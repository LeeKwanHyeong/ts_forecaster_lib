# exo_policy.py (refactor)

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, Tuple

import torch
from modeling_module.utils.exogenous_utils import compose_exo_calendar_cb

try:
    from .freq_policy import FreqSpec
except Exception:  # pragma: no cover
    from freq_policy import FreqSpec  # type: ignore


# -----------------------------
# Batch layout (robust)
# -----------------------------
@dataclass(frozen=True)
class ExoBatchIndex:
    """Batch에서 exo 텐서의 위치(index) 추론 결과."""
    idx_fe: Optional[int]      # future exo (B,H,E)
    idx_pe_cont: Optional[int] # past cont exo (B,L,E)
    idx_pe_cat: Optional[int]  # past cat exo (B,L,E_cat)


def infer_exo_batch_index(batch, *, lookback: Optional[int] = None, horizon: Optional[int] = None) -> ExoBatchIndex:
    """
    MultiPartExoDataModule batch layout을 최우선으로 '확정 인덱스'로 처리.
    그 외 loader는 기존 fallback(추론)로 처리.
    """
    if not isinstance(batch, (tuple, list)):
        return ExoBatchIndex(None, None, None)

    n = len(batch)

    # ------------------------------------------------------------------
    # (0) MultiPartExoDataModule 확정 레이아웃 (가장 우선)
    #   - train/val loader: (x, y, uid_list, fe, pe_cont, pe_cat)  len=6
    #   - inference loader: (x, id, fe, pe_cont, pe_cat)          len=5
    # ------------------------------------------------------------------
    if n == 6:
        # TrainCollateWithFutureExo.__call__의 반환 형식 그대로
        # x: Tensor[B,L,1], y: Tensor[B,H], uid_list: list[str], fe: Tensor[B,H,E], pe_cont: Tensor[B,L,E], pe_cat: Tensor[B,L,K]
        if torch.is_tensor(batch[0]) and torch.is_tensor(batch[1]) and isinstance(batch[2], list) and torch.is_tensor(batch[3]):
            return ExoBatchIndex(idx_fe=3, idx_pe_cont=4, idx_pe_cat=5)

        # (참고) dataset을 직접 iterate하는 경우: (x, y, uid, start_idx, pe_cont, pe_cat)
        # 이 경우 future exo tensor는 batch에 없고, pe는 4/5
        if torch.is_tensor(batch[0]) and torch.is_tensor(batch[1]) and isinstance(batch[2], str):
            return ExoBatchIndex(idx_fe=None, idx_pe_cont=4, idx_pe_cat=5)

    if n == 5:
        # Anchored inference dataset: (x, id, fe, pe_cont, pe_cat)
        if torch.is_tensor(batch[0]) and (isinstance(batch[1], str) or isinstance(batch[1], list)) and torch.is_tensor(batch[2]):
            return ExoBatchIndex(idx_fe=2, idx_pe_cont=3, idx_pe_cat=4)

    # ------------------------------------------------------------------
    # (1) 이하 기존 추론(fallback) 로직 유지 (다른 loader 대비)
    # ------------------------------------------------------------------
    def _is_3d_tensor_at(i: int) -> bool:
        return 0 <= i < n and torch.is_tensor(batch[i]) and batch[i].ndim == 3

    def _match_future(i: int) -> bool:
        if not _is_3d_tensor_at(i):
            return False
        t = batch[i]
        if t.shape[-1] <= 0:
            return False
        if horizon is not None and int(t.shape[1]) != int(horizon):
            return False
        return True

    def _match_past(i: int, *, want_cat: bool) -> bool:
        if not _is_3d_tensor_at(i):
            return False
        t = batch[i]
        if lookback is not None and int(t.shape[1]) != int(lookback):
            return False
        is_cat = t.dtype in (torch.int32, torch.int64)
        return is_cat if want_cat else (not is_cat)

    candidates: list[ExoBatchIndex] = []
    # (기존 후보는 필요시 유지하되, MultiPart 확정 레이아웃이 이미 위에서 처리됨)
    if n == 5:
        candidates.append(ExoBatchIndex(idx_fe=2, idx_pe_cont=3, idx_pe_cat=4))
    if n >= 6:
        candidates.append(ExoBatchIndex(idx_fe=2, idx_pe_cont=4, idx_pe_cat=5))

    for cand in candidates:
        ok_fe = (cand.idx_fe is None) or _match_future(cand.idx_fe)
        ok_pc = (cand.idx_pe_cont is None) or _match_past(cand.idx_pe_cont, want_cat=False)
        ok_pcat = (cand.idx_pe_cat is None) or _match_past(cand.idx_pe_cat, want_cat=True)
        if ok_fe and ok_pc and ok_pcat:
            return cand

    idx_fe = idx_pe_cont = idx_pe_cat = None
    for i, v in enumerate(batch):
        if not torch.is_tensor(v) or v.ndim != 3:
            continue
        t = int(v.shape[1])
        if horizon is not None and t == int(horizon) and idx_fe is None and v.shape[-1] > 0:
            idx_fe = i
        if lookback is not None and t == int(lookback):
            if v.dtype in (torch.int32, torch.int64):
                if idx_pe_cat is None:
                    idx_pe_cat = i
            else:
                if idx_pe_cont is None:
                    idx_pe_cont = i

    return ExoBatchIndex(idx_fe, idx_pe_cont, idx_pe_cat)


# -----------------------------
# Resolved spec (future + past)
# -----------------------------
@dataclass(frozen=True)
class ExoSpec:
    """Resolved exogenous configuration (future + past)."""
    use_exogenous_mode: bool

    # future
    has_loader_future_exo: bool
    loader_exo_dim: int
    exo_dim: int
    future_exo_cb: Optional[Callable]
    source: str  # "none" | "loader" | "callback"

    # past
    past_cont_dim: int = 0
    past_cat_dim: int = 0
    has_loader_past_exo: bool = False


def wrap_future_exo_cb(future_exo_cb):
    """Wrap callback to absorb `device=` kwarg and move torch.Tensor output."""
    if future_exo_cb is None:
        return None

    def _wrapped(t0, H, *args, **kwargs):
        device = kwargs.pop("device", None)
        out = future_exo_cb(t0, H)
        if device is not None and isinstance(out, torch.Tensor):
            out = out.to(device)
        return out

    return _wrapped


def infer_future_exo_spec_from_loader(loader, *, lookback: Optional[int] = None, horizon: Optional[int] = None) -> tuple[bool, int]:
    """(has_fe, fe_dim)"""
    try:
        b = next(iter(loader))
    except Exception:
        return False, 0

    idx = infer_exo_batch_index(b, lookback=lookback, horizon=horizon).idx_fe
    if idx is None:
        return False, 0

    fe = b[idx]
    if fe is None:
        return False, 0
    if torch.is_tensor(fe) and fe.ndim in (2, 3):
        return True, int(fe.shape[-1])
    return True, 0


def infer_past_exo_dim_from_loader_for_exotst(loader, *, lookback: Optional[int] = None, horizon: Optional[int] = None) -> Tuple[int, int]:
    """(past_cont_dim, past_cat_dim)"""
    try:
        b = next(iter(loader))
    except Exception:
        return 0, 0


    idxs = infer_exo_batch_index(b, lookback=lookback, horizon=horizon)

    d_cont = 0
    d_cat = 0

    if idxs.idx_pe_cont is not None:
        pe_cont = b[idxs.idx_pe_cont]
        d_cont = int(pe_cont.shape[-1]) if torch.is_tensor(pe_cont) and pe_cont.ndim == 3 else 0

    if idxs.idx_pe_cat is not None:
        pe_cat = b[idxs.idx_pe_cat]
        d_cat = int(pe_cat.shape[-1]) if torch.is_tensor(pe_cat) and pe_cat.ndim == 3 else 0

    return d_cont, d_cat


def infer_exo_dim_from_cb(future_exo_cb, horizon: int, device: str = "cpu") -> int:
    if future_exo_cb is None:
        return 0
    fe = future_exo_cb(0, horizon, device=device)  # (H,E)
    if torch.is_tensor(fe):
        return int(fe.shape[-1])
    try:
        return int(fe.shape[-1])
    except Exception:
        return 0


def resolve_exogenous(
    train_loader,
    *,
    freq_spec: FreqSpec,
    use_exogenous_mode: bool,
    lookback: Optional[int] = None,
    horizon: Optional[int] = None,
    allow_past_only: bool = False,
) -> ExoSpec:
    """
    Future + Past를 한 번에 resolve.
    - use_exogenous_mode=False면 future/past 모두 0으로 강제
    - future: loader(fe_cont) 우선, 없으면 callback(compose_exo_calendar_cb)
    - past : loader(pe_cont/pe_cat)에서 dim만 추론(없으면 0)
    - allow_past_only=True면 future exo가 없어도 callback fallback 없이 past exo만 유지
    """
    has_fe, fe_dim = infer_future_exo_spec_from_loader(train_loader, lookback=lookback, horizon=horizon)
    d_past_cont, d_past_cat = infer_past_exo_dim_from_loader_for_exotst(train_loader, lookback=lookback, horizon=horizon)
    has_past = bool(d_past_cont > 0 or d_past_cat > 0)

    print(f"[exo_policy] use_exo={use_exogenous_mode} | future(has={has_fe}, dim={fe_dim}) | past(cont={d_past_cont}, cat={d_past_cat})")

    if not use_exogenous_mode:
        if has_fe and fe_dim > 0:
            print(f"[exo_policy][WARN] use_exogenous_mode=False but loader provides future exo dim={fe_dim}. Ignoring.")
        if has_past:
            print(f"[exo_policy][WARN] use_exogenous_mode=False but loader provides past exo (cont={d_past_cont}, cat={d_past_cat}). Ignoring.")
        return ExoSpec(
            use_exogenous_mode=False,
            has_loader_future_exo=bool(has_fe),
            loader_exo_dim=int(fe_dim),
            exo_dim=0,
            future_exo_cb=None,
            source="none",
            past_cont_dim=0,
            past_cat_dim=0,
            has_loader_past_exo=has_past,
        )

    # use_exogenous_mode == True
    if has_fe:
        if fe_dim <= 0:
            if not allow_past_only:
                raise RuntimeError(
                    "[exo_policy] use_exogenous_mode=True but loader future-exo dim is invalid. "
                    f"fe_dim={fe_dim}. Check datamodule wiring / feature selection."
                )
        else:
            return ExoSpec(
                use_exogenous_mode=True,
                has_loader_future_exo=True,
                loader_exo_dim=int(fe_dim),
                exo_dim=int(fe_dim),
                future_exo_cb=None,   # loader provides it
                source="loader",
                past_cont_dim=int(d_past_cont),
                past_cat_dim=int(d_past_cat),
                has_loader_past_exo=has_past,
            )

    if allow_past_only:
        return ExoSpec(
            use_exogenous_mode=True,
            has_loader_future_exo=False,
            loader_exo_dim=0,
            exo_dim=0,
            future_exo_cb=None,
            source="none",
            past_cont_dim=int(d_past_cont),
            past_cat_dim=int(d_past_cat),
            has_loader_past_exo=has_past,
        )

    # fallback: calendar callback
    cb = compose_exo_calendar_cb(date_type=freq_spec.dt_char)
    cb = wrap_future_exo_cb(cb)

    # 가능하면 callback 실행으로 dim 추론(heuristic 제거)
    cb_dim = infer_exo_dim_from_cb(cb, int(horizon) if horizon is not None else 1, device="cpu")
    exo_dim = int(cb_dim) if cb_dim > 0 else (4 if freq_spec.freq in ("daily", "hourly") else 2)

    return ExoSpec(
        use_exogenous_mode=True,
        has_loader_future_exo=False,
        loader_exo_dim=0,
        exo_dim=exo_dim,
        future_exo_cb=cb,
        source="callback",
        past_cont_dim=int(d_past_cont),
        past_cat_dim=int(d_past_cat),
        has_loader_past_exo=has_past,
    )


# Backward-compatible aliases
resolve_future_exogenous = resolve_exogenous
_infer_future_exo_spec_from_loader = infer_future_exo_spec_from_loader
_wrap_future_exo_cb = wrap_future_exo_cb

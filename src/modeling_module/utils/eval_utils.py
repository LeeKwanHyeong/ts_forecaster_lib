import torch
import numpy as np
from typing import Optional, Callable, Tuple, Any, Dict, Literal

from modeling_module.utils.metrics import mae, rmse, smape

# 프로젝트 경로에 맞게 import 경로만 조정하세요.
# (eval_utils.py가 utils에 있고, exo_policy.py가 training/model_trainers에 있는 경우가 흔함)
try:
    from modeling_module.training.model_trainers.exo_policy import infer_exo_batch_index
except Exception:  # pragma: no cover
    # 필요 시 상대경로/대체 경로로 수정
    from modeling_module.training.exo_policy import infer_exo_batch_index  # type: ignore


MismatchPolicy = Literal["error", "auto", "zeros"]

# # 정상 평가(권장): 모델 요구에 맞추거나, use_exo_inputs=None 자동
# y, yhat = eval_on_loader_quantile(model, loader, device, use_exo_inputs=None, mismatch_policy="error")
#
# # 의도적 ablation: exo를 0으로 제거하고 성능 변화 확인
# y, yhat = eval_on_loader_quantile(model, loader, device, use_exo_inputs=False, mismatch_policy="zeros")

# ---------------------------------------------------------------------
# CKPT loader
# ---------------------------------------------------------------------
def load_model_ckpt(model, ckpt_path: str, device: str, *, strict: bool = True):
    """
    단일 ckpt 포맷: {"model_state": state_dict} 를 로드.
    strict=False는 head mismatch 등 허용할 때만 사용 권장.
    """
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state"], strict=strict)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------
# Model exo requirements (best-effort)
# ---------------------------------------------------------------------
def _get_model_cfg(model) -> Any:
    return (
        getattr(model, "configs", None)
        or getattr(model, "config", None)
        or getattr(model, "cfg", None)
    )


def _cfg_class_name(cfg) -> str:
    return cfg.__class__.__name__ if cfg is not None else ""

def _model_class_name(model) -> str:
    return model.__class__.__name__ if model is not None else ""

def _get_model_exo_requirements(model) -> Tuple[int, int, int]:
    cfg = _get_model_cfg(model)
    if cfg is None:
        return 0, 0, 0

    cfg_name = _cfg_class_name(cfg)
    model_name = _model_class_name(model)

    # ------------------------------------------------------------
    # 1) PatchTST family: d_future / d_past_* 만 신뢰
    # ------------------------------------------------------------
    if cfg_name == "PatchTSTConfig" or "PatchTST" in model_name:
        need_future = int(getattr(cfg, "future_exo_dim", 0) or 0)
        need_past_cont = int(getattr(cfg, "past_exo_cont_dim", 0) or 0)
        need_past_cat = int(getattr(cfg, "past_exo_cat_dim", 0) or 0)

        print(f"[exo_req][PatchTST] future_exo_dim={need_future}, past_exo_cont_dim={need_past_cont}, past_exo_cat_dim={need_past_cat}")
        return need_future, need_past_cont, need_past_cat

    # ------------------------------------------------------------
    # 2) Titan family: exo_dim / past_exo_*_dim 만 신뢰
    # ------------------------------------------------------------
    if cfg_name == "TitanConfig" or "Titan" in model_name:
        need_future = int(getattr(cfg, "future_exo_dim", 0) or 0)
        need_past_cont = int(getattr(cfg, "past_exo_cont_dim", 0) or 0)
        need_past_cat = int(getattr(cfg, "past_exo_cat_dim", 0) or 0)

        print(f"[exo_req][Titan] future_exo_dim={need_future}, past_exo_cont_dim={need_past_cont}, past_exo_cat_dim={need_past_cat}")
        return need_future, need_past_cont, need_past_cat

    # ------------------------------------------------------------
    # 3) PatchMixer family: (프로젝트 config 필드명에 맞춰 조정)
    # ------------------------------------------------------------
    if "PatchMixer" in model_name or cfg_name == "PatchMixerConfig":
        # 예: cfg.exo_dim / cfg.past_exo_cont_dim / cfg.past_exo_cat_dim
        need_future = int(getattr(cfg, "future_exo_dim", 0) or 0)
        need_past_cont = int(getattr(cfg, "past_exo_cont_dim", 0) or 0)
        need_past_cat = int(getattr(cfg, "past_exo_cat_dim", 0) or 0)

        print(f"[exo_req][PatchMixer] future_exo_dim={need_future}, past_exo_cont_dim={need_past_cont}, past_exo_cat_dim={need_past_cat}")
        return need_future, need_past_cont, need_past_cat

    # ------------------------------------------------------------
    # 4) ExoTST family: (프로젝트 config 필드명에 맞춰 조정)
    # ------------------------------------------------------------
    if "ExoTST" in model_name or cfg_name == "ExoTSTConfig":
        # 예: cfg.exo_dim_future / cfg.exo_dim_past
        need_future = int(getattr(cfg, "exo_dim_future", 0) or 0)
        need_past_cont = int(getattr(cfg, "exo_dim_past", 0) or 0)
        need_past_cat = 0  # 설계상 cat 미지원이면 0 고정

        print(f"[exo_req][ExoTST] future={need_future}, past_cont={need_past_cont}, past_cat={need_past_cat}")
        return need_future, need_past_cont, need_past_cat

    # ------------------------------------------------------------
    # 5) Generic fallback: "필드 존재" 기반으로만 선택 (0이면 그대로 0)
    # ------------------------------------------------------------
    if hasattr(cfg, "d_future"):
        need_future = int(getattr(cfg, "d_future", 0) or 0)
    elif hasattr(cfg, "exo_dim"):
        need_future = int(getattr(cfg, "exo_dim", 0) or 0)
    else:
        need_future = 0

    if hasattr(cfg, "d_past_cont"):
        need_past_cont = int(getattr(cfg, "d_past_cont", 0) or 0)
    elif hasattr(cfg, "past_exo_cont_dim"):
        need_past_cont = int(getattr(cfg, "past_exo_cont_dim", 0) or 0)
    else:
        need_past_cont = 0

    if hasattr(cfg, "d_past_cat"):
        need_past_cat = int(getattr(cfg, "d_past_cat", 0) or 0)
    elif hasattr(cfg, "past_exo_cat_dim"):
        need_past_cat = int(getattr(cfg, "past_exo_cat_dim", 0) or 0)
    else:
        need_past_cat = 0

    print(f"[exo_req][Generic] future={need_future}, past_cont={need_past_cont}, past_cat={need_past_cat}")
    return need_future, need_past_cont, need_past_cat


# ---------------------------------------------------------------------
# start_idx getter (for future_exo_cb)
# ---------------------------------------------------------------------
def default_start_idx_getter(batch) -> int:
    """
    기본은 batch[2]를 start_idx로 가정 (사용자 기존 코드 호환).
    프로젝트에 따라 batch[2]가 part_ids일 수 있으므로 필요 시 override 권장.
    """
    if not isinstance(batch, (tuple, list)) or len(batch) <= 2:
        return 0
    v = batch[2]
    if torch.is_tensor(v):
        if v.numel() == 0:
            return 0
        return int(v.view(-1)[0].item())
    try:
        return int(v)
    except Exception:
        return 0


def _make_future_exo_from_cb(
    future_exo_cb: Callable,
    start_idx: int,
    horizon: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """
    future_exo_cb(t0, H) -> [H,E] or [B,H,E] 를 [B,H,E]로 표준화.
    """
    fe = future_exo_cb(start_idx, horizon, device=device)
    if torch.is_tensor(fe):
        pass
    else:
        fe = torch.tensor(fe, device=device)

    if fe.dim() == 2:  # [H,E]
        fe = fe.unsqueeze(0).expand(batch_size, -1, -1)
    elif fe.dim() == 3:  # [B,H,E]
        if fe.shape[0] != batch_size:
            # broadcast 가능한 경우만 처리
            if fe.shape[0] == 1:
                fe = fe.expand(batch_size, -1, -1)
            else:
                raise RuntimeError(f"future_exo_cb returned B={fe.shape[0]} but batch_size={batch_size}")
    else:
        raise RuntimeError(f"future_exo_cb returned invalid shape={tuple(fe.shape)}")

    return fe.to(device)


def _zeros_past_cont_like(x: torch.Tensor, dim: int) -> torch.Tensor:
    # x: [B,L,1] -> zeros [B,L,dim]
    return torch.zeros((x.shape[0], x.shape[1], int(dim)), device=x.device, dtype=x.dtype)


def _zeros_past_cat_like(x: torch.Tensor, dim: int, dtype: torch.dtype = torch.long) -> torch.Tensor:
    # cat은 보통 embedding index. 기본 long.
    return torch.zeros((x.shape[0], x.shape[1], int(dim)), device=x.device, dtype=dtype)


# ---------------------------------------------------------------------
# Exo resolution for eval (SSOT aligned with infer_exo_batch_index)
# ---------------------------------------------------------------------
def resolve_eval_exo_inputs(
    *,
    model,
    batch,
    x: torch.Tensor,
    device: str,
    use_exo_inputs: Optional[bool],
    mismatch_policy: MismatchPolicy,
    future_exo_cb: Optional[Callable],
    horizon: int,
    lookback: Optional[int] = None,
    start_idx_getter: Callable = default_start_idx_getter,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], bool]:
    """
    반환:
      future_exo: [B,H,E] or None
      past_exo_cont: [B,L,E] or None
      past_exo_cat: [B,L,K] or None
      effective_use_exo: bool
    """
    need_future, need_past_cont, need_past_cat = _get_model_exo_requirements(model)

    # use_exo_inputs의 의미:
    # - None: 모델 구조에 맞춰 자동 (need_* >0 이면 공급 시도)
    # - True/False: 사용자 강제
    if use_exo_inputs is None:
        want_exo = bool((need_future > 0) or (need_past_cont > 0) or (need_past_cat > 0))
    else:
        want_exo = bool(use_exo_inputs)

    # batch에서 exo 텐서 위치 추론 (exo_policy 규칙)
    idxs = infer_exo_batch_index(batch, lookback=lookback, horizon=horizon)

    # helper: batch tensor getter
    def _get_tensor_at(i: Optional[int]) -> Optional[torch.Tensor]:
        if i is None:
            return None
        if not isinstance(batch, (tuple, list)) or i >= len(batch):
            return None
        t = batch[i]
        if t is None:
            return None
        if torch.is_tensor(t):
            return t.to(device)
        try:
            return torch.tensor(t, device=device)
        except Exception:
            return None

    # 1) 미래 exo: loader 우선, 없으면 cb
    future_exo = None
    if want_exo:
        fe = _get_tensor_at(idxs.idx_fe)
        # fe가 (B,H,0) 같이 들어올 수 있으므로 dim 검사
        if fe is not None and fe.ndim == 3 and fe.shape[-1] > 0:
            future_exo = fe
        elif future_exo_cb is not None and (need_future > 0 or use_exo_inputs):
            start_idx = start_idx_getter(batch)
            future_exo = _make_future_exo_from_cb(
                future_exo_cb,
                start_idx=start_idx,
                horizon=horizon,
                batch_size=int(x.shape[0]),
                device=device,
            )

    # 2) 과거 exo cont/cat: loader에서 가져오되 dim==0이면 None 처리
    past_exo_cont = None
    past_exo_cat = None
    if want_exo:
        pe_cont = _get_tensor_at(idxs.idx_pe_cont)
        if pe_cont is not None and pe_cont.ndim == 3 and pe_cont.shape[-1] > 0:
            past_exo_cont = pe_cont

        pe_cat = _get_tensor_at(idxs.idx_pe_cat)
        if pe_cat is not None and pe_cat.ndim == 3 and pe_cat.shape[-1] > 0:
            past_exo_cat = pe_cat

    # 3) mismatch handling:
    #    모델이 exo를 "구조적으로" 요구하는데(None)로 들어갈 상황이면 정책 적용
    missing_future = (need_future > 0) and (future_exo is None)
    missing_past_cont = (need_past_cont > 0) and (past_exo_cont is None)
    missing_past_cat = (need_past_cat > 0) and (past_exo_cat is None)

    if (missing_future or missing_past_cont or missing_past_cat):
        if mismatch_policy == "error":
            raise RuntimeError(
                "[eval_utils] Exo mismatch: model requires exo but inputs are missing.\n"
                f"  - need_future={need_future}, got_future={'yes' if future_exo is not None else 'no'}\n"
                f"  - need_past_cont={need_past_cont}, got_past_cont={'yes' if past_exo_cont is not None else 'no'}\n"
                f"  - need_past_cat={need_past_cat}, got_past_cat={'yes' if past_exo_cat is not None else 'no'}\n"
                "Fix: set use_exo_inputs=True or ensure loader/callback provides required exo tensors."
            )

        if mismatch_policy == "auto":
            # auto는 "가능하면 공급" 철학.
            # future는 cb로 공급 가능(이미 위에서 시도), past는 loader 없으면 공급 불가 -> 에러 유지
            if missing_past_cont or missing_past_cat:
                raise RuntimeError(
                    "[eval_utils] Exo mismatch (auto): model requires past exo but loader did not provide it.\n"
                    f"  - need_past_cont={need_past_cont}, need_past_cat={need_past_cat}\n"
                    "Fix: ensure loader returns pe_cont/pe_cat, or evaluate with mismatch_policy='zeros' for ablation only."
                )
            # future만 missing이면 (cb도 없었거나 실패) -> 에러
            if missing_future:
                raise RuntimeError(
                    "[eval_utils] Exo mismatch (auto): model requires future exo but neither loader nor callback provided it."
                )

        if mismatch_policy == "zeros":
            # 연구/진단 목적: exo 제거 ablation
            if missing_past_cont and need_past_cont > 0:
                past_exo_cont = _zeros_past_cont_like(x, dim=need_past_cont)
            if missing_past_cat and need_past_cat > 0:
                # dtype는 loader가 있으면 맞추는 게 이상적. 없으면 long.
                past_exo_cat = _zeros_past_cat_like(x, dim=need_past_cat, dtype=torch.long)
            # future는 cb/loader가 없으면 zeros로 만들어도 되지만,
            # future는 일반적으로 입력 head 쪽에 들어가므로 의미가 더 왜곡될 수 있어 경고성으로 처리.
            if missing_future and need_future > 0:
                future_exo = torch.zeros((x.shape[0], horizon, need_future), device=x.device, dtype=x.dtype)

    effective_use_exo = want_exo
    return future_exo, past_exo_cont, past_exo_cat, effective_use_exo


# ---------------------------------------------------------------------
# Output extraction (point/quantile)
# ---------------------------------------------------------------------
def _select_point_from_quantile_tensor(t: torch.Tensor, *, prefer_q=0.5, horizon: Optional[int] = None) -> torch.Tensor:
    """
    t:
      - [B,H] -> 그대로
      - [B,H,Q] or [B,Q,H] -> q 선택해서 [B,H]
    """
    if t.ndim == 2:
        return t

    if t.ndim != 3:
        raise ValueError(f"Unexpected prediction tensor ndim={t.ndim}, shape={tuple(t.shape)}")

    B, d1, d2 = t.shape
    # heuristic: horizon(예: 27)이 Q(예: 3/5/9)보다 크다
    if horizon is not None:
        H = int(horizon)
        # (B,H,Q)
        if d1 == H and d2 <= 32:
            q_len = d2
            q_idx = int(round((q_len - 1) * float(prefer_q)))
            return t[:, :, q_idx]
        # (B,Q,H)
        if d2 == H and d1 <= 32:
            q_len = d1
            q_idx = int(round((q_len - 1) * float(prefer_q)))
            return t[:, q_idx, :]
    # fallback heuristic
    if d1 > d2:
        q_len = d2
        q_idx = int(round((q_len - 1) * float(prefer_q)))
        return t[:, :, q_idx]
    else:
        q_len = d1
        q_idx = int(round((q_len - 1) * float(prefer_q)))
        return t[:, q_idx, :]


def _extract_pred_from_output(out, *, prefer_q=0.5, horizon: Optional[int] = None) -> torch.Tensor:
    """
    모델 출력(out)이 tensor/tuple/list/dict 등일 수 있으므로 예측 텐서를 안전하게 꺼냄.
    반환: (B,H) 또는 (B,H,*) 형태를 point (B,H)로 변환해 반환.
    """
    if torch.is_tensor(out):
        return _select_point_from_quantile_tensor(out, prefer_q=prefer_q, horizon=horizon)

    if isinstance(out, (tuple, list)):
        first = out[0]
        if torch.is_tensor(first):
            return _select_point_from_quantile_tensor(first, prefer_q=prefer_q, horizon=horizon)
        raise TypeError(f"Unsupported tuple/list output types: {[type(x) for x in out]}")

    if isinstance(out, dict):
        # PatchMixerQuantileModel 같은 형태: {"q": (B,Q,H)} or (B,H,Q)
        if "q" in out and torch.is_tensor(out["q"]):
            return _select_point_from_quantile_tensor(out["q"], prefer_q=prefer_q, horizon=horizon)

        key_candidates = ["yhat", "pred", "prediction", "y_pred", "output", "q_pred", "quantiles", "yq", "y_hat"]
        for k in key_candidates:
            if k in out and torch.is_tensor(out[k]):
                return _select_point_from_quantile_tensor(out[k], prefer_q=prefer_q, horizon=horizon)

        tensor_items = [v for v in out.values() if torch.is_tensor(v)]
        if len(tensor_items) == 1:
            return _select_point_from_quantile_tensor(tensor_items[0], prefer_q=prefer_q, horizon=horizon)

        raise KeyError(f"Cannot find tensor prediction in dict output. keys={list(out.keys())}")

    raise TypeError(f"Unsupported model output type: {type(out)}")


# ---------------------------------------------------------------------
# Unified eval: Point
# ---------------------------------------------------------------------
@torch.no_grad()
def eval_on_loader_point(
    model,
    loader,
    device: str,
    *,
    use_exo_inputs: Optional[bool] = None,
    mismatch_policy: MismatchPolicy = "error",
    future_exo_cb: Optional[Callable] = None,
    horizon: Optional[int] = None,
    lookback: Optional[int] = None,
    start_idx_getter: Callable = default_start_idx_getter,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Point 모델 평가.
    - use_exo_inputs:
        None -> 모델이 요구하면 자동 공급 시도
        True/False -> 강제
    - mismatch_policy:
        error(기본): 불일치면 즉시 에러
        auto: 가능한 범위에서 자동 공급 (past가 없으면 에러)
        zeros: exo 제거 ablation (의도적으로 0 주입)
    """
    model.eval()
    ys, yhats = [], []

    H = int(horizon or getattr(model, "horizon", 0) or 0)
    if H <= 0:
        raise ValueError("horizon must be provided or model must have .horizon")

    for batch in loader:
        x = batch[0].to(device)  # [B,L,1]
        y = batch[1].to(device)  # [B,H] or [B,H,1]

        future_exo, past_exo_cont, past_exo_cat, _ = resolve_eval_exo_inputs(
            model=model,
            batch=batch,
            x=x,
            device=device,
            use_exo_inputs=use_exo_inputs,
            mismatch_policy=mismatch_policy,
            future_exo_cb=future_exo_cb,
            horizon=H,
            lookback=lookback,
            start_idx_getter=start_idx_getter,
        )

        out = model(
            x,
            future_exo=future_exo,
            past_exo_cont=past_exo_cont,
            past_exo_cat=past_exo_cat,
        )

        yhat = _extract_pred_from_output(out, prefer_q=0.5, horizon=H)

        # y shape normalize -> (B,H)
        if y.ndim == 3 and y.shape[-1] == 1:
            y2 = y.squeeze(-1)
        else:
            y2 = y

        # yhat shape normalize -> (B,H)
        if yhat.ndim == 3 and yhat.shape[-1] == 1:
            yhat2 = yhat.squeeze(-1)
        else:
            yhat2 = yhat

        ys.append(y2.detach().cpu().numpy())
        yhats.append(yhat2.detach().cpu().numpy())

    return np.concatenate(ys, axis=0), np.concatenate(yhats, axis=0)


# ---------------------------------------------------------------------
# Unified eval: Quantile (returns q=prefer_q as point)
# ---------------------------------------------------------------------
@torch.no_grad()
def eval_on_loader_quantile(
    model,
    loader,
    device: str,
    *,
    prefer_q: float = 0.5,
    use_exo_inputs: Optional[bool] = None,
    mismatch_policy: MismatchPolicy = "error",
    future_exo_cb: Optional[Callable] = None,
    horizon: Optional[int] = None,
    lookback: Optional[int] = None,
    start_idx_getter: Callable = default_start_idx_getter,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantile 모델 평가.
    - 모델 출력이 dict("q"), tensor, tuple 등이어도 _extract_pred_from_output로 통일.
    - 반환은 (B,H) 형태의 prefer_q 점예측.
    """
    model.eval()
    ys, yhats = [], []

    H = int(horizon or getattr(model, "horizon", 0) or 0)
    if H <= 0:
        raise ValueError("horizon must be provided or model must have .horizon")

    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device)

        future_exo, past_exo_cont, past_exo_cat, _ = resolve_eval_exo_inputs(
            model=model,
            batch=batch,
            x=x,
            device=device,
            use_exo_inputs=use_exo_inputs,
            mismatch_policy=mismatch_policy,
            future_exo_cb=future_exo_cb,
            horizon=H,
            lookback=lookback,
            start_idx_getter=start_idx_getter,
        )

        out = model(
            x,
            future_exo=future_exo,
            past_exo_cont=past_exo_cont,
            past_exo_cat=past_exo_cat,
        )

        pred = _extract_pred_from_output(out, prefer_q=prefer_q, horizon=H)

        # y normalize -> (B,H)
        if y.ndim == 3 and y.shape[-1] == 1:
            y2 = y.squeeze(-1)
        else:
            y2 = y

        if pred.ndim == 3 and pred.shape[-1] == 1:
            pred2 = pred.squeeze(-1)
        else:
            pred2 = pred

        ys.append(y2.detach().cpu().numpy())
        yhats.append(pred2.detach().cpu().numpy())

    return np.concatenate(ys, axis=0), np.concatenate(yhats, axis=0)


# ---------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------
def compute_metrics(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    y_flat = y.reshape(-1)
    yhat_flat = yhat.reshape(-1)
    return {
        "MAE": float(mae(y_flat, yhat_flat)),
        "RMSE": float(rmse(y_flat, yhat_flat)),
        "SMAPE": float(smape(y_flat, yhat_flat)),
    }
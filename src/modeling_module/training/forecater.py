# forecaster.py
from __future__ import annotations

import os
import inspect
from typing import Any, Dict, List, Sequence, Optional, Callable, Tuple, Union

import numpy as np
import torch

try:
    import polars as pl
except ImportError:
    pl = None

DEBUG_FCAST = True


# -------------------------------------------------------------------------
# Device helpers
# -------------------------------------------------------------------------
def _to_device_any(obj: Any, device: torch.device) -> Any:
    """
    임의의 객체(텐서 및 컨테이너)를 대상 장치(Device)로 재귀적 이동.
    """
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device_any(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: _to_device_any(v, device) for k, v in obj.items()}
    return obj


def _infer_d_future_expected(model: torch.nn.Module) -> Optional[int]:
    """
    모델이 기대하는 미래 외생 변수(Future Exo)의 차원(Dimension) 추론 (best-effort).
    """
    for attr in ("d_future", "exo_dim"):
        if hasattr(model, attr):
            try:
                v = int(getattr(model, attr))
                if v >= 0:
                    return v
            except Exception:
                pass

    cfg = getattr(model, "cfg", None)
    if cfg is not None:
        for attr in ("d_future", "exo_dim"):
            if hasattr(cfg, attr):
                try:
                    v = int(getattr(cfg, attr))
                    if v >= 0:
                        return v
                except Exception:
                    pass

    head = getattr(model, "head", None)
    if head is not None and hasattr(head, "d_future"):
        try:
            v = int(getattr(head, "d_future"))
            if v >= 0:
                return v
        except Exception:
            pass

    return None


# -------------------------------------------------------------------------
# Standalone Helpers (Stateless)
# -------------------------------------------------------------------------
def _safe_forward(model: torch.nn.Module, x: torch.Tensor, **kwargs):
    """
    모델 forward 시그니처를 분석하여 호환되는 인자만 전달하는 안전 호출 래퍼.
    """
    try:
        sig = inspect.signature(model.forward)
        allowed = set(sig.parameters.keys())
        fkwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return model(x, **fkwargs)
    except Exception:
        try:
            return model(x, **kwargs)
        except TypeError:
            return model(x)


def _first_usable(out: Any) -> Any:
    """
    Tuple/List 등에서 유효한 첫 결과(Tensor/Dict 우선)를 추출.
    """
    if isinstance(out, (tuple, list)):
        for t in out:
            if torch.is_tensor(t) or isinstance(t, dict):
                return t
        return out[0] if out else out
    return out


def _normalize_point_to_BH(y_any: Any, B: int, H_hint: Optional[int] = None) -> torch.Tensor:
    """
    다양한 모델 출력 형태를 표준 점 예측 텐서 [B, H]로 정규화.
    """
    y_any = _first_usable(y_any)

    if isinstance(y_any, dict):
        if "point" in y_any:
            y_any = y_any["point"]
        elif "q" in y_any:
            q = y_any["q"]
            if torch.is_tensor(q):
                if q.dim() == 3 and q.size(-1) >= 3:
                    y_any = q[..., 1]
                else:
                    y_any = q[..., 0]
            elif isinstance(q, dict) and "q50" in q:
                y_any = q["q50"]
        else:
            y_any = y_any[next(iter(y_any))]

    if torch.is_tensor(y_any):
        y = y_any
        if y.dim() == 1:
            return y.view(B, -1)
        if y.dim() == 2:
            return y

        if y.dim() == 3:
            d1, d2 = y.size(1), y.size(2)

            if H_hint is not None:
                if d1 == H_hint and d2 != H_hint:
                    return y[:, :, 0]
                if d2 == H_hint and d1 != H_hint:
                    return y[:, 0, :]
                if d1 == H_hint and d2 == H_hint:
                    return y[:, :, 0]

            if d2 in (1, 3):
                return y[:, :, 1] if d2 == 3 else y[:, :, 0]
            return y[:, 0, :]

        return y.reshape(B, -1)

    raise RuntimeError(f"Unsupported point output type={type(y_any)}")


def _extract_quantile_block(out_any: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    모델 출력에서 (q10, q50, q90) 텐서를 추출 (각각 [B,H]).
    """
    out_any = _first_usable(out_any)

    if isinstance(out_any, dict):
        if all(k in out_any for k in ("q10", "q50", "q90")):
            q10, q50, q90 = out_any["q10"], out_any["q50"], out_any["q90"]

            def _S(t: torch.Tensor):
                if t.dim() == 1:
                    return t.unsqueeze(0)
                if t.dim() == 2:
                    return t
                if t.dim() == 3 and t.size(-1) == 1:
                    return t.squeeze(-1)
                if t.dim() == 3 and t.size(1) == 1:
                    return t[:, 0, :]
                return t.reshape(t.size(0), -1)

            return _S(q10), _S(q50), _S(q90)

        if "q" in out_any:
            q = out_any["q"]
            if isinstance(q, dict) and all(k in q for k in ("q10", "q50", "q90")):
                return _extract_quantile_block(q)
            out_any = q

    if not torch.is_tensor(out_any):
        raise RuntimeError(f"Unsupported quantile output type={type(out_any)}")

    q3d = out_any
    if q3d.dim() != 3:
        raise RuntimeError(f"Quantile output must be 3D, got {tuple(q3d.shape)}")

    # (B, Q, H)
    if q3d.shape[1] in (3, 5, 9):
        Qn = q3d.shape[1]
        i10, i50, i90 = (0, 1, 2) if Qn == 3 else (1, 2, 3) if Qn == 5 else (1, 4, 7)
        return q3d[:, i10, :], q3d[:, i50, :], q3d[:, i90, :]

    # (B, H, Q)
    if q3d.shape[2] in (3, 5, 9):
        Qn = q3d.shape[2]
        i10, i50, i90 = (0, 1, 2) if Qn == 3 else (1, 2, 3) if Qn == 5 else (1, 4, 7)
        return q3d[:, :, i10], q3d[:, :, i50], q3d[:, :, i90]

    raise RuntimeError(f"Cannot infer quantile axis from shape {tuple(q3d.shape)}")


def _alpha_schedule_to_zero(
    remain: int,
    *,
    linear: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    남은 remain step 동안, 마지막 스텝에서 0에 '정확히' 도달하도록 하는 감쇠 스케줄 alpha (shape: [remain]).
      - linear:   alpha(k) = 1 - k/R  (k=1..R)
      - exp-like: alpha(k) = (exp(-λk) - exp(-λR)) / (1 - exp(-λR))  (k=1..R)
                 -> alpha(R)=0, alpha(0)=1 (개형은 지수형)
    """
    if remain <= 0:
        return torch.empty((0,), device=device, dtype=dtype)

    ks = torch.arange(1, remain + 1, device=device, dtype=torch.float32)  # 1..R
    R = float(remain)

    if linear:
        alpha = 1.0 - (ks / R)  # alpha[-1] == 0 exactly
        return alpha.to(dtype)

    # exp-shaped but hits 0 at end exactly
    lam = float(np.log(100.0) / max(1.0, R))  # shape parameter (조절 가능)
    eR = torch.exp(torch.tensor(-lam * R, device=device, dtype=torch.float32))
    alpha = (torch.exp(-lam * ks) - eR) / (1.0 - eR)  # alpha(R)=0
    return alpha.to(dtype)


# -------------------------------------------------------------------------
# Main Class
# -------------------------------------------------------------------------
class DMSForecaster:
    """
    Unified Forecaster for DMS + IMS extension.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        target_channel: int = 0,
        fill_mode: str = "copy_last",
        ttm: Optional[object] = None,
        # Guards Config
        use_winsor: bool = False,
        use_multi_guard: bool = False,
        use_dampen: bool = False,
        winsor_q: Tuple[float, float] = (0.05, 0.95),
        winsor_mul: float = 2.0,
        winsor_growth: float = 1.2,
        max_step_up: float = 0.05,
        max_step_down: float = 0.40,
        damp: float = 0.30,
        # Quantile Specific Config
        quantile_feed: str = "q50",  # 'q10' or 'q50'
    ):
        self.model = model
        self.target_channel = target_channel
        self.fill_mode = fill_mode
        self.ttm = ttm

        self.guard_cfg = dict(
            use_winsor=use_winsor,
            use_multi_guard=use_multi_guard,
            use_dampen=use_dampen,
            winsor_q=winsor_q,
            winsor_mul=winsor_mul,
            winsor_growth=winsor_growth,
            max_step_up=max_step_up,
            max_step_down=max_step_down,
            damp=damp,
        )
        self.quantile_feed = quantile_feed
        self.global_t0 = 0

        # set on predict()
        self.future_exo_cb: Optional[Callable[[int, int], torch.Tensor]] = None

    # ---------------------------------------------------------------------
    # Public Entry Point
    # ---------------------------------------------------------------------
    def predict(
        self,
        x_init: torch.Tensor,
        *,
        horizon: int,
        device: Union[str, torch.device, None] = None,
        mode: str = "eval",
        # Optional Exogenous / IDs
        part_ids: Optional[Sequence[Any]] = None,
        past_exo_cont: Optional[torch.Tensor] = None,
        past_exo_cat: Optional[torch.Tensor] = None,
        future_exo_batch: Optional[torch.Tensor] = None,
        future_exo_cb: Optional[Callable[[int, int, torch.device], torch.Tensor]] = None,
        # NEW: horizon extension policy
        is_IMS: bool = True,
        is_linear_decay: bool = True,
    ) -> Dict[str, Any]:
        """
        is_IMS=True:
          - 기존 AR(IMS) 방식 (DMS-to-IMS)
        is_IMS=False:
          - DMS로 Hm까지 만든 뒤, "마지막 예측값(outputs[-1])" 기준으로 0까지 감쇠
          - is_linear_decay=True  -> 선형 감쇠
          - is_linear_decay=False -> 지수형(개형) 감쇠(단, 마지막 스텝 0 정확히)
        """
        device = torch.device(device or next(self.model.parameters()).device)
        self.model.to(device).eval()

        part_ids = _to_device_any(part_ids, device)
        past_exo_cont = _to_device_any(past_exo_cont, device)
        past_exo_cat = _to_device_any(past_exo_cat, device)
        future_exo_batch = _to_device_any(future_exo_batch, device)

        x_raw = x_init.to(device).float().clone()
        if x_raw.dim() == 2:
            x_raw = x_raw.unsqueeze(-1)
        B = x_raw.size(0)

        # --------------------------------------------------------------
        # 1) Future exo wiring
        # --------------------------------------------------------------
        d_future_expected = _infer_d_future_expected(self.model)
        cb_final = future_exo_cb

        if torch.is_tensor(future_exo_batch):
            exb = future_exo_batch

            if exb.dim() == 2:
                def _cb_from_batch(t0: int, h_req: int, dev: torch.device):
                    s, e = int(t0), int(t0) + int(h_req)
                    Htot = exb.size(0)
                    if Htot >= e:
                        return exb[s:e, :].detach().to(dev)
                    if Htot <= s:
                        return exb[-1:, :].expand(h_req, -1).detach().to(dev)
                    tail = exb[s:, :]
                    pad = exb[-1:, :].expand(e - Htot, -1)
                    return torch.cat([tail, pad], dim=0).detach().to(dev)

                cb_final = _cb_from_batch

            elif exb.dim() == 3:
                if exb.size(0) not in (1, B):
                    raise RuntimeError(
                        f"future_exo_batch has incompatible batch dim: got {exb.size(0)}, expected 1 or {B}."
                    )

                def _cb_from_batch_b(t0: int, h_req: int, dev: torch.device):
                    s, e = int(t0), int(t0) + int(h_req)
                    Htot = exb.size(1)
                    if Htot >= e:
                        out = exb[:, s:e, :]
                    elif Htot <= s:
                        out = exb[:, -1:, :].expand(exb.size(0), h_req, -1)
                    else:
                        tail = exb[:, s:, :]
                        pad = exb[:, -1:, :].expand(exb.size(0), e - Htot, -1)
                        out = torch.cat([tail, pad], dim=1)

                    if out.size(0) == 1 and B > 1:
                        out = out.expand(B, -1, -1)
                    return out.detach().to(dev)

                cb_final = _cb_from_batch_b

        self.future_exo_cb = None
        if cb_final is not None:
            self.future_exo_cb = lambda t, h: cb_final(t, h, device)

        if d_future_expected is not None:
            if d_future_expected <= 0:
                self.future_exo_cb = None
            else:
                if self.future_exo_cb is None:
                    raise RuntimeError(
                        f"Model expects future_exo dim d_future={d_future_expected}, "
                        f"but neither future_exo_batch nor future_exo_cb was provided."
                    )

        # 2) forward kwargs
        fwd_kwargs: Dict[str, Any] = {}
        if part_ids is not None:
            fwd_kwargs["part_ids"] = part_ids

        if past_exo_cont is not None:
            fwd_kwargs["past_exo_cont"] = past_exo_cont
            fwd_kwargs["pe_cont"] = past_exo_cont
        if past_exo_cat is not None:
            fwd_kwargs["past_exo_cat"] = past_exo_cat
            fwd_kwargs["pe_cat"] = past_exo_cat

        if mode is not None:
            fwd_kwargs["mode"] = mode

        # 3) probe forward to decide output type / Hm
        H_hint = int(getattr(self.model, "horizon", getattr(self.model, "output_horizon", 0)) or 0)
        probe_H = int(horizon if H_hint == 0 else max(1, H_hint))

        exo_probe = None
        if self.future_exo_cb is not None:
            ex = self.future_exo_cb(0, probe_H)

            if d_future_expected is not None and d_future_expected > 0:
                if ex is None:
                    raise RuntimeError(f"future_exo_cb returned None, but d_future={d_future_expected} is required")
                if ex.ndim not in (2, 3):
                    raise RuntimeError(f"future_exo_cb must return (H,E) or (B,H,E); got shape={tuple(ex.shape)}")
                if int(ex.shape[-1]) != int(d_future_expected):
                    raise RuntimeError(
                        f"future_exo dim mismatch: got E={int(ex.shape[-1])}, expected d_future={int(d_future_expected)}"
                    )

            if ex.ndim == 2:
                exo_probe = ex.to(device).unsqueeze(0).expand(B, -1, -1)
            else:
                exo_probe = ex.to(device)

        with torch.no_grad():
            out0 = _safe_forward(
                self.model,
                x_raw,
                future_exo=exo_probe,
                fe_cont=exo_probe,
                **fwd_kwargs,
            )

        # 4) branch: quantile vs point
        is_quantile = False
        try:
            _extract_quantile_block(out0)
            is_quantile = True
        except Exception:
            is_quantile = False

        if is_quantile:
            return self._predict_quantile_strategy(
                x_raw, out0, horizon, device, fwd_kwargs,
                is_IMS=is_IMS, is_linear_decay=is_linear_decay
            )
        else:
            return self._predict_point_strategy(
                x_raw, out0, horizon, device, fwd_kwargs, probe_H,
                is_IMS=is_IMS, is_linear_decay=is_linear_decay
            )

    # ---------------------------------------------------------------------
    # Strategies
    # ---------------------------------------------------------------------
    def _predict_point_strategy(
        self,
        x_raw,
        out0,
        horizon,
        device,
        fwd_kwargs,
        probe_H,
        *,
        is_IMS: bool,
        is_linear_decay: bool,
    ):
        B = x_raw.size(0)
        y0 = _normalize_point_to_BH(out0, B, H_hint=probe_H)
        Hm = int(y0.size(1))

        if int(horizon) <= Hm:
            y_hat = y0[:, :int(horizon)]
        else:
            if is_IMS:
                y_hat = self._impl_point_DMS_to_IMS(
                    x_init=x_raw,
                    horizon=int(horizon),
                    model_horizon=Hm,
                    device=device,
                    fwd_kwargs=fwd_kwargs,
                )
            else:
                y_hat = self._impl_point_DMS_then_decay_to_zero(
                    x_init=x_raw,
                    horizon=int(horizon),
                    model_horizon=Hm,
                    device=device,
                    fwd_kwargs=fwd_kwargs,
                    linear=is_linear_decay,
                )

        return {"point": y_hat.detach().cpu().numpy().reshape(-1)}

    def _predict_quantile_strategy(
        self,
        x_raw,
        out0,
        horizon,
        device,
        fwd_kwargs,
        *,
        is_IMS: bool,
        is_linear_decay: bool,
    ):
        q10_blk, q50_blk, q90_blk = _extract_quantile_block(out0)
        Hm = int(q50_blk.size(1))

        if int(horizon) <= Hm:
            q10 = q10_blk[:, :int(horizon)]
            q50 = q50_blk[:, :int(horizon)]
            q90 = q90_blk[:, :int(horizon)]
        else:
            if is_IMS:
                q10, q50, q90 = self._impl_quantile_DMS_to_IMS(
                    x_init=x_raw,
                    horizon=int(horizon),
                    model_horizon=Hm,
                    device=device,
                    fwd_kwargs=fwd_kwargs,
                )
            else:
                q10, q50, q90 = self._impl_quantile_DMS_then_decay_to_zero(
                    x_init=x_raw,
                    horizon=int(horizon),
                    model_horizon=Hm,
                    device=device,
                    fwd_kwargs=fwd_kwargs,
                    linear=is_linear_decay,
                )

        return {
            "q10": q10.detach().cpu().numpy().reshape(-1),
            "q50": q50.detach().cpu().numpy().reshape(-1),
            "q90": q90.detach().cpu().numpy().reshape(-1),
            "point": q50.detach().cpu().numpy().reshape(-1),
        }

    # ---------------------------------------------------------------------
    # Implementation: Point Autoregression (IMS)
    # ---------------------------------------------------------------------
    def _impl_point_DMS_to_IMS(
        self,
        x_init: torch.Tensor,
        horizon: int,
        model_horizon: int,
        device: torch.device,
        fwd_kwargs: Dict,
    ) -> torch.Tensor:
        x_raw = x_init.clone()
        B, L, C = x_raw.shape
        Hm = model_horizon

        def _call_point(xr, need_h, step_offset):
            exo = None
            if self.future_exo_cb is not None:
                t0 = self.global_t0 + step_offset
                ex = self.future_exo_cb(t0, need_h).to(xr.device)

                if ex.ndim == 2:
                    exo = ex.unsqueeze(0).expand(B, -1, -1)
                elif ex.ndim == 3:
                    if ex.size(0) == 1 and B > 1:
                        exo = ex.expand(B, -1, -1)
                    elif ex.size(0) == B:
                        exo = ex
                    else:
                        raise RuntimeError(f"future_exo batch dim mismatch: got {ex.size(0)}, expected 1 or {B}")
                else:
                    raise RuntimeError(f"future_exo must be (H,E) or (B,H,E), got shape={tuple(ex.shape)}")

            out = _safe_forward(self.model, xr, future_exo=exo, fe_cont=exo, **fwd_kwargs)
            return _normalize_point_to_BH(out, B, H_hint=need_h)

        y_block_raw = _call_point(x_raw, Hm, 0)

        if DEBUG_FCAST:
            print(f"[DMS] Point AR Start. Hm={Hm}, H_req={horizon}")

        outputs: List[torch.Tensor] = []
        use_len = min(Hm, horizon)

        for t in range(use_len):
            if self.ttm:
                self.ttm.add_context(x_raw)
            y_step = y_block_raw[:, t]
            y_adj = self._apply_guards(x_raw, y_step)
            outputs.append(y_adj.unsqueeze(1))
            x_raw = self._prepare_next_input(x_raw, y_adj)

        if horizon > Hm:
            for t in range(horizon - Hm):
                if self.ttm:
                    self.ttm.add_context(x_raw)
                y_full = _call_point(x_raw, Hm, step_offset=(use_len + t))
                y_step = y_full[:, 0]
                y_adj = self._apply_guards(x_raw, y_step)
                outputs.append(y_adj.unsqueeze(1))
                x_raw = self._prepare_next_input(x_raw, y_adj)

        return torch.cat(outputs, dim=1)

    # ---------------------------------------------------------------------
    # Implementation: Quantile Autoregression (IMS)
    # ---------------------------------------------------------------------
    def _impl_quantile_DMS_to_IMS(
        self,
        x_init: torch.Tensor,
        horizon: int,
        model_horizon: int,
        device: torch.device,
        fwd_kwargs: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_raw = x_init.clone()
        B, L, C = x_raw.shape
        Hm = model_horizon

        def _call_quantile(xr, step_offset):
            exo = None
            if self.future_exo_cb is not None:
                ex = self.future_exo_cb(step_offset, Hm).to(xr.device)

                if ex.ndim == 2:
                    exo = ex.unsqueeze(0).expand(B, -1, -1)
                elif ex.ndim == 3:
                    if ex.size(0) == 1 and B > 1:
                        exo = ex.expand(B, -1, -1)
                    elif ex.size(0) == B:
                        exo = ex
                    else:
                        raise RuntimeError(f"future_exo batch dim mismatch: got {ex.size(0)}, expected 1 or {B}")
                else:
                    raise RuntimeError(f"future_exo must be (H,E) or (B,H,E), got shape={tuple(ex.shape)}")

            out = _safe_forward(self.model, xr, future_exo=exo, fe_cont=exo, **fwd_kwargs)
            return _extract_quantile_block(out)

        q10_blk, q50_blk, q90_blk = _call_quantile(x_raw, 0)
        use_len = min(horizon, Hm)

        q10_seq: List[torch.Tensor] = []
        q50_seq: List[torch.Tensor] = []
        q90_seq: List[torch.Tensor] = []

        for t in range(use_len):
            q10, q50, q90 = q10_blk[:, t], q50_blk[:, t], q90_blk[:, t]

            y_feed = q10 if self.quantile_feed == "q10" else q50
            y_next = self._apply_growth_guard(x_raw, y_feed)

            q10_seq.append(q10.unsqueeze(1))
            q50_seq.append(q50.unsqueeze(1))
            q90_seq.append(q90.unsqueeze(1))

            x_raw = self._prepare_next_input(x_raw, y_next)

        if horizon > Hm:
            for k in range(horizon - Hm):
                offset = use_len + k
                qb10, qb50, qb90 = _call_quantile(x_raw, offset)
                q10, q50, q90 = qb10[:, 0], qb50[:, 0], qb90[:, 0]

                y_feed = q10 if self.quantile_feed == "q10" else q50
                y_next = self._apply_growth_guard(x_raw, y_feed)

                q10_seq.append(q10.unsqueeze(1))
                q50_seq.append(q50.unsqueeze(1))
                q90_seq.append(q90.unsqueeze(1))

                x_raw = self._prepare_next_input(x_raw, y_next)

        return torch.cat(q10_seq, 1), torch.cat(q50_seq, 1), torch.cat(q90_seq, 1)

    # ---------------------------------------------------------------------
    # NEW: DMS then decay-to-zero (NO IMS)
    #   - tail is anchored at last predicted output (outputs[-1])
    #   - tail hits 0 at final step exactly
    # ---------------------------------------------------------------------
    def _impl_point_DMS_then_decay_to_zero(
        self,
        x_init: torch.Tensor,
        horizon: int,
        model_horizon: int,
        device: torch.device,
        fwd_kwargs: Dict,
        *,
        linear: bool,
    ) -> torch.Tensor:
        x_raw = x_init.clone()
        B, L, C = x_raw.shape
        Hm = model_horizon

        def _call_point(xr, need_h, step_offset):
            exo = None
            if self.future_exo_cb is not None:
                t0 = self.global_t0 + step_offset
                ex = self.future_exo_cb(t0, need_h).to(xr.device)

                if ex.ndim == 2:
                    exo = ex.unsqueeze(0).expand(B, -1, -1)
                elif ex.ndim == 3:
                    if ex.size(0) == 1 and B > 1:
                        exo = ex.expand(B, -1, -1)
                    elif ex.size(0) == B:
                        exo = ex
                    else:
                        raise RuntimeError(f"future_exo batch dim mismatch: got {ex.size(0)}, expected 1 or {B}")
                else:
                    raise RuntimeError(f"future_exo must be (H,E) or (B,H,E), got shape={tuple(ex.shape)}")

            out = _safe_forward(self.model, xr, future_exo=exo, fe_cont=exo, **fwd_kwargs)
            return _normalize_point_to_BH(out, B, H_hint=need_h)

        # 1) DMS block
        y_block_raw = _call_point(x_raw, Hm, 0)

        outputs: List[torch.Tensor] = []
        use_len = min(Hm, horizon)  # horizon > Hm in caller, so use_len == Hm

        for t in range(use_len):
            if self.ttm:
                self.ttm.add_context(x_raw)
            y_step = y_block_raw[:, t]
            y_adj = self._apply_guards(x_raw, y_step)  # DMS 구간은 기존과 동일
            outputs.append(y_adj.unsqueeze(1))
            x_raw = self._prepare_next_input(x_raw, y_adj)

        if use_len >= horizon:
            return torch.cat(outputs, dim=1)

        # 2) Tail decay (anchor = last predicted output)
        last_pred = outputs[-1].squeeze(1)  # (B,)
        remain = horizon - use_len
        alpha = _alpha_schedule_to_zero(remain, linear=linear, device=device, dtype=x_raw.dtype)  # (remain,)

        # 주의: decay 구간은 "0까지 정확히 가야" 하므로, guard를 적용하면 목표를 깨뜨릴 수 있음.
        # 따라서 decay 구간에서는 guard 없이 스케줄 그대로 사용.
        for k in range(remain):
            yk = last_pred * alpha[k]  # (B,)
            outputs.append(yk.unsqueeze(1))
            x_raw = self._prepare_next_input(x_raw, yk)

        return torch.cat(outputs, dim=1)

    def _impl_quantile_DMS_then_decay_to_zero(
        self,
        x_init: torch.Tensor,
        horizon: int,
        model_horizon: int,
        device: torch.device,
        fwd_kwargs: Dict,
        *,
        linear: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_raw = x_init.clone()
        B, L, C = x_raw.shape
        Hm = model_horizon

        def _call_quantile(xr, step_offset):
            exo = None
            if self.future_exo_cb is not None:
                ex = self.future_exo_cb(step_offset, Hm).to(xr.device)

                if ex.ndim == 2:
                    exo = ex.unsqueeze(0).expand(B, -1, -1)
                elif ex.ndim == 3:
                    if ex.size(0) == 1 and B > 1:
                        exo = ex.expand(B, -1, -1)
                    elif ex.size(0) == B:
                        exo = ex
                    else:
                        raise RuntimeError(f"future_exo batch dim mismatch: got {ex.size(0)}, expected 1 or {B}")
                else:
                    raise RuntimeError(f"future_exo must be (H,E) or (B,H,E), got shape={tuple(ex.shape)}")

            out = _safe_forward(self.model, xr, future_exo=exo, fe_cont=exo, **fwd_kwargs)
            return _extract_quantile_block(out)

        # 1) DMS block
        q10_blk, q50_blk, q90_blk = _call_quantile(x_raw, 0)

        q10_seq: List[torch.Tensor] = []
        q50_seq: List[torch.Tensor] = []
        q90_seq: List[torch.Tensor] = []

        use_len = min(horizon, Hm)  # horizon > Hm in caller, so use_len == Hm

        for t in range(use_len):
            q10, q50, q90 = q10_blk[:, t], q50_blk[:, t], q90_blk[:, t]

            # DMS 구간은 기존과 동일하게 feed/guard 적용
            y_feed = q10 if self.quantile_feed == "q10" else q50
            y_next = self._apply_growth_guard(x_raw, y_feed)

            q10_seq.append(q10.unsqueeze(1))
            q50_seq.append(q50.unsqueeze(1))
            q90_seq.append(q90.unsqueeze(1))

            x_raw = self._prepare_next_input(x_raw, y_next)

        if use_len >= horizon:
            return torch.cat(q10_seq, 1), torch.cat(q50_seq, 1), torch.cat(q90_seq, 1)

        # 2) Tail decay-to-zero anchored at last predicted quantiles
        last_q10 = q10_seq[-1].squeeze(1)  # (B,)
        last_q50 = q50_seq[-1].squeeze(1)  # (B,)
        last_q90 = q90_seq[-1].squeeze(1)  # (B,)

        remain = horizon - use_len
        alpha = _alpha_schedule_to_zero(remain, linear=linear, device=device, dtype=x_raw.dtype)

        # decay 구간은 0 도달 정확성이 중요하므로 guard 미적용
        for k in range(remain):
            a = alpha[k]
            q10k = last_q10 * a
            q50k = last_q50 * a
            q90k = last_q90 * a

            q10_seq.append(q10k.unsqueeze(1))
            q50_seq.append(q50k.unsqueeze(1))
            q90_seq.append(q90k.unsqueeze(1))

            # 입력 업데이트는 q50 기반 (일관성)
            x_raw = self._prepare_next_input(x_raw, q50k)

        return torch.cat(q10_seq, 1), torch.cat(q50_seq, 1), torch.cat(q90_seq, 1)

    # ---------------------------------------------------------------------
    # Internal Logic: Input Preparation & Guards
    # ---------------------------------------------------------------------
    def _prepare_next_input(self, x_raw, y_next_val):
        B, L, C = x_raw.shape
        y_r = y_next_val.reshape(B, 1, 1)

        if C == 1:
            new_token = y_r
        else:
            last = x_raw[:, -1:, :].clone()
            if self.fill_mode == "zeros":
                new_token = torch.zeros_like(last)
            else:
                new_token = last
            new_token[:, 0, self.target_channel] = y_r[:, 0, 0]

        return torch.cat([x_raw[:, 1:, :], new_token], dim=1)

    def _apply_guards(self, x_raw, y_step):
        cfg = self.guard_cfg
        hist_raw = x_raw[:, :, self.target_channel]
        last_raw = hist_raw[:, -1]
        y_adj = y_step.float()

        if cfg["use_winsor"]:
            y_adj = self._winsorize_clamp_raw(hist_raw, y_adj, **cfg)
        if cfg["use_multi_guard"]:
            y_adj = self._guard_multiplicative_raw(last_raw, y_adj, **cfg)
        if cfg["use_dampen"]:
            y_adj = self._dampen_to_last_raw(last_raw, y_adj, **cfg)
        return y_adj

    def _apply_growth_guard(self, x_raw, y_step):
        return self._apply_guards(x_raw, y_step)

    def _winsorize_clamp_raw(self, hist_raw, y, winsor_q, winsor_mul, winsor_growth, **kwargs):
        last = hist_raw[:, -1]
        hist_safe = torch.where(torch.isfinite(hist_raw), hist_raw, last.unsqueeze(1))

        q_lo = torch.quantile(hist_safe, winsor_q[0], dim=1)
        q_hi = torch.quantile(hist_safe, winsor_q[1], dim=1)

        cap_quant = q_hi * winsor_mul
        cap_growth = torch.where(last > 0, last * winsor_growth, cap_quant)
        max_cap = torch.minimum(cap_quant, cap_growth)

        y = torch.clamp(y, max=max_cap)
        return y

    def _guard_multiplicative_raw(self, last_raw, y, max_step_up, max_step_down, **kwargs):
        eps = 1e-6
        last_safe = torch.clamp(last_raw, min=eps)
        y_safe = torch.clamp(y, min=eps)

        ratio = y_safe / last_safe
        log_ratio = torch.log(ratio)

        log_min = torch.log(torch.tensor(1.0 - max_step_down, device=y.device))
        log_max = torch.log(torch.tensor(1.0 + max_step_up, device=y.device))

        log_ratio = torch.clamp(log_ratio, min=log_min, max=log_max)
        return last_safe * torch.exp(log_ratio)

    def _dampen_to_last_raw(self, last_raw, y, damp, **kwargs):
        if damp <= 0.0:
            return y
        return (1.0 - damp) * last_raw + damp * y


# -------------------------------------------------------------------------
# Export helpers
# -------------------------------------------------------------------------
def _unpack_batch_for_export(batch: Any) -> Dict[str, Any]:
    """
    loader가 내보내는 배치를 key dict로 정규화.
    """
    x = batch[0]
    y = batch[1] if len(batch) >= 2 else None
    part_ids = batch[2] if len(batch) >= 3 else None
    future_exo = batch[3] if len(batch) >= 4 else None
    past_exo_cont = batch[4] if len(batch) >= 5 else None
    past_exo_cat = batch[5] if len(batch) >= 6 else None

    return dict(
        x=x,
        y=y,
        part_ids=part_ids,
        future_exo=future_exo,
        past_exo_cont=past_exo_cont,
        past_exo_cat=past_exo_cat,
    )


def _to_py_id(v) -> str:
    if v is None:
        return "NA"
    if torch.is_tensor(v):
        return str(v.item()) if v.numel() == 1 else str(v.tolist())
    return str(v)


def forecast_to_parquet(
    model_dict: Dict[str, torch.nn.Module],
    loader,
    *,
    parquet_path: str,
    horizon: int,
    freq: str = "unknown",
    mode: str = "infer",
    plan_dt: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    target_channel: int = 0,
    fill_mode: str = "copy_last",
    max_samples: int = 200,
    future_exo_cb: Optional[Callable] = None,
    is_IMS: bool = True,
    is_linear_decay: bool = True,
):
    """
    여러 모델의 예측 결과를 생성하여 Parquet로 저장.
    """
    if pl is None:
        raise ImportError("polars required")

    rows: List[Dict[str, Any]] = []
    device = torch.device(device) if device else None
    sample_idx = 0

    forecasters = {
        name: DMSForecaster(
            model,
            target_channel=target_channel,
            fill_mode=fill_mode,
            use_winsor=True,
            use_multi_guard=True,
        )
        for name, model in model_dict.items()
    }

    for batch in loader:
        b = _unpack_batch_for_export(batch)
        xb = b["x"]
        B = xb.size(0)

        for i in range(B):
            if sample_idx >= max_samples:
                break

            x1 = xb[i : i + 1]
            y1 = b["y"][i] if b["y"] is not None else None

            pid = b["part_ids"][i] if b["part_ids"] is not None else None
            fe1 = b["future_exo"][i : i + 1] if b["future_exo"] is not None else None
            pec1 = b["past_exo_cont"][i : i + 1] if b["past_exo_cont"] is not None else None
            pek1 = b["past_exo_cat"][i : i + 1] if b["past_exo_cat"] is not None else None

            for name, fcaster in forecasters.items():
                pred = fcaster.predict(
                    x1,
                    horizon=int(horizon),
                    device=device,
                    mode=mode,
                    part_ids=[pid] if pid is not None else None,
                    past_exo_cont=pec1,
                    past_exo_cat=pek1,
                    future_exo_batch=fe1,
                    future_exo_cb=future_exo_cb,
                    is_IMS=is_IMS,
                    is_linear_decay=is_linear_decay,
                )

                # point is always present
                y_point = pred.get("point")
                y_point_list = y_point.tolist() if hasattr(y_point, "tolist") else list(y_point)

                # quantiles may not exist for non-quantile models
                q10 = pred.get("q10", None)
                q50 = pred.get("q50", None)
                q90 = pred.get("q90", None)

                rows.append(
                    {
                        "part_id": _to_py_id(pid),
                        "sample_idx": int(sample_idx),
                        "model": str(name),
                        "horizon": int(horizon),
                        "y_pred_point": y_point_list,
                        "y_pred_q10": q10.tolist() if hasattr(q10, "tolist") else (q10 if q10 is None else list(q10)),
                        "y_pred_q50": q50.tolist() if hasattr(q50, "tolist") else (q50 if q50 is None else list(q50)),
                        "y_pred_q90": q90.tolist() if hasattr(q90, "tolist") else (q90 if q90 is None else list(q90)),
                    }
                )

            sample_idx += 1

        if sample_idx >= max_samples:
            break

    df = pl.DataFrame(rows)
    os.makedirs(os.path.dirname(parquet_path) or ".", exist_ok=True)
    df.write_parquet(parquet_path)
    return df

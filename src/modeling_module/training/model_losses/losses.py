# loss_module_refactored.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import math

from modeling_module.training.config import TrainingConfig

Tensor = torch.Tensor
Pred = Union[Tensor, Dict[str, Any]]


# =============================================================================
# 0) Core numerical utilities
# =============================================================================
def safe_div(numer: Tensor, denom: Tensor, eps: float = 1e-12) -> Tensor:
    """
    numer / denom with NaN/Inf safety.
    - denom is clamped by eps to avoid divide-by-zero.
    - NaN/Inf are replaced with 0.
    """
    denom = denom.clamp_min(eps)
    out = numer / denom
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def ensure_2d(y: Tensor) -> Tensor:
    """
    Normalize y to [B, H].
    Accepts:
      - [B, H]
      - [B, 1, H]  -> squeeze dim=1
      - [B, H, 1]  -> squeeze dim=-1
    """
    if y.dim() == 3 and y.size(1) == 1:     # [B,1,H]
        return y.squeeze(1)
    if y.dim() == 3 and y.size(-1) == 1:    # [B,H,1]
        return y.squeeze(-1)
    if y.dim() != 2:
        raise ValueError(f"Expected y as [B,H] (or [B,1,H]/[B,H,1]), got {tuple(y.shape)}")
    return y


def ensure_3d_quantile(pred_q: Tensor) -> Tensor:
    """
    Normalize quantile prediction to [B, Q, H].
    Accepts:
      - [B, Q, H]
      - [B, Q, H, C] is NOT handled here (caller must select channel)
    """
    if pred_q.dim() != 3:
        raise ValueError(f"Expected quantile pred as [B,Q,H], got {tuple(pred_q.shape)}")
    return pred_q


def maybe_mask_like(x: Tensor, mask: Optional[Tensor]) -> Tensor:
    """
    Returns mask broadcastable to x. If mask is None -> ones_like(x).
    """
    if mask is None:
        return torch.ones_like(x)
    return mask


# =============================================================================
# 1) Classic point losses (mask-aware)
# =============================================================================
def mae_loss(y: Tensor, y_hat: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    y = ensure_2d(y); y_hat = ensure_2d(y_hat)
    m = maybe_mask_like(y_hat, mask)
    return (torch.abs(y - y_hat) * m).mean()


def mse_loss(y: Tensor, y_hat: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    y = ensure_2d(y); y_hat = ensure_2d(y_hat)
    m = maybe_mask_like(y_hat, mask)
    return (((y - y_hat) ** 2) * m).mean()


def rmse_loss(y: Tensor, y_hat: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    y = ensure_2d(y); y_hat = ensure_2d(y_hat)
    m = maybe_mask_like(y_hat, mask)
    return torch.sqrt((((y - y_hat) ** 2) * m).mean())


def mape_loss(y: Tensor, y_hat: Tensor, mask: Optional[Tensor] = None, eps: float = 1e-12) -> Tensor:
    """
    Mean Absolute Percentage Error with safe division.
    """
    y = ensure_2d(y); y_hat = ensure_2d(y_hat)
    m = maybe_mask_like(y_hat, mask)
    scale = torch.abs(y).clamp_min(eps)
    ape = torch.abs(y - y_hat) / scale
    return (ape * m).mean()


def smape_loss(y: Tensor, y_hat: Tensor, mask: Optional[Tensor] = None, eps: float = 1e-12) -> Tensor:
    """
    Symmetric MAPE (bounded in [0,2] if scaled as 2*mean(...)).
    """
    y = ensure_2d(y); y_hat = ensure_2d(y_hat)
    m = maybe_mask_like(y_hat, mask)
    delta = torch.abs(y - y_hat)
    scale = (torch.abs(y) + torch.abs(y_hat)).clamp_min(eps)
    frac = delta / scale
    return 2.0 * (frac * m).mean()


def mase_loss(
    y: Tensor,
    y_hat: Tensor,
    y_insample: Tensor,
    seasonality: int,
    mask: Optional[Tensor] = None,
    eps: float = 1e-12,
) -> Tensor:
    """
    MASE = MAE(y,yhat) / MAE(seasonal_naive)
    y_insample: [B, L]
    """
    y = ensure_2d(y); y_hat = ensure_2d(y_hat)
    y_insample = ensure_2d(y_insample)
    m = maybe_mask_like(y_hat, mask)

    delta = torch.abs(y - y_hat)

    if seasonality <= 0 or y_insample.size(1) <= seasonality:
        # fallback: scale=1 to avoid crash, but you may want to raise.
        scale = torch.ones((y.size(0),), device=y.device, dtype=y.dtype)
    else:
        naive_err = torch.abs(y_insample[:, seasonality:] - y_insample[:, :-seasonality])
        scale = naive_err.mean(dim=1).clamp_min(eps)

    return (safe_div(delta, scale[:, None], eps=eps) * m).mean()


# =============================================================================
# 2) Quantile losses
# =============================================================================
def pinball_loss(
    y: Tensor,
    y_hat: Tensor,
    tau: float,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Pinball loss for a single quantile.
    y, y_hat: [B,H]
    """
    y = ensure_2d(y); y_hat = ensure_2d(y_hat)
    m = maybe_mask_like(y_hat, mask)
    diff = y - y_hat
    loss = torch.maximum(tau * diff, (tau - 1.0) * diff)
    return (loss * m).mean()


def multi_quantile_loss(
    y: Tensor,
    y_hat_q: Tensor,
    quantiles: Sequence[float],
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Average multi-quantile pinball loss.
    y: [B,H]
    y_hat_q: [B,Q,H]
    mask: broadcastable to [B,Q,H] (optional)
    """
    y = ensure_2d(y)
    y_hat_q = ensure_3d_quantile(y_hat_q)

    q = torch.as_tensor(quantiles, device=y_hat_q.device, dtype=y_hat_q.dtype)  # [Q]
    if q.numel() < 2:
        raise ValueError(f"quantiles length must be >=2, got {q.numel()}")

    # error: [B,Q,H]
    err = y_hat_q - y.unsqueeze(1)

    # pinball per q: max(q*(y-yhat), (q-1)*(y-yhat)) but with err = yhat - y
    # -> (q * max(-err,0) + (1-q)*max(err,0))
    sq = torch.clamp(-err, min=0.0)
    s1 = torch.clamp(err, min=0.0)
    loss = q.view(1, -1, 1) * sq + (1.0 - q.view(1, -1, 1)) * s1

    if mask is not None:
        loss = loss * mask

    return loss.mean()


# =============================================================================
# 3) Spike & asym losses (point)
# =============================================================================
def huber_piecewise(err: Tensor, delta: float) -> Tensor:
    abs_e = err.abs()
    return torch.where(abs_e <= delta, 0.5 * err * err, delta * (abs_e - 0.5 * delta))


def huber_asymmetric(
    pred: Tensor,
    y: Tensor,
    *,
    delta: float = 2.0,
    up_w: float = 2.0,
    down_w: float = 1.0,
    weight: Optional[Tensor] = None,
) -> Tensor:
    """
    Asymmetric Huber on (pred - y).
    - Over-prediction (err>0) weighted by up_w; under by down_w.
    - Optional multiplicative weight [B,H].
    """
    pred = ensure_2d(pred); y = ensure_2d(y)
    err = pred - y
    hub = huber_piecewise(err, delta=delta)
    asym = torch.where(err > 0, torch.full_like(hub, up_w), torch.full_like(hub, down_w))
    loss = asym * hub
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def asymmetric_mse(
    pred: Tensor,
    y: Tensor,
    *,
    up_w: float = 2.0,
    down_w: float = 1.0,
    weight: Optional[Tensor] = None,
) -> Tensor:
    """
    Asymmetric MSE on (pred - y).
    """
    pred = ensure_2d(pred); y = ensure_2d(y)
    err = pred - y
    w_asym = torch.where(err > 0, torch.full_like(err, up_w), torch.full_like(err, down_w))
    loss = w_asym * (err ** 2)
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def mad_spike_weights(
    y_ref: Tensor,
    *,
    k: float = 3.5,
    w_spike: float = 6.0,
    w_norm: float = 1.0,
    eps: float = 1e-6,
) -> Tensor:
    """
    MAD-based spike weight computed from reference series y_ref ([B,H] or [B,1,H]).
    Returns [B,H] weights.
    """
    y_ref = ensure_2d(y_ref)
    med = torch.median(y_ref, dim=1, keepdim=True).values
    mad = torch.median(torch.abs(y_ref - med), dim=1, keepdim=True).values + eps
    z = (y_ref - med) / mad
    spike = (z > k).float()
    return torch.where(spike > 0, torch.full_like(y_ref, w_spike), torch.full_like(y_ref, w_norm))


def horizon_decay_weights(y: Tensor, *, tau_h: float = 1.0) -> Tensor:
    """
    Horizon weights: tau_h**t for t=0..H-1, then normalized to mean=1.
    y: [B,H] or compatible
    Returns: [B,H]
    """
    y = ensure_2d(y)
    H = y.size(1)
    hw = (tau_h ** torch.arange(H, device=y.device, dtype=y.dtype))
    hw = hw / (hw.mean() + 1e-12)
    return hw.unsqueeze(0).expand(y.size(0), H)


# =============================================================================
# 4) Output adapters (pred dict -> tensor)
# =============================================================================
def extract_point_pred(pred: Pred, *, q50_index: Optional[int] = None) -> Tensor:
    """
    Normalize model output to point prediction [B,H].
    Supported:
      - Tensor: [B,H] | [B,H,1] | [B,1,H]
      - Dict: {"point": ...} or {"y_pred": ...} or {"q": [B,Q,H] or [B,Q,H,C]}
            if q: choose q50 if possible else middle index
    """
    if torch.is_tensor(pred):
        return ensure_2d(pred)

    if not isinstance(pred, dict):
        raise ValueError(f"Unsupported pred type: {type(pred)}")

    if "point" in pred:
        return ensure_2d(pred["point"])

    if "y_pred" in pred:
        return ensure_2d(pred["y_pred"])

    if "q" in pred:
        q = pred["q"]
        if not torch.is_tensor(q):
            raise ValueError("pred['q'] must be a Tensor")

        # q: [B,Q,H] or [B,Q,H,C]
        if q.dim() == 4:
            # choose channel 0 by default (caller should pre-select if needed)
            q = q[..., 0]  # -> [B,Q,H]
        if q.dim() != 3:
            raise ValueError(f"Unsupported q shape: {tuple(q.shape)}")

        B, Q, H = q.shape
        idx = q50_index
        if idx is None:
            idx = Q // 2  # middle
        idx = int(max(0, min(idx, Q - 1)))
        return ensure_2d(q[:, idx, :])

    # fallback: first tensor value
    for v in pred.values():
        if torch.is_tensor(v):
            return ensure_2d(v)

    raise ValueError("No tensor found in pred dict.")


def extract_quantile_pred(pred: Pred, *, target_channel: int = 0) -> Tensor:
    """
    Normalize model output to quantile prediction [B,Q,H].
    Supported:
      - Tensor [B,Q,H]
      - Dict {"q": [B,Q,H] or [B,Q,H,C]}  -> selects target_channel if C exists.
    """
    if torch.is_tensor(pred):
        return ensure_3d_quantile(pred)

    if not isinstance(pred, dict) or "q" not in pred:
        raise ValueError("Quantile mode requires pred Tensor [B,Q,H] or dict with key 'q'.")

    q = pred["q"]
    if q.dim() == 4:
        C = q.size(-1)
        ch = int(max(0, min(target_channel, C - 1)))
        q = q[..., ch]  # [B,Q,H]
    return ensure_3d_quantile(q)



# =============================================================================
# 4.5) Distribution losses (dist mode)
# =============================================================================
def ensure_bh(x: Tensor) -> Tensor:
    """Normalize tensor to [B,H]. Accepts [B,H], [B,1,H], [B,H,1]."""
    return ensure_2d(x)


def normal_nll_elementwise(y: Tensor, loc: Tensor, scale: Tensor, *, eps: float = 1e-12) -> Tensor:
    """
    Elementwise Normal negative log-likelihood for y ~ N(loc, scale^2).
    Returns [B,H].
    """
    y = ensure_2d(y)
    loc = ensure_bh(loc)
    scale = ensure_bh(scale).clamp_min(eps)

    z = (y - loc) / scale
    # 0.5*log(2*pi) as constant (keep dtype/device)
    const = y.new_tensor(0.5 * math.log(2.0 * math.pi))
    return const + torch.log(scale) + 0.5 * (z ** 2)


def extract_dist_params(pred: Pred) -> Tuple[Tensor, Tensor]:
    """
    Extract (loc, scale) from model output for dist mode.
    Supported:
      - Dict with keys:
          * "loc" and "scale"
          * "loc" and "scale_raw" (caller may softplus)
          * (compat) "mu"/"sigma"
    Returns:
      loc: [B,H], scale_like: [B,H]
    """
    if not isinstance(pred, dict):
        raise ValueError(f"dist mode requires pred as dict, got {type(pred)}")

    if "loc" in pred:
        loc = pred["loc"]
    elif "mu" in pred:
        loc = pred["mu"]
    else:
        raise ValueError("dist mode requires key 'loc' (or 'mu').")

    if "scale" in pred:
        scale = pred["scale"]
    elif "sigma" in pred:
        scale = pred["sigma"]
    elif "scale_raw" in pred:
        scale = pred["scale_raw"]
    else:
        raise ValueError("dist mode requires key 'scale' (or 'sigma'/'scale_raw').")

    return ensure_bh(loc), ensure_bh(scale)


# =============================================================================
# 6) Optional hooks: 프로젝트 내부 util이 있으면 여기로 연결
# =============================================================================
def default_newsvendor_q_star(Cu: float, Co: float) -> float:
    Cu = float(Cu); Co = float(Co)
    return Cu / (Cu + Co + 1e-12)


def default_intermittent_weights_balanced(
    y: Tensor,
    alpha_zero: float,
    alpha_pos: float,
    gamma_run: float,
    clip_run: float = 0.5,
) -> Tensor:
    """
    프로젝트 내부 intermittent weighting이 없다면 최소 동작.
    """
    y = ensure_2d(y)
    is_zero = (y <= 0)
    return torch.where(
        is_zero,
        torch.full_like(y, float(alpha_zero)),
        torch.full_like(y, float(alpha_pos)),
    )


# =============================================================================
# 7) LossComputer (single entry point)
# =============================================================================
class LossComputer:
    """
    단일 엔트리포인트 compute()로 손실을 산출.

    - Quantile:
        multi-quantile pinball (optional intermittent weights)
    - Point:
        mae/mse/huber/pinball(q*)/huber_asym
    - Spike:
        cfg.spike_loss.enabled일 때:
          - strategy="mix": MAD spike weight + horizon weight + (Huber + AsymMSE) blend
          - strategy="direct": huber_asymmetric(단일)
    """

    def __init__(
        self,
        cfg: TrainingConfig,
        *,
        newsvendor_q_star_fn=default_newsvendor_q_star,
        intermittent_weights_fn=default_intermittent_weights_balanced,
    ):
        self.cfg = cfg
        self._newsvendor_q_star_fn = newsvendor_q_star_fn
        self._intermittent_weights_fn = intermittent_weights_fn

    def _q_star(self) -> float:
        if self.cfg.use_cost_q_star and self.cfg.point_loss == "pinball":
            return float(self._newsvendor_q_star_fn(self.cfg.Cu, self.cfg.Co))
        return float(self.cfg.q_star)

    def _maybe_intermittent_weights(self, y: Tensor) -> Optional[Tensor]:
        if not self.cfg.use_intermittent:
            return None
        return self._intermittent_weights_fn(
            ensure_2d(y),
            alpha_zero=self.cfg.alpha_zero,
            alpha_pos=self.cfg.alpha_pos,
            gamma_run=self.cfg.gamma_run,
            clip_run=0.5,
        )

    def _effective_horizon_weights(self, y: Tensor) -> Tensor:
        if not self.cfg.use_horizon_decay:
            y2 = ensure_2d(y)
            return torch.ones_like(y2)
        return horizon_decay_weights(y, tau_h=float(self.cfg.tau_h))

    def _infer_mode(self, pred: Pred) -> str:
        # explicit override
        if self.cfg.loss_mode != "auto":
            return str(self.cfg.loss_mode)

        # custom loss hint (auto mode)
        # - NeuralForecast-style DistributionLoss sets `is_distribution_output=True`.
        if self.cfg.custom_loss is not None and getattr(self.cfg.custom_loss, "is_distribution_output", False):
            return "dist"

        # auto detect quantile
        if torch.is_tensor(pred):
            if pred.dim() == 3 and pred.size(1) > 1:  # [B,Q,H]
                return "quantile"
            return "point"

        if isinstance(pred, dict):
            if "q" in pred:
                return "quantile"
            # dist signature: loc + (scale|scale_raw|sigma)
            if ("loc" in pred or "mu" in pred) and ("scale" in pred or "scale_raw" in pred or "sigma" in pred):
                return "dist"

        return "point"


    # -------------------------------------------------------------------------
    # public
    # -------------------------------------------------------------------------
    def compute(self, pred: Pred, y: Tensor, *, is_val: bool) -> Tensor:
        mode = self._infer_mode(pred)

        if mode == "dist":
            # Prefer custom distribution loss if provided (e.g., DistributionLoss).
            if self.cfg.custom_loss is not None and getattr(self.cfg.custom_loss, "is_distribution_output", False):
                return self._compute_dist_custom(pred, y, is_val=is_val)
            return self._compute_dist(pred, y, is_val=is_val)

        if mode == "quantile":
            return self._compute_quantile(pred, y, is_val=is_val)

        # point mode
        sl = self.cfg.spike_loss
        if sl.enabled:
            if sl.strategy == "mix":
                return self._compute_spike_mix(pred, y, is_val=is_val)
            if sl.strategy == "direct" and self.cfg.point_loss == "huber_asym":
                return self._compute_spike_direct(pred, y, is_val=is_val)

        return self._compute_point_base(pred, y, is_val=is_val)


    # -------------------------------------------------------------------------
    # quantile
    # -------------------------------------------------------------------------
    def _compute_quantile(self, pred: Tensor, y: Tensor, is_val: bool) -> Tensor:
        y2 = ensure_2d(y)
        q_pred = extract_quantile_pred(pred)  # expected [B,Q,H], but may come as [B,H,Q]

        # ---------------------------
        # FIX: enforce [B, Q, H]
        # ---------------------------
        q = len(self.cfg.quantiles)
        if q_pred.dim() != 3:
            raise RuntimeError(f"[LossComputer] q_pred must be 3D, got {q_pred.shape}")

        # Case A: already [B,Q,H]
        if q_pred.shape[1] == q:
            pass
        # Case B: [B,H,Q] -> transpose to [B,Q,H]
        elif q_pred.shape[2] == q:
            q_pred = q_pred.permute(0, 2, 1).contiguous()
        else:
            raise RuntimeError(
                f"[LossComputer] cannot infer quantile axis. "
                f"q_pred shape={tuple(q_pred.shape)} vs q_len={q}"
            )

        # weights
        weights = self._maybe_intermittent_weights(y2)  # <-- 사용자 코드에 존재하는 메서드
        if (is_val is True) and (self.cfg.val_use_weights is False):
            weights = torch.ones_like(weights)

        # expand weights to [B,Q,H]
        w = weights.unsqueeze(1).expand(-1, q_pred.size(1), -1)
        return multi_quantile_loss(y2, q_pred, self.cfg.quantiles, mask=w)


    # -------------------------------------------------------------------------
    # dist (distribution)
    # -------------------------------------------------------------------------
    def _compute_dist(self, pred: Pred, y: Tensor, *, is_val: bool) -> Tensor:
        """
        Distribution loss (currently Normal NLL).

        Expected pred:
          - dict with keys:
              * 'loc': [B,H]
              * 'scale': [B,H] (positive) OR 'scale_raw' (unconstrained)

        Weighting policy:
          - If cfg.use_intermittent: multiply elementwise weights (normalized) to NLL map.
          - If cfg.use_horizon_decay: multiply horizon weights (normalized) to NLL map.
          - If is_val and cfg.val_use_weights=False: disable all weights.
        """
        y2 = ensure_2d(y)

        loc, scale_like = extract_dist_params(pred)

        min_scale = float(self.cfg.dist_min_scale)

        # 예: pred가 {"loc": ..., "scale": ...} 또는 {"loc": ..., "scale_raw": ...}
        if "scale_raw" in pred:
            scale = F.softplus(pred["scale_raw"]) + min_scale
        else:
            scale = pred.get("scale", scale_like)  # scale_like는 기존 변수라면 유지
            if not bool(self.cfg.dist_scale_is_positive):
                # cfg가 positive 보장하지 않는다면 보수적으로 변환
                scale = F.softplus(scale) + min_scale

        scale = scale.clamp_min(min_scale)
        with torch.no_grad():
            s = scale.detach()
            print("[DBG] scale stats:",
                  float(s.min()), float(s.median()), float(s.mean()), float(s.max()))
            yt = y2.detach()
            print("[DBG] y_true stats:",
                  float(yt.min()), float(yt.median()), float(yt.mean()), float(yt.max()))
            err = (yt - loc).detach()
            print("[DBG] |err| median/max:",
                  float(err.abs().median()), float(err.abs().max()))

        if str(self.cfg.dist_family).lower() != "normal":
            raise ValueError(f"Unsupported dist_family={self.cfg.dist_family!r}. Only 'normal' is implemented.")

        nll_map = normal_nll_elementwise(y2, loc, scale, eps=float(self.cfg.dist_eps))  # [B,H]

        # disable weights during val if requested
        if is_val and not bool(self.cfg.val_use_weights):
            return nll_map.mean()

        w = torch.ones_like(nll_map)

        # intermittent weights
        iw = self._maybe_intermittent_weights(y2)
        if iw is not None:
            w = w * iw

        # horizon decay weights
        hw = self._effective_horizon_weights(y2)
        if hw is not None:
            w = w * hw

        # normalize weights for stable scale
        w = w / (w.mean() + 1e-12)

        return (w * nll_map).mean()
    # -------------------------------------------------------------------------
    # point base
    # -------------------------------------------------------------------------

    def _pack_distributionloss_args(self, pred: Pred, *, expected_p: int) -> Tensor:
        """Pack model outputs into (B, H, P) tensor expected by NeuralForecast-style DistributionLoss.

        Accepted formats:
          - Tensor: (B, H, P) or (B, P, H) (auto-permuted)
          - Dict: keys matching distribution params (e.g., df/loc/scale or df_raw/scale_raw, mu/sigma aliases)
        """
        if torch.is_tensor(pred):
            if pred.dim() != 3:
                raise ValueError(f"DistributionLoss expects 3D tensor, got shape={tuple(pred.shape)}")
            # Prefer (B,H,P)
            if pred.size(-1) == expected_p:
                return pred
            # Accept (B,P,H)
            if pred.size(1) == expected_p:
                return pred.permute(0, 2, 1).contiguous()
            raise ValueError(
                f"Cannot infer DistributionLoss param layout. expected P={expected_p}, got shape={tuple(pred.shape)}"
            )

        if not isinstance(pred, dict):
            raise ValueError(f"dist pred must be Tensor or dict, got {type(pred)}")

        loss = self.cfg.custom_loss
        param_names = getattr(loss, "param_names", None)
        if not param_names:
            raise ValueError("custom_loss does not expose `param_names`; cannot pack dict predictions.")

        parts: List[Tensor] = []
        for name in param_names:
            k = str(name).lstrip("-")  # '-loc' -> 'loc'
            # common aliases
            aliases = [k, f"{k}_raw"]
            if k == "loc":
                aliases += ["mu", "mean"]
            if k == "scale":
                aliases += ["sigma", "std", "stddev", "scale_raw"]
            v = None
            for a in aliases:
                if a in pred:
                    v = pred[a]
                    break
            if v is None:
                raise KeyError(
                    f"Missing distribution parameter '{k}' in pred dict. Tried aliases={aliases}. keys={list(pred.keys())[:20]}..."
                )
            if v.dim() == 3 and v.size(-1) == 1:
                v = v.squeeze(-1)
            if v.dim() != 2:
                raise ValueError(f"Each dist param must be (B,H) (or (B,H,1)), got {k} shape={tuple(v.shape)}")
            parts.append(v)

        out = torch.stack(parts, dim=2)  # (B,H,P)
        if out.size(-1) != expected_p:
            raise ValueError(f"Packed args P mismatch: expected {expected_p}, got {out.size(-1)}")
        return out

    def _compute_dist_custom(self, pred: Pred, y: Tensor, *, is_val: bool) -> Tensor:
        """Distribution loss via user-provided DistributionLoss (e.g., StudentT).

        Requirements:
          - cfg.custom_loss must be a NeuralForecast-style DistributionLoss (has `is_distribution_output=True`).
          - pred must provide distribution parameters matching custom_loss.param_names order.
        """
        loss_fn = self.cfg.custom_loss
        expected_p = int(getattr(loss_fn, "outputsize_multiplier", 0))
        if expected_p <= 0:
            raise ValueError("custom_loss missing/invalid `outputsize_multiplier`")

        y2 = ensure_2d(y)  # (B,H)
        args = self._pack_distributionloss_args(pred, expected_p=expected_p)  # (B,H,P)

        # weights/mask policy matches normal dist path
        if is_val and (not self.cfg.val_use_weights):
            mask = torch.ones_like(y2)
        else:
            mask = torch.ones_like(y2)
            if self.cfg.dist_use_weights:
                iw = self._maybe_intermittent_weights(y2)
                if iw is not None:
                    mask = mask * iw
                mask = mask * self._effective_horizon_weights(y2)

        # DistributionLoss treats `mask` as weights (via _compute_weights -> weighted_average).
        return loss_fn(y2, args, mask=mask)



    def _compute_point_base(self, pred: Pred, y: Tensor, *, is_val: bool) -> Tensor:
        y2 = ensure_2d(y)
        y_hat = extract_point_pred(pred)  # [B,H]

        pl = self.cfg.point_loss.lower()

        # val에서 가중치 끄기 (intermittent/horizon/spike 모두 off)
        if is_val and not self.cfg.val_use_weights:
            return self._point_loss_plain(y_hat, y2, pl)

        # intermittent 가중치는 외부 intermittent_point_loss가 없으므로 "가중 평균"으로 구현(최소동작)
        w = self._maybe_intermittent_weights(y2)  # [B,H] or None
        if w is None:
            return self._point_loss_plain(y_hat, y2, pl)

        # weighted reduction
        loss_map = self._point_loss_elementwise(y_hat, y2, pl)
        # 평균 스케일 안정화를 위해 w 평균으로 정규화
        w_eff = w / (w.mean() + 1e-12)
        return (w_eff * loss_map).mean()

    def _point_loss_plain(self, y_hat: Tensor, y: Tensor, pl: str) -> Tensor:
        if pl == "mae":
            return mae_loss(y, y_hat)
        if pl == "mse":
            return mse_loss(y, y_hat)
        if pl == "huber":
            return F.huber_loss(y_hat, y, delta=float(self.cfg.huber_delta))
        if pl == "pinball":
            return pinball_loss(y, y_hat, tau=float(self._q_star()))
        if pl == "huber_asym":
            sl = self.cfg.spike_loss
            return huber_asymmetric(
                y_hat, y,
                delta=float(getattr(sl, "huber_delta", self.cfg.huber_delta)),
                up_w=float(getattr(sl, "asym_up_weight", 2.0)),
                down_w=float(getattr(sl, "asym_down_weight", 1.0)),
                weight=None,
            )
        # default
        return mae_loss(y, y_hat)

    def _point_loss_elementwise(self, y_hat: Tensor, y: Tensor, pl: str) -> Tensor:
        """
        elementwise loss map [B,H] for weighting.
        """
        if pl == "mae":
            return (y_hat - y).abs()
        if pl == "mse":
            return (y_hat - y) ** 2
        if pl == "huber":
            err = y_hat - y
            return huber_piecewise(err, delta=float(self.cfg.huber_delta))
        if pl == "pinball":
            tau = float(self._q_star())
            diff = y - y_hat
            return torch.maximum(tau * diff, (tau - 1.0) * diff)
        if pl == "huber_asym":
            sl = self.cfg.spike_loss
            err = y_hat - y
            hub = huber_piecewise(err, delta=float(getattr(sl, "huber_delta", self.cfg.huber_delta)))
            asym = torch.where(
                err > 0,
                torch.full_like(hub, float(getattr(sl, "asym_up_weight", 2.0))),
                torch.full_like(hub, float(getattr(sl, "asym_down_weight", 1.0))),
            )
            return asym * hub
        return (y_hat - y).abs()

    # -------------------------------------------------------------------------
    # spike mix/direct
    # -------------------------------------------------------------------------
    def _compute_spike_mix(self, pred: Pred, y: Tensor, *, is_val: bool) -> Tensor:
        y2 = ensure_2d(y)
        y_hat = extract_point_pred(pred)  # [B,H]
        sl = self.cfg.spike_loss

        # val에서 가중치 끄기
        if is_val and not self.cfg.val_use_weights:
            # spike 로직을 비활성화한 plain point로 fallback
            return self._compute_point_base(pred, y2, is_val=True)

        # spike weights (MAD) & horizon weights
        w_sp = mad_spike_weights(
            y2, k=float(sl.mad_k), w_spike=float(sl.w_spike), w_norm=float(sl.w_norm)
        )  # [B,H]
        w_hz = self._effective_horizon_weights(y2)  # [B,H]

        w_eff = w_sp * w_hz

        # intermittent weights까지 곱하고 싶으면 여기에 확장 가능(현재는 point base에서만)
        # 스케일 안정화
        w_eff = w_eff / (w_eff.mean() + 1e-12)

        # cap to avoid exploding gradients
        if float(sl.w_cap) > 0:
            w_eff = torch.clamp(w_eff, max=float(sl.w_cap))

        # Huber component
        err = y_hat - y2
        hub = huber_piecewise(err, delta=float(sl.huber_delta))
        loss_huber = (w_eff * hub).mean()

        # Asym MSE component
        loss_asym = asymmetric_mse(
            y_hat, y2,
            up_w=float(sl.asym_up_weight),
            down_w=float(sl.asym_down_weight),
            weight=w_eff,
        )

        loss = float(sl.alpha_huber) * loss_huber + float(sl.beta_asym) * loss_asym

        if sl.mix_with_baseline:
            base = self._compute_point_base(pred, y2, is_val=is_val)
            loss = loss + float(sl.gamma_baseline) * base

        return loss

    def _compute_spike_direct(self, pred: Pred, y: Tensor, *, is_val: bool) -> Tensor:
        y2 = ensure_2d(y)
        y_hat = extract_point_pred(pred)
        sl = self.cfg.spike_loss

        if is_val and not self.cfg.val_use_weights:
            # direct도 가중 off
            return huber_asymmetric(
                y_hat, y2,
                delta=float(sl.huber_delta),
                up_w=float(sl.asym_up_weight),
                down_w=float(sl.asym_down_weight),
                weight=None,
            )

        # optional intermittent weight
        w = self._maybe_intermittent_weights(y2)
        if w is not None:
            w = w / (w.mean() + 1e-12)

        return huber_asymmetric(
            y_hat, y2,
            delta=float(sl.huber_delta),
            up_w=float(sl.asym_up_weight),
            down_w=float(sl.asym_down_weight),
            weight=w,
        )


# =============================================================================
# 8) PSPA (Past Scale & Pattern Anchoring) - cleaned
# =============================================================================
def make_pspa_fn(
    lambda_pattern: float = 0.20,
    lambda_scale: float = 0.10,
    target_channel: int = 0,
):
    """
    PSPA: Past Scale & Pattern Anchoring
    - Pattern: cosine distance between recent diffs of history and predicted diffs
    - Scale  : ratio of abs-mean scale between history segment and prediction segment
    """

    eps = 1e-8

    def _to_pred_bhc(out: Pred) -> Tensor:
        """
        Normalize output to [B,H,1] for a single target channel.
        Supported:
          - Tensor [B,H] or [B,H,C]
          - Dict {"y_pred": ...} or {"q": [B,Q,H] or [B,Q,H,C]} (uses median quantile index)
        """
        if torch.is_tensor(out):
            y = out
        elif isinstance(out, dict):
            if "y_pred" in out:
                y = out["y_pred"]
            elif "q" in out:
                q = out["q"]
                if q.dim() == 4:
                    # [B,Q,H,C] -> select channel
                    C = q.size(-1)
                    ch = int(max(0, min(target_channel, C - 1)))
                    q = q[..., ch]  # [B,Q,H]
                if q.dim() != 3:
                    raise ValueError(f"Unsupported q shape: {tuple(q.shape)}")
                q_idx = q.size(1) // 2
                y = q[:, q_idx, :]  # [B,H]
            else:
                raise ValueError("Unsupported pred dict keys (need 'y_pred' or 'q').")
        else:
            raise ValueError(f"Unsupported out type: {type(out)}")

        # [B,H] -> [B,H,1]
        if y.dim() == 2:
            y = y.unsqueeze(-1)
        # [B,H,C] -> select channel -> [B,H,1]
        elif y.dim() == 3:
            C = y.size(-1)
            ch = int(max(0, min(target_channel, C - 1)))
            y = y[..., ch : ch + 1]
        else:
            raise ValueError(f"Unsupported pred shape: {tuple(y.shape)}")

        return y

    def _select_hist_blc(x: Tensor) -> Tensor:
        """
        x: [B,L,C] -> [B,L,1] by selecting target_channel
        """
        if x.dim() != 3:
            raise ValueError(f"x must be [B,L,C], got {tuple(x.shape)}")
        C = x.size(-1)
        ch = int(max(0, min(target_channel, C - 1)))
        return x[:, :, ch : ch + 1]

    def _pspa(x: Tensor, out: Pred, cfg: Any = None) -> Tensor:
        y = _to_pred_bhc(out)          # [B,H,1]
        x_ch = _select_hist_blc(x)     # [B,L,1]

        # diffs
        diff_hist = x_ch[:, 1:, :] - x_ch[:, :-1, :]   # [B,L-1,1]
        diff_pred = y[:, 1:, :] - y[:, :-1, :]         # [B,H-1,1]

        T = min(diff_hist.size(1), diff_pred.size(1))
        if T < 1:
            return y.new_tensor(0.0)

        dh = diff_hist[:, -T:, :].reshape(diff_hist.size(0), -1)  # [B,T]
        dp = diff_pred[:, :T, :].reshape(diff_pred.size(0), -1)   # [B,T]

        # pattern (cosine distance)
        cos = F.cosine_similarity(dh, dp, dim=1)  # [B]
        L_pattern = (1.0 - cos).mean()

        # scale anchoring
        hist_seg = x_ch[:, -T:, :]   # [B,T,1]
        pred_seg = y[:, :T, :]       # [B,T,1]
        scale_hist = hist_seg.abs().mean(dim=(1, 2)) + eps
        scale_pred = pred_seg.abs().mean(dim=(1, 2)) + eps
        L_scale = ((scale_pred / scale_hist) - 1.0).abs().mean()

        return float(lambda_pattern) * L_pattern + float(lambda_scale) * L_scale

    return _pspa
#
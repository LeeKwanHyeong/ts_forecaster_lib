"""
losses.py

Loss computation utilities.

[Refactor note]
- Prefer using TrainingConfig.loss (a loss object) rather than string-based loss_mode/point_loss/dist_loss.
- Keep backward compatibility: if cfg.loss is None, fall back to legacy fields.

Shape normalization (conservative):
- y: [B,H] or [B,H,1]          -> [B,H,1]
- point y_hat: [B,H] / [B,1,H] / [B,H,1] -> [B,H,1]
- quantile y_hat:
    [B,H,Q] / [B,Q,H] / [B,H,1,Q] / [B,Q,H,1] -> normalized for each loss type
- distribution y_hat (packed):
    [B,H,M] / [B,H,1,M] / [B,1,H,M] -> [B,H,M]
    where M == loss.outputsize_multiplier

This module only normalizes shapes and maps raw network outputs to valid distribution args.
Actual loss math is delegated to loss objects in loss_module.py.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling_module.training.model_losses.loss_module import (
    MAE,
    Huber,
    QuantileLoss,
    MQLoss,
    DistributionLoss,
)


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------
def _ensure_3d_y(y: torch.Tensor) -> torch.Tensor:
    """y -> [B,H,1]"""
    if y.dim() == 2:
        return y.unsqueeze(-1)
    if y.dim() == 3:
        return y
    raise ValueError(f"y must be 2D/3D, got shape={tuple(y.shape)}")


def _mask_from_y(y_bh1: torch.Tensor) -> torch.Tensor:
    """Finite mask: [B,H,1]"""
    return torch.isfinite(y_bh1).float()


def _ensure_point_yhat(y_hat: torch.Tensor, y_bh1: torch.Tensor) -> torch.Tensor:
    """point y_hat -> [B,H,1] aligned with y=[B,H,1]."""
    if y_hat.dim() == 2:
        y_hat = y_hat.unsqueeze(-1)
    elif y_hat.dim() != 3:
        raise ValueError(f"point y_hat must be 2D/3D, got {tuple(y_hat.shape)}")

    # [B,1,H] -> [B,H,1]
    if y_hat.shape[1] == 1 and y_hat.shape[2] == y_bh1.shape[1]:
        y_hat = y_hat.permute(0, 2, 1).contiguous()

    # now should be [B,H,1] or [B,H,C] (C>1이면 point loss에서 에러 내는 게 맞음)
    return y_hat


def _ensure_quantile_yhat(y_hat: torch.Tensor, y_bh1: torch.Tensor) -> torch.Tensor:
    """
    quantile y_hat -> [B,H,Q] or [B,H,1,Q].

    Accept:
    - [B,H,Q]
    - [B,Q,H]       -> permute to [B,H,Q]
    - [B,H,1,Q]
    - [B,Q,H,1]     -> permute to [B,H,1,Q]
    - [B,1,H,Q]     -> permute to [B,H,1,Q]
    """
    H = y_bh1.shape[1]

    if y_hat.dim() == 3:
        # [B,Q,H] -> [B,H,Q]
        if y_hat.shape[1] != H and y_hat.shape[2] == H:
            y_hat = y_hat.permute(0, 2, 1).contiguous()
        return y_hat

    if y_hat.dim() == 4:
        # [B,Q,H,1] -> [B,H,1,Q]
        if y_hat.shape[2] == H and y_hat.shape[-1] == 1 and y_hat.shape[1] != H:
            y_hat = y_hat.permute(0, 2, 3, 1).contiguous()
            return y_hat

        # [B,1,H,Q] -> [B,H,1,Q]
        if y_hat.shape[1] == 1 and y_hat.shape[2] == H:
            y_hat = y_hat.permute(0, 2, 1, 3).contiguous()
            return y_hat

        # already [B,H,1,Q] (or [B,H,N,Q])
        return y_hat

    raise ValueError(f"quantile y_hat must be 3D/4D, got {tuple(y_hat.shape)}")


def _ensure_dist_yhat_packed(y_hat: torch.Tensor, y_bh1: torch.Tensor) -> torch.Tensor:
    """
    distribution packed y_hat -> [B,H,M]

    Accept:
    - [B,H,M]
    - [B,H,1,M] -> squeeze dim=2
    - [B,1,H,M] -> permute to [B,H,M]
    """
    H = y_bh1.shape[1]

    if y_hat.dim() == 3:
        return y_hat

    if y_hat.dim() == 4:
        # [B,H,1,M] -> [B,H,M]
        if y_hat.shape[1] == H and y_hat.shape[2] == 1:
            return y_hat.squeeze(2)

        # [B,1,H,M] -> [B,H,M]
        if y_hat.shape[1] == 1 and y_hat.shape[2] == H:
            return y_hat.permute(0, 2, 1, 3).contiguous().squeeze(2)

    raise ValueError(f"distribution packed y_hat must be 3D/4D, got {tuple(y_hat.shape)}")


def _strip_param_names(raw: List[Any]) -> List[str]:
    return [str(x).lstrip("-").lower() for x in raw]


def _positive(x: torch.Tensor, transform: str) -> torch.Tensor:
    transform = (transform or "softplus").lower()
    if transform in ("exp", "exponential"):
        return torch.exp(x)
    if transform in ("relu",):
        return F.relu(x)
    # default
    return F.softplus(x)


# ---------------------------------------------------------------------
# LossComputer
# ---------------------------------------------------------------------
class LossComputer:
    """
    Compute loss for a batch given TrainingConfig.

    Preferred:
        cfg.loss: nn.Module
    Backward-compat:
        cfg.loss_mode / cfg.point_loss / cfg.dist_loss
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.loss_fn = getattr(cfg, "loss", None)

        # fallback: legacy
        if self.loss_fn is None:
            loss_mode = getattr(cfg, "loss_mode", "point")
            if loss_mode in ("dist", "distribution"):
                self.loss_fn = DistributionLoss("Normal")
            elif loss_mode == "quantile":
                self.loss_fn = MQLoss(quantiles=[0.1, 0.5, 0.9])
            else:
                point_loss = getattr(cfg, "point_loss", "mae")
                if point_loss == "huber":
                    self.loss_fn = Huber(delta=getattr(cfg, "huber_delta", 1.0))
                else:
                    self.loss_fn = MAE()

    def compute(
        self,
        y_hat: Any,
        y: torch.Tensor,
        is_val: bool,
        *,
        y_insample: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y_bh1 = _ensure_3d_y(y)
        mask = _mask_from_y(y_bh1)

        # ------------------------------------------------------------
        # Unwrap dict outputs (model may return dict)
        # ------------------------------------------------------------
        if isinstance(y_hat, dict):
            if "y_hat" in y_hat:
                y_hat = y_hat["y_hat"]
            elif "loc" in y_hat and ("scale_raw" in y_hat or "scale" in y_hat):
                loc = y_hat["loc"]
                scale_raw = y_hat.get("scale_raw", y_hat.get("scale"))
                df_raw = y_hat.get("df_raw", y_hat.get("df", None))

                if loc.dim() == 2:
                    loc = loc.unsqueeze(-1)
                if scale_raw.dim() == 2:
                    scale_raw = scale_raw.unsqueeze(-1)
                if df_raw is not None and df_raw.dim() == 2:
                    df_raw = df_raw.unsqueeze(-1)

                # pack to [B,H,M]
                if df_raw is not None:
                    y_hat = torch.cat([df_raw, loc, scale_raw], dim=-1)
                else:
                    y_hat = torch.cat([loc, scale_raw], dim=-1)
            else:
                # first tensor-like
                y_hat = next((v for v in y_hat.values() if torch.is_tensor(v)), y_hat)


        name = self.loss_fn.__class__.__name__

        # ------------------------------------------------------------
        # DistributionLoss (Normal/StudentT 등)
        # ------------------------------------------------------------
        if name == "DistributionLoss":
            if not torch.is_tensor(y_hat):
                raise TypeError(f"DistributionLoss expects tensor or dict output, got {type(y_hat)}")

            # config knobs
            eps = float(getattr(self.cfg, "dist_eps", 1e-8))
            min_scale = float(getattr(self.cfg, "dist_min_scale", 0.0))
            df_min = float(getattr(self.cfg, "dist_min_df", 2.0))
            scale_is_positive = bool(getattr(self.cfg, "dist_scale_is_positive", True))
            scale_transform = str(getattr(self.cfg, "dist_scale_transform", "softplus"))

            # distribution name
            distr_name = str(
                getattr(self.loss_fn, "distribution", getattr(self.loss_fn, "dist_name", getattr(self.cfg, "dist_name", "normal")))
            ).lower()

            # expected multiplier
            m = int(getattr(self.loss_fn, "outputsize_multiplier", 2))

            y_hat_bhm = _ensure_dist_yhat_packed(y_hat, y_bh1)

            if y_hat_bhm.shape[-1] != m:
                raise ValueError(
                    f"DistributionLoss expects last dim == outputsize_multiplier={m} "
                    f"(dist={distr_name}), got y_hat shape {tuple(y_hat_bhm.shape)}. "
                    f"→ 모델 head out_features(out_mult)와 loss.outputsize_multiplier가 불일치입니다."
                )

            parts = torch.tensor_split(y_hat_bhm, m, dim=-1)  # each [B,H,1]

            raw_names = list(getattr(self.loss_fn, "param_names", []))
            names = _strip_param_names(raw_names)

            def _idx(key: str, default: Optional[int]) -> Optional[int]:
                key = key.lower()
                return names.index(key) if key in names else default

            # default ordering (when param_names absent)
            if m == 2:
                loc_i = _idx("loc", 0)
                scale_i = _idx("scale", 1)
                df_i = None
            else:
                # StudentT commonly: [df, loc, scale]
                df_i = _idx("df", 0)
                loc_i = _idx("loc", 1)
                scale_i = _idx("scale", 2)

            if loc_i is None or scale_i is None:
                raise ValueError(f"Cannot infer loc/scale indices from param_names={raw_names} (m={m})")

            loc = parts[loc_i]
            scale_raw = parts[scale_i]

            if scale_is_positive:
                scale = _positive(scale_raw, scale_transform)
            else:
                scale = scale_raw
            scale = scale + (min_scale + eps)

            if df_i is not None and ("studentt" in distr_name or m >= 3):
                df_raw = parts[df_i]
                df = _positive(df_raw, scale_transform) + df_min
                distr_args = (df, loc, scale)
            else:
                distr_args = (loc, scale)

            # some implementations want [B,H] instead of [B,H,1]
            try:
                return self.loss_fn(y=y_bh1, distr_args=distr_args, mask=mask)
            except TypeError:
                y2 = y_bh1.squeeze(-1)
                mask2 = mask.squeeze(-1) if mask is not None and mask.dim() == 3 else mask
                distr_args2 = tuple(t.squeeze(-1) for t in distr_args)
                return self.loss_fn(y=y2, distr_args=distr_args2, mask=mask2)

        # ------------------------------------------------------------
        # Quantile losses
        # ------------------------------------------------------------
        if name == "MQLoss":
            if not torch.is_tensor(y_hat):
                raise TypeError(f"MQLoss expects tensor y_hat, got {type(y_hat)}")

            qhat = _ensure_quantile_yhat(y_hat, y_bh1)
            # MQLoss expects [B,H,N,Q] (your implementation uses N=1)
            if qhat.dim() == 3:
                qhat = qhat.unsqueeze(2)  # [B,H,1,Q]
            return self.loss_fn(y=y_bh1, y_hat=qhat, y_insample=y_insample, mask=mask)

        if name == "QuantileLoss":
            if not torch.is_tensor(y_hat):
                raise TypeError(f"QuantileLoss expects tensor y_hat, got {type(y_hat)}")

            qhat = _ensure_quantile_yhat(y_hat, y_bh1)
            # QuantileLoss typically expects [B,H,Q]
            if qhat.dim() == 4 and qhat.shape[2] == 1:
                qhat = qhat.squeeze(2)
            return self.loss_fn(y=y_bh1, y_hat=qhat, y_insample=y_insample, mask=mask)

        if name == "MultiQuantilePinball":
            if not torch.is_tensor(y_hat):
                raise TypeError(f"MultiQuantilePinball expects tensor y_hat, got {type(y_hat)}")

            qhat = _ensure_quantile_yhat(y_hat, y_bh1)

            # (선택) [B,H,1,Q] 같은 형태면 N=1 squeeze
            if qhat.dim() == 4 and qhat.shape[2] == 1:
                qhat = qhat.squeeze(2)  # -> [B,H,Q]
            # print("[DBG] y    :", y_bh1.shape, y_bh1.min().item(), y_bh1.max().item(), y_bh1.mean().item())
            # print("[DBG] y_hat:", qhat.shape, qhat.min().item(), qhat.max().item(), qhat.mean().item())
            # y_insample은 필요 없을 수 있으니 try/except로 흡수
            try:
                return self.loss_fn(y=y_bh1, y_hat=qhat, y_insample=y_insample, mask=mask)
            except TypeError:
                return self.loss_fn(y=y_bh1, y_hat=qhat, mask=mask)

        # ------------------------------------------------------------
        # Default: point loss
        # ------------------------------------------------------------
        if not torch.is_tensor(y_hat):
            raise TypeError(f"Point loss expects tensor y_hat, got {type(y_hat)}")

        phat = _ensure_point_yhat(y_hat, y_bh1)
        return self.loss_fn(y=y_bh1, y_hat=phat, y_insample=y_insample, mask=mask)

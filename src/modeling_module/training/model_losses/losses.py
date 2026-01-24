"""losses.py

Loss computation utilities.

[Refactor note]
- Prefer using TrainingConfig.loss (a loss object) rather than string-based loss_mode/point_loss/dist_loss.
- Keep backward compatibility: if cfg.loss is None, fall back to legacy fields.

Also includes robust tensor-shape normalization for:
- point: y_hat in [B,H] / [B,1,H] / [B,H,1]
- quantile: y_hat in [B,H,Q] / [B,Q,H] / [B,H,1,Q] / [B,Q,H,1]
- distribution (Normal etc.): y_hat in [B,H,2] (loc, scale_raw)

This file intentionally stays conservative: it only normalizes shapes and delegates
actual loss math to the provided loss objects in loss_module.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple

import torch
import torch.nn as nn

from modeling_module.training.model_losses.loss_module import MAE, Huber, QuantileLoss, MQLoss, DistributionLoss
import torch.nn.functional as F

def _ensure_3d_y(y: torch.Tensor) -> torch.Tensor:
    """y -> [B,H,1]"""
    if y.dim() == 2:
        return y.unsqueeze(-1)
    if y.dim() == 3:
        return y
    raise ValueError(f"y must be 2D/3D, got shape={tuple(y.shape)}")


def _ensure_point_yhat(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """point y_hat -> [B,H,1] aligned with y=[B,H,1]."""
    if y_hat.dim() == 2:
        y_hat = y_hat.unsqueeze(-1)
    elif y_hat.dim() == 3:
        pass
    else:
        raise ValueError(f"point y_hat must be 2D/3D, got {tuple(y_hat.shape)}")

    # [B,1,H] -> [B,H,1]
    if y_hat.shape[1] == 1 and y_hat.shape[2] == y.shape[1]:
        y_hat = y_hat.permute(0, 2, 1).contiguous()

    # [B,H] -> [B,H,1] already handled
    return y_hat


def _ensure_quantile_yhat(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """quantile y_hat -> [B,H,1,Q] (for MQLoss) or [B,H,Q] (for QuantileLoss).

    - If input is [B,Q,H], permute -> [B,H,Q]
    - If input is [B,H,Q], keep
    - If input is [B,H,1,Q], keep
    """
    if y_hat.dim() == 4:
        # prefer [B,H,1,Q]
        # if [B,Q,H,1] -> permute
        if y_hat.shape[-1] == 1 and y_hat.shape[2] == y.shape[1]:
            y_hat = y_hat.permute(0, 2, 3, 1).contiguous()  # [B,H,1,Q]
        return y_hat

    if y_hat.dim() != 3:
        raise ValueError(f"quantile y_hat must be 3D/4D, got {tuple(y_hat.shape)}")

    # [B,Q,H] -> [B,H,Q]
    if y_hat.shape[1] != y.shape[1] and y_hat.shape[2] == y.shape[1]:
        y_hat = y_hat.permute(0, 2, 1).contiguous()

    return y_hat


def _mask_from_y(y: torch.Tensor) -> torch.Tensor:
    # NaN/Inf 방지용 기본 mask
    return torch.isfinite(y).float()


class LossComputer:
    """Compute loss for a batch given TrainingConfig.

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
            if loss_mode == "dist":
                self.loss_fn = DistributionLoss("Normal")
            elif loss_mode == "quantile":
                # default multi-quantile (0.1,0.5,0.9)
                self.loss_fn = MQLoss(quantiles=[0.1, 0.5, 0.9])
            else:
                point_loss = getattr(cfg, "point_loss", "mae")
                if point_loss == "huber":
                    self.loss_fn = Huber(delta=getattr(cfg, "huber_delta", 1.0))
                else:
                    self.loss_fn = MAE()

    def compute(self, y_hat: Any, y: torch.Tensor,is_val, *, y_insample: Optional[torch.Tensor] = None) -> torch.Tensor:
        y3 = _ensure_3d_y(y)
        mask = _mask_from_y(y3)

        # handle dict outputs
        if isinstance(y_hat, dict):
            if "y_hat" in y_hat:
                y_hat = y_hat["y_hat"]
            elif "loc" in y_hat and ("scale_raw" in y_hat or "scale" in y_hat):
                loc = y_hat["loc"]
                scale_raw = y_hat.get("scale_raw", y_hat.get("scale"))
                if loc.dim() == 2:
                    loc = loc.unsqueeze(-1)
                if scale_raw.dim() == 2:
                    scale_raw = scale_raw.unsqueeze(-1)
                y_hat = torch.cat([loc, scale_raw], dim=-1)  # [B,H,2]
            else:
                # first tensor-like
                y_hat = next((v for v in y_hat.values() if torch.is_tensor(v)), y_hat)

        # route by loss type
        name = self.loss_fn.__class__.__name__

        # if name in ("DistributionLoss",):
        #     if not torch.is_tensor(y_hat):
        #         raise ValueError(f"DistributionLoss expects tensor y_hat, got {type(y_hat)}")
        #     # y_hat: [B,H,2] for univariate
        #     return self.loss_fn(y=y3, y_hat=y_hat, y_insample=y_insample, mask=mask)

        if name in ("DistributionLoss",):
            # 1) y_hat을 (loc, scale, ...) 튜플로 변환해야 함
            #    - dict 형태면 loc/scale을 꺼내고
            #    - tensor 형태면 마지막 차원을 split

            distr_args = None

            if isinstance(y_hat, dict):
                # PatchTSTDistModel이 dict로 반환하는 케이스를 우선 지원
                if "loc" in y_hat and ("scale" in y_hat or "scale_raw" in y_hat):
                    loc = y_hat["loc"]
                    scale = y_hat.get("scale", None)
                    scale_raw = y_hat.get("scale_raw", None)

                    # shape 정규화: [B,H] -> [B,H,1]
                    if loc.dim() == 2:
                        loc = loc.unsqueeze(-1)

                    if scale is not None:
                        if scale.dim() == 2:
                            scale = scale.unsqueeze(-1)
                    else:
                        # scale_raw만 있으면 양수화(softplus)해서 scale로 만든다
                        if scale_raw is None:
                            raise ValueError("Distribution dict output requires 'scale' or 'scale_raw'.")
                        if scale_raw.dim() == 2:
                            scale_raw = scale_raw.unsqueeze(-1)
                        scale = F.softplus(scale_raw) + 1e-6  # 최소 eps

                    distr_args = (loc, scale)

                elif "y_hat" in y_hat:
                    # 혹시 y_hat key에 packed tensor가 들어오는 케이스
                    y_hat = y_hat["y_hat"]

                else:
                    # dict에서 첫 텐서 pick (fallback)
                    y_hat = next((v for v in y_hat.values() if torch.is_tensor(v)), y_hat)

            if distr_args is None:
                # 2) tensor 케이스: packed tensor -> tuple로 변환
                if not torch.is_tensor(y_hat):
                    raise ValueError(f"DistributionLoss expects tensor or dict, got {type(y_hat)}")

                # 허용 shape 예:
                #   [B,H,2] or [B,H,1,2] or [B,H,N,2]
                # 마지막 차원이 multiplier(여기서는 2: loc, scale_raw/scale)라고 가정
                if y_hat.dim() == 4:
                    # [B,H,1,2] 같은 경우 N축이 1이면 squeeze
                    if y_hat.shape[2] == 1:
                        y_hat = y_hat.squeeze(2)  # -> [B,H,2]
                    else:
                        # 멀티채널이면 우선 첫 채널만 (univariate 학습 기준)
                        # [B,H,N,2] -> [B,H,2] (N=0)
                        y_hat = y_hat[:, :, 0, :]

                if y_hat.dim() != 3:
                    raise ValueError(
                        f"DistributionLoss expects y_hat as [B,H,2] (or [B,H,1,2]), got {tuple(y_hat.shape)}")

                # multiplier = loss_fn.outputsize_multiplier (Normal이면 2)
                m = int(getattr(self.loss_fn, "outputsize_multiplier", 2))
                if y_hat.shape[-1] != m:
                    raise ValueError(
                        f"DistributionLoss expects last dim == outputsize_multiplier={m}, got y_hat shape {tuple(y_hat.shape)}"
                    )

                parts = torch.tensor_split(y_hat, m, dim=-1)  # tuple of [B,H,1]
                # Normal이면 (loc, scale_raw/scale)
                loc = parts[0]
                scale_raw = parts[1]

                # scale이 이미 양수라면 그대로 써도 되지만,
                # scale_raw일 수도 있으니 안전하게 softplus 적용
                scale = F.softplus(scale_raw) + 1e-6

                distr_args = (loc, scale)

            # 3) DistributionLoss는 __call__(y, distr_args, mask) 시그니처
            return self.loss_fn(y=y3, distr_args=distr_args, mask=mask)

        if name in ("MQLoss",):
            if not torch.is_tensor(y_hat):
                raise ValueError(f"MQLoss expects tensor y_hat, got {type(y_hat)}")
            y_hat3 = _ensure_quantile_yhat(y_hat, y3)
            if y_hat3.dim() == 3:
                # [B,H,Q] -> [B,H,1,Q]
                y_hat4 = y_hat3.unsqueeze(2)
            else:
                y_hat4 = y_hat3
            return self.loss_fn(y=y3, y_hat=y_hat4, y_insample=y_insample, mask=mask)

        if name in ("QuantileLoss",):
            if not torch.is_tensor(y_hat):
                raise ValueError(f"QuantileLoss expects tensor y_hat, got {type(y_hat)}")
            y_hat3 = _ensure_quantile_yhat(y_hat, y3)
            if y_hat3.dim() == 4 and y_hat3.shape[2] == 1:
                y_hat3 = y_hat3.squeeze(2)
            return self.loss_fn(y=y3, y_hat=y_hat3, y_insample=y_insample, mask=mask)

        # default: point loss
        if not torch.is_tensor(y_hat):
            raise ValueError(f"Point loss expects tensor y_hat, got {type(y_hat)}")
        y_hat3 = _ensure_point_yhat(y_hat, y3)
        return self.loss_fn(y=y3, y_hat=y_hat3, y_insample=y_insample, mask=mask)

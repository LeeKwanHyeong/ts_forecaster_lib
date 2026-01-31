from typing import Optional

import torch.nn as nn
import torch

from modeling_module.models.PatchTST.common import get_activation_fn


class DistHeadWithExo(nn.Module):
    """
    backbone 출력 (B, N_patch, d_model)을 pool해서 (B, d_model)로 만든 뒤,
    (B, horizon, out_mult)를 출력.
    - out_mult: Normal=2 (loc, scale_raw), StudentT=3 (df, loc, scale_raw)
    """
    def __init__(
        self,
        d_model: int,
        horizon: int,
        d_future: int = 0,
        act: str = "gelu",
        out_mult: int = 2,
        hidden: int = 128,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.out_mult = int(out_mult)
        self.d_future = int(d_future)

        # 미래 외생: (B, H, E) -> (B, H*E) -> (B, d_model)
        self.future_proj = nn.Linear(self.horizon * self.d_future, d_model) if self.d_future > 0 else None
        in_dim = d_model * 2 if self.d_future > 0 else d_model

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            get_activation_fn(act),
            nn.Linear(hidden, self.horizon * self.out_mult),
        )

    def forward(self, h: torch.Tensor, *, future_exo: Optional[torch.Tensor] = None) -> torch.Tensor:
        # h: (B, N_patch, d_model)  -> pool -> (B, d_model)
        if h.dim() != 3:
            raise ValueError(f"[DistHeadWithExo] expected (B,N,D), got {tuple(h.shape)}")
        feat = h.mean(dim=1)  # (B, d_model)

        # future_exo 결합
        if self.d_future > 0:
            if future_exo is None:
                raise ValueError("[DistHeadWithExo] future_exo is required when d_future>0")
            if future_exo.dim() == 2:  # (H,E) -> (B,H,E)
                future_exo = future_exo.unsqueeze(0).expand(h.size(0), -1, -1)
            B, H, E = future_exo.shape
            if H != self.horizon or E != self.d_future:
                raise ValueError(f"[DistHeadWithExo] future_exo shape mismatch: {tuple(future_exo.shape)}")
            f_flat = future_exo.reshape(B, -1)
            f_feat = self.future_proj(f_flat)
            feat = torch.cat([feat, f_feat], dim=-1)  # (B, 2*d_model)

        out = self.net(feat).view(h.size(0), self.horizon, self.out_mult)  # (B, H, out_mult)
        return out
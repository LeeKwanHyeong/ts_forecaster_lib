from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling_module.models.Titan import MemoryAttention


class TitanBackbone(nn.Module):
    """
    Titan block (pre-norm):
      x -> attn -> residual -> ff -> residual
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        contextual_mem_size: int,
        persistent_mem_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MemoryAttention(
            d_model=d_model,
            n_heads=n_heads,
            contextual_mem_size=contextual_mem_size,
            persistent_mem_size=persistent_mem_size,
            dropout=dropout,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # attn
        h = self.norm1(x)
        h = self.attn(h)
        x = x + self.drop1(h)

        # ff
        h = self.norm2(x)
        h = self.ff(h)
        x = x + self.drop2(h)
        return x


class MemoryEncoder(nn.Module):
    """
    Input projection + TitanBackbone x n_layers
    - learnable positional embedding (optional)
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        contextual_mem_size: int,
        persistent_mem_size: int,
        dropout: float = 0.1,
        *,
        use_context_update: bool = False,
        use_pos_emb: bool = True,
        max_len: int = 512,
    ):
        super().__init__()
        self.input_proj = nn.Linear(int(input_dim), int(d_model))

        self.layers = nn.ModuleList(
            [
                TitanBackbone(
                    d_model=int(d_model),
                    n_heads=int(n_heads),
                    d_ff=int(d_ff),
                    contextual_mem_size=int(contextual_mem_size),
                    persistent_mem_size=int(persistent_mem_size),
                    dropout=float(dropout),
                )
                for _ in range(int(n_layers))
            ]
        )

        self.use_context_update = bool(use_context_update)
        self.use_pos_emb = bool(use_pos_emb)
        self.max_len = int(max_len)

        if self.use_pos_emb:
            self.pos_emb = nn.Parameter(torch.randn(1, self.max_len, int(d_model)) * 0.02)
        else:
            self.register_parameter("pos_emb", None)

    def _get_pos(self, L: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.pos_emb is None:
            return torch.zeros(1, L, self.input_proj.out_features, device=device, dtype=dtype)

        if L <= self.pos_emb.size(1):
            return self.pos_emb[:, :L, :].to(device=device, dtype=dtype)

        # interpolate if longer than max_len
        pe = self.pos_emb.to(device=device, dtype=dtype)     # [1, max_len, D]
        pe_t = pe.transpose(1, 2)                            # [1, D, max_len]
        pe_i = F.interpolate(pe_t, size=L, mode="linear", align_corners=False)
        return pe_i.transpose(1, 2)                          # [1, L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, input_dim]
        x = self.input_proj(x)  # [B, L, D]

        if self.use_pos_emb:
            L = x.size(1)
            x = x + self._get_pos(L, x.device, x.dtype)

        for layer in self.layers:
            x = layer(x)
            if self.training and self.use_context_update:
                layer.attn.update_contextual_memory(x.detach())
        return x

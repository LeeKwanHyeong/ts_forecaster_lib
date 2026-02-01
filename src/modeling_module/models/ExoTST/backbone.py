from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Utils
# =========================================================
def num_patches(seq_len: int, patch_len: int, stride: int) -> int:
    """
    Return number of patches produced by unfold with right-padding allowed.
    We assume we pad on the right so that at least one patch exists and last patch fits.
    """
    if seq_len <= 0:
        raise ValueError('seq_len must be > 0')
    if patch_len <= 0 or stride <= 0:
        raise ValueError('patch_len/stride must be > 0')

    # 최소 1개 패치
    if seq_len <= patch_len:
        return 1

    # (n-1) * stride + patch_len >= seq_len 를 만족하는 최소 n
    n = math.ceil((seq_len - patch_len) / stride) + 1
    return int(n)

def pad_to_patches(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """
    Right-pad time dimension so that unfold can generate an integer number of patches.

    x: (B, C, T)
    """
    b, c, t = x.shape
    n = num_patches(t, patch_len, stride)
    total = (n - 1) * stride + patch_len
    pad_len = max(0, total - t)
    if pad_len > 0:
        x = F.pad(x, (0, pad_len))  # pad last dim (time)
    return x

def unfold_patches_1d(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """
    x: (B, C, T)
    return: (B, C, N, patch_len)
    """
    x = pad_to_patches(x, patch_len, stride)
    # unfold over time dimension
    patches = x.unfold(dimension=-1, size = patch_len, step = stride)   # (B, C, N, patch_len)
    return patches

# =========================================================
# Positional Encoding (learnable)
# =========================================================
class LearnablePositionEncoding(nn.Module):
    """
    Learnable position embedding for tokens.

    Input: (B, N, D) or (B*C, N, D)
    """
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std = 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(1)
        if n > self.pos.size(1):
            raise ValueError(f"token length {n} exceeds max_len {self.pos.size(1)}")
        return x + self.pos[:, :n, :]

# =========================================================
# Patch Embedding
# =========================================================
class PatchEmbedding1D(nn.Module):
    """
    Patchify (B, T, C) into tokens (B, C, N, D), optionally prepend agg token.
    - Treat each channel(feature) as an independent 'series' (channel-wise).
    - Each patch is projected by Linear(patch_len -> d_model) per channel (shared weights).

    NOTE: The linear is shared across channels; this is usually fine and keeps params small.
    """
    def __init__(
        self,
        patch_len: int,
        stride: int,
        d_model: int,
        add_agg_token: bool,
        max_tokens: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.d_model = int(d_model)
        self.add_agg_token = bool(add_agg_token)

        self.proj = nn.Linear(self.patch_len, self.d_model)
        self.pos = LearnablePositionEncoding(max_len = max_tokens, d_model = self.d_model)
        self.drop = nn.Dropout(dropout)

        if self.add_agg_token:
            self.agg = nn.Parameter(torch.zeros(1, 1, self.d_model))
            nn.init.trunc_normal_(self.agg, std = 0.02)
        else:
            self.register_parameter("agg", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)
        return tokens: (B, C, N(+1), D)
        """
        if x.dim() != 3:
            raise ValueError("PatchEmbedding1D expects (B, T, C)")

        b, t, c = x.shape
        # (B, C, T)
        xc = x.transpose(1, 2).contiguous()
        patches = unfold_patches_1d(xc, self.patch_len, self.stride)    # (B, C, N, P)
        b, c, n, p = patches.shape

        tok = self.proj(patches)    # (B, C, N, D)

        # apply position encoding along token axis N (per channel)
        tok2 = tok.view(b * c, n, self.d_model)
        tok2 = self.pos(tok2)
        tok2 = self.drop(tok2)
        tok = tok2.view(b, c, n, self.d_model)

        if self.add_agg_token:
            # prepend agg token for each channel
            agg = self.agg.expand(b * c, 1, self.d_model).view(b, c, 1, self.d_model)
            tok = torch.cat([agg, tok], dim = 2)    # (B, C, N+1, D)
        return tok

# =========================================================
# Exogenous Encoder (channel-wise Transformer Encoder)
# =========================================================
class ExoEncoder(nn.Module):
    """
    Encode exogenous tokens (B, Cx, N, D) with TransformerEncoder per channel.

    Implementation: reshape to (B*Cx, N, D) and run encoder.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, layers: int, dropout: float, attn_dropout: float):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = n_heads,
            dim_feedforward = d_ff,
            dropout = dropout,
            batch_first = True,
            activation = 'gelu',
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers = layers)

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        """
        tok: (B, C, N, D)
        return: (B, C, N, D)
        """
        if tok.dim() != 4:
            raise ValueError('ExoEncoder expects (B, C, N, D)')
        b, c, n, d = tok.shape
        x = tok.reshape(b * c, n, d)
        x = self.enc(x)
        return x.reshape(b, c, n, d)


# =========================================================
# Cross-Temporal Modality Fusion Layer (agg-query cross-attn)
# =========================================================
class CrossTemporalFusionLayer(nn.Module):
    """
    Update only aggregation tokens via cross-attention between past-exo and future-exo modalities.

    hp: (B, C, Np, D) with hp[:, :, 0, :] = agg token
    hf: (B, C, Nf, D) with hf[:, :, 0, :] = agg token

    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.p_to_f = nn.MultiheadAttention(d_model, n_heads, dropout = dropout, batch_first = True)
        self.f_to_p = nn.MultiheadAttention(d_model, n_heads, dropout = dropout, batch_first = True)

        self.norm_p = nn.LayerNorm(d_model)
        self.norm_f = nn.LayerNorm(d_model)

        # lightweight FFN for agg refinement
        self.ff_p = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

        self.ff_f = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm_p2 = nn.LayerNorm(d_model)
        self.norm_f2 = nn.LayerNorm(d_model)

    def forward(self, hp: torch.Tensor, hf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, np, d = hp.shape
        _, _, nf, _ = hf.shape

        if np < 1 or nf < 1:
            raise ValueError('Fusion requires agg tokens (N>=1)')

        # reshape channel-wise: (B*C, N, D)
        Hp = hp.reshape(b * c, np, d)
        Hf = hf.reshape(b * c, nf, d)

        # agg queries
        qp = Hp[:, :1, :]   # (B * C, 1, D)
        qf = Hf[:, :1, :]   # (B * C, 1, D)

        # keys/values are full token sets of opposite modality
        # past agg attends to future tokens
        ap, _ = self.p_to_f(qp, Hf, Hf) # (B*C, 1, D)
        # future agg attends to past tokens
        af, _ = self.f_to_p(qf, Hp, Hp) # (B*C, 1, D)

        # residual + norm + FFN (only agg token updated)
        qp2 = self.norm_p(qp + ap)
        qp3 = self.norm_p2(qp2 + self.ff_p(qp2))

        qf2 = self.norm_f(qf + af)
        qf3 = self.norm_f2(qf2 + self.ff_p(qf2))

        # write back updated agg tokens, keep patch tokens unchanged
        Hp = torch.cat([qp3, Hp[:, 1:, :]], dim=1)
        Hf = torch.cat([qf3, Hf[:, 1:, :]], dim=1)

        hp_out = Hp.reshape(b, c, np, d)
        hf_out = Hf.reshape(b, c, nf, d)
        return hp_out, hf_out

# =========================================================
# Endogenous Decoder (channel-wise Transformer Decoder)
# =========================================================
class EndoDecoder(nn.Module):
    """
    Decode endogenous tokens (B, Cy, Ny, D) with self-attn + cross-attn to exo memory.

    exo_mem: (B, M, D)
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, layers: int, dropout: float):
        super().__init__()
        dec_layer = nn.TransformerDecoderLayer(
            d_model = d_model,
            nhead = n_heads,
            dim_feedforward = d_ff,
            dropout = dropout,
            batch_first = True,
            activation = 'gelu',
        )
        self.dec = nn.TransformerDecoder(dec_layer, num_layers = layers)

    def forward(self, y_tok: torch.Tensor, exo_mem: torch.Tensor) -> torch.Tensor:
        """
        y_tok: (B, Cy, Ny, D)
        exo_mem: (B, M, D)
        return: (B, Cy, Ny, D)
        """
        if y_tok.dim() != 4:
            raise ValueError('EndoDecoder expects y_tok (B, Cy, Ny, D)')
        if exo_mem.dim() != 3:
            raise ValueError('EndoDecoder expects exo_mem (B, M, D)')

        b, cy, ny, d = y_tok.shape
        _, m, dm = exo_mem.shape
        if dm != d:
            raise ValueError(f"exo_mem d_model mismatch: {dm} vs {d}")

        y = y_tok.reshape(b * cy, ny, d)

        # broadcast memory per channel
        mem = exo_mem.unsqueeze(1).expand(b, cy, m, d).reshape(b * cy, m, d)

        z = self.dec(tgt = y, memory = mem) # (B*Cy, Ny, D)
        return z.reshape(b, cy, ny, d)


# =========================================================
# Simple Horizon Head (flatten tokens -> horizon)
# =========================================================
class HorizonMLPHead(nn.Module):
    """
    A simple head:
        (B, Cy, Ny, D) -> flatten Ny*D -> Linear -> (B, H, Cy)
    """
    def __init__(self, ny: int, d_model: int, horizon: int, y_dim: int, dropout: float):
        super().__init__()
        self.ny = int(ny)
        self.d_model = int(d_model)
        self.horizon = int(horizon)
        self.y_dim = int(y_dim)

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(self.ny * self.d_model, self.horizon)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, Cy, Ny, D)
        return yhat: (B, H, Cy)
        """
        b, cy, ny, d = z.shape
        if ny != self.ny or d != self.d_model:
            raise ValueError(f"Head expects Ny={self.ny}, D={self.d_model} but got Ny={ny}, D={d}")
        x = z.reshape(b, cy, ny * d)
        x = self.drop(x)
        y = self.fc(x)  # (B, Cy, H)
        y = y.transpose(1, 2).contiguous()  # (B, H, Cy)
        return y

# =========================================================
# Distribution Horizon Head (flatten tokens -> horizon)
# =========================================================
class HorizonDistMLPHead(nn.Module):
    """
    Distribution head:
      z: (B, Cy, Ny, D) -> (B, H, out_mult)  (Cy=1 가정 우선)
    """

    def __init__(self, ny: int, d_model: int, horizon: int, y_dim: int, out_mult: int, dropout: float):
        super().__init__()
        self.ny = int(ny)
        self.d_model = int(d_model)
        self.horizon = int(horizon)
        self.y_dim = int(y_dim)
        self.out_mult = int(out_mult)

        self.drop = nn.Dropout(dropout)
        # (Ny*D) -> (H*out_mult)
        self.fc = nn.Linear(self.ny * self.d_model, self.horizon * self.out_mult)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, Cy, Ny, D)
        return:
          - if y_dim==1: (B, H, out_mult)
          - else: (B, H, Cy, out_mult)  # 확장 여지
        """
        b, cy, ny, d = z.shape
        if ny != self.ny or d != self.d_model:
            raise ValueError(f"Head expects Ny={self.ny}, D={self.d_model} but got Ny={ny}, D={d}")

        x = z.reshape(b, cy, ny * d)
        x = self.drop(x)
        y = self.fc(x)  # (B, Cy, H*out_mult)
        y = y.view(b, cy, self.horizon, self.out_mult)  # (B, Cy, H, out_mult)
        y = y.permute(0, 2, 1, 3).contiguous()          # (B, H, Cy, out_mult)

        if self.y_dim == 1:
            return y.squeeze(2)  # (B, H, out_mult)
        return y

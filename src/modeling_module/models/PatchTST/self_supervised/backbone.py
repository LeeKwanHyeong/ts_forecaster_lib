# modeling_module/models/PatchTST/self_supervised/backbone.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


def _cfg_get(cfg, name: str, default):
    return getattr(cfg, name, default)


def _sinusoidal_pos_emb(n: int, d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Returns:
        [1, n, d]
    """
    if d <= 0:
        raise ValueError(f"d_model must be > 0. got d={d}")
    pos = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)  # [n,1]
    i = torch.arange(d, device=device, dtype=dtype).unsqueeze(0)    # [1,d]
    angle_rates = 1.0 / torch.pow(10000.0, (2 * (i // 2)) / d)
    angles = pos * angle_rates
    pe = torch.zeros((n, d), device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe.unsqueeze(0)  # [1,n,d]


class Patchify(nn.Module):
    """
    Patchify: x[B,C,L] -> patches[B,N,C*patch_len]

    - N = floor((L - patch_len) / stride) + 1  (if L >= patch_len)
    - If L < patch_len: pad right to patch_len and make N=1
    """

    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        if self.patch_len <= 0:
            raise ValueError("patch_len must be > 0")
        if self.stride <= 0:
            raise ValueError("stride must be > 0")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Args:
            x: [B,C,L]
        Returns:
            patches: [B,N,C*patch_len]
            N: int
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x as [B,C,L]. got {tuple(x.shape)}")

        B, C, L = x.shape
        if L < self.patch_len:
            pad = self.patch_len - L
            x = torch.nn.functional.pad(x, (0, pad))  # pad right on time dim
            L = self.patch_len

        # x_unf: [B,C,N,patch_len]
        x_unf = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        N = x_unf.size(-2)
        # [B,N,C,patch_len] -> [B,N,C*patch_len]
        patches = x_unf.permute(0, 2, 1, 3).contiguous().view(B, N, C * self.patch_len)
        return patches, N


class PatchTSTSelfSupBackbone(nn.Module):
    """
    Self-Supervised PatchTST Backbone (Masked Patch Reconstruction)

    Shapes:
        x:            [B,C,L]
        patches:       [B,N,C*patch_len]
        z:             [B,N,d_model]
        patch_pred:    [B,N,C*patch_len]  (decoder output, patch reconstruction)

    Notes:
    - If your repo provides internal TSTEncoder/TSTEncoderLayer, this class will use them.
      Otherwise it falls back to torch.nn.TransformerEncoder.
    """

    def __init__(self, cfg, attn_core=None):
        super().__init__()

        # keep a reference for debug / optional flags
        self.cfg = cfg

        self.patch_len = int(_cfg_get(cfg, "patch_len", 16))
        self.stride = int(_cfg_get(cfg, "stride", self.patch_len))
        self.d_model = int(_cfg_get(cfg, "d_model", 128))
        self.n_heads = int(_cfg_get(cfg, "n_heads", 8))
        self.e_layers = int(_cfg_get(cfg, "e_layers", 3))
        self.d_ff = int(_cfg_get(cfg, "d_ff", self.d_model * 4))
        self.dropout = float(_cfg_get(cfg, "dropout", 0.1))
        self.activation = str(_cfg_get(cfg, "activation", "gelu"))

        # number of variables/channels
        self.n_vars = int(_cfg_get(cfg, "n_vars", _cfg_get(cfg, "enc_in", 1)))

        # patchify
        self.patchify = Patchify(patch_len=self.patch_len, stride=self.stride)

        # learnable mask token in patch space (C*patch_len)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.n_vars * self.patch_len))

        # patch embedding: (C*patch_len) -> d_model
        self.patch_embed = nn.Linear(self.n_vars * self.patch_len, self.d_model)

        # encoder (internal if available, else fallback)
        self.encoder = self._build_encoder(attn_core=attn_core)

        # output norm
        self.norm_out = nn.LayerNorm(self.d_model)

        # decoder/head: d_model -> (C*patch_len)
        self.pretrain_head = nn.Linear(self.d_model, self.n_vars * self.patch_len)

    def _build_encoder(self, attn_core=None) -> nn.Module:
        """
        IMPORTANT
        - 이 프로젝트의 supervised encoder는 build_attention(cfg.attn) 기반으로 mha를 만들고,
          TSTEncoder(q_len, cfg)가 내부에서 layer stack을 구성합니다.
        - 따라서 self-supervised에서도 MultiHeadAttention를 직접 만들지 말고,
          TSTEncoder를 그대로 사용해야 state_dict key-space가 일치합니다.
        """

        # ---- cfg aliasing (self-supervised cfg <-> supervised encoder cfg contract) ----
        cfg = self.cfg

        # self-supervised에서 쓰던 e_layers -> supervised의 n_layers로 매핑
        if not hasattr(cfg, "n_layers"):
            setattr(cfg, "n_layers", int(getattr(cfg, "e_layers", self.e_layers)))

        # d_ff, act, norm, dropout, pre_norm, store_attn 등 encoder.py가 참조하는 키를 보장
        if not hasattr(cfg, "d_ff"):
            setattr(cfg, "d_ff", int(getattr(cfg, "d_ff", self.d_ff)))
        if not hasattr(cfg, "act"):
            setattr(cfg, "act", str(getattr(cfg, "activation", self.activation)))
        if not hasattr(cfg, "norm"):
            setattr(cfg, "norm", str(getattr(cfg, "norm", "BatchNorm")))
        if not hasattr(cfg, "dropout"):
            setattr(cfg, "dropout", float(getattr(cfg, "dropout", self.dropout)))
        if not hasattr(cfg, "pre_norm"):
            setattr(cfg, "pre_norm", bool(getattr(cfg, "pre_norm", False)))
        if not hasattr(cfg, "store_attn"):
            setattr(cfg, "store_attn", bool(getattr(cfg, "store_attn", False)))

        # cfg.d_model / cfg.n_heads 정합성 보장
        if not hasattr(cfg, "d_model"):
            setattr(cfg, "d_model", int(self.d_model))
        if not hasattr(cfg, "n_heads"):
            setattr(cfg, "n_heads", int(self.n_heads))

        # ---- build encoder (NO direct MultiHeadAttention construction) ----
        from modeling_module.models.PatchTST.common.encoder import \
            TSTEncoder  # 이 파일의 contract 기준 :contentReference[oaicite:2]{index=2}

        # q_len은 현재 구현의 TSTEncoder에서는 실질적으로 사용되지 않지만, 안전하게 patch_num 추정
        seq_len = getattr(cfg, "seq_len", getattr(cfg, "lookback", None))
        if seq_len is None:
            q_len = 1
        else:
            q_len = self.infer_patch_num(int(seq_len))

        return TSTEncoder(q_len=q_len, cfg=cfg)
    @torch.no_grad()
    def infer_patch_num(self, L: int) -> int:
        L = int(L)
        if L < self.patch_len:
            return 1
        return ((L - self.patch_len) // self.stride) + 1

    def forward_from_patches(
        self,
        patches: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patches:    [B,N,C*patch_len]
            patch_mask: [B,N] bool tensor, True means "masked (to predict)"
        Returns:
            z:          [B,N,d_model]
            patch_pred: [B,N,C*patch_len]
        """
        if patches.dim() != 3:
            raise ValueError(f"Expected patches as [B,N,D]. got {tuple(patches.shape)}")

        if patch_mask is not None:
            if patch_mask.dtype != torch.bool:
                patch_mask = patch_mask.bool()
            if patch_mask.dim() != 2 or patch_mask.shape[:2] != patches.shape[:2]:
                raise ValueError(f"patch_mask must be [B,N] matching patches. got {tuple(patch_mask.shape)}")

            # replace masked patches with mask token
            mask_token = self.mask_token.expand(patches.size(0), patches.size(1), -1)  # [B,N,D]
            patches = torch.where(patch_mask.unsqueeze(-1), mask_token, patches)

        # embed + positional encoding
        z = self.patch_embed(patches)  # [B,N,d_model]
        z = z + _sinusoidal_pos_emb(z.size(1), z.size(2), device=z.device, dtype=z.dtype)
        z = torch.nn.functional.dropout(z, p=self.dropout, training=self.training)

        # encode
        z = self.encoder(z)
        z = self.norm_out(z)

        # decode to patch space
        patch_pred = self.pretrain_head(z)  # [B,N,C*patch_len]
        return z, patch_pred

    def forward(
        self,
        x: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:         [B,C,L]
            patch_mask:[B,N] bool
        Returns:
            patches:    [B,N,C*patch_len] (target)
            z:          [B,N,d_model]
            patch_pred: [B,N,C*patch_len]
        """
        patches, _ = self.patchify(x)
        z, patch_pred = self.forward_from_patches(patches=patches, patch_mask=patch_mask)
        return patches, z, patch_pred
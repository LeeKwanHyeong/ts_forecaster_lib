from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling_module.models.TimeXer.configs import TimeXerConfig
from modeling_module.models.common_layers.Attention import AttentionLayer, FullAttention
from modeling_module.models.common_layers.Embed import PositionalEmbedding


class FlattenHead(nn.Module):
    """Project per-variable latent patches into the forecast horizon."""

    def __init__(self, n_vars: int, nf: int, target_window: int, head_dropout: float = 0.0):
        super().__init__()
        self.n_vars = int(n_vars)
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)


class ExogenousEmbedding(nn.Module):
    """
    TimeXer's inverted exogenous embedding.

    The official implementation treats each exogenous channel as a token whose
    feature vector is the historical window itself.
    """

    def __init__(self, lookback: int, d_model: int, dropout: float):
        super().__init__()
        self.value_embedding = nn.Linear(int(lookback), int(d_model))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, E] -> [B, E, L] -> [B, E, D]
        x = x.permute(0, 2, 1)
        x = self.value_embedding(x)
        return self.dropout(x)


class EndogenousPatchEmbedding(nn.Module):
    """
    Patch the target history and append one global token per target channel.

    This mirrors the paper implementation while keeping the API local to this repo.
    """

    def __init__(self, n_vars: int, d_model: int, patch_len: int, dropout: float):
        super().__init__()
        self.patch_len = int(patch_len)
        self.value_embedding = nn.Linear(int(patch_len), int(d_model), bias=False)
        self.global_token = nn.Parameter(torch.randn(1, int(n_vars), 1, int(d_model)))
        self.position_embedding = PositionalEmbedding(int(d_model))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        # x: [B, Cy, L]
        n_vars = int(x.shape[1])
        global_tokens = self.global_token.repeat(x.shape[0], 1, 1, 1)

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, global_tokens], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class TimeXerEncoderLayer(nn.Module):
    """
    One TimeXer block:
    - self-attention over target patches
    - cross-attention from the per-target global token to exogenous tokens
    - feed-forward refinement
    """

    def __init__(
        self,
        self_attention: AttentionLayer,
        cross_attention: AttentionLayer,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        d_ff = int(d_ff or (4 * d_model))
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, cross: torch.Tensor) -> torch.Tensor:
        batch_size, _, d_model = cross.shape

        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=None)[0])
        x = self.norm1(x)

        x_global_original = x[:, -1, :].unsqueeze(1)
        x_global = torch.reshape(x_global_original, (batch_size, -1, d_model))
        x_global_attn = self.dropout(self.cross_attention(x_global, cross, cross, attn_mask=None)[0])
        x_global_attn = torch.reshape(
            x_global_attn,
            (x_global_attn.shape[0] * x_global_attn.shape[1], x_global_attn.shape[2]),
        ).unsqueeze(1)
        x_global = self.norm2(x_global_original + x_global_attn)

        y = x = torch.cat([x[:, :-1, :], x_global], dim=1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class TimeXerEncoder(nn.Module):
    """Thin wrapper that applies multiple TimeXer encoder layers."""

    def __init__(self, layers: list[TimeXerEncoderLayer], norm_layer: Optional[nn.Module] = None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, cross: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cross)
        if self.norm is not None:
            x = self.norm(x)
        return x


class TimeXerModel(nn.Module):
    """
    Paper-aligned TimeXer v1 for this library.

    Important contract:
    - `x` contains only endogenous target history.
    - `past_exo_cont` contains historical continuous exogenous features.
    - future exogenous inputs are intentionally rejected in v1 to preserve the
      original paper/official-code contract.
    """

    def __init__(self, cfg: TimeXerConfig):
        super().__init__()
        self.cfg = cfg
        self.lookback = int(cfg.lookback)
        self.horizon = int(cfg.horizon)
        self.y_dim = int(cfg.y_dim)
        self.past_exo_cont_dim = int(cfg.past_exo_cont_dim)
        self.patch_len = int(cfg.patch_len)
        self.use_norm = bool(cfg.use_norm)

        if self.patch_len <= 0:
            raise ValueError(f"patch_len must be positive, got {self.patch_len}")
        if self.lookback < self.patch_len:
            raise ValueError(f"lookback={self.lookback} must be >= patch_len={self.patch_len}")
        if self.lookback % self.patch_len != 0:
            raise ValueError(
                f"TimeXer requires non-overlapping patches: lookback={self.lookback} "
                f"must be divisible by patch_len={self.patch_len}."
            )
        if self.past_exo_cont_dim <= 0:
            raise ValueError("TimeXer requires past_exo_cont_dim > 0.")

        self.patch_num = int(self.lookback // self.patch_len)

        self.endogenous_embedding = EndogenousPatchEmbedding(
            n_vars=self.y_dim,
            d_model=cfg.d_model,
            patch_len=self.patch_len,
            dropout=cfg.dropout,
        )
        self.exogenous_embedding = ExogenousEmbedding(
            lookback=self.lookback,
            d_model=cfg.d_model,
            dropout=cfg.dropout,
        )

        self.encoder = TimeXerEncoder(
            [
                TimeXerEncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            factor=cfg.factor,
                            attention_dropout=cfg.dropout,
                            output_attention=False,
                        ),
                        cfg.d_model,
                        cfg.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            factor=cfg.factor,
                            attention_dropout=cfg.dropout,
                            output_attention=False,
                        ),
                        cfg.d_model,
                        cfg.n_heads,
                    ),
                    d_model=cfg.d_model,
                    d_ff=cfg.d_ff,
                    dropout=cfg.dropout,
                    activation=cfg.activation,
                )
                for _ in range(int(cfg.e_layers))
            ],
            norm_layer=nn.LayerNorm(cfg.d_model),
        )

        head_nf = int(cfg.d_model) * (self.patch_num + 1)
        self.head = FlattenHead(
            n_vars=self.y_dim,
            nf=head_nf,
            target_window=self.horizon,
            head_dropout=cfg.dropout,
        )

    @classmethod
    def from_config(cls, config: TimeXerConfig) -> "TimeXerModel":
        return cls(cfg=config)

    def forward(
        self,
        x: torch.Tensor,
        future_exo: Optional[torch.Tensor] = None,
        past_exo_cont: Optional[torch.Tensor] = None,
        past_exo_cat: Optional[torch.Tensor] = None,
        part_ids: Optional[torch.Tensor] = None,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        del part_ids, mode

        if future_exo is not None and int(future_exo.shape[-1]) > 0:
            raise ValueError("TimeXer v1 does not consume future exogenous inputs.")
        if past_exo_cat is not None and int(past_exo_cat.shape[-1]) > 0:
            raise ValueError("TimeXer v1 supports only past continuous exogenous inputs.")
        if past_exo_cont is None:
            raise ValueError("TimeXer requires `past_exo_cont`.")
        if x.dim() != 3:
            raise ValueError(f"x must be 3D [B, L, Cy], got {tuple(x.shape)}")
        if past_exo_cont.dim() != 3:
            raise ValueError(
                f"past_exo_cont must be 3D [B, L, E], got {tuple(past_exo_cont.shape)}"
            )
        if int(x.shape[1]) != self.lookback:
            raise ValueError(f"x lookback mismatch: expected {self.lookback}, got {tuple(x.shape)}")
        if int(past_exo_cont.shape[1]) != self.lookback:
            raise ValueError(
                f"past_exo_cont lookback mismatch: expected {self.lookback}, got {tuple(past_exo_cont.shape)}"
            )
        if int(x.shape[2]) != self.y_dim:
            raise ValueError(f"x channel mismatch: expected {self.y_dim}, got {tuple(x.shape)}")
        if int(past_exo_cont.shape[2]) != self.past_exo_cont_dim:
            raise ValueError(
                "past_exo_cont feature mismatch: "
                f"expected {self.past_exo_cont_dim}, got {tuple(past_exo_cont.shape)}"
            )

        if self.use_norm:
            means = x.mean(dim=1, keepdim=True).detach()
            x_norm = x - means
            stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_norm = x_norm / stdev
        else:
            x_norm = x
            means = None
            stdev = None

        endogenous_tokens, n_vars = self.endogenous_embedding(x_norm.permute(0, 2, 1))
        exogenous_tokens = self.exogenous_embedding(past_exo_cont)

        encoded = self.encoder(endogenous_tokens, exogenous_tokens)
        encoded = torch.reshape(
            encoded,
            (-1, n_vars, encoded.shape[-2], encoded.shape[-1]),
        )
        encoded = encoded.permute(0, 1, 3, 2)

        forecast = self.head(encoded).permute(0, 2, 1)

        if self.use_norm and means is not None and stdev is not None:
            forecast = forecast * stdev[:, 0, :].unsqueeze(1).repeat(1, self.horizon, 1)
            forecast = forecast + means[:, 0, :].unsqueeze(1).repeat(1, self.horizon, 1)

        return forecast

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .configs import AutoTimesConfig
from .backbone import (
    build_segment_mlp,
    freeze_backbone,
    infer_backbone_hidden_size,
    load_autotimes_backbone,
)
from .timestamp_artifact import TimestampEmbeddingArtifact


class AutoTimesModel(nn.Module):
    """AutoTimes numeric-token model with frozen-backbone autoregressive decoding."""

    model_key = "autotimes_base"

    def __init__(self, cfg: AutoTimesConfig, *, backbone: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        self.lookback = int(cfg.lookback)
        self.horizon = int(cfg.horizon)
        self.y_dim = int(cfg.y_dim)
        self.token_len = int(cfg.token_len)
        self.icl_past_exogenous_dim = int(cfg.icl_past_exogenous_dim)
        self.icl_future_exogenous_dim = int(cfg.icl_future_exogenous_dim)
        self.icl_exogenous_dim = (
            self.icl_past_exogenous_dim + self.icl_future_exogenous_dim
        )
        self.backbone = freeze_backbone(
            backbone if backbone is not None else load_autotimes_backbone(cfg)
        )
        self.hidden_size = infer_backbone_hidden_size(self.backbone)

        self.tokenizer = build_segment_mlp(
            self.token_len,
            self.hidden_size,
            hidden_dim=int(cfg.mlp_hidden_dim),
            hidden_layers=int(cfg.mlp_hidden_layers),
            dropout=float(cfg.dropout),
            activation=cfg.mlp_activation,
        )
        self.detokenizer = build_segment_mlp(
            self.hidden_size,
            self.token_len,
            hidden_dim=int(cfg.mlp_hidden_dim),
            hidden_layers=int(cfg.mlp_hidden_layers),
            dropout=float(cfg.dropout),
            activation=cfg.mlp_activation,
        )
        self.icl_exogenous_tokenizer = (
            build_segment_mlp(
                self.token_len * self.icl_exogenous_dim,
                self.hidden_size,
                hidden_dim=int(cfg.mlp_hidden_dim),
                hidden_layers=int(cfg.mlp_hidden_layers),
                dropout=float(cfg.dropout),
                activation=cfg.mlp_activation,
            )
            if self.icl_exogenous_dim
            else None
        )
        self.timestamp_scale = (
            nn.Parameter(torch.ones(())) if cfg.mix_timestamp_embeddings else None
        )
        self._timestamp_artifact: Optional[TimestampEmbeddingArtifact] = None
        if cfg.timestamp_artifact_path is not None:
            self._timestamp_artifact = TimestampEmbeddingArtifact.load(
                cfg.timestamp_artifact_path,
                str(cfg.timestamp_artifact_sha256),
            )
            if int(self._timestamp_artifact.tensor.shape[-1]) != self.hidden_size:
                raise ValueError(
                    "Timestamp artifact hidden size does not match the AutoTimes backbone: "
                    f"{self._timestamp_artifact.tensor.shape[-1]} != {self.hidden_size}."
                )

    @classmethod
    def from_config(cls, cfg: AutoTimesConfig) -> "AutoTimesModel":
        return cls(cfg)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _resolve_timestamp_embeddings(
        self,
        explicit: Optional[torch.Tensor],
        *,
        batch: int,
        channels: int,
        token_count: int,
        step: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.cfg.mix_timestamp_embeddings:
            return None
        source = explicit
        if source is None and self._timestamp_artifact is not None:
            source = self._timestamp_artifact.tensor
        if source is None:
            return None
        source = source.to(device=device, dtype=dtype)
        end = int(step) + int(token_count)
        if source.shape[-2] < end:
            raise ValueError(
                "Timestamp embeddings do not cover the requested autoregressive window: "
                f"required={end}, available={source.shape[-2]}."
            )
        source = source[..., step:end, :]
        if int(source.shape[-1]) != self.hidden_size:
            raise ValueError("Timestamp embedding hidden size does not match the backbone.")
        if source.ndim == 2:
            source = source.unsqueeze(0).expand(batch * channels, -1, -1)
        elif source.ndim == 3:
            if int(source.shape[0]) not in (1, batch):
                raise ValueError("Timestamp batch dimension must be 1 or match input batch size.")
            source = source.expand(batch, -1, -1)
            source = source[:, None, :, :].expand(batch, channels, -1, -1)
            source = source.reshape(batch * channels, token_count, self.hidden_size)
        elif source.ndim == 4:
            if int(source.shape[0]) not in (1, batch) or int(source.shape[1]) not in (1, channels):
                raise ValueError("Timestamp [B,C] dimensions must broadcast to the input shape.")
            source = source.expand(batch, channels, -1, -1)
            source = source.reshape(batch * channels, token_count, self.hidden_size)
        else:
            raise ValueError("Timestamp embeddings must be rank 2, 3, or 4.")
        return source

    def _next_token(
        self,
        context: torch.Tensor,
        *,
        timestamp_embeddings: Optional[torch.Tensor],
        step: int,
        context_exogenous: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        means = context.mean(dim=1, keepdim=True).detach()
        normalized = context - means
        stdev = torch.sqrt(normalized.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized = normalized / stdev

        batch, _, channels = normalized.shape
        context_length = int(normalized.shape[1])
        if context_length % self.token_len != 0:
            raise ValueError(
                "AutoTimes context length must be divisible by token_len: "
                f"{context_length} % {self.token_len} != 0."
            )
        flat = normalized.permute(0, 2, 1).reshape(batch * channels, context_length)
        tokens = flat.unfold(-1, self.token_len, self.token_len)
        embeddings = self.tokenizer(tokens)
        if context_exogenous is not None:
            if self.icl_exogenous_tokenizer is None:
                raise ValueError("AutoTimes checkpoint has no ICL exogenous tokenizer.")
            if tuple(context_exogenous.shape[:2]) != (batch, context_length):
                raise ValueError("AutoTimes ICL exogenous context must match [B,P].")
            if int(context_exogenous.shape[-1]) != self.icl_exogenous_dim:
                raise ValueError("AutoTimes ICL exogenous context width does not match config.")
            exogenous_tokens = context_exogenous.unfold(
                1,
                self.token_len,
                self.token_len,
            ).permute(0, 1, 3, 2).contiguous()
            exogenous_tokens = exogenous_tokens.reshape(
                batch,
                int(tokens.shape[1]),
                self.token_len * self.icl_exogenous_dim,
            )
            exogenous_embeddings = self.icl_exogenous_tokenizer(exogenous_tokens)
            exogenous_embeddings = exogenous_embeddings[:, None, :, :].expand(
                batch,
                channels,
                -1,
                -1,
            ).reshape(batch * channels, int(tokens.shape[1]), self.hidden_size)
            embeddings = embeddings + exogenous_embeddings.to(dtype=embeddings.dtype)
        timestamp = self._resolve_timestamp_embeddings(
            timestamp_embeddings,
            batch=batch,
            channels=channels,
            token_count=int(tokens.shape[1]),
            step=step,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        if timestamp is not None:
            embeddings = nn.functional.normalize(embeddings, dim=-1)
            timestamp = nn.functional.normalize(timestamp, dim=-1)
            embeddings = embeddings + self.timestamp_scale * timestamp

        hidden = self.backbone(inputs_embeds=embeddings).last_hidden_state
        next_normalized = self.detokenizer(hidden[:, -1, :])
        next_normalized = next_normalized.reshape(batch, channels, self.token_len).permute(0, 2, 1)
        return next_normalized * stdev + means

    def _autoregressive_forecast(
        self,
        context: torch.Tensor,
        *,
        timestamp_embeddings: Optional[torch.Tensor],
        context_exogenous: Optional[torch.Tensor] = None,
        future_exogenous: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        steps = math.ceil(self.horizon / self.token_len)
        if context_exogenous is not None:
            if future_exogenous is None:
                raise ValueError("AutoTimes ICL exogenous forecast requires future features.")
            required = steps * self.token_len
            if int(future_exogenous.shape[1]) != self.horizon:
                raise ValueError("AutoTimes ICL future exogenous horizon does not match config.")
            if required > self.horizon:
                padding = future_exogenous[:, -1:, :].expand(
                    -1,
                    required - self.horizon,
                    -1,
                )
                future_exogenous = torch.cat([future_exogenous, padding], dim=1)
        for step in range(steps):
            next_token = self._next_token(
                context,
                timestamp_embeddings=timestamp_embeddings,
                step=step,
                context_exogenous=context_exogenous,
            )
            pieces.append(next_token)
            context = torch.cat([context[:, self.token_len :, :], next_token], dim=1)
            if context_exogenous is not None:
                assert future_exogenous is not None
                start = step * self.token_len
                next_exogenous = future_exogenous[:, start : start + self.token_len, :]
                context_exogenous = torch.cat(
                    [context_exogenous[:, self.token_len :, :], next_exogenous],
                    dim=1,
                )
        return torch.cat(pieces, dim=1)[:, : self.horizon, :]

    def forward_icl(
        self,
        packed_context: torch.Tensor,
        *,
        prompt_mask: torch.Tensor,
        timestamp_embeddings: Optional[torch.Tensor] = None,
        packed_exogenous: Optional[torch.Tensor] = None,
        query_target_exogenous: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forecast from demonstration pairs followed by one query context."""

        if not bool(self.cfg.icl_enabled):
            raise RuntimeError("AutoTimes checkpoint was not configured for ICL execution.")
        if packed_context.ndim != 3:
            raise ValueError("packed_context must be [B,P,C].")
        if int(packed_context.shape[1]) <= self.lookback:
            raise ValueError("AutoTimes ICL context must include at least one prompt.")
        if int(packed_context.shape[2]) != self.y_dim:
            raise ValueError("AutoTimes ICL channel count does not match the config.")
        if prompt_mask.ndim != 2 or int(prompt_mask.shape[0]) != int(packed_context.shape[0]):
            raise ValueError("prompt_mask must be [B,K] and match the context batch.")
        if not bool(prompt_mask.all()):
            raise ValueError("AutoTimes ICL v1 requires every prompt slot to be populated.")
        if not torch.isfinite(packed_context).all():
            raise ValueError("AutoTimes ICL context must contain finite values only.")
        if self.icl_exogenous_dim:
            if packed_exogenous is None or query_target_exogenous is None:
                raise ValueError("AutoTimes ICL exogenous checkpoint requires exogenous inputs.")
            if int(packed_exogenous.shape[-1]) != self.icl_exogenous_dim:
                raise ValueError("AutoTimes packed exogenous width does not match config.")
            if int(query_target_exogenous.shape[-1]) != self.icl_future_exogenous_dim:
                raise ValueError("AutoTimes future exogenous width does not match config.")
            future_role = torch.cat(
                [
                    query_target_exogenous.new_zeros(
                        *query_target_exogenous.shape[:-1],
                        self.icl_past_exogenous_dim,
                    ),
                    query_target_exogenous,
                ],
                dim=-1,
            )
        elif packed_exogenous is not None or query_target_exogenous is not None:
            raise ValueError("AutoTimes checkpoint was configured without ICL exogenous inputs.")
        else:
            future_role = None
        return self._autoregressive_forecast(
            packed_context,
            timestamp_embeddings=timestamp_embeddings,
            context_exogenous=packed_exogenous,
            future_exogenous=future_role,
        )

    def forward(
        self,
        x: torch.Tensor,
        future_exo: Optional[torch.Tensor] = None,
        past_exo_cont: Optional[torch.Tensor] = None,
        past_exo_cat: Optional[torch.Tensor] = None,
        part_ids=None,
        mode: Optional[str] = None,
        timestamp_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del future_exo, past_exo_cont, past_exo_cat, part_ids, mode
        if x.ndim != 3:
            raise ValueError(f"AutoTimes input must be [B,L,C], got {tuple(x.shape)}.")
        if int(x.shape[1]) != self.lookback or int(x.shape[2]) != self.y_dim:
            raise ValueError(
                "AutoTimes input shape does not match config: "
                f"expected [B,{self.lookback},{self.y_dim}], got {tuple(x.shape)}."
            )
        if not torch.isfinite(x).all():
            raise ValueError("AutoTimes input must contain finite values only.")

        return self._autoregressive_forecast(
            x,
            timestamp_embeddings=timestamp_embeddings,
        )

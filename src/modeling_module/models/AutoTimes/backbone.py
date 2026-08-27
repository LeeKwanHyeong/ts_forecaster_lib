from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from .configs import AutoTimesConfig


def _activation(name: str) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported AutoTimes MLP activation: {name!r}")


def build_segment_mlp(
    input_dim: int,
    output_dim: int,
    *,
    hidden_dim: int,
    hidden_layers: int,
    dropout: float,
    activation: str,
) -> nn.Module:
    """Build the upstream-style numeric tokenizer or detokenizer."""

    if int(hidden_layers) == 0:
        return nn.Linear(input_dim, output_dim)
    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        _activation(activation),
    ]
    if float(dropout) > 0:
        layers.append(nn.Dropout(float(dropout)))
    for _ in range(int(hidden_layers) - 1):
        layers.extend(
            [nn.Linear(hidden_dim, hidden_dim), _activation(activation)]
        )
        if float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class MockAutoTimesBackbone(nn.Module):
    """Small decoder-shaped backbone for contract tests and CPU smoke runs."""

    def __init__(
        self,
        hidden_size: int,
        *,
        layers: int = 1,
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(hidden_size) % int(heads) != 0:
            raise ValueError(
                "Mock AutoTimes hidden_size must be divisible by heads."
            )
        block = nn.TransformerEncoderLayer(
            d_model=int(hidden_size),
            nhead=int(heads),
            dim_feedforward=max(int(hidden_size) * 2, 8),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=int(layers))
        self.config = SimpleNamespace(hidden_size=int(hidden_size))

    def forward(self, *, inputs_embeds: torch.Tensor):
        token_count = int(inputs_embeds.shape[1])
        causal_mask = torch.full(
            (token_count, token_count),
            float("-inf"),
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        ).triu(diagonal=1)
        return SimpleNamespace(
            last_hidden_state=self.encoder(inputs_embeds, mask=causal_mask)
        )


def load_autotimes_backbone(cfg: AutoTimesConfig) -> nn.Module:
    """Load the configured Mock or Hugging Face decoder backbone."""

    if cfg.backbone_type == "mock":
        return MockAutoTimesBackbone(
            int(cfg.hidden_size),
            layers=int(cfg.mock_layers),
            heads=int(cfg.mock_heads),
            dropout=float(cfg.dropout),
        )
    try:
        from transformers import AutoModel
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise ImportError(
            "AutoTimes LLM backbones require `modeling-module[autotimes]`."
        ) from exc

    load_kwargs: dict[str, object] = {}
    if cfg.llm_source == "local":
        local_path = Path(str(cfg.llm_local_path or "")).expanduser()
        if not local_path.is_dir():
            raise FileNotFoundError(
                f"AutoTimes local LLM directory does not exist: {local_path}"
            )
        target = str(local_path)
        load_kwargs["local_files_only"] = True
    else:
        target = str(cfg.llm_model_name or "").strip()
        if not target:
            raise ValueError(
                "llm_model_name is required for a Hugging Face AutoTimes backbone."
            )
        if cfg.llm_revision:
            load_kwargs["revision"] = str(cfg.llm_revision)
        load_kwargs["local_files_only"] = bool(cfg.local_files_only)
    return AutoModel.from_pretrained(target, **load_kwargs)


def infer_backbone_hidden_size(backbone: nn.Module) -> int:
    """Resolve the decoder embedding width from supported HF-style configs."""

    config = getattr(backbone, "config", None)
    for name in ("hidden_size", "n_embd", "d_model", "word_embed_proj_dim"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Unable to infer AutoTimes backbone hidden size.")


def freeze_backbone(backbone: nn.Module) -> nn.Module:
    """Freeze and switch the decoder to evaluation mode."""

    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return backbone


__all__ = [
    "MockAutoTimesBackbone",
    "build_segment_mlp",
    "freeze_backbone",
    "infer_backbone_hidden_size",
    "load_autotimes_backbone",
]

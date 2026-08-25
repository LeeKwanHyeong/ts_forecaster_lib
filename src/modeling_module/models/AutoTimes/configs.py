from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from modeling_module.training.config import TrainingConfig


@dataclass
class AutoTimesConfig(TrainingConfig):
    """Product configuration for the AutoTimes numeric-token forecaster."""

    y_dim: int = 1
    token_len: int = 13
    backbone_type: Literal["mock", "llama", "gpt2", "opt"] = "gpt2"
    llm_source: Literal["huggingface", "local"] = "local"
    llm_model_name: Optional[str] = None
    llm_local_path: Optional[str] = None
    llm_revision: Optional[str] = None
    local_files_only: bool = True
    freeze_llm: bool = True
    hidden_size: int = 128
    mock_layers: int = 1
    mock_heads: int = 4
    mlp_hidden_dim: int = 256
    mlp_hidden_layers: int = 1
    mlp_activation: Literal["relu", "gelu", "tanh"] = "gelu"
    dropout: float = 0.1
    mix_timestamp_embeddings: bool = True
    timestamp_artifact_path: Optional[str] = None
    timestamp_artifact_sha256: Optional[str] = None
    icl_enabled: bool = False
    icl_past_exogenous_dim: int = 0
    icl_future_exogenous_dim: int = 0
    icl_exogenous_schema_hash: Optional[str] = None

    def __post_init__(self) -> None:
        if int(self.lookback) <= 0 or int(self.horizon) <= 0:
            raise ValueError("AutoTimes lookback and horizon must be positive.")
        if int(self.token_len) <= 0:
            raise ValueError("AutoTimes token_len must be positive.")
        if int(self.lookback) % int(self.token_len) != 0:
            raise ValueError(
                "AutoTimes lookback must be divisible by token_len; "
                f"got lookback={self.lookback}, token_len={self.token_len}."
            )
        if int(self.y_dim) <= 0:
            raise ValueError("AutoTimes y_dim must be positive.")
        if not bool(self.freeze_llm):
            raise ValueError("autotimes_base requires a frozen LLM backbone.")
        if int(self.hidden_size) <= 0:
            raise ValueError("AutoTimes hidden_size must be positive.")
        if int(self.mlp_hidden_layers) < 0:
            raise ValueError("AutoTimes mlp_hidden_layers cannot be negative.")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("AutoTimes dropout must be in [0, 1).")
        past_dim = int(self.icl_past_exogenous_dim)
        future_dim = int(self.icl_future_exogenous_dim)
        if past_dim < 0 or future_dim < 0:
            raise ValueError("AutoTimes ICL exogenous dimensions must be non-negative.")
        if (past_dim == 0) != (future_dim == 0):
            raise ValueError(
                "AutoTimes ICL exogenous execution requires both past and future features."
            )
        schema_hash = str(self.icl_exogenous_schema_hash or "").strip()
        if bool(past_dim) != bool(schema_hash):
            raise ValueError(
                "AutoTimes ICL exogenous dimensions and schema hash must be configured together."
            )
        if past_dim and not bool(self.icl_enabled):
            raise ValueError("AutoTimes ICL exogenous inputs require icl_enabled=True.")
        if schema_hash and (
            len(schema_hash) != 64
            or any(character not in "0123456789abcdef" for character in schema_hash)
        ):
            raise ValueError("AutoTimes ICL exogenous schema hash must be lowercase SHA256.")
        self.icl_past_exogenous_dim = past_dim
        self.icl_future_exogenous_dim = future_dim
        self.icl_exogenous_schema_hash = schema_hash or None
        has_path = bool(str(self.timestamp_artifact_path or "").strip())
        has_hash = bool(str(self.timestamp_artifact_sha256 or "").strip())
        if has_path != has_hash:
            raise ValueError(
                "timestamp_artifact_path and timestamp_artifact_sha256 must be provided together."
            )

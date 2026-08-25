from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

from modeling_module.training.config import TrainingConfig


DEFAULT_SELLM_LLM_MODEL_NAME = "Qwen/Qwen2-0.5B"
DEFAULT_SELLM_LLM_REVISION = "91d2aff3f957f99e4c74c962f2f408dcc88a18d8"


@dataclass
class SELLMConfig(TrainingConfig):
    """
    Semantic-enhanced LLM forecaster configuration.

    The model can run in three modes:
    - use_pretrained_llm=True, llm_source="huggingface": load a Hub model ID.
    - use_pretrained_llm=True, llm_source="local": load an on-premise model directory.
    - use_pretrained_llm=False: use the built-in Transformer fallback for smoke tests.
    """

    # Data / IO
    y_dim: int = 1
    future_exo_dim: int = 0

    # Architecture lineage. Legacy remains the default so checkpoints saved before
    # paper_v1 was introduced rebuild with their original state-dict schema.
    architecture_variant: Literal["legacy_v1", "paper_v1"] = "legacy_v1"

    # Numeric tokenization
    token_len: int = 8
    d_model: int = 128
    n_heads: int = 4
    dropout: float = 0.1
    mlp_hidden_dim: int = 256
    mlp_activation: Literal["relu", "gelu", "tanh"] = "gelu"

    # Semantic space / TSCC
    semantic_vocab_size: int = 256
    semantic_top_k: int = 32
    tscc_latent_dim: int = 8
    tscc_hidden_dim: int = 64
    tscc_kl_weight: float = 1e-4

    # LLM backbone
    use_pretrained_llm: bool = True
    llm_source: Literal["huggingface", "local"] = "huggingface"
    llm_model_name: str = DEFAULT_SELLM_LLM_MODEL_NAME
    llm_local_path: Optional[str] = None
    llm_revision: Optional[str] = DEFAULT_SELLM_LLM_REVISION
    freeze_llm: bool = True
    use_time_adapter: bool = True
    time_adapter_rank: int = 8
    time_adapter_layers: int = 2

    # Fallback encoder used when use_pretrained_llm=False.
    fallback_layers: int = 2
    d_ff: int = 256

    # Output head
    head_hidden_dim: int = 128
    use_norm: bool = True
    final_nonneg: bool = False
    negative_output_penalty_weight: float = 0.0
    icl_enabled: bool = False
    icl_past_exogenous_dim: int = 0
    icl_future_exogenous_dim: int = 0
    icl_exogenous_schema_hash: Optional[str] = None

    def __post_init__(self) -> None:
        weight = float(self.negative_output_penalty_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                "negative_output_penalty_weight must be finite and >= 0, "
                f"got {self.negative_output_penalty_weight!r}"
            )
        self.negative_output_penalty_weight = weight
        past_dim = int(self.icl_past_exogenous_dim)
        future_dim = int(self.icl_future_exogenous_dim)
        if past_dim < 0 or future_dim < 0:
            raise ValueError("SELLM ICL exogenous dimensions must be non-negative.")
        if (past_dim == 0) != (future_dim == 0):
            raise ValueError("SELLM ICL requires both past and future exogenous features.")
        schema_hash = str(self.icl_exogenous_schema_hash or "").strip()
        if bool(past_dim) != bool(schema_hash):
            raise ValueError(
                "SELLM ICL exogenous dimensions and schema hash must be configured together."
            )
        if past_dim and not bool(self.icl_enabled):
            raise ValueError("SELLM ICL exogenous configuration requires icl_enabled=True.")
        if schema_hash and (
            len(schema_hash) != 64
            or any(character not in "0123456789abcdef" for character in schema_hash)
        ):
            raise ValueError("SELLM ICL exogenous schema hash must be lowercase SHA256.")
        self.icl_past_exogenous_dim = past_dim
        self.icl_future_exogenous_dim = future_dim
        self.icl_exogenous_schema_hash = schema_hash or None

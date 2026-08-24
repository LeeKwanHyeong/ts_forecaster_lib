from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from modeling_module.training.config import TrainingConfig


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
    semantic_vocab_size: int = 1024
    semantic_top_k: int = 32
    tscc_latent_dim: int = 8
    tscc_hidden_dim: int = 64
    tscc_kl_weight: float = 1e-4

    # LLM backbone
    use_pretrained_llm: bool = True
    llm_source: Literal["huggingface", "local"] = "huggingface"
    llm_model_name: str = "Qwen/Qwen2-0.5B"
    llm_local_path: Optional[str] = None
    llm_revision: Optional[str] = None
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

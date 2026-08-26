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
    output_head_mode: Literal[
        "identity",
        "softplus",
        "zero_inflated_softplus",
    ] = "identity"
    output_head_hidden_dim: int = 16
    output_head_softplus_beta: float = 1.0
    output_head_initial_nonzero_probability: float = 0.5
    output_calibration_mode: Literal["none", "validation_scalar"] = "none"
    output_calibration_scale: float = 1.0
    output_calibration_min_scale: float = 0.5
    output_calibration_max_scale: float = 1.5
    output_calibration_fitted: bool = False
    output_calibration_source_fingerprint: Optional[str] = None
    negative_output_penalty_weight: float = 0.0
    icl_enabled: bool = False
    icl_past_exogenous_dim: int = 0
    icl_future_exogenous_dim: int = 0
    icl_exogenous_schema_hash: Optional[str] = None

    def __post_init__(self) -> None:
        output_head_mode = str(self.output_head_mode).strip().lower()
        supported_output_heads = {
            "identity",
            "softplus",
            "zero_inflated_softplus",
        }
        if output_head_mode not in supported_output_heads:
            raise ValueError(
                "output_head_mode must be one of "
                f"{sorted(supported_output_heads)}, got {self.output_head_mode!r}"
            )
        hidden_dim = int(self.output_head_hidden_dim)
        if hidden_dim <= 0:
            raise ValueError("output_head_hidden_dim must be positive.")
        softplus_beta = float(self.output_head_softplus_beta)
        if not math.isfinite(softplus_beta) or softplus_beta <= 0.0:
            raise ValueError("output_head_softplus_beta must be finite and > 0.")
        initial_probability = float(self.output_head_initial_nonzero_probability)
        if not math.isfinite(initial_probability) or not 0.0 < initial_probability < 1.0:
            raise ValueError(
                "output_head_initial_nonzero_probability must be finite and in (0, 1)."
            )
        weight = float(self.negative_output_penalty_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                "negative_output_penalty_weight must be finite and >= 0, "
                f"got {self.negative_output_penalty_weight!r}"
            )
        self.negative_output_penalty_weight = weight
        if output_head_mode != "identity" and bool(self.final_nonneg):
            raise ValueError(
                "final_nonneg and output_head_mode cannot both apply an output transform."
            )
        if output_head_mode != "identity" and weight > 0.0:
            raise ValueError(
                "negative_output_penalty_weight is redundant with a non-negative "
                "SELLM output_head_mode."
            )
        self.output_head_mode = output_head_mode
        self.output_head_hidden_dim = hidden_dim
        self.output_head_softplus_beta = softplus_beta
        self.output_head_initial_nonzero_probability = initial_probability
        calibration_mode = str(self.output_calibration_mode).strip().lower()
        if calibration_mode not in {"none", "validation_scalar"}:
            raise ValueError(
                "output_calibration_mode must be 'none' or 'validation_scalar', "
                f"got {self.output_calibration_mode!r}."
            )
        calibration_scale = float(self.output_calibration_scale)
        calibration_min_scale = float(self.output_calibration_min_scale)
        calibration_max_scale = float(self.output_calibration_max_scale)
        if not all(
            math.isfinite(value)
            for value in (
                calibration_scale,
                calibration_min_scale,
                calibration_max_scale,
            )
        ):
            raise ValueError("SELLM output calibration scales must be finite.")
        if calibration_min_scale <= 0.0 or calibration_max_scale < calibration_min_scale:
            raise ValueError(
                "SELLM output calibration bounds must satisfy "
                "0 < min_scale <= max_scale."
            )
        if not calibration_min_scale <= calibration_scale <= calibration_max_scale:
            raise ValueError(
                "output_calibration_scale must be within the configured calibration bounds."
            )
        calibration_fitted = bool(self.output_calibration_fitted)
        calibration_fingerprint = str(
            self.output_calibration_source_fingerprint or ""
        ).strip()
        if calibration_mode == "none":
            if (
                calibration_scale != 1.0
                or calibration_fitted
                or calibration_fingerprint
            ):
                raise ValueError(
                    "output_calibration_mode='none' requires scale=1.0, "
                    "fitted=False, and no source fingerprint."
                )
        elif output_head_mode != "softplus":
            raise ValueError(
                "validation_scalar output calibration is supported only with "
                "output_head_mode='softplus'."
            )
        elif not bool(self.icl_enabled):
            raise ValueError(
                "validation_scalar output calibration requires SELLM ICL training."
            )
        elif not calibration_fitted and calibration_scale != 1.0:
            raise ValueError(
                "An unfitted validation_scalar calibration must start with scale=1.0."
            )
        if calibration_fitted != bool(calibration_fingerprint):
            raise ValueError(
                "A fitted SELLM output calibration requires a source fingerprint, "
                "and an unfitted calibration must not provide one."
            )
        if calibration_fingerprint and (
            len(calibration_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in calibration_fingerprint
            )
        ):
            raise ValueError(
                "output_calibration_source_fingerprint must be lowercase SHA256."
            )
        self.output_calibration_mode = calibration_mode
        self.output_calibration_scale = calibration_scale
        self.output_calibration_min_scale = calibration_min_scale
        self.output_calibration_max_scale = calibration_max_scale
        self.output_calibration_fitted = calibration_fitted
        self.output_calibration_source_fingerprint = calibration_fingerprint or None
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

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from modeling_module.training.config import TrainingConfig


@dataclass
class TimeXerConfig(TrainingConfig):
    """
    Minimal TimeXer configuration aligned to the paper's historical-exogenous setup.

    Notes:
    - v1 integrates only past continuous exogenous inputs.
    - The official TimeXer implementation uses non-overlapping patches, so `lookback`
      is expected to be divisible by `patch_len`.
    """

    # Data / IO
    y_dim: int = 1
    past_exo_cont_dim: int = 0

    # Core architecture
    patch_len: int = 16
    d_model: int = 128
    n_heads: int = 8
    d_ff: int = 256
    e_layers: int = 3
    dropout: float = 0.1
    factor: int = 5
    activation: Literal["relu", "gelu"] = "gelu"

    # Normalization from the official implementation.
    use_norm: bool = True

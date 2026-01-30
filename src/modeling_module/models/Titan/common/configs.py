from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TitanConfig:
    # -------------------------
    # Data / IO
    # -------------------------
    lookback: int = 52
    horizon: int = 27

    # past exogenous
    past_exo_cont_dim: int = 0
    past_exo_cat_dims: Optional[List[int]] = None
    past_exo_cat_embed_dims: Optional[List[int]] = None

    final_clamp_nonneg: bool = False

    # future exogenous
    exo_dim: int = 0  # future exo dim

    # -------------------------
    # Model dims
    # -------------------------
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 512
    dropout: float = 0.1

    # -------------------------
    # Memory (attention-side)
    # -------------------------
    contextual_mem_size: int = 32
    persistent_mem_size: int = 32
    use_context_update: bool = False

    # -------------------------
    # Positional embedding (encoder)
    # -------------------------
    use_pos_emb: bool = True
    max_len: int = 512

    # -------------------------
    # LMM (local memory matching)
    # -------------------------
    mem_size: int = 128
    mem_topk: int = 8

    # -------------------------
    # RevIN
    # -------------------------
    use_revin: bool = True

    # -------------------------
    # Output / head
    # -------------------------
    clamp_min: Optional[float] = 0.0
    clamp_max: Optional[float] = None


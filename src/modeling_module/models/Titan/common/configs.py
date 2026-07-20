from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class TitanConfig:
    # -------------------------
    # Data / IO
    # -------------------------
    lookback: int = 52
    horizon: int = 27

    # future exogenous
    future_exo_dim: int = 0  # future exo dim (B, H, E_f)

    # past exogenous
    past_exo_cont_dim: int = 0  # 연속형 과거 외생 변수 차원 (B, L, E_p_cont)
    past_exo_cat_dim: int = 0   # (단일) categorical vocab size (cardinality). 0이면 미사용
    past_exo_cat_embed_dim: Optional[int] = None  # (단일) cat embedding dim. None이면 자동 결정

    final_clamp_nonneg: bool = False

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
    loss: Any = None
    loss_mode: str = "point"
    out_mul: int = 1
    param_names: Optional[List[str]] = None
    dist_name: Optional[str] = None
    clamp_min: Optional[float] = 0.0
    clamp_max: Optional[float] = None

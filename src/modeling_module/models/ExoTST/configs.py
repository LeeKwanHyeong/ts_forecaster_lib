# modeling_module/models/ExoTST/configs.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from modeling_module.training.config import TrainingConfig


@dataclass
class ExoTSTConfig(TrainingConfig):
    # -------------------------
    # Data / IO
    # -------------------------
    lookback: int = 52
    horizon: int = 27

    y_dim: int = 1  # endogenous channel count (Cy)

    # exogenous feature dims (continuous)
    exo_dim_past: int = 0     # E_p
    exo_dim_future: int = 0   # E_f

    use_past_exo: bool = True
    use_future_exo: bool = True

    # missing handling for exogenous
    # - "zero": NaN -> 0
    # - "zero+indicator": NaN -> 0 and append missing-indicator channels (same dim)
    exo_nan_policy: Literal["zero", "zero+indicator"] = "zero+indicator"

    # -------------------------
    # Patching
    # -------------------------
    patch_len: int = 16
    stride: int = 8  # patch stride

    # -------------------------
    # Model dims
    # -------------------------
    d_model: int = 128
    n_heads: int = 8
    d_ff: int = 256

    dropout: float = 0.1
    attn_dropout: float = 0.1

    # exogenous encoders
    exo_enc_layers: int = 2

    # fusion blocks (cross-temporal modality fusion)
    fusion_layers: int = 2

    # endogenous decoder
    endo_dec_layers: int = 2

    # memory build mode
    # - "all": concat(past tokens + future tokens) for each exo channel
    # - "agg": use only aggregation tokens from past/future (2 tokens per exo channel)
    exo_memory_mode: Literal["all", "agg"] = "all"

    # -------------------------
    # Normalization
    # -------------------------
    use_revin: bool = True
    revin_affine: bool = True
    revin_eps: float = 1e-5
    revin_subtract_last: bool = False

    # -------------------------
    # Head
    # -------------------------
    head_type: Literal["point"] = "point"  # 확장: quantile, dist 등

    # -------------------------
    # Safety / Debug
    # -------------------------
    strict_shape: bool = True  # lookback/horizon mismatch 시 즉시 에러
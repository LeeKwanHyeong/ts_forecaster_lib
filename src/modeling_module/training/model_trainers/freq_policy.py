"""freq_policy.py

Frequency-dependent hyperparameter policy.

This module centralizes:
- calendar date_type (dt_char) used by exogenous calendar callbacks
- patching hyperparameters (patch_len, stride)
- canonical season period (season_period)

Keeping this in one place prevents scattered if/else blocks across training code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


FreqName = Literal["weekly", "monthly", "daily", "hourly"]


@dataclass(frozen=True)
class FreqSpec:
    """Frequency-dependent hyperparameters used by patch-based models."""
    freq: str
    dt_char: str
    patch_len: int
    stride: int
    season_period: int


def get_freq_spec(freq: str) -> FreqSpec:
    """Centralized frequency policy (patching + seasonal period + calendar dtype).

    Args:
        freq: One of {"weekly","monthly","daily","hourly"} (case-insensitive).
              Unknown values fall back to "weekly" dt_char and "monthly" patch policy.

    Returns:
        FreqSpec
    """
    f = str(freq).strip().lower()
    dt_char = {"weekly": "W", "monthly": "M", "daily": "D", "hourly": "H"}.get(f, "W")

    if f == "hourly":
        return FreqSpec(freq=f, dt_char=dt_char, patch_len=24, stride=12, season_period=24)
    if f == "daily":
        return FreqSpec(freq=f, dt_char=dt_char, patch_len=14, stride=7, season_period=7)
    if f == "weekly":
        return FreqSpec(freq=f, dt_char=dt_char, patch_len=27, stride=8, season_period=52)

    # monthly (default)
    return FreqSpec(freq=f, dt_char=dt_char, patch_len=6, stride=3, season_period=12)


# Backward-compatible alias (used by older codepaths)
_get_freq_spec = get_freq_spec

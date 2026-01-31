"""total_train_modular.py

Facade module to preserve existing import paths.

If you plan to replace an existing `total_train.py`, you can either:
- rename this file to `total_train.py`, OR
- update your import sites to use `total_train.py` directly.

This file simply re-exports the public entrypoints from orchestration.
"""

from __future__ import annotations

try:
    from .total_train import (
        run_total_train,
        run_total_train_weekly,
        run_total_train_monthly,
        run_total_train_daily,
        run_total_train_hourly,
    )
except Exception:  # pragma: no cover
    from total_train import (  # type: ignore
        run_total_train,
        run_total_train_weekly,
        run_total_train_monthly,
        run_total_train_daily,
        run_total_train_hourly,
    )

# Backward-compatible alias used in some older notebooks/scripts
_run_total_train_generic = run_total_train

__all__ = [
    "run_total_train",
    "run_total_train_weekly",
    "run_total_train_monthly",
    "run_total_train_daily",
    "run_total_train_hourly",
    "_run_total_train_generic",
]

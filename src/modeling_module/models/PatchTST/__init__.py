# src/modeling_module/models/PatchTST/__init__.py
from __future__ import annotations

from typing import TYPE_CHECKING

from .common import PatchTSTConfig, PatchTSTConfigMonthly, PatchTSTConfigWeekly

__all__ = [
    # configs (eager)
    "PatchTSTConfig",
    "PatchTSTConfigMonthly",
    "PatchTSTConfigWeekly",
    # supervised (lazy)
    "SupervisedBackbone",
    "PointHead",
    "QuantileHead",
    "PatchTSTPointModel",
    "PatchTSTQuantileModel",
    # subpackages
    "common",
    "supervised",
    "self_supervised",
]

if TYPE_CHECKING:
    from .supervised import SupervisedBackbone, PointHead, QuantileHead, PatchTSTPointModel, PatchTSTQuantileModel

_LAZY = {
    "SupervisedBackbone": (".supervised", "SupervisedBackbone"),
    "PointHead": (".supervised", "PointHead"),
    "QuantileHead": (".supervised", "QuantileHead"),
    "PatchTSTPointModel": (".supervised", "PatchTSTPointModel"),
    "PatchTSTQuantileModel": (".supervised", "PatchTSTQuantileModel"),
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_path, attr = _LAZY[name]
    from importlib import import_module

    try:
        mod = import_module(module_path, package=__name__)
        value = getattr(mod, attr)
    except ImportError as e:
        raise ImportError(
            f"Failed to import '{name}' from PatchTST. "
            f"Check torch/numpy installation and active environment isolation."
        ) from e

    globals()[name] = value
    return value
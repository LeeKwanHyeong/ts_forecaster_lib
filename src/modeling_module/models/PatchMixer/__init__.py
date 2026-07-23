# src/modeling_module/models/PatchMixer/__init__.py
from __future__ import annotations

from typing import TYPE_CHECKING

from .common import PatchMixerConfig, PatchMixerExogenousConfig

__all__ = [
    "PatchMixerConfig",
    "PatchMixerExogenousConfig",
    "PatchMixerModel",
    "PatchMixerExogenousModel",
]

if TYPE_CHECKING:
    from .PatchMixer import PatchMixerModel
    from .variants import PatchMixerExogenousModel

_LAZY = {
    "PatchMixerModel": (".PatchMixer", "PatchMixerModel"),
    "PatchMixerExogenousModel": (".variants", "PatchMixerExogenousModel"),
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
            f"Failed to import '{name}' from PatchMixer. "
            f"Check torch/numpy installation and environment isolation."
        ) from e

    globals()[name] = value
    return value

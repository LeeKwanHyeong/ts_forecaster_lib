# src/modeling_module/models/PatchMixer/__init__.py
from __future__ import annotations

from typing import TYPE_CHECKING

# Config는 가볍게 eager
from .common import PatchMixerConfig

__all__ = [
    "PatchMixerConfig",
    "BaseModel",
    "QuantileModel",
    "PatchMixerBackbone",
    "MultiScalePatchMixerBackbone",
    "make_patch_cfgs",
]

if TYPE_CHECKING:
    from .PatchMixer import BaseModel, QuantileModel, make_patch_cfgs
    from .backbone import PatchMixerBackbone, MultiScalePatchMixerBackbone

_LAZY = {
    "BaseModel": (".PatchMixer", "BaseModel"),
    "QuantileModel": (".PatchMixer", "QuantileModel"),
    "make_patch_cfgs": (".PatchMixer", "make_patch_cfgs"),
    "PatchMixerBackbone": (".backbone", "PatchMixerBackbone"),
    "MultiScalePatchMixerBackbone": (".backbone", "MultiScalePatchMixerBackbone"),
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
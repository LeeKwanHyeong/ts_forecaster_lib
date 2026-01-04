# src/modeling_module/models/PatchMixer/common/__init__.py
from __future__ import annotations

from typing import TYPE_CHECKING

# Config는 가볍기 때문에 eager import 권장
from .configs import PatchMixerConfig

__all__ = [
    "PatchMixerConfig",
    "SimpleUnfoldProjector",
    "DynamicPatcherMoS",
    "DynamicOffsetPatcher",
]

if TYPE_CHECKING:
    # 타입체크 시에만 실제 import
    from .patching import SimpleUnfoldProjector, DynamicPatcherMoS, DynamicOffsetPatcher

_LAZY = {
    "SimpleUnfoldProjector": (".patching", "SimpleUnfoldProjector"),
    "DynamicPatcherMoS": (".patching", "DynamicPatcherMoS"),
    "DynamicOffsetPatcher": (".patching", "DynamicOffsetPatcher"),
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
            f"Failed to import '{name}'. "
            f"This usually indicates missing/broken binary deps (torch/numpy) "
            f"in the active environment."
        ) from e

    globals()[name] = value
    return value
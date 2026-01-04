# src/modeling_module/models/Titan/common/__init__.py
from __future__ import annotations

from .configs import TitanConfig

__all__ = [
    "TitanConfig",
    "TitanDecoder",
    "TitanDecoderLayer",
    "MemoryAttention",
    "PositionWiseFFN",
    "LMM",
]

_LAZY = {
    "TitanDecoder": (".decoder", "TitanDecoder"),
    "TitanDecoderLayer": (".decoder", "TitanDecoderLayer"),
    "MemoryAttention": (".memory", "MemoryAttention"),
    "PositionWiseFFN": (".memory", "PositionWiseFFN"),
    "LMM": (".memory", "LMM"),
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_path, attr = _LAZY[name]
    from importlib import import_module

    mod = import_module(module_path, package=__name__)
    value = getattr(mod, attr)
    globals()[name] = value  # cache
    return value
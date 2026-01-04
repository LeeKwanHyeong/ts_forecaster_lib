# src/modeling_module/models/Titan/__init__.py
from __future__ import annotations

from typing import TYPE_CHECKING
from .common.configs import TitanConfig

__all__ = [
    "TitanConfig",
    "TitanBaseModel",
    "TitanLMMModel",
    "TitanSeq2SeqModel",
    "TitanBackbone",
    "MemoryEncoder",
    # common exports (optional convenience)
    "TitanDecoder",
    "TitanDecoderLayer",
    "MemoryAttention",
    "PositionWiseFFN",
    "LMM",
]

if TYPE_CHECKING:
    from .Titans import TitanBaseModel, TitanLMMModel, TitanSeq2SeqModel
    from .backbone import TitanBackbone, MemoryEncoder
    from .common.decoder import TitanDecoder, TitanDecoderLayer
    from .common.memory import MemoryAttention, PositionWiseFFN, LMM

_LAZY = {
    "TitanBaseModel": (".Titans", "TitanBaseModel"),
    "TitanLMMModel": (".Titans", "TitanLMMModel"),
    "TitanSeq2SeqModel": (".Titans", "TitanSeq2SeqModel"),
    "TitanBackbone": (".backbone", "TitanBackbone"),
    "MemoryEncoder": (".backbone", "MemoryEncoder"),
    "TitanDecoder": (".common.decoder", "TitanDecoder"),
    "TitanDecoderLayer": (".common.decoder", "TitanDecoderLayer"),
    "MemoryAttention": (".common.memory", "MemoryAttention"),
    "PositionWiseFFN": (".common.memory", "PositionWiseFFN"),
    "LMM": (".common.memory", "LMM"),
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_path, attr = _LAZY[name]
    from importlib import import_module

    mod = import_module(module_path, package=__name__)
    value = getattr(mod, attr)
    globals()[name] = value
    return value
# src/modeling_module/models/PatchTST/common/__init__.py
from __future__ import annotations

from typing import TYPE_CHECKING

from .configs import (
    AttentionConfig,
    HeadConfig,
    PatchTSTConfig,
    PatchTSTConfigMonthly,
    PatchTSTConfigWeekly,
)

__all__ = [
    # configs (eager)
    "AttentionConfig",
    "HeadConfig",
    "PatchTSTConfig",
    "PatchTSTConfigMonthly",
    "PatchTSTConfigWeekly",
    # basics (lazy)
    "Transpose",
    "SigmoidRange",
    "LinBnDrop",
    "sigmoid_range",
    "get_activation_fn",
    # patching (lazy)
    "compute_patch_num",
    "do_patch",
    "unpatch",
    # positional encoding (lazy)
    "PositionalEncoding",
    "positional_encoding",
    # encoder (lazy)
    "TSTEncoderLayer",
    "TSTEncoder",
    "TSTiEncoder",
    # base backbone (lazy)
    "PatchBackboneBase",
]

if TYPE_CHECKING:
    from .basics import Transpose, SigmoidRange, LinBnDrop, sigmoid_range, get_activation_fn
    from .patching import compute_patch_num, do_patch, unpatch
    from .pos_encoding import PositionalEncoding, positional_encoding
    from .encoder import TSTEncoderLayer, TSTEncoder, TSTiEncoder
    from .backbone_base import PatchBackboneBase

_LAZY = {
    # basics
    "Transpose": (".basics", "Transpose"),
    "SigmoidRange": (".basics", "SigmoidRange"),
    "LinBnDrop": (".basics", "LinBnDrop"),
    "sigmoid_range": (".basics", "sigmoid_range"),
    "get_activation_fn": (".basics", "get_activation_fn"),
    # patching
    "compute_patch_num": (".patching", "compute_patch_num"),
    "do_patch": (".patching", "do_patch"),
    "unpatch": (".patching", "unpatch"),
    # positional encoding
    "PositionalEncoding": (".pos_encoding", "PositionalEncoding"),
    "positional_encoding": (".pos_encoding", "positional_encoding"),
    # encoder
    "TSTEncoderLayer": (".encoder", "TSTEncoderLayer"),
    "TSTEncoder": (".encoder", "TSTEncoder"),
    "TSTiEncoder": (".encoder", "TSTiEncoder"),
    # base backbone
    "PatchBackboneBase": (".backbone_base", "PatchBackboneBase"),
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
            f"Failed to import '{name}' from PatchTST.common. "
            f"Check torch/numpy installation and active environment isolation."
        ) from e

    globals()[name] = value
    return value
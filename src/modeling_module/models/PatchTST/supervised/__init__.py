# # src/modeling_module/models/PatchTST/supervised/__init__.py
# from __future__ import annotations
#
# from typing import TYPE_CHECKING
#
# __all__ = [
#     "SupervisedBackbone",
#     "PointHead",
#     "QuantileHead",
#     "PatchTSTPointModel",
#     "PatchTSTQuantileModel",
# ]
#
# if TYPE_CHECKING:
#     from .backbone import SupervisedBackbone
#     from .PatchTST import PointHead, QuantileHead, PatchTSTPointModel, PatchTSTQuantileModel
#
# _LAZY = {
#     "SupervisedBackbone": (".backbone", "SupervisedBackbone"),
#     "PointHead": (".PatchTST", "PointHead"),
#     "QuantileHead": (".PatchTST", "QuantileHead"),
#     "PatchTSTPointModel": (".PatchTST", "PatchTSTPointModel"),
#     "PatchTSTQuantileModel": (".PatchTST", "PatchTSTQuantileModel"),
# }
#
#
# def __getattr__(name: str):
#     if name not in _LAZY:
#         raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
#
#     module_path, attr = _LAZY[name]
#     from importlib import import_module
#
#     try:
#         mod = import_module(module_path, package=__name__)
#         value = getattr(mod, attr)
#     except ImportError as e:
#         raise ImportError(
#             f"Failed to import '{name}' from PatchTST.supervised. "
#             f"Check torch/numpy installation and active environment isolation."
#         ) from e
#
#     globals()[name] = value
#     return value
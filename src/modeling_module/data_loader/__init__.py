'''
* 사용법 *
from modeling_module.data_loader import MultiPartExoDataModule
'''

# src/modeling_module/data_loader/__init__.py
from __future__ import annotations

from typing import TYPE_CHECKING

# ---------------------------------------
# Public exports (stable API surface)
# ---------------------------------------
__all__ = [
    # MultiPart (no-exo)
    "MultiPartDataModule",
    "MultiPartInferenceDataset",
    "MultiPartTrainingDataset",
    "MultiPartAnchoredInferenceByYYYYWW",
    "MultiPartAnchoredInferenceByYYYYMM",

    # MultiPart (exo)
    "MultiPartExoDataModule",
    "MultiPartExoTrainingDataset",
    "MultiPartExoAnchoredInferenceDataset",
    "CategoryIndexer",
]

# ---------------------------------------
# Type checking imports (IDE friendly)
# ---------------------------------------
if TYPE_CHECKING:
    from .MultiPartDataModule import (
        MultiPartDataModule,
        MultiPartInferenceDataset,
        MultiPartTrainingDataset,
        MultiPartAnchoredInferenceByYYYYWW,
        MultiPartAnchoredInferenceByYYYYMM,
    )
    from .MultiPartExoDataModule import (
        MultiPartExoDataModule,
        MultiPartExoTrainingDataset,
        MultiPartExoAnchoredInferenceDataset,
        CategoryIndexer,
    )

# ---------------------------------------
# Lazy import (avoid heavy import at package import time)
# ---------------------------------------
_LAZY = {
    # MultiPart (no-exo)
    "MultiPartDataModule": (".MultiPartDataModule", "MultiPartDataModule"),
    "MultiPartInferenceDataset": (".MultiPartDataModule", "MultiPartInferenceDataset"),
    "MultiPartTrainingDataset": (".MultiPartDataModule", "MultiPartTrainingDataset"),
    "MultiPartAnchoredInferenceByYYYYWW": (".MultiPartDataModule", "MultiPartAnchoredInferenceByYYYYWW"),
    "MultiPartAnchoredInferenceByYYYYMM": (".MultiPartDataModule", "MultiPartAnchoredInferenceByYYYYMM"),

    # MultiPart (exo)
    "MultiPartExoDataModule": (".MultiPartExoDataModule", "MultiPartExoDataModule"),
    "MultiPartExoTrainingDataset": (".MultiPartExoDataModule", "MultiPartExoTrainingDataset"),
    "MultiPartExoAnchoredInferenceDataset": (".MultiPartExoDataModule", "MultiPartExoAnchoredInferenceDataset"),
    "CategoryIndexer": (".MultiPartExoDataModule", "CategoryIndexer"),
}


def __getattr__(name: str):
    """
    Lazy attribute resolution:
    - `import modeling_module.data_loader as dl` 시점에는 torch/polars import를 최대한 미룸
    - 실제로 dl.MultiPartDataModule 같은 속성 접근 시에만 해당 모듈을 import
    """
    if name not in _LAZY:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_path, attr = _LAZY[name]
    from importlib import import_module

    mod = import_module(module_path, package=__name__)
    value = getattr(mod, attr)
    globals()[name] = value  # 캐싱
    return value
"""Compatibility imports for the historical capitalized module path.

The authoritative implementations live in lowercase modular files. External
Consumers should use ``modeling_module`` or ``modeling_module.api`` instead of
importing these concrete classes.
"""

from modeling_module.data_loader.multi_part_exo_data_module import (
    CategoryIndexer,
    MultiPartExoDataModule,
)
from modeling_module.data_loader.multi_part_exo_dataset import (
    MultiPartExoAnchoredInferenceDataset,
    MultiPartExoTrainingDataset,
)

__all__ = [
    "CategoryIndexer",
    "MultiPartExoAnchoredInferenceDataset",
    "MultiPartExoDataModule",
    "MultiPartExoTrainingDataset",
]

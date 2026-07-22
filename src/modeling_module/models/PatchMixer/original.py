"""Compatibility imports for the canonical PatchMixer implementation.

The canonical config now lives in ``common.configs`` and the model classes live
in ``PatchMixer``. This module remains as a stable import path for existing
callers and checkpoints. The implementation is pinned to the upstream source
identified in ``provenance.py``.
"""

from .PatchMixer import (
    PatchMixerOriginalBackbone,
    PatchMixerOriginalLayer,
    PatchMixerOriginalModel,
    PatchMixerOriginalRevIN,
)
from .common.configs import PatchMixerOriginalConfig

__all__ = [
    "PatchMixerOriginalConfig",
    "PatchMixerOriginalRevIN",
    "PatchMixerOriginalLayer",
    "PatchMixerOriginalBackbone",
    "PatchMixerOriginalModel",
]

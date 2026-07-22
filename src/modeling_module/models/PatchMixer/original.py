"""Compatibility imports for the canonical PatchMixer implementation.

The canonical config lives in ``common.configs``, the computation layers live
in ``backbone``, and the public model wrapper lives in ``PatchMixer``. This
module remains as a stable import path for existing callers and checkpoints.
The implementation is pinned to the upstream source identified in
``provenance.py``.
"""

from .backbone import (
    PatchMixerOriginalBackbone,
    PatchMixerOriginalLayer,
    PatchMixerOriginalRevIN,
)
from .PatchMixer import PatchMixerOriginalModel
from .common.configs import PatchMixerOriginalConfig

__all__ = [
    "PatchMixerOriginalConfig",
    "PatchMixerOriginalRevIN",
    "PatchMixerOriginalLayer",
    "PatchMixerOriginalBackbone",
    "PatchMixerOriginalModel",
]

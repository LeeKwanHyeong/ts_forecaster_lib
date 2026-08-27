"""Configuration contracts for nearest-lifecycle retrieval forecasting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping

import numpy as np

from modeling_module.models.CGMM.configs import (
    CGMMCorrectionConfig,
    CGMMPreprocessingConfig,
)


class SimilarLifecycleDistanceProfile(StrEnum):
    """Feature groups allowed to influence lifecycle retrieval distance."""

    DEMAND_SHAPE = "demand_shape"
    DEMAND_SHAPE_SCALE = "demand_shape_scale"
    DEMAND_SHAPE_CATEGORIES = "demand_shape_categories"
    DEMAND_SHAPE_STATIC = "demand_shape_static"
    DEMAND_SHAPE_SALES = "demand_shape_sales"
    ALL = "all"


@dataclass(frozen=True, slots=True)
class SimilarLifecycleConfig:
    """Nearest-neighbor capacity, distance, and interval policy."""

    neighbor_count: int = 15
    distance_profile: SimilarLifecycleDistanceProfile = (
        SimilarLifecycleDistanceProfile.ALL
    )
    distance_floor: float = 1e-8
    query_batch_size: int = 256
    interval_z: float = 1.6448536269514722

    def __post_init__(self) -> None:
        for field_name in ("neighbor_count", "query_batch_size"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if not isinstance(
            self.distance_profile,
            SimilarLifecycleDistanceProfile,
        ):
            raise TypeError(
                "distance_profile must be SimilarLifecycleDistanceProfile"
            )
        for field_name in ("distance_floor", "interval_z"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")

    @classmethod
    def from_config(
        cls,
        value: "SimilarLifecycleConfig | Mapping[str, Any] | Any",
    ) -> "SimilarLifecycleConfig":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            payload = dict(value)
        elif hasattr(value, "__dict__"):
            payload = dict(vars(value))
        else:
            raise TypeError(
                f"Unsupported Similar Lifecycle config type: {type(value)!r}"
            )
        if "distance_profile" in payload and not isinstance(
            payload["distance_profile"],
            SimilarLifecycleDistanceProfile,
        ):
            payload["distance_profile"] = SimilarLifecycleDistanceProfile(
                payload["distance_profile"]
            )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["distance_profile"] = self.distance_profile.value
        return payload


def default_similar_lifecycle_preprocessing() -> CGMMPreprocessingConfig:
    """Return the frozen DSDM-compatible lifecycle feature layout."""

    return CGMMPreprocessingConfig(feature_profile="static_observed_v1")


# The cohort/tail equations and train-only condition transform are shared with
# CGMM so both lifecycle baselines consume exactly the same frozen features.
SimilarLifecycleCorrectionConfig = CGMMCorrectionConfig
SimilarLifecyclePreprocessingConfig = CGMMPreprocessingConfig


__all__ = [
    "SimilarLifecycleConfig",
    "SimilarLifecycleCorrectionConfig",
    "SimilarLifecycleDistanceProfile",
    "SimilarLifecyclePreprocessingConfig",
    "default_similar_lifecycle_preprocessing",
]

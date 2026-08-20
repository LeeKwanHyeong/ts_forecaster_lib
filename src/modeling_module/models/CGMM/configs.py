"""Configuration contracts for conditional lifecycle GMM forecasting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final, Literal, Mapping, TypeAlias

import numpy as np

from modeling_module.data_loader.lifecycle_contracts import LTB_FORECAST_MONTHS


CGMMPreprocessingProfile: TypeAlias = Literal[
    "generic_v1",
    "static_observed_v1",
    "static_observed_m0_v1",
]
CGMM_PREPROCESSING_PROFILES: Final = frozenset(
    {"generic_v1", "static_observed_v1", "static_observed_m0_v1"}
)


@dataclass(frozen=True, slots=True)
class CGMMPreprocessingConfig:
    """Numerical safeguards fitted from training samples only."""

    quantity_scale_floor: float = 1.0
    standard_deviation_floor: float = 1e-8
    feature_profile: CGMMPreprocessingProfile = "generic_v1"

    def __post_init__(self) -> None:
        for field_name in (
            "quantity_scale_floor",
            "standard_deviation_floor",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
        if self.feature_profile not in CGMM_PREPROCESSING_PROFILES:
            raise ValueError(
                "feature_profile must be 'generic_v1', "
                "'static_observed_v1', or 'static_observed_m0_v1'"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CGMMConfig:
    """Capacity and numerical policy for the conditional Gaussian mixture."""

    component_count: int = 2
    target_component_count: int = 2
    covariance_regularization: float = 1e-4
    initialization_count: int = 2
    max_iterations: int = 200
    random_seed: int = 42
    interval_z: float = 1.6448536269514722

    def __post_init__(self) -> None:
        for field_name in (
            "component_count",
            "target_component_count",
            "initialization_count",
            "max_iterations",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if isinstance(self.random_seed, bool) or not isinstance(
            self.random_seed, int
        ):
            raise ValueError("random_seed must be an integer")
        for field_name in ("covariance_regularization", "interval_z"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
        if self.target_component_count > LTB_FORECAST_MONTHS:
            raise ValueError(
                f"target_component_count cannot exceed {LTB_FORECAST_MONTHS}"
            )

    @classmethod
    def from_config(cls, value: "CGMMConfig | Mapping[str, Any] | Any") -> "CGMMConfig":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        if hasattr(value, "__dict__"):
            return cls(**dict(vars(value)))
        raise TypeError(f"Unsupported CGMM config type: {type(value)!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CGMMCorrectionConfig:
    """Leakage-safe cohort and long-tail correction policy."""

    name: str = "cohort-half-tail72"
    cohort_strength: float = 0.5
    maximum_monthly_log_slope: float = 0.03
    tail_start_month: int = 36
    tail_half_life_months: float | None = 72.0
    scale_gate_quantile: float | None = None
    minimum_calibration_cohorts: int = 3
    correction_floor: float = 0.20
    correction_ceiling: float = 1.50

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be non-empty text")
        for field_name in (
            "cohort_strength",
            "maximum_monthly_log_slope",
            "correction_floor",
            "correction_ceiling",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if not 0.0 <= self.cohort_strength <= 1.0:
            raise ValueError("cohort_strength must be between zero and one")
        if self.maximum_monthly_log_slope <= 0.0:
            raise ValueError("maximum_monthly_log_slope must be positive")
        if (
            isinstance(self.tail_start_month, bool)
            or not isinstance(self.tail_start_month, int)
            or not 1 <= self.tail_start_month < LTB_FORECAST_MONTHS
        ):
            raise ValueError(
                f"tail_start_month must be an integer from 1 to {LTB_FORECAST_MONTHS - 1}"
            )
        if self.tail_half_life_months is not None and (
            isinstance(self.tail_half_life_months, bool)
            or not np.isfinite(self.tail_half_life_months)
            or self.tail_half_life_months <= 0.0
        ):
            raise ValueError("tail_half_life_months must be positive when set")
        if self.scale_gate_quantile is not None and (
            isinstance(self.scale_gate_quantile, bool)
            or not np.isfinite(self.scale_gate_quantile)
            or not 0.0 < self.scale_gate_quantile < 1.0
        ):
            raise ValueError("scale_gate_quantile must be between zero and one")
        if (
            isinstance(self.minimum_calibration_cohorts, bool)
            or not isinstance(self.minimum_calibration_cohorts, int)
            or self.minimum_calibration_cohorts < 2
        ):
            raise ValueError("minimum_calibration_cohorts must be at least two")
        if not 0.0 < self.correction_floor < self.correction_ceiling:
            raise ValueError(
                "correction bounds must be positive and strictly ordered"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "CGMM_PREPROCESSING_PROFILES",
    "CGMMConfig",
    "CGMMCorrectionConfig",
    "CGMMPreprocessingConfig",
    "CGMMPreprocessingProfile",
]

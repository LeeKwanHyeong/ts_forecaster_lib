"""Immutable contracts for Similar Lifecycle forecasting and artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from modeling_module.data_loader.lifecycle_contracts import LTB_FORECAST_MONTHS
from modeling_module.models.CGMM.contracts import (
    CGMMCorrectionState,
    CGMMPreprocessingState,
    fingerprint_payload,
    freeze_float_array,
    require_sha256,
)


SIMILAR_LIFECYCLE_MODEL_KEY = "similar_lifecycle"
SIMILAR_LIFECYCLE_MODEL_ID = "modeling-module.similar-lifecycle.v1"
SIMILAR_LIFECYCLE_ARTIFACT_ID = (
    "modeling-module.similar-lifecycle-artifact.v1"
)
SIMILAR_LIFECYCLE_ARTIFACT_VERSION = 1

FloatArray = NDArray[np.float64]
SimilarLifecycleCorrectionState = CGMMCorrectionState
SimilarLifecyclePreprocessingState = CGMMPreprocessingState


class SimilarLifecycleContractError(ValueError):
    """Raised when a Similar Lifecycle value violates its public contract."""


@dataclass(frozen=True, slots=True)
class SimilarLifecycleRepositoryState:
    """Completed lifecycle repository required for nearest-neighbor inference."""

    sample_ids: tuple[str, ...]
    lifecycle_start_months: tuple[date, ...]
    distance_feature_names: tuple[str, ...]
    train_condition: FloatArray
    train_future_ratio: FloatArray

    def __post_init__(self) -> None:
        sample_count = len(self.sample_ids)
        if sample_count == 0 or len(set(self.sample_ids)) != sample_count:
            raise SimilarLifecycleContractError(
                "repository sample_ids must be non-empty and unique"
            )
        if any(not isinstance(value, str) or not value for value in self.sample_ids):
            raise SimilarLifecycleContractError(
                "repository sample_ids must contain non-empty text"
            )
        if len(self.lifecycle_start_months) != sample_count or any(
            not isinstance(value, date) or value.day != 1
            for value in self.lifecycle_start_months
        ):
            raise SimilarLifecycleContractError(
                "repository lifecycle_start_months must be month starts"
            )
        if not self.distance_feature_names or len(
            set(self.distance_feature_names)
        ) != len(self.distance_feature_names):
            raise SimilarLifecycleContractError(
                "distance_feature_names must be non-empty and unique"
            )
        condition = freeze_float_array(
            self.train_condition,
            field_name="train_condition",
            shape=(sample_count, len(self.distance_feature_names)),
        )
        future_ratio = freeze_float_array(
            self.train_future_ratio,
            field_name="train_future_ratio",
            shape=(sample_count, LTB_FORECAST_MONTHS),
            non_negative=True,
        )
        object.__setattr__(self, "train_condition", condition)
        object.__setattr__(self, "train_future_ratio", future_ratio)


@dataclass(frozen=True, slots=True)
class SimilarLifecyclePrediction:
    """Weighted lifecycle forecast, interval, and retrieval evidence."""

    sample_ids: tuple[str, ...]
    mean_forecast: FloatArray
    forecast_std: FloatArray
    lower_bound: FloatArray
    upper_bound: FloatArray
    neighbor_sample_ids: tuple[tuple[str, ...], ...]
    neighbor_weights: FloatArray
    neighbor_distances: FloatArray
    model_key: str
    model_id: str
    model_fingerprint: str
    preprocessing_fingerprint: str
    correction_fingerprint: str | None = None

    def __post_init__(self) -> None:
        sample_count = len(self.sample_ids)
        if sample_count == 0 or len(set(self.sample_ids)) != sample_count:
            raise SimilarLifecycleContractError(
                "prediction sample_ids must be non-empty and unique"
            )
        for field_name in (
            "mean_forecast",
            "forecast_std",
            "lower_bound",
            "upper_bound",
        ):
            object.__setattr__(
                self,
                field_name,
                freeze_float_array(
                    getattr(self, field_name),
                    field_name=field_name,
                    shape=(sample_count, LTB_FORECAST_MONTHS),
                    non_negative=True,
                ),
            )
        if (self.lower_bound > self.upper_bound).any():
            raise SimilarLifecycleContractError(
                "lower_bound cannot exceed upper_bound"
            )

        weights = np.asarray(self.neighbor_weights, dtype=np.float64)
        if weights.ndim != 2 or weights.shape[0] != sample_count:
            raise SimilarLifecycleContractError(
                "neighbor_weights must have shape (samples, neighbors)"
            )
        neighbor_count = weights.shape[1]
        if neighbor_count == 0 or len(self.neighbor_sample_ids) != sample_count:
            raise SimilarLifecycleContractError(
                "neighbor evidence must match the prediction sample count"
            )
        if any(
            len(values) != neighbor_count or any(not value for value in values)
            for values in self.neighbor_sample_ids
        ):
            raise SimilarLifecycleContractError(
                "each prediction must contain the same non-empty neighbor set"
            )
        weights = freeze_float_array(
            weights,
            field_name="neighbor_weights",
            shape=(sample_count, neighbor_count),
            non_negative=True,
        )
        if not np.allclose(weights.sum(axis=1), 1.0, rtol=1e-8, atol=1e-10):
            raise SimilarLifecycleContractError(
                "neighbor_weights must sum to one"
            )
        distances = freeze_float_array(
            self.neighbor_distances,
            field_name="neighbor_distances",
            shape=(sample_count, neighbor_count),
            non_negative=True,
        )
        object.__setattr__(self, "neighbor_weights", weights)
        object.__setattr__(self, "neighbor_distances", distances)

        if self.model_key != SIMILAR_LIFECYCLE_MODEL_KEY:
            raise SimilarLifecycleContractError(
                f"model_key must be {SIMILAR_LIFECYCLE_MODEL_KEY!r}"
            )
        if self.model_id != SIMILAR_LIFECYCLE_MODEL_ID:
            raise SimilarLifecycleContractError(
                f"model_id must be {SIMILAR_LIFECYCLE_MODEL_ID!r}"
            )
        require_sha256(self.model_fingerprint, field_name="model_fingerprint")
        require_sha256(
            self.preprocessing_fingerprint,
            field_name="preprocessing_fingerprint",
        )
        if self.correction_fingerprint is not None:
            require_sha256(
                self.correction_fingerprint,
                field_name="correction_fingerprint",
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible response payload."""

        return {
            "sample_ids": list(self.sample_ids),
            "mean_forecast": self.mean_forecast.tolist(),
            "forecast_std": self.forecast_std.tolist(),
            "lower_bound": self.lower_bound.tolist(),
            "upper_bound": self.upper_bound.tolist(),
            "neighbor_sample_ids": [
                list(values) for values in self.neighbor_sample_ids
            ],
            "neighbor_weights": self.neighbor_weights.tolist(),
            "neighbor_distances": self.neighbor_distances.tolist(),
            "metadata": {
                "model_key": self.model_key,
                "model_id": self.model_id,
                "model_fingerprint": self.model_fingerprint,
                "preprocessing_fingerprint": self.preprocessing_fingerprint,
                "correction_fingerprint": self.correction_fingerprint,
            },
        }


@dataclass(frozen=True, slots=True)
class SimilarLifecycleRollingEvidence:
    """One cohort predicted using only earlier completed lifecycles."""

    validation_month: date
    sample_ids: tuple[str, ...]
    observed_scale: FloatArray
    actual: FloatArray
    prediction: SimilarLifecyclePrediction

    def __post_init__(self) -> None:
        if not isinstance(self.validation_month, date) or self.validation_month.day != 1:
            raise SimilarLifecycleContractError(
                "validation_month must be a month-start date"
            )
        sample_count = len(self.sample_ids)
        if sample_count == 0 or len(set(self.sample_ids)) != sample_count:
            raise SimilarLifecycleContractError(
                "rolling sample_ids must be non-empty and unique"
            )
        if self.prediction.sample_ids != self.sample_ids:
            raise SimilarLifecycleContractError(
                "rolling prediction sample order mismatch"
            )
        object.__setattr__(
            self,
            "observed_scale",
            freeze_float_array(
                self.observed_scale,
                field_name="observed_scale",
                shape=(sample_count,),
                non_negative=True,
            ),
        )
        object.__setattr__(
            self,
            "actual",
            freeze_float_array(
                self.actual,
                field_name="actual",
                shape=(sample_count, LTB_FORECAST_MONTHS),
                non_negative=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class SimilarLifecycleArtifactReceipt:
    """Verified identity of one published Similar Lifecycle artifact."""

    artifact_dir: Path
    manifest_path: Path
    arrays_path: Path
    model_fingerprint: str
    artifact_fingerprint: str
    arrays_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "model_fingerprint",
            "artifact_fingerprint",
            "arrays_sha256",
        ):
            require_sha256(getattr(self, field_name), field_name=field_name)


__all__ = [
    "SIMILAR_LIFECYCLE_ARTIFACT_ID",
    "SIMILAR_LIFECYCLE_ARTIFACT_VERSION",
    "SIMILAR_LIFECYCLE_MODEL_ID",
    "SIMILAR_LIFECYCLE_MODEL_KEY",
    "SimilarLifecycleArtifactReceipt",
    "SimilarLifecycleContractError",
    "SimilarLifecycleCorrectionState",
    "SimilarLifecyclePrediction",
    "SimilarLifecyclePreprocessingState",
    "SimilarLifecycleRepositoryState",
    "SimilarLifecycleRollingEvidence",
    "fingerprint_payload",
]

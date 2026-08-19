"""Immutable input, output, correction, and artifact contracts for CGMM."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from modeling_module.data_loader.lifecycle_contracts import (
    LTB_FORECAST_MONTHS,
    LifecycleFeatureSchema,
)
from modeling_module.models.CGMM.configs import (
    CGMM_PREPROCESSING_PROFILES,
    CGMMCorrectionConfig,
)


CGMM_MODEL_KEY = "cgmm"
CGMM_MODEL_ID = "modeling-module.cgmm.v1"
CGMM_PREPROCESSING_ID = "modeling-module.cgmm-preprocessing.v1"
CGMM_ARTIFACT_ID = "modeling-module.cgmm-artifact.v1"
CGMM_ARTIFACT_VERSION = 1

FloatArray = NDArray[np.float64]


class CGMMContractError(ValueError):
    """Raised when a CGMM value violates its public contract."""


def fingerprint_payload(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_sha256(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CGMMContractError(f"{field_name} must be lowercase SHA-256")
    return value


def freeze_float_array(
    value: object,
    *,
    field_name: str,
    shape: tuple[int, ...] | None = None,
    non_negative: bool = False,
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise CGMMContractError(
            f"{field_name} must have shape {shape}, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise CGMMContractError(f"{field_name} must contain finite values")
    if non_negative and (array < 0.0).any():
        raise CGMMContractError(f"{field_name} must be non-negative")
    frozen = np.ascontiguousarray(array)
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True, slots=True)
class CGMMPreprocessingState:
    """Train-only feature statistics and categorical vocabularies."""

    contract_id: str
    dataset_fingerprint: str
    feature_schema: LifecycleFeatureSchema
    feature_profile: str
    quantity_scale_floor: float
    standard_deviation_floor: float
    numeric_slot_names: tuple[str, ...]
    numeric_fill_values: tuple[float, ...]
    categorical_slot_names: tuple[str, ...]
    categorical_vocabularies: tuple[tuple[str, ...], ...]
    condition_feature_names: tuple[str, ...]
    condition_means: tuple[float, ...]
    condition_scales: tuple[float, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        if self.contract_id != CGMM_PREPROCESSING_ID:
            raise CGMMContractError("unsupported CGMM preprocessing contract")
        require_sha256(
            self.dataset_fingerprint,
            field_name="dataset_fingerprint",
        )
        require_sha256(self.fingerprint, field_name="fingerprint")
        if not isinstance(self.feature_schema, LifecycleFeatureSchema):
            raise TypeError("feature_schema must be LifecycleFeatureSchema")
        if self.feature_profile not in CGMM_PREPROCESSING_PROFILES:
            raise CGMMContractError("unsupported CGMM preprocessing profile")
        if len(set(self.numeric_slot_names)) != len(self.numeric_slot_names):
            raise CGMMContractError("numeric_slot_names must be unique")
        if len(self.numeric_slot_names) != len(self.numeric_fill_values):
            raise CGMMContractError(
                "numeric fill values do not match numeric slots"
            )
        if len(set(self.categorical_slot_names)) != len(
            self.categorical_slot_names
        ):
            raise CGMMContractError("categorical_slot_names must be unique")
        if len(self.categorical_slot_names) != len(
            self.categorical_vocabularies
        ):
            raise CGMMContractError(
                "categorical vocabularies do not match categorical slots"
            )
        if any(not vocabulary for vocabulary in self.categorical_vocabularies):
            raise CGMMContractError("categorical vocabularies cannot be empty")
        feature_count = len(self.condition_feature_names)
        if (
            feature_count == 0
            or len(set(self.condition_feature_names)) != feature_count
            or len(self.condition_means) != feature_count
            or len(self.condition_scales) != feature_count
        ):
            raise CGMMContractError("condition feature statistics are invalid")
        numeric = np.asarray(
            (
                self.quantity_scale_floor,
                self.standard_deviation_floor,
                *self.numeric_fill_values,
                *self.condition_means,
                *self.condition_scales,
            ),
            dtype=np.float64,
        )
        if (
            not np.isfinite(numeric).all()
            or self.quantity_scale_floor <= 0.0
            or self.standard_deviation_floor <= 0.0
            or (np.asarray(self.condition_scales) <= 0.0).any()
        ):
            raise CGMMContractError(
                "preprocessing statistics must be finite with positive scales"
            )
        if self.fingerprint != fingerprint_payload(
            self.to_dict(include_fingerprint=False)
        ):
            raise CGMMContractError(
                "preprocessing fingerprint does not match its payload"
            )

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contract_id": self.contract_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "feature_schema": self.feature_schema.to_dict(),
            "feature_schema_fingerprint": self.feature_schema.fingerprint,
            "feature_profile": self.feature_profile,
            "quantity_scale_floor": self.quantity_scale_floor,
            "standard_deviation_floor": self.standard_deviation_floor,
            "numeric_slot_names": list(self.numeric_slot_names),
            "numeric_fill_values": list(self.numeric_fill_values),
            "categorical_slot_names": list(self.categorical_slot_names),
            "categorical_vocabularies": [
                list(vocabulary)
                for vocabulary in self.categorical_vocabularies
            ],
            "condition_feature_names": list(self.condition_feature_names),
            "condition_means": list(self.condition_means),
            "condition_scales": list(self.condition_scales),
        }
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CGMMPreprocessingState":
        if not isinstance(payload, Mapping):
            raise TypeError("preprocessing payload must be a mapping")
        expected_keys = {
            "contract_id",
            "dataset_fingerprint",
            "feature_schema",
            "feature_schema_fingerprint",
            "feature_profile",
            "quantity_scale_floor",
            "standard_deviation_floor",
            "numeric_slot_names",
            "numeric_fill_values",
            "categorical_slot_names",
            "categorical_vocabularies",
            "condition_feature_names",
            "condition_means",
            "condition_scales",
            "fingerprint",
        }
        if set(payload) != expected_keys:
            raise CGMMContractError(
                "preprocessing payload has an invalid schema"
            )
        schema = LifecycleFeatureSchema.from_dict(payload["feature_schema"])
        if payload.get("feature_schema_fingerprint") != schema.fingerprint:
            raise CGMMContractError("feature schema fingerprint mismatch")
        return cls(
            contract_id=str(payload["contract_id"]),
            dataset_fingerprint=str(payload["dataset_fingerprint"]),
            feature_schema=schema,
            feature_profile=str(payload["feature_profile"]),
            quantity_scale_floor=float(payload["quantity_scale_floor"]),
            standard_deviation_floor=float(
                payload["standard_deviation_floor"]
            ),
            numeric_slot_names=tuple(payload["numeric_slot_names"]),
            numeric_fill_values=tuple(
                float(value) for value in payload["numeric_fill_values"]
            ),
            categorical_slot_names=tuple(payload["categorical_slot_names"]),
            categorical_vocabularies=tuple(
                tuple(str(value) for value in vocabulary)
                for vocabulary in payload["categorical_vocabularies"]
            ),
            condition_feature_names=tuple(payload["condition_feature_names"]),
            condition_means=tuple(
                float(value) for value in payload["condition_means"]
            ),
            condition_scales=tuple(
                float(value) for value in payload["condition_scales"]
            ),
            fingerprint=str(payload["fingerprint"]),
        )


@dataclass(frozen=True, slots=True)
class CGMMPreparedBatch:
    """Lifecycle samples transformed by one frozen preprocessing state."""

    sample_ids: tuple[str, ...]
    lifecycle_start_months: tuple[date, ...]
    quantity_scale: FloatArray
    condition_matrix: FloatArray
    normalized_future: FloatArray | None
    preprocessing_fingerprint: str

    def __post_init__(self) -> None:
        sample_count = len(self.sample_ids)
        if sample_count == 0 or len(set(self.sample_ids)) != sample_count:
            raise CGMMContractError(
                "sample_ids must contain unique, non-empty values"
            )
        if any(not isinstance(value, str) or not value for value in self.sample_ids):
            raise CGMMContractError("sample_ids must contain non-empty text")
        if len(self.lifecycle_start_months) != sample_count or any(
            not isinstance(value, date) or value.day != 1
            for value in self.lifecycle_start_months
        ):
            raise CGMMContractError(
                "lifecycle_start_months must match sample_ids"
            )
        scale = freeze_float_array(
            self.quantity_scale,
            field_name="quantity_scale",
            shape=(sample_count,),
        )
        if (scale <= 0.0).any():
            raise CGMMContractError("quantity_scale must be positive")
        object.__setattr__(self, "quantity_scale", scale)
        condition = np.asarray(self.condition_matrix, dtype=np.float64)
        if condition.ndim != 2 or condition.shape[0] != sample_count:
            raise CGMMContractError(
                "condition_matrix must have shape (sample_count, feature_count)"
            )
        if condition.shape[1] == 0:
            raise CGMMContractError("condition_matrix must contain features")
        object.__setattr__(
            self,
            "condition_matrix",
            freeze_float_array(
                condition,
                field_name="condition_matrix",
                shape=condition.shape,
            ),
        )
        if self.normalized_future is not None:
            object.__setattr__(
                self,
                "normalized_future",
                freeze_float_array(
                    self.normalized_future,
                    field_name="normalized_future",
                    shape=(sample_count, LTB_FORECAST_MONTHS),
                    non_negative=True,
                ),
            )
        require_sha256(
            self.preprocessing_fingerprint,
            field_name="preprocessing_fingerprint",
        )


@dataclass(frozen=True, slots=True)
class CGMMPrediction:
    """Conditional candidate curves, moments, intervals, and provenance."""

    sample_ids: tuple[str, ...]
    component_probabilities: FloatArray
    candidate_curves: FloatArray
    mean_forecast: FloatArray
    forecast_std: FloatArray
    lower_bound: FloatArray
    upper_bound: FloatArray
    model_key: str
    model_id: str
    model_fingerprint: str
    preprocessing_fingerprint: str
    correction_fingerprint: str | None = None

    def __post_init__(self) -> None:
        sample_count = len(self.sample_ids)
        if sample_count == 0 or len(set(self.sample_ids)) != sample_count:
            raise CGMMContractError("prediction sample_ids must be unique")
        probabilities = np.asarray(
            self.component_probabilities,
            dtype=np.float64,
        )
        if (
            probabilities.ndim != 2
            or probabilities.shape[0] != sample_count
            or probabilities.shape[1] == 0
        ):
            raise CGMMContractError(
                "component_probabilities must have shape (samples, components)"
            )
        component_count = probabilities.shape[1]
        probabilities = freeze_float_array(
            probabilities,
            field_name="component_probabilities",
            shape=(sample_count, component_count),
            non_negative=True,
        )
        if not np.allclose(
            probabilities.sum(axis=1),
            1.0,
            rtol=1e-8,
            atol=1e-10,
        ):
            raise CGMMContractError("component probabilities must sum to one")
        object.__setattr__(self, "component_probabilities", probabilities)
        candidates = freeze_float_array(
            self.candidate_curves,
            field_name="candidate_curves",
            shape=(sample_count, component_count, LTB_FORECAST_MONTHS),
            non_negative=True,
        )
        object.__setattr__(self, "candidate_curves", candidates)
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
            raise CGMMContractError("lower_bound cannot exceed upper_bound")
        weighted = np.einsum(
            "nk,nkh->nh",
            self.component_probabilities,
            self.candidate_curves,
        )
        if not np.allclose(
            weighted,
            self.mean_forecast,
            rtol=1e-7,
            atol=1e-8,
        ):
            raise CGMMContractError(
                "mean_forecast must equal probability-weighted candidates"
            )
        if self.model_key != CGMM_MODEL_KEY:
            raise CGMMContractError(f"model_key must be {CGMM_MODEL_KEY!r}")
        if self.model_id != CGMM_MODEL_ID:
            raise CGMMContractError(f"model_id must be {CGMM_MODEL_ID!r}")
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
            "component_probabilities": self.component_probabilities.tolist(),
            "candidate_curves": self.candidate_curves.tolist(),
            "mean_forecast": self.mean_forecast.tolist(),
            "forecast_std": self.forecast_std.tolist(),
            "lower_bound": self.lower_bound.tolist(),
            "upper_bound": self.upper_bound.tolist(),
            "metadata": {
                "model_key": self.model_key,
                "model_id": self.model_id,
                "model_fingerprint": self.model_fingerprint,
                "preprocessing_fingerprint": self.preprocessing_fingerprint,
                "correction_fingerprint": self.correction_fingerprint,
            },
        }


@dataclass(frozen=True, slots=True)
class CGMMRollingEvidence:
    """One validation cohort predicted using only earlier completed cohorts."""

    validation_month: date
    sample_ids: tuple[str, ...]
    observed_scale: FloatArray
    actual: FloatArray
    prediction: CGMMPrediction

    def __post_init__(self) -> None:
        if not isinstance(self.validation_month, date) or self.validation_month.day != 1:
            raise CGMMContractError("validation_month must be a month-start date")
        sample_count = len(self.sample_ids)
        if sample_count == 0 or len(set(self.sample_ids)) != sample_count:
            raise CGMMContractError("rolling sample_ids must be non-empty and unique")
        if self.prediction.sample_ids != self.sample_ids:
            raise CGMMContractError("rolling prediction sample order mismatch")
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
class CGMMCorrectionState:
    """Correction fitted only from chronological rolling evidence."""

    config: CGMMCorrectionConfig
    reference_month_ordinal: int
    block_log_intercepts: FloatArray
    block_monthly_log_slopes: FloatArray
    scale_gate_pivot: float | None
    evidence_months: tuple[str, ...]
    evidence_fingerprint: str
    fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.config, CGMMCorrectionConfig):
            raise TypeError("config must be CGMMCorrectionConfig")
        if isinstance(self.reference_month_ordinal, bool) or not isinstance(
            self.reference_month_ordinal, int
        ):
            raise TypeError("reference_month_ordinal must be an integer")
        for field_name in (
            "block_log_intercepts",
            "block_monthly_log_slopes",
        ):
            object.__setattr__(
                self,
                field_name,
                freeze_float_array(
                    getattr(self, field_name),
                    field_name=field_name,
                    shape=(3,),
                ),
            )
        if self.scale_gate_pivot is not None and (
            not np.isfinite(self.scale_gate_pivot)
            or self.scale_gate_pivot < 0.0
        ):
            raise CGMMContractError(
                "scale_gate_pivot must be finite and non-negative"
            )
        if not self.evidence_months:
            raise CGMMContractError("evidence_months cannot be empty")
        require_sha256(
            self.evidence_fingerprint,
            field_name="evidence_fingerprint",
        )
        require_sha256(self.fingerprint, field_name="fingerprint")
        if self.fingerprint != fingerprint_payload(
            self.to_dict(include_fingerprint=False)
        ):
            raise CGMMContractError(
                "correction fingerprint does not match its payload"
            )

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "config": self.config.to_dict(),
            "reference_month_ordinal": self.reference_month_ordinal,
            "block_log_intercepts": self.block_log_intercepts.tolist(),
            "block_monthly_log_slopes": self.block_monthly_log_slopes.tolist(),
            "scale_gate_pivot": self.scale_gate_pivot,
            "evidence_months": list(self.evidence_months),
            "evidence_fingerprint": self.evidence_fingerprint,
        }
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CGMMCorrectionState":
        if not isinstance(payload, Mapping):
            raise TypeError("correction payload must be a mapping")
        expected_keys = {
            "config",
            "reference_month_ordinal",
            "block_log_intercepts",
            "block_monthly_log_slopes",
            "scale_gate_pivot",
            "evidence_months",
            "evidence_fingerprint",
            "fingerprint",
        }
        if set(payload) != expected_keys:
            raise CGMMContractError("correction payload has an invalid schema")
        return cls(
            config=CGMMCorrectionConfig(**dict(payload["config"])),
            reference_month_ordinal=int(payload["reference_month_ordinal"]),
            block_log_intercepts=np.asarray(
                payload["block_log_intercepts"],
                dtype=np.float64,
            ),
            block_monthly_log_slopes=np.asarray(
                payload["block_monthly_log_slopes"],
                dtype=np.float64,
            ),
            scale_gate_pivot=(
                None
                if payload["scale_gate_pivot"] is None
                else float(payload["scale_gate_pivot"])
            ),
            evidence_months=tuple(payload["evidence_months"]),
            evidence_fingerprint=str(payload["evidence_fingerprint"]),
            fingerprint=str(payload["fingerprint"]),
        )


@dataclass(frozen=True, slots=True)
class CGMMArtifactReceipt:
    """Verified paths and identities produced by artifact publication."""

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
    "CGMM_ARTIFACT_ID",
    "CGMM_ARTIFACT_VERSION",
    "CGMM_MODEL_ID",
    "CGMM_MODEL_KEY",
    "CGMM_PREPROCESSING_ID",
    "CGMMArtifactReceipt",
    "CGMMContractError",
    "CGMMCorrectionState",
    "CGMMPreparedBatch",
    "CGMMPrediction",
    "CGMMPreprocessingState",
    "CGMMRollingEvidence",
]

"""Final chronological validation for selected lifecycle configurations."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from typing import Any, Iterable

import numpy as np

from modeling_module._internal.lifecycle_runtime import (
    ConditionalGaussianMixtureForecaster,
    LTB_FORECAST_MONTHS,
    LifecycleSample,
    LifecycleSamplePurpose,
    SimilarLifecycleForecaster,
    build_cgmm_rolling_evidence,
    build_similar_lifecycle_rolling_evidence,
    fit_cgmm_correction,
    fit_similar_lifecycle_correction,
    require_sha256,
)
from modeling_module.api.lifecycle_selection import (
    CGMMSelectionCandidate,
    LifecycleSelectionMetrics,
    SimilarLifecycleSelectionCandidate,
    _split_inner_rolling_samples,
    lifecycle_selection_metrics,
)


VALIDATION_COMPARISON_CONTRACT_ID = (
    "lifecycle-selected-configuration-validation-v1"
)
_HORIZON_BLOCKS = (
    ("months_1_12", 0, 12),
    ("months_13_36", 12, 36),
    ("months_37_72", 36, 72),
)


class LifecycleValidationComparisonError(ValueError):
    """Raised when final validation would violate its frozen boundary."""


@dataclass(frozen=True, slots=True)
class LifecycleHorizonMetrics:
    """Point metrics for one contiguous forecast-horizon block."""

    label: str
    start_month: int
    end_month: int
    mae: float
    wape: float
    smape: float
    normalized_bias: float

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise LifecycleValidationComparisonError(
                "horizon label must be non-empty text"
            )
        if not 1 <= self.start_month <= self.end_month <= LTB_FORECAST_MONTHS:
            raise LifecycleValidationComparisonError(
                "horizon months must be inside the 72-month forecast"
            )
        values = np.asarray(
            (self.mae, self.wape, self.smape, self.normalized_bias),
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or (values[:3] < 0.0).any():
            raise LifecycleValidationComparisonError(
                "horizon metrics must be finite with non-negative errors"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LifecycleIntervalMetrics:
    """Coverage, width, and interval score for one prediction interval."""

    nominal_coverage: float
    empirical_coverage: float
    absolute_coverage_error: float
    mean_width: float
    normalized_mean_width: float
    mean_interval_score: float

    def __post_init__(self) -> None:
        values = np.asarray(tuple(asdict(self).values()), dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise LifecycleValidationComparisonError(
                "interval metrics must be finite and non-negative"
            )
        if not 0.0 < self.nominal_coverage < 1.0:
            raise LifecycleValidationComparisonError(
                "nominal_coverage must be between zero and one"
            )
        if not 0.0 <= self.empirical_coverage <= 1.0:
            raise LifecycleValidationComparisonError(
                "empirical_coverage must be between zero and one"
            )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LifecycleHorizonEvaluation:
    """Point and interval evidence for one horizon block."""

    point: LifecycleHorizonMetrics
    interval: LifecycleIntervalMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": self.point.to_dict(),
            "interval": self.interval.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class LifecycleCohortEvaluation:
    """Validation evidence for one lifecycle-start cohort."""

    lifecycle_start_month: str
    sample_count: int
    point: LifecycleSelectionMetrics
    interval: LifecycleIntervalMetrics

    def __post_init__(self) -> None:
        try:
            parsed = date.fromisoformat(self.lifecycle_start_month)
        except (TypeError, ValueError) as exc:
            raise LifecycleValidationComparisonError(
                "lifecycle_start_month must be an ISO date"
            ) from exc
        if parsed.day != 1:
            raise LifecycleValidationComparisonError(
                "lifecycle_start_month must be a month start"
            )
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count <= 0
        ):
            raise LifecycleValidationComparisonError(
                "cohort sample_count must be positive"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lifecycle_start_month": self.lifecycle_start_month,
            "sample_count": self.sample_count,
            "point": self.point.to_dict(),
            "interval": self.interval.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class LifecycleValidationRun:
    """One selected model fitted on all Train samples and scored once."""

    model_key: str
    candidate_name: str
    random_seed: int | None
    validation_sample_count: int
    fit_dataset_fingerprint: str
    model_fingerprint: str
    preprocessing_fingerprint: str
    correction_fingerprint: str
    correction_evidence_fingerprint: str
    correction_evidence_months: tuple[str, ...]
    prediction_sha256: str
    point: LifecycleSelectionMetrics
    interval: LifecycleIntervalMetrics
    horizons: tuple[LifecycleHorizonEvaluation, ...]
    cohorts: tuple[LifecycleCohortEvaluation, ...]

    def __post_init__(self) -> None:
        if self.model_key not in {"cgmm", "similar_lifecycle"}:
            raise LifecycleValidationComparisonError(
                "validation run has an unsupported model_key"
            )
        if not isinstance(self.candidate_name, str) or not self.candidate_name:
            raise LifecycleValidationComparisonError(
                "candidate_name must be non-empty text"
            )
        if (
            isinstance(self.validation_sample_count, bool)
            or not isinstance(self.validation_sample_count, int)
            or self.validation_sample_count <= 0
        ):
            raise LifecycleValidationComparisonError(
                "validation_sample_count must be positive"
            )
        for field_name in (
            "fit_dataset_fingerprint",
            "model_fingerprint",
            "preprocessing_fingerprint",
            "correction_fingerprint",
            "correction_evidence_fingerprint",
            "prediction_sha256",
        ):
            require_sha256(getattr(self, field_name), field_name=field_name)
        if not self.correction_evidence_months:
            raise LifecycleValidationComparisonError(
                "correction_evidence_months cannot be empty"
            )
        if len(self.horizons) != len(_HORIZON_BLOCKS):
            raise LifecycleValidationComparisonError(
                "validation run must report all horizon blocks"
            )
        if sum(item.sample_count for item in self.cohorts) != (
            self.validation_sample_count
        ):
            raise LifecycleValidationComparisonError(
                "cohort counts must equal validation_sample_count"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "candidate_name": self.candidate_name,
            "random_seed": self.random_seed,
            "validation_sample_count": self.validation_sample_count,
            "fit_dataset_fingerprint": self.fit_dataset_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "preprocessing_fingerprint": self.preprocessing_fingerprint,
            "correction_fingerprint": self.correction_fingerprint,
            "correction_evidence_fingerprint": (
                self.correction_evidence_fingerprint
            ),
            "correction_evidence_months": list(
                self.correction_evidence_months
            ),
            "prediction_sha256": self.prediction_sha256,
            "point": self.point.to_dict(),
            "interval": self.interval.to_dict(),
            "horizons": [item.to_dict() for item in self.horizons],
            "cohorts": [item.to_dict() for item in self.cohorts],
        }


@dataclass(frozen=True, slots=True)
class LifecycleValidationModelSummary:
    """Seed-aggregated final validation result for one model family."""

    model_key: str
    candidate_name: str
    run_count: int
    point_mean: LifecycleSelectionMetrics
    wape_std: float
    normalized_bias_std: float
    interval_mean: LifecycleIntervalMetrics
    empirical_coverage_std: float
    mean_interval_score_std: float
    horizons_mean: tuple[LifecycleHorizonEvaluation, ...]

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.wape_std,
                self.normalized_bias_std,
                self.empirical_coverage_std,
                self.mean_interval_score_std,
            ),
            dtype=np.float64,
        )
        if (
            isinstance(self.run_count, bool)
            or not isinstance(self.run_count, int)
            or self.run_count <= 0
            or not np.isfinite(values).all()
            or (values < 0.0).any()
        ):
            raise LifecycleValidationComparisonError(
                "model summary spread values must be finite and non-negative"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "candidate_name": self.candidate_name,
            "run_count": self.run_count,
            "point_mean": self.point_mean.to_dict(),
            "wape_std": self.wape_std,
            "normalized_bias_std": self.normalized_bias_std,
            "interval_mean": self.interval_mean.to_dict(),
            "empirical_coverage_std": self.empirical_coverage_std,
            "mean_interval_score_std": self.mean_interval_score_std,
            "horizons_mean": [
                item.to_dict() for item in self.horizons_mean
            ],
        }


@dataclass(frozen=True, slots=True)
class LifecycleValidationComparisonRequest:
    """Frozen selected settings and one chronological Validation split."""

    training_samples: tuple[LifecycleSample, ...]
    validation_samples: tuple[LifecycleSample, ...]
    dataset_fingerprint: str
    selected_cgmm_candidate: CGMMSelectionCandidate
    selected_similar_lifecycle_candidate: (
        SimilarLifecycleSelectionCandidate
    )
    cgmm_random_seeds: tuple[int, ...] = (11, 22, 33)
    correction_rolling_fraction: float = 0.2

    def __post_init__(self) -> None:
        training = _completed_samples(
            self.training_samples,
            field_name="training_samples",
        )
        validation = _completed_samples(
            self.validation_samples,
            field_name="validation_samples",
        )
        object.__setattr__(self, "training_samples", training)
        object.__setattr__(self, "validation_samples", validation)
        require_sha256(
            self.dataset_fingerprint,
            field_name="dataset_fingerprint",
        )
        if not isinstance(
            self.selected_cgmm_candidate,
            CGMMSelectionCandidate,
        ):
            raise TypeError(
                "selected_cgmm_candidate must be CGMMSelectionCandidate"
            )
        if not isinstance(
            self.selected_similar_lifecycle_candidate,
            SimilarLifecycleSelectionCandidate,
        ):
            raise TypeError(
                "selected_similar_lifecycle_candidate must be "
                "SimilarLifecycleSelectionCandidate"
            )
        overlap = {sample.sample_id for sample in training}.intersection(
            sample.sample_id for sample in validation
        )
        if overlap:
            raise LifecycleValidationComparisonError(
                "Train and Validation sample IDs must be disjoint"
            )
        if training[0].feature_schema != validation[0].feature_schema:
            raise LifecycleValidationComparisonError(
                "Train and Validation feature schemas must match"
            )
        if max(sample.lifecycle_start_month for sample in training) >= min(
            sample.lifecycle_start_month for sample in validation
        ):
            raise LifecycleValidationComparisonError(
                "Validation cohorts must start after all Train cohorts"
            )
        if (
            not self.cgmm_random_seeds
            or len(set(self.cgmm_random_seeds))
            != len(self.cgmm_random_seeds)
            or any(
                isinstance(seed, bool) or not isinstance(seed, int)
                for seed in self.cgmm_random_seeds
            )
        ):
            raise LifecycleValidationComparisonError(
                "cgmm_random_seeds must contain unique integers"
            )
        fraction = self.correction_rolling_fraction
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(float(fraction))
            or not 0.0 < float(fraction) < 1.0
        ):
            raise LifecycleValidationComparisonError(
                "correction_rolling_fraction must be between zero and one"
            )


@dataclass(frozen=True, slots=True)
class LifecycleValidationPredictionBundle:
    """Raw Validation curves retained in memory for diagnostics and plots."""

    sample_ids: tuple[str, ...]
    lifecycle_start_months: tuple[str, ...]
    cgmm_random_seeds: tuple[int, ...]
    actual: np.ndarray
    cgmm_seed_mean_forecast: np.ndarray
    cgmm_seed_mean_lower_bound: np.ndarray
    cgmm_seed_mean_upper_bound: np.ndarray
    similar_lifecycle_forecast: np.ndarray
    similar_lifecycle_lower_bound: np.ndarray
    similar_lifecycle_upper_bound: np.ndarray

    def __post_init__(self) -> None:
        sample_count = len(self.sample_ids)
        if sample_count == 0 or len(set(self.sample_ids)) != sample_count:
            raise LifecycleValidationComparisonError(
                "prediction bundle sample_ids must be non-empty and unique"
            )
        if len(self.lifecycle_start_months) != sample_count:
            raise LifecycleValidationComparisonError(
                "prediction bundle lifecycle months must match sample_ids"
            )
        for value in self.lifecycle_start_months:
            try:
                parsed = date.fromisoformat(value)
            except (TypeError, ValueError) as exc:
                raise LifecycleValidationComparisonError(
                    "prediction bundle lifecycle months must be ISO dates"
                ) from exc
            if parsed.day != 1:
                raise LifecycleValidationComparisonError(
                    "prediction bundle lifecycle months must be month starts"
                )
        if not self.cgmm_random_seeds:
            raise LifecycleValidationComparisonError(
                "prediction bundle must retain CGMM seed identities"
            )
        expected_shape = (sample_count, LTB_FORECAST_MONTHS)
        for field_name in (
            "actual",
            "cgmm_seed_mean_forecast",
            "cgmm_seed_mean_lower_bound",
            "cgmm_seed_mean_upper_bound",
            "similar_lifecycle_forecast",
            "similar_lifecycle_lower_bound",
            "similar_lifecycle_upper_bound",
        ):
            array = np.asarray(getattr(self, field_name), dtype=np.float64)
            if (
                array.shape != expected_shape
                or not np.isfinite(array).all()
                or (array < 0.0).any()
            ):
                raise LifecycleValidationComparisonError(
                    f"{field_name} must be finite non-negative shape "
                    f"{expected_shape}"
                )
            frozen = np.ascontiguousarray(array)
            frozen.setflags(write=False)
            object.__setattr__(self, field_name, frozen)
        if (
            self.cgmm_seed_mean_lower_bound
            > self.cgmm_seed_mean_upper_bound
        ).any():
            raise LifecycleValidationComparisonError(
                "CGMM plot bounds must be ordered"
            )
        if (
            self.similar_lifecycle_lower_bound
            > self.similar_lifecycle_upper_bound
        ).any():
            raise LifecycleValidationComparisonError(
                "Similar Lifecycle plot bounds must be ordered"
            )


@dataclass(frozen=True, slots=True)
class LifecycleValidationComparisonResult:
    """Final point and interval comparison for selected lifecycle models."""

    training_sample_count: int
    validation_sample_count: int
    training_end_month: str
    validation_start_month: str
    validation_end_month: str
    correction_initial_fit_sample_count: int
    correction_rolling_sample_count: int
    correction_initial_fit_end_month: str
    correction_rolling_start_month: str
    training_sample_ids_sha256: str
    validation_sample_ids_sha256: str
    validation_targets_sha256: str
    selected_cgmm_candidate: CGMMSelectionCandidate
    selected_similar_lifecycle_candidate: SimilarLifecycleSelectionCandidate
    cgmm_runs: tuple[LifecycleValidationRun, ...]
    cgmm_summary: LifecycleValidationModelSummary
    similar_lifecycle_run: LifecycleValidationRun
    similar_lifecycle_summary: LifecycleValidationModelSummary
    point_wape_winner: str
    interval_coverage_winner: str
    cgmm_minus_similar_wape: float
    cgmm_minus_similar_absolute_coverage_error: float
    predictions: LifecycleValidationPredictionBundle = field(
        repr=False,
        compare=False,
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_contract_id": VALIDATION_COMPARISON_CONTRACT_ID,
            "evaluation_boundary": {
                "training_sample_count": self.training_sample_count,
                "validation_sample_count": self.validation_sample_count,
                "training_end_month": self.training_end_month,
                "validation_start_month": self.validation_start_month,
                "validation_end_month": self.validation_end_month,
                "training_sample_ids_sha256": (
                    self.training_sample_ids_sha256
                ),
                "validation_sample_ids_sha256": (
                    self.validation_sample_ids_sha256
                ),
                "validation_targets_sha256": (
                    self.validation_targets_sha256
                ),
                "validation_passed_to_fit": False,
                "validation_passed_to_correction": False,
            },
            "correction_boundary": {
                "initial_fit_sample_count": (
                    self.correction_initial_fit_sample_count
                ),
                "rolling_sample_count": self.correction_rolling_sample_count,
                "initial_fit_end_month": (
                    self.correction_initial_fit_end_month
                ),
                "rolling_start_month": self.correction_rolling_start_month,
                "source": "training_samples_only",
            },
            "conditional_gmm": {
                "selected_candidate": self.selected_cgmm_candidate.to_dict(),
                "summary": self.cgmm_summary.to_dict(),
                "runs": [item.to_dict() for item in self.cgmm_runs],
            },
            "similar_lifecycle": {
                "selected_candidate": (
                    self.selected_similar_lifecycle_candidate.to_dict()
                ),
                "summary": self.similar_lifecycle_summary.to_dict(),
                "runs": [self.similar_lifecycle_run.to_dict()],
            },
            "comparison": {
                "point_wape_winner": self.point_wape_winner,
                "interval_coverage_winner": (
                    self.interval_coverage_winner
                ),
                "cgmm_minus_similar_wape": (
                    self.cgmm_minus_similar_wape
                ),
                "cgmm_minus_similar_absolute_coverage_error": (
                    self.cgmm_minus_similar_absolute_coverage_error
                ),
            },
        }


def _completed_samples(
    values: Iterable[LifecycleSample],
    *,
    field_name: str,
) -> tuple[LifecycleSample, ...]:
    samples = tuple(values)
    if not samples:
        raise LifecycleValidationComparisonError(
            f"{field_name} cannot be empty"
        )
    if any(not isinstance(sample, LifecycleSample) for sample in samples):
        raise TypeError(f"{field_name} must contain LifecycleSample values")
    if any(
        sample.purpose is not LifecycleSamplePurpose.TRAINING
        or sample.future_target is None
        for sample in samples
    ):
        raise LifecycleValidationComparisonError(
            f"{field_name} must contain completed lifecycle samples"
        )
    sample_ids = tuple(sample.sample_id for sample in samples)
    if len(set(sample_ids)) != len(sample_ids):
        raise LifecycleValidationComparisonError(
            f"{field_name} sample IDs must be unique"
        )
    schema = samples[0].feature_schema
    if any(sample.feature_schema != schema for sample in samples[1:]):
        raise LifecycleValidationComparisonError(
            f"{field_name} must share one ordered feature schema"
        )
    return tuple(
        sorted(
            samples,
            key=lambda item: (item.lifecycle_start_month, item.sample_id),
        )
    )


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sample_ids_sha256(samples: tuple[LifecycleSample, ...]) -> str:
    return hashlib.sha256(
        "\n".join(sample.sample_id for sample in samples).encode("utf-8")
    ).hexdigest()


def _validation_targets_sha256(
    samples: tuple[LifecycleSample, ...],
) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.sample_id.encode("utf-8"))
        digest.update(b"\0")
        target = np.ascontiguousarray(sample.future_target, dtype="<f8")
        digest.update(target.tobytes())
    return digest.hexdigest()


def _fit_dataset_fingerprint(
    source_fingerprint: str,
    samples: tuple[LifecycleSample, ...],
) -> str:
    return _canonical_sha256(
        {
            "source_dataset_fingerprint": source_fingerprint,
            "purpose": "lifecycle-selected-model-full-train-fit",
            "training_sample_ids": [sample.sample_id for sample in samples],
        }
    )


def _prediction_sha256(prediction: Any) -> str:
    digest = hashlib.sha256()
    for sample_id in prediction.sample_ids:
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\0")
    for array in (
        prediction.mean_forecast,
        prediction.forecast_std,
        prediction.lower_bound,
        prediction.upper_bound,
    ):
        contiguous = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _nominal_coverage(interval_z: float) -> float:
    return math.erf(float(interval_z) / math.sqrt(2.0))


def lifecycle_interval_metrics(
    actual: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    *,
    nominal_coverage: float,
) -> LifecycleIntervalMetrics:
    """Evaluate empirical coverage, width, and the proper interval score."""

    actual_array = np.asarray(actual, dtype=np.float64)
    lower = np.asarray(lower_bound, dtype=np.float64)
    upper = np.asarray(upper_bound, dtype=np.float64)
    if (
        actual_array.ndim != 2
        or actual_array.size == 0
        or lower.shape != actual_array.shape
        or upper.shape != actual_array.shape
    ):
        raise LifecycleValidationComparisonError(
            "interval arrays must share one non-empty two-dimensional shape"
        )
    if (
        not np.isfinite(actual_array).all()
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or (actual_array < 0.0).any()
        or (lower < 0.0).any()
        or (lower > upper).any()
    ):
        raise LifecycleValidationComparisonError(
            "interval arrays must be finite, ordered, and non-negative"
        )
    if (
        isinstance(nominal_coverage, bool)
        or not isinstance(nominal_coverage, (int, float))
        or not math.isfinite(float(nominal_coverage))
        or not 0.0 < float(nominal_coverage) < 1.0
    ):
        raise LifecycleValidationComparisonError(
            "nominal_coverage must be between zero and one"
        )
    resolved_coverage = float(nominal_coverage)
    alpha = 1.0 - resolved_coverage
    width = upper - lower
    covered = (actual_array >= lower) & (actual_array <= upper)
    score = (
        width
        + (2.0 / alpha)
        * np.maximum(lower - actual_array, 0.0)
        + (2.0 / alpha)
        * np.maximum(actual_array - upper, 0.0)
    )
    empirical = float(covered.mean())
    return LifecycleIntervalMetrics(
        nominal_coverage=resolved_coverage,
        empirical_coverage=empirical,
        absolute_coverage_error=abs(empirical - resolved_coverage),
        mean_width=float(width.mean()),
        normalized_mean_width=float(width.sum())
        / max(float(actual_array.sum()), 1e-12),
        mean_interval_score=float(score.mean()),
    )


def _horizon_point_metrics(
    actual: np.ndarray,
    forecast: np.ndarray,
    *,
    label: str,
    start: int,
    stop: int,
) -> LifecycleHorizonMetrics:
    actual_slice = actual[:, start:stop]
    forecast_slice = forecast[:, start:stop]
    error = forecast_slice - actual_slice
    denominator = np.abs(actual_slice) + np.abs(forecast_slice)
    return LifecycleHorizonMetrics(
        label=label,
        start_month=start + 1,
        end_month=stop,
        mae=float(np.abs(error).mean()),
        wape=float(np.abs(error).sum())
        / max(float(actual_slice.sum()), 1e-12),
        smape=float(
            np.mean(
                np.divide(
                    2.0 * np.abs(error),
                    denominator,
                    out=np.zeros_like(error),
                    where=denominator > 0.0,
                )
            )
        ),
        normalized_bias=float(error.sum())
        / max(float(actual_slice.sum()), 1e-12),
    )


def _evaluate_prediction(
    *,
    model_key: str,
    candidate_name: str,
    random_seed: int | None,
    validation_samples: tuple[LifecycleSample, ...],
    prediction: Any,
    fit_dataset_fingerprint: str,
    correction_state: Any,
    interval_z: float,
) -> LifecycleValidationRun:
    sample_ids = tuple(sample.sample_id for sample in validation_samples)
    if prediction.sample_ids != sample_ids:
        raise LifecycleValidationComparisonError(
            "prediction sample order does not match Validation"
        )
    actual = np.asarray(
        [sample.future_target for sample in validation_samples],
        dtype=np.float64,
    )
    nominal = _nominal_coverage(interval_z)
    horizons = tuple(
        LifecycleHorizonEvaluation(
            point=_horizon_point_metrics(
                actual,
                prediction.mean_forecast,
                label=label,
                start=start,
                stop=stop,
            ),
            interval=lifecycle_interval_metrics(
                actual[:, start:stop],
                prediction.lower_bound[:, start:stop],
                prediction.upper_bound[:, start:stop],
                nominal_coverage=nominal,
            ),
        )
        for label, start, stop in _HORIZON_BLOCKS
    )
    cohorts: list[LifecycleCohortEvaluation] = []
    months = np.asarray(
        [sample.lifecycle_start_month for sample in validation_samples],
        dtype=object,
    )
    for month in sorted(set(months.tolist())):
        indices = np.flatnonzero(months == month)
        cohorts.append(
            LifecycleCohortEvaluation(
                lifecycle_start_month=month.isoformat(),
                sample_count=len(indices),
                point=lifecycle_selection_metrics(
                    actual[indices],
                    prediction.mean_forecast[indices],
                ),
                interval=lifecycle_interval_metrics(
                    actual[indices],
                    prediction.lower_bound[indices],
                    prediction.upper_bound[indices],
                    nominal_coverage=nominal,
                ),
            )
        )
    return LifecycleValidationRun(
        model_key=model_key,
        candidate_name=candidate_name,
        random_seed=random_seed,
        validation_sample_count=len(validation_samples),
        fit_dataset_fingerprint=fit_dataset_fingerprint,
        model_fingerprint=prediction.model_fingerprint,
        preprocessing_fingerprint=prediction.preprocessing_fingerprint,
        correction_fingerprint=correction_state.fingerprint,
        correction_evidence_fingerprint=(
            correction_state.evidence_fingerprint
        ),
        correction_evidence_months=correction_state.evidence_months,
        prediction_sha256=_prediction_sha256(prediction),
        point=lifecycle_selection_metrics(
            actual,
            prediction.mean_forecast,
        ),
        interval=lifecycle_interval_metrics(
            actual,
            prediction.lower_bound,
            prediction.upper_bound,
            nominal_coverage=nominal,
        ),
        horizons=horizons,
        cohorts=tuple(cohorts),
    )


def _mean_dataclass(
    values: tuple[Any, ...],
    data_type: type[Any],
) -> Any:
    fields = asdict(values[0])
    return data_type(
        **{
            name: float(np.mean([getattr(value, name) for value in values]))
            for name in fields
        }
    )


def _mean_horizons(
    runs: tuple[LifecycleValidationRun, ...],
) -> tuple[LifecycleHorizonEvaluation, ...]:
    summaries: list[LifecycleHorizonEvaluation] = []
    for index, (label, start, stop) in enumerate(_HORIZON_BLOCKS):
        points = tuple(run.horizons[index].point for run in runs)
        intervals = tuple(run.horizons[index].interval for run in runs)
        summaries.append(
            LifecycleHorizonEvaluation(
                point=LifecycleHorizonMetrics(
                    label=label,
                    start_month=start + 1,
                    end_month=stop,
                    mae=float(np.mean([item.mae for item in points])),
                    wape=float(np.mean([item.wape for item in points])),
                    smape=float(np.mean([item.smape for item in points])),
                    normalized_bias=float(
                        np.mean([item.normalized_bias for item in points])
                    ),
                ),
                interval=_mean_dataclass(
                    intervals,
                    LifecycleIntervalMetrics,
                ),
            )
        )
    return tuple(summaries)


def _summarize_runs(
    runs: tuple[LifecycleValidationRun, ...],
) -> LifecycleValidationModelSummary:
    points = tuple(run.point for run in runs)
    intervals = tuple(run.interval for run in runs)
    return LifecycleValidationModelSummary(
        model_key=runs[0].model_key,
        candidate_name=runs[0].candidate_name,
        run_count=len(runs),
        point_mean=_mean_dataclass(points, LifecycleSelectionMetrics),
        wape_std=float(np.std([item.wape for item in points])),
        normalized_bias_std=float(
            np.std([item.normalized_bias for item in points])
        ),
        interval_mean=_mean_dataclass(
            intervals,
            LifecycleIntervalMetrics,
        ),
        empirical_coverage_std=float(
            np.std([item.empirical_coverage for item in intervals])
        ),
        mean_interval_score_std=float(
            np.std([item.mean_interval_score for item in intervals])
        ),
        horizons_mean=_mean_horizons(runs),
    )


def _point_winner(
    cgmm: LifecycleValidationModelSummary,
    similar: LifecycleValidationModelSummary,
) -> str:
    left = cgmm.point_mean.wape
    right = similar.point_mean.wape
    if math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
        return "tie"
    return cgmm.model_key if left < right else similar.model_key


def _interval_winner(
    cgmm: LifecycleValidationModelSummary,
    similar: LifecycleValidationModelSummary,
) -> str:
    left = (
        cgmm.interval_mean.absolute_coverage_error,
        cgmm.interval_mean.mean_interval_score,
    )
    right = (
        similar.interval_mean.absolute_coverage_error,
        similar.interval_mean.mean_interval_score,
    )
    if all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)
        for a, b in zip(left, right, strict=True)
    ):
        return "tie"
    return cgmm.model_key if left < right else similar.model_key


def evaluate_lifecycle_validation_comparison(
    request: LifecycleValidationComparisonRequest,
) -> LifecycleValidationComparisonResult:
    """Fit selected settings on all Train rows and score Validation once."""

    if not isinstance(request, LifecycleValidationComparisonRequest):
        raise TypeError(
            "request must be LifecycleValidationComparisonRequest"
        )
    initial, rolling = _split_inner_rolling_samples(
        request.training_samples,
        validation_fraction=float(request.correction_rolling_fraction),
    )
    fit_fingerprint = _fit_dataset_fingerprint(
        request.dataset_fingerprint,
        request.training_samples,
    )

    cgmm_runs: list[LifecycleValidationRun] = []
    cgmm_predictions: list[Any] = []
    cgmm_candidate = request.selected_cgmm_candidate
    for seed in request.cgmm_random_seeds:
        model_config = replace(
            cgmm_candidate.model_config,
            random_seed=seed,
        )
        evidence = build_cgmm_rolling_evidence(
            initial,
            rolling,
            dataset_fingerprint=request.dataset_fingerprint,
            model_config=model_config,
            preprocessing_config=cgmm_candidate.preprocessing_config,
        )
        correction_state = fit_cgmm_correction(
            evidence,
            cgmm_candidate.correction_config,
        )
        model = ConditionalGaussianMixtureForecaster(
            model_config,
            preprocessing_config=cgmm_candidate.preprocessing_config,
        ).fit(
            request.training_samples,
            dataset_fingerprint=fit_fingerprint,
        )
        prediction = model.attach_correction(correction_state).predict(
            request.validation_samples
        )
        cgmm_predictions.append(prediction)
        cgmm_runs.append(
            _evaluate_prediction(
                model_key="cgmm",
                candidate_name=cgmm_candidate.name,
                random_seed=seed,
                validation_samples=request.validation_samples,
                prediction=prediction,
                fit_dataset_fingerprint=fit_fingerprint,
                correction_state=correction_state,
                interval_z=model_config.interval_z,
            )
        )

    similar_candidate = request.selected_similar_lifecycle_candidate
    similar_evidence = build_similar_lifecycle_rolling_evidence(
        initial,
        rolling,
        dataset_fingerprint=request.dataset_fingerprint,
        model_config=similar_candidate.model_config,
        preprocessing_config=similar_candidate.preprocessing_config,
    )
    similar_correction_state = fit_similar_lifecycle_correction(
        similar_evidence,
        similar_candidate.correction_config,
    )
    similar_model = SimilarLifecycleForecaster(
        similar_candidate.model_config,
        preprocessing_config=similar_candidate.preprocessing_config,
    ).fit(
        request.training_samples,
        dataset_fingerprint=fit_fingerprint,
    )
    similar_prediction = similar_model.attach_correction(
        similar_correction_state
    ).predict(request.validation_samples)
    similar_run = _evaluate_prediction(
        model_key="similar_lifecycle",
        candidate_name=similar_candidate.name,
        random_seed=None,
        validation_samples=request.validation_samples,
        prediction=similar_prediction,
        fit_dataset_fingerprint=fit_fingerprint,
        correction_state=similar_correction_state,
        interval_z=similar_candidate.model_config.interval_z,
    )

    cgmm_run_tuple = tuple(cgmm_runs)
    cgmm_summary = _summarize_runs(cgmm_run_tuple)
    similar_summary = _summarize_runs((similar_run,))
    return LifecycleValidationComparisonResult(
        training_sample_count=len(request.training_samples),
        validation_sample_count=len(request.validation_samples),
        training_end_month=max(
            sample.lifecycle_start_month for sample in request.training_samples
        ).isoformat(),
        validation_start_month=min(
            sample.lifecycle_start_month
            for sample in request.validation_samples
        ).isoformat(),
        validation_end_month=max(
            sample.lifecycle_start_month
            for sample in request.validation_samples
        ).isoformat(),
        correction_initial_fit_sample_count=len(initial),
        correction_rolling_sample_count=len(rolling),
        correction_initial_fit_end_month=max(
            sample.lifecycle_start_month for sample in initial
        ).isoformat(),
        correction_rolling_start_month=min(
            sample.lifecycle_start_month for sample in rolling
        ).isoformat(),
        training_sample_ids_sha256=_sample_ids_sha256(
            request.training_samples
        ),
        validation_sample_ids_sha256=_sample_ids_sha256(
            request.validation_samples
        ),
        validation_targets_sha256=_validation_targets_sha256(
            request.validation_samples
        ),
        selected_cgmm_candidate=cgmm_candidate,
        selected_similar_lifecycle_candidate=similar_candidate,
        cgmm_runs=cgmm_run_tuple,
        cgmm_summary=cgmm_summary,
        similar_lifecycle_run=similar_run,
        similar_lifecycle_summary=similar_summary,
        point_wape_winner=_point_winner(cgmm_summary, similar_summary),
        interval_coverage_winner=_interval_winner(
            cgmm_summary,
            similar_summary,
        ),
        cgmm_minus_similar_wape=(
            cgmm_summary.point_mean.wape
            - similar_summary.point_mean.wape
        ),
        cgmm_minus_similar_absolute_coverage_error=(
            cgmm_summary.interval_mean.absolute_coverage_error
            - similar_summary.interval_mean.absolute_coverage_error
        ),
        predictions=LifecycleValidationPredictionBundle(
            sample_ids=tuple(
                sample.sample_id for sample in request.validation_samples
            ),
            lifecycle_start_months=tuple(
                sample.lifecycle_start_month.isoformat()
                for sample in request.validation_samples
            ),
            cgmm_random_seeds=request.cgmm_random_seeds,
            actual=np.asarray(
                [
                    sample.future_target
                    for sample in request.validation_samples
                ],
                dtype=np.float64,
            ),
            cgmm_seed_mean_forecast=np.mean(
                [item.mean_forecast for item in cgmm_predictions],
                axis=0,
            ),
            cgmm_seed_mean_lower_bound=np.mean(
                [item.lower_bound for item in cgmm_predictions],
                axis=0,
            ),
            cgmm_seed_mean_upper_bound=np.mean(
                [item.upper_bound for item in cgmm_predictions],
                axis=0,
            ),
            similar_lifecycle_forecast=(
                similar_prediction.mean_forecast
            ),
            similar_lifecycle_lower_bound=similar_prediction.lower_bound,
            similar_lifecycle_upper_bound=similar_prediction.upper_bound,
        ),
    )


__all__ = [
    "VALIDATION_COMPARISON_CONTRACT_ID",
    "LifecycleCohortEvaluation",
    "LifecycleHorizonEvaluation",
    "LifecycleHorizonMetrics",
    "LifecycleIntervalMetrics",
    "LifecycleValidationComparisonError",
    "LifecycleValidationComparisonRequest",
    "LifecycleValidationComparisonResult",
    "LifecycleValidationModelSummary",
    "LifecycleValidationPredictionBundle",
    "LifecycleValidationRun",
    "evaluate_lifecycle_validation_comparison",
    "lifecycle_interval_metrics",
]

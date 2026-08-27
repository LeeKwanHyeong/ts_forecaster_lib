"""Train-only temporal configuration selection for lifecycle forecasters."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from typing import Any, Callable, Iterable

import numpy as np

from modeling_module._internal.lifecycle_runtime import (
    CGMMConfig,
    CGMMCorrectionConfig,
    CGMMPreprocessingConfig,
    LTB_FORECAST_MONTHS,
    LifecycleSample,
    LifecycleSamplePurpose,
    SimilarLifecycleConfig,
    SimilarLifecycleCorrectionConfig,
    SimilarLifecyclePreprocessingConfig,
    apply_cgmm_correction,
    apply_similar_lifecycle_correction,
    build_cgmm_rolling_evidence,
    build_similar_lifecycle_rolling_evidence,
    fit_cgmm_correction,
    fit_similar_lifecycle_correction,
    require_sha256,
)


class LifecycleSelectionError(ValueError):
    """Raised when train-only model selection cannot preserve chronology."""


@dataclass(frozen=True, slots=True)
class LifecycleSelectionMetrics:
    """Point metrics used consistently across lifecycle model families."""

    mae: float
    wape: float
    smape: float
    normalized_bias: float
    mae_months_1_12: float
    mae_months_13_36: float
    mae_months_37_72: float
    wape_months_1_12: float
    wape_months_13_36: float
    wape_months_37_72: float

    def __post_init__(self) -> None:
        values = np.asarray(tuple(asdict(self).values()), dtype=np.float64)
        if not np.isfinite(values).all():
            raise LifecycleSelectionError("selection metrics must be finite")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CGMMSelectionCandidate:
    """One capacity, normalization, and correction combination."""

    name: str
    model_config: CGMMConfig = field(default_factory=CGMMConfig)
    preprocessing_config: CGMMPreprocessingConfig = field(
        default_factory=CGMMPreprocessingConfig
    )
    correction_config: CGMMCorrectionConfig = field(
        default_factory=CGMMCorrectionConfig
    )

    def __post_init__(self) -> None:
        _require_candidate_name(self.name)
        if not isinstance(self.model_config, CGMMConfig):
            raise TypeError("model_config must be CGMMConfig")
        if not isinstance(self.preprocessing_config, CGMMPreprocessingConfig):
            raise TypeError(
                "preprocessing_config must be CGMMPreprocessingConfig"
            )
        if not isinstance(self.correction_config, CGMMCorrectionConfig):
            raise TypeError("correction_config must be CGMMCorrectionConfig")

    def to_dict(self) -> dict[str, Any]:
        model_config = self.model_config.to_dict()
        model_config.pop("random_seed")
        return {
            "name": self.name,
            "model_config": model_config,
            "preprocessing_config": self.preprocessing_config.to_dict(),
            "correction_config": self.correction_config.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SimilarLifecycleSelectionCandidate:
    """One retrieval, normalization, and correction combination."""

    name: str
    model_config: SimilarLifecycleConfig = field(
        default_factory=SimilarLifecycleConfig
    )
    preprocessing_config: SimilarLifecyclePreprocessingConfig = field(
        default_factory=lambda: SimilarLifecyclePreprocessingConfig(
            feature_profile="static_observed_v1"
        )
    )
    correction_config: SimilarLifecycleCorrectionConfig = field(
        default_factory=SimilarLifecycleCorrectionConfig
    )

    def __post_init__(self) -> None:
        _require_candidate_name(self.name)
        if not isinstance(self.model_config, SimilarLifecycleConfig):
            raise TypeError("model_config must be SimilarLifecycleConfig")
        if not isinstance(
            self.preprocessing_config,
            SimilarLifecyclePreprocessingConfig,
        ):
            raise TypeError(
                "preprocessing_config must be "
                "SimilarLifecyclePreprocessingConfig"
            )
        if self.preprocessing_config.feature_profile not in {
            "static_observed_v1",
            "static_observed_m0_v1",
        }:
            raise LifecycleSelectionError(
                "Similar Lifecycle selection requires a static observed "
                "preprocessing profile"
            )
        if not isinstance(
            self.correction_config,
            SimilarLifecycleCorrectionConfig,
        ):
            raise TypeError(
                "correction_config must be "
                "SimilarLifecycleCorrectionConfig"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_config": self.model_config.to_dict(),
            "preprocessing_config": self.preprocessing_config.to_dict(),
            "correction_config": self.correction_config.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class LifecycleSelectionRun:
    """One candidate and seed evaluated on forward-only inner folds."""

    model_key: str
    candidate_name: str
    random_seed: int | None
    fold_months: tuple[str, ...]
    evaluation_sample_count: int
    metrics: LifecycleSelectionMetrics
    fold_wape_mean: float
    fold_wape_std: float
    fold_wape_max: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "candidate_name": self.candidate_name,
            "random_seed": self.random_seed,
            "fold_months": list(self.fold_months),
            "evaluation_sample_count": self.evaluation_sample_count,
            "metrics": self.metrics.to_dict(),
            "fold_wape_mean": self.fold_wape_mean,
            "fold_wape_std": self.fold_wape_std,
            "fold_wape_max": self.fold_wape_max,
        }


@dataclass(frozen=True, slots=True)
class LifecycleSelectionSummary:
    """Seed-aggregated evidence used for deterministic selection."""

    model_key: str
    candidate_name: str
    run_count: int
    wape_mean: float
    wape_std: float
    mae_mean: float
    smape_mean: float
    normalized_bias_mean: float
    maximum_fold_wape: float
    complexity: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LifecycleRollingSelectionRequest:
    """Selection request that accepts completed Train samples only."""

    training_samples: tuple[LifecycleSample, ...]
    dataset_fingerprint: str
    cgmm_candidates: tuple[CGMMSelectionCandidate, ...]
    similar_lifecycle_candidates: tuple[
        SimilarLifecycleSelectionCandidate, ...
    ]
    cgmm_random_seeds: tuple[int, ...] = (11, 22, 33)
    rolling_validation_fraction: float = 0.2

    def __post_init__(self) -> None:
        samples = _completed_samples(self.training_samples)
        object.__setattr__(self, "training_samples", samples)
        require_sha256(
            self.dataset_fingerprint,
            field_name="dataset_fingerprint",
        )
        if any(
            not isinstance(item, CGMMSelectionCandidate)
            for item in self.cgmm_candidates
        ):
            raise TypeError(
                "cgmm_candidates must contain CGMMSelectionCandidate values"
            )
        if any(
            not isinstance(item, SimilarLifecycleSelectionCandidate)
            for item in self.similar_lifecycle_candidates
        ):
            raise TypeError(
                "similar_lifecycle_candidates must contain "
                "SimilarLifecycleSelectionCandidate values"
            )
        _require_unique_candidates(
            self.cgmm_candidates,
            field_name="cgmm_candidates",
        )
        _require_unique_candidates(
            self.similar_lifecycle_candidates,
            field_name="similar_lifecycle_candidates",
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
            raise LifecycleSelectionError(
                "cgmm_random_seeds must contain unique integers"
            )
        fraction = self.rolling_validation_fraction
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(float(fraction))
            or not 0.0 < float(fraction) < 1.0
        ):
            raise LifecycleSelectionError(
                "rolling_validation_fraction must be between zero and one"
            )


@dataclass(frozen=True, slots=True)
class LifecycleRollingSelectionResult:
    """Complete Train-only selection evidence for both model families."""

    training_sample_count: int
    initial_fit_sample_count: int
    rolling_sample_count: int
    initial_fit_end_month: str
    rolling_start_month: str
    rolling_end_month: str
    training_sample_ids_sha256: str
    selected_cgmm_candidate: CGMMSelectionCandidate
    selected_similar_lifecycle_candidate: SimilarLifecycleSelectionCandidate
    cgmm_runs: tuple[LifecycleSelectionRun, ...]
    cgmm_summaries: tuple[LifecycleSelectionSummary, ...]
    similar_lifecycle_runs: tuple[LifecycleSelectionRun, ...]
    similar_lifecycle_summaries: tuple[LifecycleSelectionSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_contract_id": "lifecycle-train-only-rolling-selection-v1",
            "training_boundary": {
                "training_sample_count": self.training_sample_count,
                "initial_fit_sample_count": self.initial_fit_sample_count,
                "rolling_sample_count": self.rolling_sample_count,
                "initial_fit_end_month": self.initial_fit_end_month,
                "rolling_start_month": self.rolling_start_month,
                "rolling_end_month": self.rolling_end_month,
                "training_sample_ids_sha256": (
                    self.training_sample_ids_sha256
                ),
                "external_validation_accepted": False,
            },
            "conditional_gmm": {
                "selected_candidate": self.selected_cgmm_candidate.to_dict(),
                "summaries": [item.to_dict() for item in self.cgmm_summaries],
                "runs": [item.to_dict() for item in self.cgmm_runs],
            },
            "similar_lifecycle": {
                "selected_candidate": (
                    self.selected_similar_lifecycle_candidate.to_dict()
                ),
                "summaries": [
                    item.to_dict()
                    for item in self.similar_lifecycle_summaries
                ],
                "runs": [
                    item.to_dict() for item in self.similar_lifecycle_runs
                ],
            },
        }


def _require_candidate_name(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise LifecycleSelectionError("candidate name must be non-empty text")


def _require_unique_candidates(
    values: tuple[Any, ...],
    *,
    field_name: str,
) -> None:
    if not isinstance(values, tuple) or not values:
        raise LifecycleSelectionError(f"{field_name} must be a non-empty tuple")
    names = [value.name for value in values]
    if len(set(names)) != len(names):
        raise LifecycleSelectionError(
            f"{field_name} must use unique candidate names"
        )


def _completed_samples(
    values: Iterable[LifecycleSample],
) -> tuple[LifecycleSample, ...]:
    samples = tuple(values)
    if not samples:
        raise LifecycleSelectionError("training_samples cannot be empty")
    if any(not isinstance(sample, LifecycleSample) for sample in samples):
        raise TypeError("training_samples must contain LifecycleSample values")
    if any(
        sample.purpose is not LifecycleSamplePurpose.TRAINING
        or sample.future_target is None
        for sample in samples
    ):
        raise LifecycleSelectionError(
            "selection accepts completed training samples only"
        )
    ids = tuple(sample.sample_id for sample in samples)
    if len(set(ids)) != len(ids):
        raise LifecycleSelectionError("training sample IDs must be unique")
    schema = samples[0].feature_schema
    if any(sample.feature_schema != schema for sample in samples[1:]):
        raise LifecycleSelectionError(
            "training samples must share one ordered feature schema"
        )
    return tuple(
        sorted(samples, key=lambda item: (item.lifecycle_start_month, item.sample_id))
    )


def _split_inner_rolling_samples(
    samples: tuple[LifecycleSample, ...],
    *,
    validation_fraction: float,
) -> tuple[tuple[LifecycleSample, ...], tuple[LifecycleSample, ...]]:
    counts: dict[date, int] = {}
    for sample in samples:
        counts[sample.lifecycle_start_month] = (
            counts.get(sample.lifecycle_start_month, 0) + 1
        )
    cohorts = tuple(sorted(counts))
    if len(cohorts) < 3:
        raise LifecycleSelectionError(
            "rolling selection requires at least three lifecycle cohorts"
        )
    target = len(samples) * (1.0 - validation_fraction)
    cumulative = 0
    candidates: list[tuple[float, date]] = []
    for cohort in cohorts[:-1]:
        cumulative += counts[cohort]
        candidates.append((abs(cumulative - target), cohort))
    cutoff = min(candidates)[1]
    initial = tuple(
        sample for sample in samples if sample.lifecycle_start_month <= cutoff
    )
    rolling = tuple(
        sample for sample in samples if sample.lifecycle_start_month > cutoff
    )
    if not initial or not rolling:
        raise LifecycleSelectionError(
            "rolling selection must leave non-empty fit and validation samples"
        )
    return initial, rolling


def lifecycle_selection_metrics(
    actual: np.ndarray,
    forecast: np.ndarray,
) -> LifecycleSelectionMetrics:
    """Evaluate one non-negative 72-month lifecycle point forecast."""

    actual_array = np.asarray(actual, dtype=np.float64)
    forecast_array = np.asarray(forecast, dtype=np.float64)
    if (
        actual_array.ndim != 2
        or actual_array.shape[1] != LTB_FORECAST_MONTHS
        or forecast_array.shape != actual_array.shape
        or actual_array.shape[0] == 0
    ):
        raise LifecycleSelectionError(
            "actual and forecast must share non-empty shape (N, 72)"
        )
    if (
        not np.isfinite(actual_array).all()
        or not np.isfinite(forecast_array).all()
        or (actual_array < 0.0).any()
        or (forecast_array < 0.0).any()
    ):
        raise LifecycleSelectionError(
            "selection arrays must be finite and non-negative"
        )
    error = forecast_array - actual_array

    def mae(start: int, stop: int) -> float:
        return float(np.abs(error[:, start:stop]).mean())

    def wape(start: int, stop: int) -> float:
        numerator = float(np.abs(error[:, start:stop]).sum())
        denominator = max(float(actual_array[:, start:stop].sum()), 1e-12)
        return numerator / denominator

    denominator = np.abs(actual_array) + np.abs(forecast_array)
    smape = float(
        np.mean(
            np.divide(
                2.0 * np.abs(error),
                denominator,
                out=np.zeros_like(error),
                where=denominator > 0.0,
            )
        )
    )
    return LifecycleSelectionMetrics(
        mae=mae(0, 72),
        wape=wape(0, 72),
        smape=smape,
        normalized_bias=float(error.sum())
        / max(float(actual_array.sum()), 1e-12),
        mae_months_1_12=mae(0, 12),
        mae_months_13_36=mae(12, 36),
        mae_months_37_72=mae(36, 72),
        wape_months_1_12=wape(0, 12),
        wape_months_13_36=wape(12, 36),
        wape_months_37_72=wape(36, 72),
    )


def _config_key(*payloads: dict[str, Any]) -> str:
    return json.dumps(
        payloads,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _evaluate_corrected_evidence(
    *,
    model_key: str,
    candidate_name: str,
    random_seed: int | None,
    evidence: tuple[Any, ...],
    samples_by_id: dict[str, LifecycleSample],
    correction_config: Any,
    common_start: int,
    fit_correction: Callable[[tuple[Any, ...], Any], Any],
    apply_correction: Callable[[Any, tuple[LifecycleSample, ...], Any], Any],
) -> LifecycleSelectionRun:
    if len(evidence) <= common_start:
        raise LifecycleSelectionError(
            "rolling evidence does not leave a correction evaluation cohort"
        )
    actual_blocks: list[np.ndarray] = []
    forecast_blocks: list[np.ndarray] = []
    fold_wape: list[float] = []
    fold_months: list[str] = []
    for index in range(common_start, len(evidence)):
        state = fit_correction(evidence[:index], correction_config)
        fold = evidence[index]
        fold_samples = tuple(samples_by_id[value] for value in fold.sample_ids)
        corrected = apply_correction(fold.prediction, fold_samples, state)
        metrics = lifecycle_selection_metrics(
            fold.actual,
            corrected.mean_forecast,
        )
        actual_blocks.append(fold.actual)
        forecast_blocks.append(corrected.mean_forecast)
        fold_wape.append(metrics.wape)
        fold_months.append(fold.validation_month.isoformat())
    actual = np.concatenate(actual_blocks, axis=0)
    forecast = np.concatenate(forecast_blocks, axis=0)
    fold_values = np.asarray(fold_wape, dtype=np.float64)
    return LifecycleSelectionRun(
        model_key=model_key,
        candidate_name=candidate_name,
        random_seed=random_seed,
        fold_months=tuple(fold_months),
        evaluation_sample_count=actual.shape[0],
        metrics=lifecycle_selection_metrics(actual, forecast),
        fold_wape_mean=float(fold_values.mean()),
        fold_wape_std=float(fold_values.std()),
        fold_wape_max=float(fold_values.max()),
    )


def _summaries(
    model_key: str,
    runs: tuple[LifecycleSelectionRun, ...],
    *,
    complexity_by_name: dict[str, int],
) -> tuple[LifecycleSelectionSummary, ...]:
    grouped: dict[str, list[LifecycleSelectionRun]] = {}
    for run in runs:
        grouped.setdefault(run.candidate_name, []).append(run)
    return tuple(
        LifecycleSelectionSummary(
            model_key=model_key,
            candidate_name=name,
            run_count=len(group),
            wape_mean=float(np.mean([item.metrics.wape for item in group])),
            wape_std=float(np.std([item.metrics.wape for item in group])),
            mae_mean=float(np.mean([item.metrics.mae for item in group])),
            smape_mean=float(np.mean([item.metrics.smape for item in group])),
            normalized_bias_mean=float(
                np.mean([item.metrics.normalized_bias for item in group])
            ),
            maximum_fold_wape=max(item.fold_wape_max for item in group),
            complexity=complexity_by_name[name],
        )
        for name, group in sorted(grouped.items())
    )


def _select_summary(
    summaries: tuple[LifecycleSelectionSummary, ...],
) -> LifecycleSelectionSummary:
    return min(
        summaries,
        key=lambda item: (
            item.wape_mean,
            item.wape_std,
            abs(item.normalized_bias_mean),
            item.maximum_fold_wape,
            item.complexity,
            item.candidate_name,
        ),
    )


def select_lifecycle_model_configurations(
    request: LifecycleRollingSelectionRequest,
) -> LifecycleRollingSelectionResult:
    """Select CGMM and Similar Lifecycle settings without outer validation."""

    if not isinstance(request, LifecycleRollingSelectionRequest):
        raise TypeError("request must be LifecycleRollingSelectionRequest")
    initial, rolling = _split_inner_rolling_samples(
        request.training_samples,
        validation_fraction=float(request.rolling_validation_fraction),
    )
    samples_by_id = {sample.sample_id: sample for sample in rolling}
    dataset_fingerprint = require_sha256(
        request.dataset_fingerprint,
        field_name="dataset_fingerprint",
    )

    cgmm_common_start = max(
        item.correction_config.minimum_calibration_cohorts
        for item in request.cgmm_candidates
    )
    cgmm_cache: dict[tuple[str, int], tuple[Any, ...]] = {}
    cgmm_runs: list[LifecycleSelectionRun] = []
    for candidate in request.cgmm_candidates:
        model_payload = candidate.model_config.to_dict()
        model_payload.pop("random_seed")
        cache_name = _config_key(
            model_payload,
            candidate.preprocessing_config.to_dict(),
        )
        for seed in request.cgmm_random_seeds:
            cache_key = (cache_name, seed)
            evidence = cgmm_cache.get(cache_key)
            if evidence is None:
                evidence = build_cgmm_rolling_evidence(
                    initial,
                    rolling,
                    dataset_fingerprint=dataset_fingerprint,
                    model_config=replace(
                        candidate.model_config,
                        random_seed=seed,
                    ),
                    preprocessing_config=candidate.preprocessing_config,
                )
                cgmm_cache[cache_key] = evidence
            cgmm_runs.append(
                _evaluate_corrected_evidence(
                    model_key="cgmm",
                    candidate_name=candidate.name,
                    random_seed=seed,
                    evidence=evidence,
                    samples_by_id=samples_by_id,
                    correction_config=candidate.correction_config,
                    common_start=cgmm_common_start,
                    fit_correction=fit_cgmm_correction,
                    apply_correction=apply_cgmm_correction,
                )
            )

    similar_common_start = max(
        item.correction_config.minimum_calibration_cohorts
        for item in request.similar_lifecycle_candidates
    )
    similar_cache: dict[str, tuple[Any, ...]] = {}
    similar_runs: list[LifecycleSelectionRun] = []
    for candidate in request.similar_lifecycle_candidates:
        cache_key = _config_key(
            candidate.model_config.to_dict(),
            candidate.preprocessing_config.to_dict(),
        )
        evidence = similar_cache.get(cache_key)
        if evidence is None:
            evidence = build_similar_lifecycle_rolling_evidence(
                initial,
                rolling,
                dataset_fingerprint=dataset_fingerprint,
                model_config=candidate.model_config,
                preprocessing_config=candidate.preprocessing_config,
            )
            similar_cache[cache_key] = evidence
        similar_runs.append(
            _evaluate_corrected_evidence(
                model_key="similar_lifecycle",
                candidate_name=candidate.name,
                random_seed=None,
                evidence=evidence,
                samples_by_id=samples_by_id,
                correction_config=candidate.correction_config,
                common_start=similar_common_start,
                fit_correction=fit_similar_lifecycle_correction,
                apply_correction=apply_similar_lifecycle_correction,
            )
        )

    cgmm_run_tuple = tuple(cgmm_runs)
    similar_run_tuple = tuple(similar_runs)
    cgmm_summaries = _summaries(
        "cgmm",
        cgmm_run_tuple,
        complexity_by_name={
            item.name: (
                item.model_config.component_count
                * item.model_config.target_component_count
            )
            for item in request.cgmm_candidates
        },
    )
    similar_summaries = _summaries(
        "similar_lifecycle",
        similar_run_tuple,
        complexity_by_name={
            item.name: item.model_config.neighbor_count
            for item in request.similar_lifecycle_candidates
        },
    )
    selected_cgmm_name = _select_summary(cgmm_summaries).candidate_name
    selected_similar_name = _select_summary(similar_summaries).candidate_name
    sample_ids_sha256 = hashlib.sha256(
        "\n".join(sample.sample_id for sample in request.training_samples).encode(
            "utf-8"
        )
    ).hexdigest()
    return LifecycleRollingSelectionResult(
        training_sample_count=len(request.training_samples),
        initial_fit_sample_count=len(initial),
        rolling_sample_count=len(rolling),
        initial_fit_end_month=max(
            sample.lifecycle_start_month for sample in initial
        ).isoformat(),
        rolling_start_month=min(
            sample.lifecycle_start_month for sample in rolling
        ).isoformat(),
        rolling_end_month=max(
            sample.lifecycle_start_month for sample in rolling
        ).isoformat(),
        training_sample_ids_sha256=sample_ids_sha256,
        selected_cgmm_candidate=next(
            item
            for item in request.cgmm_candidates
            if item.name == selected_cgmm_name
        ),
        selected_similar_lifecycle_candidate=next(
            item
            for item in request.similar_lifecycle_candidates
            if item.name == selected_similar_name
        ),
        cgmm_runs=cgmm_run_tuple,
        cgmm_summaries=cgmm_summaries,
        similar_lifecycle_runs=similar_run_tuple,
        similar_lifecycle_summaries=similar_summaries,
    )


__all__ = [
    "CGMMSelectionCandidate",
    "LifecycleRollingSelectionRequest",
    "LifecycleRollingSelectionResult",
    "LifecycleSelectionError",
    "LifecycleSelectionMetrics",
    "LifecycleSelectionRun",
    "LifecycleSelectionSummary",
    "SimilarLifecycleSelectionCandidate",
    "lifecycle_selection_metrics",
    "select_lifecycle_model_configurations",
]

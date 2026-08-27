"""Forward-only cohort and tail correction for Similar Lifecycle forecasts."""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Iterable

import numpy as np

from modeling_module.data_loader.lifecycle_contracts import (
    LifecycleSample,
    LifecycleSamplePurpose,
)
from modeling_module.models.CGMM.configs import (
    CGMMCorrectionConfig,
    CGMMPreprocessingConfig,
)
from modeling_module.models.CGMM.contracts import (
    CGMMCorrectionState,
    fingerprint_payload,
    require_sha256,
)
from modeling_module.models.CGMM.correction import (
    HORIZON_BLOCKS,
    cgmm_correction_factors,
)
from modeling_module.models.SimilarLifecycle.configs import (
    SimilarLifecycleConfig,
)
from modeling_module.models.SimilarLifecycle.contracts import (
    SIMILAR_LIFECYCLE_MODEL_ID,
    SIMILAR_LIFECYCLE_MODEL_KEY,
    SimilarLifecycleContractError,
    SimilarLifecyclePrediction,
    SimilarLifecycleRollingEvidence,
)


class SimilarLifecycleCorrectionError(SimilarLifecycleContractError):
    """Raised when correction would violate its chronological boundary."""


def _month_ordinal(value: date) -> int:
    return value.year * 12 + value.month


def _completed_samples(
    samples: Iterable[LifecycleSample],
    *,
    field_name: str,
) -> tuple[LifecycleSample, ...]:
    materialized = tuple(samples)
    if not materialized:
        raise SimilarLifecycleCorrectionError(f"{field_name} cannot be empty")
    if any(not isinstance(sample, LifecycleSample) for sample in materialized):
        raise TypeError(f"{field_name} must contain LifecycleSample values")
    if any(
        sample.purpose is not LifecycleSamplePurpose.TRAINING
        or sample.future_target is None
        for sample in materialized
    ):
        raise SimilarLifecycleCorrectionError(
            f"{field_name} must contain completed training samples"
        )
    sample_ids = tuple(sample.sample_id for sample in materialized)
    if len(set(sample_ids)) != len(sample_ids):
        raise SimilarLifecycleCorrectionError(
            f"{field_name} sample_ids must be unique"
        )
    schema = materialized[0].feature_schema
    if any(sample.feature_schema != schema for sample in materialized[1:]):
        raise SimilarLifecycleCorrectionError(
            f"{field_name} must use one ordered feature schema"
        )
    return materialized


def build_similar_lifecycle_rolling_evidence(
    train_samples: Iterable[LifecycleSample],
    validation_samples: Iterable[LifecycleSample],
    *,
    dataset_fingerprint: str,
    model_config: SimilarLifecycleConfig | None = None,
    preprocessing_config: CGMMPreprocessingConfig | None = None,
) -> tuple[SimilarLifecycleRollingEvidence, ...]:
    """Fit each validation cohort from train plus earlier cohorts only."""

    from modeling_module.models.SimilarLifecycle.model import (
        SimilarLifecycleForecaster,
    )

    train = _completed_samples(train_samples, field_name="train_samples")
    validation = _completed_samples(
        validation_samples,
        field_name="validation_samples",
    )
    if train[0].feature_schema != validation[0].feature_schema:
        raise SimilarLifecycleCorrectionError(
            "train and validation feature schemas must match"
        )
    if {sample.sample_id for sample in train}.intersection(
        sample.sample_id for sample in validation
    ):
        raise SimilarLifecycleCorrectionError(
            "train and validation samples must be disjoint"
        )
    dataset_fingerprint = require_sha256(
        dataset_fingerprint,
        field_name="dataset_fingerprint",
    )
    resolved_config = model_config or SimilarLifecycleConfig()
    repository = train
    evidence: list[SimilarLifecycleRollingEvidence] = []
    for validation_month in sorted(
        {sample.lifecycle_start_month for sample in validation}
    ):
        fold = tuple(
            sample
            for sample in validation
            if sample.lifecycle_start_month == validation_month
        )
        fold_fingerprint = fingerprint_payload(
            {
                "dataset_fingerprint": dataset_fingerprint,
                "purpose": "similar-lifecycle-forward-rolling-validation",
                "validation_month": validation_month.isoformat(),
                "repository_sample_ids": [
                    sample.sample_id for sample in repository
                ],
            }
        )
        model = SimilarLifecycleForecaster(
            resolved_config,
            preprocessing_config=preprocessing_config,
        ).fit(repository, dataset_fingerprint=fold_fingerprint)
        prediction = model.predict(fold, apply_correction=False)
        observed = np.asarray(
            [sample.observed_target for sample in fold],
            dtype=np.float64,
        )
        actual = np.asarray(
            [sample.future_target for sample in fold],
            dtype=np.float64,
        )
        evidence.append(
            SimilarLifecycleRollingEvidence(
                validation_month=validation_month,
                sample_ids=tuple(sample.sample_id for sample in fold),
                observed_scale=np.maximum(observed.mean(axis=1), 1.0),
                actual=actual,
                prediction=prediction,
            )
        )
        repository = (*repository, *fold)
    return tuple(evidence)


def _evidence_fingerprint(
    evidence: tuple[SimilarLifecycleRollingEvidence, ...],
) -> str:
    digest = hashlib.sha256()
    for fold in evidence:
        digest.update(fold.validation_month.isoformat().encode("ascii"))
        for sample_id in fold.sample_ids:
            digest.update(sample_id.encode("utf-8"))
            digest.update(b"\0")
        digest.update(fold.prediction.model_fingerprint.encode("ascii"))
        for array in (
            fold.observed_scale,
            fold.actual,
            fold.prediction.mean_forecast,
        ):
            contiguous = np.ascontiguousarray(array, dtype="<f8")
            digest.update(str(contiguous.shape).encode("ascii"))
            digest.update(contiguous.tobytes())
    return digest.hexdigest()


def fit_similar_lifecycle_correction(
    evidence: Iterable[SimilarLifecycleRollingEvidence],
    config: CGMMCorrectionConfig | None = None,
) -> CGMMCorrectionState:
    """Fit the shared blockwise cohort and long-tail correction policy."""

    resolved_config = config or CGMMCorrectionConfig()
    if not isinstance(resolved_config, CGMMCorrectionConfig):
        raise TypeError("config must be SimilarLifecycleCorrectionConfig")
    materialized = tuple(evidence)
    if len(materialized) < resolved_config.minimum_calibration_cohorts:
        raise SimilarLifecycleCorrectionError(
            "insufficient prior cohorts for Similar Lifecycle correction"
        )
    if any(
        not isinstance(item, SimilarLifecycleRollingEvidence)
        for item in materialized
    ):
        raise TypeError(
            "evidence must contain SimilarLifecycleRollingEvidence values"
        )
    months = tuple(item.validation_month for item in materialized)
    if tuple(sorted(months)) != months or len(set(months)) != len(months):
        raise SimilarLifecycleCorrectionError(
            "correction evidence must contain unique chronological cohorts"
        )
    month_ordinals = np.asarray(
        [_month_ordinal(value) for value in months],
        dtype=np.float64,
    )
    reference = int(month_ordinals[-1])
    centered_months = month_ordinals - reference
    sample_weights = np.asarray(
        [len(item.sample_ids) for item in materialized],
        dtype=np.float64,
    )
    sample_weights /= sample_weights.mean()
    design = np.column_stack(
        (np.ones(len(materialized), dtype=np.float64), centered_months)
    )
    weighted_design = design * np.sqrt(sample_weights)[:, None]

    intercepts: list[float] = []
    slopes: list[float] = []
    for start, stop in HORIZON_BLOCKS:
        ratios = np.asarray(
            [
                item.actual[:, start:stop].sum()
                / max(
                    item.prediction.mean_forecast[:, start:stop].sum(),
                    1e-12,
                )
                for item in materialized
            ],
            dtype=np.float64,
        )
        log_ratios = np.log(
            np.clip(
                ratios,
                resolved_config.correction_floor,
                resolved_config.correction_ceiling,
            )
        )
        coefficients = np.linalg.lstsq(
            weighted_design,
            log_ratios * np.sqrt(sample_weights),
            rcond=None,
        )[0]
        intercepts.append(float(coefficients[0]))
        slopes.append(
            float(
                np.clip(
                    coefficients[1],
                    -resolved_config.maximum_monthly_log_slope,
                    resolved_config.maximum_monthly_log_slope,
                )
            )
        )

    scale_gate_pivot = None
    if resolved_config.scale_gate_quantile is not None:
        scale_gate_pivot = float(
            np.quantile(
                np.concatenate(
                    [item.observed_scale for item in materialized],
                    axis=0,
                ),
                resolved_config.scale_gate_quantile,
            )
        )
    evidence_fingerprint = _evidence_fingerprint(materialized)
    payload = {
        "config": resolved_config.to_dict(),
        "reference_month_ordinal": reference,
        "block_log_intercepts": intercepts,
        "block_monthly_log_slopes": slopes,
        "scale_gate_pivot": scale_gate_pivot,
        "evidence_months": [value.isoformat() for value in months],
        "evidence_fingerprint": evidence_fingerprint,
    }
    return CGMMCorrectionState(
        config=resolved_config,
        reference_month_ordinal=reference,
        block_log_intercepts=np.asarray(intercepts, dtype=np.float64),
        block_monthly_log_slopes=np.asarray(slopes, dtype=np.float64),
        scale_gate_pivot=scale_gate_pivot,
        evidence_months=tuple(value.isoformat() for value in months),
        evidence_fingerprint=evidence_fingerprint,
        fingerprint=fingerprint_payload(payload),
    )


def similar_lifecycle_correction_factors(
    samples: Iterable[LifecycleSample],
    state: CGMMCorrectionState,
) -> np.ndarray:
    """Return the shared positive correction factor by sample and horizon."""

    return cgmm_correction_factors(samples, state)


def apply_similar_lifecycle_correction(
    prediction: SimilarLifecyclePrediction,
    samples: Iterable[LifecycleSample],
    state: CGMMCorrectionState,
) -> SimilarLifecyclePrediction:
    """Scale point, spread, and interval outputs by the same positive factor."""

    if not isinstance(prediction, SimilarLifecyclePrediction):
        raise TypeError("prediction must be SimilarLifecyclePrediction")
    materialized = tuple(samples)
    if tuple(sample.sample_id for sample in materialized) != prediction.sample_ids:
        raise SimilarLifecycleCorrectionError(
            "correction sample order does not match prediction"
        )
    factors = similar_lifecycle_correction_factors(materialized, state)
    return SimilarLifecyclePrediction(
        sample_ids=prediction.sample_ids,
        mean_forecast=prediction.mean_forecast * factors,
        forecast_std=prediction.forecast_std * factors,
        lower_bound=prediction.lower_bound * factors,
        upper_bound=prediction.upper_bound * factors,
        neighbor_sample_ids=prediction.neighbor_sample_ids,
        neighbor_weights=prediction.neighbor_weights,
        neighbor_distances=prediction.neighbor_distances,
        model_key=SIMILAR_LIFECYCLE_MODEL_KEY,
        model_id=SIMILAR_LIFECYCLE_MODEL_ID,
        model_fingerprint=prediction.model_fingerprint,
        preprocessing_fingerprint=prediction.preprocessing_fingerprint,
        correction_fingerprint=state.fingerprint,
    )


__all__ = [
    "SimilarLifecycleCorrectionError",
    "apply_similar_lifecycle_correction",
    "build_similar_lifecycle_rolling_evidence",
    "fit_similar_lifecycle_correction",
    "similar_lifecycle_correction_factors",
]

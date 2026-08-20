"""Forward-only rolling validation and distribution-preserving CGMM correction."""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Iterable

import numpy as np

from modeling_module.data_loader.lifecycle_contracts import (
    LTB_FORECAST_MONTHS,
    LifecycleSample,
    LifecycleSamplePurpose,
)
from modeling_module.models.CGMM.configs import (
    CGMMConfig,
    CGMMCorrectionConfig,
    CGMMPreprocessingConfig,
)
from modeling_module.models.CGMM.contracts import (
    CGMM_MODEL_KEY,
    CGMM_MODEL_ID,
    CGMMContractError,
    CGMMCorrectionState,
    CGMMPrediction,
    CGMMRollingEvidence,
    fingerprint_payload,
    freeze_float_array,
    require_sha256,
)


HORIZON_BLOCKS = ((0, 12), (12, 36), (36, 72))
HORIZON_BLOCK_CENTERS = np.asarray((5.5, 23.5, 53.5), dtype=np.float64)


class CGMMCorrectionError(CGMMContractError):
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
        raise CGMMCorrectionError(f"{field_name} cannot be empty")
    if any(not isinstance(sample, LifecycleSample) for sample in materialized):
        raise TypeError(f"{field_name} must contain LifecycleSample values")
    if any(
        sample.purpose is not LifecycleSamplePurpose.TRAINING
        or sample.future_target is None
        for sample in materialized
    ):
        raise CGMMCorrectionError(
            f"{field_name} must contain completed training samples"
        )
    ids = [sample.sample_id for sample in materialized]
    if len(set(ids)) != len(ids):
        raise CGMMCorrectionError(f"{field_name} sample_ids must be unique")
    schema = materialized[0].feature_schema
    if any(sample.feature_schema != schema for sample in materialized[1:]):
        raise CGMMCorrectionError(
            f"{field_name} must use one ordered feature schema"
        )
    return materialized


def build_cgmm_rolling_evidence(
    train_samples: Iterable[LifecycleSample],
    validation_samples: Iterable[LifecycleSample],
    *,
    dataset_fingerprint: str,
    model_config: CGMMConfig | None = None,
    preprocessing_config: CGMMPreprocessingConfig | None = None,
) -> tuple[CGMMRollingEvidence, ...]:
    """Fit every validation cohort from train plus earlier cohorts only."""

    from modeling_module.models.CGMM.model import (
        ConditionalGaussianMixtureForecaster,
    )

    train = _completed_samples(train_samples, field_name="train_samples")
    validation = _completed_samples(
        validation_samples,
        field_name="validation_samples",
    )
    if train[0].feature_schema != validation[0].feature_schema:
        raise CGMMCorrectionError(
            "train and validation feature schemas must match"
        )
    overlap = {sample.sample_id for sample in train}.intersection(
        sample.sample_id for sample in validation
    )
    if overlap:
        raise CGMMCorrectionError("train and validation samples must be disjoint")
    dataset_fingerprint = require_sha256(
        dataset_fingerprint,
        field_name="dataset_fingerprint",
    )
    resolved_config = model_config or CGMMConfig()
    repository = train
    evidence: list[CGMMRollingEvidence] = []
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
                "purpose": "cgmm-forward-rolling-validation",
                "validation_month": validation_month.isoformat(),
                "repository_sample_ids": [
                    sample.sample_id for sample in repository
                ],
            }
        )
        model = ConditionalGaussianMixtureForecaster(
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
            CGMMRollingEvidence(
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
    evidence: tuple[CGMMRollingEvidence, ...],
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


def fit_cgmm_correction(
    evidence: Iterable[CGMMRollingEvidence],
    config: CGMMCorrectionConfig | None = None,
) -> CGMMCorrectionState:
    """Fit blockwise temporal correction from prior rolling evidence."""

    resolved_config = config or CGMMCorrectionConfig()
    if not isinstance(resolved_config, CGMMCorrectionConfig):
        raise TypeError("config must be CGMMCorrectionConfig")
    materialized = tuple(evidence)
    if len(materialized) < resolved_config.minimum_calibration_cohorts:
        raise CGMMCorrectionError(
            "insufficient prior cohorts for CGMM correction"
        )
    if any(not isinstance(item, CGMMRollingEvidence) for item in materialized):
        raise TypeError("evidence must contain CGMMRollingEvidence values")
    months = tuple(item.validation_month for item in materialized)
    if tuple(sorted(months)) != months or len(set(months)) != len(months):
        raise CGMMCorrectionError(
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


def cgmm_correction_factors(
    samples: Iterable[LifecycleSample],
    state: CGMMCorrectionState,
) -> np.ndarray:
    """Return one positive correction factor for each sample and horizon."""

    materialized = tuple(samples)
    if not materialized or any(
        not isinstance(sample, LifecycleSample) for sample in materialized
    ):
        raise TypeError("samples must contain LifecycleSample values")
    if not isinstance(state, CGMMCorrectionState):
        raise TypeError("state must be CGMMCorrectionState")
    observed = np.asarray(
        [sample.observed_target for sample in materialized],
        dtype=np.float64,
    )
    observed_scale = np.maximum(observed.mean(axis=1), 1.0)
    query_months = np.asarray(
        [
            _month_ordinal(sample.lifecycle_start_month)
            for sample in materialized
        ],
        dtype=np.float64,
    )
    month_offset = query_months - state.reference_month_ordinal
    block_strengths = np.full(
        len(HORIZON_BLOCKS),
        state.config.cohort_strength,
        dtype=np.float64,
    )
    if state.config.short_horizon_cohort_strength is not None:
        block_strengths[0] = state.config.short_horizon_cohort_strength
    block_log_factors = block_strengths[None, :] * (
        state.block_log_intercepts[None, :]
        + month_offset[:, None] * state.block_monthly_log_slopes[None, :]
    )
    horizon = np.arange(LTB_FORECAST_MONTHS, dtype=np.float64)
    smooth_log_factor = np.stack(
        [
            np.interp(horizon, HORIZON_BLOCK_CENTERS, row)
            for row in block_log_factors
        ],
        axis=0,
    )
    if state.config.tail_half_life_months is not None:
        horizon_month = horizon + 1.0
        smooth_log_factor += (
            -np.log(2.0)
            * np.maximum(
                horizon_month - state.config.tail_start_month,
                0.0,
            )
            / state.config.tail_half_life_months
        )[None, :]
    if state.scale_gate_pivot is not None:
        gate = observed_scale / (observed_scale + state.scale_gate_pivot)
        smooth_log_factor *= gate[:, None]
    return freeze_float_array(
        np.clip(
            np.exp(smooth_log_factor),
            state.config.correction_floor,
            state.config.correction_ceiling,
        ),
        field_name="correction_factors",
        shape=(len(materialized), LTB_FORECAST_MONTHS),
        non_negative=True,
    )


def apply_cgmm_correction(
    prediction: CGMMPrediction,
    samples: Iterable[LifecycleSample],
    state: CGMMCorrectionState,
) -> CGMMPrediction:
    """Apply the same positive factor to all distribution-valued outputs."""

    if not isinstance(prediction, CGMMPrediction):
        raise TypeError("prediction must be CGMMPrediction")
    materialized = tuple(samples)
    sample_ids = tuple(sample.sample_id for sample in materialized)
    if sample_ids != prediction.sample_ids:
        raise CGMMCorrectionError(
            "correction sample order does not match prediction"
        )
    factors = cgmm_correction_factors(materialized, state)
    return CGMMPrediction(
        sample_ids=prediction.sample_ids,
        component_probabilities=prediction.component_probabilities,
        candidate_curves=prediction.candidate_curves * factors[:, None, :],
        mean_forecast=prediction.mean_forecast * factors,
        forecast_std=prediction.forecast_std * factors,
        lower_bound=prediction.lower_bound * factors,
        upper_bound=prediction.upper_bound * factors,
        model_key=CGMM_MODEL_KEY,
        model_id=CGMM_MODEL_ID,
        model_fingerprint=prediction.model_fingerprint,
        preprocessing_fingerprint=prediction.preprocessing_fingerprint,
        correction_fingerprint=state.fingerprint,
    )


__all__ = [
    "CGMMCorrectionError",
    "HORIZON_BLOCKS",
    "apply_cgmm_correction",
    "build_cgmm_rolling_evidence",
    "cgmm_correction_factors",
    "fit_cgmm_correction",
]

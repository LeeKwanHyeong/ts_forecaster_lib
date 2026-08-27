from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest

from modeling_module import (
    CGMMConfig,
    CGMMCorrectionConfig,
    CGMMCorrectionState,
    CGMMFitRequest,
    CGMMForecastRequest,
    CGMMPreprocessingConfig,
    CGMMRollingEvidence,
    LifecycleFeatureSchema,
    LifecycleSample,
    LifecycleSamplePurpose,
    add_calendar_months,
    build_cgmm_rolling_evidence,
    cgmm_correction_factors,
    fit_cgmm,
    fit_cgmm_correction,
    forecast_cgmm,
    load_cgmm_artifact,
    save_cgmm_artifact,
)
from modeling_module.models import build_model
from modeling_module.models.CGMM import CGMMArtifactError
from modeling_module.models.registry import get_model_spec, resolve_training_request_key


DATASET_FINGERPRINT = "a" * 64


def _schema() -> LifecycleFeatureSchema:
    return LifecycleFeatureSchema(
        static_cont_names=("defect_rate",),
        static_cat_names=("part_family",),
        observed_cont_names=("sales_qty",),
    )


def _sample(
    index: int,
    *,
    inference: bool = False,
    start: date | None = None,
    family: str | None = None,
) -> LifecycleSample:
    start = start or add_calendar_months(date(2010, 1, 1), index)
    cluster = index % 2
    observed = np.asarray(
        [
            (10.0 + index % 5) * np.exp(-month / (18.0 + 6.0 * cluster))
            + cluster * (month >= 7) * 2.0
            for month in range(12)
        ],
        dtype=np.float64,
    )
    future = np.asarray(
        [
            observed[-1]
            * np.exp(-month / (20.0 + 14.0 * cluster))
            + cluster * 1.5 * np.exp(-((month - 18.0) / 8.0) ** 2)
            for month in range(1, 73)
        ],
        dtype=np.float64,
    )
    purpose = (
        LifecycleSamplePurpose.INFERENCE
        if inference
        else LifecycleSamplePurpose.TRAINING
    )
    cutoff = add_calendar_months(start, 11 if inference else 83)
    return LifecycleSample(
        sample_id=f"sample-{index}",
        purpose=purpose,
        lifecycle_start_month=start,
        source_cutoff_month=cutoff,
        observed_target=tuple(observed.tolist()),
        future_target=None if inference else tuple(future.tolist()),
        feature_schema=_schema(),
        static_cont=(None if index % 7 == 0 else 0.01 * (index + 1),),
        static_cat=(family or f"family-{cluster}",),
        observed_cont=tuple(
            (float(120 - month * 3 + cluster * 5),)
            for month in range(12)
        ),
    )


def _config(seed: int = 7) -> CGMMConfig:
    return CGMMConfig(
        component_count=2,
        target_component_count=2,
        initialization_count=1,
        max_iterations=100,
        random_seed=seed,
    )


def _fit_model():
    request = CGMMFitRequest(
        samples=tuple(_sample(index) for index in range(24)),
        dataset_fingerprint=DATASET_FINGERPRINT,
        config=_config(),
    )
    return fit_cgmm(request)


def test_static_observed_profile_preserves_legacy_condition_layout() -> None:
    samples = tuple(_sample(index) for index in range(24))
    result = fit_cgmm(
        CGMMFitRequest(
            samples=samples,
            dataset_fingerprint=DATASET_FINGERPRINT,
            config=_config(),
            preprocessing=CGMMPreprocessingConfig(
                feature_profile="static_observed_v1"
            ),
        )
    )
    state = result.model.preprocessing_state
    observed_target = np.asarray(
        [sample.observed_target for sample in samples],
        dtype=np.float64,
    )
    quantity_scale = np.maximum(observed_target.mean(axis=1), 1.0)
    static = np.asarray(
        [
            [np.nan if value is None else value for value in sample.static_cont]
            for sample in samples
        ],
        dtype=np.float64,
    )
    static_fill = np.nanmedian(static, axis=0)
    static_missing = np.isnan(static)
    static_filled = np.where(static_missing, static_fill[None, :], static)
    observed_cont = np.asarray(
        [sample.observed_cont for sample in samples],
        dtype=np.float64,
    )
    families = np.asarray(
        [sample.static_cat[0] for sample in samples],
        dtype=np.str_,
    )
    vocabulary = tuple(sorted(set(families.tolist())))
    expected_blocks = [
        np.log1p(observed_target / quantity_scale[:, None]),
        np.log1p(quantity_scale)[:, None],
        np.log1p(static_filled),
        static_missing.astype(np.float64),
        np.log1p(observed_cont).reshape(len(samples), -1),
    ]
    expected_blocks.extend(
        (families == category).astype(np.float64)[:, None]
        for category in vocabulary[1:]
    )
    expected_raw = np.concatenate(expected_blocks, axis=1)
    expected_raw = np.concatenate(
        (
            expected_raw,
            (~np.isin(families, np.asarray(vocabulary))).astype(np.float64)[
                :, None
            ],
        ),
        axis=1,
    )

    assert state.feature_profile == "static_observed_v1"
    assert state.condition_feature_names == (
        *(f"observed_target_log_ratio_m{month:02d}" for month in range(12)),
        "log_quantity_scale",
        "static_log1p:defect_rate",
        "static_missing:defect_rate",
        *(f"observed_log1p:sales_qty:m{month:02d}" for month in range(12)),
        "category:part_family=family-1",
        "category:part_family=<UNK>",
    )
    np.testing.assert_allclose(state.condition_means, expected_raw.mean(axis=0))
    expected_scale = expected_raw.std(axis=0)
    expected_scale[expected_scale < 1e-8] = 1.0
    np.testing.assert_allclose(state.condition_scales, expected_scale)


def test_static_observed_m0_profile_adds_month_ordinal_and_restores_artifact(
    tmp_path,
) -> None:
    samples = tuple(_sample(index) for index in range(24))
    result = fit_cgmm(
        CGMMFitRequest(
            samples=samples,
            dataset_fingerprint=DATASET_FINGERPRINT,
            config=_config(),
            preprocessing=CGMMPreprocessingConfig(
                feature_profile="static_observed_m0_v1"
            ),
        )
    )
    state = result.model.preprocessing_state
    ordinal_index = state.condition_feature_names.index(
        "lifecycle_start_month_ordinal"
    )
    expected_ordinals = np.asarray(
        [
            sample.lifecycle_start_month.year * 12
            + sample.lifecycle_start_month.month
            - 1
            for sample in samples
        ],
        dtype=np.float64,
    )

    assert state.feature_profile == "static_observed_m0_v1"
    assert ordinal_index == 13
    assert state.condition_means[ordinal_index] == expected_ordinals.mean()
    assert state.condition_scales[ordinal_index] == expected_ordinals.std()

    inference = (
        _sample(130, inference=True, family="new-family"),
        _sample(131, inference=True, family="new-family"),
    )
    prepared = result.model.preprocessor.transform(inference)
    assert prepared.condition_matrix[0, ordinal_index] != (
        prepared.condition_matrix[1, ordinal_index]
    )
    expected = result.model.predict(inference)
    receipt = save_cgmm_artifact(result.model, tmp_path / "cgmm-m0")
    restored = load_cgmm_artifact(receipt.artifact_dir)
    actual = restored.predict(inference)

    assert restored.preprocessing_state == state
    for field_name in (
        "component_probabilities",
        "candidate_curves",
        "mean_forecast",
        "forecast_std",
        "lower_bound",
        "upper_bound",
    ):
        np.testing.assert_array_equal(
            getattr(actual, field_name),
            getattr(expected, field_name),
        )


def test_cgmm_public_fit_and_forecast_preserve_distribution_contract() -> None:
    result = _fit_model()
    inference = (
        _sample(100, inference=True, family="unseen-family"),
        _sample(101, inference=True),
    )

    prediction = forecast_cgmm(
        CGMMForecastRequest(model=result.model, samples=inference)
    )

    assert result.model_key == "cgmm"
    assert prediction.model_id == "modeling-module.cgmm.v1"
    assert prediction.component_probabilities.shape == (2, 2)
    assert prediction.candidate_curves.shape == (2, 2, 72)
    assert prediction.mean_forecast.shape == (2, 72)
    assert prediction.forecast_std.shape == (2, 72)
    assert prediction.lower_bound.shape == (2, 72)
    assert prediction.upper_bound.shape == (2, 72)
    response = prediction.to_dict()
    assert response["metadata"]["model_key"] == "cgmm"
    json.dumps(response, allow_nan=False)
    np.testing.assert_allclose(
        np.einsum(
            "nk,nkh->nh",
            prediction.component_probabilities,
            prediction.candidate_curves,
        ),
        prediction.mean_forecast,
    )
    assert "str:unseen-family" not in (
        result.model.preprocessing_state.categorical_vocabularies[0]
    )


def test_cgmm_registry_builds_model_but_keeps_generic_deep_learning_train_separate() -> None:
    model = build_model("cgmm", _config())
    spec = get_model_spec("lifecycle-cgmm")

    assert model.model_key == "cgmm"
    assert spec.family == "cgmm"
    assert spec.trainable is False
    assert spec.load_only is False
    assert spec.exogenous_policy == "lifecycle_conditional"
    with pytest.raises(ValueError, match="not trainable through the public training API"):
        resolve_training_request_key("cgmm")


def test_cgmm_correction_scales_candidates_and_all_distribution_outputs() -> None:
    result = _fit_model()
    evidence: list[CGMMRollingEvidence] = []
    for cohort_index in range(3):
        samples = tuple(
            _sample(cohort_index * 2 + offset)
            for offset in range(2)
        )
        raw = result.model.predict(samples, apply_correction=False)
        evidence.append(
            CGMMRollingEvidence(
                validation_month=add_calendar_months(
                    date(2020, 1, 1),
                    cohort_index,
                ),
                sample_ids=raw.sample_ids,
                observed_scale=np.asarray(
                    [np.mean(sample.observed_target) for sample in samples]
                ),
                actual=raw.mean_forecast * (0.80 - cohort_index * 0.05),
                prediction=raw,
            )
        )
    state = fit_cgmm_correction(
        evidence,
        CGMMCorrectionConfig(
            cohort_strength=0.5,
            tail_start_month=36,
            tail_half_life_months=72.0,
        ),
    )
    result.model.attach_correction(state)
    inference = (_sample(120, inference=True),)
    raw = result.model.predict(inference, apply_correction=False)
    corrected = result.model.predict(inference)
    factors = cgmm_correction_factors(inference, state)

    np.testing.assert_array_equal(
        corrected.component_probabilities,
        raw.component_probabilities,
    )
    np.testing.assert_allclose(
        corrected.candidate_curves,
        raw.candidate_curves * factors[:, None, :],
    )
    for field_name in (
        "mean_forecast",
        "forecast_std",
        "lower_bound",
        "upper_bound",
    ):
        np.testing.assert_allclose(
            getattr(corrected, field_name),
            getattr(raw, field_name) * factors,
        )
    assert corrected.correction_fingerprint == state.fingerprint


def test_short_horizon_strength_preserves_late_factors_and_legacy_payload() -> None:
    result = _fit_model()
    evidence: list[CGMMRollingEvidence] = []
    for cohort_index in range(3):
        samples = tuple(
            _sample(cohort_index * 2 + offset)
            for offset in range(2)
        )
        raw = result.model.predict(samples, apply_correction=False)
        evidence.append(
            CGMMRollingEvidence(
                validation_month=add_calendar_months(
                    date(2020, 1, 1),
                    cohort_index,
                ),
                sample_ids=raw.sample_ids,
                observed_scale=np.asarray(
                    [np.mean(sample.observed_target) for sample in samples]
                ),
                actual=raw.mean_forecast * (0.80 - cohort_index * 0.05),
                prediction=raw,
            )
        )
    baseline = fit_cgmm_correction(
        evidence,
        CGMMCorrectionConfig(
            cohort_strength=0.25,
            tail_start_month=36,
            tail_half_life_months=48.0,
        ),
    )
    strengthened = fit_cgmm_correction(
        evidence,
        CGMMCorrectionConfig(
            cohort_strength=0.25,
            short_horizon_cohort_strength=0.75,
            tail_start_month=36,
            tail_half_life_months=48.0,
        ),
    )
    inference = (_sample(120, inference=True),)
    baseline_factors = cgmm_correction_factors(inference, baseline)
    strengthened_factors = cgmm_correction_factors(inference, strengthened)

    assert "short_horizon_cohort_strength" not in (
        baseline.config.to_dict()
    )
    assert strengthened.config.to_dict()[
        "short_horizon_cohort_strength"
    ] == 0.75
    assert not np.array_equal(
        strengthened_factors[:, :12],
        baseline_factors[:, :12],
    )
    np.testing.assert_array_equal(
        strengthened_factors[:, 24:],
        baseline_factors[:, 24:],
    )
    assert CGMMCorrectionState.from_dict(strengthened.to_dict()).to_dict() == (
        strengthened.to_dict()
    )


def test_cgmm_artifact_strict_load_restores_identical_prediction(tmp_path) -> None:
    result = _fit_model()
    evidence: list[CGMMRollingEvidence] = []
    for cohort_index in range(3):
        samples = (_sample(cohort_index),)
        prediction = result.model.predict(samples, apply_correction=False)
        evidence.append(
            CGMMRollingEvidence(
                validation_month=add_calendar_months(date(2021, 1, 1), cohort_index),
                sample_ids=prediction.sample_ids,
                observed_scale=np.asarray(
                    [np.mean(samples[0].observed_target)]
                ),
                actual=prediction.mean_forecast * 0.8,
                prediction=prediction,
            )
        )
    correction = fit_cgmm_correction(
        evidence,
        CGMMCorrectionConfig(short_horizon_cohort_strength=0.75),
    )
    result.model.attach_correction(correction)
    inference = (_sample(130, inference=True, family="new-family"),)
    expected = result.model.predict(inference)

    receipt = save_cgmm_artifact(result.model, tmp_path / "cgmm")
    restored = load_cgmm_artifact(receipt.artifact_dir)
    actual = forecast_cgmm(
        CGMMForecastRequest(model=receipt.artifact_dir, samples=inference)
    )

    assert restored.model_fingerprint == result.model.model_fingerprint
    assert restored.preprocessing_state == result.model.preprocessing_state
    assert restored.correction_state is not None
    assert restored.correction_state.fingerprint == correction.fingerprint
    np.testing.assert_array_equal(
        restored.correction_state.block_log_intercepts,
        correction.block_log_intercepts,
    )
    for field_name in (
        "component_probabilities",
        "candidate_curves",
        "mean_forecast",
        "forecast_std",
        "lower_bound",
        "upper_bound",
    ):
        np.testing.assert_allclose(
            getattr(actual, field_name),
            getattr(expected, field_name),
            rtol=0.0,
            atol=0.0,
        )

    with receipt.arrays_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(CGMMArtifactError, match="SHA-256 mismatch"):
        load_cgmm_artifact(receipt.artifact_dir)


def test_cgmm_rolling_validation_uses_prior_completed_cohorts_only() -> None:
    train = tuple(_sample(index) for index in range(10))
    validation = tuple(
        _sample(
            200 + cohort * 2 + offset,
            start=add_calendar_months(date(2022, 1, 1), cohort),
        )
        for cohort in range(3)
        for offset in range(2)
    )

    evidence = build_cgmm_rolling_evidence(
        train,
        validation,
        dataset_fingerprint=DATASET_FINGERPRINT,
        model_config=_config(seed=11),
    )

    assert tuple(item.validation_month for item in evidence) == (
        date(2022, 1, 1),
        date(2022, 2, 1),
        date(2022, 3, 1),
    )
    assert all(item.prediction.correction_fingerprint is None for item in evidence)
    assert len({item.prediction.model_fingerprint for item in evidence}) == 3

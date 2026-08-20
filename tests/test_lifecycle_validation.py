from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from modeling_module import (
    CGMMConfig,
    CGMMCorrectionConfig,
    CGMMPreprocessingConfig,
    CGMMSelectionCandidate,
    LifecycleFeatureSchema,
    LifecycleSample,
    LifecycleSamplePurpose,
    LifecycleValidationComparisonError,
    LifecycleValidationComparisonRequest,
    SimilarLifecycleConfig,
    SimilarLifecycleCorrectionConfig,
    SimilarLifecycleSelectionCandidate,
    add_calendar_months,
    evaluate_lifecycle_validation_comparison,
    lifecycle_interval_metrics,
)


DATASET_FINGERPRINT = "b" * 64
SCHEMA = LifecycleFeatureSchema(static_cat_names=("family",))


def _sample(
    index: int,
    *,
    cohort_offset: int | None = None,
    future_scale: float = 1.0,
) -> LifecycleSample:
    cohort = index // 6 if cohort_offset is None else cohort_offset
    start = add_calendar_months(date(2018, 1, 1), cohort)
    scale = 8.0 + (index % 6) + cohort * 0.4
    observed = tuple(scale * (1.0 + month * 0.03) for month in range(12))
    future = tuple(
        future_scale
        * scale
        * (1.0 + min(month, 8) * 0.02)
        * (0.985**month)
        for month in range(72)
    )
    return LifecycleSample(
        sample_id=f"sample-{index:03d}",
        purpose=LifecycleSamplePurpose.TRAINING,
        lifecycle_start_month=start,
        source_cutoff_month=add_calendar_months(start, 83),
        observed_target=observed,
        future_target=future,
        feature_schema=SCHEMA,
        static_cat=("late-family" if cohort >= 8 else f"family-{index % 2}",),
    )


def _request(
    *,
    validation_scale: float = 1.0,
) -> LifecycleValidationComparisonRequest:
    correction = CGMMCorrectionConfig(
        name="tail-72",
        cohort_strength=0.0,
        tail_half_life_months=72.0,
        minimum_calibration_cohorts=2,
    )
    return LifecycleValidationComparisonRequest(
        training_samples=tuple(_sample(index) for index in range(48)),
        validation_samples=tuple(
            _sample(index, future_scale=validation_scale)
            for index in range(48, 60)
        ),
        dataset_fingerprint=DATASET_FINGERPRINT,
        selected_cgmm_candidate=CGMMSelectionCandidate(
            name="gmm1-pca1-tail-72",
            model_config=CGMMConfig(
                component_count=1,
                target_component_count=1,
                initialization_count=1,
                max_iterations=30,
            ),
            preprocessing_config=CGMMPreprocessingConfig(
                feature_profile="static_observed_v1"
            ),
            correction_config=correction,
        ),
        selected_similar_lifecycle_candidate=(
            SimilarLifecycleSelectionCandidate(
                name="all-k3-tail-72",
                model_config=SimilarLifecycleConfig(neighbor_count=3),
                preprocessing_config=CGMMPreprocessingConfig(
                    feature_profile="static_observed_v1"
                ),
                correction_config=SimilarLifecycleCorrectionConfig(
                    **correction.to_dict()
                ),
            )
        ),
        cgmm_random_seeds=(11, 22),
        correction_rolling_fraction=0.5,
    )


def test_interval_metrics_report_coverage_width_and_interval_score() -> None:
    metrics = lifecycle_interval_metrics(
        np.asarray([[1.0, 2.0]]),
        np.asarray([[0.0, 1.0]]),
        np.asarray([[2.0, 3.0]]),
        nominal_coverage=0.9,
    )

    assert metrics.empirical_coverage == 1.0
    assert metrics.absolute_coverage_error == pytest.approx(0.1)
    assert metrics.mean_width == 2.0
    assert metrics.normalized_mean_width == pytest.approx(4.0 / 3.0)
    assert metrics.mean_interval_score == 2.0


def test_selected_models_fit_all_train_and_score_validation_once() -> None:
    first = evaluate_lifecycle_validation_comparison(_request())
    second = evaluate_lifecycle_validation_comparison(_request())

    assert first.to_dict() == second.to_dict()
    assert first.training_sample_count == 48
    assert first.validation_sample_count == 12
    assert first.training_end_month == "2018-08-01"
    assert first.validation_start_month == "2018-09-01"
    assert first.validation_end_month == "2018-10-01"
    assert first.correction_initial_fit_sample_count == 24
    assert first.correction_rolling_sample_count == 24
    assert len(first.cgmm_runs) == 2
    assert first.cgmm_summary.run_count == 2
    assert first.similar_lifecycle_summary.run_count == 1
    assert [
        item.point.label for item in first.cgmm_summary.horizons_mean
    ] == ["months_1_12", "months_13_36", "months_37_72"]
    assert first.cgmm_summary.interval_mean.nominal_coverage == pytest.approx(
        0.9
    )
    assert 0.0 <= (
        first.cgmm_summary.interval_mean.empirical_coverage
    ) <= 1.0
    assert first.predictions.actual.shape == (12, 72)
    assert first.predictions.cgmm_seed_mean_forecast.shape == (12, 72)
    assert first.predictions.similar_lifecycle_forecast.shape == (12, 72)
    assert first.predictions.actual.flags.writeable is False
    assert first.predictions.cgmm_random_seeds == (11, 22)
    boundary = first.to_dict()["evaluation_boundary"]
    assert boundary["validation_passed_to_fit"] is False
    assert boundary["validation_passed_to_correction"] is False


def test_validation_targets_cannot_change_fit_or_prediction_state() -> None:
    baseline = evaluate_lifecycle_validation_comparison(_request())
    shifted = evaluate_lifecycle_validation_comparison(
        _request(validation_scale=1.4)
    )

    assert baseline.validation_targets_sha256 != (
        shifted.validation_targets_sha256
    )
    np.testing.assert_allclose(
        baseline.predictions.cgmm_seed_mean_forecast,
        shifted.predictions.cgmm_seed_mean_forecast,
    )
    np.testing.assert_allclose(
        baseline.predictions.similar_lifecycle_forecast,
        shifted.predictions.similar_lifecycle_forecast,
    )
    assert not np.array_equal(
        baseline.predictions.actual,
        shifted.predictions.actual,
    )
    for left, right in zip(
        baseline.cgmm_runs,
        shifted.cgmm_runs,
        strict=True,
    ):
        assert left.fit_dataset_fingerprint == right.fit_dataset_fingerprint
        assert left.model_fingerprint == right.model_fingerprint
        assert left.preprocessing_fingerprint == right.preprocessing_fingerprint
        assert left.correction_fingerprint == right.correction_fingerprint
        assert left.prediction_sha256 == right.prediction_sha256
    assert (
        baseline.similar_lifecycle_run.model_fingerprint
        == shifted.similar_lifecycle_run.model_fingerprint
    )
    assert (
        baseline.similar_lifecycle_run.correction_fingerprint
        == shifted.similar_lifecycle_run.correction_fingerprint
    )
    assert (
        baseline.similar_lifecycle_run.prediction_sha256
        == shifted.similar_lifecycle_run.prediction_sha256
    )
    assert baseline.cgmm_summary.point_mean.wape != (
        shifted.cgmm_summary.point_mean.wape
    )


def test_comparison_rejects_overlap_and_non_chronological_validation() -> None:
    request = _request()
    with pytest.raises(
        LifecycleValidationComparisonError,
        match="sample IDs must be disjoint",
    ):
        LifecycleValidationComparisonRequest(
            training_samples=request.training_samples,
            validation_samples=(
                request.training_samples[0],
                *request.validation_samples[1:],
            ),
            dataset_fingerprint=request.dataset_fingerprint,
            selected_cgmm_candidate=request.selected_cgmm_candidate,
            selected_similar_lifecycle_candidate=(
                request.selected_similar_lifecycle_candidate
            ),
        )

    early = replace(
        request.validation_samples[0],
        lifecycle_start_month=date(2018, 8, 1),
        source_cutoff_month=add_calendar_months(date(2018, 8, 1), 83),
    )
    with pytest.raises(
        LifecycleValidationComparisonError,
        match="must start after all Train cohorts",
    ):
        LifecycleValidationComparisonRequest(
            training_samples=request.training_samples,
            validation_samples=(early, *request.validation_samples[1:]),
            dataset_fingerprint=request.dataset_fingerprint,
            selected_cgmm_candidate=request.selected_cgmm_candidate,
            selected_similar_lifecycle_candidate=(
                request.selected_similar_lifecycle_candidate
            ),
        )

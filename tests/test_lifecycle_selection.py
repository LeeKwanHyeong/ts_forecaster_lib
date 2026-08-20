from __future__ import annotations

from datetime import date

import pytest

from modeling_module import (
    CGMMConfig,
    CGMMCorrectionConfig,
    CGMMPreprocessingConfig,
    CGMMSelectionCandidate,
    LifecycleFeatureSchema,
    LifecycleRollingSelectionRequest,
    LifecycleSample,
    LifecycleSamplePurpose,
    LifecycleSelectionError,
    SimilarLifecycleConfig,
    SimilarLifecycleCorrectionConfig,
    SimilarLifecycleSelectionCandidate,
    add_calendar_months,
    select_lifecycle_model_configurations,
)


DATASET_FINGERPRINT = "a" * 64
SCHEMA = LifecycleFeatureSchema(static_cat_names=("family",))


def _sample(index: int, *, inference: bool = False) -> LifecycleSample:
    cohort = index // 6
    start = add_calendar_months(date(2018, 1, 1), cohort)
    scale = 8.0 + (index % 6) + cohort * 0.4
    observed = tuple(scale * (1.0 + month * 0.03) for month in range(12))
    future = tuple(
        scale * (1.0 + min(month, 8) * 0.02) * (0.985**month)
        for month in range(72)
    )
    return LifecycleSample(
        sample_id=f"sample-{index:03d}",
        purpose=(
            LifecycleSamplePurpose.INFERENCE
            if inference
            else LifecycleSamplePurpose.TRAINING
        ),
        lifecycle_start_month=start,
        source_cutoff_month=add_calendar_months(start, 11 if inference else 83),
        observed_target=observed,
        future_target=None if inference else future,
        feature_schema=SCHEMA,
        static_cat=("late-family" if cohort >= 6 else f"family-{index % 2}",),
    )


def _request() -> LifecycleRollingSelectionRequest:
    baseline = CGMMCorrectionConfig(
        name="raw",
        cohort_strength=0.0,
        tail_half_life_months=None,
        minimum_calibration_cohorts=2,
    )
    return LifecycleRollingSelectionRequest(
        training_samples=tuple(_sample(index) for index in range(48)),
        dataset_fingerprint=DATASET_FINGERPRINT,
        cgmm_candidates=(
            CGMMSelectionCandidate(
                name="gmm1-pca1-raw",
                model_config=CGMMConfig(
                    component_count=1,
                    target_component_count=1,
                    initialization_count=1,
                    max_iterations=30,
                ),
                preprocessing_config=CGMMPreprocessingConfig(
                    feature_profile="static_observed_v1"
                ),
                correction_config=baseline,
            ),
        ),
        similar_lifecycle_candidates=(
            SimilarLifecycleSelectionCandidate(
                name="all-k3-raw",
                model_config=SimilarLifecycleConfig(neighbor_count=3),
                correction_config=SimilarLifecycleCorrectionConfig(
                    name="raw",
                    cohort_strength=0.0,
                    tail_half_life_months=None,
                    minimum_calibration_cohorts=2,
                ),
            ),
        ),
        cgmm_random_seeds=(11, 22),
        rolling_validation_fraction=0.5,
    )


def test_train_only_selection_is_chronological_and_reproducible() -> None:
    first = select_lifecycle_model_configurations(_request())
    second = select_lifecycle_model_configurations(_request())

    assert first.to_dict() == second.to_dict()
    assert first.training_sample_count == 48
    assert first.initial_fit_sample_count == 24
    assert first.rolling_sample_count == 24
    assert first.initial_fit_end_month == "2018-04-01"
    assert first.rolling_start_month == "2018-05-01"
    assert first.rolling_end_month == "2018-08-01"
    assert first.selected_cgmm_candidate.name == "gmm1-pca1-raw"
    assert first.selected_similar_lifecycle_candidate.name == "all-k3-raw"
    assert len(first.cgmm_runs) == 2
    assert len(first.similar_lifecycle_runs) == 1
    assert first.to_dict()["training_boundary"]["external_validation_accepted"] is False
    assert all(
        run.fold_months == ("2018-07-01", "2018-08-01")
        for run in (*first.cgmm_runs, *first.similar_lifecycle_runs)
    )


def test_train_only_selection_accepts_m0_profile_for_both_models() -> None:
    baseline = CGMMCorrectionConfig(
        name="tail-72",
        cohort_strength=0.0,
        tail_half_life_months=72.0,
        minimum_calibration_cohorts=2,
    )
    m0_preprocessing = CGMMPreprocessingConfig(
        feature_profile="static_observed_m0_v1"
    )
    result = select_lifecycle_model_configurations(
        LifecycleRollingSelectionRequest(
            training_samples=tuple(_sample(index) for index in range(48)),
            dataset_fingerprint=DATASET_FINGERPRINT,
            cgmm_candidates=(
                CGMMSelectionCandidate(
                    name="gmm1-pca1-m0-tail72",
                    model_config=CGMMConfig(
                        component_count=1,
                        target_component_count=1,
                        initialization_count=1,
                        max_iterations=30,
                    ),
                    preprocessing_config=m0_preprocessing,
                    correction_config=baseline,
                ),
            ),
            similar_lifecycle_candidates=(
                SimilarLifecycleSelectionCandidate(
                    name="all-k3-m0-tail72",
                    model_config=SimilarLifecycleConfig(neighbor_count=3),
                    preprocessing_config=m0_preprocessing,
                    correction_config=SimilarLifecycleCorrectionConfig(
                        **baseline.to_dict()
                    ),
                ),
            ),
            cgmm_random_seeds=(11,),
            rolling_validation_fraction=0.5,
        )
    )

    assert result.selected_cgmm_candidate.preprocessing_config == (
        m0_preprocessing
    )
    assert (
        result.selected_similar_lifecycle_candidate.preprocessing_config
        == m0_preprocessing
    )
    assert all(
        run.evaluation_sample_count > 0
        for run in (*result.cgmm_runs, *result.similar_lifecycle_runs)
    )


def test_selection_rejects_inference_samples_and_duplicate_candidates() -> None:
    request = _request()
    with pytest.raises(LifecycleSelectionError, match="completed training"):
        LifecycleRollingSelectionRequest(
            training_samples=(*request.training_samples[:-1], _sample(47, inference=True)),
            dataset_fingerprint=DATASET_FINGERPRINT,
            cgmm_candidates=request.cgmm_candidates,
            similar_lifecycle_candidates=request.similar_lifecycle_candidates,
        )

    with pytest.raises(LifecycleSelectionError, match="unique candidate names"):
        LifecycleRollingSelectionRequest(
            training_samples=request.training_samples,
            dataset_fingerprint=DATASET_FINGERPRINT,
            cgmm_candidates=(
                request.cgmm_candidates[0],
                request.cgmm_candidates[0],
            ),
            similar_lifecycle_candidates=request.similar_lifecycle_candidates,
        )

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from modeling_module import (
    LifecycleFeatureSchema,
    LifecycleSample,
    LifecycleSamplePurpose,
    SimilarLifecycleArtifactError,
    SimilarLifecycleConfig,
    SimilarLifecycleCorrectionConfig,
    SimilarLifecycleFitRequest,
    SimilarLifecycleForecastRequest,
    add_calendar_months,
    build_similar_lifecycle_rolling_evidence,
    fit_similar_lifecycle,
    fit_similar_lifecycle_correction,
    forecast_similar_lifecycle,
    save_similar_lifecycle_artifact,
    similar_lifecycle_correction_factors,
)
from modeling_module.models import build_model
from modeling_module.models.registry import (
    get_model_spec,
    resolve_training_request_key,
)


DATASET_FINGERPRINT = "b" * 64


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
    cluster = index % 3
    observed = np.asarray(
        [
            (9.0 + index % 7) * np.exp(-month / (15.0 + 4.0 * cluster))
            + cluster * (month >= 6)
            for month in range(12)
        ],
        dtype=np.float64,
    )
    future = np.asarray(
        [
            observed[-1] * np.exp(-month / (20.0 + 8.0 * cluster))
            + cluster * np.exp(-((month - 20.0) / 9.0) ** 2)
            for month in range(1, 73)
        ],
        dtype=np.float64,
    )
    purpose = (
        LifecycleSamplePurpose.INFERENCE
        if inference
        else LifecycleSamplePurpose.TRAINING
    )
    return LifecycleSample(
        sample_id=f"sample-{index}",
        purpose=purpose,
        lifecycle_start_month=start,
        source_cutoff_month=add_calendar_months(
            start,
            11 if inference else 83,
        ),
        observed_target=tuple(observed.tolist()),
        future_target=None if inference else tuple(future.tolist()),
        feature_schema=_schema(),
        static_cont=(None if index % 11 == 0 else 0.005 * (index + 1),),
        static_cat=(family or f"family-{cluster}",),
        observed_cont=tuple(
            (float(150 - month * 4 + cluster * 8),)
            for month in range(12)
        ),
    )


def _config() -> SimilarLifecycleConfig:
    return SimilarLifecycleConfig(neighbor_count=4)


def _fit_result():
    return fit_similar_lifecycle(
        SimilarLifecycleFitRequest(
            samples=tuple(_sample(index) for index in range(24)),
            dataset_fingerprint=DATASET_FINGERPRINT,
            config=_config(),
        )
    )


def test_public_fit_and_forecast_return_retrieval_and_interval_evidence() -> None:
    result = _fit_result()
    inference = (
        _sample(100, inference=True, family="unseen-family"),
        _sample(101, inference=True),
    )
    prediction = forecast_similar_lifecycle(
        SimilarLifecycleForecastRequest(
            model=result.model,
            samples=inference,
        )
    )

    assert result.model_key == "similar_lifecycle"
    assert result.training_sample_count == 24
    assert result.distance_feature_names == result.model.distance_feature_names
    assert prediction.model_id == "modeling-module.similar-lifecycle.v1"
    assert prediction.mean_forecast.shape == (2, 72)
    assert prediction.forecast_std.shape == (2, 72)
    assert prediction.neighbor_weights.shape == (2, 4)
    assert np.allclose(prediction.neighbor_weights.sum(axis=1), 1.0)
    assert (prediction.lower_bound <= prediction.upper_bound).all()
    assert prediction.to_dict()["metadata"]["model_key"] == (
        "similar_lifecycle"
    )


def test_training_query_excludes_its_own_sample_id() -> None:
    samples = tuple(_sample(index) for index in range(18))
    result = fit_similar_lifecycle(
        SimilarLifecycleFitRequest(
            samples=samples,
            dataset_fingerprint=DATASET_FINGERPRINT,
            config=SimilarLifecycleConfig(neighbor_count=3),
        )
    )
    prediction = forecast_similar_lifecycle(
        SimilarLifecycleForecastRequest(
            model=result.model,
            samples=samples[:4],
            apply_correction=False,
        )
    )

    for sample_id, neighbor_ids in zip(
        prediction.sample_ids,
        prediction.neighbor_sample_ids,
        strict=True,
    ):
        assert sample_id not in neighbor_ids


def test_tail_correction_scales_complete_distribution_contract() -> None:
    train = tuple(_sample(index) for index in range(18))
    validation = tuple(
        _sample(
            100 + cohort * 2 + offset,
            start=add_calendar_months(date(2020, 1, 1), cohort),
        )
        for cohort in range(4)
        for offset in range(2)
    )
    evidence = build_similar_lifecycle_rolling_evidence(
        train,
        validation,
        dataset_fingerprint=DATASET_FINGERPRINT,
        model_config=SimilarLifecycleConfig(neighbor_count=3),
    )
    correction = fit_similar_lifecycle_correction(
        evidence[:3],
        SimilarLifecycleCorrectionConfig(
            name="cohort-half-tail72",
            cohort_strength=0.5,
            tail_half_life_months=72.0,
        ),
    )
    query = validation[-2:]
    fit_result = fit_similar_lifecycle(
        SimilarLifecycleFitRequest(
            samples=train,
            dataset_fingerprint=DATASET_FINGERPRINT,
            config=SimilarLifecycleConfig(neighbor_count=3),
            correction_state=correction,
        )
    )
    raw = forecast_similar_lifecycle(
        SimilarLifecycleForecastRequest(
            model=fit_result.model,
            samples=query,
            apply_correction=False,
        )
    )
    corrected = forecast_similar_lifecycle(
        SimilarLifecycleForecastRequest(
            model=fit_result.model,
            samples=query,
        )
    )
    factors = similar_lifecycle_correction_factors(query, correction)

    np.testing.assert_allclose(corrected.mean_forecast, raw.mean_forecast * factors)
    np.testing.assert_allclose(corrected.forecast_std, raw.forecast_std * factors)
    np.testing.assert_allclose(corrected.lower_bound, raw.lower_bound * factors)
    np.testing.assert_allclose(corrected.upper_bound, raw.upper_bound * factors)
    np.testing.assert_array_equal(corrected.neighbor_weights, raw.neighbor_weights)
    assert corrected.neighbor_sample_ids == raw.neighbor_sample_ids
    assert corrected.correction_fingerprint == correction.fingerprint


def test_artifact_embeds_correction_and_restores_exact_prediction(tmp_path) -> None:
    train = tuple(_sample(index) for index in range(20))
    validation = tuple(
        _sample(
            200 + cohort,
            start=add_calendar_months(date(2021, 1, 1), cohort),
        )
        for cohort in range(3)
    )
    evidence = build_similar_lifecycle_rolling_evidence(
        train,
        validation,
        dataset_fingerprint=DATASET_FINGERPRINT,
        model_config=SimilarLifecycleConfig(neighbor_count=3),
    )
    correction = fit_similar_lifecycle_correction(evidence)
    result = fit_similar_lifecycle(
        SimilarLifecycleFitRequest(
            samples=(*train, *validation),
            dataset_fingerprint=DATASET_FINGERPRINT,
            config=SimilarLifecycleConfig(neighbor_count=3),
            correction_state=correction,
        )
    )
    query = (_sample(999, inference=True, family="unknown"),)
    expected = forecast_similar_lifecycle(
        SimilarLifecycleForecastRequest(model=result.model, samples=query)
    )
    receipt = save_similar_lifecycle_artifact(
        result.model,
        tmp_path / "similar-lifecycle",
    )
    actual = forecast_similar_lifecycle(
        SimilarLifecycleForecastRequest(
            model=receipt.artifact_dir,
            samples=query,
        )
    )

    for field_name in (
        "mean_forecast",
        "forecast_std",
        "lower_bound",
        "upper_bound",
        "neighbor_weights",
        "neighbor_distances",
    ):
        np.testing.assert_array_equal(
            getattr(actual, field_name),
            getattr(expected, field_name),
        )
    assert actual.neighbor_sample_ids == expected.neighbor_sample_ids
    assert actual.correction_fingerprint == correction.fingerprint

    with receipt.arrays_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(SimilarLifecycleArtifactError, match="SHA-256 mismatch"):
        forecast_similar_lifecycle(
            SimilarLifecycleForecastRequest(
                model=receipt.artifact_dir,
                samples=query,
            )
        )


def test_rolling_evidence_never_retrieves_current_or_future_cohorts() -> None:
    train = tuple(_sample(index) for index in range(12))
    validation = tuple(
        _sample(
            300 + cohort * 2 + offset,
            start=add_calendar_months(date(2022, 1, 1), cohort),
        )
        for cohort in range(4)
        for offset in range(2)
    )
    evidence = build_similar_lifecycle_rolling_evidence(
        train,
        validation,
        dataset_fingerprint=DATASET_FINGERPRINT,
        model_config=SimilarLifecycleConfig(neighbor_count=3),
    )
    allowed = {sample.sample_id for sample in train}
    for fold in evidence:
        assert {
            neighbor
            for row in fold.prediction.neighbor_sample_ids
            for neighbor in row
        }.issubset(allowed)
        allowed.update(fold.sample_ids)


def test_registry_builds_retrieval_model_but_excludes_generic_training() -> None:
    model = build_model("similar_lifecycle", _config())
    spec = get_model_spec("lifecycleknn")

    assert type(model).__name__ == "SimilarLifecycleForecaster"
    assert spec.key == "similar_lifecycle"
    assert spec.trainable is False
    with pytest.raises(ValueError, match="not trainable through the public"):
        resolve_training_request_key("similar_lifecycle")

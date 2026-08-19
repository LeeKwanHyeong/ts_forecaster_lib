from __future__ import annotations

from datetime import date

import pytest

from modeling_module import (
    LIFECYCLE_INPUT_CONTRACT_ID,
    LifecycleFeatureSchema,
    LifecycleSample,
    LifecycleSamplePurpose,
    LifecycleValidationError,
    LifecycleWindowSpec,
    add_calendar_months,
)


def _schema() -> LifecycleFeatureSchema:
    return LifecycleFeatureSchema(
        static_cont_names=("defect_rate",),
        static_cat_names=("part_family_id",),
        observed_cont_names=("product_sales_qty",),
    )


def test_lifecycle_window_is_frozen_to_12_observed_and_72_future_months() -> None:
    window = LifecycleWindowSpec()

    assert window.observed_months == 12
    assert window.forecast_months == 72
    assert window.total_months == 84
    assert window.forecast_start_index == 12
    assert window.forecast_end_index == 83
    assert window.to_dict()["contract_id"] == LIFECYCLE_INPUT_CONTRACT_ID

    with pytest.raises(LifecycleValidationError, match="fixed to 12 observed"):
        LifecycleWindowSpec(observed_months=13, forecast_months=71)


def test_lifecycle_feature_schema_preserves_order_and_rejects_role_conflicts() -> None:
    schema = _schema()

    restored = LifecycleFeatureSchema.from_dict(schema.to_dict())
    assert restored == schema
    assert restored.fingerprint == schema.fingerprint

    with pytest.raises(LifecycleValidationError, match="both continuous and categorical"):
        LifecycleFeatureSchema(
            observed_cont_names=("sales_qty",),
            known_future_cat_names=("sales_qty",),
        )


def test_training_sample_requires_complete_84_month_target_coverage() -> None:
    sample = LifecycleSample(
        sample_id="sample-1",
        purpose=LifecycleSamplePurpose.TRAINING,
        lifecycle_start_month=date(2018, 1, 1),
        source_cutoff_month=date(2024, 12, 1),
        observed_target=tuple(range(12)),
        future_target=tuple(range(72)),
        feature_schema=_schema(),
        static_cont=(0.01,),
        static_cat=("FAMILY-A",),
        observed_cont=tuple((float(index),) for index in range(12)),
    )

    assert sample.observation_end_month == date(2018, 12, 1)
    assert sample.forecast_start_month == date(2019, 1, 1)
    assert sample.lifecycle_end_month == date(2024, 12, 1)
    assert LifecycleSample.from_dict(sample.to_dict()) == sample

    with pytest.raises(LifecycleValidationError, match="full 84-month lifecycle"):
        LifecycleSample(
            sample_id="sample-1",
            purpose=LifecycleSamplePurpose.TRAINING,
            lifecycle_start_month=date(2018, 1, 1),
            source_cutoff_month=date(2024, 11, 1),
            observed_target=(0.0,) * 12,
            future_target=(0.0,) * 72,
        )


def test_inference_sample_has_12_observations_and_never_accepts_future_labels() -> None:
    sample = LifecycleSample(
        sample_id="sample-2",
        purpose=LifecycleSamplePurpose.INFERENCE,
        lifecycle_start_month=date(2025, 1, 1),
        source_cutoff_month=date(2025, 12, 1),
        observed_target=(0.0,) * 12,
    )
    assert sample.forecast_start_month == date(2026, 1, 1)
    assert sample.future_target is None

    with pytest.raises(LifecycleValidationError, match="must not contain future_target"):
        LifecycleSample(
            sample_id="sample-2",
            purpose=LifecycleSamplePurpose.INFERENCE,
            lifecycle_start_month=date(2025, 1, 1),
            source_cutoff_month=date(2025, 12, 1),
            observed_target=(0.0,) * 12,
            future_target=(0.0,) * 72,
        )


def test_lifecycle_sample_validates_target_and_feature_shapes() -> None:
    with pytest.raises(LifecycleValidationError, match="exactly 12"):
        LifecycleSample(
            sample_id="sample-3",
            purpose=LifecycleSamplePurpose.INFERENCE,
            lifecycle_start_month=date(2025, 1, 1),
            source_cutoff_month=date(2025, 12, 1),
            observed_target=(0.0,) * 11,
        )

    with pytest.raises(LifecycleValidationError, match="exactly 12 rows"):
        LifecycleSample(
            sample_id="sample-3",
            purpose=LifecycleSamplePurpose.INFERENCE,
            lifecycle_start_month=date(2025, 1, 1),
            source_cutoff_month=date(2025, 12, 1),
            observed_target=(0.0,) * 12,
            feature_schema=_schema(),
            static_cont=(None,),
            static_cat=(None,),
            observed_cont=((1.0,),),
        )

    with pytest.raises(LifecycleValidationError, match="non-negative"):
        LifecycleSample(
            sample_id="sample-3",
            purpose=LifecycleSamplePurpose.INFERENCE,
            lifecycle_start_month=date(2025, 1, 1),
            source_cutoff_month=date(2025, 12, 1),
            observed_target=(-1.0,) + (0.0,) * 11,
        )


def test_calendar_month_arithmetic_crosses_years_without_yyyymm_math() -> None:
    assert add_calendar_months(date(2025, 11, 1), 2) == date(2026, 1, 1)

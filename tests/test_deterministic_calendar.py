from __future__ import annotations

import math

import polars as pl
import pytest

from modeling_module.data_loader.deterministic_calendar import (
    WEEKLY_CALENDAR_CONTINUOUS_FEATURES,
    attach_weekly_calendar_features,
    weekly_calendar_schema_fingerprint,
)


def test_weekly_calendar_features_match_demand_engine_formula_and_order():
    frame = pl.DataFrame(
        {
            "oper_part_no": ["B", "A", "A"],
            "demand_dt": [202545, 202501, 202452],
            "demand_qty": [3.0, 1.0, 2.0],
        }
    )

    result = attach_weekly_calendar_features(
        frame,
        date_column="demand_dt",
    )

    assert result.columns == [
        "oper_part_no",
        "demand_dt",
        "demand_qty",
        *WEEKLY_CALENDAR_CONTINUOUS_FEATURES,
    ]
    assert result["oper_part_no"].to_list() == ["B", "A", "A"]
    first_week = result.row(1, named=True)
    assert first_week["sin_annual"] == pytest.approx(
        math.sin(2.0 * math.pi / 52)
    )
    assert first_week["cos_semi"] == pytest.approx(
        math.cos(2.0 * math.pi / 26)
    )
    assert first_week["week_of_year"] == 1.0
    assert first_week["peak_season_flag"] == 1.0
    # ISO 2025-W01 starts on 2024-12-30, matching Demand Engine's Monday date.
    assert first_week["is_year_start"] == 0.0
    assert first_week["is_year_end"] == 1.0
    assert first_week["is_q_start"] == 0.0
    assert first_week["is_q_end"] == 1.0
    assert all(
        result.schema[column] == pl.Float64
        for column in WEEKLY_CALENDAR_CONTINUOUS_FEATURES
    )


def test_weekly_calendar_schema_fingerprint_is_order_sensitive():
    baseline = weekly_calendar_schema_fingerprint()
    reordered = weekly_calendar_schema_fingerprint(
        tuple(reversed(WEEKLY_CALENDAR_CONTINUOUS_FEATURES))
    )

    assert len(baseline) == 64
    assert baseline != reordered


def test_weekly_calendar_feature_contract_fails_fast_on_schema_drift():
    frame = pl.DataFrame({"demand_dt": [202501]})

    with pytest.raises(ValueError, match="unsupported"):
        attach_weekly_calendar_features(
            frame,
            date_column="demand_dt",
            feature_columns=("weather_index",),
        )
    with pytest.raises(ValueError, match="already exist"):
        attach_weekly_calendar_features(
            frame.with_columns(pl.lit(0.0).alias("sin_annual")),
            date_column="demand_dt",
            feature_columns=("sin_annual",),
        )
    with pytest.raises(ValueError, match="date column"):
        attach_weekly_calendar_features(
            pl.DataFrame({"demand_dt": [None]}),
            date_column="demand_dt",
        )

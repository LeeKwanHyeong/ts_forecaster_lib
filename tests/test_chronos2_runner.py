"""주차 변환, 제한된 parquet 조회, 평가 정답 분리를 확인하는 테스트."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from model_test.chronos2.run_zero_shot import forecast_weeks, metrics, parquet_input, synthetic_input, week_id


def test_forecast_calendar_uses_iso_week_53_and_exactly_27_rows():
    weeks = forecast_weeks(202053)
    assert len(weeks) == 27 and weeks[:2] == [202053, 202101]
    assert len(set(weeks)) == 27


def test_synthetic_history_and_holdout_are_separate():
    ids, x, truth, plan = synthetic_input(52)
    assert len(ids) == 3 and x.shape == (3, 52, 1) and truth.shape == (3, 27)
    assert plan == 202601 and not np.shares_memory(x, truth)


def test_unknown_future_actual_is_not_zero():
    result = metrics(np.full((1, 27), np.nan), np.zeros((1, 27)))
    assert result == {"observed_points": 0, "mae": None, "wape_pct": None}
    assert metrics(np.zeros((1, 27)), np.zeros((1, 27)))["wape_pct"] is None


def test_parquet_selection_uses_only_history(tmp_path):
    origin = date.fromisocalendar(202601 // 100, 202601 % 100, 1)
    weeks = [week_id(origin + timedelta(weeks=step)) for step in range(-52, 27)]
    rows = [{"oper_part_no": part, "demand_dt": week, "demand_qty": float(value)}
        for part, value in (("A", 10), ("B", 20)) for week in weeks]
    path = tmp_path / "input.parquet"
    pl.DataFrame(rows).write_parquet(path)
    first = parquet_input(path, 202601, 52, 1)
    pl.DataFrame(rows).with_columns(pl.when(pl.col("demand_dt") >= 202601).then(9999.0).otherwise(pl.col("demand_qty")).alias("demand_qty")).write_parquet(path)
    second = parquet_input(path, 202601, 52, 1)
    assert first[0] == second[0] == ["A"]
    np.testing.assert_array_equal(first[1], second[1])
    assert not np.array_equal(first[2], second[2])
    with pytest.raises(ValueError, match="max_parts"):
        parquet_input(path, 202601, 52, 17)

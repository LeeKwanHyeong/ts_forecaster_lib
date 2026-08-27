from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from tools.run_autotimes_icl_production_refit_5090 import (
    BATCH_SIZE,
    EPOCHS,
    HORIZON,
    LEARNING_RATE,
    LOOKBACK,
    MODEL_KEY,
    SEED,
    STRIDE,
    TRAIN_END_WEEK,
    _complete_training_series,
    _parser,
)


def _week(offset: int) -> int:
    monday = date.fromisocalendar(TRAIN_END_WEEK // 100, TRAIN_END_WEEK % 100, 1)
    value = monday + timedelta(weeks=offset)
    year, week, _ = value.isocalendar()
    return year * 100 + week


def test_autotimes_production_policy_is_frozen() -> None:
    assert MODEL_KEY == "autotimes_base"
    assert (LOOKBACK, HORIZON, TRAIN_END_WEEK) == (52, 26, 202509)
    assert (SEED, BATCH_SIZE, EPOCHS) == (42, 4, 5)
    assert STRIDE == 26
    assert LEARNING_RATE == 1e-3


def test_complete_training_series_keeps_all_eligible_and_reaches_cutoff() -> None:
    weeks = [_week(offset) for offset in range(-239, 1)]
    short_weeks = weeks[-80:]
    stale_weeks = [_week(offset) for offset in range(-240, 0)]
    frame = pl.DataFrame(
        [
            {
                "oper_part_no": part,
                "demand_dt": week,
                "demand_qty": float(index % 5),
            }
            for part, values in (
                ("eligible-a", weeks),
                ("eligible-b", weeks),
                ("short", short_weeks),
                ("stale", stale_weeks),
            )
            for index, week in enumerate(values)
        ]
    )

    selected, receipt = _complete_training_series(frame)

    assert selected["oper_part_no"].unique().sort().to_list() == [
        "eligible-a",
        "eligible-b",
    ]
    assert receipt["source_series_count"] == 4
    assert receipt["eligible_series_count"] == 2
    assert receipt["excluded_series"] == {
        "does_not_reach_cutoff": 1,
        "insufficient_history": 1,
        "non_contiguous": 0,
    }
    assert selected.group_by("oper_part_no").agg(
        pl.col("demand_dt").max()
    )["demand_dt"].to_list() == [TRAIN_END_WEEK, TRAIN_END_WEEK]


def test_autotimes_refit_parser_requires_all_governed_sources() -> None:
    required = {
        action.dest
        for action in _parser()._actions
        if getattr(action, "required", False)
    }
    assert required == {
        "target_source",
        "input_manifest",
        "operation_part_source",
        "operation_part_manifest",
        "llm_local_path",
        "output_root",
    }

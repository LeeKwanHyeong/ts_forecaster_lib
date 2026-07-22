from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest


TOOL = Path(__file__).resolve().parents[1] / "tools" / "compare_exogenous_models_5090.py"
SPEC = importlib.util.spec_from_file_location("_exogenous_5090_comparison", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_comparison_has_paired_endogenous_and_exogenous_cases():
    assert [(case.key, case.family, case.exogenous) for case in MODULE.MODEL_CASES] == [
        ("patchtst_endogenous", "patchtst", False),
        ("patchtst_exogenous", "patchtst", True),
        ("patchmixer_endogenous", "patchmixer", False),
        ("patchmixer_exogenous", "patchmixer", True),
    ]


def test_comparison_uses_nonempty_past_and_future_feature_contracts():
    assert MODULE.PAST_EXOGENOUS_COLUMNS
    assert MODULE.FUTURE_EXOGENOUS_COLUMNS
    assert len(set(MODULE.PAST_EXOGENOUS_COLUMNS)) == len(MODULE.PAST_EXOGENOUS_COLUMNS)
    assert len(set(MODULE.FUTURE_EXOGENOUS_COLUMNS)) == len(MODULE.FUTURE_EXOGENOUS_COLUMNS)


def test_series_split_is_deterministic_and_disjoint():
    first = MODULE._split_ids(
        [str(index) for index in range(20)],
        seed=11,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    second = MODULE._split_ids(
        [str(index) for index in range(20)],
        seed=11,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    assert first == second
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["validation"]).isdisjoint(first["test"])


def test_split_rejects_invalid_ratios():
    with pytest.raises(ValueError, match="less than 1"):
        MODULE._split_ids(
            ["A", "B", "C"],
            seed=11,
            val_ratio=0.5,
            test_ratio=0.5,
        )


def _valid_frame() -> pl.DataFrame:
    data = {
        "unique_id": ["A", "A"],
        "date": [202601, 202602],
        "y": [1.0, 2.0],
    }
    for column in (
        *MODULE.PAST_EXOGENOUS_COLUMNS,
        *MODULE.FUTURE_EXOGENOUS_COLUMNS,
    ):
        data[column] = [0.0, 1.0]
    return pl.DataFrame(data)


def test_frame_validation_accepts_finite_unique_rows():
    MODULE._validate_frame(_valid_frame())


@pytest.mark.parametrize("bad_value", [None, float("nan"), float("inf")])
def test_frame_validation_rejects_invalid_numeric_values(bad_value):
    frame = _valid_frame().with_columns(
        pl.Series(MODULE.PAST_EXOGENOUS_COLUMNS[0], [bad_value, 1.0])
    )

    with pytest.raises(ValueError, match="null|non-finite"):
        MODULE._validate_frame(frame)


def test_frame_validation_rejects_duplicate_series_dates():
    frame = _valid_frame().with_columns(pl.lit(202601).alias("date"))

    with pytest.raises(ValueError, match="duplicate"):
        MODULE._validate_frame(frame)

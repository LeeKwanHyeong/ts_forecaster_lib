from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest


TOOL = Path(__file__).resolve().parents[1] / "tools" / "compare_exogenous_models_5090.py"
SPEC = importlib.util.spec_from_file_location("_exogenous_5090_comparison", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_comparison_has_paired_endogenous_and_exogenous_cases():
    assert [
        (
            case.key,
            case.family,
            case.past_exogenous,
            case.future_exogenous,
            case.exogenous,
        )
        for case in MODULE.MODEL_CASES
    ] == [
        ("patchtst_endogenous", "patchtst", False, False, False),
        ("patchtst_exogenous", "patchtst", True, True, True),
        ("patchmixer_endogenous", "patchmixer", False, False, False),
        ("patchmixer_past_gate", "patchmixer", True, False, True),
        ("patchmixer_future_shift", "patchmixer", False, True, True),
        ("patchmixer_exogenous", "patchmixer", True, True, True),
    ]


def test_patchmixer_ablation_pairs_isolate_gate_and_future_shift():
    assert MODULE.PATCHMIXER_ABLATION_PAIRS == {
        "past_gate_vs_endogenous": (
            "patchmixer_endogenous",
            "patchmixer_past_gate",
        ),
        "future_shift_vs_endogenous": (
            "patchmixer_endogenous",
            "patchmixer_future_shift",
        ),
        "full_vs_endogenous": (
            "patchmixer_endogenous",
            "patchmixer_exogenous",
        ),
        "full_vs_future_shift": (
            "patchmixer_future_shift",
            "patchmixer_exogenous",
        ),
        "full_vs_past_gate": (
            "patchmixer_past_gate",
            "patchmixer_exogenous",
        ),
    }


def test_patchmixer_ablation_configs_enable_only_requested_inputs():
    cases = {case.key: case for case in MODULE.MODEL_CASES}
    expected = {
        "patchmixer_endogenous": (0, 0),
        "patchmixer_past_gate": (len(MODULE.PAST_EXOGENOUS_COLUMNS), 0),
        "patchmixer_future_shift": (0, len(MODULE.FUTURE_EXOGENOUS_COLUMNS)),
        "patchmixer_exogenous": (
            len(MODULE.PAST_EXOGENOUS_COLUMNS),
            len(MODULE.FUTURE_EXOGENOUS_COLUMNS),
        ),
    }
    for key, widths in expected.items():
        config = MODULE._patchmixer_config(cases[key])
        assert (config.past_exo_cont_dim, config.future_exo_dim) == widths
        assert config.past_exo_mode == "z_gate"


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


def _prediction_payload(predictions: list[list[float]]) -> dict[str, object]:
    values = np.asarray(predictions, dtype=np.float64)
    return {
        "targets": np.ones_like(values),
        "predictions": values,
        "uids": np.asarray(["A"] * len(values), dtype=object),
    }


def test_candidate_summary_reports_positive_improvement_for_lower_error():
    baseline = _prediction_payload([[2.0, 2.0]])
    candidate = _prediction_payload([[1.5, 1.5]])

    result = MODULE._candidate_summary(
        baseline,
        candidate,
        baseline_name="endogenous",
        candidate_name="past_gate",
    )

    assert result["candidate_relative_improvement_pct"]["mae"] == pytest.approx(50.0)
    assert result["overall_mae_winner"] == "past_gate"
    assert result["pointwise_absolute_error_win_rate"]["past_gate"] == 1.0


def test_input_ablation_reports_positive_degradation_when_inputs_help():
    full = _prediction_payload([[1.25, 1.25]])
    ablated = _prediction_payload([[1.5, 1.5]])

    result = MODULE._input_ablation_summary(full, ablated)

    assert result["relative_error_degradation_pct"]["mae"] == pytest.approx(100.0)
    assert result["prediction_mean_absolute_delta"] == pytest.approx(0.25)


def _performance_record(
    key: str,
    *,
    parameters: int,
    training_ms: float,
    inference_ms: float,
    training_memory: float,
    inference_memory: float,
) -> dict[str, object]:
    return {
        "model": key,
        "parameters": parameters,
        "timing_ms": {"mean": training_ms},
        "throughput": {"samples_per_second": 1000.0 / training_ms},
        "memory_mib": {"peak_allocated": training_memory},
        "inference": {
            "timing_ms": {"mean": inference_ms},
            "throughput": {"samples_per_second": 1000.0 / inference_ms},
            "memory_mib": {"peak_allocated": inference_memory},
        },
    }


def test_performance_delta_separates_training_and_inference_overhead():
    baseline = _performance_record(
        "baseline",
        parameters=100,
        training_ms=2.0,
        inference_ms=1.0,
        training_memory=20.0,
        inference_memory=10.0,
    )
    candidate = _performance_record(
        "candidate",
        parameters=125,
        training_ms=2.5,
        inference_ms=1.1,
        training_memory=24.0,
        inference_memory=11.0,
    )

    result = MODULE._performance_delta(baseline, candidate)

    assert result["training_step_time_overhead_pct"] == pytest.approx(25.0)
    assert result["inference_step_time_overhead_pct"] == pytest.approx(10.0)
    assert result["parameter_overhead"] == 25
    assert result["training_peak_allocated_overhead_mib"] == pytest.approx(4.0)
    assert result["inference_peak_allocated_overhead_mib"] == pytest.approx(1.0)

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch


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


def test_focused_patchmixer_shift_space_cases_isolate_the_shift_coordinate():
    assert [
        (
            case.key,
            case.past_exogenous,
            case.future_exogenous,
            case.future_shift_space,
            case.future_normalized_residual_limit,
        )
        for case in MODULE.PATCHMIXER_SHIFT_SPACE_CASES
    ] == [
        ("patchmixer_endogenous", False, False, None, None),
        ("patchmixer_future_shift", False, True, "output", None),
        ("patchmixer_future_shift_normalized", False, True, "normalized", None),
        (
            "patchmixer_future_shift_normalized_bounded",
            False,
            True,
            "normalized",
            0.15,
        ),
    ]
    assert MODULE.PATCHMIXER_SHIFT_SPACE_PAIRS == {
        "output_vs_endogenous": (
            "patchmixer_endogenous",
            "patchmixer_future_shift",
        ),
        "normalized_vs_endogenous": (
            "patchmixer_endogenous",
            "patchmixer_future_shift_normalized",
        ),
        "normalized_vs_output": (
            "patchmixer_future_shift",
            "patchmixer_future_shift_normalized",
        ),
        "bounded_vs_endogenous": (
            "patchmixer_endogenous",
            "patchmixer_future_shift_normalized_bounded",
        ),
        "bounded_vs_output": (
            "patchmixer_future_shift",
            "patchmixer_future_shift_normalized_bounded",
        ),
        "bounded_vs_normalized": (
            "patchmixer_future_shift_normalized",
            "patchmixer_future_shift_normalized_bounded",
        ),
    }


def test_focused_shift_space_configs_share_architecture_and_change_only_space():
    cases = {case.key: case for case in MODULE.PATCHMIXER_SHIFT_SPACE_CASES}
    output = MODULE._patchmixer_config(cases["patchmixer_future_shift"])
    normalized = MODULE._patchmixer_config(
        cases["patchmixer_future_shift_normalized"]
    )
    bounded = MODULE._patchmixer_config(
        cases["patchmixer_future_shift_normalized_bounded"]
    )

    assert output.future_exo_dim == normalized.future_exo_dim == len(
        MODULE.FUTURE_EXOGENOUS_COLUMNS
    )
    assert output.past_exo_cont_dim == normalized.past_exo_cont_dim == 0
    assert output.future_exo_shift_space == "output"
    assert normalized.future_exo_shift_space == "normalized"
    assert bounded.future_exo_shift_space == "normalized"
    assert output.future_exo_normalized_residual_limit is None
    assert normalized.future_exo_normalized_residual_limit is None
    assert bounded.future_exo_normalized_residual_limit == 0.15


def test_focused_case_metadata_records_bounded_residual_limit():
    metadata = [
        MODULE._case_metadata(case)
        for case in MODULE.PATCHMIXER_SHIFT_SPACE_CASES
    ]

    assert metadata[-1]["key"] == "patchmixer_future_shift_normalized_bounded"
    assert metadata[-1]["future_normalized_residual_limit"] == 0.15
    assert all(
        row["future_normalized_residual_limit"] is None
        for row in metadata[:-1]
    )


def test_case_set_cli_defaults_to_historical_comparison_and_accepts_focused_set():
    parser = MODULE._build_parser()
    common = ["--data", "input.parquet", "--output", "result.json"]

    assert parser.parse_args(common).case_set == "all"
    assert (
        parser.parse_args([*common, "--case-set", "patchmixer-shift-space"]).case_set
        == "patchmixer-shift-space"
    )


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
        assert (
            int(getattr(config, "past_exo_cont_dim", 0)),
            int(getattr(config, "future_exo_dim", 0)),
        ) == widths
        expected_mode = "none" if key == "patchmixer_endogenous" else "z_gate"
        assert getattr(config, "past_exo_mode", "none") == expected_mode
        assert getattr(config, "future_exo_shift_space", "output") == "output"


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
        "history_std": np.ones(values.shape[0], dtype=np.float64),
        "history_features": np.ones((values.shape[0], 1), dtype=np.float64),
        "history_feature_names": ("history_std",),
        "uids": np.asarray(["A"] * len(values), dtype=object),
    }


def _diagnostic_payload(
    predictions: list[list[float]],
    *,
    history_std: list[float],
) -> dict[str, object]:
    values = np.asarray(predictions, dtype=np.float64)
    return {
        "targets": np.zeros_like(values),
        "predictions": values,
        "history_std": np.asarray(history_std, dtype=np.float64),
        "history_features": np.asarray(history_std, dtype=np.float64)[:, None],
        "history_feature_names": ("history_std",),
        "uids": np.asarray(["A", "A", "B", "B"], dtype=object),
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


def test_paired_diagnostics_decompose_error_by_scale_series_and_horizon():
    baseline = _diagnostic_payload(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        history_std=[1.0, 2.0, 3.0, 4.0],
    )
    candidate = _diagnostic_payload(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
        history_std=[1.0, 2.0, 3.0, 4.0],
    )

    result = MODULE._paired_error_diagnostics(baseline, candidate)

    assert result["overall"]["candidate_mae_delta"] == pytest.approx(2.5)
    assert result["window_mae_delta_correlation_with_history_std"] == pytest.approx(
        1.0
    )
    assert len(result["by_horizon"]) == 2
    assert result["by_series"]["A"]["candidate_mae"] == pytest.approx(1.5)
    assert result["by_series"]["B"]["candidate_mae"] == pytest.approx(3.5)
    assert [row["quartile"] for row in result["by_history_std_quartile"]] == [
        "q1",
        "q2",
        "q3",
        "q4",
    ]


def test_future_shift_diagnostics_report_raw_and_history_scaled_effect():
    without_future = _diagnostic_payload(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        history_std=[2.0, 4.0, 6.0, 8.0],
    )
    zero_future = _diagnostic_payload(
        [[0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8]],
        history_std=[2.0, 4.0, 6.0, 8.0],
    )
    full = _diagnostic_payload(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
        history_std=[2.0, 4.0, 6.0, 8.0],
    )

    result = MODULE._future_shift_diagnostics(full, without_future, zero_future)
    effect = result["total_effect"]["overall"]

    assert effect["mean_absolute"] == pytest.approx(2.5)
    assert effect["mean_absolute_in_history_std_units"] == pytest.approx(0.5)
    assert effect[
        "window_absolute_effect_correlation_with_history_std"
    ] == pytest.approx(1.0)
    assert len(result["total_effect"]["by_horizon"]) == 2
    assert result["feature_conditioned_effect"]["overall"][
        "mean_absolute_in_history_std_units"
    ] == pytest.approx(0.4)
    assert result["zero_input_bias_effect"]["overall"][
        "mean_absolute_in_history_std_units"
    ] == pytest.approx(0.1)
    assert result["error_comparison"]["overall"][
        "candidate_mae_delta"
    ] == pytest.approx(2.5)


def test_history_gate_features_are_finite_and_history_only():
    inputs = torch.arange(
        2 * MODULE.LOOKBACK,
        dtype=torch.float32,
    ).reshape(2, MODULE.LOOKBACK, 1)

    features = MODULE._history_gate_features(inputs)

    assert features.shape == (2, len(MODULE.HISTORY_GATE_FEATURE_NAMES))
    assert torch.isfinite(features).all()
    assert MODULE.HISTORY_GATE_FEATURE_NAMES == (
        "log1p_abs_mean",
        "log1p_std",
        "last_z",
        "linear_trend_z",
        "recent_4_minus_mean_z",
        "recent_12_minus_mean_z",
        "seasonal_52_gap_z",
        "range_z",
        "zero_fraction",
    )


def test_gate_oracles_bound_scalar_and_horizon_coordinates():
    base = np.zeros((2, 2), dtype=np.float64)
    full = np.ones((2, 2), dtype=np.float64)
    targets = np.asarray([[0.25, 0.75], [2.0, -1.0]], dtype=np.float64)

    scalar, scalar_weights = MODULE._mse_optimal_gate_targets(
        base,
        full,
        targets,
        horizon_shared=True,
    )
    horizon, horizon_weights = MODULE._mse_optimal_gate_targets(
        base,
        full,
        targets,
        horizon_shared=False,
    )

    np.testing.assert_allclose(scalar, [[0.5], [0.5]])
    np.testing.assert_allclose(scalar_weights, [[2.0], [2.0]])
    np.testing.assert_allclose(horizon, [[0.25, 0.75], [1.0, 0.0]])
    np.testing.assert_allclose(horizon_weights, np.ones((2, 2)))
    scalar_mse = MODULE._forecast_mse_with_gate(base, full, targets, scalar)
    assert scalar_mse <= MODULE._forecast_mse_with_gate(
        base,
        full,
        targets,
        np.zeros((2, 1)),
    )
    assert scalar_mse <= MODULE._forecast_mse_with_gate(
        base,
        full,
        targets,
        np.ones((2, 1)),
    )


def test_constant_gate_fits_unclipped_quadratic_target_before_clipping():
    base = np.zeros((2, 1), dtype=np.float64)
    full = np.ones((2, 1), dtype=np.float64)
    targets = np.asarray([[10.0], [0.0]], dtype=np.float64)
    regression_targets, weights = MODULE._mse_gate_regression_targets(
        base,
        full,
        targets,
        horizon_shared=True,
    )

    fitted = MODULE._fit_constant_gate(regression_targets, weights)

    np.testing.assert_allclose(regression_targets, [[10.0], [0.0]])
    np.testing.assert_allclose(fitted, [[1.0]])
    fitted_mse = MODULE._forecast_mse_with_gate(
        base,
        full,
        targets,
        fitted.repeat(2, 0),
    )
    off_mse = MODULE._forecast_mse_with_gate(
        base,
        full,
        targets,
        np.zeros((2, 1)),
    )
    assert fitted_mse <= off_mse


def test_nested_series_oof_gate_does_not_use_held_out_targets():
    uids = np.repeat(np.asarray(["A", "B", "C", "D"], dtype=object), 3)
    features = np.arange(12, dtype=np.float64)[:, None]
    base = np.zeros((12, 2), dtype=np.float64)
    full = np.ones((12, 2), dtype=np.float64)
    targets = np.repeat(np.linspace(0.1, 0.9, 12)[:, None], 2, axis=1)
    gate_targets, gate_weights = MODULE._mse_gate_regression_targets(
        base,
        full,
        targets,
        horizon_shared=True,
    )
    original, selected = MODULE._nested_group_oof_history_gate(
        "ridge",
        (0.1, 1.0),
        features,
        gate_targets,
        gate_weights,
        uids,
        base,
        full,
        targets,
    )

    changed_targets = targets.copy()
    changed_targets[uids == "A"] = 20.0
    changed_gate_targets, changed_gate_weights = MODULE._mse_gate_regression_targets(
        base,
        full,
        changed_targets,
        horizon_shared=True,
    )
    changed, changed_selected = MODULE._nested_group_oof_history_gate(
        "ridge",
        (0.1, 1.0),
        features,
        changed_gate_targets,
        changed_gate_weights,
        uids,
        base,
        full,
        changed_targets,
    )

    assert original[uids == "A"] == pytest.approx(changed[uids == "A"])
    assert selected["A"] == changed_selected["A"]
    assert np.all((0.0 <= original) & (original <= 1.0))


def test_validation_gate_upper_bound_separates_oof_estimates_from_oracles():
    uids = np.repeat(np.asarray(["A", "B", "C", "D"], dtype=object), 2)
    feature = np.linspace(-1.0, 1.0, 8)[:, None]
    targets = np.repeat(np.linspace(0.0, 1.0, 8)[:, None], 2, axis=1)

    def payload(predictions: np.ndarray) -> dict[str, object]:
        return {
            "targets": targets,
            "predictions": predictions,
            "history_std": np.ones(8, dtype=np.float64),
            "history_features": feature,
            "history_feature_names": ("synthetic_history",),
            "uids": uids,
        }

    result = MODULE._history_conditioned_gate_validation_upper_bound(
        payload(np.full_like(targets, 0.5)),
        payload(np.ones_like(targets)),
        payload(np.zeros_like(targets)),
    )

    assert result["protocol"]["test_targets_used"] is False
    scalar = result["normalized_residual_gate"]["window_scalar"]
    methods = scalar["evaluations"]["validation_all_rolling_windows"]["methods"]
    assert "nested_series_oof_ridge" in methods
    assert "nested_series_oof_knn" in methods
    assert methods["oracle_mse"]["metrics"]["mse"] <= methods["always_on"][
        "metrics"
    ]["mse"]
    assert methods["validation_fit_constant"]["metrics"]["mse"] <= min(
        methods["always_off"]["metrics"]["mse"],
        methods["always_on"]["metrics"]["mse"],
    )
    assert scalar["evaluations"]["validation_last_origin_per_series"][
        "windows"
    ] == 4
    aggregate = MODULE._aggregate_validation_gate_upper_bound(
        [
            {"seed": 11, "validation_gate_upper_bound": result},
            {"seed": 22, "validation_gate_upper_bound": result},
        ]
    )
    ridge = aggregate["normalized_residual_gate"]["window_scalar"][
        "validation_all_rolling_windows"
    ]["nested_series_oof_ridge"]
    assert len(ridge["records"]) == 2
    assert ridge["mean_mse"] == pytest.approx(
        methods["nested_series_oof_ridge"]["metrics"]["mse"]
    )


def test_future_shift_diagnostic_disable_retains_required_input_and_restores_scale():
    class RequiredFutureModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.exo_scale = 0.25

        def forward(
            self,
            inputs: torch.Tensor,
            *,
            future_exo: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if future_exo is None:
                raise RuntimeError("future_exo is required")
            return inputs + self.exo_scale * future_exo

    model = RequiredFutureModel()
    inputs = torch.ones(1, 2)
    future_exo = torch.full((1, 2), 4.0)

    with MODULE._temporarily_disable_future_shift(model, disabled=True):
        disabled = model(inputs, future_exo=future_exo)
        assert model.exo_scale == 0.0

    assert torch.equal(disabled, inputs)
    assert model.exo_scale == 0.25
    assert torch.equal(
        model(inputs, future_exo=future_exo),
        torch.full((1, 2), 2.0),
    )


def test_focused_accuracy_aggregate_reports_each_shift_space_pair():
    predictions = {
        "patchmixer_endogenous": {
            "all": _prediction_payload([[2.0, 2.0]]),
            "last": _prediction_payload([[2.0, 2.0]]),
        },
        "patchmixer_future_shift": {
            "all": _prediction_payload([[1.5, 1.5]]),
            "last": _prediction_payload([[1.5, 1.5]]),
        },
        "patchmixer_future_shift_normalized": {
            "all": _prediction_payload([[1.25, 1.25]]),
            "last": _prediction_payload([[1.25, 1.25]]),
        },
        "patchmixer_future_shift_normalized_bounded": {
            "all": _prediction_payload([[1.1, 1.1]]),
            "last": _prediction_payload([[1.1, 1.1]]),
        },
    }
    paired = MODULE._candidate_comparison_group(
        predictions,
        MODULE.PATCHMIXER_SHIFT_SPACE_PAIRS,
    )

    result = MODULE._aggregate_accuracy(
        [{"seed": 11, "paired_comparison": {"patchmixer_shift_space": paired}}],
        case_set="patchmixer-shift-space",
        comparison_pairs=MODULE.PATCHMIXER_SHIFT_SPACE_PAIRS,
    )

    comparisons = result["patchmixer_shift_space"]
    assert set(comparisons) == set(MODULE.PATCHMIXER_SHIFT_SPACE_PAIRS)
    normalized = comparisons["normalized_vs_output"][
        "test_all_rolling_windows"
    ]
    assert normalized["seed_wins"]["future_shift_normalized"] == 1
    assert normalized["mae_improvement_pct"]["mean"] == pytest.approx(50.0)
    bounded = comparisons["bounded_vs_output"]["test_all_rolling_windows"]
    assert bounded["seed_wins"]["future_shift_normalized_bounded"] == 1
    assert bounded["mae_improvement_pct"]["mean"] == pytest.approx(80.0)


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


def test_focused_performance_summary_reports_each_shift_space_pair():
    results = [
        _performance_record(
            "patchmixer_endogenous",
            parameters=100,
            training_ms=2.0,
            inference_ms=1.0,
            training_memory=20.0,
            inference_memory=10.0,
        ),
        _performance_record(
            "patchmixer_future_shift",
            parameters=110,
            training_ms=2.2,
            inference_ms=1.1,
            training_memory=21.0,
            inference_memory=10.5,
        ),
        _performance_record(
            "patchmixer_future_shift_normalized",
            parameters=110,
            training_ms=2.3,
            inference_ms=1.2,
            training_memory=21.0,
            inference_memory=10.5,
        ),
        _performance_record(
            "patchmixer_future_shift_normalized_bounded",
            parameters=110,
            training_ms=2.4,
            inference_ms=1.25,
            training_memory=21.0,
            inference_memory=10.5,
        ),
    ]

    result = MODULE._performance_summary(
        results,
        case_set="patchmixer-shift-space",
        comparison_pairs=MODULE.PATCHMIXER_SHIFT_SPACE_PAIRS,
    )

    comparisons = result["patchmixer_shift_space"]
    assert set(comparisons) == set(MODULE.PATCHMIXER_SHIFT_SPACE_PAIRS)
    assert comparisons["normalized_vs_output"]["parameter_overhead"] == 0
    assert comparisons["normalized_vs_output"][
        "training_peak_allocated_overhead_mib"
    ] == pytest.approx(0.0)
    assert comparisons["bounded_vs_normalized"]["parameter_overhead"] == 0

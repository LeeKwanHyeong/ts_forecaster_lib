from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from modeling_module.models.registry import PATCHMIXER_CAPABILITY_DEFAULTS


TOOL = Path(__file__).resolve().parents[1] / "tools" / "aggregate_patchmixer_multiseed.py"
SPEC = importlib.util.spec_from_file_location("_patchmixer_multiseed_tool", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _metric_values(value: float) -> dict[str, float]:
    return {
        "mae": value,
        "mse": value * value,
        "rmse": value,
        "smape": value / 1_000.0,
        "wape": value / 1_000.0,
    }


def _evaluation(value: float) -> dict:
    return {
        "windows": 1,
        "forecast_points": 1,
        "metrics": {"micro": _metric_values(value)},
    }


def _paired(original: float, enhanced: float, test_id: str) -> dict:
    winner = "original" if original < enhanced else "enhanced" if enhanced < original else "tie"
    original_rate = 0.6 if winner == "original" else 0.4 if winner == "enhanced" else 0.0
    enhanced_rate = 0.6 if winner == "enhanced" else 0.4 if winner == "original" else 0.0
    tie_rate = 1.0 if winner == "tie" else 0.0
    return {
        "original_relative_improvement_pct": {
            metric: MODULE._relative_improvement(
                _metric_values(original)[metric],
                _metric_values(enhanced)[metric],
            )
            for metric in MODULE.ERROR_METRICS
        },
        "overall_mae_winner": winner,
        "pointwise_absolute_error_win_rate": {
            "original": original_rate,
            "enhanced": enhanced_rate,
            "tie": tie_rate,
        },
        "series_mae_winner": {test_id: winner},
    }


def _accuracy_result(
    seed: int,
    splits: dict[str, list[str]],
    original_all: float,
    original_last: float,
) -> dict:
    enhanced_all = 100.0
    enhanced_last = 100.0
    test_id = splits["test"][0]
    source = {
        "original_upstream_commit": "upstream",
        "enhanced_baseline_commit": "enhanced",
        "git": {"commit": "enhanced", "branch": "benchmark", "working_tree_dirty": True},
    }
    environment = {
        "device": "NVIDIA GeForce RTX 5090",
        "torch": "test",
        "cuda_runtime": "test",
    }
    protocol = {
        "seed": seed,
        "initialization_seed": seed,
        "training_randomness_seed": seed + 1,
        "dataloader_seed": seed + 2,
        "split_fingerprint": f"split-{seed}",
        "endogenous_only": True,
        "precision": "float32",
        "loss": "mse",
        "batch_size": 64,
        "reference_config": {"lookback": 54, "horizon": 27},
    }
    dataset = {
        "path": "/data/walmart.parquet",
        "sha256": "dataset-sha",
        "rows": 3,
        "series": 3,
        "split_series_counts": {"train": 1, "validation": 1, "test": 1},
        "split_window_counts": {"train": 1, "validation": 1, "test_all": 1, "test_last_origin": 1},
        "total_windows": 3,
        "splits": splits,
    }

    def model_payload(name: str, all_value: float, last_value: float) -> dict:
        parameters = 10 if name == "original" else 100
        return {
            "training": {
                "best_epoch": 10,
                "best_validation_mse": all_value * all_value,
                "best_validation_rmse": all_value,
                "elapsed_seconds": 1.0,
                "epochs_completed": 10,
                "parameters": parameters,
                "peak_allocated_mib": float(parameters),
            },
            "test_all_rolling_windows": _evaluation(all_value),
            "test_last_origin_per_series": _evaluation(last_value),
        }

    return {
        "schema_version": 1,
        "source": source,
        "environment": environment,
        "protocol": protocol,
        "dataset": dataset,
        "models": {
            "original": model_payload("original", original_all, original_last),
            "enhanced": model_payload("enhanced", enhanced_all, enhanced_last),
        },
        "paired_comparison": {
            "test_all_rolling_windows": _paired(original_all, enhanced_all, test_id),
            "test_last_origin_per_series": _paired(original_last, enhanced_last, test_id),
        },
    }


def _accuracy_results() -> list[dict]:
    return [
        _accuracy_result(
            11,
            {"train": ["a"], "validation": ["b"], "test": ["c"]},
            original_all=90.0,
            original_last=95.0,
        ),
        _accuracy_result(
            22,
            {"train": ["b"], "validation": ["c"], "test": ["a"]},
            original_all=101.0,
            original_last=104.0,
        ),
        _accuracy_result(
            33,
            {"train": ["c"], "validation": ["a"], "test": ["b"]},
            original_all=80.0,
            original_last=90.0,
        ),
    ]


def _performance_result() -> dict:
    return {
        "schema_version": 1,
        "source": {
            "original_upstream_commit": "upstream",
            "enhanced_baseline_commit": "enhanced",
        },
        "environment": {"device": "NVIDIA GeForce RTX 5090"},
        "protocol": {
            "precision": "bf16",
            "measured_steps": 100,
            "batch_size": 64,
            "reference_config": {"lookback": 54, "horizon": 27},
        },
        "models": [
            {
                "model": "original",
                "parameters": 10,
                "timing_ms": {"mean": 2.0},
                "throughput": {"samples_per_second": 200.0},
                "memory_mib": {"peak_allocated": 20.0},
            },
            {
                "model": "enhanced",
                "parameters": 100,
                "timing_ms": {"mean": 4.0},
                "throughput": {"samples_per_second": 100.0},
                "memory_mib": {"peak_allocated": 200.0},
            },
        ],
    }


def test_multiseed_summary_validates_contract_and_promotes_original() -> None:
    summary = MODULE.build_summary(_accuracy_results(), _performance_result())
    rolling = summary["aggregate"]["test_all_rolling_windows"]

    assert summary["validation"]["status"] == "passed"
    assert rolling["overall_mae_seed_wins"] == {
        "original": 2,
        "enhanced": 1,
        "tie": 0,
    }
    assert rolling["original_relative_improvement_pct"]["mae"]["mean"] == pytest.approx(
        (10.0 - 1.0 + 20.0) / 3.0
    )
    assert summary["decision"]["status"] == "promote_original_for_endogenous_point"
    assert summary["decision"]["capability_defaults"] == PATCHMIXER_CAPABILITY_DEFAULTS
    assert all(check["passed"] for check in summary["decision"]["checks"])


def test_multiseed_summary_rejects_fixed_protocol_drift() -> None:
    results = _accuracy_results()
    results[1]["protocol"]["batch_size"] = 32

    with pytest.raises(ValueError, match="fixed protocol"):
        MODULE.build_summary(results, _performance_result())


def test_multiseed_summary_rejects_corrupt_paired_metric() -> None:
    results = copy.deepcopy(_accuracy_results())
    results[2]["paired_comparison"]["test_all_rolling_windows"][
        "original_relative_improvement_pct"
    ]["mae"] = -999.0

    with pytest.raises(ValueError, match="paired improvement mismatch"):
        MODULE.build_summary(results, _performance_result())

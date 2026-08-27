"""Contract tests for the H26 nonnegative-output experiment decision."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
DECISION_PATH = ROOT / "docs/DSIOV100H26NonnegativeOutputExperimentDecision.json"


def _decision() -> dict[str, object]:
    return json.loads(DECISION_PATH.read_text(encoding="utf-8"))


def test_clip_zero_is_pointwise_nonworsening_for_nonnegative_actuals() -> None:
    actual = np.asarray([0.0, 1.0, 5.0, 20.0])
    raw = np.asarray([-10.0, -0.5, 2.0, 25.0])
    clipped = np.maximum(raw, 0.0)

    assert np.all(np.abs(actual - clipped) <= np.abs(actual - raw))


def test_every_model_seed_improves_mae_after_clip_zero() -> None:
    rows = _decision()["per_seed_evidence"]

    assert len(rows) == 12
    assert all(row["clip_mae"] < row["raw_mae"] for row in rows)
    assert {(row["model_key"], row["seed"]) for row in rows} == {
        (model, seed)
        for model in ("exotst_base", "patchtst_exogenous", "timexer_base")
        for seed in (11, 22, 33, 42)
    }


def test_model_summary_recomputes_from_seed_evidence() -> None:
    decision = _decision()
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in decision["per_seed_evidence"]:
        grouped[row["model_key"]].append(row)

    for model_key, rows in grouped.items():
        summary = decision["model_summary"][model_key]
        negative_rates = [row["raw_negative_rate"] for row in rows]
        improvements = [
            100.0 * (row["raw_mae"] - row["clip_mae"]) / row["raw_mae"]
            for row in rows
        ]
        assert summary["mean_raw_negative_rate"] == pytest.approx(
            np.mean(negative_rates)
        )
        assert summary["raw_negative_rate_range"] == pytest.approx(
            [min(negative_rates), max(negative_rates)]
        )
        assert summary["mean_clip_mae_improvement_percent"] == pytest.approx(
            np.mean(improvements)
        )
        assert summary["clip_mae_improvement_range_percent"] == pytest.approx(
            [min(improvements), max(improvements)]
        )


def test_decision_keeps_runtime_guard_and_limits_scope_to_research() -> None:
    decision = _decision()

    assert decision["runtime_policy"] == {
        "current": "clip_zero",
        "decision": "KEEP_UNCHANGED",
        "reason": (
            "The current projection improves MAE for every model and seed, "
            "and remains the final demand-domain safety boundary."
        ),
    }
    assert decision["experiment_decision"]["run_training_experiment"] is True
    assert decision["experiment_decision"]["run_additional_posthoc_policy_sweep"] is False
    assert decision["experiment_decision"]["production_change_approved"] is False
    assert decision["pilot_contract"]["pilot_model"] == "exotst_base"
    assert decision["pilot_contract"]["pilot_seed"] == 42

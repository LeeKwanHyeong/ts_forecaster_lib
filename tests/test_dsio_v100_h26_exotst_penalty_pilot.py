"""Tests for the governed ExoTST negative-output penalty pilot gate."""

from __future__ import annotations

import pytest

from tools.run_dsio_v100_h26_exotst_penalty_pilot import evaluate_pilot_gate


def _rows(
    *,
    mae: float,
    negative_rate: float,
    normalized_bias: float,
) -> dict[str, dict[str, str]]:
    return {
        "raw": {
            "negative_prediction_rate": str(negative_rate),
        },
        "nonnegative": {
            "mae": str(mae),
            "normalized_bias": str(normalized_bias),
        },
    }


def test_pilot_gate_passes_only_when_all_thresholds_pass() -> None:
    result = evaluate_pilot_gate(
        control_rows=_rows(
            mae=1.25,
            negative_rate=0.24,
            normalized_bias=-0.01,
        ),
        candidate_rows=_rows(
            mae=1.20,
            negative_rate=0.10,
            normalized_bias=-0.015,
        ),
    )

    assert result["delta"]["clip_mae_improvement_percent"] == pytest.approx(4.0)
    assert result["delta"]["raw_negative_rate_reduction_percent"] == pytest.approx(
        100.0 * (0.24 - 0.10) / 0.24
    )
    assert result["passes_all_gates"] is True
    assert all(result["gates"].values())


@pytest.mark.parametrize(
    ("mae", "negative_rate", "normalized_bias", "failed_gate"),
    [
        (1.245, 0.10, -0.01, "mae_improvement"),
        (1.20, 0.13, -0.01, "negative_rate_reduction"),
        (1.20, 0.10, -0.025, "absolute_normalized_bias"),
    ],
)
def test_pilot_gate_rejects_each_failed_threshold(
    mae: float,
    negative_rate: float,
    normalized_bias: float,
    failed_gate: str,
) -> None:
    result = evaluate_pilot_gate(
        control_rows=_rows(
            mae=1.25,
            negative_rate=0.24,
            normalized_bias=-0.01,
        ),
        candidate_rows=_rows(
            mae=mae,
            negative_rate=negative_rate,
            normalized_bias=normalized_bias,
        ),
    )

    assert result["passes_all_gates"] is False
    assert result["gates"][failed_gate] is False

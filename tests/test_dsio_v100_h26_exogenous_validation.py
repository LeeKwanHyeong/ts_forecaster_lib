from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.evaluate_dsio_v100_h26_exogenous_qualification import (
    ValidationMetricAccumulator,
    _qualification_seed,
    evaluate_prediction_batches,
)
from tools.dsio_v100_h26_contract import V100H26ContractError


def test_validation_metrics_match_explicit_point_definitions():
    actual = np.asarray([[0.0, 2.0], [4.0, 0.0]])
    prediction = np.asarray([[-1.0, 4.0], [2.0, 0.0]])
    accumulator = ValidationMetricAccumulator(horizon=2)
    accumulator.update(actual, prediction)

    overall, horizons = accumulator.finalize()

    assert overall["forecast_points"] == 4
    assert overall["mae"] == pytest.approx(1.25)
    assert overall["wape"] == pytest.approx(5.0 / 6.0)
    assert overall["smape"] == pytest.approx(5.0 / 6.0)
    assert overall["bias"] == pytest.approx(-0.25)
    assert overall["normalized_bias"] == pytest.approx(-1.0 / 6.0)
    assert overall["negative_prediction_count"] == 1
    assert horizons[0]["horizon_label"] == "W0"
    assert horizons[1]["horizon_label"] == "W1"


def test_nonnegative_policy_can_be_compared_without_changing_actuals():
    actual = np.asarray([[0.0, 2.0], [4.0, 0.0]])
    raw = np.asarray([[-1.0, 4.0], [2.0, -3.0]])
    raw_metrics = ValidationMetricAccumulator(horizon=2)
    clipped_metrics = ValidationMetricAccumulator(horizon=2)
    raw_metrics.update(actual, raw)
    clipped_metrics.update(actual, np.maximum(raw, 0.0))

    raw_overall, _ = raw_metrics.finalize()
    clipped_overall, _ = clipped_metrics.finalize()

    assert raw_overall["mae"] == pytest.approx(2.0)
    assert clipped_overall["mae"] == pytest.approx(1.0)
    assert raw_overall["negative_prediction_count"] == 2
    assert clipped_overall["negative_prediction_count"] == 0
    assert raw_overall["actual_mean"] == clipped_overall["actual_mean"]


@pytest.mark.parametrize(
    ("actual", "prediction", "message"),
    [
        (
            np.zeros((2, 2)),
            np.zeros((2, 3)),
            "shape mismatch",
        ),
        (
            np.zeros((2, 2)),
            np.asarray([[0.0, np.nan], [0.0, 0.0]]),
            "finite",
        ),
    ],
)
def test_validation_metrics_reject_invalid_batches(actual, prediction, message):
    accumulator = ValidationMetricAccumulator(horizon=2)
    with pytest.raises(ValueError, match=message):
        accumulator.update(actual, prediction)


def test_streamed_batches_equal_one_shot_metrics():
    actual = np.arange(18, dtype=np.float64).reshape(3, 6)
    prediction = actual + np.asarray([[-1.0, 0.0, 1.0, 2.0, -2.0, 0.5]])
    streamed = ValidationMetricAccumulator(horizon=6)
    streamed.update(actual[:1], prediction[:1])
    streamed.update(actual[1:], prediction[1:])
    one_shot = ValidationMetricAccumulator(horizon=6)
    one_shot.update(actual, prediction)

    streamed_overall, streamed_horizons = streamed.finalize()
    one_shot_overall, one_shot_horizons = one_shot.finalize()
    assert streamed_overall.keys() == one_shot_overall.keys()
    for key in streamed_overall:
        assert streamed_overall[key] == pytest.approx(one_shot_overall[key])
    for streamed_row, one_shot_row in zip(
        streamed_horizons,
        one_shot_horizons,
        strict=True,
    ):
        assert streamed_row.keys() == one_shot_row.keys()
        for key in streamed_row:
            if key == "horizon_label":
                assert streamed_row[key] == one_shot_row[key]
            else:
                assert streamed_row[key] == pytest.approx(one_shot_row[key])


def test_public_predictor_batches_are_reshaped_and_counted():
    class FakePredictor:
        def predict(self, batch, *, horizon):
            assert horizon == 26
            return {"point": (batch[1] - 1.0).numpy().reshape(-1)}

    def batch(size):
        return (
            torch.zeros(size, 52, 1),
            torch.full((size, 26), 2.0),
            [f"P{index}" for index in range(size)],
            torch.zeros(size, 26, 12),
            torch.zeros(size, 52, 12),
            torch.zeros(size, 52, 0),
        )

    metrics, elapsed = evaluate_prediction_batches(
        predictor=FakePredictor(),
        loader=[batch(2), batch(1)],
        expected_series_count=3,
    )

    assert elapsed >= 0.0
    assert metrics["raw"]["overall"]["forecast_points"] == 78
    assert metrics["raw"]["overall"]["mae"] == pytest.approx(1.0)
    assert metrics["nonnegative"]["overall"]["mae"] == pytest.approx(1.0)


def test_qualification_seed_requires_one_seed_across_selected_models():
    receipt = {
        "seed": 11,
        "models": [
            {"training_contract": {"seed": 11}},
            {"training_contract": {"seed": 11}},
        ],
    }
    assert _qualification_seed(receipt) == 11

    receipt["models"][1]["training_contract"]["seed"] = 22
    with pytest.raises(V100H26ContractError, match="seeds disagree"):
        _qualification_seed(receipt)


def test_qualification_seed_rejects_aggregate_drift():
    with pytest.raises(V100H26ContractError, match="aggregate seed drifted"):
        _qualification_seed(
            {
                "seed": 22,
                "models": [{"training_contract": {"seed": 11}}],
            }
        )

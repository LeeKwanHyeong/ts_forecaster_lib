"""Contracts for ExoTST output-space negative regularization."""

from __future__ import annotations

import pytest
import torch

from modeling_module import load_predictor
from modeling_module.models.ExoTST.ExoTST import ExoTST
from modeling_module.models.ExoTST.configs import ExoTSTConfig
from modeling_module.training.model_trainers.exotst_train import (
    exotst_negative_output_penalty,
    make_exotst_negative_output_penalty,
)
from modeling_module.utils.checkpoint import save_model


def _tiny_config(**overrides) -> ExoTSTConfig:
    values = {
        "lookback": 6,
        "horizon": 2,
        "y_dim": 1,
        "exo_dim_past": 1,
        "exo_dim_future": 1,
        "use_past_exo": True,
        "use_future_exo": True,
        "exo_nan_policy": "zero",
        "patch_len": 2,
        "stride": 1,
        "d_model": 4,
        "n_heads": 2,
        "d_ff": 8,
        "dropout": 0.0,
        "attn_dropout": 0.0,
        "exo_enc_layers": 1,
        "fusion_layers": 1,
        "endo_dec_layers": 1,
        "exo_memory_mode": "agg",
        "use_revin": True,
        "subtract_last": True,
        "strict_shape": True,
    }
    values.update(overrides)
    return ExoTSTConfig(**values)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([
            [[2.0], [1.0], [3.0], [0.0], [2.0], [1.0]],
            [[1.0], [4.0], [2.0], [1.0], [0.0], [3.0]],
        ]),
        torch.randn(2, 6, 1),
        torch.randn(2, 2, 1),
    )


@pytest.mark.parametrize("value", [-0.1, float("inf"), float("nan")])
def test_penalty_weight_rejects_invalid_config(value: float) -> None:
    with pytest.raises(ValueError, match="finite and >= 0"):
        _tiny_config(negative_output_penalty_weight=value)


def test_zero_penalty_preserves_state_dict_and_output_exactly() -> None:
    torch.manual_seed(17)
    baseline = ExoTST(_tiny_config()).eval()
    torch.manual_seed(17)
    explicit_zero = ExoTST(
        _tiny_config(negative_output_penalty_weight=0.0)
    ).eval()
    x, past, future = _inputs()

    baseline_state = baseline.state_dict()
    explicit_state = explicit_zero.state_dict()
    assert baseline_state.keys() == explicit_state.keys()
    for key in baseline_state:
        torch.testing.assert_close(
            baseline_state[key],
            explicit_state[key],
            rtol=0.0,
            atol=0.0,
        )
    with torch.no_grad():
        baseline_output = baseline(
            x,
            past_exo_cont=past,
            future_exo=future,
        )
        explicit_output = explicit_zero(
            x,
            past_exo_cont=past,
            future_exo=future,
        )
    torch.testing.assert_close(
        baseline_output,
        explicit_output,
        rtol=0.0,
        atol=0.0,
    )
    assert make_exotst_negative_output_penalty(
        explicit_zero,
        loss_mode="point",
    ) is None


def test_negative_penalty_gradient_moves_only_negative_points_upward() -> None:
    prediction = torch.tensor(
        [[-2.0, -0.5, 0.0, 3.0]],
        requires_grad=True,
    )

    penalty = exotst_negative_output_penalty(prediction, weight=0.1)
    penalty.backward()

    assert penalty.item() == pytest.approx(0.10625)
    assert prediction.grad is not None
    assert prediction.grad[0, 0] < 0.0
    assert prediction.grad[0, 1] < 0.0
    assert prediction.grad[0, 2] == 0.0
    assert prediction.grad[0, 3] == 0.0
    updated = prediction.detach() - prediction.grad
    assert updated[0, 0] > prediction.detach()[0, 0]
    assert updated[0, 1] > prediction.detach()[0, 1]


def test_penalty_checkpoint_strict_load_preserves_config_and_prediction(
    tmp_path,
) -> None:
    torch.manual_seed(23)
    config = _tiny_config(negative_output_penalty_weight=0.1)
    model = ExoTST(config).eval()
    checkpoint_path = tmp_path / "tiny_exotst_penalty.pt"
    save_model(
        model,
        config,
        str(checkpoint_path),
        extra_meta={
            "model_key": "exotst_base",
            "family_key": "exotst",
        },
    )
    x, past, future = _inputs()
    with torch.no_grad():
        expected = model(
            x,
            past_exo_cont=past,
            future_exo=future,
        )

    predictor = load_predictor(
        str(checkpoint_path),
        device="cpu",
        strict=True,
    )
    restored = predictor.model.eval()
    with torch.no_grad():
        actual = restored(
            x,
            past_exo_cont=past,
            future_exo=future,
        )

    assert restored.cfg.negative_output_penalty_weight == 0.1
    torch.testing.assert_close(expected, actual, rtol=0.0, atol=0.0)


def test_penalty_is_point_only() -> None:
    model = ExoTST(_tiny_config(negative_output_penalty_weight=0.1))

    with pytest.raises(ValueError, match="point training"):
        make_exotst_negative_output_penalty(model, loss_mode="dist")

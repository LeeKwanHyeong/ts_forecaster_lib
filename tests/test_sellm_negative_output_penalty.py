"""Contracts for SELLM demand-space negative regularization."""

from __future__ import annotations

import pytest
import torch

from modeling_module import load_predictor
from modeling_module.models.SELLM.SELLM import SELLMModel
from modeling_module.models.SELLM.configs import SELLMConfig
from modeling_module.training.model_trainers.sellm_train import (
    make_sellm_negative_output_penalty,
    sellm_negative_output_penalty,
)
from modeling_module.utils.checkpoint import save_model


def _tiny_config(**overrides) -> SELLMConfig:
    values = {
        "lookback": 8,
        "horizon": 4,
        "y_dim": 1,
        "architecture_variant": "paper_v1",
        "token_len": 2,
        "d_model": 8,
        "n_heads": 2,
        "dropout": 0.0,
        "mlp_hidden_dim": 8,
        "semantic_vocab_size": 6,
        "semantic_top_k": 2,
        "tscc_latent_dim": 2,
        "tscc_hidden_dim": 4,
        "tscc_kl_weight": 0.0,
        "use_pretrained_llm": False,
        "fallback_layers": 1,
        "d_ff": 16,
        "use_norm": False,
        "final_nonneg": False,
    }
    values.update(overrides)
    return SELLMConfig(**values)


@pytest.mark.parametrize("value", [-0.1, float("inf"), float("nan")])
def test_penalty_weight_rejects_invalid_config(value: float) -> None:
    with pytest.raises(ValueError, match="finite and >= 0"):
        _tiny_config(negative_output_penalty_weight=value)


def test_zero_penalty_preserves_state_dict_and_output_exactly() -> None:
    torch.manual_seed(17)
    baseline = SELLMModel(_tiny_config()).eval()
    torch.manual_seed(17)
    explicit_zero = SELLMModel(
        _tiny_config(negative_output_penalty_weight=0.0)
    ).eval()
    value = torch.randn(2, 8, 1)

    for key, state in baseline.state_dict().items():
        torch.testing.assert_close(
            state,
            explicit_zero.state_dict()[key],
            rtol=0.0,
            atol=0.0,
        )
    with torch.no_grad():
        expected = baseline(value)
        actual = explicit_zero(value)
    torch.testing.assert_close(expected, actual, rtol=0.0, atol=0.0)
    assert make_sellm_negative_output_penalty(
        explicit_zero,
        loss_mode="point",
    ) is None


def test_negative_penalty_gradient_moves_only_negative_points_upward() -> None:
    prediction = torch.tensor(
        [[-2.0, -0.5, 0.0, 3.0]],
        requires_grad=True,
    )

    penalty = sellm_negative_output_penalty(prediction, weight=0.1)
    penalty.backward()

    assert penalty.item() == pytest.approx(0.10625)
    assert prediction.grad is not None
    assert prediction.grad[0, 0] < 0.0
    assert prediction.grad[0, 1] < 0.0
    assert prediction.grad[0, 2] == 0.0
    assert prediction.grad[0, 3] == 0.0
    assert torch.equal(
        prediction.grad[0, 2:],
        torch.zeros_like(prediction.grad[0, 2:]),
    )


def test_penalty_checkpoint_strict_load_preserves_config_and_prediction(
    tmp_path,
) -> None:
    torch.manual_seed(23)
    config = _tiny_config(negative_output_penalty_weight=0.1)
    model = SELLMModel(config).eval()
    checkpoint_path = tmp_path / "sellm_base.pt"
    save_model(
        model,
        config,
        str(checkpoint_path),
        extra_meta={"model_key": "sellm_base", "family": "sellm"},
    )
    value = torch.randn(2, 8, 1)
    with torch.no_grad():
        expected = model(value)

    predictor = load_predictor(str(checkpoint_path), device="cpu", strict=True)
    with torch.no_grad():
        actual = predictor.model(value)

    assert predictor.model.cfg.negative_output_penalty_weight == 0.1
    torch.testing.assert_close(expected, actual, rtol=0.0, atol=0.0)


def test_penalty_is_point_only() -> None:
    model = SELLMModel(_tiny_config(negative_output_penalty_weight=0.1))

    with pytest.raises(ValueError, match="point training"):
        make_sellm_negative_output_penalty(model, loss_mode="dist")

from __future__ import annotations

import pytest
import torch

from modeling_module.models.ExoTST.ExoTST import ExoTST
from modeling_module.models.ExoTST.configs import ExoTSTConfig


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
        "use_revin": False,
        "strict_shape": True,
    }
    values.update(overrides)
    return ExoTSTConfig(**values)


def test_exotst_point_output_and_gradient_contract():
    torch.manual_seed(7)
    model = ExoTST(_tiny_config())
    x = torch.randn(2, 6, 1, requires_grad=True)
    past = torch.randn(2, 6, 1, requires_grad=True)
    future = torch.randn(2, 2, 1, requires_grad=True)

    output = model(x, past_exo_cont=past, future_exo=future)
    output.square().mean().backward()

    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()
    for value in (x.grad, past.grad, future.grad):
        assert value is not None
        assert torch.isfinite(value).all()
        assert torch.count_nonzero(value) > 0
    parameter_gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert any(
        gradient is not None and torch.count_nonzero(gradient) > 0
        for gradient in parameter_gradients
    )
    assert all(
        gradient is None or torch.isfinite(gradient).all()
        for gradient in parameter_gradients
    )


@pytest.mark.parametrize("policy", ["zero", "zero+indicator"])
def test_exotst_nan_policy_produces_finite_output(policy):
    torch.manual_seed(11)
    model = ExoTST(_tiny_config(exo_nan_policy=policy)).eval()
    x = torch.randn(2, 6, 1)
    past = torch.randn(2, 6, 1)
    future = torch.randn(2, 2, 1)
    past[0, 2, 0] = float("nan")
    past[1, 4, 0] = float("inf")
    future[0, 1, 0] = float("nan")
    future[1, 0, 0] = float("-inf")

    with torch.no_grad():
        output = model(x, past_exo_cont=past, future_exo=future)

    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("past", "message"),
    [
        (torch.zeros(2, 6), "rank-3"),
        (torch.zeros(1, 6, 1), "batch mismatch"),
        (torch.zeros(2, 5, 1), "lookback mismatch"),
        (torch.zeros(2, 6, 2), "last dimension mismatch"),
    ],
)
def test_exotst_rejects_past_exogenous_shape_mismatch(past, message):
    model = ExoTST(_tiny_config()).eval()

    with pytest.raises(RuntimeError, match=rf"\[ExoTST\].*{message}"):
        model(
            torch.randn(2, 6, 1),
            past_exo_cont=past,
            future_exo=torch.randn(2, 2, 1),
        )


def test_exotst_rejects_categorical_past_exogenous_input():
    model = ExoTST(_tiny_config()).eval()

    with pytest.raises(RuntimeError, match="categorical past exogenous inputs are not supported"):
        model(
            torch.randn(2, 6, 1),
            past_exo_cont=torch.randn(2, 6, 1),
            past_exo_cat=torch.ones(2, 6, 1, dtype=torch.long),
            future_exo=torch.randn(2, 2, 1),
        )


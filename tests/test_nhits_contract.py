from __future__ import annotations

import pytest
import torch

from modeling_module.models.NHITS.NHITS import NHITSModel
from modeling_module.models.NHITS.configs import NHITSConfig
from modeling_module.models.model_builder import build_nhits


def _tiny_config(**overrides) -> NHITSConfig:
    values = {
        "lookback": 6,
        "horizon": 2,
        "stack_types": ("identity",),
        "n_blocks": (1,),
        "n_layers": (2,),
        "n_theta_hidden": ((8, 8),),
        "n_pool_kernel_size": (2,),
        "n_freq_downsample": (1,),
        "pooling_mode": "max",
        "interpolation_mode": "linear",
        "activation": "Softplus",
        "dropout_prob_theta": 0.0,
        "batch_normalization": False,
        "use_exogenous_mode": False,
    }
    values.update(overrides)
    return NHITSConfig(**values)


def test_nhits_public_wrapper_matches_backbone_forecast():
    torch.manual_seed(7)
    model = NHITSModel(_tiny_config()).eval()
    x = torch.tensor(
        [
            [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]],
            [[2.0], [1.0], [2.0], [1.0], [2.0], [1.0]],
        ]
    )

    actual = model(x)
    expected = model.backbone.forecast(
        insample_y=x[..., 0],
        insample_x_t=x.new_empty((2, 0, 6)),
        insample_mask=x.new_ones((2, 6)),
        outsample_x_t=x.new_empty((2, 0, 2)),
        x_s=x.new_empty((2, 0)),
    ).unsqueeze(-1)

    assert actual.shape == (2, 2, 1)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_nhits_state_dict_parameter_and_gradient_baseline():
    torch.manual_seed(11)
    model = build_nhits(_tiny_config())

    assert isinstance(model, NHITSModel)
    assert sum(parameter.numel() for parameter in model.parameters()) == 176
    assert list(model.state_dict()) == [
        "backbone.blocks.0.layers.0.weight",
        "backbone.blocks.0.layers.0.bias",
        "backbone.blocks.0.layers.2.weight",
        "backbone.blocks.0.layers.2.bias",
        "backbone.blocks.0.layers.4.weight",
        "backbone.blocks.0.layers.4.bias",
    ]

    x = torch.randn(3, 6, 1)
    loss = model(x).square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]

    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients if gradient is not None)


def test_nhits_builder_restores_mapping_config():
    model = build_nhits(
        {
            "lookback": 6,
            "horizon": 2,
            "stack_types": ["identity"],
            "n_blocks": [1],
            "n_layers": [2],
            "n_theta_hidden": [[8, 8]],
            "n_pool_kernel_size": [2],
            "n_freq_downsample": [1],
            "activation": "Softplus",
            "use_exogenous_mode": False,
        }
    )

    assert isinstance(model.cfg, NHITSConfig)
    assert model.cfg.stack_types == ("identity",)
    assert model.cfg.n_theta_hidden == ((8, 8),)


@pytest.mark.parametrize(
    ("input_value", "message"),
    [
        (torch.ones(2, 5, 1), "lookback mismatch"),
        (torch.ones(2, 6, 2), "channel mismatch"),
        (torch.ones(2, 6), "expects x with shape"),
    ],
)
def test_nhits_rejects_invalid_target_shapes(input_value, message):
    model = NHITSModel(_tiny_config())

    with pytest.raises(ValueError, match=message):
        model(input_value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("future_exo", torch.ones(2, 2, 1)),
        ("past_exo_cont", torch.ones(2, 6, 1)),
        ("past_exo_cat", torch.ones(2, 6, 1, dtype=torch.long)),
    ],
)
def test_nhits_rejects_nonempty_exogenous_inputs(field, value):
    model = NHITSModel(_tiny_config())

    with pytest.raises(RuntimeError, match=rf"endogenous-only.*{field}"):
        model(torch.ones(2, 6, 1), **{field: value})


def test_nhits_config_rejects_inconsistent_stack_contract():
    with pytest.raises(ValueError, match="n_blocks must contain one value per stack"):
        _tiny_config(
            stack_types=("identity", "identity"),
            n_blocks=(1,),
            n_layers=(2, 2),
            n_theta_hidden=((8, 8), (8, 8)),
            n_pool_kernel_size=(2, 1),
            n_freq_downsample=(2, 1),
        )


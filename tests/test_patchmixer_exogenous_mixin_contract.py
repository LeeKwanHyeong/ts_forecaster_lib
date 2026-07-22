from __future__ import annotations

import hashlib

import pytest
import torch

from modeling_module.models.PatchMixer.common.configs import PatchMixerConfig
from modeling_module.models.PatchMixer.variants import (
    PatchMixerEndogenousModel,
    PatchMixerExogenousModel,
    PatchMixerQuantileEndogenousModel,
    PatchMixerQuantileExogenousModel,
)


EXO_PARAMETER_PREFIXES = (
    "exo_head.",
    "_cat_embs.",
    "_z_exo_proj.",
    "_z_gate.",
)


def _config(
    *,
    exogenous: bool,
    distribution: bool = False,
    future_exo_shift_space: str = "output",
    future_exo_normalized_residual_limit: float | None = None,
    use_revin: bool = False,
) -> PatchMixerConfig:
    kwargs = {}
    if exogenous:
        kwargs = {
            "past_exo_cont_dim": 2,
            "past_exo_cat_dim": 2,
            "past_exo_cat_vocab_sizes": (5, 7),
            "past_exo_cat_embed_dims": (3, 4),
            "future_exo_dim": 2,
        }
    if distribution:
        kwargs.update(out_mul=2, param_names=["loc", "scale"])
    return PatchMixerConfig(
        device="cpu",
        lookback=8,
        horizon=2,
        enc_in=1,
        patch_len=4,
        stride=2,
        mixer_kernel_size=3,
        d_model=8,
        e_layers=1,
        dropout=0.0,
        head_dropout=0.0,
        f_out=8,
        head_hidden=8,
        use_revin=use_revin,
        final_nonneg=False,
        past_exo_mode="z_gate",
        future_exo_shift_space=future_exo_shift_space,
        future_exo_normalized_residual_limit=(
            future_exo_normalized_residual_limit
        ),
        patch_cfgs=((4, 2, 3),),
        per_branch_dim=4,
        fused_dim=8,
        quantiles=(0.1, 0.5, 0.9),
        **kwargs,
    )


def _future_only_config(
    *,
    distribution: bool = False,
    future_exo_shift_space: str = "output",
    future_exo_normalized_residual_limit: float | None = None,
    use_revin: bool = True,
) -> PatchMixerConfig:
    return PatchMixerConfig(
        device="cpu",
        lookback=8,
        horizon=2,
        enc_in=1,
        patch_len=4,
        stride=2,
        mixer_kernel_size=3,
        d_model=8,
        e_layers=1,
        dropout=0.0,
        head_dropout=0.0,
        f_out=8,
        head_hidden=8,
        use_revin=use_revin,
        final_nonneg=False,
        past_exo_mode="none",
        future_exo_dim=2,
        future_exo_shift_space=future_exo_shift_space,
        future_exo_normalized_residual_limit=(
            future_exo_normalized_residual_limit
        ),
        patch_cfgs=((4, 2, 3),),
        per_branch_dim=4,
        fused_dim=8,
        quantiles=(0.1, 0.5, 0.9),
        q_clip_norm=None,
        out_mul=2 if distribution else 1,
        param_names=["loc", "scale"] if distribution else None,
    )


def _future_distribution_config(
    *,
    future_exo_shift_space: str,
    param_names: tuple[str, ...],
) -> PatchMixerConfig:
    config = _future_only_config(
        distribution=True,
        future_exo_shift_space=future_exo_shift_space,
    )
    config.out_mul = len(param_names)
    config.param_names = list(param_names)
    return config


def _inputs(*, requires_grad: bool = False):
    x = torch.linspace(-1.0, 1.0, steps=16).reshape(2, 8, 1)
    past_cont = torch.linspace(-0.5, 0.75, steps=32).reshape(2, 8, 2)
    future = torch.linspace(-0.25, 0.5, steps=8).reshape(2, 2, 2)
    if requires_grad:
        past_cont.requires_grad_()
        future.requires_grad_()
    ids = torch.arange(16).reshape(2, 8)
    past_cat = torch.stack((ids % 5, ids % 7), dim=-1)
    return x, past_cont, past_cat, future


def _is_exogenous_parameter(name: str) -> bool:
    return name.startswith(EXO_PARAMETER_PREFIXES)


def _state_schema_digest(model: torch.nn.Module) -> str:
    schema = "\n".join(
        f"{key}:{tuple(value.shape)}:{value.dtype}"
        for key, value in model.state_dict().items()
    )
    return hashlib.sha256(schema.encode()).hexdigest()


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@pytest.mark.parametrize(
    (
        "endogenous_cls",
        "exogenous_cls",
        "distribution",
        "endogenous_parameters",
        "exogenous_parameters",
        "exo_parameters",
        "endogenous_state_keys",
        "exogenous_state_keys",
        "endogenous_schema",
        "exogenous_schema",
        "z_dim",
    ),
    (
        (
            PatchMixerEndogenousModel,
            PatchMixerExogenousModel,
            False,
            12_323,
            13_999,
            1_676,
            45,
            55,
            "51799c04ada71a818e9bfda1a478c4a2e838715e5b1f8699c590478a1c15df38",
            "2a03ca462ea4c458bfc0bb38ab0b10a4b4489b598bae077464c9e2359956179b",
            32,
        ),
        (
            PatchMixerQuantileEndogenousModel,
            PatchMixerQuantileExogenousModel,
            False,
            2_372,
            2_824,
            452,
            53,
            63,
            "7ac31783cb9c8e17c0014c0a9afa9492b99a1ae96c4eda6cd8b1383f7192a546",
            "a7a6fc36972a687c3beea01d5f96c0b19cb4e46eb26ba5174de6c22c98be5ecc",
            8,
        ),
        (
            PatchMixerEndogenousModel,
            PatchMixerExogenousModel,
            True,
            12_332,
            14_008,
            1_676,
            45,
            55,
            "a70314301a7ff72b06d86e4be3a6241921c6dd94ecb9e2be1a634d9acd437b28",
            "f0d7bcc96497fc4df9097895d29c341626e0f3f0251c21a094f62bbbf33b54f1",
            32,
        ),
    ),
)
def test_patchmixer_exogenous_state_dict_contract(
    endogenous_cls,
    exogenous_cls,
    distribution,
    endogenous_parameters,
    exogenous_parameters,
    exo_parameters,
    endogenous_state_keys,
    exogenous_state_keys,
    endogenous_schema,
    exogenous_schema,
    z_dim,
) -> None:
    endogenous = endogenous_cls(
        _config(exogenous=False, distribution=distribution)
    )
    exogenous = exogenous_cls(
        _config(exogenous=True, distribution=distribution)
    )

    endogenous_exo_state = {
        key: value
        for key, value in endogenous.state_dict().items()
        if _is_exogenous_parameter(key)
    }
    exogenous_exo_state = {
        key: tuple(value.shape)
        for key, value in exogenous.state_dict().items()
        if _is_exogenous_parameter(key)
    }
    expected_exo_state = {
        "exo_head.0.weight": (64, 2),
        "exo_head.0.bias": (64,),
        "exo_head.2.weight": (1, 64),
        "exo_head.2.bias": (1,),
        "_cat_embs.0.weight": (5, 3),
        "_cat_embs.1.weight": (7, 4),
        "_z_exo_proj.weight": (z_dim, 9),
        "_z_exo_proj.bias": (z_dim,),
        "_z_gate.weight": (z_dim, z_dim),
        "_z_gate.bias": (z_dim,),
    }
    actual_exo_parameters = sum(
        parameter.numel()
        for name, parameter in exogenous.named_parameters()
        if _is_exogenous_parameter(name)
    )

    assert endogenous_exo_state == {}
    assert exogenous_exo_state == expected_exo_state
    assert _parameter_count(endogenous) == endogenous_parameters
    assert _parameter_count(exogenous) == exogenous_parameters
    assert actual_exo_parameters == exo_parameters
    assert exogenous_parameters - endogenous_parameters == exo_parameters
    assert len(endogenous.state_dict()) == endogenous_state_keys
    assert len(exogenous.state_dict()) == exogenous_state_keys
    assert _state_schema_digest(endogenous) == endogenous_schema
    assert _state_schema_digest(exogenous) == exogenous_schema


@pytest.mark.parametrize(
    ("model_cls", "distribution", "seed", "expected"),
    (
        (
            PatchMixerExogenousModel,
            False,
            20260727,
            torch.tensor(
                [
                    [0.13809253, 0.12295340],
                    [1.09814000, 1.11265635],
                ]
            ),
        ),
        (
            PatchMixerQuantileExogenousModel,
            False,
            20260728,
            torch.tensor(
                [
                    [
                        [-0.71134585, -0.52838457],
                        [0.40524918, 0.58821046],
                        [1.52184415, 1.70480561],
                    ],
                    [
                        [0.33526585, 0.47948849],
                        [1.45186090, 1.59608352],
                        [2.56845593, 2.71267843],
                    ],
                ]
            ),
        ),
        (
            PatchMixerExogenousModel,
            True,
            20260730,
            torch.tensor(
                [
                    [
                        [-0.80982119, -0.59448278],
                        [-0.82483351, -0.58252615],
                    ],
                    [
                        [0.12566078, -0.70227438],
                        [-0.10492589, -0.71278965],
                    ],
                ]
            ),
        ),
    ),
)
def test_patchmixer_exogenous_output_baseline(
    model_cls,
    distribution,
    seed,
    expected,
) -> None:
    torch.manual_seed(seed)
    model = model_cls(
        _config(exogenous=True, distribution=distribution)
    ).eval()
    x, past_cont, past_cat, future = _inputs()

    with torch.no_grad():
        output = model(
            x,
            past_exo_cont=past_cont,
            past_exo_cat=past_cat,
            future_exo=future,
        )

    tensor = output["q"] if isinstance(output, dict) else output
    torch.testing.assert_close(tensor, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("model_cls", "distribution"),
    (
        (PatchMixerExogenousModel, False),
        (PatchMixerQuantileExogenousModel, False),
        (PatchMixerExogenousModel, True),
    ),
)
def test_patchmixer_future_shift_is_in_raw_output_space(
    model_cls,
    distribution: bool,
) -> None:
    torch.manual_seed(20260807)
    model = model_cls(_future_only_config(distribution=distribution)).eval()
    assert model.future_exo_shift_space == "output"
    assert all("shift_space" not in key for key in model.state_dict())
    x = torch.linspace(-1.0, 1.0, steps=16).reshape(2, 8, 1)
    x_large_scale = 1000.0 + 250.0 * x
    future = torch.linspace(-0.25, 0.5, steps=8).reshape(2, 2, 2)
    zero_future = torch.zeros_like(future)

    def output_tensor(inputs: torch.Tensor, exogenous: torch.Tensor) -> torch.Tensor:
        output = model(inputs, future_exo=exogenous)
        return output["q"] if isinstance(output, dict) else output

    with torch.no_grad():
        expected = (
            model.exo_head(future).squeeze(-1)
            - model.exo_head(zero_future).squeeze(-1)
        )
        unit_effect = output_tensor(x, future) - output_tensor(x, zero_future)
        large_effect = output_tensor(x_large_scale, future) - output_tensor(
            x_large_scale,
            zero_future,
        )

    if model_cls is PatchMixerQuantileExogenousModel:
        expected = expected.unsqueeze(1).expand_as(unit_effect)
    elif distribution:
        torch.testing.assert_close(
            unit_effect[..., 1],
            torch.zeros_like(unit_effect[..., 1]),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            large_effect[..., 1],
            torch.zeros_like(large_effect[..., 1]),
            rtol=0.0,
            atol=0.0,
        )
        unit_effect = unit_effect[..., 0]
        large_effect = large_effect[..., 0]

    torch.testing.assert_close(unit_effect, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(large_effect, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    ("param_names", "loc_idx"),
    (
        (("-loc", "-scale"), 0),
        (("-df", "-loc", "-scale"), 1),
    ),
)
def test_patchmixer_distribution_output_shift_changes_only_loc(
    param_names: tuple[str, ...],
    loc_idx: int,
) -> None:
    torch.manual_seed(20260810)
    model = PatchMixerExogenousModel(
        _future_distribution_config(
            future_exo_shift_space="output",
            param_names=param_names,
        )
    ).eval()
    x = torch.linspace(-1.0, 1.0, steps=16).reshape(2, 8, 1)
    future = torch.linspace(-0.25, 0.5, steps=8).reshape(2, 2, 2)
    zero_future = torch.zeros_like(future)

    with torch.no_grad():
        expected_loc_effect = (
            model.exo_head(future).squeeze(-1)
            - model.exo_head(zero_future).squeeze(-1)
        )
        actual_effect = model(x, future_exo=future) - model(
            x,
            future_exo=zero_future,
        )

    assert model.loc_idx == loc_idx
    torch.testing.assert_close(
        actual_effect[..., loc_idx],
        expected_loc_effect,
        rtol=1e-5,
        atol=1e-5,
    )
    for parameter_idx in set(range(len(param_names))) - {loc_idx}:
        torch.testing.assert_close(
            actual_effect[..., parameter_idx],
            torch.zeros_like(actual_effect[..., parameter_idx]),
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize(
    ("param_names", "loc_idx"),
    (
        (("-loc", "-scale"), 0),
        (("-df", "-loc", "-scale"), 1),
    ),
)
def test_patchmixer_distribution_normalized_shift_changes_only_loc(
    param_names: tuple[str, ...],
    loc_idx: int,
) -> None:
    torch.manual_seed(20260811)
    model = PatchMixerExogenousModel(
        _future_distribution_config(
            future_exo_shift_space="normalized",
            param_names=param_names,
        )
    ).eval()
    pattern = torch.linspace(-1.0, 1.0, steps=8)
    x = torch.stack((pattern, 1000.0 + 250.0 * pattern)).unsqueeze(-1)
    future = torch.linspace(-0.25, 0.5, steps=8).reshape(2, 2, 2)
    zero_future = torch.zeros_like(future)

    with torch.no_grad():
        normalized_residual = (
            model.exo_head(future).squeeze(-1)
            - model.exo_head(zero_future).squeeze(-1)
        )
        actual_effect = model(x, future_exo=future) - model(
            x,
            future_exo=zero_future,
        )

    target_std = torch.sqrt(
        x.var(dim=1, unbiased=False) + model.revin_layer.eps
    )
    expected_loc_effect = normalized_residual * target_std
    assert model.loc_idx == loc_idx
    torch.testing.assert_close(
        actual_effect[..., loc_idx],
        expected_loc_effect,
        rtol=1e-4,
        atol=1e-4,
    )
    for parameter_idx in set(range(len(param_names))) - {loc_idx}:
        torch.testing.assert_close(
            actual_effect[..., parameter_idx],
            torch.zeros_like(actual_effect[..., parameter_idx]),
            rtol=0.0,
            atol=0.0,
        )


def test_patchmixer_normalized_shift_coordinate_maps_through_target_revin_scale() -> None:
    model = PatchMixerExogenousModel(_future_only_config()).eval()
    revin = model.revin_layer

    assert revin.affine is False
    assert revin.subtract_last is True
    assert revin.use_std is True

    x = torch.tensor(
        [
            [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]],
            [[100.0], [150.0], [200.0], [250.0], [300.0], [350.0], [400.0], [450.0]],
        ]
    )
    normalized_forecast = torch.tensor(
        [[[0.2], [-0.4]], [[0.2], [-0.4]]]
    )
    normalized_shift = torch.tensor(
        [[[0.5], [-0.25]], [[0.5], [-0.25]]]
    )

    revin(x, "norm")
    raw_forecast = revin(normalized_forecast, "denorm")
    raw_shifted = revin(normalized_forecast + normalized_shift, "denorm")
    expected_effect = normalized_shift * revin.std
    scale_only_effect = revin.denorm_scale(normalized_shift)

    torch.testing.assert_close(
        raw_shifted - raw_forecast,
        expected_effect,
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        scale_only_effect,
        expected_effect,
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.allclose(
        raw_shifted[1] - raw_forecast[1],
        normalized_shift[1],
    )


@pytest.mark.parametrize(
    "model_cls",
    (PatchMixerExogenousModel, PatchMixerQuantileExogenousModel),
)
def test_patchmixer_normalized_future_shift_scales_with_target_history_std(
    model_cls,
) -> None:
    torch.manual_seed(20260808)
    model = model_cls(
        _future_only_config(future_exo_shift_space="normalized")
    ).eval()
    assert model.future_exo_shift_space == "normalized"

    pattern = torch.linspace(-1.0, 1.0, steps=8)
    x = torch.stack((pattern, 1000.0 + 250.0 * pattern)).unsqueeze(-1)
    future = torch.linspace(-0.25, 0.5, steps=8).reshape(2, 2, 2)
    zero_future = torch.zeros_like(future)

    def output_tensor(exogenous: torch.Tensor) -> torch.Tensor:
        output = model(x, future_exo=exogenous)
        return output["q"] if isinstance(output, dict) else output

    with torch.no_grad():
        normalized_residual = model.exo_scale * (
            model.exo_head(future).squeeze(-1)
            - model.exo_head(zero_future).squeeze(-1)
        )
        actual_effect = output_tensor(future) - output_tensor(zero_future)

    target_std = torch.sqrt(
        x.var(dim=1, unbiased=False) + model.revin_layer.eps
    )
    expected_effect = normalized_residual * target_std
    if model_cls is PatchMixerQuantileExogenousModel:
        expected_effect = expected_effect.unsqueeze(1).expand_as(actual_effect)

    torch.testing.assert_close(
        actual_effect,
        expected_effect,
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize(
    ("model_cls", "distribution"),
    (
        (PatchMixerExogenousModel, False),
        (PatchMixerQuantileExogenousModel, False),
        (PatchMixerExogenousModel, True),
    ),
)
def test_patchmixer_normalized_residual_soft_limit_bounds_shift_and_keeps_gradients(
    model_cls,
    distribution: bool,
) -> None:
    limit = 0.15
    model = model_cls(
        _future_only_config(
            distribution=distribution,
            future_exo_shift_space="normalized",
            future_exo_normalized_residual_limit=limit,
        )
    ).train()
    selector = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        selector.weight.zero_()
        selector.weight[0, 0] = 1.0
    model.exo_head = selector

    pattern = torch.linspace(-1.0, 1.0, steps=8)
    x = torch.stack((pattern, 1000.0 + 250.0 * pattern)).unsqueeze(-1)
    future = torch.tensor(
        [
            [[0.05, 0.0], [0.30, 0.0]],
            [[-0.30, 0.0], [1.00, 0.0]],
        ],
        requires_grad=True,
    )
    zero_future = torch.zeros_like(future)

    def output_tensor(exogenous: torch.Tensor) -> torch.Tensor:
        output = model(x, future_exo=exogenous)
        return output["q"] if isinstance(output, dict) else output

    baseline = output_tensor(zero_future)
    shifted = output_tensor(future)
    actual_effect = shifted - baseline
    target_std = torch.sqrt(
        x.var(dim=1, unbiased=False) + model.revin_layer.eps
    )
    bounded_residual = limit * torch.tanh(future[..., 0] / limit)
    expected_effect = bounded_residual * target_std

    if model_cls is PatchMixerQuantileExogenousModel:
        expected_effect = expected_effect.unsqueeze(1).expand_as(actual_effect)
    elif distribution:
        for parameter_idx in range(actual_effect.shape[-1]):
            if parameter_idx != model.loc_idx:
                torch.testing.assert_close(
                    actual_effect[..., parameter_idx],
                    torch.zeros_like(actual_effect[..., parameter_idx]),
                    rtol=0.0,
                    atol=0.0,
                )
        actual_effect = actual_effect[..., model.loc_idx]

    torch.testing.assert_close(
        actual_effect,
        expected_effect,
        rtol=1e-4,
        atol=1e-4,
    )
    shifted.square().mean().backward()
    assert future.grad is not None
    assert torch.isfinite(future.grad).all()
    assert torch.count_nonzero(future.grad) > 0


def test_patchmixer_output_shift_does_not_enter_normalized_soft_limit_path() -> None:
    torch.manual_seed(20260813)
    model = PatchMixerExogenousModel(
        _future_only_config(future_exo_shift_space="output")
    ).eval()
    assert model.future_exo_normalized_residual_limit is None
    x, _, _, future = _inputs()
    with torch.no_grad():
        expected = model(x, future_exo=future)

    def fail_if_called(_residual: torch.Tensor) -> torch.Tensor:
        raise AssertionError("output-space shift entered normalized limit path")

    model._bound_normalized_future_exo_residual = fail_if_called
    with torch.no_grad():
        actual = model(x, future_exo=future)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_patchmixer_quantile_normalized_shift_is_applied_before_eval_clip() -> None:
    model = PatchMixerQuantileExogenousModel(
        _future_only_config(future_exo_shift_space="normalized")
    ).eval()
    model.q_clip_eval = 0.05
    selector = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        selector.weight.zero_()
        selector.weight[0, 0] = 1.0
    model.exo_head = selector

    pattern = torch.linspace(-1.0, 1.0, steps=8)
    x = torch.stack((pattern, 1000.0 + 250.0 * pattern)).unsqueeze(-1)
    zero_future = torch.zeros((2, 2, 2))
    shifted_future = zero_future.clone()
    shifted_future[..., 0] = 100.0

    with torch.no_grad():
        baseline = model(x, future_exo=zero_future)["q"]
        shifted = model(x, future_exo=shifted_future)["q"]

    target_std = torch.sqrt(
        x.var(dim=1, unbiased=False) + model.revin_layer.eps
    )
    max_clipped_effect = (2.0 * model.q_clip_eval * target_std).unsqueeze(1)
    actual_effect = (shifted - baseline).abs()

    assert torch.count_nonzero(actual_effect) > 0
    assert torch.all(actual_effect <= max_clipped_effect + 2e-4)


@pytest.mark.parametrize(
    ("model_cls", "distribution"),
    (
        (PatchMixerExogenousModel, False),
        (PatchMixerQuantileExogenousModel, False),
        (PatchMixerExogenousModel, True),
    ),
)
def test_patchmixer_shift_spaces_are_exactly_equal_without_revin(
    model_cls,
    distribution: bool,
) -> None:
    torch.manual_seed(20260809)
    output_model = model_cls(
        _future_only_config(
            distribution=distribution,
            future_exo_shift_space="output",
            use_revin=False,
        )
    ).eval()
    normalized_model = model_cls(
        _future_only_config(
            distribution=distribution,
            future_exo_shift_space="normalized",
            use_revin=False,
        )
    ).eval()
    normalized_model.load_state_dict(output_model.state_dict(), strict=True)
    x, _, _, future = _inputs()

    with torch.no_grad():
        output = output_model(x, future_exo=future)
        normalized = normalized_model(x, future_exo=future)

    output_tensor = output["q"] if isinstance(output, dict) else output
    normalized_tensor = normalized["q"] if isinstance(normalized, dict) else normalized
    torch.testing.assert_close(
        normalized_tensor,
        output_tensor,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("model_cls", "distribution", "future_exo_shift_space", "use_revin"),
    (
        (PatchMixerExogenousModel, False, "output", False),
        (PatchMixerQuantileExogenousModel, False, "output", False),
        (PatchMixerExogenousModel, True, "output", False),
        (PatchMixerExogenousModel, False, "normalized", True),
        (PatchMixerQuantileExogenousModel, False, "normalized", True),
        (PatchMixerExogenousModel, True, "normalized", True),
    ),
)
def test_patchmixer_exogenous_parameters_and_inputs_receive_gradients(
    model_cls,
    distribution,
    future_exo_shift_space,
    use_revin,
) -> None:
    torch.manual_seed(20260729)
    model = model_cls(
        _config(
            exogenous=True,
            distribution=distribution,
            future_exo_shift_space=future_exo_shift_space,
            use_revin=use_revin,
        )
    ).train()
    x, past_cont, past_cat, future = _inputs(requires_grad=True)

    output = model(
        x,
        past_exo_cont=past_cont,
        past_exo_cat=past_cat,
        future_exo=future,
    )
    tensor = output["q"] if isinstance(output, dict) else output
    tensor.square().mean().backward()

    exogenous_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if _is_exogenous_parameter(name)
    }
    assert set(exogenous_parameters) == {
        "exo_head.0.weight",
        "exo_head.0.bias",
        "exo_head.2.weight",
        "exo_head.2.bias",
        "_cat_embs.0.weight",
        "_cat_embs.1.weight",
        "_z_exo_proj.weight",
        "_z_exo_proj.bias",
        "_z_gate.weight",
        "_z_gate.bias",
    }
    for name, parameter in exogenous_parameters.items():
        assert parameter.grad is not None, f"missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient: {name}"
        assert torch.count_nonzero(parameter.grad) > 0, f"zero gradient: {name}"

    assert past_cont.grad is not None
    assert future.grad is not None
    assert torch.count_nonzero(past_cont.grad) > 0
    assert torch.count_nonzero(future.grad) > 0

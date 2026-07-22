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


def _config(*, exogenous: bool, distribution: bool = False) -> PatchMixerConfig:
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
        use_revin=False,
        final_nonneg=False,
        past_exo_mode="z_gate",
        patch_cfgs=((4, 2, 3),),
        per_branch_dim=4,
        fused_dim=8,
        quantiles=(0.1, 0.5, 0.9),
        **kwargs,
    )


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
def test_patchmixer_exogenous_parameters_and_inputs_receive_gradients(
    model_cls,
    distribution,
) -> None:
    torch.manual_seed(20260729)
    model = model_cls(
        _config(exogenous=True, distribution=distribution)
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

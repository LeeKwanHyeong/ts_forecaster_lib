from __future__ import annotations

import hashlib

import pytest
import torch

from modeling_module.models.PatchMixer.PatchMixer import (
    _PatchMixerLegacyModel,
    _PatchMixerProjectCore,
)
from modeling_module.models.PatchMixer.common.configs import PatchMixerExogenousConfig
from modeling_module.models.PatchMixer.variants import PatchMixerExogenousModel


EXO_PARAMETER_PREFIXES = (
    "exo_head.",
    "_cat_embs.",
    "_z_exo_proj.",
    "_z_gate.",
)


def _config(
    *,
    future_exo_shift_space: str = "output",
    use_revin: bool = False,
    residual_limit: float | None = None,
) -> PatchMixerExogenousConfig:
    return PatchMixerExogenousConfig(
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
        past_exo_cont_dim=2,
        past_exo_cat_dim=2,
        past_exo_cat_vocab_sizes=(5, 7),
        past_exo_cat_embed_dims=(3, 4),
        future_exo_dim=2,
        future_exo_shift_space=future_exo_shift_space,
        future_exo_normalized_residual_limit=residual_limit,
    )


def _inputs(*, requires_grad: bool = False):
    x = torch.linspace(-1.0, 1.0, steps=16).reshape(2, 8, 1)
    past_cont = torch.linspace(-0.5, 0.75, steps=32).reshape(2, 8, 2)
    future = torch.linspace(-0.25, 0.5, steps=8).reshape(2, 2, 2)
    if requires_grad:
        x.requires_grad_()
        past_cont.requires_grad_()
        future.requires_grad_()
    ids = torch.arange(16).reshape(2, 8)
    past_cat = torch.stack((ids % 5, ids % 7), dim=-1)
    return x, past_cont, past_cat, future


def _state_schema_digest(model: torch.nn.Module) -> str:
    schema = "\n".join(
        f"{key}:{tuple(value.shape)}:{value.dtype}"
        for key, value in model.state_dict().items()
    )
    return hashlib.sha256(schema.encode()).hexdigest()


def test_patchmixer_exogenous_is_independent_of_retired_enhanced_identity() -> None:
    assert PatchMixerExogenousModel.__bases__ == (_PatchMixerProjectCore,)
    assert not issubclass(PatchMixerExogenousModel, _PatchMixerLegacyModel)


def test_patchmixer_exogenous_state_dict_contract() -> None:
    model = PatchMixerExogenousModel(_config())
    exogenous_state = {
        key: tuple(value.shape)
        for key, value in model.state_dict().items()
        if key.startswith(EXO_PARAMETER_PREFIXES)
    }

    assert exogenous_state == {
        "exo_head.0.weight": (64, 2),
        "exo_head.0.bias": (64,),
        "exo_head.2.weight": (1, 64),
        "exo_head.2.bias": (1,),
        "_cat_embs.0.weight": (5, 3),
        "_cat_embs.1.weight": (7, 4),
        "_z_exo_proj.weight": (32, 9),
        "_z_exo_proj.bias": (32,),
        "_z_gate.weight": (32, 32),
        "_z_gate.bias": (32,),
    }
    assert len(model.state_dict()) == 50
    assert sum(parameter.numel() for parameter in model.parameters()) == 13_992
    assert _state_schema_digest(model) == (
        "a65b168fabdbe45764e28d9d811b67f727eae4da4b4791379b1c8c86ea1f2090"
    )


def test_patchmixer_exogenous_output_baseline() -> None:
    torch.manual_seed(20260727)
    model = PatchMixerExogenousModel(_config()).eval()
    x, past_cont, past_cat, future = _inputs()

    with torch.no_grad():
        output = model(
            x,
            past_exo_cont=past_cont,
            past_exo_cat=past_cat,
            future_exo=future,
        )

    expected = torch.tensor(
        [[0.13809253, 0.12295340], [1.09814000, 1.11265635]]
    )
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("shift_space", "use_revin", "residual_limit"),
    (("output", False, None), ("normalized", True, None), ("normalized", True, 0.15)),
)
def test_patchmixer_exogenous_parameters_and_inputs_receive_gradients(
    shift_space: str,
    use_revin: bool,
    residual_limit: float | None,
) -> None:
    torch.manual_seed(20260729)
    model = PatchMixerExogenousModel(
        _config(
            future_exo_shift_space=shift_space,
            use_revin=use_revin,
            residual_limit=residual_limit,
        )
    ).train()
    x, past_cont, past_cat, future = _inputs(requires_grad=True)

    output = model(
        x,
        past_exo_cont=past_cont,
        past_exo_cat=past_cat,
        future_exo=future,
    )
    output.square().mean().backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient: {name}"
        assert torch.count_nonzero(parameter.grad) > 0, f"zero gradient: {name}"
    for tensor in (x, past_cont, future):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) > 0


def test_patchmixer_exogenous_requires_configured_and_supplied_features() -> None:
    with pytest.raises(ValueError, match="requires at least one"):
        PatchMixerExogenousModel(
            PatchMixerExogenousConfig(
                lookback=8,
                horizon=2,
                patch_len=4,
                stride=2,
            )
        )

    model = PatchMixerExogenousModel(_config()).eval()
    x, _, _, _ = _inputs()
    with pytest.raises(RuntimeError, match="future_exo.*required|missing required inputs"):
        model(x)


def test_patchmixer_exogenous_rejects_distribution_output() -> None:
    config = _config()
    config.out_mul = 2
    config.param_names = ["loc", "scale"]
    with pytest.raises(ValueError, match="point output only"):
        PatchMixerExogenousModel(config)


def test_patchmixer_exogenous_strict_load_drops_retired_point_state() -> None:
    torch.manual_seed(20260730)
    legacy_model = _PatchMixerProjectCore(_config()).eval()
    legacy_state = legacy_model.state_dict()
    assert set(_RETIRED_KEYS := {
        "out_scale",
        "out_bias",
        "dw_gain",
        "dw_head.weight",
        "dw_head.bias",
    }).issubset(legacy_state)

    torch.manual_seed(20260730)
    model = PatchMixerExogenousModel(_config()).eval()
    model.load_state_dict(legacy_state, strict=True)
    assert _RETIRED_KEYS.isdisjoint(model.state_dict())

    x, past_cont, past_cat, future = _inputs()
    with torch.no_grad():
        expected = legacy_model(
            x,
            past_exo_cont=past_cont,
            past_exo_cat=past_cat,
            future_exo=future,
        )
        actual = model(
            x,
            past_exo_cont=past_cont,
            past_exo_cat=past_cat,
            future_exo=future,
        )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

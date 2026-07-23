from __future__ import annotations

import pytest
import torch

from modeling_module.models.PatchMixer.PatchMixer import (
    PatchMixerModel,
    _PatchMixerProjectCore,
)
from modeling_module.models.PatchMixer.common.configs import (
    PatchMixerConfig,
    PatchMixerExogenousConfig,
)
from modeling_module.models.PatchMixer.variants import PatchMixerExogenousModel
from modeling_module.models.PatchTST.common.configs import AttentionConfig, PatchTSTConfig
from modeling_module.models.PatchTST.supervised.PatchTST import PatchTSTModel
from modeling_module.models.PatchTST.supervised.variants import (
    PatchTSTEndogenousModel,
    PatchTSTExogenousModel,
    PatchTSTQuantileEndogenousModel,
    PatchTSTQuantileExogenousModel,
)
from modeling_module.models.model_builder import (
    build_patch_mixer,
    build_patch_mixer_exogenous,
    build_patchTST,
    build_patchTST_exogenous,
    build_patchTST_quantile,
    build_patchTST_quantile_exogenous,
)
from modeling_module.utils.checkpoint import build_checkpoint_payload


def _patchtst_config(*, exogenous: bool) -> PatchTSTConfig:
    return PatchTSTConfig(
        lookback=8,
        horizon=2,
        patch_len=4,
        stride=2,
        padding_patch="end",
        d_model=8,
        d_ff=16,
        n_layers=1,
        dropout=0.0,
        c_in=1,
        past_exo_cont_dim=1 if exogenous else 0,
        future_exo_dim=1 if exogenous else 0,
        future_exo_fusion_dropout=0.0,
        use_revin=False,
        attn=AttentionConfig(
            n_heads=2,
            d_model=8,
            attn_dropout=0.0,
            proj_dropout=0.0,
        ),
    )


def _patchmixer_config(*, exogenous: bool):
    if not exogenous:
        return PatchMixerConfig(
            lookback=8,
            horizon=2,
            patch_len=4,
            stride=2,
            d_model=8,
            e_layers=1,
            mixer_kernel_size=3,
            dropout=0.0,
        )
    return PatchMixerExogenousConfig(
        lookback=8,
        horizon=2,
        patch_len=4,
        stride=2,
        d_model=8,
        e_layers=1,
        mixer_kernel_size=3,
        f_out=8,
        head_hidden=8,
        dropout=0.0,
        past_exo_mode="z_gate",
        past_exo_cont_dim=1 if exogenous else 0,
        future_exo_dim=1 if exogenous else 0,
        use_revin=False,
        final_nonneg=False,
    )


def test_compatibility_builders_route_by_configured_exogenous_widths():
    assert isinstance(build_patchTST(_patchtst_config(exogenous=False)), PatchTSTEndogenousModel)
    assert isinstance(build_patchTST(_patchtst_config(exogenous=True)), PatchTSTExogenousModel)
    assert isinstance(
        build_patchTST_quantile(_patchtst_config(exogenous=False)),
        PatchTSTQuantileEndogenousModel,
    )
    assert isinstance(
        build_patchTST_quantile(_patchtst_config(exogenous=True)),
        PatchTSTQuantileExogenousModel,
    )

    assert isinstance(build_patch_mixer(_patchmixer_config(exogenous=False)), PatchMixerModel)
    assert isinstance(
        build_patch_mixer_exogenous(_patchmixer_config(exogenous=True)),
        PatchMixerExogenousModel,
    )
    with pytest.raises(ValueError, match="endogenous-only"):
        build_patch_mixer(_patchmixer_config(exogenous=True))


@pytest.mark.parametrize(
    ("builder", "config"),
    (
        (build_patchTST_exogenous, _patchtst_config(exogenous=False)),
        (build_patchTST_quantile_exogenous, _patchtst_config(exogenous=False)),
        (build_patch_mixer_exogenous, _patchmixer_config(exogenous=False)),
    ),
)
def test_explicit_exogenous_builders_reject_endogenous_configs(builder, config):
    with pytest.raises(ValueError, match="requires at least one configured exogenous input"):
        builder(config)


def test_patchtst_legacy_exogenous_state_dict_strict_loads_into_split_variant():
    config = _patchtst_config(exogenous=True)
    legacy = PatchTSTModel(config)
    split = build_patchTST(config)

    assert isinstance(split, PatchTSTExogenousModel)
    assert legacy.state_dict().keys() == split.state_dict().keys()
    split.load_state_dict(legacy.state_dict(), strict=True)


def test_patchmixer_legacy_exogenous_state_dict_strict_loads_into_split_variant():
    config = _patchmixer_config(exogenous=True)
    legacy = _PatchMixerProjectCore(config)
    split = build_patch_mixer_exogenous(config)

    assert isinstance(split, PatchMixerExogenousModel)
    assert set(legacy.state_dict()) - set(split.state_dict()) == {
        "out_scale",
        "out_bias",
        "dw_gain",
        "dw_head.weight",
        "dw_head.bias",
    }
    assert set(split.state_dict()).issubset(legacy.state_dict())
    split.load_state_dict(legacy.state_dict(), strict=True)


@pytest.mark.parametrize(
    ("builder", "config"),
    (
        (build_patchTST_exogenous, _patchtst_config(exogenous=True)),
        (build_patch_mixer_exogenous, _patchmixer_config(exogenous=True)),
    ),
)
def test_explicit_exogenous_models_reject_missing_configured_inputs(builder, config):
    model = builder(config)
    x = torch.randn(2, 8, 1)

    with pytest.raises(RuntimeError, match="future_exo.*required|missing required inputs"):
        model(x)


@pytest.mark.parametrize(
    ("builder", "config"),
    (
        (build_patchTST_exogenous, _patchtst_config(exogenous=True)),
        (build_patch_mixer_exogenous, _patchmixer_config(exogenous=True)),
    ),
)
def test_explicit_exogenous_fusion_has_input_gradients(builder, config):
    torch.manual_seed(20260721)
    model = builder(config).train()
    x = torch.randn(2, 8, 1)
    past = torch.randn(2, 8, 1, requires_grad=True)
    future = torch.randn(2, 2, 1, requires_grad=True)

    output = model(x, past_exo_cont=past, future_exo=future)
    tensor = output["q"] if isinstance(output, dict) else output
    tensor.square().mean().backward()

    assert past.grad is not None
    assert future.grad is not None
    assert float(past.grad.abs().sum()) > 0.0
    assert float(future.grad.abs().sum()) > 0.0


@pytest.mark.parametrize(
    ("builder", "config", "fusion_strategy"),
    (
        (
            build_patchTST_exogenous,
            _patchtst_config(exogenous=True),
            "patch_concat+future_cross_attention",
        ),
        (
            build_patch_mixer_exogenous,
            _patchmixer_config(exogenous=True),
            "gated_residual+future_shift",
        ),
    ),
)
def test_checkpoint_records_explicit_exogenous_architecture(builder, config, fusion_strategy):
    model = builder(config)
    checkpoint = build_checkpoint_payload(model, config)

    assert checkpoint["meta"]["architecture_variant"] == "exogenous"
    assert checkpoint["meta"]["exogenous_fusion_strategy"] == fusion_strategy

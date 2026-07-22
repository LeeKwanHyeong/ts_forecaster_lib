from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path

import pytest
import torch

from modeling_module import load_predictor
from modeling_module.models.PatchMixer.common.configs import (
    PatchMixerConfig,
    PatchMixerConfigMonthly,
)
from modeling_module.models.PatchMixer.variants import (
    PatchMixerExogenousModel,
    PatchMixerQuantileEndogenousModel,
    PatchMixerQuantileExogenousModel,
)
from modeling_module.utils.checkpoint import build_checkpoint_payload


def _config(
    *,
    use_revin: bool,
    exogenous: bool = False,
    distribution: bool = False,
    q_clip_norm: float | None = 10.0,
    exo_is_normalized_default: bool = True,
    exo_is_normalized: bool = True,
    config_cls: type[PatchMixerConfig] = PatchMixerConfig,
) -> PatchMixerConfig:
    return config_cls(
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
        future_exo_dim=1 if exogenous else 0,
        patch_cfgs=((4, 2, 3),),
        per_branch_dim=4,
        fused_dim=8,
        out_mul=2 if distribution else 1,
        param_names=["loc", "scale"] if distribution else None,
        q_clip_norm=q_clip_norm,
        exo_is_normalized_default=exo_is_normalized_default,
        exo_is_normalized=exo_is_normalized,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(-1.0, 2.0, steps=16).reshape(2, 8, 1)
    future = torch.linspace(-0.5, 0.75, steps=4).reshape(2, 2, 1)
    return x, future


def _output_tensor(output):
    return output["q"] if isinstance(output, dict) else output


@pytest.mark.parametrize(
    ("model_cls", "distribution"),
    (
        (PatchMixerExogenousModel, False),
        (PatchMixerQuantileExogenousModel, False),
        (PatchMixerExogenousModel, True),
    ),
)
def test_patchmixer_legacy_exo_normalization_flag_is_a_noop(
    model_cls,
    distribution: bool,
) -> None:
    torch.manual_seed(20260802)
    model = model_cls(
        _config(use_revin=True, exogenous=True, distribution=distribution)
    ).eval()
    x, future = _inputs()

    with torch.no_grad():
        outputs = [
            _output_tensor(
                model(x, future_exo=future, exo_is_normalized=value)
            )
            for value in (None, False, True)
        ]

    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(outputs[0], outputs[2], rtol=0.0, atol=0.0)


def test_patchmixer_legacy_exo_normalization_config_fields_are_noops() -> None:
    torch.manual_seed(20260805)
    reference = PatchMixerExogenousModel(
        _config(use_revin=True, exogenous=True)
    ).eval()
    state_dict = reference.state_dict()
    x, future = _inputs()

    outputs = []
    for default_value in (False, True):
        for training_value in (False, True):
            model = PatchMixerExogenousModel(
                _config(
                    use_revin=True,
                    exogenous=True,
                    exo_is_normalized_default=default_value,
                    exo_is_normalized=training_value,
                )
            ).eval()
            model.load_state_dict(state_dict, strict=True)
            assert model.exo_is_normalized_default is default_value
            with torch.no_grad():
                outputs.append(model(x, future_exo=future))

    for output in outputs[1:]:
        torch.testing.assert_close(outputs[0], output, rtol=0.0, atol=0.0)


def test_patchmixer_quantile_training_does_not_apply_output_clip() -> None:
    torch.manual_seed(20260803)
    model = PatchMixerQuantileEndogenousModel(
        _config(use_revin=True, q_clip_norm=1e-3)
    ).train()
    x, _ = _inputs()

    model.q_clip_eval = 1e-3
    output_small = model(x)["q"]
    model.q_clip_eval = 1e3
    output_large = model(x)["q"]

    torch.testing.assert_close(output_small, output_large, rtol=0.0, atol=0.0)
    assert not hasattr(model, "q_clip_train")


@pytest.mark.parametrize(
    ("use_revin", "clip_expected"),
    ((True, True), (False, False)),
)
def test_patchmixer_quantile_eval_clip_is_revin_space_only(
    use_revin: bool,
    clip_expected: bool,
) -> None:
    torch.manual_seed(20260804)
    model = PatchMixerQuantileEndogenousModel(
        _config(use_revin=use_revin, q_clip_norm=0.05)
    ).eval()
    x = torch.linspace(0.0, 100.0, steps=16).reshape(2, 8, 1)

    with torch.no_grad():
        clipped = model(x)["q"]
        model.q_clip_eval = None
        unclipped = model(x)["q"]

    max_delta = float((clipped - unclipped).abs().max())
    if clip_expected:
        assert max_delta > 0.0
    else:
        torch.testing.assert_close(clipped, unclipped, rtol=0.0, atol=0.0)


def test_patchmixer_quantile_clip_config_and_checkpoint_contract() -> None:
    field_names = {field.name for field in fields(PatchMixerConfig)}
    assert "q_clip_norm" in field_names
    assert "q_clip_train" not in field_names
    assert "future_exo_shift_space" not in field_names

    config = _config(use_revin=True, q_clip_norm=2.5)
    model = PatchMixerQuantileEndogenousModel(config)
    payload = build_checkpoint_payload(model, config)

    assert asdict(config)["q_clip_norm"] == 2.5
    assert payload["config"]["q_clip_norm"] == 2.5
    assert payload["config"]["exo_is_normalized_default"] is True
    assert payload["config"]["exo_is_normalized"] is True
    assert model.q_clip_eval == 2.5
    assert not hasattr(model, "q_clip_train")
    assert all("clip" not in key for key in model.state_dict())

    restored_config = PatchMixerConfig(**payload["config"])
    assert restored_config.q_clip_norm == 2.5

    disabled = PatchMixerQuantileEndogenousModel(
        _config(use_revin=True, q_clip_norm=None)
    )
    assert disabled.q_clip_eval is None


@pytest.mark.parametrize(
    "config_cls",
    (PatchMixerConfig, PatchMixerConfigMonthly),
    ids=("base", "monthly"),
)
def test_legacy_checkpoint_without_q_clip_norm_strict_loads(
    tmp_path: Path,
    config_cls: type[PatchMixerConfig],
) -> None:
    torch.manual_seed(20260806)
    config = _config(
        use_revin=True,
        q_clip_norm=10.0,
        config_cls=config_cls,
    )
    model = PatchMixerQuantileEndogenousModel(config).eval()
    x, _ = _inputs()
    with torch.no_grad():
        expected = model(x)["q"]

    payload = build_checkpoint_payload(
        model,
        config,
        extra_meta={"model_key": "patchmixer_quantile", "family_key": "patchmixer"},
    )
    legacy_config = dict(payload["config"])
    legacy_config.pop("q_clip_norm")
    payload.pop("config")
    payload["cfg_state"] = dict(legacy_config)
    checkpoint_path = tmp_path / f"{config_cls.__name__}_without_q_clip_norm.pt"
    torch.save(payload, checkpoint_path)

    predictor = load_predictor(str(checkpoint_path), device="cpu", strict=True)

    assert predictor.model_key == "patchmixer_quantile"
    assert "q_clip_norm" not in payload["cfg_state"]
    assert predictor.model.q_clip_eval == 10.0
    assert predictor.model.state_dict().keys() == model.state_dict().keys()
    with torch.no_grad():
        actual = predictor.model(x)["q"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

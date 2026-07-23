from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import get_args, get_type_hints

import torch

from modeling_module import load_predictor
from modeling_module.models.PatchMixer.PatchMixer import PatchMixerQuantileModel
from modeling_module.models.PatchMixer.common.configs import PatchMixerExogenousConfig
from modeling_module.models.PatchMixer.variants import PatchMixerExogenousModel
from modeling_module.utils.checkpoint import build_checkpoint_payload


def _config(
    *,
    exogenous: bool,
    shift_space: str = "output",
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
        use_revin=True,
        final_nonneg=False,
        past_exo_mode="none",
        future_exo_dim=1 if exogenous else 0,
        future_exo_shift_space=shift_space,
        future_exo_normalized_residual_limit=residual_limit,
        q_clip_norm=2.5,
        exo_is_normalized_default=True,
        exo_is_normalized=True,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(-1.0, 2.0, steps=16).reshape(2, 8, 1)
    future = torch.linspace(-0.5, 0.75, steps=4).reshape(2, 2, 1)
    return x, future


def test_patchmixer_legacy_exo_normalization_argument_is_a_noop() -> None:
    torch.manual_seed(20260802)
    model = PatchMixerExogenousModel(_config(exogenous=True)).eval()
    x, future = _inputs()

    with torch.no_grad():
        outputs = [
            model(x, future_exo=future, exo_is_normalized=value)
            for value in (None, False, True)
        ]

    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(outputs[0], outputs[2], rtol=0.0, atol=0.0)


def test_patchmixer_exogenous_config_keeps_load_compatibility_fields() -> None:
    field_names = {field.name for field in fields(PatchMixerExogenousConfig)}
    assert "q_clip_norm" in field_names
    assert "q_clip_train" not in field_names
    assert "exo_is_normalized_default" in field_names
    assert "future_exo_shift_space" in field_names
    assert "future_exo_normalized_residual_limit" in field_names


def test_patchmixer_future_shift_space_config_roundtrips_normalized() -> None:
    shift_space_type = get_type_hints(PatchMixerExogenousConfig)[
        "future_exo_shift_space"
    ]
    assert get_args(shift_space_type) == ("output", "normalized")

    config = _config(
        exogenous=True,
        shift_space="normalized",
        residual_limit=0.15,
    )
    restored = PatchMixerExogenousConfig(**asdict(config))
    assert restored.future_exo_shift_space == "normalized"
    assert restored.future_exo_normalized_residual_limit == 0.15


def test_retired_quantile_checkpoint_without_new_fields_strict_loads(
    tmp_path: Path,
) -> None:
    torch.manual_seed(20260806)
    config = _config(exogenous=False)
    config.q_clip_norm = 10.0
    model = PatchMixerQuantileModel(config).eval()
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
    legacy_config.pop("future_exo_shift_space")
    legacy_config.pop("future_exo_normalized_residual_limit")
    payload["config"] = legacy_config

    checkpoint_path = tmp_path / "patchmixer_quantile_legacy.pt"
    torch.save(payload, checkpoint_path)
    predictor = load_predictor(str(checkpoint_path), device="cpu", strict=True)

    assert predictor.model_key == "patchmixer_quantile"
    assert predictor.model.q_clip_eval == 10.0
    with torch.no_grad():
        actual = predictor.model(x)["q"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

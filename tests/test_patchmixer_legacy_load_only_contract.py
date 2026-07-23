from __future__ import annotations

from pathlib import Path

import pytest
import torch

from modeling_module import load_predictor
from modeling_module.models.PatchMixer.PatchMixer import _PatchMixerLegacyModel
from modeling_module.models.PatchMixer.common.configs import PatchMixerExogenousConfig
from modeling_module.utils.checkpoint import build_checkpoint_payload


def _legacy_config() -> PatchMixerExogenousConfig:
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
        future_exo_dim=0,
    )


def _save_legacy_point_checkpoint(path: Path) -> tuple[_PatchMixerLegacyModel, torch.Tensor]:
    torch.manual_seed(20260723)
    config = _legacy_config()
    model = _PatchMixerLegacyModel(config).eval()
    payload = build_checkpoint_payload(
        model,
        config,
        extra_meta={"model_key": "patchmixer_base", "family_key": "patchmixer"},
    )
    torch.save(payload, path)
    return model, torch.linspace(-1.0, 2.0, steps=16).reshape(2, 8, 1)


def test_v3_enhanced_point_checkpoint_remains_exactly_loadable(tmp_path: Path) -> None:
    path = tmp_path / "patchmixer_base_v3.pt"
    expected_model, x = _save_legacy_point_checkpoint(path)

    predictor = load_predictor(str(path), device="cpu", strict=True)

    assert predictor.model_key == "patchmixer_base"
    assert type(predictor.model).__name__ == "_PatchMixerLegacyModel"
    assert predictor.model.state_dict().keys() == expected_model.state_dict().keys()
    with torch.no_grad():
        expected = expected_model(x)
        actual = predictor.model(x)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_incompatible_pre_version_basemodel_fails_closed_even_non_strict(
    tmp_path: Path,
) -> None:
    path = tmp_path / "weekly_PatchMixerBase_L52_H27.pt"
    _, _ = _save_legacy_point_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.pop("format_version")
    payload["meta"].pop("format_version")
    payload["model_class"] = "BaseModel"
    payload["state_dict"]["resid_scale"] = torch.tensor(1.0)
    torch.save(payload, path)

    with pytest.raises(ValueError, match="Unsupported pre-version PatchMixer checkpoint"):
        load_predictor(str(path), device="cpu", strict=False)

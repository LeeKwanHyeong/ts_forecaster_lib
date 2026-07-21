from __future__ import annotations

from pathlib import Path

import torch
import pytest

import modeling_module.models as public_models
from modeling_module import load_predictor
from modeling_module.api.train import (
    ArchitectureConfig,
    PatchMixerArchitectureConfig,
    _normalize_model_architecture,
)
from modeling_module.models.PatchMixer.original import (
    PatchMixerOriginalConfig,
    PatchMixerOriginalModel,
)
from modeling_module.models.PatchMixer.provenance import (
    PATCHMIXER_UPSTREAM_COMMIT,
    PATCHMIXER_UPSTREAM_REPOSITORY,
)
from modeling_module.models.registry import (
    build_model,
    expand_training_targets,
    get_model_spec,
    infer_artifact_model_key_from_checkpoint,
    list_available_model_keys,
    resolve_artifact_model_key,
)
from modeling_module.training.adapters import PatchMixerOriginalAdapter
from modeling_module.utils.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    _extract_cfg_obj,
    build_checkpoint_payload,
    save_model,
)


def _config() -> PatchMixerOriginalConfig:
    return PatchMixerOriginalConfig(
        lookback=16,
        horizon=4,
        enc_in=2,
        patch_len=4,
        stride=2,
        mixer_kernel_size=3,
        d_model=8,
        e_layers=1,
        dropout=0.0,
        head_dropout=0.0,
    )


def test_original_builder_and_registry_are_publicly_resolvable() -> None:
    config = _config()

    direct = public_models.build_patch_mixer_original(config)
    registered = build_model(
        "patchmixer-upstream",
        {
            "seq_len": config.lookback,
            "pred_len": config.horizon,
            "enc_in": config.enc_in,
            "patch_len": config.patch_len,
            "stride": config.stride,
            "mixer_kernel_size": config.mixer_kernel_size,
            "d_model": config.d_model,
            "e_layers": config.e_layers,
            "dropout": config.dropout,
            "head_dropout": config.head_dropout,
        },
    )

    assert public_models.PatchMixerOriginalConfig is PatchMixerOriginalConfig
    assert isinstance(direct, PatchMixerOriginalModel)
    assert isinstance(registered, PatchMixerOriginalModel)
    assert direct.configs == config
    assert registered.configs == config
    assert "patchmixer_original" in list_available_model_keys()
    assert resolve_artifact_model_key("PatchMixerOriginalModel") == "patchmixer_original"

    spec = get_model_spec("patchmixer_original")
    assert spec.family == "patchmixer"
    assert spec.trainable is True
    assert spec.included_in_family is False
    assert expand_training_targets(["patchmixer"]) == [
        "patchmixer_base",
        "patchmixer_quantile",
    ]
    assert expand_training_targets(["patchmixer_original"]) == [
        "patchmixer_original"
    ]


def test_public_patchmixer_config_accepts_original_kernel_override() -> None:
    normalized = _normalize_model_architecture(
        ArchitectureConfig(
            patchmixer=PatchMixerArchitectureConfig(mixer_kernel_size=3)
        )
    )

    assert normalized == {"patchmixer": {"mixer_kernel_size": 3}}


def test_original_adapter_is_shape_strict_and_rejects_exogenous_features() -> None:
    model = public_models.build_patch_mixer_original(_config())
    adapter = PatchMixerOriginalAdapter()
    x = torch.randn(2, model.configs.lookback, model.configs.enc_in)

    output = adapter.forward(model, x, mode="train")

    assert output.shape == (2, model.configs.horizon, model.configs.enc_in)
    with pytest.raises(RuntimeError, match="endogenous-only baseline"):
        adapter.forward(
            model,
            x,
            future_exo=torch.ones(2, model.configs.horizon, 1),
        )


def test_original_checkpoint_round_trip_is_strict_and_self_identifying(
    tmp_path: Path,
) -> None:
    torch.manual_seed(20260724)
    config = _config()
    model = public_models.build_patch_mixer_original(config).eval()
    x = torch.randn(3, config.lookback, config.enc_in)

    with torch.no_grad():
        expected = model(x)

    checkpoint_path = tmp_path / "patchmixer_original.pt"
    save_model(
        model,
        config,
        str(checkpoint_path),
        extra_meta={
            "model_key": "patchmixer_original",
            "family_key": "patchmixer",
        },
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert checkpoint["cfg_cls"] == "PatchMixerOriginalConfig"
    assert checkpoint["model_class"] == "PatchMixerOriginalModel"
    assert checkpoint["output_spec"] == {
        "mode": "point",
        "distribution": None,
        "out_mult": 1,
        "param_names": None,
    }
    assert checkpoint["meta"]["model_key"] == "patchmixer_original"
    assert checkpoint["meta"]["architecture_variant"] == "original"
    assert checkpoint["meta"]["upstream_repository"] == PATCHMIXER_UPSTREAM_REPOSITORY
    assert checkpoint["meta"]["upstream_commit"] == PATCHMIXER_UPSTREAM_COMMIT
    assert infer_artifact_model_key_from_checkpoint(checkpoint) == "patchmixer_original"

    predictor = load_predictor(
        str(checkpoint_path),
        device="cpu",
        strict=True,
    )
    with torch.no_grad():
        actual = predictor.model(x)

    assert predictor.model_key == "patchmixer_original"
    assert isinstance(predictor.model, PatchMixerOriginalModel)
    assert list(predictor.model.state_dict()) == list(model.state_dict())
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_original_legacy_cfg_state_rebuilds_typed_config() -> None:
    model = public_models.build_patch_mixer_original(_config())
    checkpoint = build_checkpoint_payload(model, model.configs)
    checkpoint.pop("config")

    restored = _extract_cfg_obj(checkpoint)

    assert isinstance(restored, PatchMixerOriginalConfig)
    assert restored == model.configs
    assert infer_artifact_model_key_from_checkpoint(checkpoint) == "patchmixer_original"

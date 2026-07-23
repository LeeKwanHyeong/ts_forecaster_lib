from __future__ import annotations

# Canonical public names and checkpoint identity for the paper model.

from pathlib import Path

import pytest
import torch

import modeling_module.models as public_models
from modeling_module import load_predictor
from modeling_module.api.train import (
    ArchitectureConfig,
    PatchMixerArchitectureConfig,
    _normalize_model_architecture,
)
from modeling_module.models.PatchMixer import PatchMixerConfig, PatchMixerModel
from modeling_module.models.PatchMixer.backbone import PatchMixerBackbone, PatchMixerLayer
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
from modeling_module.training.adapters import PatchMixerEndogenousAdapter
from modeling_module.utils.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    _extract_cfg_obj,
    build_checkpoint_payload,
    save_model,
)


def _config() -> PatchMixerConfig:
    return PatchMixerConfig(
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


def test_patchmixer_paper_model_owns_canonical_public_names() -> None:
    assert public_models.PatchMixerConfig is PatchMixerConfig
    assert PatchMixerConfig.__name__ == "PatchMixerConfig"
    assert PatchMixerModel.__name__ == "PatchMixerModel"
    assert PatchMixerLayer.__module__.endswith("PatchMixer.backbone")
    assert PatchMixerBackbone.__module__.endswith("PatchMixer.backbone")
    assert PatchMixerModel.architecture_variant == "endogenous"


def test_patchmixer_builder_registry_and_legacy_aliases_are_resolvable() -> None:
    config = _config()
    direct = public_models.build_patch_mixer(config)
    registered = build_model("patchmixer-upstream", config)

    assert isinstance(direct, PatchMixerModel)
    assert isinstance(registered, PatchMixerModel)
    assert direct.configs == config
    assert registered.configs == config
    assert [
        key for key in list_available_model_keys() if key.startswith("patchmixer")
    ] == ["patchmixer", "patchmixer_exo"]
    assert resolve_artifact_model_key("patchmixer_original") == "patchmixer"
    assert resolve_artifact_model_key("PatchMixerOriginalModel") == "patchmixer"

    spec = get_model_spec("patchmixer")
    assert spec.family == "patchmixer"
    assert spec.trainable is True
    assert spec.load_only is False
    assert expand_training_targets(["patchmixer"]) == ["patchmixer"]
    assert expand_training_targets(["patchmixer_original"]) == ["patchmixer"]


def test_public_patchmixer_config_accepts_kernel_override() -> None:
    normalized = _normalize_model_architecture(
        ArchitectureConfig(
            patchmixer=PatchMixerArchitectureConfig(mixer_kernel_size=3)
        )
    )
    assert normalized == {"patchmixer": {"mixer_kernel_size": 3}}


def test_endogenous_adapter_is_shape_strict_and_rejects_exogenous_features() -> None:
    model = public_models.build_patch_mixer(_config())
    adapter = PatchMixerEndogenousAdapter()
    x = torch.randn(2, model.configs.lookback, model.configs.enc_in)

    output = adapter.forward(model, x, mode="train")
    assert output.shape == (2, model.configs.horizon, model.configs.enc_in)
    with pytest.raises(RuntimeError, match="endogenous-only"):
        adapter.forward(
            model,
            x,
            future_exo=torch.ones(2, model.configs.horizon, 1),
        )


def test_patchmixer_checkpoint_round_trip_is_strict_and_self_identifying(
    tmp_path: Path,
) -> None:
    torch.manual_seed(20260724)
    config = _config()
    model = public_models.build_patch_mixer(config).eval()
    x = torch.randn(3, config.lookback, config.enc_in)
    with torch.no_grad():
        expected = model(x)

    checkpoint_path = tmp_path / "patchmixer.pt"
    save_model(
        model,
        config,
        str(checkpoint_path),
        extra_meta={"model_key": "patchmixer", "family_key": "patchmixer"},
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert checkpoint["cfg_cls"] == "PatchMixerConfig"
    assert checkpoint["model_class"] == "PatchMixerModel"
    assert checkpoint["meta"]["model_key"] == "patchmixer"
    assert checkpoint["meta"]["architecture_variant"] == "endogenous"
    assert checkpoint["meta"]["upstream_repository"] == PATCHMIXER_UPSTREAM_REPOSITORY
    assert checkpoint["meta"]["upstream_commit"] == PATCHMIXER_UPSTREAM_COMMIT
    assert infer_artifact_model_key_from_checkpoint(checkpoint) == "patchmixer"

    predictor = load_predictor(str(checkpoint_path), device="cpu", strict=True)
    with torch.no_grad():
        actual = predictor.model(x)

    assert predictor.model_key == "patchmixer"
    assert isinstance(predictor.model, PatchMixerModel)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_patchmixer_cfg_state_rebuilds_typed_config() -> None:
    model = public_models.build_patch_mixer(_config())
    checkpoint = build_checkpoint_payload(
        model,
        model.configs,
        extra_meta={"model_key": "patchmixer"},
    )
    checkpoint.pop("config")

    restored = _extract_cfg_obj(checkpoint)

    assert isinstance(restored, PatchMixerConfig)
    assert restored == model.configs

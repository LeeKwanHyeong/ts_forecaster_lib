from __future__ import annotations

from pathlib import Path

import pytest
import torch

import modeling_module as mm
import modeling_module.models as public_models
from modeling_module.api.train import (
    ArchitectureConfig,
    TimeMixerArchitectureConfig,
    _normalize_model_architecture,
)
from modeling_module.models.TimeMixer import TimeMixerConfig, TimeMixerModel
from modeling_module.models.TimeMixer.backbone import TimeMixerBackbone
from modeling_module.models.TimeMixer.provenance import (
    TIMEMIXER_UPSTREAM_COMMIT,
    TIMEMIXER_UPSTREAM_REPOSITORY,
)
from modeling_module.models.registry import (
    build_model,
    expand_training_targets,
    get_model_spec,
    infer_artifact_model_key_from_checkpoint,
    list_available_model_keys,
    resolve_artifact_model_key,
)
from modeling_module.utils.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    _extract_cfg_obj,
    build_checkpoint_payload,
    save_model,
)


def _tiny_config(**overrides) -> TimeMixerConfig:
    values = {
        "lookback": 16,
        "horizon": 4,
        "d_model": 4,
        "d_ff": 8,
        "e_layers": 1,
        "moving_avg": 3,
        "down_sampling_layers": 2,
        "down_sampling_window": 2,
        "dropout": 0.0,
        "device": "cpu",
    }
    values.update(overrides)
    return TimeMixerConfig(**values)


def test_timemixer_wrapper_preserves_backbone_state_and_numerics() -> None:
    config = _tiny_config()
    torch.manual_seed(20260730)
    expected_model = TimeMixerBackbone(config).eval()
    torch.manual_seed(20260730)
    model = TimeMixerModel(config).eval()
    x = torch.randn(2, config.lookback, 1, requires_grad=True)

    expected = expected_model(x.detach())
    actual = model(x)

    assert list(model.state_dict()) == list(expected_model.state_dict())
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_039
    assert actual.shape == (2, config.horizon, 1)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    actual.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.parameters()
    )


@pytest.mark.parametrize(
    ("value", "error_type", "message"),
    (
        ([[[1.0]]], TypeError, "floating tensor"),
        (torch.ones(2, 16), ValueError, "shape"),
        (torch.ones(0, 16, 1), ValueError, "non-empty batch"),
        (torch.ones(2, 15, 1), ValueError, "lookback mismatch"),
        (torch.ones(2, 16, 2), ValueError, "channel mismatch"),
        (torch.ones(2, 16, 1, dtype=torch.long), TypeError, "floating input"),
        (
            torch.full((2, 16, 1), float("nan")),
            ValueError,
            "finite values",
        ),
        (
            torch.full((2, 16, 1), float("inf")),
            ValueError,
            "finite values",
        ),
    ),
)
def test_timemixer_wrapper_rejects_invalid_endogenous_inputs(
    value,
    error_type,
    message,
) -> None:
    model = TimeMixerModel(_tiny_config())

    with pytest.raises(error_type, match=message):
        model(value)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("future_exo", torch.ones(2, 4, 1)),
        ("past_exo_cont", torch.ones(2, 16, 1)),
        ("past_exo_cat", torch.ones(2, 16, 1, dtype=torch.long)),
    ),
)
def test_timemixer_wrapper_rejects_nonempty_exogenous_inputs(
    field: str,
    value: torch.Tensor,
) -> None:
    model = TimeMixerModel(_tiny_config())

    with pytest.raises(RuntimeError, match=rf"endogenous-only.*{field}"):
        model(torch.ones(2, 16, 1), **{field: value})


def test_timemixer_wrapper_accepts_structurally_empty_exogenous_inputs() -> None:
    model = TimeMixerModel(_tiny_config()).eval()
    x = torch.randn(2, 16, 1)

    with torch.no_grad():
        expected = model(x)
        actual = model(
            x,
            future_exo=x.new_empty((2, 4, 0)),
            past_exo_cont=x.new_empty((2, 16, 0)),
            past_exo_cat=torch.empty((2, 16, 0), dtype=torch.long),
        )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_timemixer_builder_registry_and_public_config_are_connected() -> None:
    config = _tiny_config()
    direct = public_models.build_timemixer(config)
    registered = build_model(
        "timemixercanonical",
        {
            "lookback": 16,
            "horizon": 4,
            "d_model": 4,
            "d_ff": 8,
            "e_layers": 1,
            "moving_avg": 3,
            "down_sampling_layers": 2,
            "down_sampling_window": 2,
            "dropout": 0.0,
            "device": "cpu",
        },
    )

    assert isinstance(direct, TimeMixerModel)
    assert isinstance(registered, TimeMixerModel)
    assert public_models.TimeMixerConfig is TimeMixerConfig
    assert mm.TimeMixerArchitectureConfig is TimeMixerArchitectureConfig
    assert "timemixer" in list_available_model_keys()
    assert resolve_artifact_model_key("timemixerbase") == "timemixer"
    assert resolve_artifact_model_key("TimeMixerCanonical") == "timemixer"

    spec = get_model_spec("timemixer")
    assert spec.family == "timemixer"
    assert spec.exogenous_policy == "none"
    assert spec.load_only is False
    assert spec.trainable is False
    with pytest.raises(ValueError, match="not trainable"):
        expand_training_targets(["timemixer"])

    normalized = _normalize_model_architecture(
        ArchitectureConfig(
            timemixer=TimeMixerArchitectureConfig(
                d_model=32,
                moving_avg=7,
                down_sampling_layers=2,
                use_norm=False,
            )
        )
    )
    assert normalized == {
        "timemixer": {
            "d_model": 32,
            "moving_avg": 7,
            "down_sampling_layers": 2,
            "use_norm": False,
        }
    }


def test_timemixer_checkpoint_round_trip_is_strict_and_self_identifying(
    tmp_path: Path,
) -> None:
    torch.manual_seed(20260731)
    config = _tiny_config()
    model = public_models.build_timemixer(config).eval()
    x = torch.randn(3, config.lookback, 1)
    with torch.no_grad():
        expected = model(x)

    checkpoint_path = tmp_path / "timemixer.pt"
    save_model(
        model,
        config,
        str(checkpoint_path),
        extra_meta={"model_key": "timemixer", "family_key": "timemixer"},
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert checkpoint["cfg_cls"] == "TimeMixerConfig"
    assert checkpoint["model_class"] == "TimeMixerModel"
    assert checkpoint["output_spec"]["mode"] == "point"
    assert checkpoint["meta"]["model_key"] == "timemixer"
    assert checkpoint["meta"]["architecture_variant"] == "endogenous"
    assert checkpoint["meta"]["exogenous_fusion_strategy"] == "none"
    assert checkpoint["meta"]["upstream_repository"] == TIMEMIXER_UPSTREAM_REPOSITORY
    assert checkpoint["meta"]["upstream_commit"] == TIMEMIXER_UPSTREAM_COMMIT
    assert infer_artifact_model_key_from_checkpoint(checkpoint) == "timemixer"
    assert infer_artifact_model_key_from_checkpoint(
        {"model_class": "TimeMixerModel"}
    ) == "timemixer"

    predictor = mm.load_predictor(
        str(checkpoint_path),
        device="cpu",
        strict=True,
    )
    with torch.no_grad():
        actual = predictor.model(x)

    assert predictor.model_key == "timemixer"
    assert predictor.family_key == "timemixer"
    assert predictor.default_horizon == config.horizon
    assert isinstance(predictor.model, TimeMixerModel)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_timemixer_cfg_state_rebuilds_typed_config() -> None:
    model = public_models.build_timemixer(_tiny_config())
    checkpoint = build_checkpoint_payload(
        model,
        model.configs,
        extra_meta={"model_key": "timemixer"},
    )
    checkpoint.pop("config")

    restored = _extract_cfg_obj(checkpoint)

    assert isinstance(restored, TimeMixerConfig)
    assert restored.lookback == model.configs.lookback
    assert restored.horizon == model.configs.horizon
    assert restored.scale_lengths == model.configs.scale_lengths

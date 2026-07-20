from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch

from modeling_module import load_predictor
from modeling_module.training.model_losses.loss_module import DistributionLoss


pytestmark = pytest.mark.filterwarnings("ignore:Initializing zero-element tensors is a no-op:UserWarning")

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "legacy_distribution_checkpoints"
MANIFEST = json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))
ARTIFACTS = MANIFEST["artifacts"]
SUCCESS_CASES = [(name, case) for name, case in ARTIFACTS.items() if case["restore"] == "success"]
REJECTED_CASES = [(name, case) for name, case in ARTIFACTS.items() if case["restore"] == "reject"]
POINT_CONTROL_CASES = [
    (name, case) for name, case in ARTIFACTS.items() if case["restore"] == "point_control"
]

EXPECTED_PARAMS = {
    "Normal": ["-loc", "-scale"],
    "StudentT": ["-df", "-loc", "-scale"],
}
EXPECTED_HEADS = {
    "patchtst_base": "DistHeadWithExo",
    "patchmixer_base": "Sequential",
}
EXPECTED_POINT_HEADS = {
    "patchtst_base": "PointHeadWithExo",
    "patchmixer_base": "Sequential",
    "titan_base": "Linear",
    "exotst_base": "HorizonMLPHead",
}
REJECTION_CONTRACTS = {
    "config_schema_incompatible": (
        (TypeError, ValueError),
        None,
    ),
    "ambiguous_distribution_metadata": (
        ValueError,
        "distribution-shaped head.*no persisted distribution metadata",
    ),
    "invalid_distribution_head": (
        ValueError,
        "declares DistributionLoss.*inferred_out_mult=1",
    ),
}


def _config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return checkpoint.get("config", checkpoint["cfg_state"])


def _saved_out_mult(checkpoint: dict[str, Any], model_key: str) -> int:
    state = checkpoint["state_dict"]
    horizon = int(_config(checkpoint)["horizon"])
    if model_key == "patchtst_base":
        head_key = "head.net.2.weight" if "head.net.2.weight" in state else "head.proj.weight"
        return int(state[head_key].shape[0]) // horizon
    if model_key == "patchmixer_base":
        return int(state["head.2.weight"].shape[0])
    if model_key == "titan_base":
        return int(state["head.weight"].shape[0])
    if model_key == "exotst_base":
        return int(state["head.fc.weight"].shape[0]) // horizon
    raise AssertionError(f"Unhandled fixture model: {model_key}")


def _model_out_mult(model: torch.nn.Module) -> int:
    value = getattr(model, "out_mult", getattr(model, "out_mul", None))
    assert value is not None
    return int(value)


@pytest.mark.parametrize(
    "filename,case",
    list(ARTIFACTS.items()),
    ids=[Path(name).stem for name in ARTIFACTS],
)
def test_frozen_legacy_checkpoint_fixture_integrity(filename: str, case: dict[str, Any]):
    path = FIXTURE_ROOT / filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == case["sha256"]

    # Historical writers persisted torch.__version__ as TorchVersion. The hash is
    # checked before loading, so these repository-owned fixtures are trusted input.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert "output_spec" not in checkpoint
    if case["format"] == "v1":
        assert set(checkpoint) == {"cfg_state", "cfg_cls", "model_class", "state_dict", "meta"}
        assert "format_version" not in checkpoint
    else:
        assert set(checkpoint) == {
            "format_version",
            "config",
            "cfg_state",
            "cfg_cls",
            "model_class",
            "state_dict",
            "meta",
        }
        assert checkpoint["format_version"] == "modeling_module.ckpt.v2"
        assert checkpoint["meta"]["format_version"] == "modeling_module.ckpt.v2"
        assert checkpoint["config"] == checkpoint["cfg_state"]

    config = _config(checkpoint)
    if case["model_key"] == "titan_base":
        assert "loss" not in config
    elif case["restore"] == "point_control":
        assert config["loss"] == "MAE"
    else:
        assert config["loss"] == "DistributionLoss"
    assert _saved_out_mult(checkpoint, case["model_key"]) == case["saved_out_mult"]


@pytest.mark.parametrize(
    "filename,case",
    SUCCESS_CASES,
    ids=[Path(name).stem for name, _ in SUCCESS_CASES],
)
def test_supported_legacy_distribution_fixture_restores_structural_contract(
    filename: str,
    case: dict[str, Any],
):
    path = FIXTURE_ROOT / filename
    with pytest.warns(RuntimeWarning, match="Restoring a legacy distribution checkpoint"):
        predictor = load_predictor(str(path), device="cpu", strict=True)

    expected_params = EXPECTED_PARAMS[case["distribution"]]
    restored_loss = predictor.config["loss"]
    assert isinstance(restored_loss, DistributionLoss)
    assert restored_loss.distribution == case["distribution"]
    assert restored_loss.param_names == expected_params
    assert restored_loss.outputsize_multiplier == case["saved_out_mult"]
    # v1/v2 discarded behavioral loss options. The legacy contract deliberately
    # restores the historical fallback instead of pretending the original value is known.
    assert restored_loss.num_samples == 1000
    assert restored_loss.return_params is False
    assert restored_loss.horizon_weight is None
    assert restored_loss.distribution_kwargs == {}
    torch.testing.assert_close(restored_loss.quantiles, torch.tensor([0.1, 0.5, 0.9]))

    assert predictor.model_key == case["model_key"]
    assert type(predictor.model.head).__name__ == EXPECTED_HEADS[case["model_key"]]
    assert _model_out_mult(predictor.model) == case["saved_out_mult"]
    assert list(predictor.model.param_names) == expected_params

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    restored_state = predictor.model.state_dict()
    saved_state = checkpoint["state_dict"]
    assert restored_state.keys() == saved_state.keys()
    for key, saved_value in saved_state.items():
        torch.testing.assert_close(restored_state[key].cpu(), saved_value.cpu())


@pytest.mark.parametrize(
    "filename,case",
    POINT_CONTROL_CASES,
    ids=[Path(name).stem for name, _ in POINT_CONTROL_CASES],
)
def test_v2_legacy_point_fixture_is_not_rejected_as_distribution(
    filename: str,
    case: dict[str, Any],
):
    path = FIXTURE_ROOT / filename
    for strict in (False, True):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            predictor = load_predictor(str(path), device="cpu", strict=strict)

        assert not any(isinstance(item.message, RuntimeWarning) for item in caught)
        assert predictor.model_key == case["model_key"]
        assert type(predictor.model.head).__name__ == EXPECTED_POINT_HEADS[case["model_key"]]
        assert _model_out_mult(predictor.model) == 1
        if case["model_key"] == "titan_base":
            assert "loss" not in predictor.config
        else:
            assert predictor.config["loss"] == "MAE"


@pytest.mark.parametrize(
    "filename,case",
    REJECTED_CASES,
    ids=[Path(name).stem for name, _ in REJECTED_CASES],
)
def test_legacy_distribution_fixture_rejection_is_fail_closed(
    filename: str,
    case: dict[str, Any],
):
    path = FIXTURE_ROOT / filename
    expected_exception, error_pattern = REJECTION_CONTRACTS[case["rejection_reason"]]

    # An impossible/ambiguous distribution artifact must never fall through to
    # the public loader's non-strict partial-state path as a point predictor.
    for strict in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(expected_exception, match=error_pattern):
                load_predictor(str(path), device="cpu", strict=strict)

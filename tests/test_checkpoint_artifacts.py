import copy
from dataclasses import dataclass

import pytest
import torch

from modeling_module.training.model_losses.loss_module import DistributionLoss
from modeling_module.utils.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    TRAINING_MANIFEST_VERSION,
    _distribution_loss_from_spec,
    _distribution_loss_to_spec,
    _extract_cfg_obj,
    _extract_state_dict,
    save_model,
    save_training_manifest,
    summarize_training_results,
)


@dataclass
class DummyConfig:
    lookback: int = 4
    horizon: int = 2
    loss: str | None = "mae"


def test_distribution_loss_spec_roundtrip_and_contract_validation():
    loss = DistributionLoss(
        "StudentT",
        num_samples=17,
        horizon_weight=torch.tensor([0.25, 0.75]),
        validate_args=False,
    )

    spec = _distribution_loss_to_spec(loss)
    assert spec is not None
    restored = _distribution_loss_from_spec(spec)

    assert restored.distribution == loss.distribution
    assert restored.param_names == loss.param_names
    assert restored.outputsize_multiplier == loss.outputsize_multiplier
    assert restored.output_names == loss.output_names
    assert restored.num_samples == loss.num_samples
    assert restored.return_params == loss.return_params
    assert restored.distribution_kwargs == loss.distribution_kwargs
    torch.testing.assert_close(restored.quantiles, loss.quantiles)
    torch.testing.assert_close(restored.horizon_weight, loss.horizon_weight)

    tampered = copy.deepcopy(spec)
    tampered["contract"]["out_mult"] = 2
    with pytest.raises(ValueError, match="contract does not match"):
        _distribution_loss_from_spec(tampered)


def test_save_model_checkpoint_format(tmp_path):
    model = torch.nn.Linear(4, 2)
    cfg = DummyConfig()
    ckpt_path = tmp_path / "dummy.pt"

    save_model(model, cfg, str(ckpt_path), extra_meta={"runner": "unit-test"})

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    assert ckpt["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert ckpt["meta"]["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert ckpt["meta"]["runner"] == "unit-test"
    assert ckpt["cfg_cls"] == "DummyConfig"
    assert _extract_cfg_obj(ckpt)["lookback"] == 4
    assert set(_extract_state_dict(ckpt).keys()) == set(model.state_dict().keys())


def test_extract_cfg_obj_rejects_ambiguous_legacy_distribution_head_shape():
    ckpt = {
        "model_class": "TitanBaseModel",
        "cfg_state": {"horizon": 2},
        "cfg_cls": "UnknownConfig",
        "state_dict": {
            "head.weight": torch.zeros(2, 4),
            "head.bias": torch.zeros(2),
        },
    }

    with pytest.raises(ValueError, match="distribution-shaped head.*no persisted distribution metadata"):
        _extract_cfg_obj(ckpt)


def test_summarize_results_and_manifest(tmp_path):
    results = {
        "PatchTST": {
            "model": torch.nn.Linear(2, 1),
            "cfg": DummyConfig(),
            "ckpt_path": str(tmp_path / "patchtst.pt"),
            "pretrain_ckpt_path": str(tmp_path / "pretrain.pt"),
            "note": "ok",
        }
    }

    summary = summarize_training_results(results)
    assert "model" not in summary["PatchTST"]
    assert "cfg" not in summary["PatchTST"]
    assert summary["PatchTST"]["ckpt_path"].endswith("patchtst.pt")

    manifest_path = save_training_manifest(
        tmp_path,
        request={"models_to_run": ["patchtst"]},
        results=results,
    )
    manifest = (tmp_path / "training_manifest.json").read_text(encoding="utf-8")

    assert manifest_path.endswith("training_manifest.json")
    assert TRAINING_MANIFEST_VERSION in manifest

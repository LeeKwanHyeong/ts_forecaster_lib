from dataclasses import dataclass

import torch

from modeling_module.utils.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    TRAINING_MANIFEST_VERSION,
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

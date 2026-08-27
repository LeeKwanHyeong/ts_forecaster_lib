from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from tools.reseal_icl_production_checkpoint import (
    load_backbone_contract,
    reseal_checkpoint,
    state_dict_sha256,
    synchronize_production_config,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint() -> dict:
    config = {
        "epochs": 1,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "max_grad_norm": 30.0,
        "training_mode": "qualification",
        "random_seed": None,
        "lookback": 52,
        "horizon": 26,
    }
    return {
        "config": dict(config),
        "cfg_state": dict(config),
        "cfg_cls": "AutoTimesConfig",
        "state_dict": {
            "weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "bias": torch.tensor([1.0, -1.0], dtype=torch.bfloat16),
        },
        "meta": {
            "model_key": "autotimes_base",
            "training_mode": "production_refit",
            "validation_enabled": False,
            "state_selection": "final_epoch",
            "random_seed": 42,
            "epochs": 5,
        },
    }


def test_synchronize_production_config_does_not_mutate_source() -> None:
    source = _checkpoint()

    corrected = synchronize_production_config(
        source,
        expected_model_key="autotimes_base",
        learning_rate=1e-3,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )

    assert source["config"]["training_mode"] == "qualification"
    assert source["config"]["random_seed"] is None
    assert corrected["config"] == corrected["cfg_state"]
    assert corrected["config"]["training_mode"] == "production_refit"
    assert corrected["config"]["random_seed"] == 42
    assert corrected["config"]["epochs"] == 5
    assert corrected["config"]["lr"] == 1e-3
    assert corrected["config"]["weight_decay"] == 0.0
    assert corrected["config"]["max_grad_norm"] == 1.0
    assert state_dict_sha256(corrected["state_dict"]) == state_dict_sha256(
        source["state_dict"]
    )


def test_reseal_checkpoint_preserves_source_and_state_dict(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    destination = tmp_path / "corrected.pt"
    source_receipt = tmp_path / "source-receipt.json"
    correction_receipt = tmp_path / "correction-receipt.json"
    torch.save(_checkpoint(), source)
    source_sha = _file_sha256(source)
    source_receipt.write_text(
        json.dumps({"checkpoint": {"sha256": source_sha}}),
        encoding="utf-8",
    )

    result = reseal_checkpoint(
        source,
        destination,
        source_receipt=source_receipt,
        receipt_path=correction_receipt,
        expected_model_key="autotimes_base",
        learning_rate=1e-3,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )

    assert _file_sha256(source) == source_sha
    assert result["status"] == "PASS"
    assert result["state_dict_unchanged"] is True
    assert result["source_checkpoint"]["sha256"] == source_sha
    assert result["corrected_checkpoint"]["sha256"] == _file_sha256(destination)
    corrected = torch.load(destination, map_location="cpu", weights_only=False)
    assert corrected["config"]["training_mode"] == "production_refit"
    assert corrected["config"]["epochs"] == 5
    assert corrected["meta"]["metadata_correction"]["source_checkpoint_sha256"] == (
        source_sha
    )
    assert correction_receipt.exists()


def test_backbone_manifest_is_verified_and_sealed_into_checkpoint(
    tmp_path: Path,
) -> None:
    manifest = {
        "model_id": "Qwen/Qwen2-0.5B",
        "revision": "revision-r1",
        "files": {},
    }
    canonical = json.dumps(
        manifest,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    manifest["manifest_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    manifest_path = tmp_path / "backbone-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    contract = load_backbone_contract(manifest_path)
    corrected = synchronize_production_config(
        _checkpoint(),
        expected_model_key="autotimes_base",
        learning_rate=1e-3,
        weight_decay=0.0,
        max_grad_norm=1.0,
        backbone_contract=contract,
    )

    assert corrected["config"]["llm_model_name"] == "Qwen/Qwen2-0.5B"
    assert corrected["config"]["llm_revision"] == "revision-r1"
    assert corrected["meta"]["backbone_contract"] == manifest

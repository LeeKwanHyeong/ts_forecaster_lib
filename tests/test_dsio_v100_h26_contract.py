from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import pytest

from tools.dsio_v100_h26_contract import (
    EXPECTED_RESULT_COLUMNS,
    MODEL_SPECS,
    V100H26ContractError,
    canonical_json_sha256,
    file_sha256,
    load_training_input_manifest,
    model_signature_payload,
    validate_checkpoint_contract,
)
from tools.qualify_dsio_v100_h26_checkpoints import (
    load_sealed_registry,
    validate_predictions,
    validate_qualification_receipt,
)
from tools.run_dsio_v100_h26_refit import (
    build_training_command,
    run_refit,
)


HASH_A = "a" * 64
HASH_B = "b" * 64


def _write_manifest(path: Path, target: Path) -> None:
    payload = {
        "format": "demand-engine-v100-h26-production-refit-target",
        "format_version": 1,
        "canonical_run_id": "test-run",
        "source_bundle_sha256": HASH_A,
        "source_binding_sha256": HASH_B,
        "training_contract": {
            "site_cd": "V100",
            "train_end_week": 202509,
            "first_excluded_week": 202510,
            "lookback": 52,
            "horizon": 26,
            "forecast_offsets": {"minimum": 0, "maximum": 25, "count": 26},
            "seed": 42,
            "mode": "production_refit",
            "state_selection": "final_epoch",
        },
        "dataset": {
            "row_count": 78,
            "maximum_week": 202509,
            "null_target_count": 0,
        },
        "artifact": {
            "name": target.name,
            "sha256": file_sha256(target),
            "size_bytes": target.stat().st_size,
        },
    }
    path.write_text(
        json.dumps(
            {
                "payload": payload,
                "payload_sha256": canonical_json_sha256(payload),
            }
        ),
        encoding="utf-8",
    )


def _write_registry(path: Path, checkpoint_root: Path) -> dict[str, object]:
    models = []
    for spec in MODEL_SPECS:
        checkpoint = checkpoint_root / spec.checkpoint_filename
        checkpoint.write_bytes(f"checkpoint:{spec.model_key}".encode("ascii"))
        models.append(
            {
                "model_key": spec.model_key,
                "plan_model_name": spec.plan_model_name,
                "checkpoint": {
                    "path": spec.checkpoint_filename,
                    "sha256": file_sha256(checkpoint),
                },
            }
        )
    payload: dict[str, object] = {
        "dataset": {"train_end_week": 202509, "forecast_origin": 202510},
        "runtime": {
            "frequency": "weekly",
            "lookback": 52,
            "maximum_horizon": 26,
        },
        "models": models,
    }
    sealed = {**payload, "registry_sha256": canonical_json_sha256(payload)}
    path.write_text(json.dumps(sealed), encoding="utf-8")
    return sealed


def test_model_inventory_uses_exact_h26_filenames_and_epochs() -> None:
    assert [spec.model_key for spec in MODEL_SPECS] == [
        "patchtst_base",
        "patchtst_quantile",
        "patchmixer",
        "nhits_base",
        "timemixer",
    ]
    assert [spec.epochs for spec in MODEL_SPECS] == [8, 3, 3, 31, 33]
    assert [spec.checkpoint_filename for spec in MODEL_SPECS] == [
        "weekly_PatchTST_L52_H26.pt",
        "weekly_PatchTSTQuantile_L52_H26.pt",
        "weekly_PatchMixer_L52_H26.pt",
        "weekly_NHITSBase_L52_H26.pt",
        "weekly_TimeMixer_L52_H26.pt",
    ]


def test_training_command_is_explicit_and_does_not_change_generic_defaults(
    tmp_path: Path,
) -> None:
    command = build_training_command(
        python_executable=Path(sys.executable),
        target_source=tmp_path / "target.parquet",
        artifact_root=tmp_path / "training",
        model_key="patchtst_base",
        epochs=8,
        preflight_only=False,
        device="cuda",
        num_workers=8,
    )

    joined = " ".join(command)
    assert "--training-mode production_refit" in joined
    assert "--lookback 52 --horizon 26" in joined
    assert "--train-end-week 202509 --forecast-origin 202510" in joined
    assert "--warmup-epochs 8 --spike-epochs 0" in joined
    assert "--seed 42" in joined


def test_validate_only_seals_commands_without_creating_output(tmp_path: Path) -> None:
    target = tmp_path / "tb_master_target.parquet"
    target.write_bytes(b"sealed target")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, target)
    output = tmp_path / "refit"

    result = run_refit(
        target_source=target,
        input_manifest=manifest,
        output_root=output,
        python_executable=Path(sys.executable),
        device="cuda",
        num_workers=8,
        validate_only=True,
    )

    assert result["status"] == "VALIDATED"
    assert len(result["commands"]) == 5
    assert not output.exists()


def test_training_manifest_rejects_post_cutoff_contract(tmp_path: Path) -> None:
    target = tmp_path / "tb_master_target.parquet"
    target.write_bytes(b"sealed target")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, target)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["payload"]["training_contract"]["train_end_week"] = 202510
    raw["payload_sha256"] = canonical_json_sha256(raw["payload"])
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(V100H26ContractError, match="train_end_week"):
        load_training_input_manifest(manifest, target_source=target)


def test_checkpoint_contract_requires_h26_and_final_epoch() -> None:
    spec = MODEL_SPECS[0]
    checkpoint = {
        "meta": {
            "model_key": spec.model_key,
            "training_mode": "production_refit",
            "validation_enabled": False,
            "state_selection": "final_epoch",
            "configured_epochs": 8,
            "completed_epochs": 8,
            "random_seed": 42,
        },
        "config": {"lookback": 52, "horizon": 26},
    }

    validate_checkpoint_contract(checkpoint, spec=spec)
    checkpoint["config"]["horizon"] = 27
    with pytest.raises(V100H26ContractError, match="L52/H26"):
        validate_checkpoint_contract(checkpoint, spec=spec)


def test_prediction_contract_rejects_w26_row() -> None:
    valid = pl.DataFrame(
        {
            "series_id": ["V100_SMOKE_0001"] * 26,
            "model_key": ["patchtst_base"] * 26,
            "forecast_origin": [202510] * 26,
            "horizon_step": list(range(26)),
            "point": [1.0] * 26,
            "q10": [None] * 26,
            "q50": [None] * 26,
            "q90": [None] * 26,
        }
    )
    validate_predictions(valid, model_key="patchtst_base")

    with pytest.raises(V100H26ContractError, match="27 rows"):
        validate_predictions(
            pl.concat(
                [
                    valid,
                    valid.tail(1).with_columns(
                        pl.lit(26)
                        .cast(valid.schema["horizon_step"])
                        .alias("horizon_step")
                    ),
                ]
            ),
            model_key="patchtst_base",
        )


def test_receipt_validation_recomputes_signatures_and_file_hashes(
    tmp_path: Path,
) -> None:
    target = tmp_path / "tb_master_target.parquet"
    target.write_bytes(b"sealed target")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, target)
    manifest_sha256 = file_sha256(manifest)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    registry_path = tmp_path / "registry.json"
    registry = _write_registry(registry_path, checkpoints)

    models = []
    for definition, spec in zip(registry["models"], MODEL_SPECS, strict=True):
        checkpoint_sha256 = definition["checkpoint"]["sha256"]
        models.append(
            {
                "plan_model_name": spec.plan_model_name,
                "model_key": spec.model_key,
                "checkpoint": {
                    "path": spec.checkpoint_filename,
                    "sha256": checkpoint_sha256,
                },
                "model_signature_sha256": canonical_json_sha256(
                    model_signature_payload(
                        model_key=spec.model_key,
                        checkpoint_sha256=checkpoint_sha256,
                        input_manifest_sha256=manifest_sha256,
                    )
                ),
                "public_forecast": {
                    "candidate_rows": 26,
                    "horizon_step_min": 0,
                    "horizon_step_max": 25,
                    "point_min": 0.0,
                    "point_max": 1.0,
                    "ordered_result_columns": list(EXPECTED_RESULT_COLUMNS),
                },
            }
        )
    receipt: dict[str, object] = {
        "receipt_format_version": 1,
        "status": "PASS",
        "scope": {
            "site_cd": "V100",
            "train_end_week": 202509,
            "forecast_origin": 202510,
            "lookback": 52,
            "horizon": 26,
            "offsets": "W0-W25",
        },
        "registry": {
            "path": registry_path.name,
            "file_sha256": file_sha256(registry_path),
            "registry_sha256": registry["registry_sha256"],
        },
        "training_input_manifest": {
            "path": manifest.name,
            "sha256": manifest_sha256,
        },
        "runtime": {
            "modeling_module_version": "test",
            "device": "cpu",
            "db_write_enabled": False,
        },
        "models": models,
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="ascii")

    validate_qualification_receipt(
        receipt_path=receipt_path,
        registry_path=registry_path,
        checkpoint_root=checkpoints,
        input_manifest_path=manifest,
    )

    receipt["models"][0]["model_signature_sha256"] = HASH_A
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256")
    receipt["receipt_sha256"] = canonical_json_sha256(unsigned)
    receipt_path.write_text(json.dumps(receipt), encoding="ascii")
    with pytest.raises(V100H26ContractError, match="model signature"):
        validate_qualification_receipt(
            receipt_path=receipt_path,
            registry_path=registry_path,
            checkpoint_root=checkpoints,
            input_manifest_path=manifest,
        )


def test_registry_loader_rejects_h27_filename(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    registry_path = tmp_path / "registry.json"
    registry = _write_registry(registry_path, checkpoints)
    registry["models"][0]["checkpoint"]["path"] = "weekly_PatchTST_L52_H27.pt"
    unsigned = dict(registry)
    unsigned.pop("registry_sha256")
    registry["registry_sha256"] = canonical_json_sha256(unsigned)
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(V100H26ContractError, match="inventory drifted"):
        load_sealed_registry(registry_path)

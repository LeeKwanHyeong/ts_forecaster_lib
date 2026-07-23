from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "evaluate_dsio_qualification.py"
MODULE_NAME = "_dsio_qualification_evaluator"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_metric_values_use_micro_wape_and_zero_safe_smape():
    metrics = MODULE.metric_values(
        np.asarray([0.0, 2.0]),
        np.asarray([0.0, 4.0]),
    )

    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["wape"] == pytest.approx(1.0)
    assert metrics["smape"] == pytest.approx(1.0 / 3.0)


def test_metric_values_reject_shape_mismatch_and_nonfinite_values():
    with pytest.raises(ValueError, match="same shape"):
        MODULE.metric_values(np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="finite"):
        MODULE.metric_values(np.asarray([1.0]), np.asarray([np.nan]))


def test_training_log_parser_scopes_epochs_to_exact_model_headings(tmp_path):
    log_path = tmp_path / "qualification.log"
    log_path.write_text(
        "\n".join(
            [
                "PatchTST (Weekly)",
                "Epoch 1/3 | LR 0.001000 | Train 2.000000 | Val 3.000000",
                "Epoch 2/3 | LR 0.000500 | Train 1.000000 | Val 1.500000",
                "Epoch 3/3 | LR 0.000100 | Train 0.500000 | Val 2.000000",
                "PatchTST Quantile (Weekly)",
                "Epoch 1/2 | LR 0.001000 | Train 1.000000 | Val 0.800000",
                "Epoch 2/2 | LR 0.000100 | Train 0.700000 | Val 0.900000",
            ]
        ),
        encoding="utf-8",
    )

    histories = MODULE.parse_training_log(log_path)

    assert list(histories) == ["patchtst_base", "patchtst_quantile"]
    assert [record.epoch for record in histories["patchtst_base"]] == [1, 2, 3]
    assert histories["patchtst_quantile"][0].validation_loss == 0.8


def test_refit_policy_uses_best_validation_epoch_and_checks_manifest(tmp_path):
    log_path = tmp_path / "qualification.log"
    log_path.write_text(
        "\n".join(
            [
                "PatchMixer (Weekly) mode=point",
                "Epoch 1/3 | LR 0.001000 | Train 2.000000 | Val 3.000000",
                "Epoch 2/3 | LR 0.000500 | Train 1.000000 | Val 1.250000",
                "Epoch 3/3 | LR 0.000100 | Train 0.500000 | Val 1.500000",
            ]
        ),
        encoding="utf-8",
    )
    histories = MODULE.parse_training_log(log_path)
    manifest = {"results": {"patchmixer": {"best_val_loss": 1.2500004}}}

    policy = MODULE.build_refit_policy(
        histories=histories,
        training_manifest=manifest,
        model_keys=["patchmixer"],
    )

    assert policy[0]["production_refit_epochs"] == 2
    assert policy[0]["qualification_total_epochs"] == 3
    assert "do not early-stop" in policy[0]["refit_contract"]

    manifest["results"]["patchmixer"]["best_val_loss"] = 1.3
    with pytest.raises(ValueError, match="differs"):
        MODULE.build_refit_policy(
            histories=histories,
            training_manifest=manifest,
            model_keys=["patchmixer"],
        )


def test_checkpoint_resolution_falls_back_to_portable_artifact_name(tmp_path):
    checkpoint = tmp_path / "weekly_PatchMixer_L52_H27.pt"
    checkpoint.write_bytes(b"checkpoint")

    resolved = MODULE._resolve_checkpoint_path(
        tmp_path,
        {"ckpt_path": "/old/server/weekly_PatchMixer_L52_H27.pt"},
    )

    assert resolved == checkpoint


def test_parser_requires_artifact_directory_and_training_log():
    parser = MODULE.build_parser()
    args = parser.parse_args(
        [
            "--artifact-dir",
            "artifacts/qualification/endo_only",
            "--training-log",
            "logs/qualification.log",
        ]
    )

    assert args.device == "cuda"
    assert args.batch_size == 1024
    assert args.num_workers == 4
    assert args.pin_memory is True

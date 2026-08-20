from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from modeling_module.data_loader.temporal import add_period
from tools.dsio_v100_h26_contract import (
    FORECAST_ORIGIN,
    HORIZON,
    LOOKBACK,
    TRAIN_END_WEEK,
    V100H26ContractError,
)
from tools.run_dsio_v100_h26_exogenous_qualification import MODEL_SPECS_BY_KEY
from tools.run_dsio_v100_h26_exogenous_refit import (
    EPOCH_POLICY_EVIDENCE,
    PRODUCTION_MODEL_SPECS,
    PRODUCTION_REFIT_EPOCHS,
    PRODUCTION_REFIT_SEED,
    _build_datamodule,
    _production_canary_batch,
    _validate_point_canary_output,
    build_worker_command,
)


def _h26_frame() -> pl.DataFrame:
    weeks = [
        add_period(TRAIN_END_WEEK, -offset, "weekly")
        for offset in reversed(range(120))
    ]
    rows = []
    for part_index, part_id in enumerate(("A", "B")):
        for week_index, week in enumerate(weeks):
            rows.append(
                {
                    "oper_part_no": part_id,
                    "demand_dt": week,
                    "demand_qty": float(part_index + week_index % 5),
                }
            )
    return pl.DataFrame(rows)


def test_production_epoch_and_seed_policy_is_frozen():
    assert PRODUCTION_REFIT_SEED == 42
    assert PRODUCTION_REFIT_EPOCHS == {
        "exotst_base": 40,
        "patchtst_exogenous": 35,
    }
    assert tuple(spec.model_key for spec in PRODUCTION_MODEL_SPECS) == (
        "exotst_base",
        "patchtst_exogenous",
    )
    assert EPOCH_POLICY_EVIDENCE["selection_rule"] == (
        "lowest_four_seed_mean_validation_loss_by_epoch"
    )
    assert EPOCH_POLICY_EVIDENCE["qualification_seeds"] == [11, 22, 33, 42]


def test_production_split_trains_through_202509_without_validation():
    datamodule = _build_datamodule(
        _h26_frame(),
        spec=MODEL_SPECS_BY_KEY["exotst_base"],
        training_mode="production_refit",
    )
    summary = datamodule.summary

    assert summary["train_target_max_week"] == TRAIN_END_WEEK
    assert summary["validation_windows"] == 0
    assert summary["validation_target_min_week"] is None
    assert summary["validation_target_max_week"] is None
    assert datamodule.val_dataset is None


def test_production_canary_uses_202510_origin_and_h26_calendar():
    spec = MODEL_SPECS_BY_KEY["patchtst_exogenous"]
    datamodule = _build_datamodule(
        _h26_frame(),
        spec=spec,
        training_mode="production_refit",
    )
    batch, evidence = _production_canary_batch(datamodule, spec=spec)

    assert batch[0].shape == (2, LOOKBACK, 1)
    assert batch[3].shape == (2, HORIZON, 12)
    assert batch[4].shape == (2, LOOKBACK, 12)
    assert evidence["history_end_week"] == TRAIN_END_WEEK
    assert evidence["forecast_start_week"] == FORECAST_ORIGIN
    assert evidence["forecast_end_week"] == add_period(
        FORECAST_ORIGIN,
        HORIZON - 1,
        "weekly",
    )


def test_worker_command_uses_governed_policy_without_epoch_or_seed_override():
    command = build_worker_command(
        python_executable=Path("/opt/python"),
        target_source=Path("/data/target.parquet"),
        input_manifest=Path("/data/manifest.json"),
        output_root=Path("/artifacts/refit"),
        model_key="patchtst_exogenous",
        batch_size=512,
        num_workers=8,
        device="cuda",
        sample_part_count=8,
        preflight_only=True,
    )

    assert command[0] == "/opt/python"
    assert command[command.index("--model-key") + 1] == "patchtst_exogenous"
    assert "--epochs" not in command
    assert "--seed" not in command
    assert command[-1] == "--preflight-only"


@pytest.mark.parametrize("shape", [(52,), (2, 26)])
def test_point_canary_accepts_public_flat_and_matrix_outputs(shape):
    points, matrix_shape = _validate_point_canary_output(
        np.zeros(shape, dtype=np.float32),
        batch_size=2,
    )

    assert points.shape == shape
    assert matrix_shape == (2, HORIZON)


def test_point_canary_rejects_wrong_output_size():
    with pytest.raises(V100H26ContractError, match="expected 52 finite points"):
        _validate_point_canary_output(
            np.zeros((2, HORIZON - 1), dtype=np.float32),
            batch_size=2,
        )


def test_worker_command_can_resume_completed_training_artifact():
    command = build_worker_command(
        python_executable=Path("/opt/python"),
        target_source=Path("/data/target.parquet"),
        input_manifest=Path("/data/manifest.json"),
        output_root=Path("/artifacts/refit"),
        model_key="exotst_base",
        batch_size=512,
        num_workers=8,
        device="cuda",
        sample_part_count=None,
        preflight_only=False,
        resume_existing=True,
    )

    assert command[-1] == "--resume-existing"

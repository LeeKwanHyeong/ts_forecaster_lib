from __future__ import annotations

from pathlib import Path

import polars as pl

from modeling_module.data_loader.deterministic_calendar import (
    WEEKLY_CALENDAR_CONTINUOUS_FEATURES,
)
from modeling_module.data_loader.temporal import add_period
from tools.run_dsio_v100_h26_exogenous_qualification import (
    HORIZON,
    LOOKBACK,
    MODEL_SPECS,
    MODEL_SPECS_BY_KEY,
    TRAIN_END_WEEK,
    VALIDATION_ORIGIN,
    _batch_contract,
    _build_datamodule,
    build_architecture,
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


def test_approved_exogenous_model_inventory_and_checkpoint_names():
    assert tuple(spec.model_key for spec in MODEL_SPECS) == (
        "exotst_base",
        "timexer_base",
        "patchtst_exogenous",
    )
    assert tuple(spec.checkpoint_filename for spec in MODEL_SPECS) == (
        "weekly_ExoTSTBase_L52_H26.pt",
        "weekly_TimeXerBase_L52_H26.pt",
        "weekly_PatchTSTExogenous_L52_H26.pt",
    )
    assert MODEL_SPECS_BY_KEY["timexer_base"].uses_future_continuous is False
    assert "weather_index" not in WEEKLY_CALENDAR_CONTINUOUS_FEATURES
    assert "macro_index" not in WEEKLY_CALENDAR_CONTINUOUS_FEATURES
    assert "promo_flag" not in WEEKLY_CALENDAR_CONTINUOUS_FEATURES


def test_h26_exogenous_split_and_batch_match_endogenous_calendar():
    spec = MODEL_SPECS_BY_KEY["exotst_base"]
    datamodule = _build_datamodule(_h26_frame(), spec=spec)
    summary = datamodule.summary

    assert summary["source_max_week"] == TRAIN_END_WEEK
    assert summary["train_target_max_week"] == 202435
    assert summary["validation_target_min_week"] == VALIDATION_ORIGIN
    assert summary["validation_target_max_week"] == TRAIN_END_WEEK
    assert summary["validation_windows"] == 2
    assert summary["past_cont_dim"] == 12
    assert summary["future_cont_dim"] == 12

    batch = next(iter(datamodule.get_val_loader(batch_size=2)))
    evidence = _batch_contract(batch, spec=spec)
    assert evidence["x_shape"] == [2, LOOKBACK, 1]
    assert evidence["y_shape"] == [2, HORIZON]
    assert evidence["past_cont_shape"] == [2, LOOKBACK, 12]
    assert evidence["future_cont_shape"] == [2, HORIZON, 12]


def test_timexer_h26_split_is_past_only():
    spec = MODEL_SPECS_BY_KEY["timexer_base"]
    datamodule = _build_datamodule(_h26_frame(), spec=spec)
    batch = next(iter(datamodule.get_val_loader(batch_size=2)))

    assert datamodule.summary["future_cont_dim"] == 0
    assert batch[3].shape == (2, HORIZON, 0)
    assert datamodule.exogenous_schema.future_cont_names == ()


def test_worker_command_carries_governed_runtime_values():
    command = build_worker_command(
        python_executable=Path("/opt/python"),
        target_source=Path("/data/target.parquet"),
        input_manifest=Path("/data/manifest.json"),
        output_root=Path("/artifacts/exo"),
        model_key="exotst_base",
        epochs=40,
        batch_size=512,
        num_workers=8,
        device="cuda",
        sample_part_count=8,
        preflight_only=True,
    )

    assert command[0] == "/opt/python"
    assert command[command.index("--model-key") + 1] == "exotst_base"
    assert command[command.index("--epochs") + 1] == "40"
    assert command[command.index("--sample-part-count") + 1] == "8"
    assert command[-1] == "--preflight-only"


def test_h26_architecture_is_frozen_for_all_three_families():
    architecture = build_architecture()

    assert architecture.patchtst.patch_len == 13
    assert architecture.patchtst.stride == 6
    assert architecture.patchtst.d_model == 128
    assert architecture.exotst.patch_len == 13
    assert architecture.exotst.stride == 6
    assert architecture.exotst.d_model == 128
    assert architecture.timexer.patch_len == 13
    assert LOOKBACK % architecture.timexer.patch_len == 0

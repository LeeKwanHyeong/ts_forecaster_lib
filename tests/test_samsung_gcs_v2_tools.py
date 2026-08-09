from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import pytest

from model_test.exogenous_test.far_metrics import compute_revision_far
from model_test.exogenous_test.run_samsung_gcs_patchtst_sweep import (
    ARCHITECTURE_CASES,
    build_sweep_runs,
)


def _sweep_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        artifact_root=tmp_path,
        plan_weeks=[202538, 202539],
        architectures=["arch_small", "arch_base"],
        batch_sizes=[64],
        seeds=[11, 22],
        lookback=52,
        horizon=27,
        sample_part_count=128,
        warmup_epochs=2,
        spike_epochs=1,
        num_workers=4,
        device="cpu",
        future_exo_source="columns",
        clean_output=False,
    )


def test_samsung_sweep_builds_unique_modern_ab_commands(tmp_path: Path) -> None:
    runs = build_sweep_runs(_sweep_args(tmp_path))

    assert len(runs) == 8
    assert len({run.case_name for run in runs}) == len(runs)
    first = list(runs[0].command)
    assert "patchtst_no_future" in first
    assert "patchtst_token_cross_attn" in first
    assert "head_flatten" not in first
    assert "--patchtst-d-model" in first
    assert str(ARCHITECTURE_CASES["arch_small"].d_model) in first
    assert not any("E:/" in value for value in first)


def test_revision_far_preserves_eight_revision_weighting() -> None:
    rows: list[dict[str, object]] = []
    for plan_week in range(202501, 202509):
        rows.extend(
            [
                {
                    "model_name": "patchtst_token_cross_attn",
                    "part_no": "A",
                    "plan_week": plan_week,
                    "forecast_week": 202509,
                    "prediction": 10.0,
                    "actual": 10.0,
                },
                {
                    "model_name": "patchtst_token_cross_attn",
                    "part_no": "B",
                    "plan_week": plan_week,
                    "forecast_week": 202509,
                    "prediction": 5.0,
                    "actual": 11.0,
                },
            ]
        )

    result = compute_revision_far(pl.DataFrame(rows))

    assert result.height == 1
    assert result.item(0, "row_count") == 2
    assert result.item(0, "nonzero_row_count") == 2
    assert result.item(0, "fcst_qty_total") == pytest.approx(15.0)
    assert result.item(0, "weighted_far") == pytest.approx(2.0 / 3.0)


def test_revision_far_excludes_zero_forecast_parts() -> None:
    rows: list[dict[str, object]] = []
    for plan_week in range(202501, 202509):
        rows.extend(
            [
                {
                    "model_name": "patchtst_no_future",
                    "part_no": "A",
                    "plan_week": plan_week,
                    "forecast_week": 202509,
                    "prediction": 10.0,
                    "actual": 10.0,
                },
                {
                    "model_name": "patchtst_no_future",
                    "part_no": "B",
                    "plan_week": plan_week,
                    "forecast_week": 202509,
                    "prediction": 0.0,
                    "actual": 10.0,
                },
            ]
        )

    result = compute_revision_far(pl.DataFrame(rows))

    assert result.item(0, "row_count") == 2
    assert result.item(0, "nonzero_row_count") == 1
    assert result.item(0, "weighted_far") == pytest.approx(1.0)


def test_revision_far_rejects_duplicate_revision_rows() -> None:
    frame = pl.DataFrame(
        {
            "model_name": ["patchtst", "patchtst"],
            "part_no": ["A", "A"],
            "plan_week": [202501, 202501],
            "forecast_week": [202509, 202509],
            "prediction": [10.0, 10.0],
            "actual": [10.0, 10.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        compute_revision_far(frame)

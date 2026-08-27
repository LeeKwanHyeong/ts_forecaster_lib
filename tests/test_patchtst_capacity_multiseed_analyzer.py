from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "analyze_patchtst_capacity_multiseed.py"
MODULE_NAME = "_patchtst_capacity_multiseed_analyzer"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_run_spec_requires_capacity_seed_artifact_and_log(tmp_path):
    spec = MODULE.parse_run_spec(
        f"Small,11,{tmp_path / 'artifact'},{tmp_path / 'train.log'}"
    )

    assert spec.capacity == "small"
    assert spec.seed == 11
    assert spec.artifact_dir == (tmp_path / "artifact").resolve()

    with pytest.raises(Exception, match="run spec"):
        MODULE.parse_run_spec("small,11,artifact")
    with pytest.raises(Exception, match="integer"):
        MODULE.parse_run_spec("small,seed,artifact,train.log")


def test_build_demand_cohorts_matches_adi_cv2_contract():
    rows = []
    values = {
        "dense": [1.0, 2.0, 1.0, 2.0],
        "intermittent": [0.0, 2.0, 0.0, 2.0],
        "no_demand": [0.0, 0.0, 0.0, 0.0],
        "insufficient": [0.0, 0.0, 0.0, 2.0],
    }
    for part, series in values.items():
        for index, value in enumerate(series, start=1):
            rows.append(
                {
                    "oper_part_no": part,
                    "demand_dt": 202500 + index,
                    "demand_qty": value,
                }
            )
    target = pl.DataFrame(rows)

    cohorts = MODULE.build_demand_cohorts(
        target,
        train_cutoff=202504,
        min_periods=3,
    )
    by_part = {
        row["oper_part_no"]: row
        for row in cohorts.select(
            "oper_part_no", "demand_type", "cohort"
        ).to_dicts()
    }

    assert by_part["dense"]["cohort"] == "dense"
    assert by_part["intermittent"]["cohort"] == "intermittent"
    assert by_part["no_demand"]["cohort"] == "no_demand"
    assert by_part["insufficient"]["cohort"] == "insufficient"


def test_select_capacity_and_refit_epoch_uses_mean_mae_and_earliest_minimum():
    capacity_summary = pl.DataFrame(
        {
            "capacity": ["current", "small"],
            "parameter_count": [1000, 100],
            "mae_mean": [10.0, 9.0],
            "wape_mean": [0.6, 0.5],
            "smape_mean": [1.4, 1.3],
        }
    )
    epoch_summary = pl.DataFrame(
        {
            "capacity": ["small", "small", "small", "current"],
            "epoch": [1, 2, 3, 1],
            "validation_loss_mean": [3.0, 2.0, 2.0, 1.0],
        }
    )

    policy = MODULE.select_capacity_and_refit_epoch(
        capacity_summary,
        epoch_summary,
    )

    assert policy["selected_capacity"] == "small"
    assert policy["production_refit_epochs"] == 2


def test_summarize_pairwise_counts_candidate_seed_wins():
    frame = pl.DataFrame(
        {
            "cohort": ["dense", "dense", "intermittent", "intermittent"],
            "seed": [11, 22, 11, 22],
            "candidate_mae": [1.0, 2.0, 4.0, 5.0],
            "control_mae": [2.0, 1.0, 3.0, 6.0],
            "mae_delta": [-1.0, 1.0, 1.0, -1.0],
        }
    )

    summary = MODULE.summarize_pairwise(
        frame,
        dimensions=("cohort",),
    )

    assert summary["seed_count"].to_list() == [2, 2]
    assert summary["candidate_seed_wins"].to_list() == [1, 1]

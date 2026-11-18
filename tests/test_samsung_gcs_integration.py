from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from model_test.exogenous_test.auto_run_architecture_hyper_plot_utils import (
    _ordered_week_axis,
    save_sample_lineplots_for_plan_week_long_3col,
)
from model_test.exogenous_test.auto_run_architecture_hyper_validation_utils import evaluate_sweep, load_sweep_forecasts
from model_test.exogenous_test.far_metrics import compute_revision_far
from model_test.exogenous_test.run_samsung_gcs_patchtst_sweep import build_parser, build_sweep_runs, write_manifest
from model_test.patchtst_loss_comparison.common import SETTINGS, scores, sha256
from model_test.patchtst_loss_comparison.run import resolve_clean_commit, validate_cached_summary, validate_source_override


ROOT = Path(__file__).resolve().parents[1]


def iso_week(day):
    year, week, _ = day.isocalendar()
    return year * 100 + week


def sweep_args(root, *, seeds=(42,)):
    # 2020년의 53주에서 다음 해 1주로 넘어가는 실제 달력 구간.
    weeks = [iso_week(date(2020, 11, 16) + timedelta(weeks=i)) for i in range(8)]
    return build_parser().parse_args([
        "--artifact-root", str(root), "--plan-weeks", *map(str, weeks),
        "--architectures", "arch_small", "--seeds", *map(str, seeds), "--num-workers", "0",
    ])


def completed_sweep(root, *, seeds=(42,)):
    args = sweep_args(root, seeds=seeds)
    runs = build_sweep_runs(args)
    manifest = write_manifest(args, runs)
    for run in runs:
        start = date.fromisocalendar(run.plan_week // 100, run.plan_week % 100, 1)
        rows = [{
            "model_name": model, "part_no": part, "plan_week": run.plan_week,
            "forecast_week": iso_week(start + timedelta(weeks=step)),
            "prediction": 10.0 if run.seed == 42 else 20.0, "actual": 10.0,
        } for model in ("patchtst_no_future", "patchtst_token_cross_attn")
            for part in ("synthetic_a", "synthetic_b") for step in range(args.horizon)]
        path = Path(run.artifact_root) / "forecasts" / "all_models_forecast_long.parquet"
        path.parent.mkdir(parents=True)
        pl.DataFrame(rows).write_parquet(path)
    return manifest, runs


def test_week_axis_crosses_year_without_numeric_gap():
    weeks = list(range(202538, 202553)) + list(range(202601, 202614))
    x, labels, ticks, tick_labels = _ordered_week_axis(weeks, max_ticks=12)
    np.testing.assert_array_equal(np.diff(x), np.ones(len(weeks) - 1))
    assert labels[14:16] == ["202552", "202601"]
    assert len(ticks) == 12
    assert tick_labels[0] == "202538" and tick_labels[-1] == "202613"


@pytest.mark.parametrize("weeks,max_ticks", [([202553], 12), ([202501.5], 12), ([202501], 1)])
def test_week_axis_rejects_invalid_contract(weeks, max_ticks):
    with pytest.raises(ValueError):
        _ordered_week_axis(weeks, max_ticks=max_ticks)


def test_point_forecast_plot_does_not_require_quantile_column(tmp_path):
    frame = pl.DataFrame({"part_no": ["synthetic_a"] * 2, "yyyyww": [202053, 202101],
                          "demand_qty": [10.0, 11.0], "base_forecast": [9.0, 12.0]})
    save_sample_lineplots_for_plan_week_long_3col(frame, ["synthetic_a"], 202053, tmp_path)
    path = tmp_path / "plan_202053_sample_parts_3col.png"
    assert path.is_file() and plt.imread(path).size > 0
    assert not plt.get_fignums()


def test_sweep_forwards_sources_and_records_explicit_metadata(tmp_path):
    args = sweep_args(tmp_path)
    args.target_source = tmp_path / "target.parquet"
    args.exo_source = tmp_path / "exo.parquet"
    runs = build_sweep_runs(args)
    payload = json.loads(write_manifest(args, runs).read_text())
    assert payload["format_version"] == 2 and payload["horizon"] == 27
    assert payload["runs"][0]["architecture"] == "arch_small"
    assert payload["runs"][0]["plan_week"] == args.plan_weeks[0]
    assert "--target-source" in runs[0].command and str(args.exo_source) in runs[0].command
    write_manifest(args, runs)  # 같은 설정의 재확인은 기존 증적을 변경하지 않음.
    args.scope_label = "another_scope"
    with pytest.raises(FileExistsError):
        write_manifest(args, runs)


@pytest.mark.parametrize("field,values", [("plan_weeks", [202553]), ("seeds", [42, 42]), ("batch_sizes", [0]), ("architectures", [])])
def test_sweep_rejects_invalid_runs(tmp_path, field, values):
    args = sweep_args(tmp_path)
    setattr(args, field, values)
    with pytest.raises(ValueError):
        build_sweep_runs(args)


def test_completed_sweep_aggregates_each_seed_separately_and_renders(tmp_path):
    manifest, _ = completed_sweep(tmp_path / "runs", seeds=(42, 142))
    far, summary = evaluate_sweep(manifest, tmp_path / "evaluation")
    assert far.height == 4 and summary.height == 4
    assert far["target_week"].unique().to_list() == [202101]
    assert set(far.filter(pl.col("seed") == 42)["weighted_far"]) == {1.0}
    assert set(far.filter(pl.col("seed") == 142)["weighted_far"]) == {0.5}
    assert list((tmp_path / "evaluation/plots").glob("*.png"))
    assert not plt.get_fignums()


def test_missing_plan_result_is_not_silently_skipped(tmp_path):
    manifest, runs = completed_sweep(tmp_path / "runs")
    (Path(runs[-1].artifact_root) / "forecasts/all_models_forecast_long.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        evaluate_sweep(manifest, tmp_path / "evaluation", make_plots=False)
    assert not (tmp_path / "evaluation").exists()


@pytest.mark.parametrize("column,value", [("model_name", "unknown"), ("plan_week", 202001)])
def test_manifest_and_forecast_mismatch_rejected(tmp_path, column, value):
    manifest, runs = completed_sweep(tmp_path / "runs")
    path = Path(runs[0].artifact_root) / "forecasts/all_models_forecast_long.parquet"
    pl.read_parquet(path).with_columns(pl.lit(value).alias(column)).write_parquet(path)
    with pytest.raises(ValueError):
        load_sweep_forecasts(manifest)


def test_old_manifest_requires_explicit_regeneration(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"format_version": 1}')
    with pytest.raises(ValueError, match="version-2"):
        load_sweep_forecasts(path)


def test_negative_clipping_is_explicit_and_raw_forecast_preserved(tmp_path):
    manifest, runs = completed_sweep(tmp_path / "runs")
    path = Path(runs[0].artifact_root) / "forecasts/all_models_forecast_long.parquet"
    frame = pl.read_parquet(path).with_columns(
        pl.when(pl.col("part_no") == "synthetic_a").then(-1.0).otherwise(pl.col("prediction")).alias("prediction"))
    frame.write_parquet(path)
    with pytest.raises(ValueError, match="non-negative"):
        evaluate_sweep(manifest, tmp_path / "rejected", make_plots=False)
    evaluate_sweep(manifest, tmp_path / "clipped", clip_negative=True, make_plots=False)
    policy = json.loads((tmp_path / "clipped/evaluation_policy.json").read_text())
    assert policy["forecast_postprocess"] == "clip_negative_to_zero"
    assert policy["raw_negative_prediction_count"] > 0
    assert pl.read_parquet(tmp_path / "clipped/raw_forecasts.parquet")["prediction"].min() == -1


def far_frame():
    return pl.DataFrame({"model_name": ["patchtst"] * 8, "part_no": ["synthetic_a"] * 8,
                         "plan_week": list(range(202501, 202509)), "forecast_week": [202509] * 8,
                         "prediction": [10.0] * 8, "actual": [10.0] * 8})


@pytest.mark.parametrize("column,value", [("prediction", float("nan")), ("prediction", float("inf")),
    ("prediction", -1.0), ("prediction", None), ("actual", -1.0), ("actual", float("inf")),
    ("part_no", ""), ("model_name", None), ("plan_week", 202501.5), ("plan_week", 202553)])
def test_far_rejects_invalid_inputs(column, value):
    frame = far_frame().with_columns(pl.lit(value).alias(column))
    with pytest.raises(ValueError):
        compute_revision_far(frame)


def test_far_does_not_bridge_missing_calendar_week():
    frame = far_frame().with_columns(pl.Series("plan_week", [202501, 202502, 202503, 202504, 202505, 202506, 202508, 202509]))
    assert compute_revision_far(frame).is_empty()


def test_unknown_actual_is_not_replaced_by_zero():
    assert compute_revision_far(far_frame().with_columns(pl.lit(None).alias("actual"))).is_empty()


def test_cached_loss_result_must_match_code_and_dataset():
    expected = {"code_commit": "abc", "data_sha256": "def", "seed": 42, "pretrain_lr": 0.001,
                "smoke": True, "settings": {**SETTINGS, "pretrain_lr": 0.001}}
    kwargs = dict(code_commit="abc", data_sha256="def", seed=42, lr=0.001, smoke=True)
    validate_cached_summary(expected, **kwargs)
    for key in expected:
        with pytest.raises(ValueError, match="mismatch"):
            validate_cached_summary({**expected, key: "changed"}, **kwargs)
    with pytest.raises(ValueError):
        scores([], [])


def test_historical_entrypoint_help_uses_v2_cli():
    result = subprocess.run([sys.executable, str(ROOT / "src/model_test/exogenous_test/auto_run_architecture_hyper.py"), "--help"],
                            cwd=ROOT, text=True, capture_output=True, check=True)
    assert "--target-source" in result.stdout and "--execute" in result.stdout


def test_comparison_requires_clean_code_worktree(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "check_output", lambda command, **kwargs: " M dirty.py" if command[1] == "status" else "abc")
    with pytest.raises(ValueError, match="must be clean"):
        resolve_clean_commit(tmp_path)
    monkeypatch.setattr(subprocess, "check_output", lambda command, **kwargs: "" if command[1] == "status" else "abc")
    assert resolve_clean_commit(tmp_path) == "abc"


def test_source_override_cannot_reuse_stale_dataset(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("first")
    manifest = {"source": str(source), "source_sha256": sha256(source)}
    validate_source_override(manifest, source)
    source.write_text("changed")
    with pytest.raises(ValueError, match="differs"):
        validate_source_override(manifest, source)


def test_native_v2_pretrain_finetune_comparison_smoke(tmp_path):
    # 학습·저장·복원 경로만 소규모 합성 데이터로 확인하며 운영 데이터 미사용.
    rng = np.random.default_rng(42)
    arrays = {"parts": np.array(["synthetic_a", "synthetic_b"])}
    for split in ("train", "val", "test"):
        arrays.update({f"{split}_x": rng.uniform(1, 5, (2, 52, 1)).astype("float32"),
                       f"{split}_y": rng.uniform(1, 5, (2, 27, 1)).astype("float32"),
                       f"{split}_f": np.zeros((2, 27, 4), dtype="float32"), f"{split}_part": np.arange(2)})
    data = tmp_path / "data.npz"
    np.savez_compressed(data, **arrays)
    output = tmp_path / "native"
    result = subprocess.run([sys.executable, "-m", "model_test.patchtst_loss_comparison.worker",
        "--code-root", str(ROOT), "--data", str(data), "--output", str(output), "--variant", "after", "--seed", "42", "--smoke"],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["smoke"] and summary["test_metrics"]["n"] == 54
    assert summary["pretrain_best_epoch"] > 0
    assert np.isfinite(summary["pretrain_best_val"]) and np.isfinite(summary["finetune_best_val"])

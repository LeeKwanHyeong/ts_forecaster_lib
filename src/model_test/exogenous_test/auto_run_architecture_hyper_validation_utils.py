"""v2 long-format 예측 결과를 검증하고 삼성 FAR와 비교 그래프 생성.

학습이나 모델 로딩 없이 저장된 sweep manifest와 parquet만 사용한다.
서로 다른 아키텍처, 배치 크기, 시드의 예측 갱신 이력은 섞지 않는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from model_test.exogenous_test.far_metrics import compute_revision_far


FORECAST_COLUMNS = ["model_name", "part_no", "plan_week", "forecast_week", "prediction", "actual"]
GROUP_COLUMNS = ["architecture", "batch_size", "seed"]


def load_sweep_forecasts(manifest_path: Path) -> tuple[dict, pl.DataFrame]:
    """모든 계획 주차의 결과가 존재하는지 먼저 확인한 뒤 결합."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("format_version") != 2:
        raise ValueError("A version-2 sweep manifest with explicit run metadata is required")
    runs = manifest.get("runs", [])
    if not runs:
        raise ValueError("Sweep manifest contains no runs")
    expected_models = set(manifest["models"])
    frames = []
    seen = set()
    for run in runs:
        key = tuple(run[name] for name in GROUP_COLUMNS + ["plan_week"])
        if key in seen:
            raise ValueError(f"Duplicate sweep run: {key}")
        seen.add(key)
        path = Path(run["artifact_root"]) / "forecasts" / "all_models_forecast_long.parquet"
        frame = pl.read_parquet(path)
        missing = sorted(set(FORECAST_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing forecast columns in {path}: {missing}")
        if set(frame["model_name"].to_list()) != expected_models:
            raise ValueError(f"Missing or unexpected models in {path}")
        if set(frame["plan_week"].to_list()) != {run["plan_week"]}:
            raise ValueError(f"plan_week does not match manifest in {path}")
        frames.append(frame.select(FORECAST_COLUMNS).with_columns([
            pl.lit(run[name]).alias(name) for name in GROUP_COLUMNS
        ]))
    return manifest, pl.concat(frames, how="vertical_relaxed")


def evaluate_sweep(
    manifest_path: Path,
    output_dir: Path,
    *,
    clip_negative: bool = False,
    make_plots: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """동일 실험의 연속된 8회 갱신만 FAR로 집계하고 원 예측 별도 보존."""
    manifest, forecasts = load_sweep_forecasts(manifest_path)
    negative_count = forecasts.filter(pl.col("prediction") < 0).height
    evaluated = forecasts
    if clip_negative:
        # 음수 절단은 명시적으로 요청한 경우만 적용하고 평가 정책에 기록한다.
        evaluated = forecasts.with_columns(pl.col("prediction").clip(lower_bound=0))
    rows = []
    for group, frame in evaluated.partition_by(GROUP_COLUMNS, as_dict=True).items():
        plan_weeks = frame["plan_week"].unique().to_list()
        far = compute_revision_far(frame).filter(pl.col("target_week").is_in(plan_weeks))
        if not far.is_empty():
            rows.append(far.with_columns([
                pl.lit(value).alias(name) for name, value in zip(GROUP_COLUMNS, group)
            ]))
    if not rows:
        raise ValueError("No FAR target has eight consecutive forecast revisions; provide more completed plan weeks")
    far_by_week = pl.concat(rows).with_columns([
        pl.col("model_name").alias("ab_case_name"),
        pl.col("architecture").alias("arch_case_name"),
        pl.lit(manifest.get("scope_label", "samsung_gcs")).alias("plant"),
    ]).sort(GROUP_COLUMNS + ["model_name", "target_week"])
    summary = far_by_week.group_by(GROUP_COLUMNS + ["model_name", "ab_case_name", "arch_case_name", "plant"]).agg(
        pl.col("weighted_far").mean().alias("far_mean"),
        pl.col("weighted_far").median().alias("far_median"),
        pl.col("weighted_far").min().alias("far_min"),
        pl.col("weighted_far").max().alias("far_max"),
        pl.col("fcst_qty_total").sum().alias("fcst_qty_total_sum"),
        pl.len().alias("eval_target_week_count"),
    ).sort(GROUP_COLUMNS + ["model_name"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    forecasts.write_parquet(output_dir / "raw_forecasts.parquet")
    far_by_week.write_parquet(output_dir / "far_by_target_week.parquet")
    far_by_week.write_csv(output_dir / "far_by_target_week.csv")
    summary.write_parquet(output_dir / "far_summary.parquet")
    summary.write_csv(output_dir / "far_summary.csv")
    policy = {
        "manifest": str(Path(manifest_path).resolve()),
        "forecast_postprocess": "clip_negative_to_zero" if clip_negative else "none",
        "raw_negative_prediction_count": negative_count,
        "revisions": 8,
        "target_scope": "completed plan weeks with eight consecutive revisions",
        "scope_label_is_data_filter": False,
    }
    (output_dir / "evaluation_policy.json").write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    if make_plots:
        from model_test.exogenous_test.auto_run_architecture_hyper_plot_utils import (
            plot_far_mean_base_ab_case_comparisons,
            plot_far_trend,
        )
        plot_far_mean_base_ab_case_comparisons(summary, output_dir / "plots", base_ab_case_names=manifest["models"])
        for index, (_, frame) in enumerate(far_by_week.partition_by(GROUP_COLUMNS + ["model_name"], as_dict=True).items()):
            plot_far_trend(frame, output_dir / "plots" / f"far_trend_{index}.png")
    return far_by_week, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-negative", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    far, summary = evaluate_sweep(args.manifest, args.output_dir, clip_negative=args.clip_negative, make_plots=not args.no_plots)
    print(f"FAR rows={far.height}, summary rows={summary.height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

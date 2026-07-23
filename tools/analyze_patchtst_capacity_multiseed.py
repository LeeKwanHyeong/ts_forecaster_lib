#!/usr/bin/env python3
"""Aggregate isolated PatchTST capacity qualification runs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_dsio_qualification import EPSILON, parse_training_log  # noqa: E402


PREDICTION_KEYS = (
    "oper_part_no",
    "qualification_origin",
    "horizon_step",
    "demand_dt",
)
METRIC_NAMES = ("mae", "wape", "smape")


@dataclass(frozen=True)
class RunSpec:
    capacity: str
    seed: int
    artifact_dir: Path
    training_log: Path


def parse_run_spec(value: str) -> RunSpec:
    """Parse CAPACITY,SEED,ARTIFACT_DIR,TRAINING_LOG."""

    parts = [part.strip() for part in value.split(",", maxsplit=3)]
    if len(parts) != 4 or not all(parts):
        raise argparse.ArgumentTypeError(
            "run spec must be CAPACITY,SEED,ARTIFACT_DIR,TRAINING_LOG"
        )
    capacity, seed_text, artifact_dir, training_log = parts
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"run seed must be an integer, got {seed_text!r}"
        ) from exc
    return RunSpec(
        capacity=capacity.casefold(),
        seed=seed,
        artifact_dir=Path(artifact_dir).expanduser().resolve(),
        training_log=Path(training_log).expanduser().resolve(),
    )


def build_demand_cohorts(
    target: pl.DataFrame,
    *,
    train_cutoff: int,
    id_col: str = "oper_part_no",
    date_col: str = "demand_dt",
    target_col: str = "demand_qty",
    adi_threshold: float = 1.32,
    cv2_threshold: float = 0.49,
    epsilon: float = 0.0,
    min_periods: int = 10,
) -> pl.DataFrame:
    """Classify series with the Demand Engine ADI/CV2 contract."""

    required = {id_col, date_col, target_col}
    missing = sorted(required.difference(target.columns))
    if missing:
        raise ValueError(f"target data is missing columns: {missing}")
    if target.select((pl.col(target_col) < 0).any()).item():
        raise ValueError("target values must be non-negative")

    history = target.filter(pl.col(date_col) <= int(train_cutoff))
    if history.is_empty():
        raise ValueError(f"no target history exists at or before {train_cutoff}")

    is_nonzero = pl.col(target_col) > float(epsilon)
    stats = (
        history.group_by(id_col)
        .agg(
            pl.len().alias("n_periods"),
            (~is_nonzero).sum().alias("n_zero"),
            is_nonzero.sum().alias("n_nz"),
            pl.col(target_col).filter(is_nonzero).mean().alias("nz_mean"),
            pl.col(target_col)
            .filter(is_nonzero)
            .std(ddof=1)
            .alias("nz_std"),
        )
        .with_columns(
            (pl.col("n_zero") / pl.col("n_periods")).alias("zero_ratio"),
            pl.when(pl.col("n_nz") > 0)
            .then(pl.col("n_periods") / pl.col("n_nz"))
            .otherwise(None)
            .alias("ADI"),
            pl.when(
                (pl.col("nz_mean") > 0) & pl.col("nz_std").is_not_null()
            )
            .then((pl.col("nz_std") / pl.col("nz_mean")) ** 2)
            .otherwise(None)
            .alias("CV2"),
        )
    )

    demand_type = (
        pl.when(pl.col("n_nz") == 0)
        .then(pl.lit("no_demand"))
        .when(
            (pl.col("n_periods") < int(min_periods))
            | pl.col("ADI").is_null()
            | pl.col("CV2").is_null()
        )
        .then(pl.lit("insufficient"))
        .when(
            (pl.col("ADI") < float(adi_threshold))
            & (pl.col("CV2") < float(cv2_threshold))
        )
        .then(pl.lit("smooth"))
        .when(pl.col("ADI") < float(adi_threshold))
        .then(pl.lit("erratic"))
        .when(pl.col("CV2") < float(cv2_threshold))
        .then(pl.lit("intermittent"))
        .otherwise(pl.lit("lumpy"))
        .alias("demand_type")
    )
    return (
        stats.with_columns(demand_type)
        .with_columns(
            pl.when(pl.col("demand_type").is_in(["smooth", "erratic"]))
            .then(pl.lit("dense"))
            .when(
                pl.col("demand_type").is_in(["intermittent", "lumpy"])
            )
            .then(pl.lit("intermittent"))
            .otherwise(pl.col("demand_type"))
            .alias("cohort")
        )
        .sort(id_col)
    )


def aggregate_metrics(
    frame: pl.DataFrame,
    *,
    group_by: Sequence[str],
) -> pl.DataFrame:
    expressions = [
        pl.col("oper_part_no").n_unique().alias("series_count"),
        pl.len().alias("observation_count"),
        pl.col("absolute_error").mean().alias("mae"),
        (
            pl.col("absolute_error").sum()
            / (pl.col("absolute_actual").sum() + EPSILON)
        ).alias("wape"),
        pl.col("smape_component").mean().alias("smape"),
        pl.col("absolute_actual").sum().alias("absolute_actual_sum"),
    ]
    return (
        frame.group_by(list(group_by))
        .agg(expressions)
        .sort(list(group_by))
    )


def summarize_seed_metrics(seed_metrics: pl.DataFrame) -> pl.DataFrame:
    expressions: list[pl.Expr] = [
        pl.col("seed").n_unique().alias("seed_count"),
        pl.col("parameter_count").first().alias("parameter_count"),
    ]
    for metric in METRIC_NAMES:
        expressions.extend(
            [
                pl.col(metric).mean().alias(f"{metric}_mean"),
                pl.col(metric).std(ddof=1).fill_null(0.0).alias(f"{metric}_std"),
                pl.col(metric).min().alias(f"{metric}_min"),
                pl.col(metric).max().alias(f"{metric}_max"),
            ]
        )
    return (
        seed_metrics.group_by("capacity")
        .agg(expressions)
        .sort(["mae_mean", "parameter_count"])
    )


def select_capacity_and_refit_epoch(
    capacity_summary: pl.DataFrame,
    epoch_summary: pl.DataFrame,
) -> dict[str, Any]:
    """Select mean-MAE capacity and its minimum mean-validation epoch."""

    if capacity_summary.is_empty() or epoch_summary.is_empty():
        raise ValueError("capacity and epoch summaries must not be empty")
    selected_row = capacity_summary.sort(
        ["mae_mean", "parameter_count", "capacity"]
    ).row(0, named=True)
    selected_capacity = str(selected_row["capacity"])
    selected_curve = epoch_summary.filter(
        pl.col("capacity") == selected_capacity
    )
    if selected_curve.is_empty():
        raise ValueError(
            f"epoch summary has no rows for {selected_capacity!r}"
        )
    refit_row = selected_curve.sort(
        ["validation_loss_mean", "epoch"]
    ).row(0, named=True)
    return {
        "selected_capacity": selected_capacity,
        "production_refit_epochs": int(refit_row["epoch"]),
        "mean_qualification_mae": float(selected_row["mae_mean"]),
        "mean_qualification_wape": float(selected_row["wape_mean"]),
        "mean_qualification_smape": float(selected_row["smape_mean"]),
        "refit_epoch_mean_validation_loss": float(
            refit_row["validation_loss_mean"]
        ),
        "capacity_selection_basis": (
            "minimum mean public strict-load qualification MAE across the "
            "shared isolated seed set; parameter count and name break exact ties"
        ),
        "epoch_selection_basis": (
            "earliest minimum of the selected capacity's epoch-wise mean "
            "validation loss across the shared isolated seed set"
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _contract_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    source = manifest.get("source")
    dataset = manifest.get("dataset")
    temporal = manifest.get("temporal_contract")
    if not all(isinstance(item, dict) for item in (source, dataset, temporal)):
        raise ValueError("data manifest is missing source/dataset/temporal data")
    return {
        "source_sha256": source["sha256"],
        "row_count": dataset["row_count"],
        "series_count": dataset["series_count"],
        "train_windows": dataset["train_windows"],
        "validation_windows": dataset["validation_windows"],
        "temporal_contract": temporal,
    }


def _load_run(
    spec: RunSpec,
) -> tuple[dict[str, Any], pl.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    evaluation_dir = spec.artifact_dir / "qualification_evaluation"
    metrics_path = evaluation_dir / "qualification_metrics.csv"
    predictions_path = evaluation_dir / "qualification_predictions.parquet"
    manifest_path = spec.artifact_dir / "data_manifest.json"
    for path in (
        metrics_path,
        predictions_path,
        manifest_path,
        spec.training_log,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    metrics = pl.read_csv(metrics_path)
    if metrics.height != 1:
        raise ValueError(f"{metrics_path} must contain exactly one model row")
    metric_row = metrics.row(0, named=True)
    if metric_row["model_key"] != "patchtst_base":
        raise ValueError(
            f"{metrics_path} is not a patchtst_base qualification result"
        )
    metric_row.update(capacity=spec.capacity, seed=spec.seed)

    predictions = pl.read_parquet(predictions_path)
    required_prediction_columns = {
        *PREDICTION_KEYS,
        "actual",
        "prediction",
        "absolute_error",
        "absolute_actual",
        "smape_component",
    }
    missing = sorted(required_prediction_columns.difference(predictions.columns))
    if missing:
        raise ValueError(f"{predictions_path} is missing columns: {missing}")
    predictions = predictions.with_columns(
        pl.lit(spec.capacity).alias("capacity"),
        pl.lit(spec.seed).alias("seed"),
    )

    histories = parse_training_log(spec.training_log)
    records = histories.get("patchtst_base")
    if not records:
        raise ValueError(f"{spec.training_log} has no patchtst_base history")
    epochs = [
        {
            "capacity": spec.capacity,
            "seed": spec.seed,
            "epoch": record.epoch,
            "total_epochs": record.total_epochs,
            "learning_rate": record.learning_rate,
            "train_loss": record.train_loss,
            "validation_loss": record.validation_loss,
        }
        for record in records
    ]
    contract = _contract_from_manifest(_read_json(manifest_path))
    return metric_row, predictions, epochs, contract


def _summary_by_seed(
    frame: pl.DataFrame,
    *,
    dimensions: Sequence[str],
) -> pl.DataFrame:
    expressions: list[pl.Expr] = [
        pl.col("series_count").first().alias("series_count"),
        pl.col("observation_count").first().alias("observation_count"),
    ]
    for metric in METRIC_NAMES:
        expressions.extend(
            [
                pl.col(metric).mean().alias(f"{metric}_mean"),
                pl.col(metric).std(ddof=1).fill_null(0.0).alias(f"{metric}_std"),
                pl.col(metric).min().alias(f"{metric}_min"),
                pl.col(metric).max().alias(f"{metric}_max"),
            ]
        )
    return (
        frame.group_by(["capacity", *dimensions])
        .agg(expressions)
        .sort(["capacity", *dimensions])
    )


def summarize_pairwise(
    frame: pl.DataFrame,
    *,
    dimensions: Sequence[str],
) -> pl.DataFrame:
    return (
        frame.group_by(list(dimensions))
        .agg(
            pl.col("seed").n_unique().alias("seed_count"),
            pl.col("candidate_mae").mean().alias("candidate_mae_mean"),
            pl.col("control_mae").mean().alias("control_mae_mean"),
            pl.col("mae_delta").mean().alias("mae_delta_mean"),
            pl.col("mae_delta")
            .std(ddof=1)
            .fill_null(0.0)
            .alias("mae_delta_std"),
            (pl.col("mae_delta") < 0).sum().alias("candidate_seed_wins"),
        )
        .sort(list(dimensions))
    )


def _paired_frames(
    predictions: pl.DataFrame,
    *,
    candidate: str,
    control: str,
    seeds: Iterable[int],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    seed_rows: list[dict[str, Any]] = []
    cohort_frames: list[pl.DataFrame] = []
    horizon_frames: list[pl.DataFrame] = []
    for seed in seeds:
        candidate_frame = predictions.filter(
            (pl.col("capacity") == candidate) & (pl.col("seed") == seed)
        ).select(
            *PREDICTION_KEYS,
            "cohort",
            pl.col("actual").alias("candidate_actual"),
            pl.col("absolute_error").alias("candidate_absolute_error"),
        )
        control_frame = predictions.filter(
            (pl.col("capacity") == control) & (pl.col("seed") == seed)
        ).select(
            *PREDICTION_KEYS,
            pl.col("actual").alias("control_actual"),
            pl.col("absolute_error").alias("control_absolute_error"),
        )
        paired = candidate_frame.join(
            control_frame,
            on=list(PREDICTION_KEYS),
            how="inner",
            validate="1:1",
        )
        if paired.height != candidate_frame.height or paired.height != control_frame.height:
            raise ValueError(f"prediction rows do not align for seed {seed}")
        max_actual_delta = paired.select(
            (pl.col("candidate_actual") - pl.col("control_actual"))
            .abs()
            .max()
        ).item()
        if not math.isclose(float(max_actual_delta or 0.0), 0.0, abs_tol=1e-8):
            raise ValueError(f"actual values differ between capacities for seed {seed}")

        series = paired.group_by("oper_part_no").agg(
            pl.col("candidate_absolute_error").mean().alias("candidate_mae"),
            pl.col("control_absolute_error").mean().alias("control_mae"),
        )
        horizon = (
            paired.group_by("horizon_step")
            .agg(
                pl.col("candidate_absolute_error").mean().alias("candidate_mae"),
                pl.col("control_absolute_error").mean().alias("control_mae"),
            )
            .with_columns(
                pl.lit(seed).alias("seed"),
                (pl.col("candidate_mae") - pl.col("control_mae")).alias(
                    "mae_delta"
                ),
            )
        )
        cohort = (
            paired.group_by("cohort")
            .agg(
                pl.col("oper_part_no").n_unique().alias("series_count"),
                pl.len().alias("observation_count"),
                pl.col("candidate_absolute_error").mean().alias("candidate_mae"),
                pl.col("control_absolute_error").mean().alias("control_mae"),
            )
            .with_columns(
                pl.lit(seed).alias("seed"),
                (pl.col("candidate_mae") - pl.col("control_mae")).alias(
                    "mae_delta"
                ),
            )
        )
        seed_rows.append(
            {
                "seed": seed,
                "observation_win_rate": paired.select(
                    (
                        pl.col("candidate_absolute_error")
                        < pl.col("control_absolute_error")
                    ).mean()
                ).item(),
                "observation_tie_rate": paired.select(
                    (
                        pl.col("candidate_absolute_error")
                        == pl.col("control_absolute_error")
                    ).mean()
                ).item(),
                "series_wins": series.filter(
                    pl.col("candidate_mae") < pl.col("control_mae")
                ).height,
                "series_count": series.height,
                "horizon_wins": horizon.filter(pl.col("mae_delta") < 0).height,
                "horizon_count": horizon.height,
            }
        )
        cohort_frames.append(cohort)
        horizon_frames.append(horizon)
    return (
        pl.DataFrame(seed_rows).sort("seed"),
        pl.concat(cohort_frames).sort(["seed", "cohort"]),
        pl.concat(horizon_frames).sort(["seed", "horizon_step"]),
    )


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    specs = list(args.run)
    if not specs:
        raise ValueError("at least one --run is required")
    identities = [(spec.capacity, spec.seed) for spec in specs]
    if len(identities) != len(set(identities)):
        raise ValueError("capacity/seed run identities must be unique")

    capacity_seeds: dict[str, set[int]] = {}
    for spec in specs:
        capacity_seeds.setdefault(spec.capacity, set()).add(spec.seed)
    if args.candidate_capacity not in capacity_seeds:
        raise ValueError(f"missing candidate capacity {args.candidate_capacity!r}")
    if args.control_capacity not in capacity_seeds:
        raise ValueError(f"missing control capacity {args.control_capacity!r}")
    shared_seeds = capacity_seeds[args.candidate_capacity]
    if shared_seeds != capacity_seeds[args.control_capacity]:
        raise ValueError("candidate and control must have the same seed set")

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pl.DataFrame] = []
    epoch_rows: list[dict[str, Any]] = []
    contracts: list[dict[str, Any]] = []
    for spec in specs:
        metric, predictions, epochs, contract = _load_run(spec)
        metric_rows.append(metric)
        prediction_frames.append(predictions)
        epoch_rows.extend(epochs)
        contracts.append(contract)
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("qualification data contracts differ across runs")

    target = pl.read_parquet(args.target_source)
    cohorts = build_demand_cohorts(
        target,
        train_cutoff=args.train_cutoff,
        adi_threshold=args.adi_threshold,
        cv2_threshold=args.cv2_threshold,
        epsilon=args.epsilon,
        min_periods=args.min_periods,
    )
    predictions = pl.concat(prediction_frames, how="vertical").join(
        cohorts.select("oper_part_no", "demand_type", "cohort"),
        on="oper_part_no",
        how="left",
        validate="m:1",
    ).with_columns(
        pl.when(pl.col("horizon_step") <= 4)
        .then(pl.lit("H01-04"))
        .when(pl.col("horizon_step") <= 13)
        .then(pl.lit("H05-13"))
        .otherwise(pl.lit("H14-27"))
        .alias("horizon_band")
    )
    if predictions["cohort"].null_count() > 0:
        raise ValueError("some qualification series have no cohort assignment")

    seed_metrics = pl.DataFrame(metric_rows).sort(["capacity", "seed"])
    capacity_summary = summarize_seed_metrics(seed_metrics)
    cohort_metrics = aggregate_metrics(
        predictions,
        group_by=("capacity", "seed", "cohort"),
    )
    cohort_summary = _summary_by_seed(
        cohort_metrics,
        dimensions=("cohort",),
    )
    horizon_metrics = aggregate_metrics(
        predictions,
        group_by=("capacity", "seed", "horizon_step"),
    )
    horizon_summary = _summary_by_seed(
        horizon_metrics,
        dimensions=("horizon_step",),
    )
    horizon_band_metrics = aggregate_metrics(
        predictions,
        group_by=("capacity", "seed", "horizon_band"),
    )
    horizon_band_summary = _summary_by_seed(
        horizon_band_metrics,
        dimensions=("horizon_band",),
    )
    cohort_horizon_metrics = aggregate_metrics(
        predictions,
        group_by=("capacity", "seed", "cohort", "horizon_step"),
    )
    cohort_horizon_summary = _summary_by_seed(
        cohort_horizon_metrics,
        dimensions=("cohort", "horizon_step"),
    )

    epoch_curves = pl.DataFrame(epoch_rows).sort(
        ["capacity", "seed", "epoch"]
    )
    epoch_summary = (
        epoch_curves.group_by(["capacity", "epoch"])
        .agg(
            pl.col("train_loss").mean().alias("train_loss_mean"),
            pl.col("train_loss").std(ddof=1).fill_null(0.0).alias(
                "train_loss_std"
            ),
            pl.col("validation_loss").mean().alias("validation_loss_mean"),
            pl.col("validation_loss").std(ddof=1).fill_null(0.0).alias(
                "validation_loss_std"
            ),
        )
        .sort(["capacity", "epoch"])
    )
    seed_best_epochs = (
        epoch_curves.sort(["capacity", "seed", "validation_loss", "epoch"])
        .group_by(["capacity", "seed"], maintain_order=True)
        .first()
        .select(
            "capacity",
            "seed",
            pl.col("epoch").alias("best_epoch"),
            pl.col("validation_loss").alias("best_validation_loss"),
        )
        .sort(["capacity", "seed"])
    )

    pairwise_seed, pairwise_cohort, pairwise_horizon = _paired_frames(
        predictions,
        candidate=args.candidate_capacity,
        control=args.control_capacity,
        seeds=sorted(shared_seeds),
    )
    pairwise_seed = pairwise_seed.join(
        seed_metrics.filter(
            pl.col("capacity") == args.candidate_capacity
        ).select(
            "seed",
            pl.col("mae").alias("candidate_mae"),
            pl.col("wape").alias("candidate_wape"),
            pl.col("smape").alias("candidate_smape"),
        ),
        on="seed",
    ).join(
        seed_metrics.filter(
            pl.col("capacity") == args.control_capacity
        ).select(
            "seed",
            pl.col("mae").alias("control_mae"),
            pl.col("wape").alias("control_wape"),
            pl.col("smape").alias("control_smape"),
        ),
        on="seed",
    ).with_columns(
        (pl.col("candidate_mae") - pl.col("control_mae")).alias("mae_delta"),
        (
            100.0
            * (pl.col("control_mae") - pl.col("candidate_mae"))
            / pl.col("control_mae")
        ).alias("mae_improvement_pct"),
    ).sort("seed")
    pairwise_cohort_summary = summarize_pairwise(
        pairwise_cohort,
        dimensions=("cohort",),
    )
    pairwise_horizon_summary = summarize_pairwise(
        pairwise_horizon,
        dimensions=("horizon_step",),
    )

    policy = select_capacity_and_refit_epoch(
        capacity_summary,
        epoch_summary,
    )
    policy.update(
        {
            "candidate_capacity": args.candidate_capacity,
            "control_capacity": args.control_capacity,
            "seeds": sorted(shared_seeds),
            "candidate_seed_mae_wins": pairwise_seed.filter(
                pl.col("mae_delta") < 0
            ).height,
            "seed_count": len(shared_seeds),
            "qualification_train_cutoff": args.train_cutoff,
        }
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "cohort_assignments": output_dir / "cohort_assignments.parquet",
        "capacity_seed_metrics": output_dir / "capacity_seed_metrics.csv",
        "capacity_summary": output_dir / "capacity_summary.csv",
        "capacity_cohort_metrics": output_dir / "capacity_cohort_metrics.csv",
        "capacity_cohort_summary": output_dir / "capacity_cohort_summary.csv",
        "capacity_horizon_metrics": output_dir / "capacity_horizon_metrics.csv",
        "capacity_horizon_summary": output_dir / "capacity_horizon_summary.csv",
        "capacity_horizon_band_metrics": output_dir
        / "capacity_horizon_band_metrics.csv",
        "capacity_horizon_band_summary": output_dir
        / "capacity_horizon_band_summary.csv",
        "capacity_cohort_horizon_metrics": output_dir
        / "capacity_cohort_horizon_metrics.csv",
        "capacity_cohort_horizon_summary": output_dir
        / "capacity_cohort_horizon_summary.csv",
        "capacity_epoch_curves": output_dir / "capacity_epoch_curves.csv",
        "capacity_epoch_summary": output_dir / "capacity_epoch_summary.csv",
        "capacity_seed_best_epochs": output_dir
        / "capacity_seed_best_epochs.csv",
        "capacity_pairwise_seed": output_dir / "capacity_pairwise_seed.csv",
        "capacity_pairwise_cohort": output_dir
        / "capacity_pairwise_cohort.csv",
        "capacity_pairwise_cohort_summary": output_dir
        / "capacity_pairwise_cohort_summary.csv",
        "capacity_pairwise_horizon": output_dir
        / "capacity_pairwise_horizon.csv",
        "capacity_pairwise_horizon_summary": output_dir
        / "capacity_pairwise_horizon_summary.csv",
        "production_refit_policy": output_dir / "production_refit_policy.json",
    }
    cohorts.write_parquet(outputs["cohort_assignments"])
    seed_metrics.write_csv(outputs["capacity_seed_metrics"])
    capacity_summary.write_csv(outputs["capacity_summary"])
    cohort_metrics.write_csv(outputs["capacity_cohort_metrics"])
    cohort_summary.write_csv(outputs["capacity_cohort_summary"])
    horizon_metrics.write_csv(outputs["capacity_horizon_metrics"])
    horizon_summary.write_csv(outputs["capacity_horizon_summary"])
    horizon_band_metrics.write_csv(outputs["capacity_horizon_band_metrics"])
    horizon_band_summary.write_csv(outputs["capacity_horizon_band_summary"])
    cohort_horizon_metrics.write_csv(outputs["capacity_cohort_horizon_metrics"])
    cohort_horizon_summary.write_csv(outputs["capacity_cohort_horizon_summary"])
    epoch_curves.write_csv(outputs["capacity_epoch_curves"])
    epoch_summary.write_csv(outputs["capacity_epoch_summary"])
    seed_best_epochs.write_csv(outputs["capacity_seed_best_epochs"])
    pairwise_seed.write_csv(outputs["capacity_pairwise_seed"])
    pairwise_cohort.write_csv(outputs["capacity_pairwise_cohort"])
    pairwise_cohort_summary.write_csv(
        outputs["capacity_pairwise_cohort_summary"]
    )
    pairwise_horizon.write_csv(outputs["capacity_pairwise_horizon"])
    pairwise_horizon_summary.write_csv(
        outputs["capacity_pairwise_horizon_summary"]
    )
    outputs["production_refit_policy"].write_text(
        json.dumps(policy, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema_version": 1,
        "protocol": {
            **contracts[0],
            "train_cutoff": args.train_cutoff,
            "adi_threshold": args.adi_threshold,
            "cv2_threshold": args.cv2_threshold,
            "epsilon": args.epsilon,
            "min_periods": args.min_periods,
        },
        "runs": [
            {
                **asdict(spec),
                "artifact_dir": str(spec.artifact_dir),
                "training_log": str(spec.training_log),
            }
            for spec in specs
        ],
        "cohort_counts": cohorts.group_by("cohort")
        .len()
        .sort("cohort")
        .to_dicts(),
        "decision": policy,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    summary_path = output_dir / "capacity_multiseed_summary.json"
    summary["outputs"]["capacity_multiseed_summary"] = str(summary_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_spec,
        required=True,
        metavar="CAPACITY,SEED,ARTIFACT_DIR,TRAINING_LOG",
    )
    parser.add_argument("--target-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-capacity", default="small")
    parser.add_argument("--control-capacity", default="current")
    parser.add_argument("--train-cutoff", type=int, default=202517)
    parser.add_argument("--adi-threshold", type=float, default=1.32)
    parser.add_argument("--cv2-threshold", type=float, default=0.49)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--min-periods", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.target_source = args.target_source.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    summary = run_analysis(args)
    decision = summary["decision"]
    print(
        "[capacity] selected="
        f"{decision['selected_capacity']} "
        f"production_refit_epochs={decision['production_refit_epochs']}"
    )
    print(f"[capacity] outputs={args.output_dir}")


if __name__ == "__main__":
    main()

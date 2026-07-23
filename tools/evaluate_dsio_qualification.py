#!/usr/bin/env python3
"""Evaluate DSIO endogenous qualification checkpoints and fix refit epochs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modeling_module.api import load_predictor  # noqa: E402
from modeling_module.data_loader.indexed_temporal_data_module import (  # noqa: E402
    IndexedTemporalDataModule,
)
from modeling_module.data_loader.temporal import add_period  # noqa: E402


EPSILON = 1e-8
MODEL_HEADING_PREFIXES = (
    ("PatchTST Quantile (", "patchtst_quantile"),
    ("PatchTST (", "patchtst_base"),
    ("PatchMixer (", "patchmixer"),
    ("N-HiTS (", "nhits_base"),
    ("TimeMixer (", "timemixer"),
)
EPOCH_PATTERN = re.compile(
    r"^Epoch (?P<epoch>\d+)/(?P<total>\d+)"
    r" \| LR (?P<lr>[-+0-9.eE]+)"
    r" \| Train (?P<train>[-+0-9.eE]+)"
    r" \| Val (?P<validation>[-+0-9.eE]+)$"
)


@dataclass(frozen=True)
class EpochRecord:
    model_key: str
    epoch: int
    total_epochs: int
    learning_rate: float
    train_loss: float
    validation_loss: float


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_training_log(path: Path) -> dict[str, list[EpochRecord]]:
    """Parse model-scoped epoch records from one total-train log."""

    if not path.is_file():
        raise FileNotFoundError(path)

    current_model: str | None = None
    histories: dict[str, list[EpochRecord]] = {}
    seen_epochs: set[tuple[str, int]] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        for prefix, model_key in MODEL_HEADING_PREFIXES:
            if line.startswith(prefix):
                current_model = model_key
                histories.setdefault(model_key, [])
                break

        match = EPOCH_PATTERN.match(line)
        if match is None:
            continue
        if current_model is None:
            raise ValueError(f"Epoch record appeared before a model heading: {line}")

        epoch = int(match.group("epoch"))
        identity = (current_model, epoch)
        if identity in seen_epochs:
            raise ValueError(
                f"Duplicate epoch {epoch} for model {current_model!r} in {path}."
            )
        seen_epochs.add(identity)
        histories[current_model].append(
            EpochRecord(
                model_key=current_model,
                epoch=epoch,
                total_epochs=int(match.group("total")),
                learning_rate=float(match.group("lr")),
                train_loss=float(match.group("train")),
                validation_loss=float(match.group("validation")),
            )
        )

    empty_models = [key for key, records in histories.items() if not records]
    if empty_models:
        raise ValueError(f"Model headings have no epoch records: {empty_models}.")
    if not histories:
        raise ValueError(f"No epoch records found in {path}.")
    return histories


def build_refit_policy(
    *,
    histories: Mapping[str, Sequence[EpochRecord]],
    training_manifest: Mapping[str, Any],
    model_keys: Sequence[str],
    loss_tolerance: float = 5e-6,
) -> list[dict[str, Any]]:
    """Select each fixed refit epoch and verify it against checkpoint metadata."""

    results = training_manifest.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("training_manifest.json is missing the `results` object.")

    policies: list[dict[str, Any]] = []
    for model_key in model_keys:
        records = list(histories.get(model_key, ()))
        if not records:
            raise ValueError(f"Training log has no epoch history for {model_key!r}.")
        result = results.get(model_key)
        if not isinstance(result, Mapping):
            raise ValueError(
                f"Training manifest has no result metadata for {model_key!r}."
            )

        best = min(records, key=lambda item: (item.validation_loss, item.epoch))
        manifest_best = float(result["best_val_loss"])
        if not math.isclose(
            best.validation_loss,
            manifest_best,
            rel_tol=1e-7,
            abs_tol=loss_tolerance,
        ):
            raise ValueError(
                f"{model_key} best validation loss differs between log "
                f"({best.validation_loss}) and manifest ({manifest_best})."
            )

        policies.append(
            {
                "model_key": model_key,
                "production_refit_epochs": best.epoch,
                "qualification_best_epoch": best.epoch,
                "qualification_total_epochs": max(
                    item.total_epochs for item in records
                ),
                "qualification_best_validation_loss": manifest_best,
                "selection_basis": (
                    "minimum model-specific validation loss on the fixed "
                    "last-origin holdout"
                ),
                "refit_contract": (
                    "train from scratch on all targets through train_end_week "
                    "for this fixed epoch count; do not early-stop on the "
                    "consumed qualification holdout"
                ),
            }
        )
    return policies


def build_epoch_extension_analysis(
    *,
    histories: Mapping[str, Sequence[EpochRecord]],
    model_keys: Sequence[str],
    baseline_max_epoch: int,
) -> list[dict[str, Any]]:
    """Compare the original epoch window with its explicit extension."""

    cutoff = int(baseline_max_epoch)
    if cutoff <= 0:
        raise ValueError("baseline_max_epoch must be positive.")

    analyses: list[dict[str, Any]] = []
    for model_key in model_keys:
        records = list(histories.get(model_key, ()))
        baseline = [record for record in records if record.epoch <= cutoff]
        extension = [record for record in records if record.epoch > cutoff]
        if not baseline:
            raise ValueError(
                f"{model_key} has no epoch at or before baseline cutoff {cutoff}."
            )
        if not extension:
            raise ValueError(
                f"{model_key} has no epoch after baseline cutoff {cutoff}."
            )

        baseline_best = min(
            baseline,
            key=lambda item: (item.validation_loss, item.epoch),
        )
        extension_best = min(
            extension,
            key=lambda item: (item.validation_loss, item.epoch),
        )
        overall_best = min(
            records,
            key=lambda item: (item.validation_loss, item.epoch),
        )
        delta = extension_best.validation_loss - baseline_best.validation_loss
        analyses.append(
            {
                "model_key": model_key,
                "baseline_epoch_start": min(record.epoch for record in baseline),
                "baseline_epoch_end": cutoff,
                "baseline_best_epoch": baseline_best.epoch,
                "baseline_best_validation_loss": baseline_best.validation_loss,
                "extension_epoch_start": min(
                    record.epoch for record in extension
                ),
                "extension_epoch_end": max(record.epoch for record in extension),
                "extension_best_epoch": extension_best.epoch,
                "extension_best_validation_loss": extension_best.validation_loss,
                "extension_minus_baseline_validation_loss": delta,
                "extension_improved": delta < 0.0,
                "overall_best_epoch": overall_best.epoch,
                "overall_best_validation_loss": overall_best.validation_loss,
            }
        )
    return analyses


def metric_values(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    """Return the repository-standard micro MAE, WAPE, and sMAPE ratios."""

    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if actual.shape != prediction.shape:
        raise ValueError(
            f"Metric arrays must have the same shape, got "
            f"{actual.shape} and {prediction.shape}."
        )
    if actual.size == 0:
        raise ValueError("Metric arrays must not be empty.")
    if not np.isfinite(actual).all() or not np.isfinite(prediction).all():
        raise ValueError("Metric arrays must contain only finite values.")

    absolute_error = np.abs(prediction - actual)
    return {
        "mae": float(absolute_error.mean()),
        "wape": float(
            absolute_error.sum() / (np.abs(actual).sum() + EPSILON)
        ),
        "smape": float(
            np.mean(
                2.0
                * absolute_error
                / (np.abs(actual) + np.abs(prediction) + EPSILON)
            )
        ),
    }


def _with_error_columns(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        (pl.col("prediction") - pl.col("actual"))
        .abs()
        .alias("absolute_error"),
        pl.col("actual").abs().alias("absolute_actual"),
    ).with_columns(
        (
            2.0
            * pl.col("absolute_error")
            / (
                pl.col("absolute_actual")
                + pl.col("prediction").abs()
                + EPSILON
            )
        ).alias("smape_component")
    )


def _aggregate_metrics(
    frame: pl.DataFrame,
    *,
    group_by: Sequence[str],
) -> pl.DataFrame:
    expressions = [
        pl.len().alias("observation_count"),
        pl.col("absolute_error").mean().alias("mae"),
        (
            pl.col("absolute_error").sum()
            / (pl.col("absolute_actual").sum() + EPSILON)
        ).alias("wape"),
        pl.col("smape_component").mean().alias("smape"),
        pl.col("absolute_actual").sum().alias("absolute_actual_sum"),
    ]
    if not group_by:
        return frame.select(expressions)
    return frame.group_by(list(group_by)).agg(expressions).sort(list(group_by))


def _resolve_target_source(
    data_manifest: Mapping[str, Any],
    override: Path | None,
) -> Path:
    if override is not None:
        path = override.expanduser().resolve()
    else:
        source = data_manifest.get("source")
        if not isinstance(source, Mapping) or not source.get("path"):
            raise ValueError(
                "data_manifest.json has no source path; pass --target-source."
            )
        path = Path(str(source["path"])).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _resolve_checkpoint_path(
    artifact_dir: Path,
    result: Mapping[str, Any],
) -> Path:
    raw_path = result.get("ckpt_path")
    if not raw_path:
        raise ValueError("Training result is missing `ckpt_path`.")
    original = Path(str(raw_path)).expanduser()
    if original.is_file():
        return original
    portable = artifact_dir / original.name
    if portable.is_file():
        return portable
    raise FileNotFoundError(
        f"Checkpoint not found at {original} or portable path {portable}."
    )


def _synchronize_cuda(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _evaluate_model(
    *,
    model_key: str,
    checkpoint_path: Path,
    data_module: IndexedTemporalDataModule,
    horizon: int,
    validation_origin: int,
    device: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    load_started = time.perf_counter()
    predictor = load_predictor(
        str(checkpoint_path),
        device=device,
        strict=True,
    )
    load_seconds = time.perf_counter() - load_started
    if predictor.model_key != model_key:
        raise ValueError(
            f"Checkpoint model key mismatch: expected {model_key!r}, "
            f"loaded {predictor.model_key!r}."
        )

    loader = data_module.get_val_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2,
    )
    demand_weeks = np.asarray(
        [
            add_period(validation_origin, step, "weekly")
            for step in range(horizon)
        ],
        dtype=np.int64,
    )
    frames: list[pl.DataFrame] = []
    _synchronize_cuda(device)
    inference_started = time.perf_counter()
    for batch_index, (x, y, part_ids) in enumerate(loader, start=1):
        batch_size_actual = len(part_ids)
        output = predictor.predict(
            {"x": x, "part_ids": list(part_ids)},
            horizon=horizon,
        )
        if not isinstance(output, Mapping) or "point" not in output:
            raise ValueError(
                f"{model_key} public predictor did not return a `point` output."
            )
        prediction = np.asarray(output["point"], dtype=np.float64)
        expected_values = batch_size_actual * horizon
        if prediction.size != expected_values:
            raise ValueError(
                f"{model_key} returned {prediction.size} point values; "
                f"expected {expected_values}."
            )
        prediction = prediction.reshape(batch_size_actual, horizon)
        actual = y.detach().cpu().numpy().astype(np.float64, copy=False)
        if actual.shape != prediction.shape:
            raise ValueError(
                f"{model_key} actual/prediction shapes differ: "
                f"{actual.shape} vs {prediction.shape}."
            )
        if not np.isfinite(prediction).all():
            raise ValueError(f"{model_key} produced non-finite predictions.")

        frame = pl.DataFrame(
            {
                "oper_part_no": np.repeat(
                    np.asarray(list(part_ids), dtype=str),
                    horizon,
                ),
                "horizon_step": np.tile(
                    np.arange(1, horizon + 1, dtype=np.int16),
                    batch_size_actual,
                ),
                "demand_dt": np.tile(demand_weeks, batch_size_actual),
                "actual": actual.reshape(-1),
                "prediction": prediction.reshape(-1),
            }
        ).with_columns(
            pl.lit(model_key).alias("model_key"),
            pl.lit(validation_origin).alias("qualification_origin"),
        )
        frames.append(
            frame.select(
                "model_key",
                "oper_part_no",
                "qualification_origin",
                "horizon_step",
                "demand_dt",
                "actual",
                "prediction",
            )
        )
        if batch_index % 10 == 0:
            print(
                f"[qualification] {model_key}: "
                f"{batch_index * batch_size:,} series processed",
                flush=True,
            )

    _synchronize_cuda(device)
    inference_seconds = time.perf_counter() - inference_started
    predictions = pl.concat(frames, how="vertical")
    expected_rows = len(data_module.val_dataset or ()) * horizon
    if predictions.height != expected_rows:
        raise ValueError(
            f"{model_key} produced {predictions.height} rows; "
            f"expected {expected_rows}."
        )

    metrics = metric_values(
        predictions["actual"].to_numpy(),
        predictions["prediction"].to_numpy(),
    )
    series_count = predictions["oper_part_no"].n_unique()
    return predictions, {
        "model_key": model_key,
        "mae": metrics["mae"],
        "wape": metrics["wape"],
        "wape_pct": metrics["wape"] * 100.0,
        "smape": metrics["smape"],
        "smape_pct": metrics["smape"] * 100.0,
        "series_count": series_count,
        "observation_count": predictions.height,
        "parameter_count": sum(
            parameter.numel() for parameter in predictor.model.parameters()
        ),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "series_per_second": series_count / max(inference_seconds, EPSILON),
        "point_output": "q50" if "quantile" in model_key else "point",
    }


def _add_metric_ranks(
    metrics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in metrics]
    for metric in ("mae", "wape", "smape"):
        ordered = sorted(
            rows,
            key=lambda row: (float(row[metric]), str(row["model_key"])),
        )
        ranks = {
            str(row["model_key"]): rank
            for rank, row in enumerate(ordered, start=1)
        }
        for row in rows:
            row[f"{metric}_rank"] = ranks[str(row["model_key"])]
    return rows


def _validate_source_contract(
    *,
    source: Path,
    data_manifest: Mapping[str, Any],
) -> None:
    source_meta = data_manifest.get("source")
    if not isinstance(source_meta, Mapping):
        raise ValueError("data_manifest.json is missing the `source` object.")
    expected_sha = source_meta.get("sha256")
    if expected_sha:
        actual_sha = _sha256_file(source)
        if actual_sha != expected_sha:
            raise ValueError(
                f"Target source SHA-256 mismatch: expected {expected_sha}, "
                f"got {actual_sha}."
            )


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = args.artifact_dir.expanduser().resolve()
    data_manifest_path = artifact_dir / "data_manifest.json"
    training_manifest_path = artifact_dir / "training_manifest.json"
    data_manifest = _read_json(data_manifest_path)
    training_manifest = _read_json(training_manifest_path)
    source = _resolve_target_source(data_manifest, args.target_source)
    _validate_source_contract(source=source, data_manifest=data_manifest)

    schema = data_manifest.get("schema")
    temporal = data_manifest.get("temporal_contract")
    if not isinstance(schema, Mapping) or not isinstance(temporal, Mapping):
        raise ValueError(
            "data_manifest.json requires `schema` and `temporal_contract` objects."
        )
    model_keys = [str(value) for value in data_manifest["artifact_models"]]
    results = training_manifest.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("training_manifest.json is missing the `results` object.")

    histories = parse_training_log(args.training_log.expanduser().resolve())
    refit_policy = build_refit_policy(
        histories=histories,
        training_manifest=training_manifest,
        model_keys=model_keys,
    )
    extension_analysis = None
    if args.baseline_max_epoch is not None:
        extension_analysis = build_epoch_extension_analysis(
            histories=histories,
            model_keys=model_keys,
            baseline_max_epoch=args.baseline_max_epoch,
        )

    columns = [
        str(schema["id_col"]),
        str(schema["date_col"]),
        str(schema["target_col"]),
    ]
    target_frame = pl.read_parquet(source, columns=columns)
    data_module = IndexedTemporalDataModule(
        target_frame,
        lookback=int(temporal["lookback"]),
        horizon=int(temporal["horizon"]),
        train_end_week=int(temporal["train_end_week"]),
        forecast_origin=int(temporal["forecast_origin"]),
        validation_origin=int(temporal["validation_origin"]),
        window_stride=int(temporal["window_stride"]),
        seed=int(data_manifest.get("selection", {}).get("seed", 42)),
        part_col=str(schema["id_col"]),
        date_col=str(schema["date_col"]),
        qty_col=str(schema["target_col"]),
    )
    summary = data_module.summary
    expected_dataset = data_manifest.get("dataset")
    if isinstance(expected_dataset, Mapping):
        for key in (
            "row_count",
            "series_count",
            "source_min_week",
            "source_max_week",
            "train_windows",
            "validation_windows",
        ):
            if key in expected_dataset and int(expected_dataset[key]) != summary[key]:
                raise ValueError(
                    f"Dataset manifest mismatch for {key}: "
                    f"expected {expected_dataset[key]}, got {summary[key]}."
                )

    all_predictions: list[pl.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for model_key in model_keys:
        result = results.get(model_key)
        if not isinstance(result, Mapping):
            raise ValueError(f"No training result for {model_key!r}.")
        checkpoint_path = _resolve_checkpoint_path(artifact_dir, result)
        print(
            f"[qualification] evaluating {model_key}: {checkpoint_path}",
            flush=True,
        )
        predictions, metrics = _evaluate_model(
            model_key=model_key,
            checkpoint_path=checkpoint_path,
            data_module=data_module,
            horizon=int(temporal["horizon"]),
            validation_origin=int(temporal["validation_origin"]),
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        all_predictions.append(predictions)
        metric_rows.append(metrics)
        print(
            "[qualification] "
            f"{model_key}: MAE={metrics['mae']:.6f}, "
            f"WAPE={metrics['wape_pct']:.4f}%, "
            f"sMAPE={metrics['smape_pct']:.4f}%",
            flush=True,
        )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else artifact_dir / "qualification_evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = _with_error_columns(pl.concat(all_predictions, how="vertical"))
    metrics = _add_metric_ranks(metric_rows)
    metric_frame = pl.DataFrame(metrics).sort("mae_rank")
    by_series = _aggregate_metrics(
        predictions,
        group_by=("model_key", "oper_part_no"),
    )
    by_horizon = _aggregate_metrics(
        predictions,
        group_by=("model_key", "horizon_step", "demand_dt"),
    )

    predictions.write_parquet(
        output_dir / "qualification_predictions.parquet",
        compression="zstd",
    )
    metric_frame.write_csv(output_dir / "qualification_metrics.csv")
    by_series.write_parquet(
        output_dir / "qualification_metrics_by_series.parquet",
        compression="zstd",
    )
    by_horizon.write_parquet(
        output_dir / "qualification_metrics_by_horizon.parquet",
        compression="zstd",
    )
    (output_dir / "production_refit_epochs.json").write_text(
        json.dumps(refit_policy, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if extension_analysis is not None:
        (output_dir / "epoch_extension_analysis.json").write_text(
            json.dumps(extension_analysis, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    payload = {
        "schema_version": 1,
        "artifact_dir": str(artifact_dir),
        "target_source": str(source),
        "target_source_sha256": _sha256_file(source),
        "training_log": str(args.training_log.expanduser().resolve()),
        "device": args.device,
        "batch_size": args.batch_size,
        "dataset": summary,
        "temporal_contract": dict(temporal),
        "metric_contract": {
            "aggregation": "micro over every series-horizon observation",
            "point_output": (
                "public predictor point output; PatchTST Quantile uses q50"
            ),
            "mae": "mean(abs(prediction - actual))",
            "wape": (
                "sum(abs(prediction - actual)) / "
                "(sum(abs(actual)) + 1e-8)"
            ),
            "smape": (
                "mean(2 * abs(prediction - actual) / "
                "(abs(actual) + abs(prediction) + 1e-8))"
            ),
            "ratio_fields": ["wape", "smape"],
            "percentage_fields": ["wape_pct", "smape_pct"],
        },
        "metrics": metrics,
        "production_refit_policy": refit_policy,
        "epoch_extension_analysis": extension_analysis,
        "epoch_history": {
            key: [asdict(record) for record in histories[key]]
            for key in model_keys
        },
        "outputs": {
            "metrics_csv": "qualification_metrics.csv",
            "predictions_parquet": "qualification_predictions.parquet",
            "by_series_parquet": "qualification_metrics_by_series.parquet",
            "by_horizon_parquet": "qualification_metrics_by_horizon.parquet",
            "refit_epochs_json": "production_refit_epochs.json",
        },
    }
    if extension_analysis is not None:
        payload["outputs"][
            "epoch_extension_analysis_json"
        ] = "epoch_extension_analysis.json"
    (output_dir / "qualification_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[qualification] outputs: {output_dir}", flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DSIO last-origin qualification checkpoints and derive "
            "fixed production-refit epoch counts."
        )
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="endo_only directory containing data/training manifests and checkpoints",
    )
    parser.add_argument(
        "--training-log",
        type=Path,
        required=True,
        help="total-train log containing model headings and epoch loss lines",
    )
    parser.add_argument(
        "--target-source",
        type=Path,
        default=None,
        help="canonical target Parquet override; defaults to data manifest source",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="defaults to <artifact-dir>/qualification_evaluation",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--baseline-max-epoch",
        type=int,
        default=None,
        help=(
            "compare the best validation epoch at or before this cutoff "
            "against all later logged epochs"
        ),
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.baseline_max_epoch is not None and args.baseline_max_epoch <= 0:
        raise ValueError("--baseline-max-epoch must be positive.")
    run_evaluation(args)


if __name__ == "__main__":
    main()

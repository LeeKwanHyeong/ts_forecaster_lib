#!/usr/bin/env python3
"""Strictly restore and forecast with a DSIO production-refit checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modeling_module import load_predictor
from modeling_module.data_loader.temporal import add_period


ID_COL = "oper_part_no"
DATE_COL = "demand_dt"
Y_COL = "demand_qty"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synchronize_cuda(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _parse_expected_config(items: list[str] | None) -> dict[str, Any]:
    expected: dict[str, Any] = {}
    for item in items or []:
        key, separator, raw_value = item.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ValueError(
                "--expected-config values must use KEY=JSON_VALUE syntax, "
                f"got {item!r}."
            )
        try:
            expected[key] = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON value in --expected-config {item!r}."
            ) from exc
    return expected


def _normalize_contract_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalize_contract_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_contract_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_contract_value(item)
            for key, item in value.items()
        }
    return value


def _load_inference_history(
    source: Path,
    *,
    lookback: int,
    train_end_week: int,
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    frame = (
        pl.read_parquet(source)
        .select(ID_COL, DATE_COL, Y_COL)
        .with_columns(
            pl.col(ID_COL).cast(pl.String),
            pl.col(DATE_COL).cast(pl.Int64),
            pl.col(Y_COL).cast(pl.Float32),
        )
        .sort([ID_COL, DATE_COL])
    )
    if frame.is_empty():
        raise ValueError("Target source is empty.")
    if frame.null_count().select(pl.sum_horizontal(pl.all())).item() > 0:
        raise ValueError("Target source contains nulls in required columns.")
    duplicate_count = (
        frame.group_by([ID_COL, DATE_COL])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate_count:
        raise ValueError(
            f"Target source contains {duplicate_count} duplicate part/week keys."
        )
    source_max_week = int(frame[DATE_COL].max())
    if source_max_week != train_end_week:
        raise ValueError(
            "Target source upper bound mismatch: "
            f"expected {train_end_week}, got {source_max_week}."
        )

    recent = (
        frame.group_by(ID_COL, maintain_order=True)
        .tail(lookback)
        .sort([ID_COL, DATE_COL])
    )
    ids: list[str] = []
    histories: list[np.ndarray] = []
    for group in recent.partition_by(ID_COL, maintain_order=True):
        part_id = str(group[ID_COL][0])
        if group.height != lookback:
            raise ValueError(
                f"Series {part_id!r} has {group.height} inference rows; "
                f"expected {lookback}."
            )
        weeks = group[DATE_COL].to_numpy()
        if int(weeks[-1]) != train_end_week:
            raise ValueError(
                f"Series {part_id!r} ends at {int(weeks[-1])}, "
                f"expected {train_end_week}."
            )
        ordinals = np.asarray(
            [
                date.fromisocalendar(int(week) // 100, int(week) % 100, 1)
                .toordinal()
                for week in weeks
            ],
            dtype=np.int64,
        )
        if not np.all(np.diff(ordinals) == 7):
            raise ValueError(
                f"Series {part_id!r} is not continuous over its inference history."
            )
        ids.append(part_id)
        histories.append(
            group[Y_COL].to_numpy().astype(np.float32, copy=False)
        )

    x = np.stack(histories, axis=0)[..., None]
    return ids, x, {
        "row_count": frame.height,
        "series_count": len(ids),
        "source_min_week": int(frame[DATE_COL].min()),
        "source_max_week": source_max_week,
    }


def _validate_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    *,
    expected_model_key: str,
    expected_config: Mapping[str, Any],
    lookback: int,
    horizon: int,
    expected_seed: int,
    expected_epochs: int,
) -> None:
    meta = checkpoint.get("meta") or {}
    config = checkpoint.get("config") or checkpoint.get("cfg_state") or {}
    expected_meta = {
        "model_key": expected_model_key,
        "training_mode": "production_refit",
        "validation_enabled": False,
        "state_selection": "final_epoch",
        "configured_epochs": expected_epochs,
        "completed_epochs": expected_epochs,
        "random_seed": expected_seed,
    }
    for key, expected in expected_meta.items():
        actual = meta.get(key)
        if actual != expected:
            raise ValueError(
                f"Checkpoint metadata mismatch for {key!r}: "
                f"expected {expected!r}, got {actual!r}."
            )
    config_contract = {
        "lookback": lookback,
        "horizon": horizon,
        **dict(expected_config),
    }
    for key, expected in config_contract.items():
        actual = config.get(key)
        if _normalize_contract_value(actual) != _normalize_contract_value(expected):
            raise ValueError(
                f"Checkpoint config mismatch for {key!r}: "
                f"expected {expected!r}, got {actual!r}."
            )


def verify(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    target_source = args.target_source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_origin = add_period(args.train_end_week, 1, "weekly")
    if args.forecast_origin != expected_origin:
        raise ValueError(
            f"forecast_origin must be {expected_origin}, "
            f"got {args.forecast_origin}."
        )

    raw_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    expected_config = _parse_expected_config(args.expected_config)
    if not expected_config and args.expected_model_key == "patchtst_base":
        # Preserve the original PatchTST Small verifier invocation.
        expected_config = {
            "d_model": 128,
            "n_layers": 2,
            "d_ff": 512,
        }
    _validate_checkpoint_contract(
        raw_checkpoint,
        expected_model_key=args.expected_model_key,
        expected_config=expected_config,
        lookback=args.lookback,
        horizon=args.horizon,
        expected_seed=args.expected_seed,
        expected_epochs=args.expected_epochs,
    )
    part_ids, x, source_summary = _load_inference_history(
        target_source,
        lookback=args.lookback,
        train_end_week=args.train_end_week,
    )

    load_started = time.perf_counter()
    predictor = load_predictor(
        str(checkpoint_path),
        device=args.device,
        strict=True,
    )
    load_seconds = time.perf_counter() - load_started
    if predictor.model_key != args.expected_model_key:
        raise ValueError(
            f"Strict restore resolved {predictor.model_key!r}, "
            f"expected {args.expected_model_key!r}."
        )

    predictions: list[np.ndarray] = []
    _synchronize_cuda(args.device)
    inference_started = time.perf_counter()
    for start in range(0, len(part_ids), args.batch_size):
        stop = min(start + args.batch_size, len(part_ids))
        batch_ids = part_ids[start:stop]
        output = predictor.predict(
            {
                "x": torch.from_numpy(x[start:stop]),
                "part_ids": batch_ids,
            },
            horizon=args.horizon,
        )
        if not isinstance(output, Mapping) or "point" not in output:
            raise ValueError("Public predictor did not return a `point` output.")
        point = np.asarray(output["point"], dtype=np.float32)
        expected_size = len(batch_ids) * args.horizon
        if point.size != expected_size:
            raise ValueError(
                f"Prediction size mismatch: expected {expected_size}, "
                f"got {point.size}."
            )
        point = point.reshape(len(batch_ids), args.horizon)
        if not np.isfinite(point).all():
            raise ValueError("Production forecast contains non-finite values.")
        predictions.append(point)
    _synchronize_cuda(args.device)
    inference_seconds = time.perf_counter() - inference_started
    prediction = np.concatenate(predictions, axis=0)

    forecast_weeks = np.asarray(
        [
            add_period(args.forecast_origin, step, "weekly")
            for step in range(args.horizon)
        ],
        dtype=np.int64,
    )
    forecast = pl.DataFrame(
        {
            ID_COL: np.repeat(np.asarray(part_ids, dtype=str), args.horizon),
            "forecast_origin": np.repeat(
                args.forecast_origin,
                len(part_ids) * args.horizon,
            ),
            "horizon_step": np.tile(
                np.arange(1, args.horizon + 1, dtype=np.int16),
                len(part_ids),
            ),
            DATE_COL: np.tile(forecast_weeks, len(part_ids)),
            "prediction": prediction.reshape(-1),
        }
    )
    forecast_path = output_dir / (
        f"production_forecast_{args.forecast_origin}.parquet"
    )
    forecast.write_parquet(forecast_path)

    report = {
        "schema_version": 1,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256_file(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "model_key": predictor.model_key,
            "parameter_count": sum(
                parameter.numel() for parameter in predictor.model.parameters()
            ),
            "cfg_cls": raw_checkpoint.get("cfg_cls"),
            "model_class": raw_checkpoint.get("model_class"),
            "output_spec": raw_checkpoint.get("output_spec"),
            "verified_config": {
                key: (
                    raw_checkpoint.get("config")
                    or raw_checkpoint.get("cfg_state")
                    or {}
                ).get(key)
                for key in ("lookback", "horizon", *expected_config)
            },
            "meta": raw_checkpoint["meta"],
        },
        "source": {
            "path": str(target_source),
            "sha256": _sha256_file(target_source),
            **source_summary,
        },
        "forecast": {
            "path": str(forecast_path),
            "sha256": _sha256_file(forecast_path),
            "forecast_origin": args.forecast_origin,
            "forecast_end_week": int(forecast_weeks[-1]),
            "horizon": args.horizon,
            "row_count": forecast.height,
            "series_count": len(part_ids),
            "minimum": float(prediction.min()),
            "maximum": float(prediction.max()),
            "mean": float(prediction.mean()),
            "negative_count": int((prediction < 0).sum()),
        },
        "runtime": {
            "device": args.device,
            "strict_load": True,
            "batch_size": args.batch_size,
            "load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "series_per_second": len(part_ids) / max(inference_seconds, 1e-12),
        },
    }
    report_path = output_dir / "production_artifact_verification.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly verify a DSIO production-refit checkpoint and "
            "materialize the 202545 forecast."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lookback", type=int, default=52)
    parser.add_argument("--horizon", type=int, default=27)
    parser.add_argument("--train-end-week", type=int, default=202544)
    parser.add_argument("--forecast-origin", type=int, default=202545)
    parser.add_argument("--expected-model-key", default="patchtst_base")
    parser.add_argument(
        "--expected-config",
        action="append",
        default=[],
        metavar="KEY=JSON_VALUE",
        help=(
            "Repeatable checkpoint architecture assertion. Values are parsed "
            "as JSON, for example d_model=384 or n_blocks=[1,1,1]."
        ),
    )
    parser.add_argument("--expected-seed", type=int, default=42)
    parser.add_argument("--expected-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    verify(build_parser().parse_args())


if __name__ == "__main__":
    main()

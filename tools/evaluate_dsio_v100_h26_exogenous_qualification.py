#!/usr/bin/env python3
"""Evaluate all DSIO V100 L52/H26 exogenous qualification checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np
import polars as pl
import torch


ROOT: Final = Path(__file__).resolve().parents[1]
SRC_ROOT: Final = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modeling_module import load_predictor  # noqa: E402
from tools.dsio_v100_h26_contract import (  # noqa: E402
    HORIZON,
    LOOKBACK,
    SEED,
    TRAIN_END_WEEK,
    VALIDATION_ORIGIN,
    V100H26ContractError,
    canonical_json_sha256,
    file_sha256,
    load_training_input_manifest,
    write_secure_json,
)
from tools.run_dsio_v100_h26_exogenous_qualification import (  # noqa: E402
    MODEL_SPECS,
    _build_datamodule,
    _load_target,
    configure_torch_runtime,
)


EXPECTED_SOURCE_SERIES: Final = 7_000
EXPECTED_ELIGIBLE_SERIES: Final = 6_952
EXPECTED_EXCLUDED_SERIES: Final = 48
OUTPUT_POLICIES: Final = ("raw", "nonnegative")


def _source_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V100H26ContractError(f"{label} must be one JSON object")
    return value


def _load_qualification_receipt(path: Path) -> dict[str, Any]:
    raw = dict(
        _require_mapping(
            json.loads(path.read_text(encoding="ascii")),
            label="qualification receipt",
        )
    )
    receipt_sha256 = raw.pop("receipt_sha256", None)
    if receipt_sha256 != canonical_json_sha256(raw):
        raise V100H26ContractError("qualification receipt seal mismatch")
    if raw.get("status") != "PASS":
        raise V100H26ContractError("qualification receipt status must be PASS")
    models = raw.get("models")
    if not isinstance(models, list) or len(models) != len(MODEL_SPECS):
        raise V100H26ContractError("qualification model inventory drifted")
    if [item.get("model_key") for item in models] != [
        spec.model_key for spec in MODEL_SPECS
    ]:
        raise V100H26ContractError("qualification model order drifted")
    return {**raw, "receipt_sha256": receipt_sha256}


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


@dataclass
class ValidationMetricAccumulator:
    """Accumulate point metrics without retaining all series predictions."""

    horizon: int

    def __post_init__(self) -> None:
        self.count = np.zeros(self.horizon, dtype=np.int64)
        self.absolute_error_sum = np.zeros(self.horizon, dtype=np.float64)
        self.error_sum = np.zeros(self.horizon, dtype=np.float64)
        self.smape_sum = np.zeros(self.horizon, dtype=np.float64)
        self.actual_absolute_sum = np.zeros(self.horizon, dtype=np.float64)
        self.actual_sum = np.zeros(self.horizon, dtype=np.float64)
        self.prediction_sum = np.zeros(self.horizon, dtype=np.float64)
        self.negative_prediction_count = np.zeros(self.horizon, dtype=np.int64)

    def update(self, actual: np.ndarray, prediction: np.ndarray) -> None:
        actual = np.asarray(actual, dtype=np.float64)
        prediction = np.asarray(prediction, dtype=np.float64)
        if actual.shape != prediction.shape:
            raise ValueError(
                f"actual/prediction shape mismatch: {actual.shape} != "
                f"{prediction.shape}"
            )
        if actual.ndim != 2 or actual.shape[1] != self.horizon:
            raise ValueError(
                f"metrics require [B,{self.horizon}], got {actual.shape}"
            )
        if not np.isfinite(actual).all() or not np.isfinite(prediction).all():
            raise ValueError("metrics require finite actuals and predictions")

        error = prediction - actual
        absolute_error = np.abs(error)
        denominator = np.abs(actual) + np.abs(prediction)
        smape = np.divide(
            2.0 * absolute_error,
            denominator,
            out=np.zeros_like(absolute_error),
            where=denominator > 0.0,
        )
        self.count += actual.shape[0]
        self.absolute_error_sum += absolute_error.sum(axis=0)
        self.error_sum += error.sum(axis=0)
        self.smape_sum += smape.sum(axis=0)
        self.actual_absolute_sum += np.abs(actual).sum(axis=0)
        self.actual_sum += actual.sum(axis=0)
        self.prediction_sum += prediction.sum(axis=0)
        self.negative_prediction_count += (prediction < 0.0).sum(axis=0)

    @staticmethod
    def _metrics(
        *,
        count: int,
        absolute_error_sum: float,
        error_sum: float,
        smape_sum: float,
        actual_absolute_sum: float,
        actual_sum: float,
        prediction_sum: float,
        negative_prediction_count: int,
    ) -> dict[str, int | float]:
        mae = _safe_ratio(absolute_error_sum, float(count))
        wape = _safe_ratio(absolute_error_sum, actual_absolute_sum)
        smape = _safe_ratio(smape_sum, float(count))
        bias = _safe_ratio(error_sum, float(count))
        normalized_bias = _safe_ratio(error_sum, actual_absolute_sum)
        return {
            "forecast_points": count,
            "mae": mae,
            "wape": wape,
            "wape_percent": 100.0 * wape,
            "smape": smape,
            "smape_percent": 100.0 * smape,
            "bias": bias,
            "normalized_bias": normalized_bias,
            "normalized_bias_percent": 100.0 * normalized_bias,
            "actual_mean": _safe_ratio(actual_sum, float(count)),
            "prediction_mean": _safe_ratio(prediction_sum, float(count)),
            "negative_prediction_count": negative_prediction_count,
            "negative_prediction_rate": _safe_ratio(
                float(negative_prediction_count), float(count)
            ),
        }

    def finalize(self) -> tuple[dict[str, int | float], list[dict[str, Any]]]:
        if not self.count.size or np.any(self.count == 0):
            raise ValueError("cannot finalize empty validation metrics")
        overall = self._metrics(
            count=int(self.count.sum()),
            absolute_error_sum=float(self.absolute_error_sum.sum()),
            error_sum=float(self.error_sum.sum()),
            smape_sum=float(self.smape_sum.sum()),
            actual_absolute_sum=float(self.actual_absolute_sum.sum()),
            actual_sum=float(self.actual_sum.sum()),
            prediction_sum=float(self.prediction_sum.sum()),
            negative_prediction_count=int(self.negative_prediction_count.sum()),
        )
        by_horizon = []
        for step in range(self.horizon):
            by_horizon.append(
                {
                    "horizon_step": step,
                    "horizon_label": f"W{step}",
                    **self._metrics(
                        count=int(self.count[step]),
                        absolute_error_sum=float(self.absolute_error_sum[step]),
                        error_sum=float(self.error_sum[step]),
                        smape_sum=float(self.smape_sum[step]),
                        actual_absolute_sum=float(
                            self.actual_absolute_sum[step]
                        ),
                        actual_sum=float(self.actual_sum[step]),
                        prediction_sum=float(self.prediction_sum[step]),
                        negative_prediction_count=int(
                            self.negative_prediction_count[step]
                        ),
                    ),
                }
            )
        return overall, by_horizon


def evaluate_prediction_batches(
    *,
    predictor: Any,
    loader: Any,
    expected_series_count: int,
) -> tuple[dict[str, dict[str, Any]], float]:
    accumulators = {
        policy: ValidationMetricAccumulator(HORIZON)
        for policy in OUTPUT_POLICIES
    }
    observed_series = 0
    started = time.perf_counter()
    for batch in loader:
        actual = np.asarray(batch[1], dtype=np.float64)
        prediction_payload = predictor.predict(batch, horizon=HORIZON)
        if "point" not in prediction_payload:
            raise V100H26ContractError("predictor did not return point output")
        raw = np.asarray(prediction_payload["point"], dtype=np.float64)
        if raw.size != actual.size:
            raise V100H26ContractError(
                f"prediction size mismatch: {raw.size} != {actual.size}"
            )
        raw = raw.reshape(actual.shape)
        accumulators["raw"].update(actual, raw)
        accumulators["nonnegative"].update(actual, np.maximum(raw, 0.0))
        observed_series += actual.shape[0]
    elapsed = time.perf_counter() - started
    if observed_series != expected_series_count:
        raise V100H26ContractError(
            "validation series count drifted: "
            f"{observed_series} != {expected_series_count}"
        )

    result: dict[str, dict[str, Any]] = {}
    for policy, accumulator in accumulators.items():
        overall, by_horizon = accumulator.finalize()
        result[policy] = {
            "overall": overall,
            "by_horizon": by_horizon,
        }
    return result, elapsed


def _validate_data_summary(summary: Mapping[str, Any]) -> None:
    expected = {
        "source_series_count": EXPECTED_SOURCE_SERIES,
        "series_count": EXPECTED_ELIGIBLE_SERIES,
        "excluded_series_count": EXPECTED_EXCLUDED_SERIES,
        "validation_windows": EXPECTED_ELIGIBLE_SERIES,
        "validation_target_min_week": VALIDATION_ORIGIN,
        "validation_target_max_week": TRAIN_END_WEEK,
    }
    drift = {
        key: (summary.get(key), value)
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if drift:
        raise V100H26ContractError(
            f"validation data summary drifted: {drift}"
        )


def _checkpoint_from_receipt(
    *,
    qualification_root: Path,
    model_receipt: Mapping[str, Any],
    model_key: str,
    checkpoint_filename: str,
) -> tuple[Path, str]:
    checkpoint = _require_mapping(
        model_receipt.get("checkpoint"),
        label=f"{model_key} checkpoint receipt",
    )
    checkpoint_path = qualification_root / model_key / checkpoint_filename
    if Path(str(checkpoint.get("path", ""))).name != checkpoint_filename:
        raise V100H26ContractError(
            f"{model_key} checkpoint filename drifted"
        )
    expected_sha256 = str(checkpoint.get("checkpoint_sha256", ""))
    observed_sha256 = file_sha256(checkpoint_path)
    if observed_sha256 != expected_sha256:
        raise V100H26ContractError(
            f"{model_key} checkpoint SHA-256 mismatch"
        )
    return checkpoint_path, observed_sha256


def evaluate(
    *,
    target_source: Path,
    input_manifest: Path,
    qualification_root: Path,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    device: str,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if output_dir.exists():
        raise V100H26ContractError(
            f"evaluation output already exists; refusing overwrite: {output_dir}"
        )
    manifest = load_training_input_manifest(
        input_manifest,
        target_source=target_source,
    )
    qualification_receipt_path = qualification_root / "qualification-receipt.json"
    qualification_receipt = _load_qualification_receipt(
        qualification_receipt_path
    )
    target = _load_target(target_source, sample_part_count=None)
    configure_torch_runtime()

    output_dir.mkdir(parents=True)
    model_results: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    try:
        for spec, model_receipt in zip(
            MODEL_SPECS,
            qualification_receipt["models"],
            strict=True,
        ):
            datamodule = _build_datamodule(target, spec=spec)
            _validate_data_summary(datamodule.summary)
            loader = datamodule.get_val_loader(
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=device.startswith("cuda"),
                persistent_workers=num_workers > 0,
                prefetch_factor=4,
            )
            checkpoint_path, checkpoint_sha256 = _checkpoint_from_receipt(
                qualification_root=qualification_root,
                model_receipt=model_receipt,
                model_key=spec.model_key,
                checkpoint_filename=spec.checkpoint_filename,
            )
            predictor = load_predictor(
                str(checkpoint_path),
                device=device,
                strict=True,
            )
            metrics, inference_seconds = evaluate_prediction_batches(
                predictor=predictor,
                loader=loader,
                expected_series_count=EXPECTED_ELIGIBLE_SERIES,
            )
            for policy in OUTPUT_POLICIES:
                overall_rows.append(
                    {
                        "model_key": spec.model_key,
                        "plan_model_name": spec.plan_model_name,
                        "output_policy": policy,
                        "series_count": EXPECTED_ELIGIBLE_SERIES,
                        **metrics[policy]["overall"],
                    }
                )
                for horizon_metrics in metrics[policy]["by_horizon"]:
                    horizon_rows.append(
                        {
                            "model_key": spec.model_key,
                            "plan_model_name": spec.plan_model_name,
                            "output_policy": policy,
                            **horizon_metrics,
                        }
                    )
            model_results.append(
                {
                    "model_key": spec.model_key,
                    "plan_model_name": spec.plan_model_name,
                    "checkpoint": {
                        "path": str(checkpoint_path),
                        "sha256": checkpoint_sha256,
                    },
                    "inference_seconds": inference_seconds,
                    "series_per_second": EXPECTED_ELIGIBLE_SERIES
                    / inference_seconds,
                    "metrics": metrics,
                }
            )
            del predictor, datamodule, loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        overall_path = output_dir / "validation-overall.csv"
        horizon_path = output_dir / "validation-by-horizon.csv"
        pl.DataFrame(overall_rows).sort(
            ["output_policy", "mae", "model_key"]
        ).write_csv(overall_path)
        pl.DataFrame(horizon_rows).sort(
            ["output_policy", "horizon_step", "mae", "model_key"]
        ).write_csv(horizon_path)
        ranking = sorted(
            (
                {
                    "rank": 0,
                    "model_key": result["model_key"],
                    "mae": result["metrics"]["nonnegative"]["overall"]["mae"],
                    "wape": result["metrics"]["nonnegative"]["overall"]["wape"],
                    "smape": result["metrics"]["nonnegative"]["overall"]["smape"],
                    "bias": result["metrics"]["nonnegative"]["overall"]["bias"],
                    "normalized_bias": result["metrics"]["nonnegative"]["overall"]["normalized_bias"],
                }
                for result in model_results
            ),
            key=lambda row: (row["mae"], row["model_key"]),
        )
        for rank, row in enumerate(ranking, start=1):
            row["rank"] = rank

        receipt: dict[str, Any] = {
            "evaluation_format_version": 1,
            "status": "PASS",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "contract": "dsio-v100-weekly-l52-h26-exogenous-validation-v1",
            "source_commit": _source_commit(),
            "scope": {
                "lookback": LOOKBACK,
                "horizon": HORIZON,
                "validation_origin": VALIDATION_ORIGIN,
                "validation_end_week": TRAIN_END_WEEK,
                "source_series_count": EXPECTED_SOURCE_SERIES,
                "eligible_series_count": EXPECTED_ELIGIBLE_SERIES,
                "excluded_series_count": EXPECTED_EXCLUDED_SERIES,
                "forecast_points_per_model": (
                    EXPECTED_ELIGIBLE_SERIES * HORIZON
                ),
                "seed": SEED,
            },
            "metric_contract": {
                "mae": "mean(abs(prediction - actual))",
                "wape": "sum(abs(prediction - actual)) / sum(abs(actual))",
                "smape": (
                    "mean(2 * abs(prediction - actual) / "
                    "(abs(prediction) + abs(actual))); zero/zero=0"
                ),
                "bias": "mean(prediction - actual)",
                "normalized_bias": (
                    "sum(prediction - actual) / sum(abs(actual))"
                ),
                "output_policies": {
                    "raw": "checkpoint point output without clipping",
                    "nonnegative": "max(0, raw point output)",
                },
            },
            "inputs": {
                "target_source": {
                    "path": str(target_source),
                    "sha256": file_sha256(target_source),
                },
                "input_manifest": {
                    "path": str(input_manifest),
                    "sha256": manifest["file_sha256"],
                },
                "qualification_receipt": {
                    "path": str(qualification_receipt_path),
                    "file_sha256": file_sha256(qualification_receipt_path),
                    "receipt_sha256": qualification_receipt["receipt_sha256"],
                },
            },
            "runtime": {
                "device": device,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_name": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
                "batch_size": batch_size,
                "num_workers": num_workers,
            },
            "ranking_by_nonnegative_mae": ranking,
            "models": model_results,
            "artifacts": {
                "overall_csv": {
                    "path": overall_path.name,
                    "sha256": file_sha256(overall_path),
                },
                "horizon_csv": {
                    "path": horizon_path.name,
                    "sha256": file_sha256(horizon_path),
                },
            },
        }
        receipt["receipt_sha256"] = canonical_json_sha256(receipt)
        write_secure_json(output_dir / "validation-metrics.json", receipt)
        return receipt
    except BaseException:
        (output_dir / "evaluation-status.txt").write_text(
            "FAILED\n", encoding="ascii"
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-source", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    qualification_root = args.qualification_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else qualification_root / "validation-evaluation"
    )
    receipt = evaluate(
        target_source=args.target_source.expanduser().resolve(),
        input_manifest=args.input_manifest.expanduser().resolve(),
        qualification_root=qualification_root,
        output_dir=output_dir,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=str(args.device),
    )
    print(json.dumps(receipt["ranking_by_nonnegative_mae"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

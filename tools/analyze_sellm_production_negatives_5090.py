#!/usr/bin/env python3
"""Analyze SELLM production raw negatives and clip-zero distribution shift."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
import polars as pl
import torch


ROOT: Final = Path(__file__).resolve().parents[1]
SRC_ROOT: Final = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modeling_module.api import load_predictor  # noqa: E402
from tools.dsio_v100_h26_contract import (  # noqa: E402
    V100H26ContractError,
    canonical_json_sha256,
    file_sha256,
    load_training_input_manifest,
    write_secure_json,
)
from tools.run_sellm_production_refit_5090 import (  # noqa: E402
    BATCH_SIZE,
    FORECAST_ORIGIN,
    HORIZON,
    LOOKBACK,
    TRAIN_END_WEEK,
    _build_datamodule,
    _latest_histories,
    _load_target,
)


ZERO_RATIO_LABELS: Final = (
    "0-25%",
    "25-50%",
    "50-75%",
    "75-<100%",
    "100%",
)
HISTORY_MEAN_LABELS: Final = (
    "zero",
    "(0,0.5]",
    "(0.5,1]",
    "(1,3]",
    "(3,10]",
    ">10",
)


def _source_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not np.isfinite(denominator) or denominator == 0.0:
        return None
    return float(numerator / denominator)


def _load_included_part_ids(path: Path) -> tuple[list[str], dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pl.read_parquet(path)
    elif suffix == ".csv":
        frame = pl.read_csv(path)
    else:
        raise V100H26ContractError(
            "included-part source must be Parquet or CSV: " + str(path)
        )
    if "oper_part_no" not in frame.columns:
        raise V100H26ContractError(
            "included-part source must contain oper_part_no"
        )
    values = frame.get_column("oper_part_no").cast(pl.String)
    if values.null_count() != 0:
        raise V100H26ContractError("included oper_part_no values cannot be null")
    part_ids = values.unique(maintain_order=True).to_list()
    if not part_ids:
        raise V100H26ContractError("included-part source is empty")
    return part_ids, {
        "path": str(path),
        "sha256": file_sha256(path),
        "part_count": len(part_ids),
    }


def _filter_histories(
    histories: np.ndarray,
    part_ids: Sequence[str],
    included_part_ids: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    positions = {part_id: index for index, part_id in enumerate(part_ids)}
    if len(positions) != len(part_ids):
        raise V100H26ContractError("target source contains duplicate series IDs")
    included = set(included_part_ids)
    missing = sorted(included - positions.keys())
    if missing:
        raise V100H26ContractError(
            "included-part source contains IDs absent from target source: "
            + ", ".join(missing[:10])
        )
    ordered = [part_id for part_id in part_ids if part_id in included]
    indices = [positions[part_id] for part_id in ordered]
    return histories[indices], ordered


def _zero_ratio_bins(values: np.ndarray) -> np.ndarray:
    return np.select(
        [
            values < 0.25,
            values < 0.50,
            values < 0.75,
            values < 1.0,
        ],
        ZERO_RATIO_LABELS[:-1],
        default=ZERO_RATIO_LABELS[-1],
    )


def _history_mean_bins(values: np.ndarray) -> np.ndarray:
    return np.select(
        [
            values == 0.0,
            values <= 0.5,
            values <= 1.0,
            values <= 3.0,
            values <= 10.0,
        ],
        HISTORY_MEAN_LABELS[:-1],
        default=HISTORY_MEAN_LABELS[-1],
    )


def _group_summary(
    labels: np.ndarray,
    ordered_labels: Sequence[str],
    raw: np.ndarray,
) -> list[dict[str, Any]]:
    clipped = np.maximum(raw, 0.0)
    rows: list[dict[str, Any]] = []
    for label in ordered_labels:
        mask = labels == label
        if not mask.any():
            continue
        grouped_raw = raw[mask]
        grouped_clipped = clipped[mask]
        negative_count = int((grouped_raw < 0).sum())
        clip_added = float(grouped_clipped.sum() - grouped_raw.sum())
        rows.append(
            {
                "group": label,
                "series_count": int(mask.sum()),
                "point_count": int(grouped_raw.size),
                "raw_negative_count": negative_count,
                "raw_negative_rate": negative_count / int(grouped_raw.size),
                "series_any_negative_rate": float(
                    (grouped_raw < 0).any(axis=1).mean()
                ),
                "raw_mean": float(grouped_raw.mean()),
                "clipped_mean": float(grouped_clipped.mean()),
                "clip_added_total": clip_added,
                "clip_added_share_of_clipped": _safe_ratio(
                    clip_added,
                    float(grouped_clipped.sum()),
                ),
                "raw_min": float(grouped_raw.min()),
                "raw_max": float(grouped_raw.max()),
            }
        )
    return rows


def _horizon_summary(raw: np.ndarray) -> list[dict[str, Any]]:
    clipped = np.maximum(raw, 0.0)
    rows: list[dict[str, Any]] = []
    for horizon in range(raw.shape[1]):
        horizon_raw = raw[:, horizon]
        horizon_clipped = clipped[:, horizon]
        negative = horizon_raw < 0
        rows.append(
            {
                "horizon": horizon,
                "offset": f"W{horizon}",
                "token_segment": 1 if horizon < 13 else 2,
                "raw_negative_count": int(negative.sum()),
                "raw_negative_rate": float(negative.mean()),
                "raw_mean": float(horizon_raw.mean()),
                "clipped_mean": float(horizon_clipped.mean()),
                "clip_uplift_mean": float(
                    horizon_clipped.mean() - horizon_raw.mean()
                ),
                "raw_min": float(horizon_raw.min()),
                "raw_p05": float(np.quantile(horizon_raw, 0.05)),
                "raw_median": float(np.median(horizon_raw)),
                "raw_p95": float(np.quantile(horizon_raw, 0.95)),
                "raw_max": float(horizon_raw.max()),
            }
        )
    return rows


def _negative_magnitude_summary(raw: np.ndarray) -> list[dict[str, Any]]:
    magnitudes = -raw[raw < 0]
    labels = ("<=0.001", "(0.001,0.1]", "(0.1,1]", "(1,5]", ">5")
    masks = (
        magnitudes <= 0.001,
        (magnitudes > 0.001) & (magnitudes <= 0.1),
        (magnitudes > 0.1) & (magnitudes <= 1.0),
        (magnitudes > 1.0) & (magnitudes <= 5.0),
        magnitudes > 5.0,
    )
    total_count = max(int(magnitudes.size), 1)
    total_volume = max(float(magnitudes.sum()), np.finfo(np.float64).eps)
    return [
        {
            "magnitude": label,
            "count": int(mask.sum()),
            "count_share": int(mask.sum()) / total_count,
            "negative_volume": float(magnitudes[mask].sum()),
            "negative_volume_share": float(magnitudes[mask].sum()) / total_volume,
        }
        for label, mask in zip(labels, masks, strict=True)
    ]


def _qualification_summary(
    receipt_paths: Sequence[Path],
    *,
    fixed_epoch: int,
) -> dict[str, Any] | None:
    if not receipt_paths:
        return None
    rows: list[dict[str, Any]] = []
    for path in receipt_paths:
        payload = json.loads(path.read_text(encoding="ascii"))
        matches = [
            row for row in payload.get("epochs", [])
            if int(row.get("epoch", -1)) == fixed_epoch
        ]
        if len(matches) != 1:
            raise V100H26ContractError(
                f"qualification receipt has no unique epoch {fixed_epoch}: {path}"
            )
        row = matches[0]
        rows.append(
            {
                "path": str(path),
                "seed": int(payload["training"]["seed"]),
                "mae_after_clip": float(row["mae"]),
                "wape_after_clip": float(row["wape"]),
                "smape_after_clip": float(row["smape"]),
                "bias_after_clip": float(row["bias"]),
                "raw_negative_rate": float(row["raw_negative_rate"]),
                "raw_min": float(row["raw_min"]),
            }
        )
    metric_keys = (
        "mae_after_clip",
        "wape_after_clip",
        "smape_after_clip",
        "bias_after_clip",
        "raw_negative_rate",
        "raw_min",
    )
    return {
        "fixed_epoch": fixed_epoch,
        "rows": rows,
        "means": {
            key: float(np.mean([row[key] for row in rows]))
            for key in metric_keys
        },
        "note": (
            "Accuracy and bias metrics use clip_zero outputs. Production-origin "
            "actuals are unavailable, so they cannot validate the production artifact."
        ),
    }


def _predict_public_and_direct(
    predictor,
    histories: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    public_parts: list[np.ndarray] = []
    max_abs_error = 0.0
    mismatch_count = 0
    predictor.model.eval()
    for start in range(0, len(histories), batch_size):
        cpu_batch = torch.from_numpy(histories[start : start + batch_size])
        with torch.inference_mode():
            direct = (
                predictor.model(cpu_batch.to(device))
                .squeeze(-1)
                .detach()
                .cpu()
                .numpy()
            )
        public = np.asarray(
            predictor.predict(cpu_batch, horizon=HORIZON)["point"],
            dtype=np.float32,
        ).reshape(len(cpu_batch), HORIZON)
        difference = np.abs(public - direct)
        max_abs_error = max(max_abs_error, float(difference.max()))
        mismatch_count += int((difference != 0.0).sum())
        public_parts.append(public)
    return np.concatenate(public_parts), {
        "public_direct_max_abs_error": max_abs_error,
        "public_direct_exact_mismatch_count": mismatch_count,
        "public_direct_point_count": int(len(histories) * HORIZON),
    }


def analyze(
    *,
    checkpoint: Path,
    target_source: Path,
    input_manifest: Path,
    output_root: Path,
    qualification_receipts: Sequence[Path],
    device: str,
    batch_size: int,
    included_parts: Path | None = None,
) -> dict[str, Any]:
    if output_root.exists():
        raise V100H26ContractError(
            f"output root already exists; refusing overwrite: {output_root}"
        )
    load_training_input_manifest(input_manifest, target_source=target_source)
    datamodule = _build_datamodule(_load_target(target_source))
    histories = _latest_histories(datamodule)
    part_ids = [series.part_id for series in datamodule._series]
    included_parts_source = None
    if included_parts is not None:
        included_part_ids, included_parts_source = _load_included_part_ids(
            included_parts
        )
        histories, part_ids = _filter_histories(
            histories,
            part_ids,
            included_part_ids,
        )
    predictor = load_predictor(str(checkpoint), device=device, strict=True)
    raw, parity = _predict_public_and_direct(
        predictor,
        histories,
        device=device,
        batch_size=batch_size,
    )
    if not np.isfinite(raw).all():
        raise V100H26ContractError("production raw output contains non-finite values")
    if parity["public_direct_max_abs_error"] != 0.0:
        raise V100H26ContractError(f"public/direct prediction drifted: {parity}")

    clipped = np.maximum(raw, 0.0)
    history_2d = histories[:, :, 0]
    history_mean = history_2d.mean(axis=1)
    history_std = history_2d.std(axis=1)
    zero_ratio = (history_2d == 0.0).mean(axis=1)
    negative = raw < 0
    clip_added_by_series = (clipped - raw).sum(axis=1)
    expected_history_volume = history_mean * HORIZON
    zero_labels = _zero_ratio_bins(zero_ratio)
    scale_labels = _history_mean_bins(history_mean)

    series_frame = pl.DataFrame(
        {
            "oper_part_no": part_ids,
            "history_mean": history_mean,
            "history_std": history_std,
            "history_zero_ratio": zero_ratio,
            "zero_ratio_group": zero_labels,
            "history_mean_group": scale_labels,
            "raw_negative_count": negative.sum(axis=1),
            "raw_negative_rate": negative.mean(axis=1),
            "raw_min": raw.min(axis=1),
            "raw_max": raw.max(axis=1),
            "raw_forecast_total": raw.sum(axis=1),
            "clipped_forecast_total": clipped.sum(axis=1),
            "clip_added_total": clip_added_by_series,
            "history_scaled_h26_total": expected_history_volume,
        }
    )
    horizon_rows = _horizon_summary(raw)
    zero_rows = _group_summary(
        zero_labels,
        ZERO_RATIO_LABELS,
        raw,
    )
    scale_rows = _group_summary(
        scale_labels,
        HISTORY_MEAN_LABELS,
        raw,
    )
    magnitude_rows = _negative_magnitude_summary(raw)
    qualification = _qualification_summary(
        qualification_receipts,
        fixed_epoch=6,
    )

    clip_added_total = float(clipped.sum() - raw.sum())
    summary = {
        "series_count": len(part_ids),
        "point_count": int(raw.size),
        "raw_negative_count": int(negative.sum()),
        "raw_negative_rate": float(negative.mean()),
        "series_any_negative_count": int(negative.any(axis=1).sum()),
        "series_any_negative_rate": float(negative.any(axis=1).mean()),
        "series_majority_negative_count": int((negative.mean(axis=1) > 0.5).sum()),
        "series_majority_negative_rate": float(
            (negative.mean(axis=1) > 0.5).mean()
        ),
        "series_all_negative_count": int(negative.all(axis=1).sum()),
        "series_all_negative_rate": float(negative.all(axis=1).mean()),
        "raw_total": float(raw.sum()),
        "clipped_total": float(clipped.sum()),
        "clip_added_total": clip_added_total,
        "clip_added_mean_per_point": float((clipped - raw).mean()),
        "clip_added_share_of_clipped": _safe_ratio(
            clip_added_total,
            float(clipped.sum()),
        ),
        "raw_mean": float(raw.mean()),
        "clipped_mean": float(clipped.mean()),
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "history_mean": float(history_mean.mean()),
        "history_scaled_h26_total": float(expected_history_volume.sum()),
        "raw_to_history_scaled_volume_ratio": _safe_ratio(
            float(raw.sum()),
            float(expected_history_volume.sum()),
        ),
        "clipped_to_history_scaled_volume_ratio": _safe_ratio(
            float(clipped.sum()),
            float(expected_history_volume.sum()),
        ),
    }
    output_root.mkdir(parents=True)
    series_frame.write_parquet(output_root / "series-analysis.parquet")
    pl.DataFrame(horizon_rows).write_csv(output_root / "horizon-analysis.csv")
    pl.DataFrame(zero_rows).write_csv(output_root / "zero-ratio-analysis.csv")
    pl.DataFrame(scale_rows).write_csv(output_root / "history-scale-analysis.csv")
    pl.DataFrame(magnitude_rows).write_csv(
        output_root / "negative-magnitude-analysis.csv"
    )
    _write_json(output_root / "qualification-baseline.json", qualification)
    receipt: dict[str, Any] = {
        "receipt_format_version": 1,
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": "sellm-production-negative-analysis-v1",
        "source_commit": _source_commit(),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
        },
        "target_source": {
            "path": str(target_source),
            "sha256": file_sha256(target_source),
        },
        "included_parts_source": included_parts_source,
        "forecast_contract": {
            "history_end_week": TRAIN_END_WEEK,
            "forecast_origin": FORECAST_ORIGIN,
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "offsets": "W0-W25",
        },
        "parity": parity,
        "summary": summary,
        "horizon": horizon_rows,
        "zero_ratio_groups": zero_rows,
        "history_scale_groups": scale_rows,
        "negative_magnitude": magnitude_rows,
        "qualification_baseline": qualification,
        "interpretation_boundary": {
            "actual_future_available": False,
            "clip_mae_property": (
                "For nonnegative actual demand, clip_zero is pointwise non-worsening "
                "for MAE and WAPE numerator."
            ),
            "bias_limit": (
                "Actual production bias is unknown. clip_zero increases forecast mean "
                "by exactly clip_added_mean_per_point relative to raw output."
            ),
        },
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    write_secure_json(output_root / "analysis-receipt.json", receipt)
    del predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-source", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--qualification-receipt",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--included-parts", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = analyze(
        checkpoint=args.checkpoint.expanduser().resolve(),
        target_source=args.target_source.expanduser().resolve(),
        input_manifest=args.input_manifest.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        qualification_receipts=[
            path.expanduser().resolve() for path in args.qualification_receipt
        ],
        device=str(args.device),
        batch_size=int(args.batch_size),
        included_parts=(
            args.included_parts.expanduser().resolve()
            if args.included_parts is not None
            else None
        ),
    )
    print(json.dumps(receipt["summary"], ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

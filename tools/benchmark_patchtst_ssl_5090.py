#!/usr/bin/env python3
"""Run reproducible PatchTST sl_only versus full SSL qualification on RTX 5090."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "src/model_test/total_train/dsio_total_running.py"
EVALUATOR = ROOT / "tools/evaluate_dsio_qualification.py"
MODEL_KEY = "patchtst_base"
MODES = ("sl_only", "full")
PATCH_LEN = 13
SUPERVISED_STRIDE = 6


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _gpu_memory_used_mib() -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    first_line = result.stdout.strip().splitlines()[0]
    return float(first_line.strip())


def _gpu_identity() -> dict[str, str] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    values = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
    if len(values) != 3:
        return None
    return {
        "name": values[0],
        "memory_total_mib": values[1],
        "driver_version": values[2],
    }


def _tail(path: Path, line_count: int = 50) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def run_measured_process(
    command: Sequence[str],
    *,
    log_path: Path,
    poll_seconds: float,
) -> dict[str, Any]:
    """Run one process and sample whole-device memory without shell wrappers."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_mib = _gpu_memory_used_mib()
    peak_mib = baseline_mib
    started_at = _utc_now()
    started = time.perf_counter()
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            list(command),
            cwd=ROOT,
            env=env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            sampled_mib = _gpu_memory_used_mib()
            if sampled_mib is not None:
                peak_mib = (
                    sampled_mib
                    if peak_mib is None
                    else max(peak_mib, sampled_mib)
                )
            time.sleep(poll_seconds)
        return_code = process.wait()

    elapsed_seconds = time.perf_counter() - started
    finished_at = _utc_now()
    measurement = {
        "command": list(command),
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": elapsed_seconds,
        "return_code": return_code,
        "gpu_memory": {
            "baseline_mib": baseline_mib,
            "peak_mib": peak_mib,
            "peak_delta_mib": (
                None
                if baseline_mib is None or peak_mib is None
                else peak_mib - baseline_mib
            ),
            "sampling_basis": "nvidia-smi whole-device memory.used",
            "poll_seconds": poll_seconds,
        },
        "log_path": str(log_path),
    }
    if return_code != 0:
        raise RuntimeError(
            f"Command failed with exit code {return_code}: {command}\n"
            f"{_tail(log_path)}"
        )
    return measurement


def _runner_command(
    *,
    python: Path,
    target_source: Path,
    case_root: Path,
    mode: str,
    seed: int,
    pretrain_epochs: int,
    pretrain_stride: int,
    mask_ratio: float,
    supervised_epochs: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> list[str]:
    return [
        str(python),
        str(RUNNER),
        "--mode",
        "endo",
        "--training-mode",
        "qualification",
        "--artifact-root",
        str(case_root),
        "--target-source",
        str(target_source),
        "--ssl-mode",
        mode,
        "--ssl-pretrain-epochs",
        str(pretrain_epochs),
        "--ssl-pretrain-stride",
        str(pretrain_stride),
        "--ssl-mask-ratio",
        str(mask_ratio),
        "--lookback",
        "52",
        "--horizon",
        "27",
        "--train-end-week",
        "202544",
        "--forecast-origin",
        "202545",
        "--validation-origin",
        "202518",
        "--window-stride",
        "4",
        "--warmup-epochs",
        str(supervised_epochs),
        "--spike-epochs",
        "0",
        "--endo-models",
        MODEL_KEY,
        "--clean-output",
        "--device",
        "cuda",
        "--seed",
        str(seed),
        "--endo-batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--prefetch-factor",
        str(prefetch_factor),
        "--patchtst-d-model",
        "128",
        "--patchtst-layers",
        "2",
        "--patchtst-d-ff",
        "512",
        "--patch-len",
        str(PATCH_LEN),
        "--stride",
        str(SUPERVISED_STRIDE),
    ]


def _evaluation_command(
    *,
    python: Path,
    target_source: Path,
    case_root: Path,
    training_log: Path,
    batch_size: int,
    num_workers: int,
) -> list[str]:
    artifact_dir = case_root / "endo_only"
    return [
        str(python),
        str(EVALUATOR),
        "--artifact-dir",
        str(artifact_dir),
        "--training-log",
        str(training_log),
        "--target-source",
        str(target_source),
        "--output-dir",
        str(artifact_dir / "qualification_evaluation"),
        "--device",
        "cuda",
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
    ]


def _read_qualification_metric(case_root: Path) -> dict[str, Any]:
    path = (
        case_root
        / "endo_only/qualification_evaluation/qualification_metrics.csv"
    )
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    row = next(item for item in rows if item["model_key"] == MODEL_KEY)
    numeric_fields = (
        "mae",
        "wape",
        "wape_pct",
        "smape",
        "smape_pct",
        "load_seconds",
        "inference_seconds",
        "series_per_second",
    )
    result: dict[str, Any] = dict(row)
    for field in numeric_fields:
        result[field] = float(row[field])
    for field in ("series_count", "observation_count", "parameter_count"):
        result[field] = int(row[field])
    return result


def _read_supervised_selection(case_root: Path) -> dict[str, Any]:
    path = (
        case_root
        / "endo_only/qualification_evaluation/production_refit_epochs.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON array: {path}")
    policy = next(
        item
        for item in value
        if item.get("model_key") == MODEL_KEY
    )
    return {
        "best_epoch": int(policy["qualification_best_epoch"]),
        "best_validation_loss": float(
            policy["qualification_best_validation_loss"]
        ),
        "qualification_total_epochs": int(
            policy["qualification_total_epochs"]
        ),
        "selection_basis": policy["selection_basis"],
    }


def _read_pretrain_metadata(case_root: Path) -> dict[str, Any] | None:
    path = case_root / "endo_only/pretrain/patchtst_pretrain_best.pt"
    if not path.is_file():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "checkpoint_path": str(path),
        "checkpoint_sha256": _sha256_file(path),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_validation_loss": checkpoint.get("best_val"),
        "validation_mask_seed": checkpoint.get("validation_mask_seed"),
        "pretrain_contract": checkpoint.get("pretrain_contract"),
        "history": checkpoint.get("history", []),
    }


def _case_conditions(
    *,
    mode: str,
    pretrain_epochs: int,
    pretrain_stride: int,
    mask_ratio: float,
    supervised_epochs: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    return {
        "lookback": 52,
        "horizon": 27,
        "train_end_week": 202544,
        "forecast_origin": 202545,
        "validation_origin": 202518,
        "window_stride": 4,
        "supervised_epochs": supervised_epochs,
        "pretrain_epochs": pretrain_epochs if mode == "full" else 0,
        "patch_len": PATCH_LEN,
        "supervised_stride": SUPERVISED_STRIDE,
        "pretrain_stride": (
            pretrain_stride if mode == "full" else None
        ),
        "mask_ratio": mask_ratio if mode == "full" else None,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "patchtst_capacity": {
            "d_model": 128,
            "n_layers": 2,
            "d_ff": 512,
        },
    }


def run_case(
    *,
    phase_root: Path,
    python: Path,
    target_source: Path,
    mode: str,
    seed: int,
    pretrain_epochs: int,
    pretrain_stride: int,
    mask_ratio: float,
    supervised_epochs: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    poll_seconds: float,
    resume: bool,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"Unsupported mode: {mode}")
    expected_conditions = _case_conditions(
        mode=mode,
        pretrain_epochs=pretrain_epochs,
        pretrain_stride=pretrain_stride,
        mask_ratio=mask_ratio,
        supervised_epochs=supervised_epochs,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    case_root = phase_root / mode / f"seed_{seed}"
    record_path = case_root / "benchmark_runtime.json"
    if resume and record_path.is_file():
        record = _read_json(record_path)
        if record.get("status") == "complete":
            if record.get("conditions") != expected_conditions:
                raise RuntimeError(
                    "Completed benchmark case does not match the requested "
                    "SSL patching contract. Use a new artifact root or rerun "
                    "with --no-resume."
                )
            if "supervised_selection" not in record:
                record["supervised_selection"] = (
                    _read_supervised_selection(case_root)
                )
                record_path.write_text(
                    json.dumps(record, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            print(f"[benchmark] reuse complete case: {mode} seed={seed}")
            return record

    case_root.mkdir(parents=True, exist_ok=True)
    training_log = case_root / "training.log"
    print(f"[benchmark] train mode={mode} seed={seed}", flush=True)
    training = run_measured_process(
        _runner_command(
            python=python,
            target_source=target_source,
            case_root=case_root,
            mode=mode,
            seed=seed,
            pretrain_epochs=pretrain_epochs,
            pretrain_stride=pretrain_stride,
            mask_ratio=mask_ratio,
            supervised_epochs=supervised_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        ),
        log_path=training_log,
        poll_seconds=poll_seconds,
    )

    evaluation_log = case_root / "evaluation.log"
    print(f"[benchmark] evaluate mode={mode} seed={seed}", flush=True)
    evaluation = run_measured_process(
        _evaluation_command(
            python=python,
            target_source=target_source,
            case_root=case_root,
            training_log=training_log,
            batch_size=batch_size,
            num_workers=max(0, min(num_workers, 4)),
        ),
        log_path=evaluation_log,
        poll_seconds=poll_seconds,
    )

    checkpoint_path = (
        case_root / "endo_only/weekly_PatchTST_L52_H27.pt"
    )
    data_manifest = _read_json(case_root / "endo_only/data_manifest.json")
    record = {
        "schema_version": 1,
        "status": "complete",
        "mode": mode,
        "seed": seed,
        "conditions": expected_conditions,
        "training": training,
        "evaluation": evaluation,
        "metrics": _read_qualification_metric(case_root),
        "supervised_selection": _read_supervised_selection(case_root),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256_file(checkpoint_path),
        },
        "pretrain": _read_pretrain_metadata(case_root),
        "data_manifest": data_manifest,
    }
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return record


def _contiguous_ranges(values: Iterable[int]) -> list[list[int]]:
    sorted_values = sorted(set(values))
    if not sorted_values:
        return []
    ranges: list[list[int]] = []
    start = previous = sorted_values[0]
    for value in sorted_values[1:]:
        if value != previous + 1:
            ranges.append([start, previous])
            start = value
        previous = value
    ranges.append([start, previous])
    return ranges


def calculate_overlap_exposure(
    *,
    patch_len: int,
    stride: int,
    patch_count: int,
    mask_ratio: float,
) -> dict[str, Any]:
    """Calculate exact masked-value exposure from overlapping unmasked patches."""
    if patch_len <= 0 or stride <= 0 or patch_count <= 0:
        raise ValueError("patch_len, stride and patch_count must be positive.")
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be between zero and one.")

    starts = [index * stride for index in range(patch_count)]
    weighted_exposure = 0.0
    weighted_full_exposure = 0.0
    weighted_masked_patch_count = 0.0
    for pattern in range(1 << patch_count):
        masked = [
            bool(pattern & (1 << index))
            for index in range(patch_count)
        ]
        masked_count = sum(masked)
        probability = (
            mask_ratio ** masked_count
            * (1.0 - mask_ratio) ** (patch_count - masked_count)
        )
        if masked_count == 0:
            continue

        for patch_index, is_masked in enumerate(masked):
            if not is_masked:
                continue
            patch_start = starts[patch_index]
            exposed_values = 0
            for time_index in range(patch_start, patch_start + patch_len):
                exposed = any(
                    not masked[other_index]
                    and starts[other_index] <= time_index
                    < starts[other_index] + patch_len
                    for other_index in range(patch_count)
                )
                exposed_values += int(exposed)
            exposure_fraction = exposed_values / patch_len
            weighted_exposure += probability * exposure_fraction
            weighted_full_exposure += probability * int(
                exposure_fraction == 1.0
            )
            weighted_masked_patch_count += probability

    return {
        "patch_len": patch_len,
        "stride": stride,
        "patch_count": patch_count,
        "mask_ratio": mask_ratio,
        "expected_exposed_value_fraction_per_masked_patch": (
            weighted_exposure / weighted_masked_patch_count
        ),
        "fully_exposed_masked_patch_fraction": (
            weighted_full_exposure / weighted_masked_patch_count
        ),
        "calculation": (
            "exact enumeration of independent Bernoulli patch masks"
        ),
    }


def analyze_pretrain_history(
    history: Sequence[Mapping[str, Any]],
    *,
    tolerance_fraction: float = 0.01,
    rolling_window: int = 3,
) -> dict[str, Any]:
    rows = [
        {
            "epoch": int(row["global_epoch"]),
            "train_loss": float(row["train_loss"]),
            "validation_loss": float(row["validation_loss"]),
        }
        for row in history
        if row.get("validation_loss") is not None
    ]
    if not rows:
        raise ValueError("Pretrain history has no validation loss.")
    if rolling_window <= 0 or rolling_window > len(rows):
        raise ValueError("rolling_window must fit within pretrain history.")

    best = min(rows, key=lambda row: (row["validation_loss"], row["epoch"]))
    threshold = best["validation_loss"] * (1.0 + tolerance_fraction)
    near_best_epochs = [
        row["epoch"]
        for row in rows
        if row["validation_loss"] <= threshold
    ]
    rolling = []
    for offset in range(len(rows) - rolling_window + 1):
        window_rows = rows[offset:offset + rolling_window]
        rolling.append(
            {
                "start_epoch": window_rows[0]["epoch"],
                "end_epoch": window_rows[-1]["epoch"],
                "mean_validation_loss": statistics.fmean(
                    row["validation_loss"] for row in window_rows
                ),
            }
        )
    best_rolling = min(
        rolling,
        key=lambda row: (
            row["mean_validation_loss"],
            row["start_epoch"],
        ),
    )
    return {
        "epoch_count": len(rows),
        "best_epoch": best["epoch"],
        "best_validation_loss": best["validation_loss"],
        "near_best_tolerance_fraction": tolerance_fraction,
        "near_best_threshold": threshold,
        "near_best_epoch_ranges": _contiguous_ranges(near_best_epochs),
        "rolling_window": rolling_window,
        "best_rolling_window": best_rolling,
        "history": rows,
    }


def _nested_float(record: Mapping[str, Any], path: Sequence[str]) -> float:
    value: Any = record
    for key in path:
        value = value[key]
    return float(value)


def build_comparison_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_mode = {
        mode: [record for record in records if record["mode"] == mode]
        for mode in MODES
    }
    for mode, mode_records in by_mode.items():
        if not mode_records:
            raise ValueError(f"No records for mode {mode}.")

    fields = {
        "mae": ("metrics", "mae"),
        "wape_pct": ("metrics", "wape_pct"),
        "smape_pct": ("metrics", "smape_pct"),
        "training_seconds": ("training", "elapsed_seconds"),
        "training_peak_delta_mib": (
            "training",
            "gpu_memory",
            "peak_delta_mib",
        ),
        "inference_seconds": ("metrics", "inference_seconds"),
        "evaluation_peak_delta_mib": (
            "evaluation",
            "gpu_memory",
            "peak_delta_mib",
        ),
    }
    mode_summary: dict[str, Any] = {}
    for mode, mode_records in by_mode.items():
        metrics = {}
        for field, path in fields.items():
            values = [_nested_float(record, path) for record in mode_records]
            metrics[field] = {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "population_std": statistics.pstdev(values),
                "values": values,
            }
        mode_summary[mode] = {
            "seeds": [int(record["seed"]) for record in mode_records],
            "supervised_best_epochs": [
                int(record["supervised_selection"]["best_epoch"])
                for record in mode_records
            ],
            "supervised_best_validation_losses": [
                float(
                    record["supervised_selection"][
                        "best_validation_loss"
                    ]
                )
                for record in mode_records
            ],
            "pretrain_best_epochs": [
                int(record["pretrain"]["best_epoch"])
                for record in mode_records
                if record.get("pretrain") is not None
            ],
            "metrics": metrics,
        }

    indexed = {
        (str(record["mode"]), int(record["seed"])): record
        for record in records
    }
    shared_seeds = sorted(
        set(mode_summary["sl_only"]["seeds"])
        & set(mode_summary["full"]["seeds"])
    )
    paired = []
    for seed in shared_seeds:
        baseline = indexed[("sl_only", seed)]
        candidate = indexed[("full", seed)]
        deltas = {}
        for field, path in fields.items():
            baseline_value = _nested_float(baseline, path)
            candidate_value = _nested_float(candidate, path)
            deltas[field] = {
                "sl_only": baseline_value,
                "full": candidate_value,
                "full_minus_sl_only": candidate_value - baseline_value,
                "full_vs_sl_only_pct": (
                    (candidate_value / baseline_value - 1.0) * 100.0
                    if baseline_value != 0.0
                    else None
                ),
            }
        paired.append({"seed": seed, "deltas": deltas})

    accuracy_fields = ("mae", "wape_pct", "smape_pct")
    wins = {
        field: sum(
            pair["deltas"][field]["full"] < pair["deltas"][field]["sl_only"]
            for pair in paired
        )
        for field in accuracy_fields
    }
    return {
        "schema_version": 1,
        "modes": mode_summary,
        "paired_by_seed": paired,
        "full_accuracy_wins": wins,
        "paired_seed_count": len(shared_seeds),
    }


def _write_case_csv(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    fieldnames = [
        "mode",
        "seed",
        "pretrain_epochs",
        "supervised_epochs",
        "pretrain_best_epoch",
        "supervised_best_epoch",
        "supervised_best_validation_loss",
        "mae",
        "wape_pct",
        "smape_pct",
        "training_seconds",
        "training_peak_delta_mib",
        "inference_seconds",
        "evaluation_peak_delta_mib",
        "checkpoint_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "mode": record["mode"],
                    "seed": record["seed"],
                    "pretrain_epochs": record["conditions"][
                        "pretrain_epochs"
                    ],
                    "supervised_epochs": record["conditions"][
                        "supervised_epochs"
                    ],
                    "pretrain_best_epoch": (
                        record["pretrain"]["best_epoch"]
                        if record.get("pretrain") is not None
                        else ""
                    ),
                    "supervised_best_epoch": record[
                        "supervised_selection"
                    ]["best_epoch"],
                    "supervised_best_validation_loss": record[
                        "supervised_selection"
                    ]["best_validation_loss"],
                    "mae": record["metrics"]["mae"],
                    "wape_pct": record["metrics"]["wape_pct"],
                    "smape_pct": record["metrics"]["smape_pct"],
                    "training_seconds": record["training"][
                        "elapsed_seconds"
                    ],
                    "training_peak_delta_mib": record["training"][
                        "gpu_memory"
                    ]["peak_delta_mib"],
                    "inference_seconds": record["metrics"][
                        "inference_seconds"
                    ],
                    "evaluation_peak_delta_mib": record["evaluation"][
                        "gpu_memory"
                    ]["peak_delta_mib"],
                    "checkpoint_sha256": record["checkpoint"]["sha256"],
                }
            )


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--target-source", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--pretrain-epochs", type=int, required=True)
    parser.add_argument("--pretrain-stride", type=int, default=13)
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument("--supervised-epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pilot = subparsers.add_parser("pilot")
    _add_common_arguments(pilot)
    pilot.add_argument("--seed", type=int, default=42)
    pilot.add_argument("--near-best-tolerance", type=float, default=0.01)
    pilot.add_argument("--rolling-window", type=int, default=3)

    compare = subparsers.add_parser("compare")
    _add_common_arguments(compare)
    compare.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    if not args.target_source.is_file():
        raise FileNotFoundError(args.target_source)
    if args.pretrain_epochs <= 0:
        raise ValueError("--pretrain-epochs must be positive.")
    if args.pretrain_stride <= 0:
        raise ValueError("--pretrain-stride must be positive.")
    if not 0.0 < args.mask_ratio <= 1.0:
        raise ValueError("--mask-ratio must be in (0, 1].")
    if args.supervised_epochs <= 0:
        raise ValueError("--supervised-epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be positive.")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive.")


def _common_case_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        # Preserve a venv's python symlink so sys.prefix and site-packages stay
        # bound to the approved experiment environment.
        "python": args.python.absolute(),
        "target_source": args.target_source.resolve(),
        "pretrain_epochs": args.pretrain_epochs,
        "pretrain_stride": args.pretrain_stride,
        "mask_ratio": args.mask_ratio,
        "supervised_epochs": args.supervised_epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "poll_seconds": args.poll_seconds,
        "resume": args.resume,
    }


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    phase_root = args.artifact_root.resolve() / "pilot"
    record = run_case(
        phase_root=phase_root,
        mode="full",
        seed=args.seed,
        **_common_case_kwargs(args),
    )
    pretrain = record.get("pretrain")
    if not isinstance(pretrain, Mapping):
        raise RuntimeError("Pilot full run did not create pretrain metadata.")
    analysis = analyze_pretrain_history(
        pretrain["history"],
        tolerance_fraction=args.near_best_tolerance,
        rolling_window=args.rolling_window,
    )
    payload = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "gpu": _gpu_identity(),
        "case": {
            "mode": record["mode"],
            "seed": record["seed"],
            "conditions": record["conditions"],
        },
        "pretrain_checkpoint": {
            key: value
            for key, value in pretrain.items()
            if key != "history"
        },
        "overlap_exposure_diagnostic": calculate_overlap_exposure(
            patch_len=PATCH_LEN,
            stride=args.pretrain_stride,
            patch_count=((52 - PATCH_LEN) // args.pretrain_stride) + 1,
            mask_ratio=args.mask_ratio,
        ),
        "analysis": analysis,
    }
    output_path = phase_root / "pilot_pretrain_analysis.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[benchmark] pilot analysis: {output_path}", flush=True)
    return payload


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    phase_root = args.artifact_root.resolve() / "comparison"
    records = []
    for seed_index, seed in enumerate(args.seeds):
        order = MODES if seed_index % 2 == 0 else tuple(reversed(MODES))
        for mode in order:
            records.append(
                run_case(
                    phase_root=phase_root,
                    mode=mode,
                    seed=seed,
                    **_common_case_kwargs(args),
                )
            )

    summary = build_comparison_summary(records)
    summary.update(
        {
            "generated_at": _utc_now(),
            "gpu": _gpu_identity(),
            "experiment": {
                "seeds": list(args.seeds),
                "pretrain_epochs": args.pretrain_epochs,
                "pretrain_stride": args.pretrain_stride,
                "mask_ratio": args.mask_ratio,
                "supervised_epochs": args.supervised_epochs,
                "execution_order": (
                    "alternating sl_only/full by seed index"
                ),
                "target_source": str(args.target_source.resolve()),
                "target_source_sha256": _sha256_file(
                    args.target_source.resolve()
                ),
            },
        }
    )
    json_path = phase_root / "comparison_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_case_csv(records, phase_root / "comparison_cases.csv")
    print(f"[benchmark] comparison summary: {json_path}", flush=True)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    if args.command == "pilot":
        run_pilot(args)
    else:
        run_comparison(args)


if __name__ == "__main__":
    main()

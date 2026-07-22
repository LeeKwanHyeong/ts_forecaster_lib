#!/usr/bin/env python3
"""Run a controlled single-seed PatchMixer accuracy comparison on an RTX 5090."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import nullcontext, redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modeling_module.data_loader.multi_part_exo_dataset import (
    MultiPartExoTrainingDataset,
)
from modeling_module.models.PatchMixer.PatchMixer import PatchMixerModel
from modeling_module.models.PatchMixer.common.configs import PatchMixerConfig
from modeling_module.models.PatchMixer import (
    PatchMixerOriginalConfig,
    PatchMixerOriginalModel,
)
from modeling_module.models.PatchMixer.provenance import (
    PATCHMIXER_ENHANCED_BASELINE_COMMIT,
    PATCHMIXER_REFERENCE_CONFIG,
    PATCHMIXER_REFERENCE_PARAMETER_COUNTS,
    PATCHMIXER_UPSTREAM_COMMIT,
)


MODEL_NAMES = ("original", "enhanced")
SEASONAL_PERIOD = 52


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _ratio(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("ratio must be in (0, 1)")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=_positive_int, default=100)
    parser.add_argument("--patience", type=_positive_int, default=15)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=30.0)
    parser.add_argument("--val-ratio", type=_ratio, default=0.15)
    parser.add_argument("--test-ratio", type=_ratio, default=0.15)
    parser.add_argument("--precision", choices=("float32", "bf16"), default="float32")
    parser.add_argument(
        "--expected-device",
        default="NVIDIA GeForce RTX 5090",
        help="Fail on an unexpected GPU; pass an empty string to disable.",
    )
    return parser


def _run_text(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command,
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _git_metadata() -> dict[str, Any]:
    status = _run_text(["git", "status", "--porcelain"])
    return {
        "branch": _run_text(["git", "branch", "--show-current"]),
        "commit": _run_text(["git", "rev-parse", "HEAD"]),
        "working_tree_dirty": bool(status),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value.astype("<f4", copy=False))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _configure_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _seed_everything(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")


def _validate_cuda(expected_device: str) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this comparison must not run on CPU or MPS.")
    device_name = torch.cuda.get_device_name(0)
    if expected_device and device_name != expected_device:
        raise RuntimeError(
            f"Expected device {expected_device!r}, but CUDA device 0 is {device_name!r}."
        )
    return device_name


def _split_ids(
    ids: Iterable[str],
    *,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[str]]:
    ordered = np.asarray(sorted(str(uid) for uid in ids), dtype=object)
    if len(ordered) < 3:
        raise ValueError("At least three series are required for train/validation/test splits.")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1.")

    rng = np.random.default_rng(seed)
    rng.shuffle(ordered)
    n_test = max(1, int(math.floor(len(ordered) * test_ratio + 0.5)))
    n_val = max(1, int(math.floor(len(ordered) * val_ratio + 0.5)))
    n_train = len(ordered) - n_val - n_test
    if n_train <= 0:
        raise ValueError("Split ratios leave no training series.")

    return {
        "train": sorted(str(uid) for uid in ordered[n_test + n_val :]),
        "validation": sorted(str(uid) for uid in ordered[n_test : n_test + n_val]),
        "test": sorted(str(uid) for uid in ordered[:n_test]),
    }


def _indices_for_ids(
    dataset: MultiPartExoTrainingDataset,
    ids: Iterable[str],
) -> list[int]:
    indices: list[int] = []
    for uid in ids:
        indices.extend(dataset.id_to_indices[str(uid)])
    return indices


def _last_indices_for_ids(
    dataset: MultiPartExoTrainingDataset,
    ids: Iterable[str],
) -> list[int]:
    return [dataset.id_to_indices[str(uid)][-1] for uid in ids]


def _make_loader(
    dataset: MultiPartExoTrainingDataset,
    indices: list[int],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )


def _build_model(name: str) -> torch.nn.Module:
    config_values = dict(PATCHMIXER_REFERENCE_CONFIG)
    if name == "original":
        return PatchMixerOriginalModel(
            PatchMixerOriginalConfig.from_config(config_values)
        )
    if name == "enhanced":
        config = PatchMixerConfig(**config_values)
        with redirect_stdout(StringIO()):
            return PatchMixerModel(config)
    raise ValueError(f"Unsupported model: {name}")


def _point_prediction(output: torch.Tensor, *, horizon: int) -> torch.Tensor:
    if output.ndim == 3 and output.shape[-1] == 1:
        output = output.squeeze(-1)
    if output.ndim != 2 or output.shape[1] != horizon:
        raise RuntimeError(
            f"Expected [B,{horizon}] or [B,{horizon},1], got {tuple(output.shape)}."
        )
    return output


def _autocast_factory(precision: str) -> Callable[[], Any]:
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected CUDA device does not support BF16.")
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext


@torch.no_grad()
def _mse_on_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    horizon: int,
    autocast_context: Callable[[], Any],
) -> float:
    model.eval()
    squared_error = 0.0
    count = 0
    for batch in loader:
        inputs = batch[0].cuda(non_blocking=True)
        targets = batch[1].cuda(non_blocking=True)
        with autocast_context():
            predictions = _point_prediction(model(inputs), horizon=horizon)
        squared_error += float(
            F.mse_loss(predictions.float(), targets.float(), reduction="sum")
        )
        count += targets.numel()
    return squared_error / max(1, count)


def _train_model(
    name: str,
    *,
    dataset: MultiPartExoTrainingDataset,
    train_indices: list[int],
    validation_indices: list[int],
    horizon: int,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    precision: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    _seed_everything(seed)
    model = _build_model(name).cuda().train()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    expected_count = dict(PATCHMIXER_REFERENCE_PARAMETER_COUNTS)[name]
    if parameter_count != expected_count:
        raise RuntimeError(
            f"{name} parameter-count drift: got {parameter_count:,}, "
            f"expected {expected_count:,}."
        )

    # Keep training randomness independent from differing initialization graphs.
    _seed_everything(seed + 1)
    train_loader = _make_loader(
        dataset,
        train_indices,
        batch_size=batch_size,
        shuffle=True,
        seed=seed + 2,
    )
    validation_loader = _make_loader(
        dataset,
        validation_indices,
        batch_size=batch_size,
        shuffle=False,
        seed=seed + 2,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.01,
    )
    autocast_context = _autocast_factory(precision)

    best_validation_mse = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_squared_error = 0.0
        train_count = 0
        for batch in train_loader:
            inputs = batch[0].cuda(non_blocking=True)
            targets = batch[1].cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                predictions = _point_prediction(model(inputs), horizon=horizon)
                loss = F.mse_loss(predictions, targets)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(
                    f"{name} produced a non-finite gradient norm at epoch {epoch}."
                )
            optimizer.step()
            train_squared_error += float(
                F.mse_loss(
                    predictions.detach().float(),
                    targets.float(),
                    reduction="sum",
                )
            )
            train_count += targets.numel()

        train_mse = train_squared_error / max(1, train_count)
        validation_mse = _mse_on_loader(
            model,
            validation_loader,
            horizon=horizon,
            autocast_context=autocast_context,
        )
        current_lr = float(optimizer.param_groups[0]["lr"])
        improved = validation_mse < best_validation_mse
        if improved:
            best_validation_mse = validation_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_mse": train_mse,
                "validation_mse": validation_mse,
            }
        )
        if epoch == 1 or epoch % 10 == 0 or improved:
            print(
                "ACCURACY_PROGRESS="
                + json.dumps(
                    {
                        "model": name,
                        "epoch": epoch,
                        "train_mse": train_mse,
                        "validation_mse": validation_mse,
                        "best_epoch": best_epoch,
                        "improved": improved,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        scheduler.step()
        if epochs_without_improvement >= patience:
            break

    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()
    training_result = {
        "parameters": parameter_count,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation_mse,
        "best_validation_rmse": math.sqrt(best_validation_mse),
        "elapsed_seconds": elapsed_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "history": history,
    }
    del optimizer, scheduler, train_loader, validation_loader, best_state
    return model, training_result


@torch.no_grad()
def _predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    lookback: int,
    horizon: int,
    autocast_context: Callable[[], Any],
) -> dict[str, Any]:
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    last_value_predictions: list[np.ndarray] = []
    seasonal_predictions: list[np.ndarray] = []
    uids: list[str] = []
    seasonal_start = lookback - SEASONAL_PERIOD
    if seasonal_start < 0 or seasonal_start + horizon > lookback:
        raise ValueError(
            "lookback must contain every seasonal-naive source required by the horizon."
        )

    for batch in loader:
        inputs = batch[0].cuda(non_blocking=True)
        batch_targets = batch[1].cuda(non_blocking=True)
        with autocast_context():
            batch_predictions = _point_prediction(model(inputs), horizon=horizon)

        last_values = inputs[:, -1, 0].unsqueeze(1).expand(-1, horizon)
        seasonal_values = inputs[
            :, seasonal_start : seasonal_start + horizon, 0
        ]
        targets.append(batch_targets.float().cpu().numpy())
        predictions.append(batch_predictions.float().cpu().numpy())
        last_value_predictions.append(last_values.float().cpu().numpy())
        seasonal_predictions.append(seasonal_values.float().cpu().numpy())
        uids.extend(str(uid) for uid in batch[2])

    return {
        "targets": np.concatenate(targets, axis=0).astype(np.float64),
        "predictions": np.concatenate(predictions, axis=0).astype(np.float64),
        "last_value": np.concatenate(last_value_predictions, axis=0).astype(np.float64),
        "seasonal_naive_52": np.concatenate(seasonal_predictions, axis=0).astype(np.float64),
        "uids": np.asarray(uids, dtype=object),
    }


def _metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    errors = predictions - targets
    absolute_errors = np.abs(errors)
    epsilon = 1e-8
    return {
        "mae": float(np.mean(absolute_errors)),
        "mse": float(np.mean(np.square(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "smape": float(
            np.mean(
                2.0
                * absolute_errors
                / (np.abs(targets) + np.abs(predictions) + epsilon)
            )
        ),
        "wape": float(np.sum(absolute_errors) / (np.sum(np.abs(targets)) + epsilon)),
        "bias_ratio": float(np.sum(errors) / (np.sum(np.abs(targets)) + epsilon)),
    }


def _metric_bundle(
    targets: np.ndarray,
    predictions: np.ndarray,
    uids: np.ndarray,
) -> dict[str, Any]:
    per_series = {
        str(uid): _metrics(targets[uids == uid], predictions[uids == uid])
        for uid in sorted(set(str(value) for value in uids))
    }
    metric_names = tuple(next(iter(per_series.values())).keys())
    macro = {
        name: float(np.mean([metrics[name] for metrics in per_series.values()]))
        for name in metric_names
    }
    return {
        "micro": _metrics(targets, predictions),
        "macro_series": macro,
        "per_series": per_series,
    }


def _summarize_predictions(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "windows": int(payload["targets"].shape[0]),
        "forecast_points": int(payload["targets"].size),
        "prediction_sha256": _array_sha256(payload["predictions"]),
        "metrics": _metric_bundle(
            payload["targets"],
            payload["predictions"],
            payload["uids"],
        ),
    }


def _summarize_baselines(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "last_value": _metric_bundle(
            payload["targets"], payload["last_value"], payload["uids"]
        ),
        "seasonal_naive_52": _metric_bundle(
            payload["targets"], payload["seasonal_naive_52"], payload["uids"]
        ),
    }


def _paired_comparison(
    original: dict[str, Any],
    enhanced: dict[str, Any],
) -> dict[str, Any]:
    if not np.array_equal(original["targets"], enhanced["targets"]):
        raise RuntimeError("Model evaluations do not share identical targets.")
    if not np.array_equal(original["uids"], enhanced["uids"]):
        raise RuntimeError("Model evaluations do not share identical series ordering.")

    targets = original["targets"]
    original_predictions = original["predictions"]
    enhanced_predictions = enhanced["predictions"]
    original_errors = np.abs(original_predictions - targets)
    enhanced_errors = np.abs(enhanced_predictions - targets)
    original_metrics = _metrics(targets, original_predictions)
    enhanced_metrics = _metrics(targets, enhanced_predictions)
    lower_is_better = ("mae", "mse", "rmse", "smape", "wape")

    relative_improvement = {
        name: float(
            100.0
            * (enhanced_metrics[name] - original_metrics[name])
            / max(abs(enhanced_metrics[name]), 1e-12)
        )
        for name in lower_is_better
    }
    series_mae_winner: dict[str, str] = {}
    for uid in sorted(set(str(value) for value in original["uids"])):
        mask = original["uids"] == uid
        original_mae = float(np.mean(original_errors[mask]))
        enhanced_mae = float(np.mean(enhanced_errors[mask]))
        if original_mae < enhanced_mae:
            series_mae_winner[uid] = "original"
        elif enhanced_mae < original_mae:
            series_mae_winner[uid] = "enhanced"
        else:
            series_mae_winner[uid] = "tie"

    return {
        "original_relative_improvement_pct": relative_improvement,
        "pointwise_absolute_error_win_rate": {
            "original": float(np.mean(original_errors < enhanced_errors)),
            "enhanced": float(np.mean(enhanced_errors < original_errors)),
            "tie": float(np.mean(original_errors == enhanced_errors)),
        },
        "series_mae_winner": series_mae_winner,
        "overall_mae_winner": (
            "original"
            if original_metrics["mae"] < enhanced_metrics["mae"]
            else "enhanced"
            if enhanced_metrics["mae"] < original_metrics["mae"]
            else "tie"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    experiment_started_at = datetime.now(timezone.utc)
    experiment_started = time.perf_counter()
    data_path = args.data.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    _configure_determinism(args.seed)
    device_name = _validate_cuda(args.expected_device)
    reference = dict(PATCHMIXER_REFERENCE_CONFIG)
    lookback = int(reference["lookback"])
    horizon = int(reference["horizon"])

    frame = pl.read_parquet(data_path).select(["unique_id", "date", "y"])
    dataset = MultiPartExoTrainingDataset(
        frame,
        lookback,
        horizon,
        "weekly",
        id_col="unique_id",
        date_col="date",
        qty_col="y",
    )
    splits = _split_ids(
        dataset.id_to_indices.keys(),
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    train_indices = _indices_for_ids(dataset, splits["train"])
    validation_indices = _indices_for_ids(dataset, splits["validation"])
    test_indices = _indices_for_ids(dataset, splits["test"])
    last_test_indices = _last_indices_for_ids(dataset, splits["test"])

    split_fingerprint = hashlib.sha256(
        json.dumps(splits, sort_keys=True).encode("utf-8")
    ).hexdigest()
    protocol = {
        "seed": args.seed,
        "initialization_seed": args.seed,
        "training_randomness_seed": args.seed + 1,
        "dataloader_seed": args.seed + 2,
        "split": "series_id_disjoint",
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "split_fingerprint": split_fingerprint,
        "reference_config": reference,
        "endogenous_only": True,
        "precision": args.precision,
        "loss": "mse",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "eta_min": args.lr * 0.01,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "deterministic_algorithms": True,
        "model_selection": "lowest validation MSE",
        "test_evaluations": ["all_rolling_windows", "last_origin_per_series"],
    }
    dataset_metadata = {
        "path": str(data_path),
        "sha256": _file_sha256(data_path),
        "rows": frame.height,
        "series": len(dataset.id_to_indices),
        "total_windows": len(dataset),
        "splits": splits,
        "split_series_counts": {name: len(ids) for name, ids in splits.items()},
        "split_window_counts": {
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test_all": len(test_indices),
            "test_last_origin": len(last_test_indices),
        },
    }

    model_results: dict[str, Any] = {}
    all_predictions: dict[str, dict[str, Any]] = {}
    last_predictions: dict[str, dict[str, Any]] = {}
    autocast_context = _autocast_factory(args.precision)
    for name in MODEL_NAMES:
        model, training_result = _train_model(
            name,
            dataset=dataset,
            train_indices=train_indices,
            validation_indices=validation_indices,
            horizon=horizon,
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            precision=args.precision,
        )
        all_loader = _make_loader(
            dataset,
            test_indices,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed + 2,
        )
        last_loader = _make_loader(
            dataset,
            last_test_indices,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed + 2,
        )
        all_payload = _predict_loader(
            model,
            all_loader,
            lookback=lookback,
            horizon=horizon,
            autocast_context=autocast_context,
        )
        last_payload = _predict_loader(
            model,
            last_loader,
            lookback=lookback,
            horizon=horizon,
            autocast_context=autocast_context,
        )
        model_results[name] = {
            "training": training_result,
            "test_all_rolling_windows": _summarize_predictions(all_payload),
            "test_last_origin_per_series": _summarize_predictions(last_payload),
        }
        all_predictions[name] = all_payload
        last_predictions[name] = last_payload
        del model, all_loader, last_loader
        gc.collect()
        torch.cuda.empty_cache()

    experiment_finished_at = datetime.now(timezone.utc)
    result = {
        "schema_version": 1,
        "started_at_utc": experiment_started_at.isoformat(),
        "finished_at_utc": experiment_finished_at.isoformat(),
        "elapsed_seconds": time.perf_counter() - experiment_started,
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": device_name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "source": {
            "git": _git_metadata(),
            "original_upstream_commit": PATCHMIXER_UPSTREAM_COMMIT,
            "enhanced_baseline_commit": PATCHMIXER_ENHANCED_BASELINE_COMMIT,
        },
        "protocol": protocol,
        "dataset": dataset_metadata,
        "baselines": {
            "test_all_rolling_windows": _summarize_baselines(
                all_predictions["original"]
            ),
            "test_last_origin_per_series": _summarize_baselines(
                last_predictions["original"]
            ),
        },
        "models": model_results,
        "paired_comparison": {
            "test_all_rolling_windows": _paired_comparison(
                all_predictions["original"], all_predictions["enhanced"]
            ),
            "test_last_origin_per_series": _paired_comparison(
                last_predictions["original"], last_predictions["enhanced"]
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("ACCURACY_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

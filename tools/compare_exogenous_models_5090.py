#!/usr/bin/env python3
"""Compare endogenous and exogenous Patch models on an RTX 5090.

Accuracy uses deterministic, series-disjoint Walmart splits. Performance uses
fixed CUDA-resident synthetic batches so data loading and host transfer are not
part of the measured training-step latency.
"""

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
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
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

from modeling_module.data_loader.multi_part_exo_dataset import (  # noqa: E402
    MultiPartExoTrainingDataset,
)
from modeling_module.models.PatchMixer.common.configs import (  # noqa: E402
    PatchMixerConfig,
)
from modeling_module.models.PatchTST.common.configs import (  # noqa: E402
    AttentionConfig,
    PatchTSTConfig,
)
from modeling_module.models.model_builder import (  # noqa: E402
    build_patch_mixer,
    build_patch_mixer_exogenous,
    build_patchTST,
    build_patchTST_exogenous,
)


LOOKBACK = 54
HORIZON = 27
SEASONAL_PERIOD = 52
PAST_EXOGENOUS_COLUMNS = (
    "exo_p_y_lag_1w",
    "exo_p_y_lag_2w",
    "exo_p_y_lag_52w",
    "exo_p_y_rollmean_4w",
    "exo_p_y_rollmean_12w",
    "exo_p_y_rollstd_4w",
    "exo_p_temperature",
    "exo_p_fuel_price",
    "exo_p_cpi",
    "exo_p_unemployment",
    "exo_p_markdown_sum",
)
FUTURE_EXOGENOUS_COLUMNS = (
    "exo_is_holiday",
    "exo_temperature",
    "exo_fuel_price",
    "exo_cpi",
    "exo_unemployment",
    "exo_markdown_sum",
)


@dataclass(frozen=True)
class ModelCase:
    key: str
    family: str
    exogenous: bool


MODEL_CASES = (
    ModelCase("patchtst_endogenous", "patchtst", False),
    ModelCase("patchtst_exogenous", "patchtst", True),
    ModelCase("patchmixer_endogenous", "patchmixer", False),
    ModelCase("patchmixer_exogenous", "patchmixer", True),
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _ratio(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("ratio must be in (0, 1)")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--epochs", type=_positive_int, default=30)
    parser.add_argument("--patience", type=_positive_int, default=8)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=30.0)
    parser.add_argument("--val-ratio", type=_ratio, default=0.15)
    parser.add_argument("--test-ratio", type=_ratio, default=0.15)
    parser.add_argument(
        "--accuracy-precision",
        choices=("float32", "bf16"),
        default="float32",
    )
    parser.add_argument(
        "--performance-precision",
        choices=("float32", "bf16"),
        default="bf16",
    )
    parser.add_argument("--performance-steps", type=_positive_int, default=100)
    parser.add_argument("--warmup-steps", type=_nonnegative_int, default=20)
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
    return {
        "branch": _run_text(["git", "branch", "--show-current"]),
        "commit": _run_text(["git", "rev-parse", "HEAD"]),
        "working_tree_dirty": bool(_run_text(["git", "status", "--porcelain"])),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_frame(frame: pl.DataFrame) -> None:
    if frame.is_empty():
        raise ValueError("Benchmark dataset is empty.")

    null_counts = dict(zip(frame.columns, frame.null_count().row(0)))
    null_columns = [name for name, count in null_counts.items() if count]
    if null_columns:
        raise ValueError(f"Benchmark columns contain null values: {null_columns}.")

    numeric_columns = ["y", *PAST_EXOGENOUS_COLUMNS, *FUTURE_EXOGENOUS_COLUMNS]
    finite_flags = frame.select(
        [pl.col(name).is_finite().all().alias(name) for name in numeric_columns]
    ).row(0, named=True)
    nonfinite_columns = [name for name, finite in finite_flags.items() if not finite]
    if nonfinite_columns:
        raise ValueError(
            f"Benchmark columns contain non-finite values: {nonfinite_columns}."
        )

    duplicate_rows = frame.select(
        pl.struct("unique_id", "date").is_duplicated().any().alias("has_duplicates")
    ).item()
    if duplicate_rows:
        raise ValueError("Benchmark dataset contains duplicate (unique_id, date) rows.")


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value.astype("<f4", copy=False))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _configure_accuracy(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _seed_everything(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")


def _configure_performance(seed: int) -> None:
    _seed_everything(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")


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


def _standardize_exogenous(
    frame: pl.DataFrame,
    *,
    train_ids: list[str],
) -> tuple[pl.DataFrame, dict[str, dict[str, float]], list[str], list[str]]:
    train_frame = frame.filter(pl.col("unique_id").is_in(train_ids))
    stats: dict[str, dict[str, float]] = {}
    expressions: list[pl.Expr] = []
    standardized_names: dict[str, str] = {}
    for column in (*PAST_EXOGENOUS_COLUMNS, *FUTURE_EXOGENOUS_COLUMNS):
        mean = float(train_frame[column].mean())
        std = float(train_frame[column].std())
        if not math.isfinite(mean) or not math.isfinite(std):
            raise ValueError(f"Non-finite scaler statistics for {column!r}.")
        if std < 1e-8:
            std = 1.0
        output_name = f"{column}__z"
        standardized_names[column] = output_name
        stats[column] = {"mean": mean, "std": std}
        expressions.append(
            ((pl.col(column).cast(pl.Float64) - mean) / std)
            .cast(pl.Float32)
            .alias(output_name)
        )
    return (
        frame.with_columns(expressions),
        stats,
        [standardized_names[name] for name in PAST_EXOGENOUS_COLUMNS],
        [standardized_names[name] for name in FUTURE_EXOGENOUS_COLUMNS],
    )


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


def _patchtst_config(*, exogenous: bool) -> PatchTSTConfig:
    return PatchTSTConfig(
        lookback=LOOKBACK,
        horizon=HORIZON,
        c_in=1,
        patch_len=12,
        stride=8,
        padding_patch="end",
        past_exo_cont_dim=(len(PAST_EXOGENOUS_COLUMNS) if exogenous else 0),
        future_exo_dim=(len(FUTURE_EXOGENOUS_COLUMNS) if exogenous else 0),
        d_model=128,
        n_layers=3,
        d_ff=256,
        norm="LayerNorm",
        dropout=0.1,
        pre_norm=True,
        use_revin=True,
        future_exo_fusion_dropout=0.1,
        attn=AttentionConfig(
            n_heads=8,
            d_model=128,
            attn_dropout=0.1,
            proj_dropout=0.1,
        ),
    )


def _patchmixer_config(*, exogenous: bool) -> PatchMixerConfig:
    return PatchMixerConfig(
        lookback=LOOKBACK,
        horizon=HORIZON,
        enc_in=1,
        patch_len=12,
        stride=8,
        mixer_kernel_size=5,
        d_model=128,
        e_layers=6,
        dropout=0.1,
        head_dropout=0.02,
        f_out=256,
        head_hidden=256,
        past_exo_mode="z_gate",
        past_exo_cont_dim=(len(PAST_EXOGENOUS_COLUMNS) if exogenous else 0),
        future_exo_dim=(len(FUTURE_EXOGENOUS_COLUMNS) if exogenous else 0),
        use_revin=True,
    )


def _build_model(case: ModelCase) -> torch.nn.Module:
    with redirect_stdout(StringIO()):
        if case.family == "patchtst":
            config = _patchtst_config(exogenous=case.exogenous)
            return build_patchTST_exogenous(config) if case.exogenous else build_patchTST(config)
        if case.family == "patchmixer":
            config = _patchmixer_config(exogenous=case.exogenous)
            return (
                build_patch_mixer_exogenous(config)
                if case.exogenous
                else build_patch_mixer(config)
            )
    raise ValueError(f"Unsupported model case: {case}")


def _point_prediction(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        output = output.get("pred", output.get("point"))
    if not torch.is_tensor(output):
        raise TypeError(f"Expected tensor point output, got {type(output).__name__}.")
    if output.ndim == 3 and output.shape[-1] == 1:
        output = output.squeeze(-1)
    if output.ndim != 2 or output.shape[1] != HORIZON:
        raise RuntimeError(f"Expected [B,{HORIZON}], got {tuple(output.shape)}.")
    return output


def _autocast_factory(precision: str) -> Callable[[], Any]:
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected CUDA device does not support BF16.")
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext


def _forward_batch(
    model: torch.nn.Module,
    case: ModelCase,
    batch: Any,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    inputs = batch[0].cuda(non_blocking=True)
    targets = batch[1].cuda(non_blocking=True)
    uids = [str(uid) for uid in batch[2]]
    if not case.exogenous:
        return _point_prediction(model(inputs)), targets, uids

    future_exo = batch[3].cuda(non_blocking=True)
    past_exo_cont = batch[4].cuda(non_blocking=True)
    output = model(
        inputs,
        future_exo=future_exo,
        past_exo_cont=past_exo_cont,
    )
    return _point_prediction(output), targets, uids


@torch.no_grad()
def _mse_on_loader(
    model: torch.nn.Module,
    case: ModelCase,
    loader: DataLoader,
    *,
    autocast_context: Callable[[], Any],
) -> float:
    model.eval()
    squared_error = 0.0
    count = 0
    for batch in loader:
        with autocast_context():
            predictions, targets, _ = _forward_batch(model, case, batch)
        squared_error += float(
            F.mse_loss(predictions.float(), targets.float(), reduction="sum")
        )
        count += targets.numel()
    return squared_error / max(1, count)


def _train_model(
    case: ModelCase,
    *,
    dataset: MultiPartExoTrainingDataset,
    train_indices: list[int],
    validation_indices: list[int],
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
    model = _build_model(case).cuda().train()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

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
    torch.cuda.synchronize()
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_squared_error = 0.0
        train_count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                predictions, targets, _ = _forward_batch(model, case, batch)
                loss = F.mse_loss(predictions, targets)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(
                    f"{case.key} produced a non-finite gradient at epoch {epoch}."
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
            case,
            validation_loader,
            autocast_context=autocast_context,
        )
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
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_mse": train_mse,
                "validation_mse": validation_mse,
            }
        )
        if epoch == 1 or epoch % 5 == 0 or improved:
            print(
                "ACCURACY_PROGRESS="
                + json.dumps(
                    {
                        "seed": seed,
                        "model": case.key,
                        "epoch": epoch,
                        "train_mse": train_mse,
                        "validation_mse": validation_mse,
                        "best_epoch": best_epoch,
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
    result = {
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
    return model, result


@torch.no_grad()
def _predict_loader(
    model: torch.nn.Module,
    case: ModelCase,
    loader: DataLoader,
    *,
    autocast_context: Callable[[], Any],
) -> dict[str, Any]:
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    last_values: list[np.ndarray] = []
    seasonal_values: list[np.ndarray] = []
    uids: list[str] = []
    seasonal_start = LOOKBACK - SEASONAL_PERIOD
    for batch in loader:
        inputs = batch[0].cuda(non_blocking=True)
        with autocast_context():
            batch_predictions, batch_targets, batch_uids = _forward_batch(
                model,
                case,
                batch,
            )
        targets.append(batch_targets.float().cpu().numpy())
        predictions.append(batch_predictions.float().cpu().numpy())
        last_values.append(
            inputs[:, -1, 0]
            .unsqueeze(1)
            .expand(-1, HORIZON)
            .float()
            .cpu()
            .numpy()
        )
        seasonal_values.append(
            inputs[:, seasonal_start : seasonal_start + HORIZON, 0]
            .float()
            .cpu()
            .numpy()
        )
        uids.extend(batch_uids)
    return {
        "targets": np.concatenate(targets).astype(np.float64),
        "predictions": np.concatenate(predictions).astype(np.float64),
        "last_value": np.concatenate(last_values).astype(np.float64),
        "seasonal_naive_52": np.concatenate(seasonal_values).astype(np.float64),
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


def _metric_bundle(payload: dict[str, Any]) -> dict[str, Any]:
    targets = payload["targets"]
    predictions = payload["predictions"]
    uids = payload["uids"]
    per_series = {
        uid: _metrics(targets[uids == uid], predictions[uids == uid])
        for uid in sorted(set(str(value) for value in uids))
    }
    return {
        "micro": _metrics(targets, predictions),
        "macro_series": {
            name: float(np.mean([value[name] for value in per_series.values()]))
            for name in next(iter(per_series.values()))
        },
        "per_series": per_series,
    }


def _prediction_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "windows": int(payload["targets"].shape[0]),
        "forecast_points": int(payload["targets"].size),
        "prediction_sha256": _array_sha256(payload["predictions"]),
        "metrics": _metric_bundle(payload),
    }


def _paired_summary(
    endogenous: dict[str, Any],
    exogenous: dict[str, Any],
) -> dict[str, Any]:
    if not np.array_equal(endogenous["targets"], exogenous["targets"]):
        raise RuntimeError("Paired evaluations do not share identical targets.")
    if not np.array_equal(endogenous["uids"], exogenous["uids"]):
        raise RuntimeError("Paired evaluations do not share identical series ordering.")
    targets = endogenous["targets"]
    endogenous_predictions = endogenous["predictions"]
    exogenous_predictions = exogenous["predictions"]
    endogenous_absolute_error = np.abs(endogenous_predictions - targets)
    exogenous_absolute_error = np.abs(exogenous_predictions - targets)
    endogenous_metrics = _metrics(targets, endogenous_predictions)
    exogenous_metrics = _metrics(targets, exogenous_predictions)
    lower_is_better = ("mae", "mse", "rmse", "smape", "wape")
    return {
        "exogenous_relative_improvement_pct": {
            name: float(
                100.0
                * (endogenous_metrics[name] - exogenous_metrics[name])
                / max(abs(endogenous_metrics[name]), 1e-12)
            )
            for name in lower_is_better
        },
        "pointwise_absolute_error_win_rate": {
            "endogenous": float(
                np.mean(endogenous_absolute_error < exogenous_absolute_error)
            ),
            "exogenous": float(
                np.mean(exogenous_absolute_error < endogenous_absolute_error)
            ),
            "tie": float(np.mean(exogenous_absolute_error == endogenous_absolute_error)),
        },
        "overall_mae_winner": (
            "exogenous"
            if exogenous_metrics["mae"] < endogenous_metrics["mae"]
            else "endogenous"
            if endogenous_metrics["mae"] < exogenous_metrics["mae"]
            else "tie"
        ),
    }


def _run_accuracy_seed(
    frame: pl.DataFrame,
    *,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    precision: str,
) -> dict[str, Any]:
    _configure_accuracy(seed)
    splits = _split_ids(
        frame["unique_id"].unique().to_list(),
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    normalized, scaler, past_columns, future_columns = _standardize_exogenous(
        frame,
        train_ids=splits["train"],
    )
    dataset = MultiPartExoTrainingDataset(
        normalized,
        LOOKBACK,
        HORIZON,
        "weekly",
        id_col="unique_id",
        date_col="date",
        qty_col="y",
        past_exo_cont_cols=past_columns,
        future_exo_cont_cols=future_columns,
    )
    train_indices = _indices_for_ids(dataset, splits["train"])
    validation_indices = _indices_for_ids(dataset, splits["validation"])
    test_indices = _indices_for_ids(dataset, splits["test"])
    last_test_indices = _last_indices_for_ids(dataset, splits["test"])
    autocast_context = _autocast_factory(precision)

    model_results: dict[str, Any] = {}
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    for case in MODEL_CASES:
        model, training = _train_model(
            case,
            dataset=dataset,
            train_indices=train_indices,
            validation_indices=validation_indices,
            seed=seed,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            precision=precision,
        )
        all_loader = _make_loader(
            dataset,
            test_indices,
            batch_size=batch_size,
            shuffle=False,
            seed=seed + 2,
        )
        last_loader = _make_loader(
            dataset,
            last_test_indices,
            batch_size=batch_size,
            shuffle=False,
            seed=seed + 2,
        )
        all_payload = _predict_loader(
            model,
            case,
            all_loader,
            autocast_context=autocast_context,
        )
        last_payload = _predict_loader(
            model,
            case,
            last_loader,
            autocast_context=autocast_context,
        )
        model_results[case.key] = {
            "training": training,
            "test_all_rolling_windows": _prediction_summary(all_payload),
            "test_last_origin_per_series": _prediction_summary(last_payload),
        }
        predictions[case.key] = {"all": all_payload, "last": last_payload}
        del model, all_loader, last_loader
        gc.collect()
        torch.cuda.empty_cache()

    paired: dict[str, Any] = {}
    for family in ("patchtst", "patchmixer"):
        endogenous_key = f"{family}_endogenous"
        exogenous_key = f"{family}_exogenous"
        paired[family] = {
            "test_all_rolling_windows": _paired_summary(
                predictions[endogenous_key]["all"],
                predictions[exogenous_key]["all"],
            ),
            "test_last_origin_per_series": _paired_summary(
                predictions[endogenous_key]["last"],
                predictions[exogenous_key]["last"],
            ),
        }

    split_fingerprint = hashlib.sha256(
        json.dumps(splits, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "seed": seed,
        "split_fingerprint": split_fingerprint,
        "splits": splits,
        "split_series_counts": {name: len(ids) for name, ids in splits.items()},
        "split_window_counts": {
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test_all": len(test_indices),
            "test_last_origin": len(last_test_indices),
        },
        "exogenous_scaler": scaler,
        "models": model_results,
        "paired_comparison": paired,
    }


def _aggregate_accuracy(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for family in ("patchtst", "patchmixer"):
        output[family] = {}
        for evaluation in (
            "test_all_rolling_windows",
            "test_last_origin_per_series",
        ):
            records = [
                {
                    "seed": result["seed"],
                    "winner": result["paired_comparison"][family][evaluation][
                        "overall_mae_winner"
                    ],
                    "mae_improvement_pct": result["paired_comparison"][family][
                        evaluation
                    ]["exogenous_relative_improvement_pct"]["mae"],
                    "endogenous_mae": result["models"][f"{family}_endogenous"][
                        evaluation
                    ]["metrics"]["micro"]["mae"],
                    "exogenous_mae": result["models"][f"{family}_exogenous"][
                        evaluation
                    ]["metrics"]["micro"]["mae"],
                }
                for result in seed_results
            ]
            improvements = [record["mae_improvement_pct"] for record in records]
            output[family][evaluation] = {
                "records": records,
                "seed_wins": {
                    name: sum(record["winner"] == name for record in records)
                    for name in ("endogenous", "exogenous", "tie")
                },
                "mae_improvement_pct": {
                    "mean": statistics.fmean(improvements),
                    "population_stddev": statistics.pstdev(improvements),
                    "min": min(improvements),
                    "max": max(improvements),
                },
            }
    return output


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _make_cuda_batch(
    *,
    batch_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    inputs = torch.randn(
        batch_size,
        LOOKBACK,
        1,
        device="cuda",
        generator=generator,
    )
    past = torch.randn(
        batch_size,
        LOOKBACK,
        len(PAST_EXOGENOUS_COLUMNS),
        device="cuda",
        generator=generator,
    )
    future = torch.randn(
        batch_size,
        HORIZON,
        len(FUTURE_EXOGENOUS_COLUMNS),
        device="cuda",
        generator=generator,
    )
    noise = 0.05 * torch.randn(
        batch_size,
        HORIZON,
        device="cuda",
        generator=generator,
    )
    targets = inputs[:, -1, 0].unsqueeze(1) + 0.2 * future[..., 0] + noise
    return inputs, targets, past, future


def _benchmark_case(
    case: ModelCase,
    *,
    steps: int,
    warmup_steps: int,
    batch_size: int,
    precision: str,
    seed: int,
    lr: float,
    weight_decay: float,
) -> dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    _seed_everything(seed)
    model = _build_model(case).cuda().train()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    inputs, targets, past, future = _make_cuda_batch(
        batch_size=batch_size,
        seed=seed + 1,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    autocast_context = _autocast_factory(precision)

    def training_step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            if case.exogenous:
                prediction = _point_prediction(
                    model(
                        inputs,
                        future_exo=future,
                        past_exo_cont=past,
                    )
                )
            else:
                prediction = _point_prediction(model(inputs))
            loss = F.mse_loss(prediction, targets)
        loss.backward()
        optimizer.step()
        return loss.detach()

    for _ in range(warmup_steps):
        training_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    first_loss: torch.Tensor | None = None
    last_loss: torch.Tensor | None = None
    for index, (start, end) in enumerate(zip(starts, ends)):
        start.record()
        loss = training_step()
        end.record()
        if index == 0:
            first_loss = loss
        last_loss = loss
    torch.cuda.synchronize()
    assert first_loss is not None and last_loss is not None
    times = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    total = sum(times)
    mean = statistics.fmean(times)
    result = {
        "model": case.key,
        "parameters": parameter_count,
        "timing_ms": {
            "total": total,
            "mean": mean,
            "median": statistics.median(times),
            "p95": _percentile(times, 0.95),
            "min": min(times),
            "max": max(times),
            "population_stddev": statistics.pstdev(times),
        },
        "throughput": {
            "steps_per_second": 1000.0 / mean,
            "samples_per_second": batch_size * steps * 1000.0 / total,
        },
        "memory_mib": {
            "peak_allocated": torch.cuda.max_memory_allocated() / (1024**2),
            "peak_reserved": torch.cuda.max_memory_reserved() / (1024**2),
        },
        "loss": {"first_measured": float(first_loss), "last_measured": float(last_loss)},
    }
    del model, optimizer, inputs, targets, past, future, starts, ends
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return result


def _performance_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {result["model"]: result for result in results}
    output: dict[str, Any] = {}
    for family in ("patchtst", "patchmixer"):
        endogenous = by_key[f"{family}_endogenous"]
        exogenous = by_key[f"{family}_exogenous"]
        endogenous_time = endogenous["timing_ms"]["mean"]
        exogenous_time = exogenous["timing_ms"]["mean"]
        output[family] = {
            "exogenous_step_time_overhead_pct": (
                100.0 * (exogenous_time - endogenous_time) / endogenous_time
            ),
            "exogenous_throughput_ratio": (
                exogenous["throughput"]["samples_per_second"]
                / endogenous["throughput"]["samples_per_second"]
            ),
            "parameter_overhead": exogenous["parameters"] - endogenous["parameters"],
            "peak_allocated_overhead_mib": (
                exogenous["memory_mib"]["peak_allocated"]
                - endogenous["memory_mib"]["peak_allocated"]
            ),
        }
    return output


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    data_path = args.data.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique.")

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    _configure_accuracy(args.seeds[0])
    device_name = _validate_cuda(args.expected_device)
    required_columns = [
        "unique_id",
        "date",
        "y",
        *PAST_EXOGENOUS_COLUMNS,
        *FUTURE_EXOGENOUS_COLUMNS,
    ]
    frame = pl.read_parquet(data_path, columns=required_columns)
    _validate_frame(frame)

    seed_results = [
        _run_accuracy_seed(
            frame,
            seed=seed,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            precision=args.accuracy_precision,
        )
        for seed in args.seeds
    ]

    performance_seed = 20260722
    _configure_performance(performance_seed)
    performance_results = [
        _benchmark_case(
            case,
            steps=args.performance_steps,
            warmup_steps=args.warmup_steps,
            batch_size=args.batch_size,
            precision=args.performance_precision,
            seed=performance_seed,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        for case in MODEL_CASES
    ]
    properties = torch.cuda.get_device_properties(0)
    result = {
        "schema_version": 1,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": device_name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_total_memory_mib": properties.total_memory / (1024**2),
        },
        "source": {"git": _git_metadata()},
        "dataset": {
            "path": str(data_path),
            "sha256": _file_sha256(data_path),
            "rows": frame.height,
            "series": frame["unique_id"].n_unique(),
            "past_exogenous_columns": list(PAST_EXOGENOUS_COLUMNS),
            "future_exogenous_columns": list(FUTURE_EXOGENOUS_COLUMNS),
            "scaler_fit_scope": "train-series rows only",
        },
        "accuracy_protocol": {
            "seeds": args.seeds,
            "split": "series_id_disjoint",
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "precision": args.accuracy_precision,
            "loss": "mse",
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "deterministic_algorithms": True,
            "model_selection": "lowest validation MSE",
            "test_evaluations": ["all_rolling_windows", "last_origin_per_series"],
        },
        "accuracy_seeds": seed_results,
        "accuracy_aggregate": _aggregate_accuracy(seed_results),
        "performance_protocol": {
            "seed": performance_seed,
            "precision": args.performance_precision,
            "batch_size": args.batch_size,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.performance_steps,
            "cuda_resident_batch": True,
            "data_loader_included": False,
            "host_to_device_transfer_included": False,
        },
        "performance_models": performance_results,
        "performance_summary": _performance_summary(performance_results),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "EXOGENOUS_COMPARISON_COMPLETE="
        + json.dumps(
            {
                "elapsed_seconds": result["elapsed_seconds"],
                "git_commit": result["source"]["git"]["commit"],
                "output": str(output_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

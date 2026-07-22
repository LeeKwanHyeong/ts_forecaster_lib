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
from contextlib import contextmanager, nullcontext, redirect_stdout
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
HISTORY_GATE_FEATURE_NAMES = (
    "log1p_abs_mean",
    "log1p_std",
    "last_z",
    "linear_trend_z",
    "recent_4_minus_mean_z",
    "recent_12_minus_mean_z",
    "seasonal_52_gap_z",
    "range_z",
    "zero_fraction",
)
HISTORY_GATE_RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
HISTORY_GATE_KNN_NEIGHBORS = (8, 16, 32, 64, 128)
HISTORY_GATE_REFERENCE_KEYS = {
    "patchmixer_future_shift",
    "patchmixer_future_shift_normalized",
}


@dataclass(frozen=True)
class ModelCase:
    key: str
    family: str
    past_exogenous: bool = False
    future_exogenous: bool = False
    future_shift_space: str | None = None
    future_normalized_residual_limit: float | None = None

    @property
    def exogenous(self) -> bool:
        return self.past_exogenous or self.future_exogenous


MODEL_CASES = (
    ModelCase("patchtst_endogenous", "patchtst"),
    ModelCase(
        "patchtst_exogenous",
        "patchtst",
        past_exogenous=True,
        future_exogenous=True,
    ),
    ModelCase("patchmixer_endogenous", "patchmixer"),
    ModelCase("patchmixer_past_gate", "patchmixer", past_exogenous=True),
    ModelCase(
        "patchmixer_future_shift",
        "patchmixer",
        future_exogenous=True,
        future_shift_space="output",
    ),
    ModelCase(
        "patchmixer_exogenous",
        "patchmixer",
        past_exogenous=True,
        future_exogenous=True,
        future_shift_space="output",
    ),
)

PATCHMIXER_SHIFT_SPACE_CASES = (
    ModelCase("patchmixer_endogenous", "patchmixer"),
    ModelCase(
        "patchmixer_future_shift",
        "patchmixer",
        future_exogenous=True,
        future_shift_space="output",
    ),
    ModelCase(
        "patchmixer_future_shift_normalized",
        "patchmixer",
        future_exogenous=True,
        future_shift_space="normalized",
    ),
    ModelCase(
        "patchmixer_future_shift_normalized_bounded",
        "patchmixer",
        future_exogenous=True,
        future_shift_space="normalized",
        future_normalized_residual_limit=0.15,
    ),
)

PATCHMIXER_ABLATION_PAIRS = {
    "past_gate_vs_endogenous": (
        "patchmixer_endogenous",
        "patchmixer_past_gate",
    ),
    "future_shift_vs_endogenous": (
        "patchmixer_endogenous",
        "patchmixer_future_shift",
    ),
    "full_vs_endogenous": (
        "patchmixer_endogenous",
        "patchmixer_exogenous",
    ),
    "full_vs_future_shift": (
        "patchmixer_future_shift",
        "patchmixer_exogenous",
    ),
    "full_vs_past_gate": (
        "patchmixer_past_gate",
        "patchmixer_exogenous",
    ),
}

PATCHMIXER_SHIFT_SPACE_PAIRS = {
    "output_vs_endogenous": (
        "patchmixer_endogenous",
        "patchmixer_future_shift",
    ),
    "normalized_vs_endogenous": (
        "patchmixer_endogenous",
        "patchmixer_future_shift_normalized",
    ),
    "normalized_vs_output": (
        "patchmixer_future_shift",
        "patchmixer_future_shift_normalized",
    ),
    "bounded_vs_endogenous": (
        "patchmixer_endogenous",
        "patchmixer_future_shift_normalized_bounded",
    ),
    "bounded_vs_output": (
        "patchmixer_future_shift",
        "patchmixer_future_shift_normalized_bounded",
    ),
    "bounded_vs_normalized": (
        "patchmixer_future_shift_normalized",
        "patchmixer_future_shift_normalized_bounded",
    ),
}


def _case_metadata(case: ModelCase) -> dict[str, Any]:
    return {
        "key": case.key,
        "family": case.family,
        "past_exogenous": case.past_exogenous,
        "future_exogenous": case.future_exogenous,
        "future_shift_space": case.future_shift_space,
        "future_normalized_residual_limit": (
            case.future_normalized_residual_limit
        ),
    }


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
    parser.add_argument(
        "--case-set",
        choices=("all", "patchmixer-shift-space"),
        default="all",
        help="Select the historical full comparison or the focused shift-space comparison.",
    )
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


def _patchtst_config(case: ModelCase) -> PatchTSTConfig:
    return PatchTSTConfig(
        lookback=LOOKBACK,
        horizon=HORIZON,
        c_in=1,
        patch_len=12,
        stride=8,
        padding_patch="end",
        past_exo_cont_dim=(
            len(PAST_EXOGENOUS_COLUMNS) if case.past_exogenous else 0
        ),
        future_exo_dim=(
            len(FUTURE_EXOGENOUS_COLUMNS) if case.future_exogenous else 0
        ),
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


def _patchmixer_config(case: ModelCase) -> PatchMixerConfig:
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
        past_exo_cont_dim=(
            len(PAST_EXOGENOUS_COLUMNS) if case.past_exogenous else 0
        ),
        future_exo_dim=(
            len(FUTURE_EXOGENOUS_COLUMNS) if case.future_exogenous else 0
        ),
        future_exo_shift_space=case.future_shift_space or "output",
        future_exo_normalized_residual_limit=(
            case.future_normalized_residual_limit
        ),
        use_revin=True,
    )


def _build_model(case: ModelCase) -> torch.nn.Module:
    with redirect_stdout(StringIO()):
        if case.family == "patchtst":
            config = _patchtst_config(case)
            return build_patchTST_exogenous(config) if case.exogenous else build_patchTST(config)
        if case.family == "patchmixer":
            config = _patchmixer_config(case)
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


@contextmanager
def _temporarily_disable_future_shift(
    model: torch.nn.Module,
    *,
    disabled: bool,
) -> Iterable[None]:
    """Disable only the additive future shift while retaining input contracts."""
    if not disabled:
        yield
        return
    if not hasattr(model, "exo_scale"):
        raise RuntimeError(
            f"{type(model).__name__} cannot disable its future shift for diagnostics."
        )

    original_scale = model.exo_scale
    model.exo_scale = 0.0
    try:
        yield
    finally:
        model.exo_scale = original_scale


def _forward_batch(
    model: torch.nn.Module,
    case: ModelCase,
    batch: Any,
    *,
    zero_past: bool = False,
    zero_future: bool = False,
    omit_future: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    inputs = batch[0].cuda(non_blocking=True)
    targets = batch[1].cuda(non_blocking=True)
    uids = [str(uid) for uid in batch[2]]
    if not case.exogenous:
        return _point_prediction(model(inputs)), targets, uids

    kwargs: dict[str, torch.Tensor] = {}
    if case.future_exogenous:
        future_exo = batch[3].cuda(non_blocking=True)
        kwargs["future_exo"] = (
            torch.zeros_like(future_exo) if zero_future else future_exo
        )
    if case.past_exogenous:
        past_exo_cont = batch[4].cuda(non_blocking=True)
        kwargs["past_exo_cont"] = (
            torch.zeros_like(past_exo_cont) if zero_past else past_exo_cont
        )
    with _temporarily_disable_future_shift(model, disabled=omit_future):
        output = model(inputs, **kwargs)
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


def _history_gate_features(inputs: torch.Tensor) -> torch.Tensor:
    """Build target-history-only features available at forecast time."""
    if inputs.ndim != 3 or inputs.shape[1] != LOOKBACK or inputs.shape[2] != 1:
        raise ValueError(
            f"Expected history [B,{LOOKBACK},1], got {tuple(inputs.shape)}."
        )

    history = inputs[:, :, 0].float()
    mean = history.mean(dim=1)
    std = torch.sqrt(history.var(dim=1, unbiased=False) + 1e-5)
    centered = history - mean.unsqueeze(1)
    time_axis = torch.linspace(
        -1.0,
        1.0,
        LOOKBACK,
        device=history.device,
        dtype=history.dtype,
    )
    linear_trend = torch.sum(centered * time_axis, dim=1) / torch.sum(
        time_axis.square()
    )
    recent_4 = history[:, -4:].mean(dim=1)
    recent_12 = history[:, -12:].mean(dim=1)
    seasonal_lag = history[:, -1 - SEASONAL_PERIOD]
    scale = std.clamp_min(1e-6)
    return torch.stack(
        (
            torch.log1p(mean.abs()),
            torch.log1p(std),
            (history[:, -1] - mean) / scale,
            linear_trend / scale,
            (recent_4 - mean) / scale,
            (recent_12 - mean) / scale,
            (history[:, -1] - seasonal_lag) / scale,
            (history.max(dim=1).values - history.min(dim=1).values) / scale,
            torch.mean((history.abs() <= 1e-8).float(), dim=1),
        ),
        dim=1,
    )


@torch.no_grad()
def _predict_loader(
    model: torch.nn.Module,
    case: ModelCase,
    loader: DataLoader,
    *,
    autocast_context: Callable[[], Any],
    zero_past: bool = False,
    zero_future: bool = False,
    omit_future: bool = False,
) -> dict[str, Any]:
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    last_values: list[np.ndarray] = []
    seasonal_values: list[np.ndarray] = []
    history_stds: list[np.ndarray] = []
    history_features: list[np.ndarray] = []
    uids: list[str] = []
    seasonal_start = LOOKBACK - SEASONAL_PERIOD
    for batch in loader:
        inputs = batch[0].cuda(non_blocking=True)
        with autocast_context():
            batch_predictions, batch_targets, batch_uids = _forward_batch(
                model,
                case,
                batch,
                zero_past=zero_past,
                zero_future=zero_future,
                omit_future=omit_future,
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
        history_stds.append(
            torch.sqrt(
                inputs[:, :, 0].float().var(dim=1, unbiased=False) + 1e-5
            )
            .cpu()
            .numpy()
        )
        history_features.append(_history_gate_features(inputs).cpu().numpy())
        uids.extend(batch_uids)
    return {
        "targets": np.concatenate(targets).astype(np.float64),
        "predictions": np.concatenate(predictions).astype(np.float64),
        "last_value": np.concatenate(last_values).astype(np.float64),
        "seasonal_naive_52": np.concatenate(seasonal_values).astype(np.float64),
        "history_std": np.concatenate(history_stds).astype(np.float64),
        "history_features": np.concatenate(history_features).astype(np.float64),
        "history_feature_names": HISTORY_GATE_FEATURE_NAMES,
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


def _safe_pearson(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.size < 2 or first.size != second.size:
        return None
    if np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return None
    value = float(np.corrcoef(first, second)[0, 1])
    return value if math.isfinite(value) else None


def _validate_diagnostic_payloads(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    for field in ("targets", "uids", "history_std", "history_features"):
        if not np.array_equal(baseline[field], candidate[field]):
            raise RuntimeError(
                f"Diagnostic evaluations do not share identical {field}."
            )
    if baseline["history_feature_names"] != candidate["history_feature_names"]:
        raise RuntimeError(
            "Diagnostic evaluations do not share identical history feature names."
        )
    if baseline["predictions"].shape != candidate["predictions"].shape:
        raise RuntimeError("Diagnostic prediction shapes do not match.")


def _mse_optimal_gate_targets(
    base_predictions: np.ndarray,
    full_predictions: np.ndarray,
    targets: np.ndarray,
    *,
    horizon_shared: bool,
) -> tuple[np.ndarray, np.ndarray]:
    delta = full_predictions - base_predictions
    numerator = delta * (targets - base_predictions)
    denominator = np.square(delta)
    if horizon_shared:
        numerator = np.sum(numerator, axis=1, keepdims=True)
        denominator = np.sum(denominator, axis=1, keepdims=True)
    gate = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 1e-12,
    )
    return np.clip(gate, 0.0, 1.0), denominator


def _mae_optimal_gate_targets(
    base_predictions: np.ndarray,
    full_predictions: np.ndarray,
    targets: np.ndarray,
    *,
    horizon_shared: bool,
) -> np.ndarray:
    delta = full_predictions - base_predictions
    roots = np.divide(
        targets - base_predictions,
        delta,
        out=np.zeros_like(delta, dtype=np.float64),
        where=np.abs(delta) > 1e-12,
    )
    if not horizon_shared:
        return np.clip(roots, 0.0, 1.0)

    gates = np.zeros((targets.shape[0], 1), dtype=np.float64)
    for index in range(targets.shape[0]):
        weights = np.abs(delta[index])
        active = weights > 1e-12
        if not np.any(active):
            continue
        active_roots = roots[index, active]
        active_weights = weights[active]
        order = np.argsort(active_roots, kind="stable")
        cumulative = np.cumsum(active_weights[order])
        median_index = int(
            np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left")
        )
        gates[index, 0] = np.clip(
            active_roots[order[min(median_index, order.size - 1)]],
            0.0,
            1.0,
        )
    return gates


def _binary_oracle_gate(
    base_predictions: np.ndarray,
    full_predictions: np.ndarray,
    targets: np.ndarray,
    *,
    horizon_shared: bool,
) -> np.ndarray:
    base_error = np.square(base_predictions - targets)
    full_error = np.square(full_predictions - targets)
    if horizon_shared:
        base_error = np.sum(base_error, axis=1, keepdims=True)
        full_error = np.sum(full_error, axis=1, keepdims=True)
    return (full_error < base_error).astype(np.float64)


def _apply_gate(
    base_predictions: np.ndarray,
    full_predictions: np.ndarray,
    gate: np.ndarray,
) -> np.ndarray:
    gate = np.asarray(gate, dtype=np.float64)
    if gate.ndim == 1:
        gate = gate[:, None]
    if gate.ndim != 2 or gate.shape[0] != base_predictions.shape[0]:
        raise ValueError("Gate must have shape [windows, 1 or horizon].")
    if gate.shape[1] not in (1, base_predictions.shape[1]):
        raise ValueError("Gate width must be one or match the forecast horizon.")
    return base_predictions + gate * (full_predictions - base_predictions)


def _fit_constant_gate(
    gate_targets: np.ndarray,
    gate_weights: np.ndarray,
) -> np.ndarray:
    numerator = np.sum(gate_targets * gate_weights, axis=0, keepdims=True)
    denominator = np.sum(gate_weights, axis=0, keepdims=True)
    return np.clip(
        np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-12,
        ),
        0.0,
        1.0,
    )


def _standardize_history_features(
    train_features: np.ndarray,
    evaluation_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train_features, axis=0, keepdims=True)
    std = np.std(train_features, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (train_features - mean) / std, (evaluation_features - mean) / std


def _ridge_gate_predict(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_weights: np.ndarray,
    evaluation_features: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    train_scaled, evaluation_scaled = _standardize_history_features(
        train_features,
        evaluation_features,
    )
    train_design = np.column_stack(
        (np.ones(train_scaled.shape[0], dtype=np.float64), train_scaled)
    )
    evaluation_design = np.column_stack(
        (
            np.ones(evaluation_scaled.shape[0], dtype=np.float64),
            evaluation_scaled,
        )
    )
    penalty = np.eye(train_design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    predictions = np.zeros(
        (evaluation_features.shape[0], train_targets.shape[1]),
        dtype=np.float64,
    )
    for output_index in range(train_targets.shape[1]):
        weights = train_weights[:, output_index]
        if np.sum(weights) <= 1e-12:
            continue
        weights = weights / max(float(np.mean(weights)), 1e-12)
        weighted_design = train_design * weights[:, None]
        system = train_design.T @ weighted_design + alpha * penalty
        right_hand_side = train_design.T @ (
            weights * train_targets[:, output_index]
        )
        coefficients = np.linalg.pinv(system) @ right_hand_side
        predictions[:, output_index] = evaluation_design @ coefficients
    return np.clip(predictions, 0.0, 1.0)


def _knn_gate_predict(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_weights: np.ndarray,
    evaluation_features: np.ndarray,
    *,
    neighbors: int,
) -> np.ndarray:
    train_scaled, evaluation_scaled = _standardize_history_features(
        train_features,
        evaluation_features,
    )
    squared_distances = np.mean(
        np.square(evaluation_scaled[:, None, :] - train_scaled[None, :, :]),
        axis=2,
    )
    neighbor_count = min(neighbors, train_scaled.shape[0])
    neighbor_indices = np.argpartition(
        squared_distances,
        kth=neighbor_count - 1,
        axis=1,
    )[:, :neighbor_count]
    neighbor_distances = np.take_along_axis(
        squared_distances,
        neighbor_indices,
        axis=1,
    )
    similarity = 1.0 / (np.sqrt(np.maximum(neighbor_distances, 0.0)) + 1e-6)
    predictions = np.zeros(
        (evaluation_features.shape[0], train_targets.shape[1]),
        dtype=np.float64,
    )
    for output_index in range(train_targets.shape[1]):
        local_weights = similarity * train_weights[
            neighbor_indices,
            output_index,
        ]
        denominator = np.sum(local_weights, axis=1)
        numerator = np.sum(
            local_weights * train_targets[neighbor_indices, output_index],
            axis=1,
        )
        predictions[:, output_index] = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-12,
        )
    return np.clip(predictions, 0.0, 1.0)


def _history_gate_predict(
    learner: str,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_weights: np.ndarray,
    evaluation_features: np.ndarray,
    hyperparameter: float | int,
) -> np.ndarray:
    if learner == "ridge":
        return _ridge_gate_predict(
            train_features,
            train_targets,
            train_weights,
            evaluation_features,
            alpha=float(hyperparameter),
        )
    if learner == "knn":
        return _knn_gate_predict(
            train_features,
            train_targets,
            train_weights,
            evaluation_features,
            neighbors=int(hyperparameter),
        )
    raise ValueError(f"Unsupported history gate learner: {learner!r}.")


def _forecast_mse_with_gate(
    base_predictions: np.ndarray,
    full_predictions: np.ndarray,
    targets: np.ndarray,
    gate: np.ndarray,
) -> float:
    predictions = _apply_gate(base_predictions, full_predictions, gate)
    return float(np.mean(np.square(predictions - targets)))


def _select_history_gate_hyperparameter(
    learner: str,
    candidates: tuple[float | int, ...],
    features: np.ndarray,
    gate_targets: np.ndarray,
    gate_weights: np.ndarray,
    uids: np.ndarray,
    base_predictions: np.ndarray,
    full_predictions: np.ndarray,
    targets: np.ndarray,
) -> float | int:
    groups = sorted(set(str(value) for value in uids))
    if len(groups) < 2:
        return candidates[0]

    scores: list[float] = []
    for candidate in candidates:
        squared_error = 0.0
        point_count = 0
        for held_out in groups:
            evaluation_mask = uids == held_out
            train_mask = ~evaluation_mask
            gate = _history_gate_predict(
                learner,
                features[train_mask],
                gate_targets[train_mask],
                gate_weights[train_mask],
                features[evaluation_mask],
                candidate,
            )
            predictions = _apply_gate(
                base_predictions[evaluation_mask],
                full_predictions[evaluation_mask],
                gate,
            )
            squared_error += float(
                np.sum(np.square(predictions - targets[evaluation_mask]))
            )
            point_count += predictions.size
        scores.append(squared_error / max(point_count, 1))
    return candidates[min(range(len(candidates)), key=lambda index: scores[index])]


def _nested_group_oof_history_gate(
    learner: str,
    candidates: tuple[float | int, ...],
    features: np.ndarray,
    gate_targets: np.ndarray,
    gate_weights: np.ndarray,
    uids: np.ndarray,
    base_predictions: np.ndarray,
    full_predictions: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    groups = sorted(set(str(value) for value in uids))
    if len(groups) < 3:
        raise ValueError(
            "Nested series-out gate validation requires at least three series."
        )

    gates = np.zeros_like(gate_targets, dtype=np.float64)
    selected: dict[str, float | int] = {}
    for held_out in groups:
        evaluation_mask = uids == held_out
        train_mask = ~evaluation_mask
        hyperparameter = _select_history_gate_hyperparameter(
            learner,
            candidates,
            features[train_mask],
            gate_targets[train_mask],
            gate_weights[train_mask],
            uids[train_mask],
            base_predictions[train_mask],
            full_predictions[train_mask],
            targets[train_mask],
        )
        gates[evaluation_mask] = _history_gate_predict(
            learner,
            features[train_mask],
            gate_targets[train_mask],
            gate_weights[train_mask],
            features[evaluation_mask],
            hyperparameter,
        )
        selected[held_out] = hyperparameter
    return gates, selected


def _group_oof_constant_gate(
    gate_targets: np.ndarray,
    gate_weights: np.ndarray,
    uids: np.ndarray,
) -> np.ndarray:
    groups = sorted(set(str(value) for value in uids))
    if len(groups) < 2:
        raise ValueError("Series-out constant validation requires two series.")
    gates = np.zeros_like(gate_targets, dtype=np.float64)
    for held_out in groups:
        evaluation_mask = uids == held_out
        fitted = _fit_constant_gate(
            gate_targets[~evaluation_mask],
            gate_weights[~evaluation_mask],
        )
        gates[evaluation_mask] = fitted
    return gates


def _last_origin_indices(uids: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.flatnonzero(uids == uid)[-1]
            for uid in sorted(set(str(value) for value in uids))
        ],
        dtype=np.int64,
    )


def _gate_value_statistics(gate: np.ndarray) -> dict[str, float]:
    values = np.asarray(gate, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(values)),
        "population_stddev": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "fraction_below_0_05": float(np.mean(values < 0.05)),
        "fraction_above_0_95": float(np.mean(values > 0.95)),
    }


def _relative_metric_improvement(
    baseline: dict[str, float],
    candidate: dict[str, float],
) -> dict[str, float]:
    return {
        name: float(
            100.0
            * (baseline[name] - candidate[name])
            / max(abs(baseline[name]), 1e-12)
        )
        for name in ("mae", "mse", "rmse")
    }


def _gate_method_evaluations(
    base_predictions: np.ndarray,
    full_predictions: np.ndarray,
    targets: np.ndarray,
    uids: np.ndarray,
    methods: dict[str, np.ndarray],
    *,
    output_reference: np.ndarray | None,
) -> dict[str, Any]:
    evaluations = {
        "validation_all_rolling_windows": np.arange(targets.shape[0]),
        "validation_last_origin_per_series": _last_origin_indices(uids),
    }
    output: dict[str, Any] = {}
    for evaluation_name, indices in evaluations.items():
        method_rows: dict[str, Any] = {}
        for method_name, gate in methods.items():
            predictions = _apply_gate(
                base_predictions[indices],
                full_predictions[indices],
                gate[indices],
            )
            method_rows[method_name] = {
                "metrics": _metrics(targets[indices], predictions),
                "gate": _gate_value_statistics(gate[indices]),
                "prediction_sha256": _array_sha256(predictions),
            }

        always_on_metrics = method_rows["always_on"]["metrics"]
        oracle_mse = method_rows["oracle_mse"]["metrics"]["mse"]
        oracle_gain = always_on_metrics["mse"] - oracle_mse
        reference_metrics = (
            _metrics(targets[indices], output_reference[indices])
            if output_reference is not None
            else None
        )
        for row in method_rows.values():
            row["relative_improvement_pct_vs_always_on"] = (
                _relative_metric_improvement(always_on_metrics, row["metrics"])
            )
            row["mse_oracle_gain_capture_pct_from_always_on"] = (
                float(
                    100.0
                    * (always_on_metrics["mse"] - row["metrics"]["mse"])
                    / oracle_gain
                )
                if oracle_gain > 1e-12
                else None
            )
            if reference_metrics is not None:
                row["relative_improvement_pct_vs_output_reference"] = (
                    _relative_metric_improvement(reference_metrics, row["metrics"])
                )
        output[evaluation_name] = {
            "windows": int(indices.size),
            "output_reference_metrics": reference_metrics,
            "methods": method_rows,
        }
    return output


def _gate_capacity_analysis(
    base: dict[str, Any],
    full: dict[str, Any],
    *,
    output_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    _validate_diagnostic_payloads(base, full)
    if output_reference is not None:
        _validate_diagnostic_payloads(base, output_reference)

    base_predictions = base["predictions"]
    full_predictions = full["predictions"]
    targets = base["targets"]
    features = base["history_features"]
    uids = base["uids"]
    output_predictions = (
        output_reference["predictions"] if output_reference is not None else None
    )
    output: dict[str, Any] = {}
    for name, horizon_shared in (
        ("window_scalar", True),
        ("window_horizon", False),
    ):
        gate_targets, gate_weights = _mse_optimal_gate_targets(
            base_predictions,
            full_predictions,
            targets,
            horizon_shared=horizon_shared,
        )
        ridge_gate, ridge_selected = _nested_group_oof_history_gate(
            "ridge",
            HISTORY_GATE_RIDGE_ALPHAS,
            features,
            gate_targets,
            gate_weights,
            uids,
            base_predictions,
            full_predictions,
            targets,
        )
        knn_gate, knn_selected = _nested_group_oof_history_gate(
            "knn",
            HISTORY_GATE_KNN_NEIGHBORS,
            features,
            gate_targets,
            gate_weights,
            uids,
            base_predictions,
            full_predictions,
            targets,
        )
        gate_width = 1 if horizon_shared else targets.shape[1]
        methods = {
            "always_off": np.zeros((targets.shape[0], gate_width)),
            "always_on": np.ones((targets.shape[0], gate_width)),
            "validation_fit_constant": np.repeat(
                _fit_constant_gate(gate_targets, gate_weights),
                targets.shape[0],
                axis=0,
            ),
            "series_oof_constant": _group_oof_constant_gate(
                gate_targets,
                gate_weights,
                uids,
            ),
            "nested_series_oof_ridge": ridge_gate,
            "nested_series_oof_knn": knn_gate,
            "oracle_binary": _binary_oracle_gate(
                base_predictions,
                full_predictions,
                targets,
                horizon_shared=horizon_shared,
            ),
            "oracle_mse": gate_targets,
            "oracle_mae": _mae_optimal_gate_targets(
                base_predictions,
                full_predictions,
                targets,
                horizon_shared=horizon_shared,
            ),
        }
        output[name] = {
            "gate_width": gate_width,
            "history_learners": {
                "nested_series_oof_ridge": {
                    "candidate_alphas": list(HISTORY_GATE_RIDGE_ALPHAS),
                    "selected_by_held_out_series": ridge_selected,
                },
                "nested_series_oof_knn": {
                    "candidate_neighbors": list(HISTORY_GATE_KNN_NEIGHBORS),
                    "selected_by_held_out_series": knn_selected,
                },
            },
            "evaluations": _gate_method_evaluations(
                base_predictions,
                full_predictions,
                targets,
                uids,
                methods,
                output_reference=output_predictions,
            ),
        }
    return output


def _history_conditioned_gate_validation_upper_bound(
    output_shift: dict[str, Any],
    normalized_shift: dict[str, Any],
    normalized_without_shift: dict[str, Any],
) -> dict[str, Any]:
    _validate_diagnostic_payloads(output_shift, normalized_shift)
    _validate_diagnostic_payloads(normalized_without_shift, normalized_shift)
    return {
        "protocol": {
            "fit_and_evaluation_scope": "validation split only",
            "test_targets_used": False,
            "history_features": list(normalized_shift["history_feature_names"]),
            "cross_fit": "nested leave-one-series-out",
            "cross_fit_scope": "post-hoc gate only",
            "base_model_selection": (
                "base model checkpoints were selected on the complete validation "
                "split before post-hoc gate cross-fitting"
            ),
            "learner_objective": "weighted MSE-optimal gate regression",
            "gate_range": [0.0, 1.0],
            "interpretation": (
                "optimistic validation capacity characterization, not a held-out "
                "generalization estimate"
            ),
            "oracle_warning": (
                "oracle methods inspect validation targets and are unattainable "
                "ceilings, not deployable estimates"
            ),
        },
        "normalized_residual_gate": _gate_capacity_analysis(
            normalized_without_shift,
            normalized_shift,
            output_reference=output_shift,
        ),
        "output_to_normalized_blend": _gate_capacity_analysis(
            output_shift,
            normalized_shift,
            output_reference=None,
        ),
    }


def _error_comparison(
    targets: np.ndarray,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
) -> dict[str, float]:
    baseline_mae = float(np.mean(np.abs(baseline_predictions - targets)))
    candidate_mae = float(np.mean(np.abs(candidate_predictions - targets)))
    return {
        "baseline_mae": baseline_mae,
        "candidate_mae": candidate_mae,
        "candidate_mae_delta": candidate_mae - baseline_mae,
        "candidate_relative_improvement_pct": float(
            100.0
            * (baseline_mae - candidate_mae)
            / max(abs(baseline_mae), 1e-12)
        ),
    }


def _history_std_quartiles(history_std: np.ndarray) -> list[tuple[str, np.ndarray]]:
    order = np.argsort(history_std, kind="stable")
    return [
        (f"q{index + 1}", indices)
        for index, indices in enumerate(np.array_split(order, 4))
        if indices.size
    ]


def _paired_error_diagnostics(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    _validate_diagnostic_payloads(baseline, candidate)
    targets = baseline["targets"]
    baseline_predictions = baseline["predictions"]
    candidate_predictions = candidate["predictions"]
    history_std = baseline["history_std"]
    uids = baseline["uids"]
    window_mae_delta = np.mean(
        np.abs(candidate_predictions - targets)
        - np.abs(baseline_predictions - targets),
        axis=1,
    )

    by_horizon = [
        {
            "horizon": horizon + 1,
            **_error_comparison(
                targets[:, horizon],
                baseline_predictions[:, horizon],
                candidate_predictions[:, horizon],
            ),
        }
        for horizon in range(targets.shape[1])
    ]
    by_series = {
        uid: {
            "windows": int(np.count_nonzero(mask)),
            "history_std_mean": float(np.mean(history_std[mask])),
            "history_std_median": float(np.median(history_std[mask])),
            **_error_comparison(
                targets[mask],
                baseline_predictions[mask],
                candidate_predictions[mask],
            ),
        }
        for uid in sorted(set(str(value) for value in uids))
        for mask in (uids == uid,)
    }
    by_history_std_quartile = [
        {
            "quartile": name,
            "windows": int(indices.size),
            "history_std_min": float(np.min(history_std[indices])),
            "history_std_mean": float(np.mean(history_std[indices])),
            "history_std_max": float(np.max(history_std[indices])),
            **_error_comparison(
                targets[indices],
                baseline_predictions[indices],
                candidate_predictions[indices],
            ),
        }
        for name, indices in _history_std_quartiles(history_std)
    ]
    return {
        "overall": _error_comparison(
            targets,
            baseline_predictions,
            candidate_predictions,
        ),
        "history_std": {
            "min": float(np.min(history_std)),
            "median": float(np.median(history_std)),
            "mean": float(np.mean(history_std)),
            "p90": float(np.quantile(history_std, 0.90)),
            "max": float(np.max(history_std)),
        },
        "window_mae_delta_correlation_with_history_std": _safe_pearson(
            history_std,
            window_mae_delta,
        ),
        "by_horizon": by_horizon,
        "by_series": by_series,
        "by_history_std_quartile": by_history_std_quartile,
    }


def _effect_statistics(
    effect: np.ndarray,
    history_std: np.ndarray,
) -> dict[str, float | None]:
    absolute = np.abs(effect)
    per_window_absolute = np.mean(absolute, axis=1)
    normalized_per_window = per_window_absolute / np.maximum(history_std, 1e-12)
    return {
        "mean_signed": float(np.mean(effect)),
        "mean_absolute": float(np.mean(absolute)),
        "median_absolute": float(np.median(absolute)),
        "p90_absolute": float(np.quantile(absolute, 0.90)),
        "p95_absolute": float(np.quantile(absolute, 0.95)),
        "p99_absolute": float(np.quantile(absolute, 0.99)),
        "max_absolute": float(np.max(absolute)),
        "mean_absolute_in_history_std_units": float(
            np.mean(normalized_per_window)
        ),
        "p95_absolute_in_history_std_units": float(
            np.quantile(normalized_per_window, 0.95)
        ),
        "window_absolute_effect_correlation_with_history_std": _safe_pearson(
            history_std,
            per_window_absolute,
        ),
    }


def _effect_breakdown(
    effect: np.ndarray,
    *,
    history_std: np.ndarray,
    uids: np.ndarray,
) -> dict[str, Any]:
    return {
        "overall": _effect_statistics(effect, history_std),
        "by_horizon": [
            {
                "horizon": horizon + 1,
                **_effect_statistics(
                    effect[:, horizon : horizon + 1],
                    history_std,
                ),
            }
            for horizon in range(effect.shape[1])
        ],
        "by_series": {
            uid: {
                "windows": int(np.count_nonzero(mask)),
                "history_std_mean": float(np.mean(history_std[mask])),
                **_effect_statistics(effect[mask], history_std[mask]),
            }
            for uid in sorted(set(str(value) for value in uids))
            for mask in (uids == uid,)
        },
        "by_history_std_quartile": [
            {
                "quartile": name,
                "windows": int(indices.size),
                "history_std_min": float(np.min(history_std[indices])),
                "history_std_mean": float(np.mean(history_std[indices])),
                "history_std_max": float(np.max(history_std[indices])),
                **_effect_statistics(effect[indices], history_std[indices]),
            }
            for name, indices in _history_std_quartiles(history_std)
        ],
    }


def _future_shift_diagnostics(
    full: dict[str, Any],
    without_future_shift: dict[str, Any],
    zero_future: dict[str, Any],
) -> dict[str, Any]:
    _validate_diagnostic_payloads(without_future_shift, full)
    _validate_diagnostic_payloads(without_future_shift, zero_future)
    history_std = full["history_std"]
    uids = full["uids"]
    total_effect = full["predictions"] - without_future_shift["predictions"]
    feature_effect = full["predictions"] - zero_future["predictions"]
    zero_input_bias_effect = (
        zero_future["predictions"] - without_future_shift["predictions"]
    )
    return {
        "without_future_shift_prediction_sha256": _array_sha256(
            without_future_shift["predictions"]
        ),
        "zero_future_prediction_sha256": _array_sha256(
            zero_future["predictions"]
        ),
        "error_comparison": _paired_error_diagnostics(
            without_future_shift,
            full,
        ),
        "total_effect": _effect_breakdown(
            total_effect,
            history_std=history_std,
            uids=uids,
        ),
        "feature_conditioned_effect": _effect_breakdown(
            feature_effect,
            history_std=history_std,
            uids=uids,
        ),
        "zero_input_bias_effect": _effect_breakdown(
            zero_input_bias_effect,
            history_std=history_std,
            uids=uids,
        ),
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


def _candidate_summary(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    baseline_name: str,
    candidate_name: str,
) -> dict[str, Any]:
    if not np.array_equal(baseline["targets"], candidate["targets"]):
        raise RuntimeError("Ablation evaluations do not share identical targets.")
    if not np.array_equal(baseline["uids"], candidate["uids"]):
        raise RuntimeError("Ablation evaluations do not share identical series ordering.")

    targets = baseline["targets"]
    baseline_predictions = baseline["predictions"]
    candidate_predictions = candidate["predictions"]
    baseline_absolute_error = np.abs(baseline_predictions - targets)
    candidate_absolute_error = np.abs(candidate_predictions - targets)
    baseline_metrics = _metrics(targets, baseline_predictions)
    candidate_metrics = _metrics(targets, candidate_predictions)
    lower_is_better = ("mae", "mse", "rmse", "smape", "wape")
    return {
        "baseline": baseline_name,
        "candidate": candidate_name,
        "candidate_relative_improvement_pct": {
            name: float(
                100.0
                * (baseline_metrics[name] - candidate_metrics[name])
                / max(abs(baseline_metrics[name]), 1e-12)
            )
            for name in lower_is_better
        },
        "pointwise_absolute_error_win_rate": {
            baseline_name: float(
                np.mean(baseline_absolute_error < candidate_absolute_error)
            ),
            candidate_name: float(
                np.mean(candidate_absolute_error < baseline_absolute_error)
            ),
            "tie": float(
                np.mean(candidate_absolute_error == baseline_absolute_error)
            ),
        },
        "overall_mae_winner": (
            candidate_name
            if candidate_metrics["mae"] < baseline_metrics["mae"]
            else baseline_name
            if baseline_metrics["mae"] < candidate_metrics["mae"]
            else "tie"
        ),
    }


def _candidate_comparison_group(
    predictions: dict[str, dict[str, dict[str, Any]]],
    comparison_pairs: dict[str, tuple[str, str]],
    *,
    include_diagnostics: bool = False,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for comparison_name, (baseline_key, candidate_key) in comparison_pairs.items():
        baseline_name = baseline_key.removeprefix("patchmixer_")
        candidate_name = candidate_key.removeprefix("patchmixer_")
        output[comparison_name] = {
            "test_all_rolling_windows": _candidate_summary(
                predictions[baseline_key]["all"],
                predictions[candidate_key]["all"],
                baseline_name=baseline_name,
                candidate_name=candidate_name,
            ),
            "test_last_origin_per_series": _candidate_summary(
                predictions[baseline_key]["last"],
                predictions[candidate_key]["last"],
                baseline_name=baseline_name,
                candidate_name=candidate_name,
            ),
        }
        if include_diagnostics:
            output[comparison_name]["diagnostics"] = {
                "test_all_rolling_windows": _paired_error_diagnostics(
                    predictions[baseline_key]["all"],
                    predictions[candidate_key]["all"],
                ),
                "test_last_origin_per_series": _paired_error_diagnostics(
                    predictions[baseline_key]["last"],
                    predictions[candidate_key]["last"],
                ),
            }
    return output


def _input_ablation_summary(
    full: dict[str, Any],
    ablated: dict[str, Any],
) -> dict[str, Any]:
    if not np.array_equal(full["targets"], ablated["targets"]):
        raise RuntimeError("Input ablation does not share identical targets.")
    if not np.array_equal(full["uids"], ablated["uids"]):
        raise RuntimeError("Input ablation does not share identical series ordering.")

    targets = full["targets"]
    full_predictions = full["predictions"]
    ablated_predictions = ablated["predictions"]
    full_metrics = _metrics(targets, full_predictions)
    ablated_metrics = _metrics(targets, ablated_predictions)
    lower_is_better = ("mae", "mse", "rmse", "smape", "wape")
    return {
        "ablated_metrics": ablated_metrics,
        "relative_error_degradation_pct": {
            name: float(
                100.0
                * (ablated_metrics[name] - full_metrics[name])
                / max(abs(full_metrics[name]), 1e-12)
            )
            for name in lower_is_better
        },
        "prediction_mean_absolute_delta": float(
            np.mean(np.abs(full_predictions - ablated_predictions))
        ),
    }


@torch.no_grad()
def _gate_statistics(
    model: torch.nn.Module,
    case: ModelCase,
    loader: DataLoader,
    *,
    autocast_context: Callable[[], Any],
) -> dict[str, float | int]:
    gate_module = getattr(model, "_z_gate", None)
    if not isinstance(gate_module, torch.nn.Module):
        raise RuntimeError(f"{case.key} does not expose a trainable _z_gate module.")

    count = 0
    value_sum = 0.0
    square_sum = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    below_005 = 0
    above_095 = 0

    def collect_gate(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        nonlocal count, value_sum, square_sum, minimum, maximum
        nonlocal below_005, above_095
        values = torch.sigmoid(output.detach().float())
        count += values.numel()
        value_sum += float(values.sum())
        square_sum += float(values.square().sum())
        minimum = min(minimum, float(values.min()))
        maximum = max(maximum, float(values.max()))
        below_005 += int(torch.count_nonzero(values < 0.05))
        above_095 += int(torch.count_nonzero(values > 0.95))

    handle = gate_module.register_forward_hook(collect_gate)
    try:
        model.eval()
        for batch in loader:
            with autocast_context():
                _forward_batch(model, case, batch)
    finally:
        handle.remove()

    if count == 0:
        raise RuntimeError(f"{case.key} gate statistics collected no activations.")
    mean = value_sum / count
    variance = max(0.0, square_sum / count - mean * mean)
    return {
        "count": count,
        "mean": mean,
        "population_stddev": math.sqrt(variance),
        "min": minimum,
        "max": maximum,
        "fraction_below_0_05": below_005 / count,
        "fraction_above_0_95": above_095 / count,
    }


def _run_accuracy_seed(
    frame: pl.DataFrame,
    *,
    cases: tuple[ModelCase, ...],
    case_set: str,
    comparison_pairs: dict[str, tuple[str, str]],
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
    validation_gate_payloads: dict[str, dict[str, dict[str, Any]]] = {}
    for case in cases:
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
        if (
            case_set == "patchmixer-shift-space"
            and case.key in HISTORY_GATE_REFERENCE_KEYS
        ):
            validation_loader = _make_loader(
                dataset,
                validation_indices,
                batch_size=batch_size,
                shuffle=False,
                seed=seed + 2,
            )
            validation_payload = _predict_loader(
                model,
                case,
                validation_loader,
                autocast_context=autocast_context,
            )
            validation_gate_payloads[case.key] = {
                "full": validation_payload,
            }
            model_results[case.key]["validation_all_rolling_windows"] = (
                _prediction_summary(validation_payload)
            )
            if case.key == "patchmixer_future_shift_normalized":
                validation_gate_payloads[case.key]["without_future_shift"] = (
                    _predict_loader(
                        model,
                        case,
                        validation_loader,
                        autocast_context=autocast_context,
                        omit_future=True,
                    )
                )
            del validation_loader
        if case_set == "patchmixer-shift-space" and case.future_exogenous:
            without_future_all = _predict_loader(
                model,
                case,
                all_loader,
                autocast_context=autocast_context,
                omit_future=True,
            )
            without_future_last = _predict_loader(
                model,
                case,
                last_loader,
                autocast_context=autocast_context,
                omit_future=True,
            )
            zero_future_all = _predict_loader(
                model,
                case,
                all_loader,
                autocast_context=autocast_context,
                zero_future=True,
            )
            zero_future_last = _predict_loader(
                model,
                case,
                last_loader,
                autocast_context=autocast_context,
                zero_future=True,
            )
            model_results[case.key]["future_shift_diagnostics"] = {
                "test_all_rolling_windows": _future_shift_diagnostics(
                    all_payload,
                    without_future_all,
                    zero_future_all,
                ),
                "test_last_origin_per_series": _future_shift_diagnostics(
                    last_payload,
                    without_future_last,
                    zero_future_last,
                ),
            }
        if case.key == "patchmixer_exogenous":
            input_ablations: dict[str, Any] = {}
            for ablation_name, zero_past, zero_future in (
                ("zero_past", True, False),
                ("zero_future", False, True),
                ("zero_all", True, True),
            ):
                ablated_all = _predict_loader(
                    model,
                    case,
                    all_loader,
                    autocast_context=autocast_context,
                    zero_past=zero_past,
                    zero_future=zero_future,
                )
                ablated_last = _predict_loader(
                    model,
                    case,
                    last_loader,
                    autocast_context=autocast_context,
                    zero_past=zero_past,
                    zero_future=zero_future,
                )
                input_ablations[ablation_name] = {
                    "test_all_rolling_windows": {
                        "prediction": _prediction_summary(ablated_all),
                        "comparison_to_full": _input_ablation_summary(
                            all_payload,
                            ablated_all,
                        ),
                    },
                    "test_last_origin_per_series": {
                        "prediction": _prediction_summary(ablated_last),
                        "comparison_to_full": _input_ablation_summary(
                            last_payload,
                            ablated_last,
                        ),
                    },
                }
            model_results[case.key]["input_ablations"] = input_ablations
            model_results[case.key]["gate_statistics"] = {
                "test_all_rolling_windows": _gate_statistics(
                    model,
                    case,
                    all_loader,
                    autocast_context=autocast_context,
                )
            }
        predictions[case.key] = {"all": all_payload, "last": last_payload}
        del model, all_loader, last_loader
        gc.collect()
        torch.cuda.empty_cache()

    paired: dict[str, Any] = {}
    validation_gate_upper_bound: dict[str, Any] | None = None
    if case_set == "all":
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
        paired["patchmixer_ablation"] = _candidate_comparison_group(
            predictions,
            comparison_pairs,
        )
    else:
        paired["patchmixer_shift_space"] = _candidate_comparison_group(
            predictions,
            comparison_pairs,
            include_diagnostics=True,
        )
        output_validation = validation_gate_payloads[
            "patchmixer_future_shift"
        ]["full"]
        normalized_validation = validation_gate_payloads[
            "patchmixer_future_shift_normalized"
        ]
        validation_gate_upper_bound = (
            _history_conditioned_gate_validation_upper_bound(
                output_validation,
                normalized_validation["full"],
                normalized_validation["without_future_shift"],
            )
        )

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
        "validation_gate_upper_bound": validation_gate_upper_bound,
    }


def _aggregate_candidate_comparisons(
    seed_results: list[dict[str, Any]],
    *,
    group_name: str,
    comparison_pairs: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for comparison_name in comparison_pairs:
        output[comparison_name] = {}
        for evaluation in (
            "test_all_rolling_windows",
            "test_last_origin_per_series",
        ):
            records = []
            for result in seed_results:
                comparison = result["paired_comparison"][group_name][
                    comparison_name
                ][evaluation]
                records.append(
                    {
                        "seed": result["seed"],
                        "baseline": comparison["baseline"],
                        "candidate": comparison["candidate"],
                        "winner": comparison["overall_mae_winner"],
                        "mae_improvement_pct": comparison[
                            "candidate_relative_improvement_pct"
                        ]["mae"],
                    }
                )
            improvements = [record["mae_improvement_pct"] for record in records]
            winner_names = {
                records[0]["baseline"],
                records[0]["candidate"],
                "tie",
            }
            output[comparison_name][evaluation] = {
                "records": records,
                "seed_wins": {
                    name: sum(record["winner"] == name for record in records)
                    for name in sorted(winner_names)
                },
                "mae_improvement_pct": {
                    "mean": statistics.fmean(improvements),
                    "population_stddev": statistics.pstdev(improvements),
                    "min": min(improvements),
                    "max": max(improvements),
                },
            }
    return output


def _aggregate_validation_gate_upper_bound(
    seed_results: list[dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for analysis_name in (
        "normalized_residual_gate",
        "output_to_normalized_blend",
    ):
        output[analysis_name] = {}
        for granularity in ("window_scalar", "window_horizon"):
            output[analysis_name][granularity] = {}
            for evaluation in (
                "validation_all_rolling_windows",
                "validation_last_origin_per_series",
            ):
                first_methods = seed_results[0]["validation_gate_upper_bound"][
                    analysis_name
                ][granularity]["evaluations"][evaluation]["methods"]
                method_output: dict[str, Any] = {}
                for method_name in first_methods:
                    records = []
                    for result in seed_results:
                        row = result["validation_gate_upper_bound"][analysis_name][
                            granularity
                        ]["evaluations"][evaluation]["methods"][method_name]
                        records.append(
                            {
                                "seed": result["seed"],
                                "mae": row["metrics"]["mae"],
                                "mse": row["metrics"]["mse"],
                                "mae_improvement_pct_vs_always_on": row[
                                    "relative_improvement_pct_vs_always_on"
                                ]["mae"],
                                "mse_improvement_pct_vs_always_on": row[
                                    "relative_improvement_pct_vs_always_on"
                                ]["mse"],
                                "mse_oracle_gain_capture_pct_from_always_on": row[
                                    "mse_oracle_gain_capture_pct_from_always_on"
                                ],
                                "mae_improvement_pct_vs_output_reference": (
                                    row.get(
                                        "relative_improvement_pct_vs_output_reference",
                                        {},
                                    ).get("mae")
                                ),
                                "mse_improvement_pct_vs_output_reference": (
                                    row.get(
                                        "relative_improvement_pct_vs_output_reference",
                                        {},
                                    ).get("mse")
                                ),
                            }
                        )
                    method_output[method_name] = {
                        "records": records,
                        "mean_mae": statistics.fmean(
                            record["mae"] for record in records
                        ),
                        "mean_mse": statistics.fmean(
                            record["mse"] for record in records
                        ),
                        "mean_mae_improvement_pct_vs_always_on": statistics.fmean(
                            record["mae_improvement_pct_vs_always_on"]
                            for record in records
                        ),
                        "mean_mse_improvement_pct_vs_always_on": statistics.fmean(
                            record["mse_improvement_pct_vs_always_on"]
                            for record in records
                        ),
                    }
                    if all(
                        record["mae_improvement_pct_vs_output_reference"]
                        is not None
                        for record in records
                    ):
                        method_output[method_name][
                            "mean_mae_improvement_pct_vs_output_reference"
                        ] = statistics.fmean(
                            float(record["mae_improvement_pct_vs_output_reference"])
                            for record in records
                        )
                        method_output[method_name][
                            "mean_mse_improvement_pct_vs_output_reference"
                        ] = statistics.fmean(
                            float(record["mse_improvement_pct_vs_output_reference"])
                            for record in records
                        )
                output[analysis_name][granularity][evaluation] = method_output
    return output


def _aggregate_accuracy(
    seed_results: list[dict[str, Any]],
    *,
    case_set: str,
    comparison_pairs: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    if case_set == "patchmixer-shift-space":
        output = {
            "patchmixer_shift_space": _aggregate_candidate_comparisons(
                seed_results,
                group_name="patchmixer_shift_space",
                comparison_pairs=comparison_pairs,
            )
        }
        if all(
            result.get("validation_gate_upper_bound") is not None
            for result in seed_results
        ):
            output["validation_gate_upper_bound"] = (
                _aggregate_validation_gate_upper_bound(seed_results)
            )
        return output

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

    output["patchmixer_ablation"] = _aggregate_candidate_comparisons(
        seed_results,
        group_name="patchmixer_ablation",
        comparison_pairs=comparison_pairs,
    )

    output["patchmixer_input_ablation"] = {}
    for ablation_name in ("zero_past", "zero_future", "zero_all"):
        output["patchmixer_input_ablation"][ablation_name] = {}
        for evaluation in (
            "test_all_rolling_windows",
            "test_last_origin_per_series",
        ):
            records = [
                {
                    "seed": result["seed"],
                    "mae_degradation_pct": result["models"][
                        "patchmixer_exogenous"
                    ]["input_ablations"][ablation_name][evaluation][
                        "comparison_to_full"
                    ]["relative_error_degradation_pct"]["mae"],
                    "prediction_mean_absolute_delta": result["models"][
                        "patchmixer_exogenous"
                    ]["input_ablations"][ablation_name][evaluation][
                        "comparison_to_full"
                    ]["prediction_mean_absolute_delta"],
                }
                for result in seed_results
            ]
            degradations = [record["mae_degradation_pct"] for record in records]
            output["patchmixer_input_ablation"][ablation_name][evaluation] = {
                "records": records,
                "mae_degradation_pct": {
                    "mean": statistics.fmean(degradations),
                    "population_stddev": statistics.pstdev(degradations),
                    "min": min(degradations),
                    "max": max(degradations),
                },
            }

    gate_records = [
        {
            "seed": result["seed"],
            **result["models"]["patchmixer_exogenous"]["gate_statistics"][
                "test_all_rolling_windows"
            ],
        }
        for result in seed_results
    ]
    output["patchmixer_gate_statistics"] = {
        "records": gate_records,
        "mean_activation": statistics.fmean(
            float(record["mean"]) for record in gate_records
        ),
        "mean_fraction_below_0_05": statistics.fmean(
            float(record["fraction_below_0_05"]) for record in gate_records
        ),
        "mean_fraction_above_0_95": statistics.fmean(
            float(record["fraction_above_0_95"]) for record in gate_records
        ),
    }
    return output


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _make_cuda_batch(
    case: ModelCase,
    *,
    batch_size: int,
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    inputs = torch.randn(
        batch_size,
        LOOKBACK,
        1,
        device="cuda",
        generator=generator,
    )
    noise = 0.05 * torch.randn(
        batch_size,
        HORIZON,
        device="cuda",
        generator=generator,
    )
    targets = inputs[:, -1, 0].unsqueeze(1) + noise
    past = (
        torch.randn(
            batch_size,
            LOOKBACK,
            len(PAST_EXOGENOUS_COLUMNS),
            device="cuda",
            generator=generator,
        )
        if case.past_exogenous
        else None
    )
    future = (
        torch.randn(
            batch_size,
            HORIZON,
            len(FUTURE_EXOGENOUS_COLUMNS),
            device="cuda",
            generator=generator,
        )
        if case.future_exogenous
        else None
    )
    return inputs, targets, past, future


def _forward_cuda_batch(
    model: torch.nn.Module,
    case: ModelCase,
    inputs: torch.Tensor,
    past: torch.Tensor | None,
    future: torch.Tensor | None,
) -> torch.Tensor:
    kwargs: dict[str, torch.Tensor] = {}
    if case.past_exogenous:
        assert past is not None
        kwargs["past_exo_cont"] = past
    if case.future_exogenous:
        assert future is not None
        kwargs["future_exo"] = future
    return _point_prediction(model(inputs, **kwargs))


def _timing_payload(
    starts: list[torch.cuda.Event],
    ends: list[torch.cuda.Event],
    *,
    batch_size: int,
) -> tuple[dict[str, float], dict[str, float]]:
    times = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    total = sum(times)
    mean = statistics.fmean(times)
    return (
        {
            "total": total,
            "mean": mean,
            "median": statistics.median(times),
            "p95": _percentile(times, 0.95),
            "min": min(times),
            "max": max(times),
            "population_stddev": statistics.pstdev(times),
        },
        {
            "steps_per_second": 1000.0 / mean,
            "samples_per_second": batch_size * len(times) * 1000.0 / total,
        },
    )


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
        case,
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
            prediction = _forward_cuda_batch(
                model,
                case,
                inputs,
                past,
                future,
            )
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
    training_timing, training_throughput = _timing_payload(
        starts,
        ends,
        batch_size=batch_size,
    )
    result = {
        "model": case.key,
        "parameters": parameter_count,
        "timing_ms": training_timing,
        "throughput": training_throughput,
        "memory_mib": {
            "peak_allocated": torch.cuda.max_memory_allocated() / (1024**2),
            "peak_reserved": torch.cuda.max_memory_reserved() / (1024**2),
        },
        "loss": {"first_measured": float(first_loss), "last_measured": float(last_loss)},
    }

    del optimizer, starts, ends, first_loss, last_loss
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    model.eval()

    def inference_step() -> torch.Tensor:
        with autocast_context():
            return _forward_cuda_batch(model, case, inputs, past, future)

    with torch.inference_mode():
        for _ in range(warmup_steps):
            inference_step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        inference_starts = [
            torch.cuda.Event(enable_timing=True) for _ in range(steps)
        ]
        inference_ends = [
            torch.cuda.Event(enable_timing=True) for _ in range(steps)
        ]
        for start, end in zip(inference_starts, inference_ends):
            start.record()
            inference_step()
            end.record()
        torch.cuda.synchronize()

    inference_timing, inference_throughput = _timing_payload(
        inference_starts,
        inference_ends,
        batch_size=batch_size,
    )
    result["inference"] = {
        "timing_ms": inference_timing,
        "throughput": inference_throughput,
        "memory_mib": {
            "peak_allocated": torch.cuda.max_memory_allocated() / (1024**2),
            "peak_reserved": torch.cuda.max_memory_reserved() / (1024**2),
        },
    }

    del model, inputs, targets, past, future, inference_starts, inference_ends
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return result


def _performance_delta(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_training_time = float(baseline["timing_ms"]["mean"])
    candidate_training_time = float(candidate["timing_ms"]["mean"])
    baseline_inference_time = float(baseline["inference"]["timing_ms"]["mean"])
    candidate_inference_time = float(candidate["inference"]["timing_ms"]["mean"])
    return {
        "baseline": baseline["model"],
        "candidate": candidate["model"],
        "training_step_time_overhead_pct": (
            100.0
            * (candidate_training_time - baseline_training_time)
            / baseline_training_time
        ),
        "training_throughput_ratio": (
            candidate["throughput"]["samples_per_second"]
            / baseline["throughput"]["samples_per_second"]
        ),
        "inference_step_time_overhead_pct": (
            100.0
            * (candidate_inference_time - baseline_inference_time)
            / baseline_inference_time
        ),
        "inference_throughput_ratio": (
            candidate["inference"]["throughput"]["samples_per_second"]
            / baseline["inference"]["throughput"]["samples_per_second"]
        ),
        "parameter_overhead": candidate["parameters"] - baseline["parameters"],
        "training_peak_allocated_overhead_mib": (
            candidate["memory_mib"]["peak_allocated"]
            - baseline["memory_mib"]["peak_allocated"]
        ),
        "inference_peak_allocated_overhead_mib": (
            candidate["inference"]["memory_mib"]["peak_allocated"]
            - baseline["inference"]["memory_mib"]["peak_allocated"]
        ),
    }


def _performance_summary(
    results: list[dict[str, Any]],
    *,
    case_set: str,
    comparison_pairs: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    by_key = {result["model"]: result for result in results}
    if case_set == "patchmixer-shift-space":
        return {
            "patchmixer_shift_space": {
                comparison_name: _performance_delta(
                    by_key[baseline_key],
                    by_key[candidate_key],
                )
                for comparison_name, (baseline_key, candidate_key) in (
                    comparison_pairs.items()
                )
            }
        }

    output: dict[str, Any] = {}
    for family in ("patchtst", "patchmixer"):
        delta = _performance_delta(
            by_key[f"{family}_endogenous"],
            by_key[f"{family}_exogenous"],
        )
        output[family] = {
            **delta,
            "exogenous_step_time_overhead_pct": delta[
                "training_step_time_overhead_pct"
            ],
            "exogenous_throughput_ratio": delta["training_throughput_ratio"],
            "peak_allocated_overhead_mib": delta[
                "training_peak_allocated_overhead_mib"
            ],
        }

    output["patchmixer_ablation"] = {
        comparison_name: _performance_delta(
            by_key[baseline_key],
            by_key[candidate_key],
        )
        for comparison_name, (baseline_key, candidate_key) in (
            comparison_pairs.items()
        )
    }
    return output


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.case_set == "patchmixer-shift-space":
        cases = PATCHMIXER_SHIFT_SPACE_CASES
        comparison_pairs = PATCHMIXER_SHIFT_SPACE_PAIRS
    else:
        cases = MODEL_CASES
        comparison_pairs = PATCHMIXER_ABLATION_PAIRS
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
            cases=cases,
            case_set=args.case_set,
            comparison_pairs=comparison_pairs,
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
        for case in cases
    ]
    properties = torch.cuda.get_device_properties(0)
    result = {
        "schema_version": 6,
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
            "future_availability_assumption": (
                "Every future exogenous value is assumed available or forecast at origin; "
                "the benchmark does not measure upstream feature-forecast error."
            ),
        },
        "accuracy_protocol": {
            "case_set": args.case_set,
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
            "model_cases": [_case_metadata(case) for case in cases],
            "comparison_pairs": comparison_pairs,
            "diagnostics": (
                [
                    "future_shift_total_effect",
                    "future_shift_feature_conditioned_effect",
                    "future_shift_zero_input_bias_effect",
                    "error_by_series",
                    "error_by_horizon",
                    "error_by_history_std_rank_quartile",
                    "validation_history_gate_nested_series_oof",
                    "validation_history_gate_target_aware_oracle_ceiling",
                ]
                if args.case_set == "patchmixer-shift-space"
                else []
            ),
            "full_model_input_ablations": (
                ["zero_past", "zero_future", "zero_all"]
                if args.case_set == "all"
                else []
            ),
        },
        "accuracy_seeds": seed_results,
        "accuracy_aggregate": _aggregate_accuracy(
            seed_results,
            case_set=args.case_set,
            comparison_pairs=comparison_pairs,
        ),
        "performance_protocol": {
            "seed": performance_seed,
            "precision": args.performance_precision,
            "batch_size": args.batch_size,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.performance_steps,
            "modes": ["training_step", "inference"],
            "cuda_resident_batch": True,
            "only_configured_exogenous_tensors_allocated": True,
            "data_loader_included": False,
            "host_to_device_transfer_included": False,
        },
        "performance_models": performance_results,
        "performance_summary": _performance_summary(
            performance_results,
            case_set=args.case_set,
            comparison_pairs=comparison_pairs,
        ),
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

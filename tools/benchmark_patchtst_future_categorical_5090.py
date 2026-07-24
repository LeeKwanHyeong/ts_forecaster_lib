#!/usr/bin/env python3
"""Compare continuous-only and categorical PatchTST paths on an RTX 5090."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from contextlib import nullcontext, redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modeling_module.models.PatchTST.common.configs import (  # noqa: E402
    AttentionConfig,
    PatchTSTConfig,
)
from modeling_module.models.model_builder import (  # noqa: E402
    build_patchTST_exogenous,
)


LOOKBACK = 52
HORIZON = 27
FUTURE_CONTINUOUS_DIM = 2
FUTURE_CATEGORICAL_CARDINALITIES = (5, 4)
MODEL_CASES = ("continuous_only", "continuous_and_categorical")


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=_positive_int, default=1000)
    parser.add_argument("--warmup-steps", type=_nonnegative_int, default=100)
    parser.add_argument("--inference-steps", type=_positive_int, default=200)
    parser.add_argument("--batch-size", type=_positive_int, default=128)
    parser.add_argument("--train-samples", type=_positive_int, default=4096)
    parser.add_argument("--validation-samples", type=_positive_int, default=1024)
    parser.add_argument(
        "--precision",
        choices=("bf16", "float32"),
        default="bf16",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--expected-device",
        default="NVIDIA GeForce RTX 5090",
    )
    parser.add_argument("--source-branch")
    parser.add_argument("--source-commit")
    parser.add_argument(
        "--source-working-tree-dirty",
        choices=("true", "false", "unknown"),
        default="unknown",
    )
    parser.add_argument("--output", type=Path)
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
        "working_tree_dirty": None if status is None else bool(status),
    }


def _source_snapshot_sha256() -> str:
    digest = hashlib.sha256()
    paths = sorted(SRC.rglob("*.py"))
    paths.append(Path(__file__).resolve())
    for path in paths:
        relative_path = path.relative_to(ROOT).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _stabilize_cuda() -> None:
    """Warm CUDA kernels and clocks before the first measured model."""
    generator = torch.Generator(device="cuda").manual_seed(20260724)
    left = torch.randn(
        4096,
        4096,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    right = torch.randn(
        4096,
        4096,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    for _ in range(20):
        torch.mm(left, right)
    torch.cuda.synchronize()
    del left, right
    torch.cuda.empty_cache()


def _autocast_factory(precision: str) -> Callable[[], Any]:
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected CUDA device does not support BF16.")
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _timing_payload(
    starts: list[torch.cuda.Event],
    ends: list[torch.cuda.Event],
    *,
    batch_size: int,
) -> tuple[dict[str, float], dict[str, float]]:
    elapsed = [
        start.elapsed_time(end)
        for start, end in zip(starts, ends)
    ]
    total = sum(elapsed)
    mean = statistics.fmean(elapsed)
    return (
        {
            "total": total,
            "mean": mean,
            "median": statistics.median(elapsed),
            "p95": _percentile(elapsed, 0.95),
            "min": min(elapsed),
            "max": max(elapsed),
            "population_stddev": statistics.pstdev(elapsed),
        },
        {
            "steps_per_second": 1000.0 / mean,
            "samples_per_second": (
                batch_size * len(elapsed) * 1000.0 / total
            ),
        },
    )


def _build_config(case: str) -> PatchTSTConfig:
    if case not in MODEL_CASES:
        raise ValueError(f"Unsupported benchmark case: {case!r}.")
    categorical = case == "continuous_and_categorical"
    return PatchTSTConfig(
        lookback=LOOKBACK,
        horizon=HORIZON,
        c_in=1,
        patch_len=8,
        stride=4,
        padding_patch="end",
        future_exo_dim=FUTURE_CONTINUOUS_DIM,
        future_exo_cat_cardinalities=(
            FUTURE_CATEGORICAL_CARDINALITIES
            if categorical
            else ()
        ),
        future_exo_cat_embedding_dim=8,
        future_exo_fusion_dropout=0.0,
        d_model=128,
        n_layers=3,
        d_ff=256,
        norm="LayerNorm",
        dropout=0.0,
        pre_norm=True,
        use_revin=True,
        attn=AttentionConfig(
            n_heads=8,
            d_model=128,
            attn_dropout=0.0,
            proj_dropout=0.0,
        ),
    )


def _build_model(case: str) -> torch.nn.Module:
    with redirect_stdout(StringIO()):
        return build_patchTST_exogenous(_build_config(case))


def _make_dataset(
    sample_count: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    time = torch.arange(
        LOOKBACK,
        device="cuda",
        dtype=torch.float32,
    )
    base = 5.0 + 1.5 * torch.randn(
        sample_count,
        1,
        device="cuda",
        generator=generator,
    )
    slope = 0.01 * torch.randn(
        sample_count,
        1,
        device="cuda",
        generator=generator,
    )
    phase = 2.0 * math.pi * torch.rand(
        sample_count,
        1,
        device="cuda",
        generator=generator,
    )
    seasonal = torch.sin(
        time.unsqueeze(0) * (2.0 * math.pi / 13.0) + phase
    )
    history = (
        base
        + slope * time.unsqueeze(0)
        + 0.4 * seasonal
        + 0.05
        * torch.randn(
            sample_count,
            LOOKBACK,
            device="cuda",
            generator=generator,
        )
    ).unsqueeze(-1)

    future_continuous = torch.randn(
        sample_count,
        HORIZON,
        FUTURE_CONTINUOUS_DIM,
        device="cuda",
        generator=generator,
    )
    promo = torch.randint(
        1,
        FUTURE_CATEGORICAL_CARDINALITIES[0],
        (sample_count, HORIZON),
        device="cuda",
        generator=generator,
    )
    holiday = torch.randint(
        1,
        FUTURE_CATEGORICAL_CARDINALITIES[1],
        (sample_count, HORIZON),
        device="cuda",
        generator=generator,
    )
    future_categorical = torch.stack((promo, holiday), dim=-1)

    horizon_axis = torch.arange(
        1,
        HORIZON + 1,
        device="cuda",
        dtype=torch.float32,
    ).unsqueeze(0)
    promo_effects = torch.tensor(
        [0.0, -0.8, 0.3, 1.0, 1.8],
        device="cuda",
    )
    holiday_effects = torch.tensor(
        [0.0, -0.4, 0.6, 1.2],
        device="cuda",
    )
    target = (
        history[:, -1, 0].unsqueeze(1)
        + slope * horizon_axis
        + 0.45 * future_continuous[..., 0]
        - 0.30 * future_continuous[..., 1]
        + promo_effects[promo]
        + holiday_effects[holiday]
        + 0.05
        * torch.randn(
            sample_count,
            HORIZON,
            device="cuda",
            generator=generator,
        )
    )
    return history, future_continuous, future_categorical, target


def _point_prediction(value: Any) -> torch.Tensor:
    if isinstance(value, dict):
        value = value.get("pred", value.get("point"))
    if not torch.is_tensor(value):
        raise TypeError(
            f"Expected tensor point output, got {type(value).__name__}."
        )
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if value.shape != (value.shape[0], HORIZON):
        raise RuntimeError(
            f"Expected [B,{HORIZON}], got {tuple(value.shape)}."
        )
    return value


def _accuracy_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    error = prediction.float() - target.float()
    denominator = prediction.float().abs() + target.float().abs()
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(error.square().mean().sqrt()),
        "smape_pct": float(
            (
                200.0
                * error.abs()
                / denominator.clamp_min(1e-6)
            ).mean()
        ),
    }


def _benchmark_case(
    case: str,
    *,
    seed: int,
    steps: int,
    warmup_steps: int,
    inference_steps: int,
    batch_size: int,
    train_samples: int,
    validation_samples: int,
    precision: str,
    lr: float,
    weight_decay: float,
) -> dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    _seed_everything(seed)

    train_data = _make_dataset(train_samples, seed=seed + 1000)
    validation_data = _make_dataset(
        validation_samples,
        seed=seed + 2000,
    )
    index_generator = torch.Generator(device="cuda").manual_seed(seed + 3000)
    batch_indices = [
        torch.randint(
            0,
            train_samples,
            (batch_size,),
            device="cuda",
            generator=index_generator,
        )
        for _ in range(warmup_steps + steps)
    ]

    _seed_everything(seed)
    model = _build_model(case).cuda().train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    autocast_context = _autocast_factory(precision)
    use_categorical = case == "continuous_and_categorical"

    def forward_batch(
        history: torch.Tensor,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
    ) -> torch.Tensor:
        return _point_prediction(
            model(
                history,
                future_exo=continuous,
                future_exo_cat=(
                    categorical if use_categorical else None
                ),
            )
        )

    def training_step(indices: torch.Tensor) -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            prediction = forward_batch(
                train_data[0][indices],
                train_data[1][indices],
                train_data[2][indices],
            )
            loss = F.l1_loss(prediction, train_data[3][indices])
        loss.backward()
        optimizer.step()
        return loss.detach()

    for indices in batch_indices[:warmup_steps]:
        training_step(indices)

    torch.cuda.synchronize()
    memory_before_training = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    starts = [
        torch.cuda.Event(enable_timing=True)
        for _ in range(steps)
    ]
    ends = [
        torch.cuda.Event(enable_timing=True)
        for _ in range(steps)
    ]
    first_loss: torch.Tensor | None = None
    last_loss: torch.Tensor | None = None
    for index, (indices, start, end) in enumerate(
        zip(batch_indices[warmup_steps:], starts, ends)
    ):
        start.record()
        loss = training_step(indices)
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
    peak_training_memory = torch.cuda.max_memory_allocated()

    optimizer.zero_grad(set_to_none=True)
    del optimizer, train_data, batch_indices
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    model.eval()
    inference_batch_size = min(batch_size, validation_samples)
    inference_batch = tuple(
        value[:inference_batch_size]
        for value in validation_data[:3]
    )
    with torch.inference_mode():
        for _ in range(warmup_steps):
            with autocast_context():
                forward_batch(*inference_batch)
        torch.cuda.synchronize()
        memory_before_inference = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        inference_starts = [
            torch.cuda.Event(enable_timing=True)
            for _ in range(inference_steps)
        ]
        inference_ends = [
            torch.cuda.Event(enable_timing=True)
            for _ in range(inference_steps)
        ]
        for start, end in zip(inference_starts, inference_ends):
            start.record()
            with autocast_context():
                forward_batch(*inference_batch)
            end.record()
        torch.cuda.synchronize()
        inference_timing, inference_throughput = _timing_payload(
            inference_starts,
            inference_ends,
            batch_size=inference_batch_size,
        )
        peak_inference_memory = torch.cuda.max_memory_allocated()

        predictions: list[torch.Tensor] = []
        for start_index in range(0, validation_samples, batch_size):
            stop_index = start_index + batch_size
            with autocast_context():
                predictions.append(
                    forward_batch(
                        validation_data[0][start_index:stop_index],
                        validation_data[1][start_index:stop_index],
                        validation_data[2][start_index:stop_index],
                    ).float()
                )
        validation_prediction = torch.cat(predictions, dim=0)

    result = {
        "case": case,
        "seed": seed,
        "parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
        "accuracy": _accuracy_metrics(
            validation_prediction,
            validation_data[3],
        ),
        "training": {
            "timing_ms": training_timing,
            "throughput": training_throughput,
            "loss": {
                "first_measured": float(first_loss),
                "last_measured": float(last_loss),
            },
            "memory_mib": {
                "allocated_before_step": (
                    memory_before_training / (1024**2)
                ),
                "peak_allocated": (
                    peak_training_memory / (1024**2)
                ),
                "peak_step_delta": (
                    (peak_training_memory - memory_before_training)
                    / (1024**2)
                ),
            },
        },
        "inference": {
            "timing_ms": inference_timing,
            "throughput": inference_throughput,
            "memory_mib": {
                "allocated_before_step": (
                    memory_before_inference / (1024**2)
                ),
                "peak_allocated": (
                    peak_inference_memory / (1024**2)
                ),
                "peak_step_delta": (
                    (peak_inference_memory - memory_before_inference)
                    / (1024**2)
                ),
            },
        },
    }

    del (
        model,
        validation_data,
        starts,
        ends,
        inference_starts,
        inference_ends,
        validation_prediction,
        predictions,
        first_loss,
        last_loss,
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return result


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _median(values: list[float]) -> float:
    return statistics.median(values)


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for case in MODEL_CASES:
        rows = [row for row in results if row["case"] == case]
        summary[case] = {
            "seeds": [row["seed"] for row in rows],
            "parameters": rows[0]["parameters"],
            "accuracy": {
                metric: _mean(
                    [row["accuracy"][metric] for row in rows]
                )
                for metric in ("mae", "rmse", "smape_pct")
            },
            "training": {
                "mean_step_ms": _mean(
                    [
                        row["training"]["timing_ms"]["mean"]
                        for row in rows
                    ]
                ),
                "median_seed_step_ms": _median(
                    [
                        row["training"]["timing_ms"]["mean"]
                        for row in rows
                    ]
                ),
                "samples_per_second": _mean(
                    [
                        row["training"]["throughput"][
                            "samples_per_second"
                        ]
                        for row in rows
                    ]
                ),
                "peak_allocated_mib": _mean(
                    [
                        row["training"]["memory_mib"]["peak_allocated"]
                        for row in rows
                    ]
                ),
                "peak_step_delta_mib": _mean(
                    [
                        row["training"]["memory_mib"]["peak_step_delta"]
                        for row in rows
                    ]
                ),
            },
            "inference": {
                "mean_step_ms": _mean(
                    [
                        row["inference"]["timing_ms"]["mean"]
                        for row in rows
                    ]
                ),
                "median_seed_step_ms": _median(
                    [
                        row["inference"]["timing_ms"]["mean"]
                        for row in rows
                    ]
                ),
                "samples_per_second": _mean(
                    [
                        row["inference"]["throughput"][
                            "samples_per_second"
                        ]
                        for row in rows
                    ]
                ),
                "peak_allocated_mib": _mean(
                    [
                        row["inference"]["memory_mib"]["peak_allocated"]
                        for row in rows
                    ]
                ),
                "peak_step_delta_mib": _mean(
                    [
                        row["inference"]["memory_mib"]["peak_step_delta"]
                        for row in rows
                    ]
                ),
            },
        }

    baseline = summary["continuous_only"]
    candidate = summary["continuous_and_categorical"]

    def percent_change(candidate_value: float, baseline_value: float) -> float:
        return 100.0 * (candidate_value - baseline_value) / baseline_value

    summary["categorical_delta"] = {
        "mae_change_pct": percent_change(
            candidate["accuracy"]["mae"],
            baseline["accuracy"]["mae"],
        ),
        "timing_delta_basis": "median of per-seed mean step times",
        "training_step_time_change_pct": percent_change(
            candidate["training"]["median_seed_step_ms"],
            baseline["training"]["median_seed_step_ms"],
        ),
        "inference_step_time_change_pct": percent_change(
            candidate["inference"]["median_seed_step_ms"],
            baseline["inference"]["median_seed_step_ms"],
        ),
        "parameter_change": (
            candidate["parameters"] - baseline["parameters"]
        ),
        "training_peak_allocated_change_mib": (
            candidate["training"]["peak_allocated_mib"]
            - baseline["training"]["peak_allocated_mib"]
        ),
        "inference_peak_allocated_change_mib": (
            candidate["inference"]["peak_allocated_mib"]
            - baseline["inference"]["peak_allocated_mib"]
        ),
    }
    return summary


def _validate_cuda(expected_device: str) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required; this benchmark must not run on CPU or MPS."
        )
    device_name = torch.cuda.get_device_name(0)
    if expected_device and device_name != expected_device:
        raise RuntimeError(
            f"Expected device {expected_device!r}, got {device_name!r}."
        )
    return device_name


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    device_name = _validate_cuda(args.expected_device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    _stabilize_cuda()

    properties = torch.cuda.get_device_properties(0)
    git_metadata = _git_metadata()
    if args.source_branch is not None:
        git_metadata["branch"] = args.source_branch
    if args.source_commit is not None:
        git_metadata["commit"] = args.source_commit
    if args.source_working_tree_dirty != "unknown":
        git_metadata["working_tree_dirty"] = (
            args.source_working_tree_dirty == "true"
        )
    results: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": device_name,
            "compute_capability": list(
                torch.cuda.get_device_capability(0)
            ),
            "gpu_total_memory_mib": (
                properties.total_memory / (1024**2)
            ),
        },
        "source": {
            "git": git_metadata,
            "python_snapshot_sha256": _source_snapshot_sha256(),
        },
        "protocol": {
            "cases": list(MODEL_CASES),
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "future_continuous_dim": FUTURE_CONTINUOUS_DIM,
            "future_categorical_cardinalities": list(
                FUTURE_CATEGORICAL_CARDINALITIES
            ),
            "category_ids": "1..cardinality-1; ID 0 reserved for UNK",
            "data": (
                "deterministic synthetic series with continuous and "
                "categorical future effects"
            ),
            "loss": "mae",
            "optimizer": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "precision": args.precision,
            "seeds": args.seeds,
            "warmup_steps": args.warmup_steps,
            "measured_training_steps": args.steps,
            "measured_inference_steps": args.inference_steps,
            "batch_size": args.batch_size,
            "train_samples": args.train_samples,
            "validation_samples": args.validation_samples,
            "data_loader_included": False,
            "host_to_device_transfer_included": False,
        },
        "results": results,
    }

    execution_order: dict[str, list[str]] = {}
    for seed_index, seed in enumerate(args.seeds):
        cases = (
            MODEL_CASES
            if seed_index % 2 == 0
            else tuple(reversed(MODEL_CASES))
        )
        execution_order[str(seed)] = list(cases)
        for case in cases:
            print(
                f"[benchmark] seed={seed} case={case}",
                file=sys.stderr,
                flush=True,
            )
            results.append(
                _benchmark_case(
                    case,
                    seed=seed,
                    steps=args.steps,
                    warmup_steps=args.warmup_steps,
                    inference_steps=args.inference_steps,
                    batch_size=args.batch_size,
                    train_samples=args.train_samples,
                    validation_samples=args.validation_samples,
                    precision=args.precision,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                )
            )

    payload["protocol"]["execution_order"] = execution_order
    payload["summary"] = _aggregate(results)
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

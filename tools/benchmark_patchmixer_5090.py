#!/usr/bin/env python3
"""Benchmark fixed PatchMixer training steps on an RTX 5090.

The benchmark compares the pinned Original and Enhanced point models with the
same synthetic CUDA-resident batch, loss, optimizer, and precision. Data-loader
and host-to-device transfer time are intentionally excluded.
"""

from __future__ import annotations

import argparse
import gc
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

from modeling_module.models.PatchMixer.PatchMixer import PatchMixerModel
from modeling_module.models.PatchMixer import (
    PatchMixerOriginalConfig,
    PatchMixerOriginalModel,
)
from modeling_module.models.PatchMixer.common.configs import PatchMixerConfig
from modeling_module.models.PatchMixer.provenance import (
    PATCHMIXER_ENHANCED_BASELINE_COMMIT,
    PATCHMIXER_REFERENCE_CONFIG,
    PATCHMIXER_REFERENCE_PARAMETER_COUNTS,
    PATCHMIXER_UPSTREAM_COMMIT,
)


MODEL_NAMES = ("original", "enhanced")


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
    parser.add_argument("--steps", type=_positive_int, default=100)
    parser.add_argument("--warmup-steps", type=_nonnegative_int, default=20)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--precision", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
    )
    parser.add_argument(
        "--expected-device",
        default="NVIDIA GeForce RTX 5090",
        help="Fail instead of benchmarking on an unexpected GPU; pass an empty string to disable.",
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
        "working_tree_dirty": bool(status),
    }


def _driver_version() -> str | None:
    value = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
            "--id=0",
        ]
    )
    return value.splitlines()[0].strip() if value else None


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _build_model(name: str) -> torch.nn.Module:
    config_values = dict(PATCHMIXER_REFERENCE_CONFIG)
    if name == "original":
        config = PatchMixerOriginalConfig.from_config(config_values)
        return PatchMixerOriginalModel(config)
    if name == "enhanced":
        config = PatchMixerConfig(**config_values)
        # The Enhanced constructor currently emits compatibility diagnostics.
        with redirect_stdout(StringIO()):
            return PatchMixerModel(config)
    raise ValueError(f"Unsupported model: {name}")


def _point_prediction(output: torch.Tensor, *, horizon: int) -> torch.Tensor:
    if output.ndim == 3 and output.shape[-1] == 1:
        output = output.squeeze(-1)
    if output.ndim != 2 or output.shape[1] != horizon:
        raise RuntimeError(
            f"Expected point output [B,{horizon}] or [B,{horizon},1], "
            f"got {tuple(output.shape)}."
        )
    return output


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


def _make_batch(
    *,
    batch_size: int,
    lookback: int,
    horizon: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    inputs = torch.randn(
        batch_size,
        lookback,
        1,
        device="cuda",
        generator=generator,
    )
    noise = 0.05 * torch.randn(
        batch_size,
        horizon,
        device="cuda",
        generator=generator,
    )
    trend = torch.linspace(0.0, 0.2, horizon, device="cuda")
    targets = inputs[:, -1, 0].unsqueeze(1) + trend.unsqueeze(0) + noise
    return inputs, targets


def _benchmark_model(
    name: str,
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

    reference = dict(PATCHMIXER_REFERENCE_CONFIG)
    expected_counts = dict(PATCHMIXER_REFERENCE_PARAMETER_COUNTS)
    lookback = int(reference["lookback"])
    horizon = int(reference["horizon"])

    model = _build_model(name).cuda().train()
    parameter_count = _parameter_count(model)
    if parameter_count != expected_counts[name]:
        raise RuntimeError(
            f"{name} parameter-count drift: got {parameter_count:,}, "
            f"expected {expected_counts[name]:,}."
        )

    inputs, targets = _make_batch(
        batch_size=batch_size,
        lookback=lookback,
        horizon=horizon,
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
            prediction = _point_prediction(model(inputs), horizon=horizon)
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
    step_times_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    total_time_ms = sum(step_times_ms)
    mean_step_ms = statistics.fmean(step_times_ms)
    result = {
        "model": name,
        "parameters": parameter_count,
        "output_shape": [batch_size, horizon],
        "warmup_steps": warmup_steps,
        "measured_steps": steps,
        "timing_ms": {
            "total": total_time_ms,
            "mean": mean_step_ms,
            "median": statistics.median(step_times_ms),
            "p95": _percentile(step_times_ms, 0.95),
            "min": min(step_times_ms),
            "max": max(step_times_ms),
            "population_stddev": statistics.pstdev(step_times_ms),
        },
        "throughput": {
            "steps_per_second": 1000.0 / mean_step_ms,
            "samples_per_second": batch_size * steps * 1000.0 / total_time_ms,
        },
        "memory_mib": {
            "peak_allocated": torch.cuda.max_memory_allocated() / (1024**2),
            "peak_reserved": torch.cuda.max_memory_reserved() / (1024**2),
            "allocated_after_run": torch.cuda.memory_allocated() / (1024**2),
            "reserved_after_run": torch.cuda.memory_reserved() / (1024**2),
        },
        "loss": {
            "first_measured": float(first_loss),
            "last_measured": float(last_loss),
        },
    }

    del optimizer, model, inputs, targets, starts, ends, first_loss, last_loss
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return result


def _validate_cuda(expected_device: str) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this benchmark must not run on CPU or MPS.")
    device_name = torch.cuda.get_device_name(0)
    if expected_device and device_name != expected_device:
        raise RuntimeError(
            f"Expected device {expected_device!r}, but CUDA device 0 is {device_name!r}."
        )
    return device_name


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    device_name = _validate_cuda(args.expected_device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    properties = torch.cuda.get_device_properties(0)
    free_memory, total_memory = torch.cuda.mem_get_info(0)
    result: dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "driver": _driver_version(),
            "device": device_name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_total_memory_mib": properties.total_memory / (1024**2),
            "gpu_free_memory_before_mib": free_memory / (1024**2),
            "gpu_total_memory_before_mib": total_memory / (1024**2),
        },
        "source": {
            "git": _git_metadata(),
            "original_upstream_commit": PATCHMIXER_UPSTREAM_COMMIT,
            "enhanced_baseline_commit": PATCHMIXER_ENHANCED_BASELINE_COMMIT,
        },
        "protocol": {
            "models": args.models,
            "reference_config": dict(PATCHMIXER_REFERENCE_CONFIG),
            "batch_size": args.batch_size,
            "precision": args.precision,
            "loss": "mse",
            "optimizer": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.steps,
            "torch_compile": False,
            "data_loader_included": False,
            "host_to_device_transfer_included": False,
        },
        "models": [],
    }

    for name in args.models:
        result["models"].append(
            _benchmark_model(
                name,
                steps=args.steps,
                warmup_steps=args.warmup_steps,
                batch_size=args.batch_size,
                precision=args.precision,
                seed=args.seed,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        )

    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("BENCHMARK_RESULT=" + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

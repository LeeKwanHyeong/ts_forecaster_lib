#!/usr/bin/env python3
"""Run the seed-42 SELLM qualification through the public shared trainer."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

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
from modeling_module.api.train import (  # noqa: E402
    ArtifactConfig,
    RuntimeConfig,
    SSLConfig,
    TrainerConfig,
    TrainRequest,
    train,
)
from modeling_module.data_loader.indexed_temporal_data_module import (  # noqa: E402
    IndexedTemporalDataModule,
)
from modeling_module.models.SELLM.training_contract import (  # noqa: E402
    SELLM_TRAINER_CONTRACT,
)
from modeling_module.training.model_losses.loss_module import MAE  # noqa: E402
from tools.benchmark_sellm_token_boundary_5090 import (  # noqa: E402
    _evaluate,
    _horizon_metrics,
)
from tools.dsio_v100_h26_contract import (  # noqa: E402
    V100H26ContractError,
    canonical_json_sha256,
    file_sha256,
    load_training_input_manifest,
    write_secure_json,
)
from tools.run_sellm_production_refit_5090 import (  # noqa: E402
    BATCH_SIZE,
    EPOCHS,
    FORECAST_ORIGIN,
    HORIZON,
    LOOKBACK,
    MODEL_KEY,
    SEED,
    TRAIN_END_WEEK,
    VALIDATION_ORIGIN,
    WINDOW_STRIDE,
    _architecture,
)


BASELINE_MAE: Final = 1.3977457284927368
BASELINE_RAW_NEGATIVE_RATE: Final = 0.1408725767902983
MAX_MAE_RELATIVE_DRIFT: Final = 0.03
RAW_NEGATIVE_RATE_RANGE: Final = (0.14, 0.17)


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_datamodule(frame: pl.DataFrame) -> IndexedTemporalDataModule:
    module = IndexedTemporalDataModule(
        frame,
        lookback=LOOKBACK,
        horizon=HORIZON,
        train_end_week=TRAIN_END_WEEK,
        forecast_origin=FORECAST_ORIGIN,
        validation_origin=VALIDATION_ORIGIN,
        window_stride=WINDOW_STRIDE,
        training_mode="qualification",
        seed=SEED,
        require_all_series_eligible=False,
    )
    module.setup()
    return module


def _checkpoint_contract(
    payload: dict[str, Any],
    *,
    negative_output_penalty_weight: float = 0.0,
    final_nonneg: bool = False,
) -> dict[str, Any]:
    meta = payload.get("meta") or {}
    expected = {
        "model_key": MODEL_KEY,
        "training_mode": "qualification",
        "validation_enabled": True,
        "state_selection": "best_validation",
        "configured_epochs": EPOCHS,
        "random_seed": SEED,
        "batch_size": BATCH_SIZE,
        "negative_output_penalty_weight": negative_output_penalty_weight,
        "final_nonneg": final_nonneg,
        **SELLM_TRAINER_CONTRACT.as_metadata(),
    }
    drift = {
        key: {"expected": value, "actual": meta.get(key)}
        for key, value in expected.items()
        if meta.get(key) != value
    }
    if drift:
        raise V100H26ContractError(
            f"SELLM qualification checkpoint metadata drifted: {drift}"
        )
    return expected


def run_parity(
    *,
    target_source: Path,
    input_manifest: Path,
    output_root: Path,
    llm_local_path: Path,
    device: str,
    num_workers: int,
    negative_output_penalty_weight: float = 0.0,
    final_nonneg: bool = False,
) -> dict[str, Any]:
    if output_root.exists():
        raise V100H26ContractError(
            f"output root already exists; refusing overwrite: {output_root}"
        )
    if not llm_local_path.is_dir():
        raise V100H26ContractError(f"Qwen local path is missing: {llm_local_path}")
    manifest = load_training_input_manifest(
        input_manifest,
        target_source=target_source,
    )
    frame = pl.read_parquet(
        target_source,
        columns=["oper_part_no", "demand_dt", "demand_qty"],
    )
    datamodule = _build_datamodule(frame)
    output_root.mkdir(parents=True)
    _seed_all(SEED)
    pin_memory = device.startswith("cuda")
    train_loader = datamodule.get_train_loader(
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4,
        drop_last=False,
    )
    val_loader = datamodule.get_val_loader(
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4,
        drop_last=False,
    )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = train(
        TrainRequest(
            train_loader=train_loader,
            val_loader=val_loader,
            freq="weekly",
            lookback=LOOKBACK,
            horizon=HORIZON,
            models=[MODEL_KEY],
            architecture=_architecture(
                llm_local_path,
                negative_output_penalty_weight=(
                    negative_output_penalty_weight
                ),
                final_nonneg=final_nonneg,
            ),
            trainer=TrainerConfig(
                warmup_epochs=EPOCHS,
                spike_epochs=0,
                loss_point=MAE(),
                use_intermittent=False,
                val_use_weights=False,
                training_mode="qualification",
                random_seed=SEED,
                **SELLM_TRAINER_CONTRACT.trainer_kwargs(),
            ),
            ssl=SSLConfig(mode="sl_only"),
            runtime=RuntimeConfig(device=device),
            artifacts=ArtifactConfig(
                save_dir=str(output_root),
                auto_save_dir=False,
            ),
            use_exogenous_mode=False,
            use_past_exogenous=False,
            use_future_exogenous=False,
        )
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    peak_training_bytes = (
        int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
    )
    if result.primary_ckpt_path is None:
        raise V100H26ContractError("SELLM qualification did not return a checkpoint")
    checkpoint_path = Path(result.primary_ckpt_path).resolve()
    checkpoint_payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_contract = _checkpoint_contract(
        checkpoint_payload,
        negative_output_penalty_weight=negative_output_penalty_weight,
        final_nonneg=final_nonneg,
    )
    predictor = load_predictor(str(checkpoint_path), device=device, strict=True)
    metrics, raw, target, inference_seconds = _evaluate(
        predictor.model,
        val_loader,
        torch.device(device),
    )
    mae_relative_drift = abs(metrics["mae"] - BASELINE_MAE) / BASELINE_MAE
    negative_rate_pass = (
        RAW_NEGATIVE_RATE_RANGE[0]
        <= metrics["raw_negative_rate"]
        <= RAW_NEGATIVE_RATE_RANGE[1]
    )
    baseline_run = (
        negative_output_penalty_weight == 0.0 and not final_nonneg
    )
    parity_pass = (
        metrics["raw_nonfinite_count"] == 0
        and (
            not baseline_run
            or (
                mae_relative_drift <= MAX_MAE_RELATIVE_DRIFT
                and negative_rate_pass
            )
        )
    )
    receipt: dict[str, Any] = {
        "receipt_format_version": 1,
        "status": "PASS" if parity_pass else "FAIL",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": "sellm-shared-trainer-seed42-parity-v1",
        "source_commit": _source_commit(),
        "target_source": {
            "path": str(target_source),
            "sha256": file_sha256(target_source),
        },
        "input_manifest": {
            "path": str(input_manifest),
            "sha256": manifest["file_sha256"],
            "payload_sha256": manifest["payload_sha256"],
        },
        "data_summary": datamodule.summary,
        "training_contract": {
            "model_key": MODEL_KEY,
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "train_end_week": TRAIN_END_WEEK,
            "validation_origin": VALIDATION_ORIGIN,
            "forecast_origin": FORECAST_ORIGIN,
            "window_stride": WINDOW_STRIDE,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "training_mode": "qualification",
            "validation_enabled": True,
            "state_selection": "best_validation",
            "negative_output_penalty_weight": negative_output_penalty_weight,
            "final_nonneg": final_nonneg,
            **SELLM_TRAINER_CONTRACT.as_metadata(),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "strict_load": True,
            "metadata": checkpoint_contract,
        },
        "baseline": {
            "mae": BASELINE_MAE,
            "raw_negative_rate": BASELINE_RAW_NEGATIVE_RATE,
        },
        "parity": {
            "baseline_run": baseline_run,
            "mae_relative_drift": mae_relative_drift,
            "max_mae_relative_drift": MAX_MAE_RELATIVE_DRIFT,
            "raw_negative_rate_range": list(RAW_NEGATIVE_RATE_RANGE),
            "passed": parity_pass,
        },
        "metrics": metrics,
        "horizon_metrics": _horizon_metrics(raw, target),
        "runtime": {
            "device": device,
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
            "peak_training_allocated_bytes": peak_training_bytes,
        },
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    write_secure_json(output_root / "qualification-parity-receipt.json", receipt)
    if not parity_pass:
        raise V100H26ContractError(
            "SELLM shared-trainer parity failed: "
            f"mae={metrics['mae']}, raw_negative_rate={metrics['raw_negative_rate']}"
        )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-source", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--llm-local-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--negative-output-penalty-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--final-nonneg", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run_parity(
        target_source=args.target_source.expanduser().resolve(),
        input_manifest=args.input_manifest.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        llm_local_path=args.llm_local_path.expanduser().resolve(),
        device=str(args.device),
        num_workers=int(args.num_workers),
        negative_output_penalty_weight=float(
            args.negative_output_penalty_weight
        ),
        final_nonneg=bool(args.final_nonneg),
    )
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

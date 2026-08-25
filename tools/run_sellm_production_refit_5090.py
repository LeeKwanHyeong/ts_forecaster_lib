#!/usr/bin/env python3
"""Run and seal the governed SELLM L52/H26 production refit on RTX 5090."""

from __future__ import annotations

import argparse
import gc
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
    ArchitectureConfig,
    ArtifactConfig,
    RuntimeConfig,
    SELLMArchitectureConfig,
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
from tools.dsio_v100_h26_contract import (  # noqa: E402
    V100H26ContractError,
    canonical_json_sha256,
    file_sha256,
    load_training_input_manifest,
    write_secure_json,
)


LOOKBACK: Final = 52
HORIZON: Final = 26
TRAIN_END_WEEK: Final = 202509
FORECAST_ORIGIN: Final = 202510
VALIDATION_ORIGIN: Final = 202436
WINDOW_STRIDE: Final = 4
SEED: Final = 42
BATCH_SIZE: Final = 256
EPOCHS: Final = 6
LEARNING_RATE: Final = SELLM_TRAINER_CONTRACT.learning_rate
TOKEN_LEN: Final = 13
SEMANTIC_VOCAB_SIZE: Final = 256
SEMANTIC_TOP_K: Final = 32
MODEL_KEY: Final = "sellm_base"
CHECKPOINT_FILENAME: Final = "weekly_SELLMBase_L52_H26.pt"


def _source_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _start_cuda_measurement(device: str) -> float:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def _finish_cuda_measurement(started: float, *, device: str) -> dict[str, Any]:
    cuda_enabled = device.startswith("cuda") and torch.cuda.is_available()
    if cuda_enabled:
        torch.cuda.synchronize()
    result: dict[str, Any] = {
        "seconds": time.perf_counter() - started,
        "cuda_peak_allocated_mib": None,
        "cuda_peak_reserved_mib": None,
    }
    if cuda_enabled:
        mib = 1024.0 * 1024.0
        result["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / mib
        result["cuda_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / mib
    return result


def _load_target(path: Path) -> pl.DataFrame:
    return pl.read_parquet(
        path,
        columns=["oper_part_no", "demand_dt", "demand_qty"],
    )


def _build_datamodule(frame: pl.DataFrame) -> IndexedTemporalDataModule:
    module = IndexedTemporalDataModule(
        frame,
        lookback=LOOKBACK,
        horizon=HORIZON,
        train_end_week=TRAIN_END_WEEK,
        forecast_origin=FORECAST_ORIGIN,
        validation_origin=VALIDATION_ORIGIN,
        window_stride=WINDOW_STRIDE,
        training_mode="production_refit",
        seed=SEED,
        require_all_series_eligible=False,
    )
    module.setup()
    return module


def _architecture(
    llm_local_path: Path,
    *,
    negative_output_penalty_weight: float = 0.0,
) -> ArchitectureConfig:
    return ArchitectureConfig(
        sellm=SELLMArchitectureConfig(
            architecture_variant="paper_v1",
            token_len=TOKEN_LEN,
            semantic_vocab_size=SEMANTIC_VOCAB_SIZE,
            semantic_top_k=SEMANTIC_TOP_K,
            dropout=0.1,
            mlp_hidden_dim=256,
            tscc_latent_dim=8,
            tscc_hidden_dim=64,
            tscc_kl_weight=1e-4,
            use_pretrained_llm=True,
            llm_source="local",
            llm_local_path=str(llm_local_path),
            freeze_llm=True,
            use_time_adapter=True,
            time_adapter_rank=8,
            time_adapter_layers=2,
            use_norm=True,
            final_nonneg=False,
            negative_output_penalty_weight=negative_output_penalty_weight,
        )
    )


def _expected_metadata(
    *,
    negative_output_penalty_weight: float = 0.0,
) -> dict[str, Any]:
    return {
        "model_key": MODEL_KEY,
        "training_mode": "production_refit",
        "validation_enabled": False,
        "state_selection": "final_epoch",
        "configured_epochs": EPOCHS,
        "completed_epochs": EPOCHS,
        "random_seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "token_len": TOKEN_LEN,
        "semantic_vocab_size": SEMANTIC_VOCAB_SIZE,
        "negative_output_penalty_weight": negative_output_penalty_weight,
        **SELLM_TRAINER_CONTRACT.as_metadata(),
    }


def _validate_checkpoint_payload(
    payload: dict[str, Any],
    *,
    negative_output_penalty_weight: float = 0.0,
) -> dict[str, Any]:
    config = payload.get("config") or payload.get("cfg_state") or {}
    meta = payload.get("meta") or {}
    expected_config = {
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "architecture_variant": "paper_v1",
        "token_len": TOKEN_LEN,
        "semantic_vocab_size": SEMANTIC_VOCAB_SIZE,
        "semantic_top_k": SEMANTIC_TOP_K,
        "llm_source": "local",
        "freeze_llm": True,
        "use_time_adapter": True,
        "time_adapter_layers": 2,
        "random_seed": SEED,
        "negative_output_penalty_weight": negative_output_penalty_weight,
    }
    config_drift = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in expected_config.items()
        if config.get(key) != expected
    }
    meta_drift = {
        key: {"expected": expected, "actual": meta.get(key)}
        for key, expected in _expected_metadata(
            negative_output_penalty_weight=negative_output_penalty_weight
        ).items()
        if meta.get(key) != expected
    }
    if config_drift or meta_drift:
        raise V100H26ContractError(
            "SELLM production checkpoint metadata drifted: "
            f"config={config_drift}, meta={meta_drift}"
        )
    final_train_loss = float(meta.get("final_train_loss", float("nan")))
    if not np.isfinite(final_train_loss):
        raise V100H26ContractError("final_train_loss must be finite")
    return {
        "config": expected_config,
        "meta": {
            **_expected_metadata(
                negative_output_penalty_weight=negative_output_penalty_weight
            ),
            "final_train_loss": final_train_loss,
        },
    }


def _latest_histories(datamodule: IndexedTemporalDataModule) -> np.ndarray:
    values = np.stack(
        [series.values[-LOOKBACK:] for series in datamodule._series]
    ).astype(np.float32, copy=False)
    if values.shape != (len(datamodule._series), LOOKBACK):
        raise V100H26ContractError(
            f"invalid production history matrix shape: {values.shape}"
        )
    return values[:, :, None]


def _predict_all(
    predictor,
    histories: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(histories), batch_size):
        batch = torch.from_numpy(histories[start : start + batch_size])
        output = predictor.predict(batch, horizon=HORIZON)
        points = np.asarray(output.get("point"), dtype=np.float32)
        expected_size = len(batch) * HORIZON
        if points.size != expected_size:
            raise V100H26ContractError(
                "SELLM W0-W25 output size drifted: "
                f"expected={expected_size}, actual={points.size}"
            )
        chunks.append(points.reshape(len(batch), HORIZON))
    predictions = np.concatenate(chunks, axis=0)
    if not np.isfinite(predictions).all():
        raise V100H26ContractError("SELLM production prediction is non-finite")
    return predictions


def _preflight(
    *,
    target_source: Path,
    input_manifest: Path,
    llm_local_path: Path,
    negative_output_penalty_weight: float = 0.0,
) -> tuple[dict[str, Any], IndexedTemporalDataModule]:
    manifest = load_training_input_manifest(
        input_manifest,
        target_source=target_source,
    )
    if not llm_local_path.is_dir():
        raise V100H26ContractError(f"Qwen local path is missing: {llm_local_path}")
    datamodule = _build_datamodule(_load_target(target_source))
    summary = datamodule.summary
    if summary["train_target_max_week"] != TRAIN_END_WEEK:
        raise V100H26ContractError("production training does not end at 202509")
    if summary["validation_windows"] != 0:
        raise V100H26ContractError("production refit must not create validation windows")
    payload = {
        "status": "PREFLIGHT_PASS",
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
        "data_summary": summary,
        "training_contract": {
            "model_key": MODEL_KEY,
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "train_end_week": TRAIN_END_WEEK,
            "forecast_origin": FORECAST_ORIGIN,
            "window_stride": WINDOW_STRIDE,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            **SELLM_TRAINER_CONTRACT.as_metadata(),
            "token_len": TOKEN_LEN,
            "semantic_vocab_size": SEMANTIC_VOCAB_SIZE,
            "semantic_top_k": SEMANTIC_TOP_K,
            "negative_output_penalty_weight": negative_output_penalty_weight,
            "training_mode": "production_refit",
            "validation_enabled": False,
            "state_selection": "final_epoch",
            "llm_local_path": str(llm_local_path),
        },
    }
    return payload, datamodule


def run_refit(
    *,
    target_source: Path,
    input_manifest: Path,
    output_root: Path,
    llm_local_path: Path,
    device: str,
    num_workers: int,
    preflight_only: bool,
    negative_output_penalty_weight: float = 0.0,
) -> dict[str, Any]:
    preflight, datamodule = _preflight(
        target_source=target_source,
        input_manifest=input_manifest,
        llm_local_path=llm_local_path,
        negative_output_penalty_weight=negative_output_penalty_weight,
    )
    if preflight_only:
        return preflight
    if output_root.exists():
        raise V100H26ContractError(
            f"output root already exists; refusing overwrite: {output_root}"
        )
    output_root.mkdir(parents=True)
    status_path = output_root / "production-refit-status.txt"
    status_path.write_text("RUNNING current=training\n", encoding="ascii")
    write_secure_json(output_root / "production-refit-data-manifest.json", preflight)

    try:
        _seed_all(SEED)
        train_loader = datamodule.get_train_loader(
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device.startswith("cuda"),
            persistent_workers=num_workers > 0,
            prefetch_factor=4,
            drop_last=False,
        )
        training_started = _start_cuda_measurement(device)
        result = train(
            TrainRequest(
                train_loader=train_loader,
                val_loader=None,
                freq="weekly",
                lookback=LOOKBACK,
                horizon=HORIZON,
                models=[MODEL_KEY],
                architecture=_architecture(
                    llm_local_path,
                    negative_output_penalty_weight=(
                        negative_output_penalty_weight
                    ),
                ),
                trainer=TrainerConfig(
                    warmup_epochs=EPOCHS,
                    spike_epochs=0,
                    loss_point=MAE(),
                    use_intermittent=False,
                    val_use_weights=False,
                    training_mode="production_refit",
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
        training_runtime = _finish_cuda_measurement(
            training_started,
            device=device,
        )
        if result.primary_ckpt_path is None:
            raise V100H26ContractError("SELLM refit did not return a checkpoint")
        checkpoint_path = Path(result.primary_ckpt_path).resolve()
        if checkpoint_path.name != CHECKPOINT_FILENAME:
            raise V100H26ContractError(
                "SELLM checkpoint filename drifted: "
                f"expected={CHECKPOINT_FILENAME}, actual={checkpoint_path.name}"
            )

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_contract = _validate_checkpoint_payload(
            payload,
            negative_output_penalty_weight=negative_output_penalty_weight,
        )
        del payload, result, train_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        load_started = _start_cuda_measurement(device)
        predictor = load_predictor(
            str(checkpoint_path),
            device=device,
            strict=True,
        )
        strict_load_runtime = _finish_cuda_measurement(load_started, device=device)

        histories = _latest_histories(datamodule)
        inference_started = _start_cuda_measurement(device)
        raw = _predict_all(
            predictor,
            histories,
            batch_size=BATCH_SIZE,
        )
        inference_runtime = _finish_cuda_measurement(
            inference_started,
            device=device,
        )
        negative_count = int((raw < 0).sum())
        prediction_count = int(raw.size)
        inference_runtime["series_per_second"] = (
            len(raw) / float(inference_runtime["seconds"])
        )
        inference_runtime["points_per_second"] = (
            prediction_count / float(inference_runtime["seconds"])
        )
        training_runtime["windows_per_second"] = (
            preflight["data_summary"]["train_windows"]
            * EPOCHS
            / float(training_runtime["seconds"])
        )

        runtime = {
            "python_version": (
                f"{sys.version_info.major}.{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "torch_version": torch.__version__,
            "device": device,
            "cuda_device_name": (
                torch.cuda.get_device_name(0)
                if device.startswith("cuda") and torch.cuda.is_available()
                else None
            ),
            "training": training_runtime,
            "strict_load": strict_load_runtime,
            "inference": inference_runtime,
        }
        receipt: dict[str, Any] = {
            "receipt_format_version": 1,
            "status": "PASS",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "contract": "sellm-l52-h26-production-refit-v1",
            "source_commit": preflight["source_commit"],
            "target_source": preflight["target_source"],
            "input_manifest": preflight["input_manifest"],
            "data_summary": preflight["data_summary"],
            "training_contract": preflight["training_contract"],
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": file_sha256(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
                "strict_load": True,
                "contract": checkpoint_contract,
            },
            "forecast_canary": {
                "history_end_week": TRAIN_END_WEEK,
                "forecast_origin": FORECAST_ORIGIN,
                "offsets": "W0-W25",
                "series_count": len(raw),
                "prediction_shape": list(raw.shape),
                "prediction_count": prediction_count,
                "raw_nonfinite_count": int((~np.isfinite(raw)).sum()),
                "raw_negative_count": negative_count,
                "raw_negative_rate": negative_count / prediction_count,
                "raw_min": float(raw.min()),
                "raw_max": float(raw.max()),
                "clip_zero_count": negative_count,
            },
            "runtime_evidence": runtime,
        }
        receipt["receipt_sha256"] = canonical_json_sha256(receipt)
        write_secure_json(output_root / "production-refit-receipt.json", receipt)
        status_path.write_text(
            f"PASS model={MODEL_KEY} seed={SEED} epochs={EPOCHS}\n",
            encoding="ascii",
        )
        return receipt
    except BaseException:
        status_path.write_text("FAILED\n", encoding="ascii")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-source", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--llm-local-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--negative-output-penalty-weight",
        type=float,
        default=0.0,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run_refit(
        target_source=args.target_source.expanduser().resolve(),
        input_manifest=args.input_manifest.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        llm_local_path=args.llm_local_path.expanduser().resolve(),
        device=str(args.device),
        num_workers=int(args.num_workers),
        preflight_only=bool(args.preflight_only),
        negative_output_penalty_weight=float(
            args.negative_output_penalty_weight
        ),
    )
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

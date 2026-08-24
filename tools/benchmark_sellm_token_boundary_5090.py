from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl
import torch

from modeling_module.api import load_predictor
from modeling_module.data_loader.indexed_temporal_data_module import (
    IndexedTemporalDataModule,
)
from modeling_module.models.SELLM.SELLM import SELLMModel
from modeling_module.models.SELLM.configs import SELLMConfig
from modeling_module.utils.checkpoint import save_model


LOOKBACK = 52
HORIZON = 26
TRAIN_END_WEEK = 202509
FORECAST_ORIGIN = 202510
VALIDATION_ORIGIN = 202436
WINDOW_STRIDE = 4
SEMANTIC_VOCAB_SIZE = 256
SEMANTIC_TOP_K = 32
SEEDS = (11, 22, 33)
TOKEN_LENGTHS = (8, 13)
SAMPLE_SERIES = 256
SAMPLE_SEED = 42
EXPECTED_SAMPLE_SHA256 = (
    "d0f7cab0fe1f236877bc17d841d7fefde034116904076104c45683486019b208"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_datamodule(frame: pl.DataFrame, seed: int) -> IndexedTemporalDataModule:
    module = IndexedTemporalDataModule(
        frame,
        lookback=LOOKBACK,
        horizon=HORIZON,
        train_end_week=TRAIN_END_WEEK,
        forecast_origin=FORECAST_ORIGIN,
        validation_origin=VALIDATION_ORIGIN,
        window_stride=WINDOW_STRIDE,
        training_mode="qualification",
        seed=seed,
        require_all_series_eligible=False,
    )
    module.setup()
    return module


def _build_model(
    *,
    token_len: int,
    seed: int,
    llm_local_path: Path,
    device: torch.device,
) -> SELLMModel:
    config = SELLMConfig(
        lookback=LOOKBACK,
        horizon=HORIZON,
        architecture_variant="paper_v1",
        token_len=token_len,
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
        random_seed=seed,
    )
    return SELLMModel(config).to(device)


def _metric_set(raw: np.ndarray, target: np.ndarray) -> dict[str, float]:
    processed = np.maximum(raw, 0.0)
    error = processed - target
    denominator = max(float(np.abs(target).sum()), 1e-8)
    return {
        "mae": float(np.abs(error).mean()),
        "wape": float(np.abs(error).sum() / denominator),
        "smape": float(
            np.mean(
                2.0
                * np.abs(error)
                / (np.abs(processed) + np.abs(target) + 1e-8)
            )
        ),
        "bias": float(error.mean()),
        "raw_negative_rate": float((raw < 0).mean()),
        "raw_min": float(raw.min()),
        "raw_nonfinite_count": int((~np.isfinite(raw)).sum()),
    }


def _evaluate(model: SELLMModel, loader, device: torch.device):
    model.eval()
    raw_parts = []
    target_parts = []
    started = time.perf_counter()
    with torch.inference_mode():
        for x, y, _ids in loader:
            raw_parts.append(
                model(x.to(device, non_blocking=True)).squeeze(-1).cpu().numpy()
            )
            target_parts.append(y.numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    raw = np.concatenate(raw_parts)
    target = np.concatenate(target_parts)
    return _metric_set(raw, target), raw, target, elapsed


def _horizon_metrics(raw: np.ndarray, target: np.ndarray) -> list[dict[str, Any]]:
    return [
        {"horizon": horizon, **_metric_set(raw[:, horizon], target[:, horizon])}
        for horizon in range(HORIZON)
    ]


def _restore_trainable_state(
    model: SELLMModel,
    state: dict[str, torch.Tensor],
) -> None:
    named = dict(model.named_parameters())
    missing = sorted(set(state) - set(named))
    if missing:
        raise RuntimeError(f"Best-state parameters are missing from model: {missing}")
    with torch.no_grad():
        for name, value in state.items():
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))


def _train(
    *,
    model: SELLMModel,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    progress_path: Path,
    run_contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], int]:
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(trainable.values(), lr=learning_rate)
    reports: list[dict[str, Any]] = []
    best_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, epochs + 1):
        model.train()
        objective_sum = 0.0
        point_count = 0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for x, y, _ids in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x).squeeze(-1)
            loss = torch.mean(torch.abs(prediction - y))
            regularization = model.reg_loss()
            if regularization is not None:
                loss = loss + regularization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(trainable.values()), 30.0)
            optimizer.step()
            objective_sum += float(loss.detach()) * y.numel()
            point_count += y.numel()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - started
        metrics, _raw, _target, validation_seconds = _evaluate(
            model, val_loader, device
        )
        report = {
            "epoch": epoch,
            "train_objective": objective_sum / point_count,
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            **metrics,
        }
        reports.append(report)
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in trainable.items()
            }
        progress = {
            "status": "RUNNING",
            "updated_at_utc": _utc_now(),
            "contract": run_contract,
            "best_epoch": best_epoch,
            "best_mae": best_mae,
            "epochs": reports,
        }
        _write_json(progress_path, progress)
        print(
            json.dumps(
                {
                    "token_len": run_contract["token_len"],
                    "seed": run_contract["seed"],
                    **report,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    del optimizer
    if not best_state:
        raise RuntimeError("Training completed without a best state.")
    return reports, best_state, best_epoch


def _sample_frame(frame: pl.DataFrame) -> tuple[pl.DataFrame, str]:
    module = _make_datamodule(frame, SAMPLE_SEED)
    eligible = sorted(buffer.part_id for buffer in module._series)
    sample_ids = sorted(random.Random(SAMPLE_SEED).sample(eligible, SAMPLE_SERIES))
    sample_sha256 = hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()
    if sample_sha256 != EXPECTED_SAMPLE_SHA256:
        raise RuntimeError(
            f"Sample fingerprint drifted: {sample_sha256} != {EXPECTED_SAMPLE_SHA256}"
        )
    sampled = frame.filter(
        pl.col("oper_part_no").cast(pl.String).is_in(sample_ids)
    )
    del module
    gc.collect()
    return sampled, sample_sha256


def run_integration(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    source = args.source.resolve()
    output = args.output_root.resolve() / "integration"
    output.mkdir(parents=True, exist_ok=True)
    frame = pl.read_parquet(source).select(
        ["oper_part_no", "demand_dt", "demand_qty"]
    )
    frame, sample_sha256 = _sample_frame(frame)
    _seed_all(SAMPLE_SEED)
    module = _make_datamodule(frame, SAMPLE_SEED)
    train_loader = module.get_train_loader(
        batch_size=args.integration_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = module.get_val_loader(
        batch_size=args.integration_batch_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    model = _build_model(
        token_len=13,
        seed=SAMPLE_SEED,
        llm_local_path=args.llm_local_path,
        device=device,
    )
    contract = {
        "token_len": 13,
        "seed": SAMPLE_SEED,
        "semantic_vocab_size": SEMANTIC_VOCAB_SIZE,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "sample_series": SAMPLE_SERIES,
    }
    reports, best_state, best_epoch = _train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.integration_epochs,
        learning_rate=args.learning_rate,
        progress_path=output / "integration-progress.json",
        run_contract=contract,
    )
    _restore_trainable_state(model, best_state)
    model.eval()
    x, _y, _ids = next(iter(val_loader))
    x = x[:2].to(device)
    with torch.inference_mode():
        expected = model(x).detach().cpu()

    checkpoint_path = output / "sellm_k256_l52_h26_token13.pt"
    save_model(
        model,
        model.cfg,
        str(checkpoint_path),
        extra_meta={
            "model_key": "sellm_base",
            "family_key": "sellm",
            "training_mode": "qualification",
            "validation_enabled": True,
            "state_selection": "best_validation",
            "random_seed": SAMPLE_SEED,
            "semantic_vocab_size": SEMANTIC_VOCAB_SIZE,
            "best_epoch": best_epoch,
        },
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_token_len = payload["config"].get("token_len")
    meta_token_len = payload["meta"].get("token_len")
    if config_token_len != 13 or meta_token_len != 13:
        raise RuntimeError(
            "Checkpoint token_len mismatch: "
            f"config={config_token_len}, meta={meta_token_len}"
        )
    del payload, best_state, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    predictor = load_predictor(
        str(checkpoint_path), device=str(device), strict=True
    )
    predictor.model.eval()
    with torch.inference_mode():
        restored = predictor.model(x).detach().cpu()
    max_abs_error = float(torch.max(torch.abs(expected - restored)))
    if max_abs_error != 0.0:
        raise RuntimeError(f"Strict-load prediction drifted: {max_abs_error}")
    if int(predictor.config["token_len"]) != 13:
        raise RuntimeError("Loaded predictor config did not preserve token_len=13.")

    receipt = {
        "status": "PASS",
        "contract": "sellm-k256-l52-h26-token13-checkpoint-v1",
        "created_at_utc": _utc_now(),
        "git_commit": args.git_commit,
        "source": {"path": str(source), "sha256": _sha256(source)},
        "sample_id_sha256": sample_sha256,
        "training": contract,
        "best_epoch": best_epoch,
        "epochs": reports,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "config_token_len": config_token_len,
            "meta_token_len": meta_token_len,
            "strict_load": True,
            "prediction_shape": list(restored.shape),
            "prediction_max_abs_error": max_abs_error,
        },
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    _write_json(output / "integration-receipt.json", receipt)
    progress_path = output / "integration-progress.json"
    if progress_path.exists():
        progress_path.unlink()
    print(json.dumps(receipt["checkpoint"], sort_keys=True), flush=True)
    return receipt


def run_qualification_case(
    args: argparse.Namespace,
    frame: pl.DataFrame,
    *,
    token_len: int,
    seed: int,
) -> dict[str, Any]:
    case_root = args.output_root.resolve() / f"token{token_len}" / f"seed{seed}"
    receipt_path = case_root / "qualification-receipt.json"
    if receipt_path.is_file() and not args.force:
        receipt = json.loads(receipt_path.read_text(encoding="ascii"))
        if receipt.get("status") == "PASS":
            print(f"[skip] completed token_len={token_len} seed={seed}", flush=True)
            return receipt

    device = torch.device(args.device)
    _seed_all(seed)
    module = _make_datamodule(frame, seed)
    train_loader = module.get_train_loader(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = module.get_val_loader(
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    model = _build_model(
        token_len=token_len,
        seed=seed,
        llm_local_path=args.llm_local_path,
        device=device,
    )
    contract = {
        "token_len": token_len,
        "seed": seed,
        "semantic_vocab_size": SEMANTIC_VOCAB_SIZE,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "train_end_week": TRAIN_END_WEEK,
        "validation_origin": VALIDATION_ORIGIN,
        "forecast_origin": FORECAST_ORIGIN,
        "window_stride": WINDOW_STRIDE,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "series_count": len(module._series),
        "train_windows": len(train_loader.dataset),
        "validation_windows": len(val_loader.dataset),
    }
    reports, best_state, best_epoch = _train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        progress_path=case_root / "qualification-progress.json",
        run_contract=contract,
    )
    peak_training_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    _restore_trainable_state(model, best_state)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_metrics, raw, target, inference_seconds = _evaluate(model, val_loader, device)
    peak_inference_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    receipt = {
        "status": "PASS",
        "contract": "sellm-full-token-boundary-qualification-v1",
        "created_at_utc": _utc_now(),
        "git_commit": args.git_commit,
        "source": {"path": str(args.source.resolve()), "sha256": args.source_sha256},
        "training": contract,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "epochs": reports,
        "horizon_metrics": _horizon_metrics(raw, target),
        "runtime": {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "mean_train_seconds_per_epoch": float(
                np.mean([report["train_seconds"] for report in reports])
            ),
            "best_inference_seconds": inference_seconds,
            "inference_series_per_second": len(val_loader.dataset) / inference_seconds,
            "peak_training_allocated_bytes": peak_training_bytes,
            "peak_inference_allocated_bytes": peak_inference_bytes,
        },
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    _write_json(receipt_path, receipt)
    progress_path = case_root / "qualification-progress.json"
    if progress_path.exists():
        progress_path.unlink()
    print(
        json.dumps(
            {
                "status": "PASS",
                "token_len": token_len,
                "seed": seed,
                "best_epoch": best_epoch,
                **best_metrics,
                **receipt["runtime"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    del model, best_state, train_loader, val_loader, module
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return receipt


def _mean_std(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std())}


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    receipts = []
    for token_len in args.token_lengths:
        for seed in args.seeds:
            path = output_root / f"token{token_len}" / f"seed{seed}" / "qualification-receipt.json"
            receipt = json.loads(path.read_text(encoding="ascii"))
            seal = receipt.pop("receipt_sha256", None)
            if receipt.get("status") != "PASS" or seal != _canonical_sha256(receipt):
                raise RuntimeError(f"Invalid qualification receipt: {path}")
            receipts.append({**receipt, "receipt_sha256": seal})

    summaries = []
    for token_len in args.token_lengths:
        selected = [
            receipt
            for receipt in receipts
            if receipt["training"]["token_len"] == token_len
        ]
        epoch_counts = {len(receipt["epochs"]) for receipt in selected}
        if len(epoch_counts) != 1:
            raise RuntimeError(
                f"Epoch count drifted for token_len={token_len}: {epoch_counts}"
            )
        epoch_count = epoch_counts.pop()
        summaries.append(
            {
                "token_len": token_len,
                "best_epoch": _mean_std(
                    receipt["best_epoch"] for receipt in selected
                ),
                **{
                    metric: _mean_std(
                        receipt["best_metrics"][metric] for receipt in selected
                    )
                    for metric in (
                        "mae",
                        "wape",
                        "smape",
                        "bias",
                        "raw_negative_rate",
                    )
                },
                "mean_train_seconds_per_epoch": _mean_std(
                    receipt["runtime"]["mean_train_seconds_per_epoch"]
                    for receipt in selected
                ),
                "inference_series_per_second": _mean_std(
                    receipt["runtime"]["inference_series_per_second"]
                    for receipt in selected
                ),
                "peak_training_allocated_bytes": _mean_std(
                    receipt["runtime"]["peak_training_allocated_bytes"]
                    for receipt in selected
                ),
                "peak_inference_allocated_bytes": _mean_std(
                    receipt["runtime"]["peak_inference_allocated_bytes"]
                    for receipt in selected
                ),
                "epoch_metrics": [
                    {
                        "epoch": epoch + 1,
                        **{
                            metric: _mean_std(
                                receipt["epochs"][epoch][metric]
                                for receipt in selected
                            )
                            for metric in (
                                "mae",
                                "wape",
                                "smape",
                                "bias",
                                "raw_negative_rate",
                                "train_objective",
                                "train_seconds",
                            )
                        },
                    }
                    for epoch in range(epoch_count)
                ],
                "horizon_metrics": [
                    {
                        "horizon": horizon,
                        **{
                            metric: _mean_std(
                                receipt["horizon_metrics"][horizon][metric]
                                for receipt in selected
                            )
                            for metric in (
                                "mae",
                                "wape",
                                "smape",
                                "bias",
                                "raw_negative_rate",
                            )
                        },
                    }
                    for horizon in range(HORIZON)
                ],
            }
        )
    comparison: dict[str, Any] = {}
    if set(args.token_lengths) == {8, 13}:
        baseline = next(item for item in summaries if item["token_len"] == 8)
        candidate = next(item for item in summaries if item["token_len"] == 13)
        comparison = {
            metric: {
                "token8": baseline[metric]["mean"],
                "token13": candidate[metric]["mean"],
                "absolute_delta": candidate[metric]["mean"]
                - baseline[metric]["mean"],
                "relative_delta": (
                    (candidate[metric]["mean"] - baseline[metric]["mean"])
                    / baseline[metric]["mean"]
                    if baseline[metric]["mean"] != 0
                    else None
                ),
            }
            for metric in (
                "mae",
                "wape",
                "smape",
                "bias",
                "raw_negative_rate",
                "mean_train_seconds_per_epoch",
                "inference_series_per_second",
                "peak_training_allocated_bytes",
                "peak_inference_allocated_bytes",
            )
        }
    aggregate_receipt = {
        "status": "PASS",
        "contract": "sellm-full-token-boundary-aggregate-v2",
        "created_at_utc": _utc_now(),
        "git_commit": args.git_commit,
        "source_sha256": args.source_sha256,
        "seeds": list(args.seeds),
        "token_lengths": list(args.token_lengths),
        "summaries": summaries,
        "comparison": comparison,
    }
    aggregate_receipt["receipt_sha256"] = _canonical_sha256(aggregate_receipt)
    _write_json(output_root / "qualification-aggregate-receipt.json", aggregate_receipt)
    printable = comparison or {
        str(summary["token_len"]): {
            "best_epoch": summary["best_epoch"],
            "mae": summary["mae"],
            "epoch_metrics": summary["epoch_metrics"],
        }
        for summary in summaries
    }
    print(json.dumps(printable, indent=2, sort_keys=True), flush=True)
    return aggregate_receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark SELLM token-boundary candidates on RTX 5090."
    )
    parser.add_argument(
        "command", choices=("integration", "qualify-all", "qualify", "aggregate")
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--llm-local-path", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--integration-epochs", type=int, default=1)
    parser.add_argument("--integration-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--token-lengths",
        type=int,
        nargs="+",
        default=list(TOKEN_LENGTHS),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.source = args.source.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.llm_local_path = args.llm_local_path.expanduser().resolve()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    if len(set(args.token_lengths)) != len(args.token_lengths) or not set(
        args.token_lengths
    ).issubset(TOKEN_LENGTHS):
        raise ValueError(
            f"token_lengths must be unique values from {TOKEN_LENGTHS}."
        )
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must contain unique integers.")
    if args.command == "qualify-all":
        args.token_lengths = list(TOKEN_LENGTHS)
        args.seeds = list(SEEDS)
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if not args.llm_local_path.is_dir():
        raise FileNotFoundError(args.llm_local_path)
    args.source_sha256 = _sha256(args.source)

    if args.command == "integration":
        run_integration(args)
        return
    if args.command == "aggregate":
        aggregate(args)
        return

    frame = pl.read_parquet(args.source).select(
        ["oper_part_no", "demand_dt", "demand_qty"]
    )
    for seed in args.seeds:
        for token_len in args.token_lengths:
            run_qualification_case(
                args,
                frame,
                token_len=token_len,
                seed=seed,
            )
    aggregate(args)


if __name__ == "__main__":
    main()

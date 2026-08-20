#!/usr/bin/env python3
"""Run governed DSIO V100 L52/H26 production refit for selected exogenous models."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
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

from modeling_module import (  # noqa: E402
    ArtifactConfig,
    RuntimeConfig,
    SSLConfig,
    TrainerConfig,
    TrainRequest,
    load_predictor,
    train,
)
from modeling_module.data_loader.deterministic_calendar import (  # noqa: E402
    WEEKLY_CALENDAR_CONTRACT_VERSION,
    WEEKLY_CALENDAR_CONTINUOUS_FEATURES,
    attach_weekly_calendar_features,
    weekly_calendar_schema_fingerprint,
)
from modeling_module.data_loader.temporal import add_period  # noqa: E402
from tools.dsio_v100_h26_contract import (  # noqa: E402
    FORECAST_ORIGIN,
    HORIZON,
    LOOKBACK,
    SITE_CD,
    TRAIN_END_WEEK,
    V100H26ContractError,
    canonical_json_sha256,
    file_sha256,
    load_training_input_manifest,
    write_secure_json,
)
from tools.run_dsio_v100_h26_exogenous_qualification import (  # noqa: E402
    DATE_COLUMN,
    DEFAULT_BATCH_SIZE,
    FREQUENCY,
    ID_COLUMN,
    MODEL_SPECS_BY_KEY,
    TARGET_COLUMN,
    WINDOW_STRIDE,
    ExogenousQualificationModelSpec,
    _batch_contract,
    _build_datamodule,
    _load_target,
    _source_commit,
    _training_metrics,
    build_architecture,
    configure_torch_runtime,
    set_global_seed,
)


PRODUCTION_REFIT_SEED: Final = 42
PRODUCTION_REFIT_EPOCHS: Final = {
    "exotst_base": 40,
    "patchtst_exogenous": 35,
}
PRODUCTION_MODEL_SPECS: Final = tuple(
    MODEL_SPECS_BY_KEY[key] for key in PRODUCTION_REFIT_EPOCHS
)
EPOCH_POLICY_EVIDENCE: Final = {
    "selection_rule": "lowest_four_seed_mean_validation_loss_by_epoch",
    "qualification_seeds": [11, 22, 33, 42],
    "models": {
        "exotst_base": {
            "seed_best_epochs": {"11": 32, "22": 36, "33": 27, "42": 30},
            "selected_epoch": 40,
            "selected_epoch_mean_validation_loss": 1.471662,
            "selected_epoch_validation_loss_std": 0.033302,
            "selected_epoch_worst_validation_loss": 1.528111,
        },
        "patchtst_exogenous": {
            "seed_best_epochs": {"11": 25, "22": 35, "33": 30, "42": 39},
            "selected_epoch": 35,
            "selected_epoch_mean_validation_loss": 1.357135,
            "selected_epoch_validation_loss_std": 0.081083,
            "selected_epoch_worst_validation_loss": 1.473626,
        },
    },
    "seed_policy": {
        "selected_seed": PRODUCTION_REFIT_SEED,
        "rule": "fixed_project_canonical_seed_not_selected_by_validation_rank",
    },
}


def build_worker_command(
    *,
    python_executable: Path,
    target_source: Path,
    input_manifest: Path,
    output_root: Path,
    model_key: str,
    batch_size: int,
    num_workers: int,
    device: str,
    sample_part_count: int | None,
    preflight_only: bool,
) -> list[str]:
    if model_key not in PRODUCTION_REFIT_EPOCHS:
        raise ValueError(f"Unsupported production model: {model_key!r}.")
    command = [
        str(python_executable),
        str(Path(__file__).resolve()),
        "--target-source",
        str(target_source),
        "--input-manifest",
        str(input_manifest),
        "--output-root",
        str(output_root),
        "--model-key",
        model_key,
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--device",
        device,
    ]
    if sample_part_count is not None:
        command.extend(["--sample-part-count", str(sample_part_count)])
    if preflight_only:
        command.append("--preflight-only")
    return command


def _production_canary_batch(datamodule, *, spec: ExogenousQualificationModelSpec):
    datamodule.setup()
    dataset = datamodule.train_dataset
    assert dataset is not None
    selected = dataset._series[: min(2, dataset.series_count)]
    if not selected:
        raise V100H26ContractError("production canary requires an eligible series")

    x = torch.from_numpy(
        np.stack([item.values[-LOOKBACK:] for item in selected])
    ).unsqueeze(-1)
    past_cont = torch.from_numpy(
        np.stack([item.past_cont[-LOOKBACK:] for item in selected])
    )
    future_weeks = [
        add_period(FORECAST_ORIGIN, offset, "weekly")
        for offset in range(HORIZON)
    ]
    future_frame = attach_weekly_calendar_features(
        pl.DataFrame({DATE_COLUMN: future_weeks}),
        date_column=DATE_COLUMN,
    )
    if spec.uses_future_continuous:
        future_values = (
            future_frame.select(WEEKLY_CALENDAR_CONTINUOUS_FEATURES)
            .to_numpy()
            .astype(np.float32, copy=False)
        )
        future_cont = torch.from_numpy(
            np.repeat(future_values[None, :, :], len(selected), axis=0)
        )
    else:
        future_cont = torch.empty((len(selected), HORIZON, 0), dtype=torch.float32)
    past_cat = torch.empty((len(selected), LOOKBACK, 0), dtype=torch.long)
    y_placeholder = torch.zeros((len(selected), HORIZON), dtype=torch.float32)
    batch = (
        x,
        y_placeholder,
        [item.part_id for item in selected],
        future_cont,
        past_cont,
        past_cat,
    )
    evidence = _batch_contract(batch, spec=spec)
    evidence.update(
        {
            "history_end_week": TRAIN_END_WEEK,
            "forecast_start_week": FORECAST_ORIGIN,
            "forecast_end_week": future_weeks[-1],
        }
    )
    return batch, evidence


def _validate_checkpoint(
    *,
    checkpoint_path: Path,
    spec: ExogenousQualificationModelSpec,
    canary_batch: tuple[Any, ...],
    device: str,
    epochs: int,
) -> dict[str, object]:
    if checkpoint_path.name != spec.checkpoint_filename:
        raise V100H26ContractError(
            "checkpoint filename drifted: "
            f"expected {spec.checkpoint_filename}, got {checkpoint_path.name}"
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint.get("config") or checkpoint.get("cfg_state") or {}
    meta = checkpoint.get("meta") or {}
    if config.get("lookback") != LOOKBACK or config.get("horizon") != HORIZON:
        raise V100H26ContractError("checkpoint window must be L52/H26")
    expected_meta = {
        "model_key": spec.model_key,
        "training_mode": "production_refit",
        "validation_enabled": False,
        "state_selection": "final_epoch",
        "configured_epochs": epochs,
        "completed_epochs": epochs,
        "random_seed": PRODUCTION_REFIT_SEED,
    }
    drift = {
        key: {"expected": expected, "actual": meta.get(key)}
        for key, expected in expected_meta.items()
        if meta.get(key) != expected
    }
    if drift:
        raise V100H26ContractError(f"production checkpoint metadata drifted: {drift}")

    predictor = load_predictor(str(checkpoint_path), device=device, strict=True)
    schema = predictor.exogenous_schema
    if schema is None:
        raise V100H26ContractError("checkpoint is missing its exogenous schema")
    expected_future = (
        WEEKLY_CALENDAR_CONTINUOUS_FEATURES
        if spec.uses_future_continuous
        else ()
    )
    if schema.past_cont_names != WEEKLY_CALENDAR_CONTINUOUS_FEATURES:
        raise V100H26ContractError("checkpoint past exogenous schema drifted")
    if schema.future_cont_names != expected_future:
        raise V100H26ContractError("checkpoint future exogenous schema drifted")

    prediction = predictor.predict(canary_batch)
    points = np.asarray(prediction.get("point"))
    expected_shape = (len(canary_batch[2]), HORIZON)
    if points.shape != expected_shape or not np.isfinite(points).all():
        raise V100H26ContractError(
            "production checkpoint canary failed: "
            f"expected finite {expected_shape}, got {points.shape}"
        )
    return {
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "prediction_shape": list(points.shape),
        "prediction_finite": True,
        "prediction_min": float(points.min()),
        "prediction_max": float(points.max()),
        "exogenous_schema_fingerprint": schema.fingerprint,
        "checkpoint_meta": expected_meta,
    }


def run_model(
    *,
    target_source: Path,
    input_manifest: Path,
    output_root: Path,
    spec: ExogenousQualificationModelSpec,
    batch_size: int,
    num_workers: int,
    device: str,
    sample_part_count: int | None,
    preflight_only: bool,
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if spec.model_key not in PRODUCTION_REFIT_EPOCHS:
        raise ValueError(f"Unsupported production model: {spec.model_key!r}.")
    epochs = PRODUCTION_REFIT_EPOCHS[spec.model_key]
    manifest = load_training_input_manifest(
        input_manifest,
        target_source=target_source,
    )
    set_global_seed(PRODUCTION_REFIT_SEED)
    configure_torch_runtime()

    target = _load_target(
        target_source,
        sample_part_count=sample_part_count,
        seed=PRODUCTION_REFIT_SEED,
    )
    datamodule = _build_datamodule(
        target,
        spec=spec,
        seed=PRODUCTION_REFIT_SEED,
        training_mode="production_refit",
    )
    summary = datamodule.summary
    train_loader = datamodule.get_train_loader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
        prefetch_factor=4,
        drop_last=summary["train_windows"] >= batch_size,
    )
    first_batch = next(iter(train_loader))
    train_batch_contract = _batch_contract(first_batch, spec=spec)
    canary_batch, canary_contract = _production_canary_batch(datamodule, spec=spec)
    preflight: dict[str, object] = {
        "status": "PREFLIGHT_PASS",
        "model_key": spec.model_key,
        "source_commit": _source_commit(),
        "target_sha256": file_sha256(target_source),
        "input_manifest_sha256": manifest["file_sha256"],
        "calendar_contract_version": WEEKLY_CALENDAR_CONTRACT_VERSION,
        "calendar_schema_fingerprint": weekly_calendar_schema_fingerprint(),
        "data_summary": summary,
        "train_batch_contract": train_batch_contract,
        "production_canary_contract": canary_contract,
        "epoch_policy": EPOCH_POLICY_EVIDENCE,
    }
    if preflight_only:
        return preflight

    model_output = output_root / spec.model_key
    if model_output.exists():
        raise V100H26ContractError(
            f"model output already exists; refusing overwrite: {model_output}"
        )
    model_output.mkdir(parents=True)
    write_secure_json(
        model_output / "production-refit-data-manifest.json",
        {
            **preflight,
            "status": "SEALED",
            "training_contract": {
                "site_cd": SITE_CD,
                "frequency": FREQUENCY,
                "lookback": LOOKBACK,
                "horizon": HORIZON,
                "train_end_week": TRAIN_END_WEEK,
                "forecast_origin": FORECAST_ORIGIN,
                "window_stride": WINDOW_STRIDE,
                "epochs": epochs,
                "seed": PRODUCTION_REFIT_SEED,
                "loss": "library_point_default",
                "training_mode": "production_refit",
                "validation_enabled": False,
                "state_selection": "final_epoch",
            },
            "feature_contract": {
                "source_kind": "deterministic_calendar",
                "past_continuous_columns": list(WEEKLY_CALENDAR_CONTINUOUS_FEATURES),
                "future_continuous_columns": (
                    list(WEEKLY_CALENDAR_CONTINUOUS_FEATURES)
                    if spec.uses_future_continuous
                    else []
                ),
                "past_categorical_columns": [],
                "future_categorical_columns": [],
            },
        },
    )

    result = train(
        TrainRequest(
            train_loader=train_loader,
            val_loader=None,
            freq=FREQUENCY,
            lookback=LOOKBACK,
            horizon=HORIZON,
            models=[spec.model_key],
            architecture=build_architecture(),
            trainer=TrainerConfig(
                warmup_epochs=epochs,
                spike_epochs=0,
                lr=1e-3,
                training_mode="production_refit",
                random_seed=PRODUCTION_REFIT_SEED,
            ),
            ssl=SSLConfig(mode="sl_only"),
            runtime=RuntimeConfig(device=device),
            artifacts=ArtifactConfig(
                save_dir=str(model_output),
                auto_save_dir=False,
            ),
            use_exogenous_mode=True,
            use_past_exogenous=True,
            use_future_exogenous=spec.uses_future_continuous,
        )
    )
    if result.primary_ckpt_path is None:
        raise V100H26ContractError(
            f"{spec.model_key} refit did not return a checkpoint"
        )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    checkpoint_path = Path(result.primary_ckpt_path).resolve()
    checkpoint_evidence = _validate_checkpoint(
        checkpoint_path=checkpoint_path,
        spec=spec,
        canary_batch=canary_batch,
        device=device,
        epochs=epochs,
    )
    model_metrics = result.results.get(spec.model_key, {})
    receipt: dict[str, object] = {
        "receipt_format_version": 1,
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_key": spec.model_key,
        "plan_model_name": spec.plan_model_name,
        "source_commit": _source_commit(),
        "input_manifest_sha256": manifest["file_sha256"],
        "training_contract": {
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "train_target_max_week": summary["train_target_max_week"],
            "forecast_origin": FORECAST_ORIGIN,
            "window_stride": WINDOW_STRIDE,
            "epochs": epochs,
            "seed": PRODUCTION_REFIT_SEED,
            "training_mode": "production_refit",
            "validation_enabled": False,
            "state_selection": "final_epoch",
        },
        "epoch_policy": EPOCH_POLICY_EVIDENCE,
        "data_summary": summary,
        "metrics": _training_metrics(model_metrics),
        "production_canary_contract": canary_contract,
        "checkpoint": {"path": str(checkpoint_path), **checkpoint_evidence},
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    write_secure_json(model_output / "production-refit-receipt.json", receipt)
    return receipt


def run_all(
    *,
    target_source: Path,
    input_manifest: Path,
    output_root: Path,
    python_executable: Path,
    batch_size: int,
    num_workers: int,
    device: str,
    sample_part_count: int | None,
    model_specs: tuple[ExogenousQualificationModelSpec, ...] = PRODUCTION_MODEL_SPECS,
) -> dict[str, object]:
    load_training_input_manifest(input_manifest, target_source=target_source)
    if output_root.exists():
        raise V100H26ContractError(
            f"output root already exists; refusing overwrite: {output_root}"
        )
    output_root.mkdir(parents=True)
    logs_dir = output_root / "logs"
    logs_dir.mkdir()
    status_path = output_root / "production-refit-status.txt"
    status_path.write_text("RUNNING current=preflight\n", encoding="ascii")

    receipts: list[dict[str, object]] = []
    try:
        for spec in model_specs:
            epochs = PRODUCTION_REFIT_EPOCHS[spec.model_key]
            status_path.write_text(
                "RUNNING "
                f"current={spec.model_key} epochs={epochs} "
                f"seed={PRODUCTION_REFIT_SEED}\n",
                encoding="ascii",
            )
            command = build_worker_command(
                python_executable=python_executable,
                target_source=target_source,
                input_manifest=input_manifest,
                output_root=output_root,
                model_key=spec.model_key,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                sample_part_count=sample_part_count,
                preflight_only=False,
            )
            log_path = logs_dir / f"training-{spec.model_key}.log"
            with log_path.open("x", encoding="utf-8") as stream:
                stream.write(
                    "command="
                    + json.dumps(command, ensure_ascii=True)
                    + "\n"
                )
                stream.flush()
                subprocess.run(
                    command,
                    cwd=ROOT,
                    check=True,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            receipt_path = (
                output_root
                / spec.model_key
                / "production-refit-receipt.json"
            )
            receipts.append(json.loads(receipt_path.read_text(encoding="ascii")))
    except BaseException:
        status_path.write_text("FAILED\n", encoding="ascii")
        raise

    aggregate: dict[str, object] = {
        "receipt_format_version": 1,
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": "dsio-v100-weekly-l52-h26-exogenous-production-refit-v1",
        "source_commit": _source_commit(),
        "target_source_sha256": file_sha256(target_source),
        "input_manifest_sha256": file_sha256(input_manifest),
        "seed": PRODUCTION_REFIT_SEED,
        "epoch_policy": EPOCH_POLICY_EVIDENCE,
        "selected_model_keys": [spec.model_key for spec in model_specs],
        "models": receipts,
    }
    aggregate["receipt_sha256"] = canonical_json_sha256(aggregate)
    write_secure_json(output_root / "production-refit-receipt.json", aggregate)
    status_path.write_text(
        f"PASS models={len(model_specs)} seed={PRODUCTION_REFIT_SEED}\n",
        encoding="ascii",
    )
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-source", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--model-key",
        choices=tuple(PRODUCTION_REFIT_EPOCHS),
        default=None,
        help="Run one worker model. Omit to refit both models sequentially.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-keys",
        nargs="+",
        choices=tuple(PRODUCTION_REFIT_EPOCHS),
        default=None,
    )
    parser.add_argument("--sample-part-count", type=int, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target_source = args.target_source.expanduser().resolve()
    input_manifest = args.input_manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if args.model_key is not None:
        receipt = run_model(
            target_source=target_source,
            input_manifest=input_manifest,
            output_root=output_root,
            spec=MODEL_SPECS_BY_KEY[args.model_key],
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=str(args.device),
            sample_part_count=args.sample_part_count,
            preflight_only=bool(args.preflight_only),
        )
    else:
        if args.preflight_only:
            raise ValueError("--preflight-only requires an explicit --model-key.")
        receipt = run_all(
            target_source=target_source,
            input_manifest=input_manifest,
            output_root=output_root,
            python_executable=args.python.expanduser().resolve(),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=str(args.device),
            sample_part_count=args.sample_part_count,
            model_specs=tuple(
                MODEL_SPECS_BY_KEY[key]
                for key in (args.model_keys or tuple(PRODUCTION_REFIT_EPOCHS))
            ),
        )
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

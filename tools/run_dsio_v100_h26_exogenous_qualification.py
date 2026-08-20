#!/usr/bin/env python3
"""Run governed DSIO V100 L52/H26 qualification for exogenous DL models."""

from __future__ import annotations

import argparse
import gc
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping

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
    ArchitectureConfig,
    ArtifactConfig,
    ExoTSTArchitectureConfig,
    PatchTSTArchitectureConfig,
    RuntimeConfig,
    SSLConfig,
    TimexerArchitectureConfig,
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
from modeling_module.data_loader.indexed_temporal_exogenous_data_module import (  # noqa: E402
    IndexedTemporalExogenousDataModule,
)
from modeling_module.data_loader.temporal import add_period  # noqa: E402
from tools.dsio_v100_h26_contract import (  # noqa: E402
    FORECAST_ORIGIN,
    HORIZON,
    LOOKBACK,
    SEED,
    SITE_CD,
    TRAIN_END_WEEK,
    VALIDATION_ORIGIN,
    V100H26ContractError,
    canonical_json_sha256,
    file_sha256,
    load_training_input_manifest,
    write_secure_json,
)


FREQUENCY: Final = "weekly"
ID_COLUMN: Final = "oper_part_no"
DATE_COLUMN: Final = "demand_dt"
TARGET_COLUMN: Final = "demand_qty"
WINDOW_STRIDE: Final = 4
DEFAULT_EPOCHS: Final = 40
DEFAULT_BATCH_SIZE: Final = 512


@dataclass(frozen=True, slots=True)
class ExogenousQualificationModelSpec:
    """One approved Demand Engine exogenous point model."""

    model_key: str
    plan_model_name: str
    checkpoint_model_name: str
    uses_future_continuous: bool

    @property
    def checkpoint_filename(self) -> str:
        return (
            f"weekly_{self.checkpoint_model_name}_"
            f"L{LOOKBACK}_H{HORIZON}.pt"
        )


MODEL_SPECS: Final = (
    ExogenousQualificationModelSpec(
        model_key="exotst_base",
        plan_model_name="ExoTST",
        checkpoint_model_name="ExoTSTBase",
        uses_future_continuous=True,
    ),
    ExogenousQualificationModelSpec(
        model_key="timexer_base",
        plan_model_name="TimeXer",
        checkpoint_model_name="TimeXerBase",
        uses_future_continuous=False,
    ),
    ExogenousQualificationModelSpec(
        model_key="patchtst_exogenous",
        plan_model_name="PatchTST_Exo",
        checkpoint_model_name="PatchTSTExogenous",
        uses_future_continuous=True,
    ),
)
MODEL_SPECS_BY_KEY: Final = {
    spec.model_key: spec for spec in MODEL_SPECS
}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_runtime() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def build_architecture() -> ArchitectureConfig:
    """Return the frozen capacity used by the governed H26 comparison."""

    return ArchitectureConfig(
        patchtst=PatchTSTArchitectureConfig(
            patch_len=13,
            stride=6,
            d_model=128,
            n_layers=2,
            d_ff=512,
            dropout=0.1,
            norm="LayerNorm",
            pre_norm=True,
            act="gelu",
            use_revin=True,
            pe="sincos",
            learn_pe=True,
            padding_patch="end",
            future_exo_fusion_dropout=0.0,
        ),
        exotst=ExoTSTArchitectureConfig(
            patch_len=13,
            stride=6,
            d_model=128,
            n_heads=8,
            d_ff=256,
            dropout=0.1,
            attn_dropout=0.1,
            exo_enc_layers=2,
            fusion_layers=2,
            endo_dec_layers=2,
            exo_memory_mode="all",
            exo_nan_policy="zero+indicator",
            use_revin=True,
            subtract_last=True,
        ),
        timexer=TimexerArchitectureConfig(
            patch_len=13,
            d_model=128,
            n_heads=8,
            d_ff=256,
            e_layers=3,
            dropout=0.1,
            factor=5,
            activation="gelu",
            use_norm=True,
        ),
    )


def build_worker_command(
    *,
    python_executable: Path,
    target_source: Path,
    input_manifest: Path,
    output_root: Path,
    model_key: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
    device: str,
    sample_part_count: int | None,
    preflight_only: bool,
    seed: int,
) -> list[str]:
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
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--device",
        device,
        "--seed",
        str(seed),
    ]
    if sample_part_count is not None:
        command.extend(["--sample-part-count", str(sample_part_count)])
    if preflight_only:
        command.append("--preflight-only")
    return command


def _source_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_target(
    source: Path,
    *,
    sample_part_count: int | None,
    seed: int = SEED,
) -> pl.DataFrame:
    if not source.is_file():
        raise FileNotFoundError(f"target source not found: {source}")
    frame = pl.read_parquet(
        source,
        columns=[ID_COLUMN, DATE_COLUMN, TARGET_COLUMN],
    )
    if sample_part_count is not None:
        if sample_part_count <= 0:
            raise ValueError("sample_part_count must be positive.")
        part_ids = sorted(
            str(value) for value in frame[ID_COLUMN].unique().to_list()
        )
        if sample_part_count > len(part_ids):
            raise ValueError(
                "sample_part_count exceeds available series: "
                f"{sample_part_count} > {len(part_ids)}."
            )
        rng = np.random.default_rng(seed)
        selected = sorted(
            str(value)
            for value in rng.choice(
                part_ids,
                size=sample_part_count,
                replace=False,
            )
        )
        frame = frame.filter(pl.col(ID_COLUMN).cast(pl.String).is_in(selected))
    return frame


def _build_datamodule(
    frame: pl.DataFrame,
    *,
    spec: ExogenousQualificationModelSpec,
    seed: int = SEED,
) -> IndexedTemporalExogenousDataModule:
    one_table = attach_weekly_calendar_features(
        frame,
        date_column=DATE_COLUMN,
    )
    future_columns = (
        WEEKLY_CALENDAR_CONTINUOUS_FEATURES
        if spec.uses_future_continuous
        else ()
    )
    datamodule = IndexedTemporalExogenousDataModule(
        one_table,
        lookback=LOOKBACK,
        horizon=HORIZON,
        train_end_week=TRAIN_END_WEEK,
        forecast_origin=FORECAST_ORIGIN,
        validation_origin=VALIDATION_ORIGIN,
        past_exo_cont_cols=WEEKLY_CALENDAR_CONTINUOUS_FEATURES,
        future_exo_cont_cols=future_columns,
        window_stride=WINDOW_STRIDE,
        seed=seed,
        part_col=ID_COLUMN,
        date_col=DATE_COLUMN,
        qty_col=TARGET_COLUMN,
        require_all_series_eligible=False,
    )
    summary = datamodule.summary
    unsupported_exclusions = tuple(
        reason
        for reason in datamodule.ineligible_series_reasons
        if ":rows=" not in reason or ",validation_index=" not in reason
    )
    if unsupported_exclusions:
        raise V100H26ContractError(
            "series may be excluded only when pre-validation history is too "
            f"short; examples: {unsupported_exclusions[:5]}"
        )
    expected_train_target_max = add_period(VALIDATION_ORIGIN, -1, "weekly")
    if summary["train_target_max_week"] != expected_train_target_max:
        raise V100H26ContractError(
            "exogenous training windows crossed the validation boundary: "
            f"expected {expected_train_target_max}, got "
            f"{summary['train_target_max_week']}"
        )
    if summary["validation_target_min_week"] != VALIDATION_ORIGIN:
        raise V100H26ContractError(
            "exogenous validation origin drifted from the H26 contract"
        )
    if summary["validation_target_max_week"] != TRAIN_END_WEEK:
        raise V100H26ContractError(
            "exogenous validation window does not end at train_end_week"
        )
    if summary["validation_windows"] != summary["series_count"]:
        raise V100H26ContractError(
            "last-origin validation must contain exactly one window per series"
        )
    return datamodule


def _batch_contract(
    batch: tuple[Any, ...],
    *,
    spec: ExogenousQualificationModelSpec,
) -> dict[str, object]:
    if len(batch) != 6:
        raise V100H26ContractError(
            f"continuous exogenous batch must be a 6-tuple, got {len(batch)}"
        )
    x, y, part_ids, future_cont, past_cont, past_cat = batch
    expected_future_dim = (
        len(WEEKLY_CALENDAR_CONTINUOUS_FEATURES)
        if spec.uses_future_continuous
        else 0
    )
    expected = {
        "x": (None, LOOKBACK, 1),
        "y": (None, HORIZON),
        "future_cont": (None, HORIZON, expected_future_dim),
        "past_cont": (
            None,
            LOOKBACK,
            len(WEEKLY_CALENDAR_CONTINUOUS_FEATURES),
        ),
        "past_cat": (None, LOOKBACK, 0),
    }
    tensors = {
        "x": x,
        "y": y,
        "future_cont": future_cont,
        "past_cont": past_cont,
        "past_cat": past_cat,
    }
    batch_size = int(x.shape[0])
    for name, tensor in tensors.items():
        actual = tuple(int(value) for value in tensor.shape)
        wanted = tuple(
            batch_size if value is None else int(value)
            for value in expected[name]
        )
        if actual != wanted:
            raise V100H26ContractError(
                f"{name} batch shape mismatch: expected {wanted}, got {actual}"
            )
    if len(part_ids) != batch_size:
        raise V100H26ContractError("part_ids batch length mismatch")
    return {
        "batch_size": batch_size,
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "future_cont_shape": list(future_cont.shape),
        "past_cont_shape": list(past_cont.shape),
        "past_cat_shape": list(past_cat.shape),
    }


def _training_metrics(result: Mapping[str, Any]) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for key, value in result.items():
        if key == "ckpt_path":
            continue
        if isinstance(value, (str, bool, int, float)) or value is None:
            metrics[str(key)] = value
    return metrics


def _validate_checkpoint(
    *,
    checkpoint_path: Path,
    spec: ExogenousQualificationModelSpec,
    validation_batch: tuple[Any, ...],
    device: str,
    seed: int,
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
    if meta.get("model_key") != spec.model_key:
        raise V100H26ContractError("checkpoint model_key mismatch")
    if meta.get("training_mode") != "qualification":
        raise V100H26ContractError("checkpoint training_mode must be qualification")
    if meta.get("state_selection") != "best_validation":
        raise V100H26ContractError(
            "qualification checkpoint must restore the best validation state"
        )
    if meta.get("random_seed") != seed:
        raise V100H26ContractError("checkpoint random seed mismatch")

    predictor = load_predictor(
        str(checkpoint_path),
        device=device,
        strict=True,
    )
    schema = predictor.exogenous_schema
    if schema is None:
        raise V100H26ContractError(
            "checkpoint is missing the ordered exogenous schema"
        )
    expected_future = (
        WEEKLY_CALENDAR_CONTINUOUS_FEATURES
        if spec.uses_future_continuous
        else ()
    )
    if schema.past_cont_names != WEEKLY_CALENDAR_CONTINUOUS_FEATURES:
        raise V100H26ContractError("checkpoint past exogenous schema drifted")
    if schema.future_cont_names != expected_future:
        raise V100H26ContractError("checkpoint future exogenous schema drifted")

    prediction = predictor.predict(validation_batch)
    if "point" not in prediction:
        raise V100H26ContractError("checkpoint canary did not return point output")
    points = np.asarray(prediction["point"])
    if points.size == 0 or not np.isfinite(points).all():
        raise V100H26ContractError(
            "checkpoint canary returned empty or non-finite predictions"
        )
    return {
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "prediction_shape": list(points.shape),
        "prediction_finite": True,
        "exogenous_schema_fingerprint": schema.fingerprint,
    }


def run_model(
    *,
    target_source: Path,
    input_manifest: Path,
    output_root: Path,
    spec: ExogenousQualificationModelSpec,
    epochs: int,
    batch_size: int,
    num_workers: int,
    device: str,
    sample_part_count: int | None,
    preflight_only: bool,
    seed: int,
) -> dict[str, object]:
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    manifest = load_training_input_manifest(
        input_manifest,
        target_source=target_source,
    )
    set_global_seed(seed)
    configure_torch_runtime()

    target = _load_target(
        target_source,
        sample_part_count=sample_part_count,
        seed=seed,
    )
    datamodule = _build_datamodule(target, spec=spec, seed=seed)
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
    val_loader = datamodule.get_val_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
        prefetch_factor=4,
    )
    validation_batch = next(iter(val_loader))
    batch_contract = _batch_contract(validation_batch, spec=spec)
    preflight = {
        "status": "PREFLIGHT_PASS",
        "model_key": spec.model_key,
        "source_commit": _source_commit(),
        "target_sha256": file_sha256(target_source),
        "input_manifest_sha256": manifest["file_sha256"],
        "calendar_contract_version": WEEKLY_CALENDAR_CONTRACT_VERSION,
        "calendar_schema_fingerprint": weekly_calendar_schema_fingerprint(),
        "data_summary": summary,
        "batch_contract": batch_contract,
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
        model_output / "qualification-data-manifest.json",
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
                "validation_origin": VALIDATION_ORIGIN,
                "window_stride": WINDOW_STRIDE,
                "epochs": epochs,
                "seed": seed,
                "loss": "library_point_default",
                "state_selection": "best_validation",
            },
            "feature_contract": {
                "source_kind": "deterministic_calendar",
                "past_continuous_columns": list(
                    WEEKLY_CALENDAR_CONTINUOUS_FEATURES
                ),
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
            val_loader=val_loader,
            freq=FREQUENCY,
            lookback=LOOKBACK,
            horizon=HORIZON,
            models=[spec.model_key],
            architecture=build_architecture(),
            trainer=TrainerConfig(
                warmup_epochs=epochs,
                spike_epochs=0,
                lr=1e-3,
                training_mode="qualification",
                random_seed=seed,
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
            f"{spec.model_key} training did not return a checkpoint"
        )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    checkpoint_path = Path(result.primary_ckpt_path).resolve()
    checkpoint_evidence = _validate_checkpoint(
        checkpoint_path=checkpoint_path,
        spec=spec,
        validation_batch=validation_batch,
        device=device,
        seed=seed,
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
            "validation_origin": VALIDATION_ORIGIN,
            "validation_end_week": TRAIN_END_WEEK,
            "window_stride": WINDOW_STRIDE,
            "epochs": epochs,
            "seed": seed,
            "state_selection": "best_validation",
        },
        "data_summary": summary,
        "metrics": _training_metrics(model_metrics),
        "checkpoint": {
            "path": str(checkpoint_path),
            **checkpoint_evidence,
        },
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    write_secure_json(
        model_output / "qualification-receipt.json",
        receipt,
    )
    return receipt


def run_all(
    *,
    target_source: Path,
    input_manifest: Path,
    output_root: Path,
    python_executable: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    device: str,
    sample_part_count: int | None,
    seed: int,
    model_specs: tuple[ExogenousQualificationModelSpec, ...] = MODEL_SPECS,
) -> dict[str, object]:
    load_training_input_manifest(
        input_manifest,
        target_source=target_source,
    )
    if output_root.exists():
        raise V100H26ContractError(
            f"output root already exists; refusing overwrite: {output_root}"
        )
    output_root.mkdir(parents=True)
    logs_dir = output_root / "logs"
    logs_dir.mkdir()
    status_path = output_root / "qualification-status.txt"
    status_path.write_text("RUNNING current=preflight\n", encoding="ascii")

    receipts: list[dict[str, object]] = []
    try:
        for spec in model_specs:
            status_path.write_text(
                f"RUNNING current={spec.model_key} epochs={epochs}\n",
                encoding="ascii",
            )
            command = build_worker_command(
                python_executable=python_executable,
                target_source=target_source,
                input_manifest=input_manifest,
                output_root=output_root,
                model_key=spec.model_key,
                epochs=epochs,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                sample_part_count=sample_part_count,
                preflight_only=False,
                seed=seed,
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
                / "qualification-receipt.json"
            )
            receipts.append(
                json.loads(receipt_path.read_text(encoding="ascii"))
            )
    except BaseException:
        status_path.write_text("FAILED\n", encoding="ascii")
        raise

    aggregate: dict[str, object] = {
        "receipt_format_version": 1,
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": "dsio-v100-weekly-l52-h26-exogenous-qualification-v1",
        "source_commit": _source_commit(),
        "target_source_sha256": file_sha256(target_source),
        "input_manifest_sha256": file_sha256(input_manifest),
        "seed": seed,
        "selected_model_keys": [spec.model_key for spec in model_specs],
        "models": receipts,
    }
    aggregate["receipt_sha256"] = canonical_json_sha256(aggregate)
    write_secure_json(
        output_root / "qualification-receipt.json",
        aggregate,
    )
    status_path.write_text(
        f"PASS models={len(model_specs)} seed={seed}\n",
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
        choices=tuple(MODEL_SPECS_BY_KEY),
        default=None,
        help="Run one worker model. Omit to run all three sequentially.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--model-keys",
        nargs="+",
        choices=tuple(MODEL_SPECS_BY_KEY),
        default=None,
        help="Run a selected model subset sequentially. Omit for all models.",
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
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=str(args.device),
            sample_part_count=args.sample_part_count,
            preflight_only=bool(args.preflight_only),
            seed=int(args.seed),
        )
    else:
        if args.preflight_only:
            raise ValueError(
                "--preflight-only requires an explicit --model-key."
            )
        receipt = run_all(
            target_source=target_source,
            input_manifest=input_manifest,
            output_root=output_root,
            python_executable=args.python.expanduser().resolve(),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=str(args.device),
            sample_part_count=args.sample_part_count,
            seed=int(args.seed),
            model_specs=tuple(
                MODEL_SPECS_BY_KEY[key]
                for key in (args.model_keys or tuple(MODEL_SPECS_BY_KEY))
            ),
        )
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

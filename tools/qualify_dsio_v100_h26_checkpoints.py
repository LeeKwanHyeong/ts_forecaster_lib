#!/usr/bin/env python3
"""Qualify and verify the DSIO V100 L52/H26 inactive checkpoint bundle."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import sys
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final


ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.dsio_v100_h26_contract import (  # noqa: E402
    EXPECTED_RESULT_COLUMNS,
    FORECAST_ORIGIN,
    HORIZON,
    LOOKBACK,
    MODEL_SPECS,
    SITE_CD,
    TRAIN_END_WEEK,
    V100H26ContractError,
    canonical_json_sha256,
    file_sha256,
    load_training_input_manifest,
    model_signature_payload,
    require_sha256,
    write_secure_json,
)


def load_sealed_registry(path: Path) -> dict[str, Any]:
    """Load a registry and enforce the exact inactive H26 model inventory."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise V100H26ContractError("registry must be one JSON object")
    registry = {str(key): value for key, value in raw.items()}
    seal = require_sha256(
        registry.pop("registry_sha256", None), label="registry_sha256"
    )
    if canonical_json_sha256(registry) != seal:
        raise V100H26ContractError("registry seal mismatch")

    dataset = registry.get("dataset")
    runtime = registry.get("runtime")
    models = registry.get("models")
    if not isinstance(dataset, Mapping) or not isinstance(runtime, Mapping):
        raise V100H26ContractError("registry dataset/runtime is invalid")
    if dataset.get("train_end_week") != TRAIN_END_WEEK:
        raise V100H26ContractError("registry train_end_week must be 202509")
    if dataset.get("forecast_origin") != FORECAST_ORIGIN:
        raise V100H26ContractError("registry forecast_origin must be 202510")
    if runtime.get("frequency") != "weekly":
        raise V100H26ContractError("registry frequency must be weekly")
    if runtime.get("lookback") != LOOKBACK or runtime.get(
        "maximum_horizon"
    ) != HORIZON:
        raise V100H26ContractError("registry window must be L52/H26")
    if not isinstance(models, list) or len(models) != len(MODEL_SPECS):
        raise V100H26ContractError("registry model inventory drifted")
    for definition, spec in zip(models, MODEL_SPECS, strict=True):
        if not isinstance(definition, Mapping):
            raise V100H26ContractError("registry model definition is invalid")
        checkpoint = definition.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise V100H26ContractError("registry checkpoint is invalid")
        if (
            definition.get("model_key") != spec.model_key
            or definition.get("plan_model_name") != spec.plan_model_name
            or checkpoint.get("path") != spec.checkpoint_filename
        ):
            raise V100H26ContractError("registry model inventory drifted")
        require_sha256(
            checkpoint.get("sha256"),
            label=f"{spec.model_key}.checkpoint.sha256",
        )
    return {**registry, "registry_sha256": seal}


def build_smoke_frame(polars: Any) -> Any:
    """Build one deterministic L52 series ending at ISO week 202509."""

    origin_date = date.fromisocalendar(2025, 10, 1)
    periods: list[int] = []
    for offset in range(LOOKBACK, 0, -1):
        historical = origin_date - timedelta(weeks=offset)
        iso_year, iso_week, _ = historical.isocalendar()
        periods.append(iso_year * 100 + iso_week)
    return polars.DataFrame(
        {
            "oper_part_no": ["V100_SMOKE_0001"] * LOOKBACK,
            "demand_dt": periods,
            "demand_qty": [float(10 + index % 13) for index in range(LOOKBACK)],
        }
    )


def validate_predictions(predictions: Any, *, model_key: str) -> dict[str, object]:
    """Enforce exactly one W0-W25 Candidate set for one model."""

    if tuple(predictions.columns) != EXPECTED_RESULT_COLUMNS:
        raise V100H26ContractError(f"{model_key} result schema drifted")
    if predictions.height != HORIZON:
        raise V100H26ContractError(
            f"{model_key} produced {predictions.height} rows instead of 26"
        )
    if predictions.get_column("horizon_step").to_list() != list(range(HORIZON)):
        raise V100H26ContractError(f"{model_key} offsets are not W0-W25")
    if predictions.get_column("model_key").unique().to_list() != [model_key]:
        raise V100H26ContractError(f"{model_key} result identity drifted")
    if predictions.get_column("forecast_origin").unique().to_list() != [
        FORECAST_ORIGIN
    ]:
        raise V100H26ContractError(f"{model_key} origin drifted")
    points = predictions.get_column("point").to_list()
    if any(value is None or not math.isfinite(float(value)) for value in points):
        raise V100H26ContractError(f"{model_key} produced a non-finite point")
    return {
        "candidate_rows": HORIZON,
        "horizon_step_min": 0,
        "horizon_step_max": HORIZON - 1,
        "point_min": min(float(value) for value in points),
        "point_max": max(float(value) for value in points),
        "ordered_result_columns": list(EXPECTED_RESULT_COLUMNS),
    }


def qualify(
    *,
    registry_path: Path,
    checkpoint_root: Path,
    input_manifest_path: Path,
    receipt_path: Path,
    device: str,
) -> dict[str, object]:
    """Hash, load, forecast, and seal all five inactive checkpoints."""

    import modeling_module
    import polars as pl
    import torch

    registry = load_sealed_registry(registry_path)
    manifest = load_training_input_manifest(input_manifest_path)
    frame = build_smoke_frame(pl)
    model_receipts: list[dict[str, object]] = []
    for definition, spec in zip(registry["models"], MODEL_SPECS, strict=True):
        checkpoint = definition["checkpoint"]
        checkpoint_path = checkpoint_root / spec.checkpoint_filename
        observed_sha256 = file_sha256(checkpoint_path)
        if observed_sha256 != checkpoint["sha256"]:
            raise V100H26ContractError(
                f"checkpoint SHA-256 mismatch: {spec.checkpoint_filename}"
            )
        request = modeling_module.ForecastRequest(
            checkpoint_path=checkpoint_path,
            expected_model_key=spec.model_key,
            data=modeling_module.DataRequest(
                df=frame,
                backend="exo",
                window=modeling_module.DataWindowConfig(
                    lookback=LOOKBACK,
                    horizon=HORIZON,
                    freq="weekly",
                ),
                columns=modeling_module.DataColumnConfig(
                    id_col="oper_part_no",
                    date_col="demand_dt",
                    y_col="demand_qty",
                ),
                exogenous=modeling_module.ExogenousConfig(
                    use_exogenous_mode=False,
                    use_past_exogenous=False,
                    use_future_exogenous=False,
                    past_exo_cont_cols=[],
                    past_exo_cat_cols=[],
                    future_exo_cont_cols=[],
                    fill_missing="zero",
                ),
            ),
            series_ids=["V100_SMOKE_0001"],
            forecast_origin=FORECAST_ORIGIN,
            runtime=modeling_module.ForecastRuntimeConfig(
                batch_size=1,
                num_workers=0,
                device=device,
                pin_memory=device.startswith("cuda"),
                persistent_workers=False,
            ),
            unknown_series_policy="error",
        )
        result = modeling_module.forecast(request)
        if result.model_key != spec.model_key:
            raise V100H26ContractError(f"{spec.model_key} result identity drifted")
        prediction = validate_predictions(
            result.predictions,
            model_key=spec.model_key,
        )
        signature_payload = model_signature_payload(
            model_key=spec.model_key,
            checkpoint_sha256=observed_sha256,
            input_manifest_sha256=manifest["file_sha256"],
        )
        model_receipts.append(
            {
                "plan_model_name": spec.plan_model_name,
                "model_key": spec.model_key,
                "checkpoint": {
                    "path": spec.checkpoint_filename,
                    "sha256": observed_sha256,
                },
                "model_signature_sha256": canonical_json_sha256(
                    signature_payload
                ),
                "public_forecast": prediction,
            }
        )
        del result
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    receipt: dict[str, object] = {
        "receipt_format_version": 1,
        "status": "PASS",
        "scope": {
            "site_cd": SITE_CD,
            "train_end_week": TRAIN_END_WEEK,
            "forecast_origin": FORECAST_ORIGIN,
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "offsets": "W0-W25",
        },
        "registry": {
            "path": registry_path.name,
            "file_sha256": file_sha256(registry_path),
            "registry_sha256": registry["registry_sha256"],
        },
        "training_input_manifest": {
            "path": input_manifest_path.name,
            "sha256": manifest["file_sha256"],
        },
        "runtime": {
            "modeling_module_version": importlib.metadata.version(
                "modeling-module"
            ),
            "device": device,
            "db_write_enabled": False,
        },
        "models": model_receipts,
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    write_secure_json(receipt_path, receipt)
    validate_qualification_receipt(
        receipt_path=receipt_path,
        registry_path=registry_path,
        checkpoint_root=checkpoint_root,
        input_manifest_path=input_manifest_path,
    )
    return receipt


def validate_qualification_receipt(
    *,
    receipt_path: Path,
    registry_path: Path,
    checkpoint_root: Path,
    input_manifest_path: Path,
) -> dict[str, Any]:
    """Recompute every external binding represented by a qualification receipt."""

    raw = json.loads(receipt_path.read_text(encoding="ascii"))
    if not isinstance(raw, Mapping):
        raise V100H26ContractError("qualification receipt is invalid")
    receipt = {str(key): value for key, value in raw.items()}
    seal = require_sha256(
        receipt.pop("receipt_sha256", None), label="receipt_sha256"
    )
    if canonical_json_sha256(receipt) != seal:
        raise V100H26ContractError("qualification receipt seal mismatch")
    registry = load_sealed_registry(registry_path)
    manifest = load_training_input_manifest(input_manifest_path)
    if receipt.get("status") != "PASS" or receipt.get("scope") != {
        "site_cd": SITE_CD,
        "train_end_week": TRAIN_END_WEEK,
        "forecast_origin": FORECAST_ORIGIN,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "offsets": "W0-W25",
    }:
        raise V100H26ContractError("qualification receipt scope drifted")
    if receipt.get("registry") != {
        "path": registry_path.name,
        "file_sha256": file_sha256(registry_path),
        "registry_sha256": registry["registry_sha256"],
    }:
        raise V100H26ContractError("qualification receipt registry binding drifted")
    if receipt.get("training_input_manifest") != {
        "path": input_manifest_path.name,
        "sha256": manifest["file_sha256"],
    }:
        raise V100H26ContractError("qualification receipt manifest binding drifted")

    models = receipt.get("models")
    if not isinstance(models, list) or len(models) != len(MODEL_SPECS):
        raise V100H26ContractError("qualification receipt model inventory drifted")
    for model, definition, spec in zip(
        models,
        registry["models"],
        MODEL_SPECS,
        strict=True,
    ):
        if not isinstance(model, Mapping):
            raise V100H26ContractError("qualification model receipt is invalid")
        checkpoint_path = checkpoint_root / spec.checkpoint_filename
        checkpoint_sha256 = file_sha256(checkpoint_path)
        if checkpoint_sha256 != definition["checkpoint"]["sha256"]:
            raise V100H26ContractError(
                f"checkpoint SHA-256 mismatch: {spec.checkpoint_filename}"
            )
        if model.get("checkpoint") != {
            "path": spec.checkpoint_filename,
            "sha256": checkpoint_sha256,
        }:
            raise V100H26ContractError("qualification checkpoint binding drifted")
        signature = canonical_json_sha256(
            model_signature_payload(
                model_key=spec.model_key,
                checkpoint_sha256=checkpoint_sha256,
                input_manifest_sha256=manifest["file_sha256"],
            )
        )
        if (
            model.get("plan_model_name") != spec.plan_model_name
            or model.get("model_key") != spec.model_key
            or model.get("model_signature_sha256") != signature
        ):
            raise V100H26ContractError("qualification model signature drifted")
        public_forecast = model.get("public_forecast")
        if not isinstance(public_forecast, Mapping) or (
            public_forecast.get("candidate_rows") != HORIZON
            or public_forecast.get("horizon_step_min") != 0
            or public_forecast.get("horizon_step_max") != HORIZON - 1
            or public_forecast.get("ordered_result_columns")
            != list(EXPECTED_RESULT_COLUMNS)
        ):
            raise V100H26ContractError("qualification Candidate contract drifted")
    return {**receipt, "receipt_sha256": seal}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("qualify", "verify-receipt"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--registry", type=Path, required=True)
        subparser.add_argument("--checkpoint-root", type=Path, required=True)
        subparser.add_argument("--input-manifest", type=Path, required=True)
        subparser.add_argument("--receipt", type=Path, required=True)
        if command == "qualify":
            subparser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    common = {
        "registry_path": args.registry.expanduser().resolve(),
        "checkpoint_root": args.checkpoint_root.expanduser().resolve(),
        "input_manifest_path": args.input_manifest.expanduser().resolve(),
        "receipt_path": args.receipt.expanduser().resolve(),
    }
    if args.command == "qualify":
        result = qualify(**common, device=str(args.device))
    else:
        result = validate_qualification_receipt(**common)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

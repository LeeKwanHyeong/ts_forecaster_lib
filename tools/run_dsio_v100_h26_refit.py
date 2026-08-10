#!/usr/bin/env python3
"""Run the governed five-model DSIO V100 Weekly L52/H26 production refit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import torch


ROOT: Final = Path(__file__).resolve().parents[1]
RUNNER: Final = (
    ROOT / "src/model_test/total_train/dsio_total_running.py"
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.dsio_v100_h26_contract import (  # noqa: E402
    FORECAST_ORIGIN,
    HORIZON,
    LOOKBACK,
    MODEL_SPECS,
    SEED,
    TRAIN_END_WEEK,
    VALIDATION_ORIGIN,
    V100H26ContractError,
    canonical_json_sha256,
    file_sha256,
    load_training_input_manifest,
    model_signature_payload,
    validate_checkpoint_contract,
    write_secure_json,
)


def build_training_command(
    *,
    python_executable: Path,
    target_source: Path,
    artifact_root: Path,
    model_key: str,
    epochs: int,
    preflight_only: bool,
    device: str,
    num_workers: int,
) -> list[str]:
    """Build one explicit model command without changing generic defaults."""

    command = [
        str(python_executable),
        str(RUNNER),
        "--mode",
        "endo",
        "--training-mode",
        "production_refit",
        "--artifact-root",
        str(artifact_root),
        "--target-source",
        str(target_source),
        "--endo-models",
        model_key,
        "--lookback",
        str(LOOKBACK),
        "--horizon",
        str(HORIZON),
        "--endo-loader-backend",
        "indexed_temporal",
        "--train-end-week",
        str(TRAIN_END_WEEK),
        "--forecast-origin",
        str(FORECAST_ORIGIN),
        "--validation-origin",
        str(VALIDATION_ORIGIN),
        "--window-stride",
        "4",
        "--endo-batch-size",
        "1024",
        "--num-workers",
        str(num_workers),
        "--prefetch-factor",
        "4",
        "--pin-memory",
        "--persistent-workers",
        "--warmup-epochs",
        str(epochs),
        "--spike-epochs",
        "0",
        "--ssl-mode",
        "sl_only",
        "--device",
        device,
        "--seed",
        str(SEED),
    ]
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


def _run_logged(command: list[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as stream:
        stream.write("command=" + json.dumps(command, ensure_ascii=True) + "\n")
        stream.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


def run_refit(
    *,
    target_source: Path,
    input_manifest: Path,
    output_root: Path,
    python_executable: Path,
    device: str,
    num_workers: int,
    validate_only: bool,
) -> dict[str, Any]:
    """Validate, preflight, train, and seal all approved model artifacts."""

    manifest = load_training_input_manifest(
        input_manifest,
        target_source=target_source,
    )
    commands: list[dict[str, object]] = []
    for spec in MODEL_SPECS:
        commands.append(
            {
                "model_key": spec.model_key,
                "epochs": spec.epochs,
                "checkpoint_filename": spec.checkpoint_filename,
                "preflight": build_training_command(
                    python_executable=python_executable,
                    target_source=target_source,
                    artifact_root=output_root / "preflight" / spec.model_key,
                    model_key=spec.model_key,
                    epochs=spec.epochs,
                    preflight_only=True,
                    device=device,
                    num_workers=num_workers,
                ),
                "training": build_training_command(
                    python_executable=python_executable,
                    target_source=target_source,
                    artifact_root=output_root / "training" / spec.model_key,
                    model_key=spec.model_key,
                    epochs=spec.epochs,
                    preflight_only=False,
                    device=device,
                    num_workers=num_workers,
                ),
            }
        )
    if validate_only:
        return {
            "status": "VALIDATED",
            "contract": "dsio-v100-weekly-l52-h26-production-refit-v1",
            "input_manifest_sha256": manifest["file_sha256"],
            "commands": commands,
        }
    if output_root.exists():
        raise V100H26ContractError(
            "output root already exists; refusing duplicate production refit"
        )

    output_root.mkdir(parents=True)
    status_path = output_root / "training-status.txt"
    status_path.write_text("RUNNING current=preflight\n", encoding="ascii")
    model_receipts: list[dict[str, object]] = []
    try:
        for spec, command_set in zip(MODEL_SPECS, commands, strict=True):
            _run_logged(
                list(command_set["preflight"]),
                log_path=output_root / "logs" / f"preflight-{spec.model_key}.log",
            )
            status_path.write_text(
                f"RUNNING current={spec.model_key} epochs={spec.epochs}\n",
                encoding="ascii",
            )
            _run_logged(
                list(command_set["training"]),
                log_path=output_root / "logs" / f"training-{spec.model_key}.log",
            )
            checkpoint_path = (
                output_root
                / "training"
                / spec.model_key
                / "endo_only"
                / spec.checkpoint_filename
            )
            if not checkpoint_path.is_file():
                raise V100H26ContractError(
                    f"missing checkpoint: {spec.checkpoint_filename}"
                )
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            validate_checkpoint_contract(checkpoint, spec=spec)
            checkpoint_sha256 = file_sha256(checkpoint_path)
            signature_payload = model_signature_payload(
                model_key=spec.model_key,
                checkpoint_sha256=checkpoint_sha256,
                input_manifest_sha256=manifest["file_sha256"],
            )
            model_receipts.append(
                {
                    "plan_model_name": spec.plan_model_name,
                    "model_key": spec.model_key,
                    "epochs": spec.epochs,
                    "checkpoint": {
                        "path": spec.checkpoint_filename,
                        "sha256": checkpoint_sha256,
                        "size_bytes": checkpoint_path.stat().st_size,
                    },
                    "model_signature_sha256": canonical_json_sha256(
                        signature_payload
                    ),
                }
            )
    except BaseException:
        status_path.write_text("FAILED\n", encoding="ascii")
        raise

    payload: dict[str, object] = {
        "receipt_format_version": 1,
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "site_cd": "V100",
            "train_end_week": TRAIN_END_WEEK,
            "forecast_origin": FORECAST_ORIGIN,
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "offsets": "W0-W25",
            "seed": SEED,
            "mode": "production_refit",
            "state_selection": "final_epoch",
        },
        "source_commit": _source_commit(),
        "training_input_manifest": {
            "path": input_manifest.name,
            "sha256": manifest["file_sha256"],
            "payload_sha256": manifest["payload_sha256"],
        },
        "models": model_receipts,
    }
    payload["receipt_sha256"] = canonical_json_sha256(payload)
    write_secure_json(output_root / "training-receipt.json", payload)
    status_path.write_text("PASS models=5\n", encoding="ascii")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-source", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run_refit(
        target_source=args.target_source.expanduser().resolve(),
        input_manifest=args.input_manifest.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        python_executable=args.python.expanduser().resolve(),
        device=str(args.device),
        num_workers=int(args.num_workers),
        validate_only=bool(args.validate_only),
    )
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

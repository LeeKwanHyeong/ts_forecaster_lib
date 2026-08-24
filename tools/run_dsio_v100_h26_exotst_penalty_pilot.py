#!/usr/bin/env python3
"""Run and gate the governed ExoTST negative-output penalty pilot."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final


ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.dsio_v100_h26_contract import (  # noqa: E402
    canonical_json_sha256,
    file_sha256,
    write_secure_json,
)


CONTRACT_ID: Final = "dsio-v100-h26-exotst-negative-penalty-pilot-v1"
MODEL_KEY: Final = "exotst_base"
PENALTY_WEIGHTS: Final = (0.01, 0.1, 1.0)
MINIMUM_MAE_IMPROVEMENT_PERCENT: Final = 1.0
MINIMUM_NEGATIVE_RATE_REDUCTION_PERCENT: Final = 50.0
MAXIMUM_ABSOLUTE_BIAS_DEGRADATION_PP: Final = 1.0


def _weight_slug(weight: float) -> str:
    return str(weight).replace(".", "p")


def _read_policy_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    selected = {
        row["output_policy"]: row
        for row in rows
        if row["model_key"] == MODEL_KEY
    }
    if set(selected) != {"raw", "nonnegative"}:
        raise ValueError(
            f"{path} must contain raw and nonnegative rows for {MODEL_KEY}"
        )
    return selected


def evaluate_pilot_gate(
    *,
    control_rows: dict[str, dict[str, str]],
    candidate_rows: dict[str, dict[str, str]],
) -> dict[str, object]:
    """Compare one candidate against the frozen seed-42 control."""

    control_raw = control_rows["raw"]
    control_clip = control_rows["nonnegative"]
    candidate_raw = candidate_rows["raw"]
    candidate_clip = candidate_rows["nonnegative"]
    control_mae = float(control_clip["mae"])
    candidate_mae = float(candidate_clip["mae"])
    control_negative_rate = float(control_raw["negative_prediction_rate"])
    candidate_negative_rate = float(candidate_raw["negative_prediction_rate"])
    control_bias = abs(float(control_clip["normalized_bias"]))
    candidate_bias = abs(float(candidate_clip["normalized_bias"]))

    mae_improvement_percent = 100.0 * (
        control_mae - candidate_mae
    ) / control_mae
    negative_rate_reduction_percent = 100.0 * (
        control_negative_rate - candidate_negative_rate
    ) / control_negative_rate
    absolute_bias_degradation_pp = 100.0 * (
        candidate_bias - control_bias
    )
    gates = {
        "mae_improvement": (
            mae_improvement_percent >= MINIMUM_MAE_IMPROVEMENT_PERCENT
        ),
        "negative_rate_reduction": (
            negative_rate_reduction_percent
            >= MINIMUM_NEGATIVE_RATE_REDUCTION_PERCENT
        ),
        "absolute_normalized_bias": (
            absolute_bias_degradation_pp
            <= MAXIMUM_ABSOLUTE_BIAS_DEGRADATION_PP
        ),
    }
    return {
        "control": {
            "clip_mae": control_mae,
            "raw_negative_rate": control_negative_rate,
            "clip_normalized_bias": float(
                control_clip["normalized_bias"]
            ),
        },
        "candidate": {
            "clip_mae": candidate_mae,
            "raw_negative_rate": candidate_negative_rate,
            "clip_normalized_bias": float(
                candidate_clip["normalized_bias"]
            ),
        },
        "delta": {
            "clip_mae_improvement_percent": mae_improvement_percent,
            "raw_negative_rate_reduction_percent": (
                negative_rate_reduction_percent
            ),
            "absolute_normalized_bias_degradation_percentage_points": (
                absolute_bias_degradation_pp
            ),
        },
        "gates": gates,
        "passes_all_gates": all(gates.values()),
    }


def _run(command: list[str], *, log_path: Path) -> float:
    started = time.perf_counter()
    with log_path.open("x", encoding="utf-8") as stream:
        stream.write("command=" + json.dumps(command) + "\n")
        stream.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    return time.perf_counter() - started


def run_pilot(args: argparse.Namespace) -> dict[str, object]:
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"pilot output root already exists: {output_root}"
        )
    output_root.mkdir(parents=True)
    logs_dir = output_root / "logs"
    logs_dir.mkdir()
    status_path = output_root / "pilot-status.txt"
    status_path.write_text("RUNNING current=setup\n", encoding="ascii")

    control_path = args.control_overall_csv.expanduser().resolve()
    control_rows = _read_policy_rows(control_path)
    candidate_results: list[dict[str, object]] = []
    try:
        for weight in PENALTY_WEIGHTS:
            slug = _weight_slug(weight)
            candidate_root = output_root / f"lambda_{slug}"
            status_path.write_text(
                f"RUNNING current=lambda_{slug} epochs={args.epochs}\n",
                encoding="ascii",
            )
            training_command = [
                str(args.python),
                str(ROOT / "tools/run_dsio_v100_h26_exogenous_qualification.py"),
                "--target-source",
                str(args.target_source),
                "--input-manifest",
                str(args.input_manifest),
                "--output-root",
                str(candidate_root),
                "--model-keys",
                MODEL_KEY,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--negative-output-penalty-weight",
                str(weight),
                "--python",
                str(args.python),
            ]
            training_seconds = _run(
                training_command,
                log_path=logs_dir / f"lambda_{slug}-training.log",
            )
            evaluation_command = [
                str(args.python),
                str(ROOT / "tools/evaluate_dsio_v100_h26_exogenous_qualification.py"),
                "--target-source",
                str(args.target_source),
                "--input-manifest",
                str(args.input_manifest),
                "--qualification-root",
                str(candidate_root),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--device",
                args.device,
                "--model-keys",
                MODEL_KEY,
            ]
            evaluation_seconds = _run(
                evaluation_command,
                log_path=logs_dir / f"lambda_{slug}-evaluation.log",
            )
            overall_path = (
                candidate_root
                / "validation-evaluation/validation-overall.csv"
            )
            checkpoint_path = (
                candidate_root
                / MODEL_KEY
                / "weekly_ExoTSTBase_L52_H26.pt"
            )
            gate = evaluate_pilot_gate(
                control_rows=control_rows,
                candidate_rows=_read_policy_rows(overall_path),
            )
            candidate_results.append({
                "negative_output_penalty_weight": weight,
                "training_seconds": training_seconds,
                "evaluation_seconds": evaluation_seconds,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "validation_overall_path": str(overall_path),
                "validation_overall_sha256": file_sha256(overall_path),
                "gate": gate,
            })
    except BaseException:
        status_path.write_text("FAILED\n", encoding="ascii")
        raise

    passing = [
        result for result in candidate_results
        if result["gate"]["passes_all_gates"]
    ]
    selected = (
        min(
            passing,
            key=lambda result: result["gate"]["candidate"]["clip_mae"],
        )
        if passing
        else None
    )
    receipt: dict[str, object] = {
        "contract_id": CONTRACT_ID,
        "status": "PASS",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "source_overlay_sha256_path": str(
            args.source_overlay_sha256.expanduser().resolve()
        ),
        "source_overlay_sha256": file_sha256(
            args.source_overlay_sha256.expanduser().resolve()
        ),
        "target_source_sha256": file_sha256(args.target_source),
        "input_manifest_sha256": file_sha256(args.input_manifest),
        "control_validation_overall_sha256": file_sha256(control_path),
        "training_contract": {
            "model_key": MODEL_KEY,
            "lookback": 52,
            "horizon": 26,
            "seed": args.seed,
            "epochs": args.epochs,
            "penalty_weights": list(PENALTY_WEIGHTS),
            "state_selection": "best_validation_base_point_loss",
        },
        "gate_contract": {
            "minimum_clip_mae_improvement_percent": (
                MINIMUM_MAE_IMPROVEMENT_PERCENT
            ),
            "minimum_raw_negative_rate_reduction_percent": (
                MINIMUM_NEGATIVE_RATE_REDUCTION_PERCENT
            ),
            "maximum_absolute_normalized_bias_degradation_percentage_points": (
                MAXIMUM_ABSOLUTE_BIAS_DEGRADATION_PP
            ),
        },
        "candidates": candidate_results,
        "decision": {
            "multiseed_approved": selected is not None,
            "selected_negative_output_penalty_weight": (
                None
                if selected is None
                else selected["negative_output_penalty_weight"]
            ),
            "reason": (
                "at_least_one_candidate_passed_all_pilot_gates"
                if selected is not None
                else "no_candidate_passed_all_pilot_gates"
            ),
        },
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    write_secure_json(output_root / "pilot-receipt.json", receipt)
    status_path.write_text(
        "PASS multiseed_approved="
        f"{str(selected is not None).lower()}\n",
        encoding="ascii",
    )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-source", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--control-overall-csv", type=Path, required=True)
    parser.add_argument("--source-overlay-sha256", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run_pilot(args)
    print(json.dumps(receipt["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

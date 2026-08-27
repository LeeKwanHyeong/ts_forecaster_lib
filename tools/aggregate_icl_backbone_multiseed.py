#!/usr/bin/env python3
"""Aggregate sealed ICL backbone qualification receipts across random seeds."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Final


ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.qualify_icl_backbones_5090 import _sha256_payload  # noqa: E402


AGGREGATE_CONTRACT: Final = "modeling_module.icl_backbone_multiseed.v1"
EXPECTED_BACKBONE: Final = "Qwen/Qwen2-0.5B"
EXPECTED_BACKBONE_REVISION: Final = "91d2aff3f957f99e4c74c962f2f408dcc88a18d8"
METRICS: Final = ("mae", "wape", "smape", "bias", "raw_negative_rate")
MODEL_KEYS: Final = ("autotimes_base", "sellm_base")


class AggregateError(RuntimeError):
    """Raised when input receipts cannot form one comparable aggregate."""


def _load_receipt(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = str(receipt.get("receipt_sha256") or "")
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    if _sha256_payload(payload) != expected:
        raise AggregateError(f"Receipt seal is invalid: {path}.")
    if receipt.get("qualification", {}).get("status") != "PASS":
        raise AggregateError(f"Qualification did not pass: {path}.")
    results = receipt.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise AggregateError(f"Receipt must contain exactly one model result: {path}.")
    return receipt


def _summary(values: list[float]) -> dict[str, float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise AggregateError("Aggregate metrics must be finite and non-empty.")
    return {
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate_receipts(
    receipt_paths: list[Path],
    *,
    expected_seeds: tuple[int, ...],
    max_mae_cv: float,
    max_abs_bias: float,
    max_raw_negative_rate: float,
) -> dict[str, Any]:
    receipts = [(path, _load_receipt(path)) for path in receipt_paths]
    groups: dict[str, list[tuple[Path, dict[str, Any]]]] = {
        key: [] for key in MODEL_KEYS
    }
    common_contracts: set[str] = set()
    model_contracts: dict[str, set[str]] = {key: set() for key in MODEL_KEYS}
    for path, receipt in receipts:
        qualification = receipt["qualification"]
        backbone = receipt["backbone"]
        result = receipt["results"][0]
        model_key = str(result.get("model_key") or "")
        if model_key not in groups:
            raise AggregateError(f"Unexpected model key in {path}: {model_key!r}.")
        if backbone.get("model_id") != EXPECTED_BACKBONE:
            raise AggregateError(f"Unexpected backbone model in {path}.")
        if backbone.get("revision") != EXPECTED_BACKBONE_REVISION:
            raise AggregateError(f"Unexpected backbone revision in {path}.")
        if qualification.get("horizons") != [26]:
            raise AggregateError(f"Only H26 receipts are comparable: {path}.")
        if int(qualification.get("sample_series") or 0) != 256:
            raise AggregateError(f"Receipt does not use 256 series: {path}.")
        if int(qualification.get("batch_size") or 0) != 4:
            raise AggregateError(f"Receipt does not use batch size 4: {path}.")
        if int(qualification.get("epochs") or 0) != 5:
            raise AggregateError(f"Receipt does not use five epochs: {path}.")
        accuracy = result.get("accuracy") or {}
        if any(name not in accuracy for name in METRICS):
            raise AggregateError(f"Receipt lacks operating metrics: {path}.")
        if float(result.get("checkpoint", {}).get("reload_max_abs_delta") or 0.0) > 1e-6:
            raise AggregateError(f"Checkpoint reload parity failed: {path}.")
        common_contracts.add(
            _sha256_payload(
                {
                    "source_revision": receipt["input"]["source_revision"],
                    "exogenous_source_revision": qualification[
                        "exogenous_source_revision"
                    ],
                    "episode_manifest_hash": receipt["episodes"]["26"][
                        "manifest_hash"
                    ],
                    "backbone_contract_sha256": backbone["contract_sha256"],
                }
            )
        )
        model_contracts[model_key].add(
            _sha256_payload(
                {
                    "model_key": model_key,
                    "learning_rate": qualification["learning_rate"],
                }
            )
        )
        groups[model_key].append((path, receipt))

    if len(common_contracts) != 1:
        raise AggregateError("Receipts do not share one source, episode, and backbone contract.")
    if any(len(contracts) != 1 for contracts in model_contracts.values()):
        raise AggregateError("Receipts do not share one learning rate per model family.")

    expected = set(int(value) for value in expected_seeds)
    model_summaries: dict[str, Any] = {}
    operational_candidates: list[str] = []
    for model_key, items in groups.items():
        observed = {int(receipt["qualification"]["seed"]) for _, receipt in items}
        if observed != expected or len(items) != len(expected):
            raise AggregateError(
                f"{model_key} seeds differ: expected={sorted(expected)}, "
                f"observed={sorted(observed)}."
            )
        ordered = sorted(items, key=lambda item: int(item[1]["qualification"]["seed"]))
        values = {
            name: [float(receipt["results"][0]["accuracy"][name]) for _, receipt in ordered]
            for name in METRICS
        }
        metrics = {name: _summary(metric_values) for name, metric_values in values.items()}
        mae_mean = metrics["mae"]["mean"]
        mae_cv = metrics["mae"]["population_std"] / mae_mean if mae_mean else 0.0
        gates = {
            "checkpoint_reload_parity": True,
            "mae_cv_within_limit": mae_cv <= float(max_mae_cv),
            "bias_within_limit": max(abs(value) for value in values["bias"])
            <= float(max_abs_bias),
            "raw_negative_rate_within_limit": metrics["raw_negative_rate"]["mean"]
            <= float(max_raw_negative_rate),
        }
        passed = all(gates.values())
        if passed:
            operational_candidates.append(model_key)
        model_summaries[model_key] = {
            "status": "PASS" if passed else "FAIL",
            "seeds": [int(receipt["qualification"]["seed"]) for _, receipt in ordered],
            "metrics": metrics,
            "seed_values": {
                name: {
                    str(receipt["qualification"]["seed"]): value
                    for (_, receipt), value in zip(ordered, metric_values)
                }
                for name, metric_values in values.items()
            },
            "mae_cv": mae_cv,
            "training_seconds": _summary(
                [float(receipt["results"][0]["training"]["seconds"]) for _, receipt in ordered]
            ),
            "peak_allocated_mib": _summary(
                [
                    float(receipt["results"][0]["training"]["peak_allocated_mib"])
                    for _, receipt in ordered
                ]
            ),
            "gates": gates,
            "receipts": [
                {
                    "path": str(path),
                    "receipt_sha256": receipt["receipt_sha256"],
                    "checkpoint_sha256": receipt["results"][0]["checkpoint"]["sha256"],
                }
                for path, receipt in ordered
            ],
        }

    recommended_default = None
    if operational_candidates:
        recommended_default = min(
            operational_candidates,
            key=lambda key: (
                model_summaries[key]["metrics"]["wape"]["mean"],
                model_summaries[key]["metrics"]["mae"]["mean"],
            ),
        )
    aggregate = {
        "contract": AGGREGATE_CONTRACT,
        "status": "PASS" if operational_candidates else "FAIL",
        "backbone": {
            "model_id": EXPECTED_BACKBONE,
            "revision": EXPECTED_BACKBONE_REVISION,
        },
        "expected_seeds": sorted(expected),
        "gates": {
            "max_mae_cv": float(max_mae_cv),
            "max_abs_bias": float(max_abs_bias),
            "max_mean_raw_negative_rate": float(max_raw_negative_rate),
        },
        "models": model_summaries,
        "operational_candidates": operational_candidates,
        "recommended_default": recommended_default,
    }
    aggregate["receipt_sha256"] = _sha256_payload(aggregate)
    return aggregate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--max-mae-cv", type=float, default=0.10)
    parser.add_argument("--max-abs-bias", type=float, default=0.10)
    parser.add_argument("--max-raw-negative-rate", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    aggregate = aggregate_receipts(
        [path.expanduser().resolve() for path in args.receipt],
        expected_seeds=tuple(args.expected_seeds),
        max_mae_cv=float(args.max_mae_cv),
        max_abs_bias=float(args.max_abs_bias),
        max_raw_negative_rate=float(args.max_raw_negative_rate),
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": aggregate["status"],
                "receipt_sha256": aggregate["receipt_sha256"],
                "operational_candidates": aggregate["operational_candidates"],
                "recommended_default": aggregate["recommended_default"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

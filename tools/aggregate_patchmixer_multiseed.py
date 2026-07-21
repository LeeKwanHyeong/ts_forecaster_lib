#!/usr/bin/env python3
"""Validate PatchMixer benchmark contracts and aggregate seeds 11, 22, and 33."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_SEEDS = (11, 22, 33)
MODELS = ("original", "enhanced")
SCOPES = (
    "test_all_rolling_windows",
    "test_last_origin_per_series",
)
ERROR_METRICS = ("mae", "mse", "rmse", "smape", "wape")
SEED_PROTOCOL_FIELDS = {
    "seed",
    "initialization_seed",
    "training_randomness_seed",
    "dataloader_seed",
    "split_fingerprint",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accuracy-results",
        type=Path,
        nargs="+",
        required=True,
        help="Single-seed accuracy JSON files; exactly seeds 11, 22, and 33 are required.",
    )
    parser.add_argument(
        "--performance-result",
        type=Path,
        required=True,
        help="RTX 5090 100-step BF16 benchmark JSON.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _without(mapping: Mapping[str, Any], fields: set[str]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in fields}


def _assert_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _relative_improvement(original: float, enhanced: float) -> float:
    if enhanced == 0.0:
        if original == 0.0:
            return 0.0
        raise ValueError("Cannot compute relative improvement against a zero Enhanced metric.")
    return (enhanced - original) / enhanced * 100.0


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty value sequence.")
    numeric = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError(f"Non-finite aggregate value: {numeric!r}")
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "sample_stddev": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
        "median": statistics.median(numeric),
        "min": min(numeric),
        "max": max(numeric),
    }


def _summarize_seed_values(seed_values: Sequence[tuple[int, float]]) -> dict[str, Any]:
    ordered = sorted((int(seed), float(value)) for seed, value in seed_values)
    return {
        "by_seed": {str(seed): value for seed, value in ordered},
        **_stats([value for _, value in ordered]),
    }


def _validate_split(dataset: Mapping[str, Any], seed: int) -> None:
    splits = dataset.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError(f"seed {seed}: dataset.splits must be an object")

    expected_names = {"train", "validation", "test"}
    _assert_equal(f"seed {seed}: split names", set(splits), expected_names)
    split_sets = {name: set(values) for name, values in splits.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if not split_sets[left].isdisjoint(split_sets[right]):
            raise ValueError(f"seed {seed}: {left}/{right} series splits overlap")

    all_ids = set.union(*split_sets.values())
    _assert_equal(f"seed {seed}: split series coverage", len(all_ids), dataset["series"])
    expected_counts = dataset["split_series_counts"]
    actual_counts = {name: len(values) for name, values in split_sets.items()}
    _assert_equal(f"seed {seed}: split series counts", actual_counts, expected_counts)


def _validate_accuracy_result(result: Mapping[str, Any]) -> None:
    _assert_equal("accuracy schema_version", result.get("schema_version"), 1)
    protocol = result["protocol"]
    seed = int(protocol["seed"])
    _assert_equal(f"seed {seed}: initialization_seed", protocol["initialization_seed"], seed)
    _assert_equal(f"seed {seed}: training_randomness_seed", protocol["training_randomness_seed"], seed + 1)
    _assert_equal(f"seed {seed}: dataloader_seed", protocol["dataloader_seed"], seed + 2)
    _assert_equal(f"seed {seed}: endogenous_only", protocol["endogenous_only"], True)
    _assert_equal(f"seed {seed}: precision", protocol["precision"], "float32")
    _assert_equal(f"seed {seed}: loss", protocol["loss"], "mse")
    _validate_split(result["dataset"], seed)

    models = result["models"]
    _assert_equal(f"seed {seed}: model names", set(models), set(MODELS))
    for model_name in MODELS:
        model = models[model_name]
        training = model["training"]
        if int(training["parameters"]) <= 0:
            raise ValueError(f"seed {seed}: {model_name} parameter count must be positive")
        for scope in SCOPES:
            evaluation = model[scope]
            if int(evaluation["windows"]) <= 0 or int(evaluation["forecast_points"]) <= 0:
                raise ValueError(f"seed {seed}: {model_name}/{scope} has no evaluation points")
            micro = evaluation["metrics"]["micro"]
            for metric in ERROR_METRICS:
                value = float(micro[metric])
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"seed {seed}: invalid {model_name}/{scope}/{metric}: {value}"
                    )

    for scope in SCOPES:
        original_eval = models["original"][scope]
        enhanced_eval = models["enhanced"][scope]
        _assert_equal(
            f"seed {seed}: {scope} windows",
            original_eval["windows"],
            enhanced_eval["windows"],
        )
        _assert_equal(
            f"seed {seed}: {scope} forecast points",
            original_eval["forecast_points"],
            enhanced_eval["forecast_points"],
        )

        paired = result["paired_comparison"][scope]
        original_micro = original_eval["metrics"]["micro"]
        enhanced_micro = enhanced_eval["metrics"]["micro"]
        for metric in ERROR_METRICS:
            expected = _relative_improvement(
                float(original_micro[metric]),
                float(enhanced_micro[metric]),
            )
            actual = float(paired["original_relative_improvement_pct"][metric])
            if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(
                    f"seed {seed}: {scope}/{metric} paired improvement mismatch: "
                    f"expected {expected}, got {actual}"
                )

        original_mae = float(original_micro["mae"])
        enhanced_mae = float(enhanced_micro["mae"])
        expected_winner = (
            "tie"
            if original_mae == enhanced_mae
            else "original"
            if original_mae < enhanced_mae
            else "enhanced"
        )
        _assert_equal(
            f"seed {seed}: {scope} overall MAE winner",
            paired["overall_mae_winner"],
            expected_winner,
        )
        win_rates = paired["pointwise_absolute_error_win_rate"]
        if not math.isclose(sum(float(value) for value in win_rates.values()), 1.0, abs_tol=1e-12):
            raise ValueError(f"seed {seed}: {scope} pointwise win rates do not sum to one")


def _validate_accuracy_results(
    results: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    if len(results) != len(EXPECTED_SEEDS):
        raise ValueError(
            f"Expected {len(EXPECTED_SEEDS)} accuracy results, got {len(results)}."
        )
    for result in results:
        _validate_accuracy_result(result)

    ordered = sorted(results, key=lambda item: int(item["protocol"]["seed"]))
    seeds = tuple(int(result["protocol"]["seed"]) for result in ordered)
    _assert_equal("accuracy seeds", seeds, EXPECTED_SEEDS)

    reference = ordered[0]
    reference_protocol = _without(reference["protocol"], SEED_PROTOCOL_FIELDS)
    reference_dataset = _without(reference["dataset"], {"splits"})
    fingerprints: list[str] = []
    parameter_counts = {
        model_name: int(reference["models"][model_name]["training"]["parameters"])
        for model_name in MODELS
    }

    for result in ordered:
        seed = int(result["protocol"]["seed"])
        _assert_equal(f"seed {seed}: source", result["source"], reference["source"])
        _assert_equal(f"seed {seed}: environment", result["environment"], reference["environment"])
        _assert_equal(
            f"seed {seed}: fixed protocol",
            _without(result["protocol"], SEED_PROTOCOL_FIELDS),
            reference_protocol,
        )
        _assert_equal(
            f"seed {seed}: fixed dataset contract",
            _without(result["dataset"], {"splits"}),
            reference_dataset,
        )
        for model_name in MODELS:
            _assert_equal(
                f"seed {seed}: {model_name} parameters",
                int(result["models"][model_name]["training"]["parameters"]),
                parameter_counts[model_name],
            )
        fingerprints.append(str(result["protocol"]["split_fingerprint"]))

    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("Each seed must produce a distinct split fingerprint.")

    invariants = {
        "seeds": list(seeds),
        "source": reference["source"],
        "environment": reference["environment"],
        "fixed_protocol": reference_protocol,
        "fixed_dataset": reference_dataset,
        "parameter_counts": parameter_counts,
        "split_fingerprints": {
            str(seed): fingerprint for seed, fingerprint in zip(seeds, fingerprints)
        },
    }
    return ordered, invariants


def _aggregate_scope(
    results: Sequence[Mapping[str, Any]],
    scope: str,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model_name in MODELS:
        models[model_name] = {
            "micro_metrics": {
                metric: _summarize_seed_values(
                    [
                        (
                            int(result["protocol"]["seed"]),
                            float(result["models"][model_name][scope]["metrics"]["micro"][metric]),
                        )
                        for result in results
                    ]
                )
                for metric in ERROR_METRICS
            }
        }

    relative_improvement = {
        metric: _summarize_seed_values(
            [
                (
                    int(result["protocol"]["seed"]),
                    float(
                        result["paired_comparison"][scope][
                            "original_relative_improvement_pct"
                        ][metric]
                    ),
                )
                for result in results
            ]
        )
        for metric in ERROR_METRICS
    }

    overall_wins = {"original": 0, "enhanced": 0, "tie": 0}
    series_wins = {"original": 0, "enhanced": 0, "tie": 0}
    for result in results:
        paired = result["paired_comparison"][scope]
        overall_wins[paired["overall_mae_winner"]] += 1
        for winner in paired["series_mae_winner"].values():
            series_wins[winner] += 1

    pointwise_win_rates = {
        model_name: _summarize_seed_values(
            [
                (
                    int(result["protocol"]["seed"]),
                    float(
                        result["paired_comparison"][scope][
                            "pointwise_absolute_error_win_rate"
                        ][model_name]
                    ),
                )
                for result in results
            ]
        )
        for model_name in (*MODELS, "tie")
    }

    return {
        "models": models,
        "original_relative_improvement_pct": relative_improvement,
        "overall_mae_seed_wins": overall_wins,
        "pointwise_absolute_error_win_rate": pointwise_win_rates,
        "series_mae_wins": series_wins,
    }


def _aggregate_training(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "best_epoch",
        "best_validation_mse",
        "best_validation_rmse",
        "elapsed_seconds",
        "epochs_completed",
        "peak_allocated_mib",
    )
    output: dict[str, Any] = {}
    for model_name in MODELS:
        output[model_name] = {
            "parameters": int(results[0]["models"][model_name]["training"]["parameters"]),
            "statistics": {
                field: _summarize_seed_values(
                    [
                        (
                            int(result["protocol"]["seed"]),
                            float(result["models"][model_name]["training"][field]),
                        )
                        for result in results
                    ]
                )
                for field in fields
            },
        }

    output["original_validation_mse_relative_improvement_pct"] = _summarize_seed_values(
        [
            (
                int(result["protocol"]["seed"]),
                _relative_improvement(
                    float(result["models"]["original"]["training"]["best_validation_mse"]),
                    float(result["models"]["enhanced"]["training"]["best_validation_mse"]),
                ),
            )
            for result in results
        ]
    )
    return output


def _validate_performance_result(
    result: Mapping[str, Any],
    invariants: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    _assert_equal("performance schema_version", result.get("schema_version"), 1)
    for field in ("original_upstream_commit", "enhanced_baseline_commit"):
        _assert_equal(
            f"performance source {field}",
            result["source"][field],
            invariants["source"][field],
        )
    _assert_equal(
        "performance reference_config",
        result["protocol"]["reference_config"],
        invariants["fixed_protocol"]["reference_config"],
    )
    _assert_equal("performance device", result["environment"]["device"], invariants["environment"]["device"])
    _assert_equal("performance precision", result["protocol"]["precision"], "bf16")
    _assert_equal("performance measured_steps", result["protocol"]["measured_steps"], 100)
    _assert_equal("performance batch_size", result["protocol"]["batch_size"], 64)

    models = {str(item["model"]): item for item in result["models"]}
    _assert_equal("performance model names", set(models), set(MODELS))
    for model_name in MODELS:
        _assert_equal(
            f"performance {model_name} parameters",
            int(models[model_name]["parameters"]),
            invariants["parameter_counts"][model_name],
        )
    return models


def _performance_summary(
    result: Mapping[str, Any],
    models: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    original = models["original"]
    enhanced = models["enhanced"]
    original_latency = float(original["timing_ms"]["mean"])
    enhanced_latency = float(enhanced["timing_ms"]["mean"])
    original_throughput = float(original["throughput"]["samples_per_second"])
    enhanced_throughput = float(enhanced["throughput"]["samples_per_second"])
    original_memory = float(original["memory_mib"]["peak_allocated"])
    enhanced_memory = float(enhanced["memory_mib"]["peak_allocated"])
    original_parameters = int(original["parameters"])
    enhanced_parameters = int(enhanced["parameters"])

    return {
        "protocol": result["protocol"],
        "models": {
            model_name: {
                "parameters": int(model["parameters"]),
                "mean_step_latency_ms": float(model["timing_ms"]["mean"]),
                "samples_per_second": float(model["throughput"]["samples_per_second"]),
                "peak_allocated_mib": float(model["memory_mib"]["peak_allocated"]),
            }
            for model_name, model in models.items()
        },
        "original_vs_enhanced": {
            "latency_reduction_pct": _relative_improvement(original_latency, enhanced_latency),
            "throughput_multiplier": original_throughput / enhanced_throughput,
            "peak_allocated_reduction_pct": _relative_improvement(original_memory, enhanced_memory),
            "parameter_reduction_pct": _relative_improvement(
                float(original_parameters), float(enhanced_parameters)
            ),
        },
    }


def _decision(aggregate: Mapping[str, Any], performance: Mapping[str, Any]) -> dict[str, Any]:
    rolling = aggregate["test_all_rolling_windows"]
    last_origin = aggregate["test_last_origin_per_series"]
    efficiency = performance["original_vs_enhanced"]

    checks = [
        {
            "name": "rolling_mae_seed_majority",
            "observed": rolling["overall_mae_seed_wins"]["original"],
            "operator": ">=",
            "threshold": 2,
            "passed": rolling["overall_mae_seed_wins"]["original"] >= 2,
        },
        {
            "name": "rolling_mae_mean_improvement_pct",
            "observed": rolling["original_relative_improvement_pct"]["mae"]["mean"],
            "operator": ">",
            "threshold": 0.0,
            "passed": rolling["original_relative_improvement_pct"]["mae"]["mean"] > 0.0,
        },
        {
            "name": "rolling_wape_mean_improvement_pct",
            "observed": rolling["original_relative_improvement_pct"]["wape"]["mean"],
            "operator": ">",
            "threshold": 0.0,
            "passed": rolling["original_relative_improvement_pct"]["wape"]["mean"] > 0.0,
        },
        {
            "name": "last_origin_mae_mean_improvement_pct",
            "observed": last_origin["original_relative_improvement_pct"]["mae"]["mean"],
            "operator": ">=",
            "threshold": 0.0,
            "passed": last_origin["original_relative_improvement_pct"]["mae"]["mean"] >= 0.0,
        },
        {
            "name": "last_origin_rmse_mean_regression_guardrail_pct",
            "observed": last_origin["original_relative_improvement_pct"]["rmse"]["mean"],
            "operator": ">=",
            "threshold": -5.0,
            "passed": last_origin["original_relative_improvement_pct"]["rmse"]["mean"] >= -5.0,
        },
        {
            "name": "last_origin_smape_mean_regression_guardrail_pct",
            "observed": last_origin["original_relative_improvement_pct"]["smape"]["mean"],
            "operator": ">=",
            "threshold": -5.0,
            "passed": last_origin["original_relative_improvement_pct"]["smape"]["mean"] >= -5.0,
        },
        {
            "name": "throughput_multiplier",
            "observed": efficiency["throughput_multiplier"],
            "operator": ">=",
            "threshold": 1.0,
            "passed": efficiency["throughput_multiplier"] >= 1.0,
        },
        {
            "name": "peak_allocated_reduction_pct",
            "observed": efficiency["peak_allocated_reduction_pct"],
            "operator": ">",
            "threshold": 0.0,
            "passed": efficiency["peak_allocated_reduction_pct"] > 0.0,
        },
    ]
    promoted = all(check["passed"] for check in checks)
    point_default = "patchmixer_original" if promoted else "patchmixer_base"

    return {
        "status": (
            "promote_original_for_endogenous_point"
            if promoted
            else "retain_enhanced_for_endogenous_point"
        ),
        "checks": checks,
        "capability_defaults": {
            "endogenous_point": point_default,
            "exogenous_point": "patchmixer_base",
            "distribution": "patchmixer_base",
            "quantile": "patchmixer_quantile",
        },
        "compatibility_contract": {
            "patchmixer_family_expansion": ["patchmixer_base", "patchmixer_quantile"],
            "existing_checkpoint_aliases_changed": False,
            "automatic_family_alias_change": False,
        },
        "caveats": [
            "Accuracy evidence covers one Walmart weekly dataset and three disjoint-series splits.",
            "Enhanced wins seed 22 on both aggregate MAE evaluations.",
            "Original has lower mean last-origin MAE but regresses mean last-origin sMAPE.",
            "Original is endogenous-only and point-only; exogenous, distribution, and quantile requests stay Enhanced.",
        ],
    }


def build_summary(
    accuracy_results: Sequence[Mapping[str, Any]],
    performance_result: Mapping[str, Any],
) -> dict[str, Any]:
    ordered, invariants = _validate_accuracy_results(accuracy_results)
    performance_models = _validate_performance_result(performance_result, invariants)
    aggregate = {
        scope: _aggregate_scope(ordered, scope)
        for scope in SCOPES
    }
    aggregate["training"] = _aggregate_training(ordered)
    performance = _performance_summary(performance_result, performance_models)
    decision = _decision(aggregate, performance)
    return {
        "schema_version": 1,
        "validation": {
            "status": "passed",
            "invariants": invariants,
        },
        "aggregate": aggregate,
        "performance": performance,
        "decision": decision,
    }


def main() -> None:
    args = _build_parser().parse_args()
    accuracy_records = [
        (path, _read_json(path))
        for path in args.accuracy_results
    ]
    performance_result = _read_json(args.performance_result)
    summary = build_summary(
        [result for _, result in accuracy_records],
        performance_result,
    )
    summary["inputs"] = {
        "accuracy_results": [
            {
                "seed": int(result["protocol"]["seed"]),
                "path": str(path),
                "sha256": _sha256(path),
            }
            for path, result in sorted(
                accuracy_records,
                key=lambda item: int(item[1]["protocol"]["seed"]),
            )
        ],
        "performance_result": {
            "path": str(args.performance_result),
            "sha256": _sha256(args.performance_result),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    rolling = summary["aggregate"]["test_all_rolling_windows"]
    last_origin = summary["aggregate"]["test_last_origin_per_series"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": summary["decision"]["status"],
                "rolling_mae_improvement_mean_pct": rolling[
                    "original_relative_improvement_pct"
                ]["mae"]["mean"],
                "last_origin_mae_improvement_mean_pct": last_origin[
                    "original_relative_improvement_pct"
                ]["mae"]["mean"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

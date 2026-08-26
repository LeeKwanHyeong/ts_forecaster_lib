#!/usr/bin/env python3
"""Compare production checkpoints across two isolated private-wheel installs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROBE_CODE = r"""
import hashlib
import json
import math
import sys
from pathlib import Path

site_path = Path(sys.argv[1]).resolve()
checkpoint_path = Path(sys.argv[2]).resolve()
model_key = sys.argv[3]
device = sys.argv[4]
lookback = int(sys.argv[5])
horizon = int(sys.argv[6])
past_dim = int(sys.argv[7])
future_dim = int(sys.argv[8])
sys.path.insert(0, str(site_path))

import numpy as np
import torch
import modeling_module
from modeling_module import load_predictor


def canonical_sha256(value):
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def tensor_payload(value):
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise RuntimeError(f"unsupported prediction dtype: {array.dtype}")
    values = array.astype(np.float64, copy=False).reshape(-1).tolist()
    if any(not math.isfinite(float(item)) for item in values):
        raise RuntimeError("prediction contains a non-finite value")
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "values": values,
    }


predictor = load_predictor(str(checkpoint_path), device=device, strict=True)
if predictor.model_key != model_key:
    raise RuntimeError(
        f"checkpoint resolved {predictor.model_key!r}, expected {model_key!r}"
    )

parameters = list(predictor.model.parameters())
state_schema = [
    {"key": key, "shape": list(value.shape), "dtype": str(value.dtype)}
    for key, value in predictor.model.state_dict().items()
]
x = (torch.arange(2 * lookback, dtype=torch.float32) % 17).reshape(
    2, lookback, 1
)
payload = {"x": x}
if past_dim:
    payload["past_exo_cont"] = torch.linspace(
        -1.0, 1.0, 2 * lookback * past_dim, dtype=torch.float32
    ).reshape(2, lookback, past_dim)
if future_dim:
    payload["future_exo_batch"] = torch.linspace(
        -0.5, 1.5, 2 * horizon * future_dim, dtype=torch.float32
    ).reshape(2, horizon, future_dim)

result = predictor.predict(payload, horizon=horizon, device=device)
if not isinstance(result, dict) or "point" not in result:
    raise RuntimeError("predictor did not return a mapping with point output")
outputs = {str(key): tensor_payload(value) for key, value in sorted(result.items())}
for name, output in outputs.items():
    if output["shape"] != [2 * horizon]:
        raise RuntimeError(
            f"{name} output shape is {output['shape']}, expected {[2 * horizon]}"
        )

print(json.dumps({
    "modeling_module_path": str(Path(modeling_module.__file__).resolve()),
    "model_key": predictor.model_key,
    "strict_load": True,
    "lookback": lookback,
    "horizon": horizon,
    "past_exogenous_dim": past_dim,
    "future_exogenous_dim": future_dim,
    "parameter_count": sum(value.numel() for value in parameters),
    "state_dict_key_count": len(state_schema),
    "state_dict_schema_sha256": canonical_sha256(state_schema),
    "outputs": outputs,
    "runtime": {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device,
        "device_name": (
            torch.cuda.get_device_name(0) if device.startswith("cuda") else None
        ),
    },
}, ensure_ascii=True, sort_keys=True))
"""


class CompatibilityError(RuntimeError):
    """Raised when a Wheel or checkpoint violates the compatibility contract."""


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompatibilityError(f"{label} must be one JSON object")
    return value


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise CompatibilityError(f"{label} must be one lowercase SHA-256")
    return value


def load_sealed_registry(path: Path) -> dict[str, Any]:
    raw = dict(
        _mapping(
            json.loads(path.read_text(encoding="utf-8")),
            label="registry",
        )
    )
    seal = _sha256(raw.pop("registry_sha256", None), label="registry_sha256")
    if canonical_json_sha256(raw) != seal:
        raise CompatibilityError("registry seal mismatch")
    models = raw.get("models")
    if not isinstance(models, list) or not models:
        raise CompatibilityError("registry model inventory is empty")
    return {**raw, "registry_sha256": seal}


def model_cases(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for raw_model in registry["models"]:
        model = _mapping(raw_model, label="model definition")
        checkpoint = _mapping(model.get("checkpoint"), label="checkpoint")
        architecture = _mapping(
            _mapping(model.get("model"), label="model metadata").get("architecture"),
            label="model architecture",
        )
        feature_schema = model.get("feature_schema") or {}
        if not isinstance(feature_schema, Mapping):
            raise CompatibilityError("feature_schema must be one JSON object")
        cases.append(
            {
                "model_key": str(model["model_key"]),
                "plan_model_name": str(model["plan_model_name"]),
                "checkpoint_filename": str(checkpoint["path"]),
                "checkpoint_sha256": _sha256(
                    checkpoint.get("sha256"), label="checkpoint.sha256"
                ),
                "parameter_count": int(
                    _mapping(model.get("model"), label="model metadata")[
                        "parameter_count"
                    ]
                ),
                "lookback": int(architecture["lookback"]),
                "horizon": int(architecture["horizon"]),
                "past_exogenous_dim": len(
                    feature_schema.get("past_continuous_columns", [])
                ),
                "future_exogenous_dim": len(
                    feature_schema.get("future_continuous_columns", [])
                ),
            }
        )
    return cases


def install_wheel(wheel: Path, target: Path) -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--target",
            str(target),
            str(wheel),
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )


def probe_model(
    *,
    site_path: Path,
    checkpoint_path: Path,
    case: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            PROBE_CODE,
            str(site_path),
            str(checkpoint_path),
            str(case["model_key"]),
            device,
            str(case["lookback"]),
            str(case["horizon"]),
            str(case["past_exogenous_dim"]),
            str(case["future_exogenous_dim"]),
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def compare_probe_reports(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    expected_parameter_count: int,
    absolute_tolerance: float,
) -> dict[str, Any]:
    required_equal = (
        "model_key",
        "lookback",
        "horizon",
        "past_exogenous_dim",
        "future_exogenous_dim",
        "parameter_count",
        "state_dict_key_count",
        "state_dict_schema_sha256",
    )
    for key in required_equal:
        if baseline.get(key) != candidate.get(key):
            raise CompatibilityError(f"probe contract differs for {key}")
    observed_parameter_count = int(candidate["parameter_count"])

    baseline_outputs = _mapping(baseline.get("outputs"), label="baseline outputs")
    candidate_outputs = _mapping(candidate.get("outputs"), label="candidate outputs")
    if set(baseline_outputs) != set(candidate_outputs):
        raise CompatibilityError("prediction output keys differ")
    output_receipts: dict[str, Any] = {}
    global_max = 0.0
    exact = True
    for key in sorted(baseline_outputs):
        left = _mapping(baseline_outputs[key], label=f"baseline {key}")
        right = _mapping(candidate_outputs[key], label=f"candidate {key}")
        if left.get("shape") != right.get("shape"):
            raise CompatibilityError(f"{key} output shape differs")
        left_values = list(left.get("values", []))
        right_values = list(right.get("values", []))
        if len(left_values) != len(right_values):
            raise CompatibilityError(f"{key} output length differs")
        differences = [
            abs(float(left_value) - float(right_value))
            for left_value, right_value in zip(left_values, right_values, strict=True)
        ]
        max_difference = max(differences, default=0.0)
        global_max = max(global_max, max_difference)
        output_exact = all(difference == 0.0 for difference in differences)
        exact = exact and output_exact
        if not math.isfinite(max_difference) or max_difference > absolute_tolerance:
            raise CompatibilityError(
                f"{key} output parity failed: {max_difference} > {absolute_tolerance}"
            )
        output_receipts[key] = {
            "shape": right["shape"],
            "finite_value_count": len(right_values),
            "exact_match": output_exact,
            "maximum_absolute_difference": max_difference,
        }
    return {
        "strict_load": bool(baseline["strict_load"] and candidate["strict_load"]),
        "parameter_count": observed_parameter_count,
        "registry_parameter_count": expected_parameter_count,
        "registry_parameter_count_matches": (
            observed_parameter_count == expected_parameter_count
        ),
        "state_dict_key_count": int(candidate["state_dict_key_count"]),
        "state_dict_schema_sha256": candidate["state_dict_schema_sha256"],
        "outputs": output_receipts,
        "exact_output_match": exact,
        "maximum_absolute_difference": global_max,
    }


def _wheel_identity(path: Path, expected_sha256: str) -> dict[str, Any]:
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise CompatibilityError(
            f"Wheel SHA-256 mismatch for {path.name}: {observed}"
        )
    return {"path": str(path), "filename": path.name, "sha256": observed}


def verify_receipt(path: Path) -> dict[str, Any]:
    receipt = dict(
        _mapping(json.loads(path.read_text(encoding="ascii")), label="receipt")
    )
    seal = _sha256(receipt.pop("receipt_sha256", None), label="receipt_sha256")
    if canonical_json_sha256(receipt) != seal:
        raise CompatibilityError("receipt seal mismatch")
    if receipt.get("status") != "PASS":
        raise CompatibilityError("compatibility receipt is not PASS")
    return {**receipt, "receipt_sha256": seal}


def _verify_icl_wheel_receipt(path: Path) -> dict[str, Any]:
    receipt = dict(
        _mapping(json.loads(path.read_text(encoding="ascii")), label="ICL receipt")
    )
    seal = _sha256(
        receipt.pop("receipt_sha256", None), label="ICL receipt_sha256"
    )
    if canonical_json_sha256(receipt) != seal:
        raise CompatibilityError("ICL Wheel receipt seal mismatch")
    verification = _mapping(
        receipt.get("verification"), label="ICL Wheel verification"
    )
    for model_key in ("sellm", "autotimes"):
        model = _mapping(verification.get(model_key), label=f"{model_key} verification")
        if model.get("strict_load") is not True:
            raise CompatibilityError(f"{model_key} strict load did not pass")
        if int(model.get("nonfinite_count", -1)) != 0:
            raise CompatibilityError(f"{model_key} produced non-finite output")
        if float(model.get("source_corrected_max_abs_diff", math.inf)) != 0.0:
            raise CompatibilityError(f"{model_key} corrected parity did not pass")
    if receipt.get("status") != "PASS":
        raise CompatibilityError("ICL Wheel receipt is not PASS")
    return {**receipt, "receipt_sha256": seal}


def aggregate_ten_model_receipt(
    *,
    legacy_receipt_path: Path,
    icl_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    legacy = verify_receipt(legacy_receipt_path)
    icl = _verify_icl_wheel_receipt(icl_receipt_path)
    legacy_wheel = _mapping(
        _mapping(legacy.get("wheels"), label="legacy Wheels").get("candidate"),
        label="legacy candidate Wheel",
    )
    icl_wheel = _mapping(icl.get("wheel"), label="ICL Wheel")
    if legacy_wheel.get("sha256") != icl_wheel.get("sha256"):
        raise CompatibilityError("legacy and ICL receipts reference different Wheels")

    legacy_models = [
        str(_mapping(model, label="legacy model")["model_key"])
        for model in legacy["models"]
    ]
    model_keys = [*legacy_models, "sellm_base", "autotimes_base"]
    if len(model_keys) != 10 or len(set(model_keys)) != 10:
        raise CompatibilityError("integrated Wheel receipt must contain 10 models")
    summary = _mapping(legacy.get("summary"), label="legacy summary")
    if any(
        int(summary.get(key, -1)) != 8
        for key in (
            "strict_load_passed",
            "state_dict_schema_passed",
            "h26_finite_output_passed",
            "exact_output_parity_passed",
        )
    ):
        raise CompatibilityError("legacy eight-model receipt is incomplete")

    receipt: dict[str, Any] = {
        "contract": "modeling_module.ten_model_wheel_compatibility.v1",
        "status": "PASS",
        "wheel": {
            "filename": icl_wheel["path"].rsplit("/", 1)[-1],
            "sha256": icl_wheel["sha256"],
            "distribution_profile": icl_wheel["distribution_profile"],
            "source_commit": icl["source_commit"],
        },
        "model_inventory": {
            "count": 10,
            "model_keys": model_keys,
        },
        "verification": {
            "legacy_eight": {
                "strict_load_passed": 8,
                "state_dict_schema_passed": 8,
                "finite_h26_output_passed": 8,
                "exact_output_parity_passed": 8,
                "maximum_absolute_difference": summary[
                    "maximum_absolute_difference"
                ],
            },
            "icl_two": {
                "strict_load_passed": 2,
                "finite_h26_output_passed": 2,
                "corrected_checkpoint_parity_passed": 2,
                "maximum_absolute_difference": 0.0,
            },
            "registry_metadata_warnings": summary.get(
                "registry_parameter_count_warnings", []
            ),
        },
        "source_receipts": {
            "legacy_eight": {
                "filename": legacy_receipt_path.name,
                "file_sha256": file_sha256(legacy_receipt_path),
                "receipt_sha256": legacy["receipt_sha256"],
            },
            "icl_two": {
                "filename": icl_receipt_path.name,
                "file_sha256": file_sha256(icl_receipt_path),
                "receipt_sha256": icl["receipt_sha256"],
            },
        },
        "runtime": legacy["runtime"],
        "safety": {
            "db_write_enabled": False,
            "runtime_activation_changed": False,
            "active_port_8011_modified": False,
        },
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return verify_receipt(output_path)


def run_compatibility(
    *,
    baseline_wheel: Path,
    baseline_wheel_sha256: str,
    candidate_wheel: Path,
    candidate_wheel_sha256: str,
    registry_path: Path,
    checkpoint_root: Path,
    receipt_path: Path,
    device: str,
    absolute_tolerance: float,
) -> dict[str, Any]:
    registry = load_sealed_registry(registry_path)
    cases = model_cases(registry)
    baseline_identity = _wheel_identity(baseline_wheel, baseline_wheel_sha256)
    candidate_identity = _wheel_identity(candidate_wheel, candidate_wheel_sha256)
    model_receipts: list[dict[str, Any]] = []
    runtime: dict[str, Any] | None = None

    with tempfile.TemporaryDirectory(prefix="wheel-compatibility-") as temp_dir:
        temp_root = Path(temp_dir)
        baseline_site = temp_root / "baseline-site"
        candidate_site = temp_root / "candidate-site"
        install_wheel(baseline_wheel, baseline_site)
        install_wheel(candidate_wheel, candidate_site)
        for case in cases:
            checkpoint_path = checkpoint_root / case["checkpoint_filename"]
            if not checkpoint_path.is_file():
                raise CompatibilityError(f"checkpoint not found: {checkpoint_path}")
            observed_sha256 = file_sha256(checkpoint_path)
            if observed_sha256 != case["checkpoint_sha256"]:
                raise CompatibilityError(
                    f"checkpoint SHA-256 mismatch: {checkpoint_path.name}"
                )
            baseline_probe = probe_model(
                site_path=baseline_site,
                checkpoint_path=checkpoint_path,
                case=case,
                device=device,
            )
            candidate_probe = probe_model(
                site_path=candidate_site,
                checkpoint_path=checkpoint_path,
                case=case,
                device=device,
            )
            runtime = candidate_probe["runtime"]
            parity = compare_probe_reports(
                baseline_probe,
                candidate_probe,
                expected_parameter_count=int(case["parameter_count"]),
                absolute_tolerance=absolute_tolerance,
            )
            model_receipts.append(
                {
                    "model_key": case["model_key"],
                    "plan_model_name": case["plan_model_name"],
                    "checkpoint": {
                        "path": case["checkpoint_filename"],
                        "sha256": observed_sha256,
                    },
                    "input_contract": {
                        "batch_size": 2,
                        "lookback": case["lookback"],
                        "horizon": case["horizon"],
                        "past_exogenous_dim": case["past_exogenous_dim"],
                        "future_exogenous_dim": case["future_exogenous_dim"],
                    },
                    "parity": parity,
                }
            )

    receipt: dict[str, Any] = {
        "receipt_format_version": 1,
        "status": "PASS",
        "scope": {
            "model_count": len(cases),
            "model_keys": [case["model_key"] for case in cases],
            "comparison": "baseline_non_sellm_to_candidate_sellm",
            "absolute_tolerance": absolute_tolerance,
            "db_write_enabled": False,
            "runtime_activation_changed": False,
        },
        "registry": {
            "path": registry_path.name,
            "file_sha256": file_sha256(registry_path),
            "registry_sha256": registry["registry_sha256"],
        },
        "wheels": {
            "baseline": baseline_identity,
            "candidate": candidate_identity,
        },
        "runtime": runtime,
        "models": model_receipts,
        "summary": {
            "strict_load_passed": len(model_receipts),
            "state_dict_schema_passed": len(model_receipts),
            "h26_finite_output_passed": len(model_receipts),
            "registry_parameter_count_matched": sum(
                bool(model["parity"]["registry_parameter_count_matches"])
                for model in model_receipts
            ),
            "registry_parameter_count_warnings": [
                model["model_key"]
                for model in model_receipts
                if not model["parity"]["registry_parameter_count_matches"]
            ],
            "exact_output_parity_passed": sum(
                bool(model["parity"]["exact_output_match"])
                for model in model_receipts
            ),
            "maximum_absolute_difference": max(
                float(model["parity"]["maximum_absolute_difference"])
                for model in model_receipts
            ),
        },
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return verify_receipt(receipt_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--baseline-wheel", type=Path, required=True)
    run.add_argument("--baseline-wheel-sha256", required=True)
    run.add_argument("--candidate-wheel", type=Path, required=True)
    run.add_argument("--candidate-wheel-sha256", required=True)
    run.add_argument("--registry", type=Path, required=True)
    run.add_argument("--checkpoint-root", type=Path, required=True)
    run.add_argument("--receipt", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument("--absolute-tolerance", type=float, default=0.0)
    verify = subparsers.add_parser("verify-receipt")
    verify.add_argument("--receipt", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate-ten-model")
    aggregate.add_argument("--legacy-receipt", type=Path, required=True)
    aggregate.add_argument("--icl-receipt", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify-receipt":
        result = verify_receipt(args.receipt.expanduser().resolve())
    elif args.command == "aggregate-ten-model":
        result = aggregate_ten_model_receipt(
            legacy_receipt_path=args.legacy_receipt.expanduser().resolve(),
            icl_receipt_path=args.icl_receipt.expanduser().resolve(),
            output_path=args.output.expanduser().resolve(),
        )
    else:
        result = run_compatibility(
            baseline_wheel=args.baseline_wheel.expanduser().resolve(),
            baseline_wheel_sha256=_sha256(
                args.baseline_wheel_sha256, label="baseline Wheel SHA-256"
            ),
            candidate_wheel=args.candidate_wheel.expanduser().resolve(),
            candidate_wheel_sha256=_sha256(
                args.candidate_wheel_sha256, label="candidate Wheel SHA-256"
            ),
            registry_path=args.registry.expanduser().resolve(),
            checkpoint_root=args.checkpoint_root.expanduser().resolve(),
            receipt_path=args.receipt.expanduser().resolve(),
            device=str(args.device),
            absolute_tolerance=float(args.absolute_tolerance),
        )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

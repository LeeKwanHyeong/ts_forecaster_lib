from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "verify_private_wheel_model_compatibility.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("_wheel_compatibility", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def _probe(*, values: list[float], schema: str = "a" * 64) -> dict:
    return {
        "model_key": "patchtst_base",
        "strict_load": True,
        "lookback": 52,
        "horizon": 26,
        "past_exogenous_dim": 0,
        "future_exogenous_dim": 0,
        "parameter_count": 3,
        "state_dict_key_count": 2,
        "state_dict_schema_sha256": schema,
        "outputs": {
            "point": {
                "dtype": "float32",
                "shape": [len(values)],
                "values": values,
            }
        },
    }


def test_compare_probe_reports_accepts_exact_output_and_state_schema():
    baseline = _probe(values=[1.0, 2.0])
    candidate = _probe(values=[1.0, 2.0])

    result = tool.compare_probe_reports(
        baseline,
        candidate,
        expected_parameter_count=3,
        absolute_tolerance=0.0,
    )

    assert result["strict_load"] is True
    assert result["exact_output_match"] is True
    assert result["maximum_absolute_difference"] == 0.0
    assert result["outputs"]["point"]["finite_value_count"] == 2
    assert result["registry_parameter_count_matches"] is True


def test_compare_probe_reports_records_preexisting_registry_count_drift():
    result = tool.compare_probe_reports(
        _probe(values=[1.0, 2.0]),
        _probe(values=[1.0, 2.0]),
        expected_parameter_count=99,
        absolute_tolerance=0.0,
    )

    assert result["parameter_count"] == 3
    assert result["registry_parameter_count"] == 99
    assert result["registry_parameter_count_matches"] is False


def test_compare_probe_reports_rejects_output_drift():
    with pytest.raises(tool.CompatibilityError, match="output parity failed"):
        tool.compare_probe_reports(
            _probe(values=[1.0, 2.0]),
            _probe(values=[1.0, 2.01]),
            expected_parameter_count=3,
            absolute_tolerance=0.0,
        )


def test_compare_probe_reports_rejects_state_dict_schema_drift():
    with pytest.raises(tool.CompatibilityError, match="state_dict_schema_sha256"):
        tool.compare_probe_reports(
            _probe(values=[1.0], schema="a" * 64),
            _probe(values=[1.0], schema="b" * 64),
            expected_parameter_count=3,
            absolute_tolerance=0.0,
        )


def test_verify_receipt_detects_tampering(tmp_path):
    payload = {"receipt_format_version": 1, "status": "PASS", "models": []}
    payload["receipt_sha256"] = tool.canonical_json_sha256(payload)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(payload), encoding="ascii")

    assert tool.verify_receipt(path)["status"] == "PASS"

    payload["status"] = "FAIL"
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(tool.CompatibilityError, match="receipt seal mismatch"):
        tool.verify_receipt(path)


def _write_sealed(path: Path, payload: dict) -> None:
    payload["receipt_sha256"] = tool.canonical_json_sha256(payload)
    path.write_text(json.dumps(payload), encoding="ascii")


def test_aggregate_ten_model_receipt_binds_both_receipts(tmp_path):
    wheel_sha = "c" * 64
    legacy_path = tmp_path / "legacy.json"
    legacy_models = [
        {"model_key": f"legacy_{index}"} for index in range(8)
    ]
    _write_sealed(
        legacy_path,
        {
            "status": "PASS",
            "models": legacy_models,
            "wheels": {"candidate": {"sha256": wheel_sha}},
            "runtime": {"device": "cuda"},
            "summary": {
                "strict_load_passed": 8,
                "state_dict_schema_passed": 8,
                "h26_finite_output_passed": 8,
                "exact_output_parity_passed": 8,
                "maximum_absolute_difference": 0.0,
                "registry_parameter_count_warnings": [],
            },
        },
    )
    icl_path = tmp_path / "icl.json"
    _write_sealed(
        icl_path,
        {
            "status": "PASS",
            "source_commit": "d" * 40,
            "wheel": {
                "path": "/tmp/modeling_module.whl",
                "sha256": wheel_sha,
                "distribution_profile": "sellm",
            },
            "verification": {
                key: {
                    "strict_load": True,
                    "nonfinite_count": 0,
                    "source_corrected_max_abs_diff": 0.0,
                }
                for key in ("sellm", "autotimes")
            },
        },
    )
    output = tmp_path / "ten.json"

    result = tool.aggregate_ten_model_receipt(
        legacy_receipt_path=legacy_path,
        icl_receipt_path=icl_path,
        output_path=output,
    )

    assert result["status"] == "PASS"
    assert result["model_inventory"]["count"] == 10
    assert result["wheel"]["sha256"] == wheel_sha
    assert result["source_receipts"]["legacy_eight"]["file_sha256"] == (
        tool.file_sha256(legacy_path)
    )

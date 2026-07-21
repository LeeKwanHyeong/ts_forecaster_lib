"""Contract tests for the proposed result-returning public forecast API."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from dataclasses import fields
from pathlib import Path

import polars as pl
import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs" / "contracts" / "public_forecast_contract.v1.json"
ADR_PATH = ROOT / "docs" / "adr" / "0001-public-anchored-forecast-api.md"
CONTRACT_ID = "modeling-module.public-anchored-forecast"
CONTRACT_VERSION = "1.0.0"
CONTRACT_SHA256 = "07e8d2d825929bd9882d413c32faf76108b3f5e0d147d6a628575e0ebda563bd"

FORECAST_COLUMNS = [
    "series_id",
    "model_key",
    "forecast_origin",
    "horizon_step",
    "point",
    "q10",
    "q50",
    "q90",
]

FORECAST_SCHEMA = {
    "series_id": pl.String,
    "model_key": pl.String,
    "forecast_origin": pl.Int64,
    "horizon_step": pl.Int32,
    "point": pl.Float64,
    "q10": pl.Float64,
    "q50": pl.Float64,
    "q90": pl.Float64,
}


def _load_contract() -> dict[str, object]:
    """Load the sealed machine-readable forecast contract."""
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_public_forecast_contract_identity_and_seal() -> None:
    contract = _load_contract()
    sealed_hash = contract.pop("contract_sha256")
    canonical = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")

    assert contract["contract_id"] == CONTRACT_ID
    assert contract["contract_version"] == CONTRACT_VERSION
    assert sealed_hash == CONTRACT_SHA256
    assert hashlib.sha256(canonical).hexdigest() == CONTRACT_SHA256


def test_accepted_adr_references_the_sealed_contract_identity() -> None:
    adr = ADR_PATH.read_text(encoding="utf-8")

    assert "Status: Accepted" in adr
    assert "Accepted: 2026-07-21" in adr
    assert CONTRACT_ID in adr
    assert CONTRACT_VERSION in adr
    assert CONTRACT_SHA256 in adr
    assert CONTRACT_PATH.name in adr


def test_machine_readable_contract_freezes_ordered_schema_and_nullability() -> None:
    contract = _load_contract()
    ordered_schema = contract["forecast_result"]["ordered_schema"]

    assert ordered_schema == [
        {"name": "series_id", "dtype": "String", "nullable": False},
        {"name": "model_key", "dtype": "String", "nullable": False},
        {"name": "forecast_origin", "dtype": "Int64", "nullable": False},
        {"name": "horizon_step", "dtype": "Int32", "nullable": False},
        {"name": "point", "dtype": "Float64", "nullable": False},
        {"name": "q10", "dtype": "Float64", "nullable": True},
        {"name": "q50", "dtype": "Float64", "nullable": True},
        {"name": "q90", "dtype": "Float64", "nullable": True},
    ]


@pytest.mark.parametrize("module_name", ["modeling_module", "modeling_module.api"])
def test_forecast_types_and_function_are_exported_from_stable_surfaces(module_name: str) -> None:
    module = importlib.import_module(module_name)

    for name in ("ForecastRequest", "ForecastRuntimeConfig", "ForecastResult", "forecast"):
        assert hasattr(module, name), f"{module_name} does not export {name}"


def test_forecast_request_and_result_field_order_matches_sealed_signature() -> None:
    from modeling_module import ForecastRequest, ForecastResult, ForecastRuntimeConfig, forecast

    assert [field.name for field in fields(ForecastRuntimeConfig)] == [
        "batch_size",
        "num_workers",
        "device",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
    ]
    assert [field.name for field in fields(ForecastRequest)] == [
        "checkpoint_path",
        "expected_model_key",
        "data",
        "series_ids",
        "forecast_origin",
        "runtime",
        "unknown_series_policy",
    ]
    assert [field.name for field in fields(ForecastResult)] == [
        "predictions",
        "model_key",
        "forecast_origin",
    ]

    runtime_signature = inspect.signature(ForecastRuntimeConfig)
    assert [parameter.name for parameter in runtime_signature.parameters.values()] == [
        "batch_size",
        "num_workers",
        "device",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
    ]
    assert [parameter.default for parameter in runtime_signature.parameters.values()] == [
        64,
        0,
        None,
        True,
        True,
        2,
    ]

    request_signature = inspect.signature(ForecastRequest)
    assert [parameter.name for parameter in request_signature.parameters.values()] == [
        "checkpoint_path",
        "expected_model_key",
        "data",
        "series_ids",
        "forecast_origin",
        "runtime",
        "unknown_series_policy",
    ]
    assert request_signature.parameters["unknown_series_policy"].default == "error"
    assert str(inspect.signature(forecast)) == "(request: 'ForecastRequest') -> 'ForecastResult'"


def test_forecast_result_schema_contract_is_exact_and_ordered() -> None:
    frame = pl.DataFrame(schema=FORECAST_SCHEMA)

    assert frame.columns == FORECAST_COLUMNS
    assert frame.schema == pl.Schema(FORECAST_SCHEMA)


def test_horizon_step_contract_is_zero_based_and_series_major() -> None:
    frame = pl.DataFrame(
        {
            "series_id": ["B", "A", "B", "A"],
            "horizon_step": [1, 1, 0, 0],
        }
    ).sort("series_id", "horizon_step")

    assert frame.rows() == [("A", 0), ("A", 1), ("B", 0), ("B", 1)]


def test_batch_partitioning_does_not_change_contract_row_identity_or_order() -> None:
    identities = [(series_id, step) for series_id in ("A", "B", "C") for step in range(3)]

    def collect(batch_size: int) -> list[tuple[str, int]]:
        batches = [identities[offset : offset + batch_size] for offset in range(0, len(identities), batch_size)]
        return [identity for batch in batches for identity in batch]

    assert collect(1) == collect(2) == collect(4) == identities

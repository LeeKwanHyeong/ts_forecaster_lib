"""Shared contract helpers for the DSIO V100 Weekly L52/H26 refit."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping


SITE_CD: Final = "V100"
TRAIN_END_WEEK: Final = 202509
FORECAST_ORIGIN: Final = 202510
VALIDATION_ORIGIN: Final = 202436
LOOKBACK: Final = 52
HORIZON: Final = 26
SEED: Final = 42
FORECAST_OFFSETS: Final = tuple(range(HORIZON))
EXPECTED_RESULT_COLUMNS: Final = (
    "series_id",
    "model_key",
    "forecast_origin",
    "horizon_step",
    "point",
    "q10",
    "q50",
    "q90",
)


class V100H26ContractError(RuntimeError):
    """Raised when an artifact violates the governed V100 H26 contract."""


@dataclass(frozen=True, slots=True)
class ProductionRefitModelSpec:
    """One approved endogenous production-refit model."""

    model_key: str
    plan_model_name: str
    checkpoint_model_name: str
    epochs: int

    @property
    def checkpoint_filename(self) -> str:
        return (
            f"weekly_{self.checkpoint_model_name}_"
            f"L{LOOKBACK}_H{HORIZON}.pt"
        )


MODEL_SPECS: Final = (
    ProductionRefitModelSpec("patchtst_base", "PatchTST", "PatchTST", 8),
    ProductionRefitModelSpec(
        "patchtst_quantile",
        "PatchTST_Quantile",
        "PatchTSTQuantile",
        3,
    ),
    ProductionRefitModelSpec("patchmixer", "PatchMixer", "PatchMixer", 3),
    ProductionRefitModelSpec("nhits_base", "NHITS", "NHITSBase", 31),
    ProductionRefitModelSpec("timemixer", "TimeMixer", "TimeMixer", 33),
)
MODEL_SPECS_BY_KEY: Final = {spec.model_key: spec for spec in MODEL_SPECS}


def canonical_json_sha256(value: object) -> str:
    """Return the ASCII canonical JSON SHA-256 used by DSIO receipts."""

    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash one regular file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: object, *, label: str) -> str:
    """Normalize and validate one SHA-256 digest."""

    if not isinstance(value, str):
        raise V100H26ContractError(f"{label} must be one SHA-256 digest")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise V100H26ContractError(f"{label} must be one SHA-256 digest")
    return normalized


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V100H26ContractError(f"{label} must be one JSON object")
    return value


def load_training_input_manifest(
    path: Path,
    *,
    target_source: Path | None = None,
) -> dict[str, Any]:
    """Load and validate the sealed V100-only refit input manifest."""

    raw = _require_mapping(
        json.loads(path.read_text(encoding="utf-8")),
        label="training input manifest",
    )
    payload = _require_mapping(raw.get("payload"), label="manifest payload")
    payload_sha256 = require_sha256(
        raw.get("payload_sha256"), label="payload_sha256"
    )
    if canonical_json_sha256(payload) != payload_sha256:
        raise V100H26ContractError("training input manifest seal mismatch")

    if payload.get("format") != "demand-engine-v100-h26-production-refit-target":
        raise V100H26ContractError("training input manifest format drifted")
    contract = _require_mapping(
        payload.get("training_contract"), label="training contract"
    )
    expected_contract = {
        "site_cd": SITE_CD,
        "train_end_week": TRAIN_END_WEEK,
        "first_excluded_week": FORECAST_ORIGIN,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "forecast_offsets": {
            "minimum": 0,
            "maximum": HORIZON - 1,
            "count": HORIZON,
        },
        "seed": SEED,
        "mode": "production_refit",
        "state_selection": "final_epoch",
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise V100H26ContractError(
                f"training input contract mismatch for {key!r}"
            )

    dataset = _require_mapping(payload.get("dataset"), label="dataset summary")
    if dataset.get("maximum_week") != TRAIN_END_WEEK:
        raise V100H26ContractError("training input exceeds the 202509 cutoff")
    if int(dataset.get("row_count", 0)) <= 0:
        raise V100H26ContractError("training input is empty")
    if int(dataset.get("null_target_count", -1)) != 0:
        raise V100H26ContractError("training input contains null targets")

    artifact = _require_mapping(payload.get("artifact"), label="input artifact")
    artifact_sha256 = require_sha256(
        artifact.get("sha256"), label="artifact.sha256"
    )
    require_sha256(
        payload.get("source_bundle_sha256"),
        label="source_bundle_sha256",
    )
    require_sha256(
        payload.get("source_binding_sha256"),
        label="source_binding_sha256",
    )
    if target_source is not None:
        if artifact.get("name") != target_source.name:
            raise V100H26ContractError("target filename differs from the manifest")
        if file_sha256(target_source) != artifact_sha256:
            raise V100H26ContractError("target SHA-256 differs from the manifest")

    return {
        "payload": dict(payload),
        "payload_sha256": payload_sha256,
        "file_sha256": file_sha256(path),
    }


def validate_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    *,
    spec: ProductionRefitModelSpec,
) -> dict[str, Any]:
    """Validate model metadata and the exact L52/H26 architecture window."""

    meta = _require_mapping(checkpoint.get("meta"), label="checkpoint metadata")
    config = _require_mapping(
        checkpoint.get("config") or checkpoint.get("cfg_state"),
        label="checkpoint config",
    )
    expected_meta = {
        "model_key": spec.model_key,
        "training_mode": "production_refit",
        "validation_enabled": False,
        "state_selection": "final_epoch",
        "configured_epochs": spec.epochs,
        "completed_epochs": spec.epochs,
        "random_seed": SEED,
    }
    for key, expected in expected_meta.items():
        if meta.get(key) != expected:
            raise V100H26ContractError(
                f"checkpoint metadata mismatch for {key!r}: "
                f"expected {expected!r}, got {meta.get(key)!r}"
            )
    if config.get("lookback") != LOOKBACK or config.get("horizon") != HORIZON:
        raise V100H26ContractError("checkpoint window must be L52/H26")
    return {"meta": dict(meta), "config": dict(config)}


def model_signature_payload(
    *,
    model_key: str,
    checkpoint_sha256: str,
    input_manifest_sha256: str,
) -> dict[str, object]:
    """Build the stable signature payload shared with Demand Engine."""

    return {
        "model_key": model_key,
        "checkpoint_sha256": require_sha256(
            checkpoint_sha256, label="checkpoint_sha256"
        ),
        "input_manifest_sha256": require_sha256(
            input_manifest_sha256, label="input_manifest_sha256"
        ),
        "train_end_week": TRAIN_END_WEEK,
        "forecast_origin": FORECAST_ORIGIN,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "ordered_result_columns": list(EXPECTED_RESULT_COLUMNS),
    }


def write_secure_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically write one private ASCII JSON receipt."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            json.dump(payload, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

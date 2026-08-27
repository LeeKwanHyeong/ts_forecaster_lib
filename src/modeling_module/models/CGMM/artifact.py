"""Pickle-free CGMM artifact publication and strict restoration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from modeling_module.models.CGMM.configs import CGMMConfig
from modeling_module.models.CGMM.contracts import (
    CGMM_ARTIFACT_ID,
    CGMM_ARTIFACT_VERSION,
    CGMM_MODEL_ID,
    CGMM_MODEL_KEY,
    CGMMArtifactReceipt,
    CGMMContractError,
    CGMMCorrectionState,
    CGMMPreprocessingState,
    fingerprint_payload,
)
from modeling_module.models.CGMM.model import (
    ConditionalGaussianMixtureForecaster,
)


CGMM_MANIFEST_FILENAME = "manifest.json"
CGMM_ARRAYS_FILENAME = "model_arrays.npz"


class CGMMArtifactError(CGMMContractError):
    """Raised when an artifact is incomplete, altered, or incompatible."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_cgmm_artifact(
    model: ConditionalGaussianMixtureForecaster,
    artifact_dir: str | Path,
) -> CGMMArtifactReceipt:
    """Publish a fitted model as a sealed JSON and NPZ directory."""

    if not isinstance(model, ConditionalGaussianMixtureForecaster):
        raise TypeError("model must be ConditionalGaussianMixtureForecaster")
    if not model.is_fitted:
        raise CGMMArtifactError("model must be fitted before publication")
    destination = Path(artifact_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise CGMMArtifactError("artifact_dir must be empty or absent")
    destination.mkdir(parents=True, exist_ok=True)
    arrays_path = destination / CGMM_ARRAYS_FILENAME
    temporary_arrays = arrays_path.with_suffix(".npz.tmp")
    with temporary_arrays.open("wb") as handle:
        np.savez_compressed(handle, **model.export_state_arrays())
    temporary_arrays.replace(arrays_path)
    arrays_sha256 = _file_sha256(arrays_path)

    correction = model.correction_state
    payload: dict[str, Any] = {
        "artifact_id": CGMM_ARTIFACT_ID,
        "artifact_version": CGMM_ARTIFACT_VERSION,
        "model_key": CGMM_MODEL_KEY,
        "model_id": CGMM_MODEL_ID,
        "model_config": model.config.to_dict(),
        "model_fingerprint": model.model_fingerprint,
        "converged": model.converged,
        "iteration_count": model.iteration_count,
        "target_component_count": model.target_component_count,
        "preprocessing": model.preprocessing_state.to_dict(),
        "correction": None if correction is None else correction.to_dict(),
        "files": {
            "arrays": {
                "name": CGMM_ARRAYS_FILENAME,
                "sha256": arrays_sha256,
            }
        },
    }
    artifact_fingerprint = fingerprint_payload(payload)
    manifest = {**payload, "artifact_fingerprint": artifact_fingerprint}
    manifest_path = destination / CGMM_MANIFEST_FILENAME
    _write_json(manifest_path, manifest)
    return CGMMArtifactReceipt(
        artifact_dir=destination,
        manifest_path=manifest_path,
        arrays_path=arrays_path,
        model_fingerprint=model.model_fingerprint,
        artifact_fingerprint=artifact_fingerprint,
        arrays_sha256=arrays_sha256,
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CGMMArtifactError("CGMM manifest cannot be read") from exc
    if not isinstance(payload, dict):
        raise CGMMArtifactError("CGMM manifest must be a JSON object")
    required = {
        "artifact_id",
        "artifact_version",
        "model_key",
        "model_id",
        "model_config",
        "model_fingerprint",
        "converged",
        "iteration_count",
        "target_component_count",
        "preprocessing",
        "correction",
        "files",
        "artifact_fingerprint",
    }
    if set(payload) != required:
        raise CGMMArtifactError("CGMM manifest has an invalid schema")
    expected_fingerprint = fingerprint_payload(
        {
            key: value
            for key, value in payload.items()
            if key != "artifact_fingerprint"
        }
    )
    if payload["artifact_fingerprint"] != expected_fingerprint:
        raise CGMMArtifactError("CGMM artifact fingerprint mismatch")
    if (
        payload["artifact_id"] != CGMM_ARTIFACT_ID
        or payload["artifact_version"] != CGMM_ARTIFACT_VERSION
        or payload["model_key"] != CGMM_MODEL_KEY
        or payload["model_id"] != CGMM_MODEL_ID
    ):
        raise CGMMArtifactError("unsupported CGMM artifact identity")
    return payload


def load_cgmm_artifact(
    artifact_dir: str | Path,
) -> ConditionalGaussianMixtureForecaster:
    """Strictly restore a CGMM artifact after schema and SHA validation."""

    directory = Path(artifact_dir).expanduser().resolve()
    manifest_path = directory / CGMM_MANIFEST_FILENAME
    manifest = _load_manifest(manifest_path)
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != {"arrays"}:
        raise CGMMArtifactError("CGMM files manifest has an invalid schema")
    arrays_descriptor = files["arrays"]
    if (
        not isinstance(arrays_descriptor, dict)
        or set(arrays_descriptor) != {"name", "sha256"}
        or arrays_descriptor["name"] != CGMM_ARRAYS_FILENAME
    ):
        raise CGMMArtifactError("CGMM arrays descriptor is invalid")
    arrays_path = directory / arrays_descriptor["name"]
    if not arrays_path.is_file():
        raise CGMMArtifactError("CGMM arrays file is missing")
    if _file_sha256(arrays_path) != arrays_descriptor["sha256"]:
        raise CGMMArtifactError("CGMM arrays SHA-256 mismatch")
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise CGMMArtifactError("CGMM arrays cannot be read") from exc

    try:
        config = CGMMConfig(**dict(manifest["model_config"]))
        preprocessing = CGMMPreprocessingState.from_dict(
            manifest["preprocessing"]
        )
        correction_payload = manifest["correction"]
        correction = (
            None
            if correction_payload is None
            else CGMMCorrectionState.from_dict(correction_payload)
        )
        model = ConditionalGaussianMixtureForecaster.restore(
            config=config,
            preprocessing_state=preprocessing,
            arrays=arrays,
            converged=bool(manifest["converged"]),
            iteration_count=int(manifest["iteration_count"]),
            expected_model_fingerprint=str(manifest["model_fingerprint"]),
            correction_state=correction,
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, CGMMArtifactError):
            raise
        raise CGMMArtifactError("CGMM artifact state is incompatible") from exc
    if model.target_component_count != int(manifest["target_component_count"]):
        raise CGMMArtifactError("CGMM target component count mismatch")
    return model


__all__ = [
    "CGMM_ARRAYS_FILENAME",
    "CGMM_MANIFEST_FILENAME",
    "CGMMArtifactError",
    "load_cgmm_artifact",
    "save_cgmm_artifact",
]

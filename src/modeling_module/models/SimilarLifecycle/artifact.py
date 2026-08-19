"""Pickle-free Similar Lifecycle artifact publication and restoration."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from modeling_module.models.CGMM.contracts import (
    CGMMCorrectionState,
    CGMMPreprocessingState,
    fingerprint_payload,
)
from modeling_module.models.SimilarLifecycle.configs import (
    SimilarLifecycleConfig,
)
from modeling_module.models.SimilarLifecycle.contracts import (
    SIMILAR_LIFECYCLE_ARTIFACT_ID,
    SIMILAR_LIFECYCLE_ARTIFACT_VERSION,
    SIMILAR_LIFECYCLE_MODEL_ID,
    SIMILAR_LIFECYCLE_MODEL_KEY,
    SimilarLifecycleArtifactReceipt,
    SimilarLifecycleContractError,
    SimilarLifecycleRepositoryState,
)
from modeling_module.models.SimilarLifecycle.model import (
    SimilarLifecycleForecaster,
)


SIMILAR_LIFECYCLE_MANIFEST_FILENAME = "manifest.json"
SIMILAR_LIFECYCLE_ARRAYS_FILENAME = "repository_arrays.npz"


class SimilarLifecycleArtifactError(SimilarLifecycleContractError):
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


def save_similar_lifecycle_artifact(
    model: SimilarLifecycleForecaster,
    artifact_dir: str | Path,
) -> SimilarLifecycleArtifactReceipt:
    """Publish a fitted retrieval model with embedded correction state."""

    if not isinstance(model, SimilarLifecycleForecaster):
        raise TypeError("model must be SimilarLifecycleForecaster")
    if not model.is_fitted:
        raise SimilarLifecycleArtifactError(
            "model must be fitted before publication"
        )
    destination = Path(artifact_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise SimilarLifecycleArtifactError(
            "artifact_dir must be empty or absent"
        )
    destination.mkdir(parents=True, exist_ok=True)
    arrays_path = destination / SIMILAR_LIFECYCLE_ARRAYS_FILENAME
    temporary_arrays = arrays_path.with_suffix(".npz.tmp")
    repository = model.repository_state
    with temporary_arrays.open("wb") as handle:
        np.savez_compressed(
            handle,
            train_condition=np.asarray(
                repository.train_condition,
                dtype="<f8",
            ),
            train_future_ratio=np.asarray(
                repository.train_future_ratio,
                dtype="<f8",
            ),
        )
    temporary_arrays.replace(arrays_path)
    arrays_sha256 = _file_sha256(arrays_path)

    correction = model.correction_state
    payload: dict[str, Any] = {
        "artifact_id": SIMILAR_LIFECYCLE_ARTIFACT_ID,
        "artifact_version": SIMILAR_LIFECYCLE_ARTIFACT_VERSION,
        "model_key": SIMILAR_LIFECYCLE_MODEL_KEY,
        "model_id": SIMILAR_LIFECYCLE_MODEL_ID,
        "model_config": model.config.to_dict(),
        "model_fingerprint": model.model_fingerprint,
        "preprocessing": model.preprocessing_state.to_dict(),
        "correction": None if correction is None else correction.to_dict(),
        "repository": {
            "sample_ids": list(repository.sample_ids),
            "lifecycle_start_months": [
                value.isoformat()
                for value in repository.lifecycle_start_months
            ],
            "distance_feature_names": list(
                repository.distance_feature_names
            ),
        },
        "files": {
            "arrays": {
                "name": SIMILAR_LIFECYCLE_ARRAYS_FILENAME,
                "sha256": arrays_sha256,
            }
        },
    }
    artifact_fingerprint = fingerprint_payload(payload)
    manifest = {**payload, "artifact_fingerprint": artifact_fingerprint}
    manifest_path = destination / SIMILAR_LIFECYCLE_MANIFEST_FILENAME
    _write_json(manifest_path, manifest)
    return SimilarLifecycleArtifactReceipt(
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
        raise SimilarLifecycleArtifactError(
            "Similar Lifecycle manifest cannot be read"
        ) from exc
    if not isinstance(payload, dict):
        raise SimilarLifecycleArtifactError(
            "Similar Lifecycle manifest must be a JSON object"
        )
    required = {
        "artifact_id",
        "artifact_version",
        "model_key",
        "model_id",
        "model_config",
        "model_fingerprint",
        "preprocessing",
        "correction",
        "repository",
        "files",
        "artifact_fingerprint",
    }
    if set(payload) != required:
        raise SimilarLifecycleArtifactError(
            "Similar Lifecycle manifest has an invalid schema"
        )
    expected_fingerprint = fingerprint_payload(
        {
            key: value
            for key, value in payload.items()
            if key != "artifact_fingerprint"
        }
    )
    if payload["artifact_fingerprint"] != expected_fingerprint:
        raise SimilarLifecycleArtifactError("artifact fingerprint mismatch")
    if (
        payload["artifact_id"] != SIMILAR_LIFECYCLE_ARTIFACT_ID
        or payload["artifact_version"]
        != SIMILAR_LIFECYCLE_ARTIFACT_VERSION
        or payload["model_key"] != SIMILAR_LIFECYCLE_MODEL_KEY
        or payload["model_id"] != SIMILAR_LIFECYCLE_MODEL_ID
    ):
        raise SimilarLifecycleArtifactError(
            "unsupported Similar Lifecycle artifact identity"
        )
    return payload


def load_similar_lifecycle_artifact(
    artifact_dir: str | Path,
) -> SimilarLifecycleForecaster:
    """Strictly restore a checksum-verified Similar Lifecycle artifact."""

    directory = Path(artifact_dir).expanduser().resolve()
    manifest = _load_manifest(
        directory / SIMILAR_LIFECYCLE_MANIFEST_FILENAME
    )
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != {"arrays"}:
        raise SimilarLifecycleArtifactError(
            "artifact files manifest has an invalid schema"
        )
    descriptor = files["arrays"]
    if (
        not isinstance(descriptor, dict)
        or set(descriptor) != {"name", "sha256"}
        or descriptor["name"] != SIMILAR_LIFECYCLE_ARRAYS_FILENAME
    ):
        raise SimilarLifecycleArtifactError(
            "repository arrays descriptor is invalid"
        )
    arrays_path = directory / descriptor["name"]
    if not arrays_path.is_file():
        raise SimilarLifecycleArtifactError(
            "repository arrays file is missing"
        )
    if _file_sha256(arrays_path) != descriptor["sha256"]:
        raise SimilarLifecycleArtifactError(
            "repository arrays SHA-256 mismatch"
        )
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "train_condition",
                "train_future_ratio",
            }:
                raise SimilarLifecycleArtifactError(
                    "repository arrays have an invalid schema"
                )
            train_condition = np.asarray(archive["train_condition"])
            train_future_ratio = np.asarray(archive["train_future_ratio"])
    except (OSError, ValueError) as exc:
        raise SimilarLifecycleArtifactError(
            "repository arrays cannot be read"
        ) from exc
    if (
        train_condition.dtype != np.dtype("float64")
        or train_future_ratio.dtype != np.dtype("float64")
    ):
        raise SimilarLifecycleArtifactError(
            "repository arrays must use float64"
        )

    try:
        config = SimilarLifecycleConfig.from_config(manifest["model_config"])
        preprocessing = CGMMPreprocessingState.from_dict(
            manifest["preprocessing"]
        )
        correction_payload = manifest["correction"]
        correction = (
            None
            if correction_payload is None
            else CGMMCorrectionState.from_dict(correction_payload)
        )
        repository_payload = manifest["repository"]
        if not isinstance(repository_payload, dict) or set(
            repository_payload
        ) != {
            "sample_ids",
            "lifecycle_start_months",
            "distance_feature_names",
        }:
            raise ValueError("repository schema")
        repository = SimilarLifecycleRepositoryState(
            sample_ids=tuple(
                str(value) for value in repository_payload["sample_ids"]
            ),
            lifecycle_start_months=tuple(
                date.fromisoformat(str(value))
                for value in repository_payload["lifecycle_start_months"]
            ),
            distance_feature_names=tuple(
                str(value)
                for value in repository_payload["distance_feature_names"]
            ),
            train_condition=train_condition,
            train_future_ratio=train_future_ratio,
        )
        return SimilarLifecycleForecaster.restore(
            config=config,
            preprocessing_state=preprocessing,
            repository_state=repository,
            expected_model_fingerprint=str(manifest["model_fingerprint"]),
            correction_state=correction,
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, SimilarLifecycleArtifactError):
            raise
        raise SimilarLifecycleArtifactError(
            "Similar Lifecycle artifact state is incompatible"
        ) from exc


__all__ = [
    "SIMILAR_LIFECYCLE_ARRAYS_FILENAME",
    "SIMILAR_LIFECYCLE_MANIFEST_FILENAME",
    "SimilarLifecycleArtifactError",
    "load_similar_lifecycle_artifact",
    "save_similar_lifecycle_artifact",
]

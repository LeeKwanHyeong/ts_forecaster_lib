"""Immutable Parquet and JSON persistence for sealed ICL episode bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from modeling_module.icl.contracts import (
    ICL_MANIFEST_CONTRACT_ID,
    ICL_MANIFEST_CONTRACT_VERSION,
    ICLContractError,
    ICLDemonstration,
    ICLDemonstrationSeriesMode,
    ICLExogenousSchema,
    ICLEpisode,
    ICLEpisodeBundle,
    ICLManifest,
    ICLPromptKind,
    ICLSplit,
    ICLWindow,
    sha256_payload,
)


ICL_ARTIFACT_CONTRACT_ID = "modeling_module.icl_episode_artifact"
ICL_ARTIFACT_CONTRACT_VERSION = "1.0.0"
ICL_EPISODE_PARQUET = "episodes.parquet"
ICL_MANIFEST_JSON = "manifest.json"


class ICLArtifactError(ICLContractError):
    """Raised when a persisted ICL artifact is incomplete or has drifted."""


@dataclass(frozen=True)
class ICLArtifactReceipt:
    artifact_dir: str
    manifest_hash: str
    manifest_file_sha256: str
    episode_file_sha256: str
    episode_count: int
    exact_replay: bool


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _window_from_payload(payload: Mapping[str, Any]) -> ICLWindow:
    exogenous = payload.get("exogenous")
    return ICLWindow(
        weeks=tuple(int(value) for value in payload["weeks"]),
        target=tuple(
            tuple(float(value) for value in row) for row in payload["target"]
        ),
        exogenous=(
            None
            if exogenous is None
            else tuple(tuple(float(value) for value in row) for row in exogenous)
        ),
    )


def _episode_from_payload(payload: Mapping[str, Any]) -> ICLEpisode:
    demonstrations = tuple(
        ICLDemonstration(
            demonstration_id=str(item["demonstration_id"]),
            series_id=str(item["series_id"]),
            kind=ICLPromptKind(str(item["kind"])),
            context=_window_from_payload(item["context"]),
            target=_window_from_payload(item["target"]),
        )
        for item in payload["demonstrations"]
    )
    return ICLEpisode(
        episode_id=str(payload["episode_id"]),
        series_id=str(payload["series_id"]),
        split=ICLSplit(str(payload["split"])),
        source_revision=str(payload["source_revision"]),
        query_context=_window_from_payload(payload["query_context"]),
        query_target=_window_from_payload(payload["query_target"]),
        demonstrations=demonstrations,
        query_target_observed=bool(
            payload.get("query_target_observed", True)
        ),
        demonstration_series_mode=ICLDemonstrationSeriesMode(
            str(payload.get("demonstration_series_mode", "same_series"))
        ),
    )


def _manifest_from_payload(payload: Mapping[str, Any]) -> ICLManifest:
    if payload.get("contract_id") != ICL_MANIFEST_CONTRACT_ID:
        raise ICLArtifactError("Unsupported ICL manifest contract ID.")
    if payload.get("contract_version") != ICL_MANIFEST_CONTRACT_VERSION:
        raise ICLArtifactError("Unsupported ICL manifest contract version.")
    hash_payload = dict(payload)
    claimed_hash = str(hash_payload.pop("manifest_hash", ""))
    if sha256_payload(hash_payload) != claimed_hash:
        raise ICLArtifactError("ICL manifest hash mismatch.")
    schema_payload = payload.get("exogenous_schema")
    exogenous_schema = None
    if schema_payload is not None:
        schema_payload = dict(schema_payload)
        claimed_schema_hash = str(schema_payload.pop("schema_hash", ""))
        exogenous_schema = ICLExogenousSchema(
            past_feature_names=tuple(schema_payload["past_feature_names"]),
            future_feature_names=tuple(schema_payload["future_feature_names"]),
            source_revision=str(schema_payload["source_revision"]),
        )
        if exogenous_schema.fingerprint != claimed_schema_hash:
            raise ICLArtifactError("ICL exogenous schema hash mismatch.")
    manifest = ICLManifest(
        dataset_kind=str(payload["dataset_kind"]),
        source_revision=str(payload["source_revision"]),
        source_hash=str(payload["source_hash"]),
        config_hash=str(payload["config_hash"]),
        source_min_week=int(payload["source_min_week"]),
        source_max_week=int(payload["source_max_week"]),
        series_count=int(payload["series_count"]),
        episode_count=int(payload["episode_count"]),
        split_counts={
            str(key): int(value)
            for key, value in dict(payload["split_counts"]).items()
        },
        episode_hashes=tuple(str(value) for value in payload["episode_hashes"]),
        manifest_hash=claimed_hash,
        exogenous_schema=exogenous_schema,
    )
    if manifest.episode_count != len(manifest.episode_hashes):
        raise ICLArtifactError("Manifest episode count does not match episode hashes.")
    return manifest


def _episode_rows(bundle: ICLEpisodeBundle) -> list[dict[str, Any]]:
    return [
        {
            "episode_id": episode.episode_id,
            "series_id": episode.series_id,
            "split": episode.split.value,
            "origin_week": episode.origin_week,
            "episode_hash": episode.episode_hash,
            "episode_json": json.dumps(
                episode.to_payload(),
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
        }
        for episode in bundle.episodes
    ]


def write_icl_episode_artifact(
    bundle: ICLEpisodeBundle,
    artifact_dir: str | Path,
) -> ICLArtifactReceipt:
    """Write one immutable ICL artifact or return its exact existing replay."""

    root = Path(artifact_dir).expanduser().resolve()
    if root.exists():
        existing_bundle, existing_receipt = read_icl_episode_artifact(root)
        if existing_bundle.manifest.manifest_hash != bundle.manifest.manifest_hash:
            raise ICLArtifactError(
                "Artifact directory already contains a different ICL manifest."
            )
        return ICLArtifactReceipt(
            artifact_dir=existing_receipt.artifact_dir,
            manifest_hash=existing_receipt.manifest_hash,
            manifest_file_sha256=existing_receipt.manifest_file_sha256,
            episode_file_sha256=existing_receipt.episode_file_sha256,
            episode_count=existing_receipt.episode_count,
            exact_replay=True,
        )

    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.", dir=str(root.parent))
    )
    try:
        episode_path = temporary / ICL_EPISODE_PARQUET
        pl.DataFrame(_episode_rows(bundle)).write_parquet(
            episode_path,
            compression="zstd",
            statistics=True,
        )
        episode_sha256 = _file_sha256(episode_path)
        envelope = {
            "contract_id": ICL_ARTIFACT_CONTRACT_ID,
            "contract_version": ICL_ARTIFACT_CONTRACT_VERSION,
            "manifest": bundle.manifest.to_payload(),
            "episode_file": {
                "name": ICL_EPISODE_PARQUET,
                "sha256": episode_sha256,
                "row_count": len(bundle.episodes),
            },
        }
        manifest_path = temporary / ICL_MANIFEST_JSON
        manifest_path.write_text(
            json.dumps(envelope, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    _, receipt = read_icl_episode_artifact(root)
    return receipt


def read_icl_episode_artifact(
    artifact_dir: str | Path,
) -> tuple[ICLEpisodeBundle, ICLArtifactReceipt]:
    """Read an ICL artifact after verifying every persisted identity."""

    root = Path(artifact_dir).expanduser().resolve()
    manifest_path = root / ICL_MANIFEST_JSON
    episode_path = root / ICL_EPISODE_PARQUET
    if not manifest_path.is_file() or not episode_path.is_file():
        raise ICLArtifactError("ICL artifact requires manifest.json and episodes.parquet.")
    envelope = json.loads(manifest_path.read_text(encoding="utf-8"))
    if envelope.get("contract_id") != ICL_ARTIFACT_CONTRACT_ID:
        raise ICLArtifactError("Unsupported ICL artifact contract ID.")
    if envelope.get("contract_version") != ICL_ARTIFACT_CONTRACT_VERSION:
        raise ICLArtifactError("Unsupported ICL artifact contract version.")
    file_contract = dict(envelope.get("episode_file") or {})
    if file_contract.get("name") != ICL_EPISODE_PARQUET:
        raise ICLArtifactError("ICL artifact references an unexpected episode file.")
    episode_sha256 = _file_sha256(episode_path)
    if episode_sha256 != str(file_contract.get("sha256", "")):
        raise ICLArtifactError("ICL episode Parquet SHA256 mismatch.")

    manifest = _manifest_from_payload(dict(envelope["manifest"]))
    frame = pl.read_parquet(episode_path)
    if frame.height != int(file_contract.get("row_count", -1)):
        raise ICLArtifactError("ICL episode Parquet row count mismatch.")
    if frame.height != manifest.episode_count:
        raise ICLArtifactError("ICL Parquet and manifest episode counts differ.")

    episodes: list[ICLEpisode] = []
    for row in frame.iter_rows(named=True):
        payload = json.loads(str(row["episode_json"]))
        episode = _episode_from_payload(payload)
        if episode.episode_id != str(row["episode_id"]):
            raise ICLArtifactError("Episode ID differs between Parquet columns and payload.")
        if episode.series_id != str(row["series_id"]):
            raise ICLArtifactError("Series ID differs between Parquet columns and payload.")
        if episode.split.value != str(row["split"]):
            raise ICLArtifactError("Split differs between Parquet columns and payload.")
        if episode.origin_week != int(row["origin_week"]):
            raise ICLArtifactError("Origin week differs between Parquet columns and payload.")
        if episode.episode_hash != str(row["episode_hash"]):
            raise ICLArtifactError("ICL episode hash mismatch.")
        episodes.append(episode)

    bundle = ICLEpisodeBundle(episodes=tuple(episodes), manifest=manifest)
    return bundle, ICLArtifactReceipt(
        artifact_dir=str(root),
        manifest_hash=manifest.manifest_hash,
        manifest_file_sha256=_file_sha256(manifest_path),
        episode_file_sha256=episode_sha256,
        episode_count=len(episodes),
        exact_replay=False,
    )

"""Stable contracts shared by endogenous and exogenous ICL datasets."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from modeling_module.data_loader.temporal import add_period, normalize_period_key


ICL_EPISODE_CONTRACT_ID = "modeling_module.icl_episode"
ICL_EPISODE_CONTRACT_VERSION = "1.0.0"
ICL_MANIFEST_CONTRACT_ID = "modeling_module.icl_manifest"
ICL_MANIFEST_CONTRACT_VERSION = "1.0.0"


class ICLContractError(ValueError):
    """Raised when an ICL artifact violates the public contract."""


class ICLSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    INFERENCE = "inference"


class ICLPromptKind(str, Enum):
    HISTORICAL = "historical"
    SEASONAL = "seasonal"


@dataclass(frozen=True)
class ICLExogenousSchema:
    """Ordered continuous feature identity for past and known-future windows."""

    past_feature_names: tuple[str, ...]
    future_feature_names: tuple[str, ...]
    source_revision: str

    def __post_init__(self) -> None:
        past = tuple(str(value).strip() for value in self.past_feature_names)
        future = tuple(str(value).strip() for value in self.future_feature_names)
        if not past:
            raise ICLContractError("Exogenous ICL requires past feature names.")
        if not future:
            raise ICLContractError("Exogenous ICL requires known-future feature names.")
        if any(not value for value in (*past, *future)):
            raise ICLContractError("Exogenous ICL feature names must not be blank.")
        if len(set(past)) != len(past) or len(set(future)) != len(future):
            raise ICLContractError("Exogenous ICL feature names must be unique by role.")
        revision = str(self.source_revision).strip()
        if not revision:
            raise ICLContractError("Exogenous ICL source_revision must not be blank.")
        object.__setattr__(self, "past_feature_names", past)
        object.__setattr__(self, "future_feature_names", future)
        object.__setattr__(self, "source_revision", revision)

    @property
    def fingerprint(self) -> str:
        return sha256_payload(self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        return {
            "past_feature_names": list(self.past_feature_names),
            "future_feature_names": list(self.future_feature_names),
            "source_revision": self.source_revision,
        }


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_payload(payload: Any) -> str:
    """Return a deterministic SHA256 for a JSON-compatible payload."""

    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_matrix(
    values: tuple[tuple[float, ...], ...],
    *,
    row_count: int,
    field_name: str,
) -> int:
    if len(values) != row_count:
        raise ICLContractError(
            f"{field_name} row count must match weeks: {len(values)} != {row_count}."
        )
    if not values:
        raise ICLContractError(f"{field_name} must not be empty.")
    width = len(values[0])
    if width <= 0:
        raise ICLContractError(f"{field_name} must contain at least one feature.")
    for row in values:
        if len(row) != width:
            raise ICLContractError(f"{field_name} rows must have a stable width.")
        if not all(math.isfinite(float(value)) for value in row):
            raise ICLContractError(f"{field_name} must contain finite values only.")
    return width


@dataclass(frozen=True)
class ICLWindow:
    """One contiguous weekly target window and its optional exogenous features."""

    weeks: tuple[int, ...]
    target: tuple[tuple[float, ...], ...]
    exogenous: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        if not self.weeks:
            raise ICLContractError("ICLWindow.weeks must not be empty.")
        normalized = tuple(
            normalize_period_key(int(week), "weekly") for week in self.weeks
        )
        if normalized != self.weeks:
            raise ICLContractError("ICLWindow.weeks must use canonical integer YYYYWW values.")
        for previous, current in zip(self.weeks, self.weeks[1:]):
            if add_period(previous, 1, "weekly") != current:
                raise ICLContractError(
                    f"ICLWindow must be continuous between {previous} and {current}."
                )
        _validate_matrix(
            self.target,
            row_count=len(self.weeks),
            field_name="ICLWindow.target",
        )
        if self.exogenous is not None:
            _validate_matrix(
                self.exogenous,
                row_count=len(self.weeks),
                field_name="ICLWindow.exogenous",
            )

    @property
    def start_week(self) -> int:
        return self.weeks[0]

    @property
    def end_week(self) -> int:
        return self.weeks[-1]

    @property
    def target_dim(self) -> int:
        return len(self.target[0])

    @property
    def exogenous_dim(self) -> int:
        return 0 if self.exogenous is None else len(self.exogenous[0])

    def to_payload(self) -> dict[str, Any]:
        return {
            "weeks": list(self.weeks),
            "target": [list(row) for row in self.target],
            "exogenous": (
                None
                if self.exogenous is None
                else [list(row) for row in self.exogenous]
            ),
        }


@dataclass(frozen=True)
class ICLDemonstration:
    """A labeled prompt example from the same series as its query episode."""

    demonstration_id: str
    series_id: str
    kind: ICLPromptKind
    context: ICLWindow
    target: ICLWindow

    def __post_init__(self) -> None:
        if not self.demonstration_id.strip():
            raise ICLContractError("demonstration_id must not be blank.")
        if not self.series_id.strip():
            raise ICLContractError("series_id must not be blank.")
        if add_period(self.context.end_week, 1, "weekly") != self.target.start_week:
            raise ICLContractError("Demonstration target must immediately follow its context.")
        if self.context.target_dim != self.target.target_dim:
            raise ICLContractError("Demonstration context and target dimensions must match.")

    @property
    def start_week(self) -> int:
        return self.context.start_week

    @property
    def end_week(self) -> int:
        return self.target.end_week

    def to_payload(self) -> dict[str, Any]:
        return {
            "demonstration_id": self.demonstration_id,
            "series_id": self.series_id,
            "kind": self.kind.value,
            "context": self.context.to_payload(),
            "target": self.target.to_payload(),
        }


@dataclass(frozen=True)
class ICLEpisode:
    """One query and its leakage-free, same-series prompt demonstrations."""

    episode_id: str
    series_id: str
    split: ICLSplit
    source_revision: str
    query_context: ICLWindow
    query_target: ICLWindow
    demonstrations: tuple[ICLDemonstration, ...]
    query_target_observed: bool = True

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise ICLContractError("episode_id must not be blank.")
        if not self.series_id.strip():
            raise ICLContractError("series_id must not be blank.")
        if not self.source_revision.strip():
            raise ICLContractError("source_revision must not be blank.")
        if add_period(self.query_context.end_week, 1, "weekly") != self.query_target.start_week:
            raise ICLContractError("Query target must immediately follow query context.")
        if self.query_context.target_dim != self.query_target.target_dim:
            raise ICLContractError("Query context and target dimensions must match.")
        if not self.demonstrations:
            raise ICLContractError("An ICL episode requires at least one demonstration.")
        if self.split is ICLSplit.INFERENCE and self.query_target_observed:
            raise ICLContractError(
                "Inference episodes must mark query_target_observed=False."
            )
        if self.split is not ICLSplit.INFERENCE and not self.query_target_observed:
            raise ICLContractError(
                "Only inference episodes may contain an unobserved query target."
            )

        occupied: list[tuple[int, int, str]] = []
        for demonstration in self.demonstrations:
            if demonstration.series_id != self.series_id:
                raise ICLContractError("All demonstrations must come from the query series.")
            if demonstration.context.target_dim != self.query_context.target_dim:
                raise ICLContractError("Prompt and query target dimensions must match.")
            if demonstration.context.exogenous_dim != self.query_context.exogenous_dim:
                raise ICLContractError("Prompt and query past exogenous dimensions must match.")
            if demonstration.target.exogenous_dim != self.query_target.exogenous_dim:
                raise ICLContractError("Prompt and query future exogenous dimensions must match.")
            if demonstration.end_week >= self.query_context.start_week:
                raise ICLContractError(
                    "Prompt demonstrations must end before the query context starts."
                )
            occupied.append(
                (
                    demonstration.start_week,
                    demonstration.end_week,
                    demonstration.demonstration_id,
                )
            )
        occupied.sort()
        for previous, current in zip(occupied, occupied[1:]):
            if previous[1] >= current[0]:
                raise ICLContractError(
                    "Prompt demonstrations must not overlap: "
                    f"{previous[2]!r} and {current[2]!r}."
                )

    @property
    def origin_week(self) -> int:
        return self.query_target.start_week

    @property
    def episode_hash(self) -> str:
        return sha256_payload(self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "contract_id": ICL_EPISODE_CONTRACT_ID,
            "contract_version": ICL_EPISODE_CONTRACT_VERSION,
            "episode_id": self.episode_id,
            "series_id": self.series_id,
            "split": self.split.value,
            "source_revision": self.source_revision,
            "query_context": self.query_context.to_payload(),
            "query_target": self.query_target.to_payload(),
            "demonstrations": [item.to_payload() for item in self.demonstrations],
        }
        # Omit the legacy default so existing qualification episode hashes remain stable.
        if not self.query_target_observed:
            payload["query_target_observed"] = False
        return payload


@dataclass(frozen=True)
class ICLManifest:
    """Deterministic seal for an immutable collection of ICL episodes."""

    dataset_kind: str
    source_revision: str
    source_hash: str
    config_hash: str
    source_min_week: int
    source_max_week: int
    series_count: int
    episode_count: int
    split_counts: Mapping[str, int]
    episode_hashes: tuple[str, ...]
    manifest_hash: str
    exogenous_schema: ICLExogenousSchema | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "split_counts",
            MappingProxyType(dict(self.split_counts)),
        )
        if self.dataset_kind == "exogenous" and self.exogenous_schema is None:
            raise ICLContractError("Exogenous ICL manifest requires a feature schema.")
        if self.dataset_kind != "exogenous" and self.exogenous_schema is not None:
            raise ICLContractError("Only exogenous ICL manifests may declare a feature schema.")

    @classmethod
    def create(
        cls,
        *,
        dataset_kind: str,
        source_revision: str,
        source_hash: str,
        config_hash: str,
        source_min_week: int,
        source_max_week: int,
        series_count: int,
        episodes: Iterable[ICLEpisode],
        exogenous_schema: ICLExogenousSchema | None = None,
    ) -> "ICLManifest":
        ordered = tuple(episodes)
        split_counts = {
            split.value: sum(item.split is split for item in ordered)
            for split in (
                ICLSplit.TRAIN,
                ICLSplit.VALIDATION,
                ICLSplit.TEST,
            )
        }
        inference_count = sum(
            item.split is ICLSplit.INFERENCE for item in ordered
        )
        if inference_count:
            split_counts[ICLSplit.INFERENCE.value] = inference_count
        episode_hashes = tuple(item.episode_hash for item in ordered)
        payload = {
            "contract_id": ICL_MANIFEST_CONTRACT_ID,
            "contract_version": ICL_MANIFEST_CONTRACT_VERSION,
            "dataset_kind": dataset_kind,
            "source_revision": source_revision,
            "source_hash": source_hash,
            "config_hash": config_hash,
            "source_min_week": source_min_week,
            "source_max_week": source_max_week,
            "series_count": series_count,
            "episode_count": len(ordered),
            "split_counts": split_counts,
            "episode_hashes": list(episode_hashes),
        }
        if exogenous_schema is not None:
            payload["exogenous_schema"] = {
                **exogenous_schema.to_payload(),
                "schema_hash": exogenous_schema.fingerprint,
            }
        return cls(
            dataset_kind=dataset_kind,
            source_revision=source_revision,
            source_hash=source_hash,
            config_hash=config_hash,
            source_min_week=source_min_week,
            source_max_week=source_max_week,
            series_count=series_count,
            episode_count=len(ordered),
            split_counts=split_counts,
            episode_hashes=episode_hashes,
            manifest_hash=sha256_payload(payload),
            exogenous_schema=exogenous_schema,
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "contract_id": ICL_MANIFEST_CONTRACT_ID,
            "contract_version": ICL_MANIFEST_CONTRACT_VERSION,
            "dataset_kind": self.dataset_kind,
            "source_revision": self.source_revision,
            "source_hash": self.source_hash,
            "config_hash": self.config_hash,
            "source_min_week": self.source_min_week,
            "source_max_week": self.source_max_week,
            "series_count": self.series_count,
            "episode_count": self.episode_count,
            "split_counts": dict(self.split_counts),
            "episode_hashes": list(self.episode_hashes),
            "manifest_hash": self.manifest_hash,
        }
        if self.exogenous_schema is not None:
            payload["exogenous_schema"] = {
                **self.exogenous_schema.to_payload(),
                "schema_hash": self.exogenous_schema.fingerprint,
            }
        return payload


@dataclass(frozen=True)
class ICLEpisodeBundle:
    episodes: tuple[ICLEpisode, ...]
    manifest: ICLManifest

    def __post_init__(self) -> None:
        hashes = tuple(episode.episode_hash for episode in self.episodes)
        if hashes != self.manifest.episode_hashes:
            raise ICLContractError("Bundle episodes do not match the sealed manifest.")

    def for_split(self, split: ICLSplit | str) -> tuple[ICLEpisode, ...]:
        normalized = ICLSplit(split)
        return tuple(item for item in self.episodes if item.split is normalized)

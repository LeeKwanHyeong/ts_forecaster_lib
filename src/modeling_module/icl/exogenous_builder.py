"""Deterministic exogenous ICL episodes with role-specific feature schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import polars as pl

from modeling_module.data_loader.temporal import normalize_period_key
from modeling_module.icl.contracts import (
    ICLContractError,
    ICLDemonstration,
    ICLEpisode,
    ICLEpisodeBundle,
    ICLExogenousSchema,
    ICLManifest,
    ICLWindow,
    sha256_payload,
)
from modeling_module.icl.endogenous_builder import (
    EndogenousICLBuilderConfig,
    EndogenousICLDatasetBuilder,
)


@dataclass(frozen=True)
class ExogenousICLBuilderConfig:
    episode: EndogenousICLBuilderConfig
    past_feature_cols: tuple[str, ...]
    future_feature_cols: tuple[str, ...]

    def __post_init__(self) -> None:
        past = tuple(str(value).strip() for value in self.past_feature_cols)
        future = tuple(str(value).strip() for value in self.future_feature_cols)
        if not past or not future:
            raise ValueError("Exogenous ICL requires past and future feature columns.")
        if any(not value for value in (*past, *future)):
            raise ValueError("Exogenous feature column names must not be blank.")
        if len(set(past)) != len(past) or len(set(future)) != len(future):
            raise ValueError("Exogenous feature columns must be unique by role.")
        object.__setattr__(self, "past_feature_cols", past)
        object.__setattr__(self, "future_feature_cols", future)


class ExogenousICLDatasetBuilder:
    """Enrich leakage-free endogenous episodes with approved exogenous features."""

    def __init__(self, config: ExogenousICLBuilderConfig) -> None:
        self.config = config

    def build(
        self,
        frame: pl.DataFrame,
        *,
        source_revision: str,
        exogenous_source_revision: str,
    ) -> ICLEpisodeBundle:
        cfg = self.config.episode
        required = tuple(
            dict.fromkeys(
                (
                    cfg.part_col,
                    cfg.week_col,
                    cfg.value_col,
                    *self.config.past_feature_cols,
                    *self.config.future_feature_cols,
                )
            )
        )
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Exogenous ICL source is missing columns: {missing}.")
        feature_cols = tuple(
            dict.fromkeys(
                (*self.config.past_feature_cols, *self.config.future_feature_cols)
            )
        )
        normalized = frame.select(required).with_columns(
            pl.col(cfg.part_col).cast(pl.String),
            pl.Series(
                cfg.week_col,
                [normalize_period_key(value, "weekly") for value in frame[cfg.week_col]],
                dtype=pl.Int64,
            ),
            pl.col(cfg.value_col).cast(pl.Float64),
            *[
                pl.col(column).cast(pl.Float64)
                for column in feature_cols
            ],
        )
        nulls = normalized.select(
            [pl.col(column).null_count().alias(column) for column in required]
        ).row(0, named=True)
        populated_nulls = {name: int(count) for name, count in nulls.items() if count}
        if populated_nulls:
            raise ValueError(f"Exogenous ICL source contains nulls: {populated_nulls}.")
        numeric_columns = (cfg.value_col, *feature_cols)
        for column in numeric_columns:
            if not np.isfinite(normalized[column].to_numpy()).all():
                raise ValueError(f"Exogenous ICL column {column!r} contains non-finite values.")

        lookup: dict[tuple[str, int], tuple[tuple[float, ...], tuple[float, ...]]] = {}
        for row in normalized.sort([cfg.part_col, cfg.week_col]).iter_rows(named=True):
            key = (str(row[cfg.part_col]), int(row[cfg.week_col]))
            value = (
                tuple(float(row[column]) for column in self.config.past_feature_cols),
                tuple(float(row[column]) for column in self.config.future_feature_cols),
            )
            existing = lookup.get(key)
            if existing is not None and existing != value:
                raise ICLContractError(
                    "Duplicate item-week rows contain conflicting exogenous values: "
                    f"{key}."
                )
            lookup[key] = value

        base = EndogenousICLDatasetBuilder(cfg).build(
            normalized,
            source_revision=source_revision,
        )

        def enrich_window(window: ICLWindow, *, role: str, series_id: str) -> ICLWindow:
            index = 0 if role == "past" else 1
            values = tuple(lookup[(series_id, week)][index] for week in window.weeks)
            return replace(window, exogenous=values)

        episodes: list[ICLEpisode] = []
        for episode in base.episodes:
            demonstrations = tuple(
                ICLDemonstration(
                    demonstration_id=item.demonstration_id,
                    series_id=item.series_id,
                    kind=item.kind,
                    context=enrich_window(
                        item.context,
                        role="past",
                        series_id=item.series_id,
                    ),
                    target=enrich_window(
                        item.target,
                        role="future",
                        series_id=item.series_id,
                    ),
                )
                for item in episode.demonstrations
            )
            episodes.append(
                ICLEpisode(
                    episode_id=episode.episode_id,
                    series_id=episode.series_id,
                    split=episode.split,
                    source_revision=episode.source_revision,
                    query_context=enrich_window(
                        episode.query_context,
                        role="past",
                        series_id=episode.series_id,
                    ),
                    query_target=enrich_window(
                        episode.query_target,
                        role="future",
                        series_id=episode.series_id,
                    ),
                    demonstrations=demonstrations,
                )
            )

        ordered = tuple(episodes)
        schema = ICLExogenousSchema(
            past_feature_names=self.config.past_feature_cols,
            future_feature_names=self.config.future_feature_cols,
            source_revision=exogenous_source_revision,
        )
        source_rows = normalized.sort([cfg.part_col, cfg.week_col]).rows(named=True)
        manifest = ICLManifest.create(
            dataset_kind="exogenous",
            source_revision=source_revision,
            source_hash=sha256_payload(source_rows),
            config_hash=sha256_payload(asdict(self.config)),
            source_min_week=base.manifest.source_min_week,
            source_max_week=base.manifest.source_max_week,
            series_count=base.manifest.series_count,
            episodes=ordered,
            exogenous_schema=schema,
        )
        return ICLEpisodeBundle(episodes=ordered, manifest=manifest)


__all__ = ["ExogenousICLBuilderConfig", "ExogenousICLDatasetBuilder"]

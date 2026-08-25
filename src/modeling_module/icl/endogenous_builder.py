"""Deterministic rolling ICL episodes for weekly endogenous demand."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import numpy as np
import polars as pl

from modeling_module.data_loader.temporal import normalize_period_key
from modeling_module.icl.contracts import (
    ICLContractError,
    ICLDemonstration,
    ICLEpisode,
    ICLEpisodeBundle,
    ICLManifest,
    ICLPromptKind,
    ICLSplit,
    ICLWindow,
    sha256_payload,
)


@dataclass(frozen=True)
class EndogenousICLBuilderConfig:
    lookback: int = 52
    horizon: int = 26
    window_stride: int = 1
    seasonal_period: int = 52
    validation_episodes_per_series: int = 1
    test_episodes_per_series: int = 1
    part_col: str = "oper_part_no"
    week_col: str = "demand_dt"
    value_col: str = "demand_qty"

    def __post_init__(self) -> None:
        positive = {
            "lookback": self.lookback,
            "horizon": self.horizon,
            "window_stride": self.window_stride,
            "seasonal_period": self.seasonal_period,
        }
        for name, value in positive.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.validation_episodes_per_series < 0:
            raise ValueError("validation_episodes_per_series must be non-negative.")
        if self.test_episodes_per_series < 0:
            raise ValueError("test_episodes_per_series must be non-negative.")


@dataclass(frozen=True)
class _Series:
    series_id: str
    weeks: tuple[int, ...]
    values: tuple[float, ...]


class EndogenousICLDatasetBuilder:
    """Build leakage-free rolling episodes from complete weekly demand history."""

    def __init__(self, config: EndogenousICLBuilderConfig | None = None) -> None:
        self.config = config or EndogenousICLBuilderConfig()

    def build(
        self,
        frame: pl.DataFrame,
        *,
        source_revision: str,
    ) -> ICLEpisodeBundle:
        if not source_revision.strip():
            raise ValueError("source_revision must not be blank.")
        normalized = self._normalize_frame(frame)
        series = self._build_series(normalized)
        episodes: list[ICLEpisode] = []
        for item in series:
            item_episodes = self._build_series_episodes(item, source_revision)
            episodes.extend(self._assign_splits(item_episodes))
        if not episodes:
            raise ICLContractError(
                "No ICL episodes were created. The source needs enough continuous history "
                "for two non-overlapping prompts, one query context, and one target."
            )
        ordered = tuple(sorted(episodes, key=lambda item: (item.series_id, item.origin_week)))
        source_payload = normalized.select(
            [self.config.part_col, self.config.week_col, self.config.value_col]
        ).rows(named=True)
        manifest = ICLManifest.create(
            dataset_kind="endogenous",
            source_revision=source_revision,
            source_hash=sha256_payload(source_payload),
            config_hash=sha256_payload(asdict(self.config)),
            source_min_week=int(normalized[self.config.week_col].min()),
            source_max_week=int(normalized[self.config.week_col].max()),
            series_count=len(series),
            episodes=ordered,
        )
        return ICLEpisodeBundle(episodes=ordered, manifest=manifest)

    def _normalize_frame(self, frame: pl.DataFrame) -> pl.DataFrame:
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(f"frame must be a polars DataFrame, got {type(frame)!r}.")
        required = [self.config.part_col, self.config.week_col, self.config.value_col]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"demand history is missing required columns: {missing}.")
        if frame.height == 0:
            raise ValueError("demand history must contain at least one row.")

        weeks = [
            normalize_period_key(value, "weekly")
            for value in frame[self.config.week_col].to_list()
        ]
        normalized = (
            frame.select(required)
            .with_columns(
                pl.col(self.config.part_col).cast(pl.String),
                pl.Series(self.config.week_col, weeks, dtype=pl.Int64),
                pl.col(self.config.value_col).cast(pl.Float64),
            )
        )
        null_counts = normalized.select(
            [pl.col(column).null_count().alias(column) for column in required]
        ).row(0, named=True)
        populated_nulls = {name: int(count) for name, count in null_counts.items() if count}
        if populated_nulls:
            raise ValueError(f"demand history contains nulls: {populated_nulls}.")
        values = normalized[self.config.value_col].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("demand history values must be finite.")
        if np.any(values < 0):
            raise ValueError("demand history values must be non-negative.")

        return (
            normalized.group_by([self.config.part_col, self.config.week_col])
            .agg(pl.col(self.config.value_col).sum())
            .sort([self.config.part_col, self.config.week_col])
        )

    def _build_series(self, frame: pl.DataFrame) -> tuple[_Series, ...]:
        output: list[_Series] = []
        for group in frame.partition_by(self.config.part_col, maintain_order=True):
            series_id = str(group[self.config.part_col][0])
            weeks = tuple(int(value) for value in group[self.config.week_col].to_list())
            ordinals = tuple(
                date.fromisocalendar(week // 100, week % 100, 1).toordinal()
                for week in weeks
            )
            for previous, current in zip(ordinals, ordinals[1:]):
                if current - previous != 7:
                    raise ICLContractError(
                        f"Series {series_id!r} contains a missing weekly period. "
                        "Zero-demand weeks must be materialized by the approved source contract."
                    )
            output.append(
                _Series(
                    series_id=series_id,
                    weeks=weeks,
                    values=tuple(float(value) for value in group[self.config.value_col].to_list()),
                )
            )
        return tuple(output)

    def _build_series_episodes(
        self,
        series: _Series,
        source_revision: str,
    ) -> list[ICLEpisode]:
        cfg = self.config
        total = cfg.lookback + cfg.horizon
        latest_start = len(series.weeks) - total
        episodes: list[ICLEpisode] = []
        for query_start in range(0, latest_start + 1, cfg.window_stride):
            query_target_start = query_start + cfg.lookback
            historical_start = query_start - total
            if historical_start < 0:
                continue

            seasonal_target_start = query_target_start - cfg.seasonal_period
            while seasonal_target_start + cfg.horizon > historical_start:
                seasonal_target_start -= cfg.seasonal_period
            seasonal_start = seasonal_target_start - cfg.lookback
            if seasonal_start < 0:
                continue

            demonstrations = (
                self._demonstration(
                    series,
                    kind=ICLPromptKind.SEASONAL,
                    start=seasonal_start,
                    source_revision=source_revision,
                ),
                self._demonstration(
                    series,
                    kind=ICLPromptKind.HISTORICAL,
                    start=historical_start,
                    source_revision=source_revision,
                ),
            )
            origin_week = series.weeks[query_target_start]
            episode_id = f"{series.series_id}:{origin_week}:{source_revision}"
            episodes.append(
                ICLEpisode(
                    episode_id=episode_id,
                    series_id=series.series_id,
                    split=ICLSplit.TRAIN,
                    source_revision=source_revision,
                    query_context=self._window(
                        series,
                        query_start,
                        query_target_start,
                    ),
                    query_target=self._window(
                        series,
                        query_target_start,
                        query_target_start + cfg.horizon,
                    ),
                    demonstrations=demonstrations,
                )
            )
        return episodes

    def _demonstration(
        self,
        series: _Series,
        *,
        kind: ICLPromptKind,
        start: int,
        source_revision: str,
    ) -> ICLDemonstration:
        target_start = start + self.config.lookback
        target_end = target_start + self.config.horizon
        start_week = series.weeks[start]
        target_week = series.weeks[target_start]
        return ICLDemonstration(
            demonstration_id=(
                f"{series.series_id}:{kind.value}:{start_week}:{target_week}:"
                f"{source_revision}"
            ),
            series_id=series.series_id,
            kind=kind,
            context=self._window(series, start, target_start),
            target=self._window(series, target_start, target_end),
        )

    @staticmethod
    def _window(series: _Series, start: int, end: int) -> ICLWindow:
        return ICLWindow(
            weeks=series.weeks[start:end],
            target=tuple((value,) for value in series.values[start:end]),
        )

    def _assign_splits(self, episodes: list[ICLEpisode]) -> tuple[ICLEpisode, ...]:
        validation_count = self.config.validation_episodes_per_series
        test_count = self.config.test_episodes_per_series
        if not episodes:
            return ()

        def select_holdout(
            candidates: list[ICLEpisode],
            *,
            count: int,
            before_week: int | None = None,
        ) -> list[ICLEpisode]:
            selected: list[ICLEpisode] = []
            boundary = before_week
            for episode in reversed(candidates):
                if boundary is not None and episode.query_target.end_week >= boundary:
                    continue
                selected.append(episode)
                boundary = episode.query_target.start_week
                if len(selected) == count:
                    break
            return list(reversed(selected))

        test = select_holdout(episodes, count=test_count) if test_count else []
        if len(test) != test_count:
            return ()
        test_ids = {episode.episode_id for episode in test}
        validation_candidates = [
            episode for episode in episodes if episode.episode_id not in test_ids
        ]
        validation = (
            select_holdout(
                validation_candidates,
                count=validation_count,
                before_week=(test[0].query_target.start_week if test else None),
            )
            if validation_count
            else []
        )
        if len(validation) != validation_count:
            return ()

        validation_ids = {episode.episode_id for episode in validation}
        holdout_boundary = (
            validation[0].query_target.start_week
            if validation
            else test[0].query_target.start_week if test else None
        )
        train = [
            episode
            for episode in episodes
            if episode.episode_id not in test_ids
            and episode.episode_id not in validation_ids
            and (
                holdout_boundary is None
                or episode.query_target.end_week < holdout_boundary
            )
        ]
        if not train:
            return ()

        split_by_id = {
            **{episode.episode_id: ICLSplit.TRAIN for episode in train},
            **{episode.episode_id: ICLSplit.VALIDATION for episode in validation},
            **{episode.episode_id: ICLSplit.TEST for episode in test},
        }
        assigned: list[ICLEpisode] = []
        for episode in episodes:
            split = split_by_id.get(episode.episode_id)
            if split is None:
                continue
            assigned.append(
                ICLEpisode(
                    episode_id=episode.episode_id,
                    series_id=episode.series_id,
                    split=split,
                    source_revision=episode.source_revision,
                    query_context=episode.query_context,
                    query_target=episode.query_target,
                    demonstrations=episode.demonstrations,
                )
            )
        return tuple(assigned)

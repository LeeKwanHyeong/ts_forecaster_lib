"""Build sealed ICL episodes for an anchored forecast with no future labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import polars as pl

from modeling_module.data_loader.temporal import add_period, normalize_period_key
from modeling_module.icl.contracts import (
    ICLContractError,
    ICLEpisode,
    ICLEpisodeBundle,
    ICLExogenousSchema,
    ICLManifest,
    ICLSplit,
    ICLWindow,
    sha256_payload,
)
from modeling_module.icl.endogenous_builder import EndogenousICLBuilderConfig
from modeling_module.icl.exogenous_builder import (
    ExogenousICLBuilderConfig,
    ExogenousICLDatasetBuilder,
)


@dataclass(frozen=True)
class ICLInferenceBuilderConfig:
    lookback: int = 52
    horizon: int = 26
    demonstration_stride: int = 26
    seasonal_period: int = 52
    part_col: str = "oper_part_no"
    week_col: str = "demand_dt"
    value_col: str = "demand_qty"
    past_feature_cols: tuple[str, ...] = ()
    future_feature_cols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        dimensions = (
            self.lookback,
            self.horizon,
            self.demonstration_stride,
            self.seasonal_period,
        )
        if min(int(value) for value in dimensions) <= 0:
            raise ValueError("ICL inference dimensions must be positive.")
        if not self.past_feature_cols or not self.future_feature_cols:
            raise ValueError(
                "ICL inference requires ordered past and future feature columns."
            )
        if len(set(self.past_feature_cols)) != len(self.past_feature_cols):
            raise ValueError("past_feature_cols must be unique.")
        if len(set(self.future_feature_cols)) != len(self.future_feature_cols):
            raise ValueError("future_feature_cols must be unique.")


class ExogenousICLInferenceBuilder:
    """Create one label-free W0-W(H-1) episode per explicitly active series."""

    def __init__(self, config: ICLInferenceBuilderConfig) -> None:
        self.config = config

    def build(
        self,
        history: pl.DataFrame,
        known_future: pl.DataFrame,
        *,
        active_series_ids: Iterable[str],
        forecast_origin: int,
        source_revision: str,
        exogenous_source_revision: str,
    ) -> ICLEpisodeBundle:
        cfg = self.config
        origin = normalize_period_key(forecast_origin, "weekly")
        active = tuple(sorted({str(value).strip() for value in active_series_ids}))
        if not active or any(not value for value in active):
            raise ValueError("active_series_ids must contain non-blank identifiers.")
        if not source_revision.strip() or not exogenous_source_revision.strip():
            raise ValueError("ICL inference source revisions must not be blank.")

        history_required = tuple(
            dict.fromkeys(
                (
                    cfg.part_col,
                    cfg.week_col,
                    cfg.value_col,
                    *cfg.past_feature_cols,
                    *cfg.future_feature_cols,
                )
            )
        )
        future_required = (
            cfg.part_col,
            cfg.week_col,
            *cfg.future_feature_cols,
        )
        self._require_columns(history, history_required, label="history")
        self._require_columns(known_future, future_required, label="known_future")

        history = self._normalize_history(history, active, origin)
        known_future = self._normalize_future(known_future, active, origin)
        observed_ids = set(history[cfg.part_col].to_list())
        future_ids = set(known_future[cfg.part_col].to_list())
        missing_history = set(active) - observed_ids
        missing_future = set(active) - future_ids
        if missing_history or missing_future:
            raise ICLContractError(
                "Active ICL series lack required inputs: "
                f"history={sorted(missing_history)}, future={sorted(missing_future)}."
            )

        training_bundle = ExogenousICLDatasetBuilder(
            ExogenousICLBuilderConfig(
                episode=EndogenousICLBuilderConfig(
                    lookback=cfg.lookback,
                    horizon=cfg.horizon,
                    window_stride=cfg.demonstration_stride,
                    seasonal_period=cfg.seasonal_period,
                    validation_episodes_per_series=0,
                    test_episodes_per_series=0,
                    part_col=cfg.part_col,
                    week_col=cfg.week_col,
                    value_col=cfg.value_col,
                ),
                past_feature_cols=cfg.past_feature_cols,
                future_feature_cols=cfg.future_feature_cols,
            )
        ).build(
            history,
            source_revision=source_revision,
            exogenous_source_revision=exogenous_source_revision,
        )
        candidates: dict[str, list[ICLEpisode]] = {}
        for episode in training_bundle.for_split(ICLSplit.TRAIN):
            candidates.setdefault(episode.series_id, []).append(episode)

        future_weeks = tuple(
            add_period(origin, step, "weekly") for step in range(cfg.horizon)
        )
        episodes: list[ICLEpisode] = []
        for series_id in active:
            series_history = history.filter(
                pl.col(cfg.part_col) == series_id
            ).sort(cfg.week_col)
            if series_history.height < cfg.lookback:
                raise ICLContractError(
                    f"Active series {series_id!r} has fewer than L{cfg.lookback} rows."
                )
            query_rows = series_history.tail(cfg.lookback)
            if add_period(int(query_rows[cfg.week_col][-1]), 1, "weekly") != origin:
                raise ICLContractError(
                    f"Active series {series_id!r} does not end immediately before W0."
                )
            query_context = ICLWindow(
                weeks=tuple(int(value) for value in query_rows[cfg.week_col]),
                target=tuple(
                    (float(value),) for value in query_rows[cfg.value_col]
                ),
                exogenous=tuple(
                    tuple(float(row[column]) for column in cfg.past_feature_cols)
                    for row in query_rows.iter_rows(named=True)
                ),
            )
            prompt_source = next(
                (
                    item
                    for item in reversed(candidates.get(series_id, []))
                    if all(
                        prompt.end_week < query_context.start_week
                        for prompt in item.demonstrations
                    )
                ),
                None,
            )
            if prompt_source is None:
                raise ICLContractError(
                    f"Active series {series_id!r} lacks two leakage-free demonstrations."
                )
            if len(prompt_source.demonstrations) != 2:
                raise ICLContractError(
                    f"Active series {series_id!r} must provide exactly two "
                    "leakage-free demonstrations."
                )
            series_future = known_future.filter(
                pl.col(cfg.part_col) == series_id
            ).sort(cfg.week_col)
            observed_future_weeks = tuple(
                int(value) for value in series_future[cfg.week_col]
            )
            if observed_future_weeks != future_weeks:
                raise ICLContractError(
                    f"Active series {series_id!r} must provide exact "
                    f"W0-W{cfg.horizon - 1} known-future rows."
                )
            query_target = ICLWindow(
                weeks=future_weeks,
                target=tuple((0.0,) for _ in future_weeks),
                exogenous=tuple(
                    tuple(float(row[column]) for column in cfg.future_feature_cols)
                    for row in series_future.iter_rows(named=True)
                ),
            )
            episodes.append(
                ICLEpisode(
                    episode_id=f"{series_id}:{origin}:{source_revision}:inference",
                    series_id=series_id,
                    split=ICLSplit.INFERENCE,
                    source_revision=source_revision,
                    query_context=query_context,
                    query_target=query_target,
                    demonstrations=tuple(prompt_source.demonstrations),
                    query_target_observed=False,
                )
            )

        ordered = tuple(sorted(episodes, key=lambda item: item.series_id))
        schema = ICLExogenousSchema(
            past_feature_names=cfg.past_feature_cols,
            future_feature_names=cfg.future_feature_cols,
            source_revision=exogenous_source_revision,
        )
        source_payload = {
            "history": history.sort([cfg.part_col, cfg.week_col]).rows(named=True),
            "known_future": known_future.sort(
                [cfg.part_col, cfg.week_col]
            ).rows(named=True),
            "active_series_ids": list(active),
            "forecast_origin": origin,
        }
        return ICLEpisodeBundle(
            episodes=ordered,
            manifest=ICLManifest.create(
                dataset_kind="exogenous",
                source_revision=source_revision,
                source_hash=sha256_payload(source_payload),
                config_hash=sha256_payload(asdict(cfg)),
                source_min_week=int(history[cfg.week_col].min()),
                source_max_week=int(known_future[cfg.week_col].max()),
                series_count=len(active),
                episodes=ordered,
                exogenous_schema=schema,
            ),
        )

    @staticmethod
    def _require_columns(
        frame: pl.DataFrame,
        required: tuple[str, ...],
        *,
        label: str,
    ) -> None:
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"ICL inference {label} is missing columns: {missing}.")

    def _normalize_history(
        self,
        frame: pl.DataFrame,
        active: tuple[str, ...],
        origin: int,
    ) -> pl.DataFrame:
        cfg = self.config
        columns = tuple(
            dict.fromkeys(
                (
                    cfg.part_col,
                    cfg.week_col,
                    cfg.value_col,
                    *cfg.past_feature_cols,
                    *cfg.future_feature_cols,
                )
            )
        )
        feature_columns = tuple(
            dict.fromkeys((*cfg.past_feature_cols, *cfg.future_feature_cols))
        )
        normalized = frame.select(columns).with_columns(
            pl.col(cfg.part_col).cast(pl.String),
            pl.Series(
                cfg.week_col,
                [normalize_period_key(value, "weekly") for value in frame[cfg.week_col]],
                dtype=pl.Int64,
            ),
            pl.col(cfg.value_col).cast(pl.Float64),
            *[pl.col(column).cast(pl.Float64) for column in feature_columns],
        )
        normalized = normalized.filter(
            pl.col(cfg.part_col).is_in(active),
            pl.col(cfg.week_col) < origin,
        ).sort([cfg.part_col, cfg.week_col])
        if any(normalized.null_count().row(0)):
            raise ValueError("ICL inference history contains null values.")
        return normalized

    def _normalize_future(
        self,
        frame: pl.DataFrame,
        active: tuple[str, ...],
        origin: int,
    ) -> pl.DataFrame:
        cfg = self.config
        final_week = add_period(origin, cfg.horizon - 1, "weekly")
        normalized = frame.select(
            cfg.part_col,
            cfg.week_col,
            *cfg.future_feature_cols,
        ).with_columns(
            pl.col(cfg.part_col).cast(pl.String),
            pl.Series(
                cfg.week_col,
                [normalize_period_key(value, "weekly") for value in frame[cfg.week_col]],
                dtype=pl.Int64,
            ),
            *[
                pl.col(column).cast(pl.Float64)
                for column in cfg.future_feature_cols
            ],
        )
        normalized = normalized.filter(
            pl.col(cfg.part_col).is_in(active),
            pl.col(cfg.week_col).is_between(
                origin,
                final_week,
                closed="both",
            ),
        ).sort([cfg.part_col, cfg.week_col])
        if any(normalized.null_count().row(0)):
            raise ValueError("ICL inference known_future contains null values.")
        return normalized


__all__ = [
    "ExogenousICLInferenceBuilder",
    "ICLInferenceBuilderConfig",
]

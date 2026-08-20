"""Leakage-safe temporal windows for continuous exogenous weekly training."""

from __future__ import annotations

import bisect
import random
from dataclasses import dataclass
from datetime import date
from typing import Literal, Sequence

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from modeling_module.data_loader.exogenous_contracts import ExogenousFeatureSchema
from modeling_module.data_loader.future_scenario_store import (
    TrainCollateWithFutureExo,
)
from modeling_module.data_loader.indexed_temporal_data_module import (
    validate_weekly_forecast_calendar,
)
from modeling_module.data_loader.temporal import normalize_period_key


Split = Literal["train", "validation"]


@dataclass(frozen=True)
class TemporalExogenousWindowMetadata:
    """Calendar bounds and feature widths for one exogenous window."""

    part_id: str
    x_start_week: int
    x_end_week: int
    y_start_week: int
    y_end_week: int
    past_cont_dim: int
    future_cont_dim: int


@dataclass(frozen=True)
class _ExogenousSeriesBuffer:
    part_id: str
    weeks: np.ndarray
    values: np.ndarray
    past_cont: np.ndarray
    future_cont: np.ndarray
    validation_index: int


def _seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _normalize_feature_columns(
    columns: Sequence[str] | None,
    *,
    label: str,
) -> tuple[str, ...]:
    normalized = tuple(str(column).strip() for column in (columns or ()))
    if any(not column for column in normalized):
        raise ValueError(f"{label} cannot contain empty column names.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must contain ordered unique columns.")
    return normalized


class IndexedTemporalExogenousWindowDataset(Dataset):
    """Index temporal exogenous windows without materializing every sample."""

    def __init__(
        self,
        series: list[_ExogenousSeriesBuffer],
        *,
        lookback: int,
        horizon: int,
        split: Split,
        window_stride: int = 1,
    ) -> None:
        if lookback <= 0 or horizon <= 0:
            raise ValueError("lookback and horizon must be positive.")
        if window_stride <= 0:
            raise ValueError("window_stride must be positive.")
        if split not in {"train", "validation"}:
            raise ValueError(f"Unsupported split: {split!r}.")

        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.split = split
        self.window_stride = int(window_stride)

        retained: list[_ExogenousSeriesBuffer] = []
        counts: list[int] = []
        for item in series:
            if split == "train":
                max_start = (
                    item.validation_index - self.lookback - self.horizon
                )
                count = (
                    max_start // self.window_stride + 1
                    if max_start >= 0
                    else 0
                )
            else:
                count = int(
                    item.validation_index >= self.lookback
                    and len(item.values) - item.validation_index >= self.horizon
                )
            if count > 0:
                retained.append(item)
                counts.append(count)

        self._series = retained
        self._counts = np.asarray(counts, dtype=np.int64)
        self._cumulative_counts = np.cumsum(self._counts, dtype=np.int64)

    @property
    def series_count(self) -> int:
        return len(self._series)

    @property
    def series_ids(self) -> tuple[str, ...]:
        return tuple(item.part_id for item in self._series)

    def __len__(self) -> int:
        if self._cumulative_counts.size == 0:
            return 0
        return int(self._cumulative_counts[-1])

    def _resolve_index(
        self,
        index: int,
    ) -> tuple[_ExogenousSeriesBuffer, int]:
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(index)

        series_index = bisect.bisect_right(self._cumulative_counts, index)
        previous_count = (
            int(self._cumulative_counts[series_index - 1])
            if series_index > 0
            else 0
        )
        local_index = index - previous_count
        item = self._series[series_index]
        if self.split == "train":
            max_start = (
                item.validation_index - self.lookback - self.horizon
            )
            first_start = max_start % self.window_stride
            start = first_start + local_index * self.window_stride
        else:
            start = item.validation_index - self.lookback
        return item, int(start)

    def window_metadata(self, index: int) -> TemporalExogenousWindowMetadata:
        item, start = self._resolve_index(index)
        y_start = start + self.lookback
        y_end = y_start + self.horizon - 1
        return TemporalExogenousWindowMetadata(
            part_id=item.part_id,
            x_start_week=int(item.weeks[start]),
            x_end_week=int(item.weeks[y_start - 1]),
            y_start_week=int(item.weeks[y_start]),
            y_end_week=int(item.weeks[y_end]),
            past_cont_dim=int(item.past_cont.shape[-1]),
            future_cont_dim=int(item.future_cont.shape[-1]),
        )

    def __getitem__(self, index: int):
        item, start = self._resolve_index(index)
        y_start = start + self.lookback
        y_end = y_start + self.horizon

        x = torch.from_numpy(item.values[start:y_start]).unsqueeze(-1)
        y = torch.from_numpy(item.values[y_start:y_end])
        past_cont = torch.from_numpy(item.past_cont[start:y_start])
        past_cat = torch.empty((self.lookback, 0), dtype=torch.long)
        if item.future_cont.shape[-1] > 0:
            future_payload: int | torch.Tensor = torch.from_numpy(
                item.future_cont[y_start:y_end]
            )
        else:
            future_payload = int(item.weeks[y_start])
        return x, y, item.part_id, future_payload, past_cont, past_cat


class IndexedTemporalExogenousDataModule:
    """Build the same last-origin split used by endogenous qualification.

    This module intentionally supports continuous exogenous features only. The
    governed H26 models use deterministic calendar signals and no categorical
    inputs; categorical vocabulary fitting remains owned by
    ``MultiPartExoDataModule`` until a temporal categorical contract is added.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        *,
        lookback: int,
        horizon: int,
        train_end_week: int,
        forecast_origin: int,
        validation_origin: int,
        past_exo_cont_cols: Sequence[str] | None = None,
        future_exo_cont_cols: Sequence[str] | None = None,
        window_stride: int = 1,
        seed: int = 42,
        part_col: str = "oper_part_no",
        date_col: str = "demand_dt",
        qty_col: str = "demand_qty",
        require_all_series_eligible: bool = True,
    ) -> None:
        if lookback <= 0 or horizon <= 0:
            raise ValueError("lookback and horizon must be positive.")
        if window_stride <= 0:
            raise ValueError("window_stride must be positive.")
        validate_weekly_forecast_calendar(
            train_end_week=train_end_week,
            forecast_origin=forecast_origin,
            validation_origin=validation_origin,
            horizon=horizon,
        )

        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.train_end_week = int(train_end_week)
        self.forecast_origin = int(forecast_origin)
        self.validation_origin = int(validation_origin)
        self.window_stride = int(window_stride)
        self.seed = int(seed)
        self.part_col = part_col
        self.date_col = date_col
        self.qty_col = qty_col
        self.require_all_series_eligible = bool(require_all_series_eligible)
        self.past_exo_cont_cols = _normalize_feature_columns(
            past_exo_cont_cols,
            label="past_exo_cont_cols",
        )
        self.future_exo_cont_cols = _normalize_feature_columns(
            future_exo_cont_cols,
            label="future_exo_cont_cols",
        )
        if not self.past_exo_cont_cols and not self.future_exo_cont_cols:
            raise ValueError(
                "At least one past or future continuous exogenous feature is required."
            )

        self.exogenous_schema = ExogenousFeatureSchema.from_columns(
            past_cont=self.past_exo_cont_cols,
            future_cont=self.future_exo_cont_cols,
        )
        self.df = self._validate_and_normalize_frame(df)
        self._ineligible_series_reasons: tuple[str, ...] = ()
        self._series = self._build_series_buffers()
        self.train_dataset: IndexedTemporalExogenousWindowDataset | None = None
        self.val_dataset: IndexedTemporalExogenousWindowDataset | None = None

    def _validate_and_normalize_frame(self, df: pl.DataFrame) -> pl.DataFrame:
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"df must be a polars DataFrame, got {type(df)!r}.")
        required = list(
            dict.fromkeys(
                (
                    self.part_col,
                    self.date_col,
                    self.qty_col,
                    *self.past_exo_cont_cols,
                    *self.future_exo_cont_cols,
                )
            )
        )
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(
                f"exogenous data is missing required columns: {missing}."
            )
        if df.is_empty():
            raise ValueError("exogenous data must contain at least one row.")

        feature_columns = list(
            dict.fromkeys(
                (*self.past_exo_cont_cols, *self.future_exo_cont_cols)
            )
        )
        normalized = (
            df.select(required)
            .with_columns(
                pl.col(self.part_col).cast(pl.String),
                pl.col(self.date_col).cast(pl.Int64),
                pl.col(self.qty_col).cast(pl.Float32),
                *[
                    pl.col(column).cast(pl.Float32)
                    for column in feature_columns
                ],
            )
            .sort([self.part_col, self.date_col])
        )
        null_counts = normalized.select(
            [pl.col(column).null_count().alias(column) for column in required]
        ).row(0, named=True)
        populated_nulls = {
            name: int(count) for name, count in null_counts.items() if count
        }
        if populated_nulls:
            raise ValueError(
                f"exogenous data contains nulls: {populated_nulls}."
            )

        duplicate_exists = normalized.select(
            pl.struct([self.part_col, self.date_col]).is_duplicated().any()
        ).item()
        if duplicate_exists:
            raise ValueError(
                "exogenous data contains duplicate "
                f"({self.part_col}, {self.date_col}) keys."
            )

        target_values = normalized[self.qty_col].to_numpy()
        if not np.isfinite(target_values).all():
            raise ValueError(f"{self.qty_col} must contain only finite values.")
        if np.any(target_values < 0):
            raise ValueError(f"{self.qty_col} must be non-negative.")
        for column in feature_columns:
            if not np.isfinite(normalized[column].to_numpy()).all():
                raise ValueError(f"{column} must contain only finite values.")

        source_max_week = int(normalized[self.date_col].max())
        if source_max_week != self.train_end_week:
            raise ValueError(
                "exogenous data upper bound does not match train_end_week: "
                f"source_max={source_max_week}, "
                f"train_end_week={self.train_end_week}."
            )
        return normalized

    def _build_series_buffers(self) -> list[_ExogenousSeriesBuffer]:
        unique_weeks = self.df[self.date_col].unique().sort().to_list()
        ordinal_by_week: dict[int, int] = {}
        for raw_week in unique_weeks:
            week = normalize_period_key(int(raw_week), "weekly")
            ordinal_by_week[week] = date.fromisocalendar(
                week // 100,
                week % 100,
                1,
            ).toordinal()

        eligible: list[_ExogenousSeriesBuffer] = []
        ineligible: list[str] = []
        for group in self.df.partition_by(self.part_col, maintain_order=True):
            part_id = str(group[self.part_col][0])
            weeks = group[self.date_col].to_numpy().astype(
                np.int64,
                copy=False,
            )
            ordinals = np.fromiter(
                (ordinal_by_week[int(week)] for week in weeks),
                dtype=np.int64,
                count=len(weeks),
            )
            gaps = np.flatnonzero(np.diff(ordinals) != 7)
            if gaps.size:
                gap_index = int(gaps[0])
                raise ValueError(
                    f"series {part_id!r} is not continuous weekly data between "
                    f"{int(weeks[gap_index])} and "
                    f"{int(weeks[gap_index + 1])}."
                )
            if int(weeks[-1]) != self.train_end_week:
                ineligible.append(f"{part_id}:series_end={int(weeks[-1])}")
                continue

            validation_matches = np.flatnonzero(
                weeks == self.validation_origin
            )
            if validation_matches.size != 1:
                ineligible.append(
                    f"{part_id}:validation_origin_count="
                    f"{validation_matches.size}"
                )
                continue
            validation_index = int(validation_matches[0])
            has_train_window = (
                validation_index - self.lookback - self.horizon >= 0
            )
            has_validation_window = (
                validation_index >= self.lookback
                and len(weeks) - validation_index >= self.horizon
            )
            if not has_train_window or not has_validation_window:
                ineligible.append(
                    f"{part_id}:rows={len(weeks)},"
                    f"validation_index={validation_index}"
                )
                continue

            values = np.array(
                group[self.qty_col].to_numpy(),
                dtype=np.float32,
                copy=True,
                order="C",
            )
            past_cont = self._feature_matrix(
                group,
                self.past_exo_cont_cols,
            )
            if self.future_exo_cont_cols == self.past_exo_cont_cols:
                future_cont = past_cont
            else:
                future_cont = self._feature_matrix(
                    group,
                    self.future_exo_cont_cols,
                )
            eligible.append(
                _ExogenousSeriesBuffer(
                    part_id=part_id,
                    weeks=np.array(
                        weeks,
                        dtype=np.int64,
                        copy=True,
                        order="C",
                    ),
                    values=values,
                    past_cont=past_cont,
                    future_cont=future_cont,
                    validation_index=validation_index,
                )
            )

        self._ineligible_series_reasons = tuple(ineligible)
        if ineligible and self.require_all_series_eligible:
            preview = ", ".join(ineligible[:5])
            raise ValueError(
                f"{len(ineligible)} series are not eligible for the temporal "
                f"exogenous split; examples: {preview}."
            )
        if not eligible:
            raise ValueError(
                "No series are eligible for temporal exogenous training."
            )
        return eligible

    @property
    def ineligible_series_reasons(self) -> tuple[str, ...]:
        """Explain source series excluded before window construction."""

        return self._ineligible_series_reasons

    @staticmethod
    def _feature_matrix(
        group: pl.DataFrame,
        columns: Sequence[str],
    ) -> np.ndarray:
        if not columns:
            return np.empty((group.height, 0), dtype=np.float32)
        return np.array(
            group.select(columns).to_numpy(),
            dtype=np.float32,
            copy=True,
            order="C",
        )

    def setup(self) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return
        self.train_dataset = IndexedTemporalExogenousWindowDataset(
            self._series,
            lookback=self.lookback,
            horizon=self.horizon,
            split="train",
            window_stride=self.window_stride,
        )
        self.val_dataset = IndexedTemporalExogenousWindowDataset(
            self._series,
            lookback=self.lookback,
            horizon=self.horizon,
            split="validation",
            window_stride=1,
        )
        if len(self.train_dataset) == 0:
            raise ValueError(
                "Temporal exogenous split produced no training windows."
            )
        if len(self.val_dataset) == 0:
            raise ValueError(
                "Temporal exogenous split produced no validation windows."
            )

    @property
    def summary(self) -> dict[str, int]:
        self.setup()
        assert self.train_dataset is not None
        assert self.val_dataset is not None
        train_target_max_week = max(
            self.train_dataset.window_metadata(index).y_end_week
            for index in (
                int(value) - 1
                for value in self.train_dataset._cumulative_counts
            )
        )
        return {
            "row_count": self.df.height,
            "source_series_count": self.df[self.part_col].n_unique(),
            "series_count": len(self._series),
            "excluded_series_count": len(self._ineligible_series_reasons),
            "source_min_week": int(self.df[self.date_col].min()),
            "source_max_week": int(self.df[self.date_col].max()),
            "train_windows": len(self.train_dataset),
            "train_target_max_week": train_target_max_week,
            "validation_windows": len(self.val_dataset),
            "validation_target_min_week": self.validation_origin,
            "validation_target_max_week": self.train_end_week,
            "past_cont_dim": len(self.past_exo_cont_cols),
            "future_cont_dim": len(self.future_exo_cont_cols),
        }

    @staticmethod
    def _loader_options(
        *,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        prefetch_factor: int,
    ) -> dict:
        workers = max(0, int(num_workers))
        options: dict = {
            "num_workers": workers,
            "pin_memory": bool(pin_memory),
        }
        if workers > 0:
            options["persistent_workers"] = bool(persistent_workers)
            options["prefetch_factor"] = max(1, int(prefetch_factor))
            options["worker_init_fn"] = _seed_worker
        return options

    def _attach_contract(self, loader: DataLoader) -> DataLoader:
        loader.exogenous_schema = self.exogenous_schema
        return loader

    def get_train_loader(
        self,
        *,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        drop_last: bool = True,
    ) -> DataLoader:
        self.setup()
        assert self.train_dataset is not None
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            self.train_dataset,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            drop_last=bool(drop_last),
            generator=generator,
            collate_fn=TrainCollateWithFutureExo(horizon=self.horizon),
            **self._loader_options(
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            ),
        )
        return self._attach_contract(loader)

    def get_val_loader(
        self,
        *,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        drop_last: bool = False,
    ) -> DataLoader:
        self.setup()
        assert self.val_dataset is not None
        loader = DataLoader(
            self.val_dataset,
            batch_size=int(batch_size),
            shuffle=False,
            drop_last=bool(drop_last),
            collate_fn=TrainCollateWithFutureExo(horizon=self.horizon),
            **self._loader_options(
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            ),
        )
        return self._attach_contract(loader)


__all__ = [
    "IndexedTemporalExogenousDataModule",
    "IndexedTemporalExogenousWindowDataset",
    "TemporalExogenousWindowMetadata",
]

from __future__ import annotations

import polars as pl
import pytest
import torch

from modeling_module.data_loader.indexed_temporal_data_module import (
    IndexedTemporalDataModule,
    validate_weekly_forecast_calendar,
)
from modeling_module.data_loader.temporal import add_period


def _weekly_frame(*, parts: tuple[str, ...] = ("A", "B")) -> pl.DataFrame:
    start_week = 202444
    weeks = [add_period(start_week, offset, "weekly") for offset in range(11)]
    rows = []
    for part_index, part_id in enumerate(parts):
        for value_index, week in enumerate(weeks):
            rows.append(
                {
                    "oper_part_no": part_id,
                    "demand_dt": week,
                    "demand_qty": float(part_index * 100 + value_index),
                }
            )
    return pl.DataFrame(rows)


def _datamodule(df: pl.DataFrame | None = None, **overrides):
    kwargs = {
        "lookback": 4,
        "horizon": 3,
        "train_end_week": 202502,
        "forecast_origin": 202503,
        "validation_origin": 202452,
        "window_stride": 1,
        "seed": 17,
    }
    kwargs.update(overrides)
    return IndexedTemporalDataModule(
        _weekly_frame() if df is None else df,
        **kwargs,
    )


def test_temporal_split_uses_last_origin_without_future_leakage():
    datamodule = _datamodule()
    datamodule.setup()
    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset
    assert train_dataset is not None
    assert val_dataset is not None

    assert len(train_dataset) == 4
    assert len(val_dataset) == 2
    assert not hasattr(train_dataset, "samples")
    assert train_dataset._series[0].values is val_dataset._series[0].values

    last_train = train_dataset.window_metadata(len(train_dataset) - 1)
    validation = val_dataset.window_metadata(0)
    assert last_train.y_end_week == 202451
    assert validation.x_start_week == 202448
    assert validation.x_end_week == 202451
    assert validation.y_start_week == 202452
    assert validation.y_end_week == 202502

    x, y, part_id = val_dataset[0]
    assert x.shape == (4, 1)
    assert y.shape == (3,)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert part_id == "A"


def test_window_stride_is_arithmetic_and_loader_order_is_reproducible():
    first = _datamodule(window_stride=2)
    second = _datamodule(window_stride=2)
    assert first.summary["train_windows"] == 2
    assert second.summary["validation_windows"] == 2
    assert first.train_dataset is not None
    assert first.train_dataset.window_metadata(0).y_end_week == 202451

    first_batch = next(
        iter(first.get_train_loader(batch_size=2, shuffle=True, drop_last=False))
    )
    second_batch = next(
        iter(second.get_train_loader(batch_size=2, shuffle=True, drop_last=False))
    )
    assert torch.equal(first_batch[0], second_batch[0])
    assert torch.equal(first_batch[1], second_batch[1])
    assert first_batch[2] == second_batch[2]


def test_gap_duplicate_and_cutoff_mismatch_fail_fast():
    gap_df = _weekly_frame().filter(
        ~(
            (pl.col("oper_part_no") == "A")
            & (pl.col("demand_dt") == 202449)
        )
    )
    with pytest.raises(ValueError, match="not continuous weekly data"):
        _datamodule(gap_df)

    duplicate_df = pl.concat([_weekly_frame(), _weekly_frame().head(1)])
    with pytest.raises(ValueError, match="duplicate"):
        _datamodule(duplicate_df)

    with pytest.raises(ValueError, match="upper bound"):
        _datamodule(
            train_end_week=202501,
            forecast_origin=202502,
            validation_origin=202451,
        )


def test_forecast_calendar_handles_iso_year_boundary_and_rejects_drift():
    validate_weekly_forecast_calendar(
        train_end_week=202502,
        forecast_origin=202503,
        validation_origin=202452,
        horizon=3,
    )
    with pytest.raises(ValueError, match="immediately after"):
        validate_weekly_forecast_calendar(
            train_end_week=202502,
            forecast_origin=202504,
            validation_origin=202452,
            horizon=3,
        )
    with pytest.raises(ValueError, match="last-origin holdout"):
        validate_weekly_forecast_calendar(
            train_end_week=202502,
            forecast_origin=202503,
            validation_origin=202451,
            horizon=3,
        )

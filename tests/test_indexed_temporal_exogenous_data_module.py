from __future__ import annotations

import math

import polars as pl
import pytest
import torch

from modeling_module.data_loader.indexed_temporal_exogenous_data_module import (
    IndexedTemporalExogenousDataModule,
)
from modeling_module.data_loader.temporal import add_period


def _weekly_frame(*, parts: tuple[str, ...] = ("A", "B")) -> pl.DataFrame:
    weeks = [add_period(202444, offset, "weekly") for offset in range(11)]
    rows = []
    for part_index, part_id in enumerate(parts):
        for value_index, week in enumerate(weeks):
            rows.append(
                {
                    "oper_part_no": part_id,
                    "demand_dt": week,
                    "demand_qty": float(part_index * 100 + value_index),
                    "past_a": float(value_index),
                    "past_b": float(part_index),
                    "future_a": float(value_index) / 10.0,
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
        "past_exo_cont_cols": ("past_a", "past_b"),
        "future_exo_cont_cols": ("future_a",),
        "window_stride": 1,
        "seed": 17,
    }
    kwargs.update(overrides)
    return IndexedTemporalExogenousDataModule(
        _weekly_frame() if df is None else df,
        **kwargs,
    )


def test_exogenous_temporal_split_matches_endogenous_last_origin_contract():
    datamodule = _datamodule()
    datamodule.setup()
    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset
    assert train_dataset is not None
    assert val_dataset is not None

    assert len(train_dataset) == 4
    assert len(val_dataset) == 2
    assert datamodule.summary == {
        "row_count": 22,
        "source_series_count": 2,
        "series_count": 2,
        "excluded_series_count": 0,
        "source_min_week": 202444,
        "source_max_week": 202502,
        "train_windows": 4,
        "train_target_max_week": 202451,
        "validation_windows": 2,
        "validation_target_min_week": 202452,
        "validation_target_max_week": 202502,
        "past_cont_dim": 2,
        "future_cont_dim": 1,
    }

    last_train = train_dataset.window_metadata(len(train_dataset) - 1)
    validation = val_dataset.window_metadata(0)
    assert last_train.y_end_week == 202451
    assert validation.x_start_week == 202448
    assert validation.x_end_week == 202451
    assert validation.y_start_week == 202452
    assert validation.y_end_week == 202502

    batch = next(iter(datamodule.get_val_loader(batch_size=2)))
    x, y, part_ids, future_cont, past_cont, past_cat = batch
    assert x.shape == (2, 4, 1)
    assert y.shape == (2, 3)
    assert part_ids == ["A", "B"]
    assert future_cont.shape == (2, 3, 1)
    assert past_cont.shape == (2, 4, 2)
    assert past_cat.shape == (2, 4, 0)
    assert future_cont.dtype == torch.float32
    assert past_cont.dtype == torch.float32


def test_past_only_loader_emits_empty_future_tensor_for_timexer():
    datamodule = _datamodule(future_exo_cont_cols=())
    loader = datamodule.get_val_loader(batch_size=2)
    batch = next(iter(loader))

    assert batch[3].shape == (2, 3, 0)
    assert batch[4].shape == (2, 4, 2)
    assert loader.exogenous_schema.future_cont_names == ()


def test_production_refit_uses_all_exogenous_targets_without_validation():
    datamodule = _datamodule(training_mode="production_refit")
    summary = datamodule.summary

    assert summary["train_windows"] == 10
    assert summary["train_target_max_week"] == 202502
    assert summary["validation_windows"] == 0
    assert summary["validation_target_min_week"] is None
    assert summary["validation_target_max_week"] is None
    assert datamodule.val_dataset is None

    batch = next(iter(datamodule.get_train_loader(batch_size=2)))
    assert batch[0].shape == (2, 4, 1)
    assert batch[1].shape == (2, 3)
    assert batch[3].shape == (2, 3, 1)
    assert batch[4].shape == (2, 4, 2)
    with pytest.raises(RuntimeError, match="has no validation loader"):
        datamodule.get_val_loader(batch_size=2)


def test_window_stride_keeps_latest_eligible_training_origin_and_is_seeded():
    first = _datamodule(window_stride=2)
    second = _datamodule(window_stride=2)
    assert first.summary["train_windows"] == 2
    assert first.train_dataset is not None
    assert first.train_dataset.window_metadata(0).y_end_week == 202451

    first_batch = next(
        iter(first.get_train_loader(batch_size=2, shuffle=True, drop_last=False))
    )
    second_batch = next(
        iter(second.get_train_loader(batch_size=2, shuffle=True, drop_last=False))
    )
    for index in (0, 1, 3, 4, 5):
        assert torch.equal(first_batch[index], second_batch[index])
    assert first_batch[2] == second_batch[2]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda frame: frame.filter(
                ~(
                    (pl.col("oper_part_no") == "A")
                    & (pl.col("demand_dt") == 202449)
                )
            ),
            "not continuous weekly data",
        ),
        (
            lambda frame: pl.concat([frame, frame.head(1)]),
            "duplicate",
        ),
        (
            lambda frame: frame.with_columns(
                pl.when(pl.col("oper_part_no") == "A")
                .then(pl.lit(math.inf))
                .otherwise(pl.col("past_a"))
                .alias("past_a")
            ),
            "finite",
        ),
        (
            lambda frame: frame.drop("future_a"),
            "missing required columns",
        ),
    ],
)
def test_invalid_exogenous_source_fails_fast(mutator, message):
    with pytest.raises(ValueError, match=message):
        _datamodule(mutator(_weekly_frame()))


def test_exogenous_source_cutoff_must_match_training_contract():
    with pytest.raises(ValueError, match="upper bound"):
        _datamodule(
            train_end_week=202501,
            forecast_origin=202502,
            validation_origin=202451,
        )


def test_too_short_series_can_be_explicitly_excluded_without_padding():
    frame = _weekly_frame().filter(
        ~(
            (pl.col("oper_part_no") == "A")
            & pl.col("demand_dt").is_in([202444, 202445])
        )
    )
    datamodule = _datamodule(
        frame,
        require_all_series_eligible=False,
    )

    assert datamodule.summary["source_series_count"] == 2
    assert datamodule.summary["series_count"] == 1
    assert datamodule.summary["excluded_series_count"] == 1
    assert datamodule.ineligible_series_reasons == (
        "A:rows=9,validation_index=6",
    )
    assert datamodule.val_dataset is not None
    assert datamodule.val_dataset.series_ids == ("B",)


def test_production_refit_recovers_series_that_only_lack_holdout_history():
    frame = _weekly_frame().filter(
        ~(
            (pl.col("oper_part_no") == "A")
            & pl.col("demand_dt").is_in([202444, 202445])
        )
    )
    datamodule = _datamodule(
        frame,
        require_all_series_eligible=False,
        training_mode="production_refit",
    )

    assert datamodule.summary["source_series_count"] == 2
    assert datamodule.summary["series_count"] == 2
    assert datamodule.summary["excluded_series_count"] == 0

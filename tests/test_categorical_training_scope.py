from __future__ import annotations

import polars as pl
import pytest
import torch
from torch.utils.data import Subset

from modeling_module import ExogenousFeatureSchema
from modeling_module.data_loader.multi_part_exo_data_module import (
    MultiPartExoDataModule,
)


def _window_frame(length: int = 12) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["A"] * length,
            "date": [20240101 + index for index in range(length)],
            "y": [float(index + 1) for index in range(length)],
            "category": [f"row-{index}" for index in range(length)],
        }
    )


def _multi_frame(length: int = 6) -> pl.DataFrame:
    rows = []
    for uid in ("A", "B", "C", "D"):
        for index in range(length):
            rows.append(
                {
                    "unique_id": uid,
                    "date": 20240101 + index,
                    "y": float(index + 1) if index < length - 1 else None,
                    "category": f"{uid}-{'tail' if index == length - 1 else index}",
                }
            )
    return pl.DataFrame(rows)


def test_window_scope_uses_only_rows_referenced_by_training_windows():
    frame = _window_frame()
    datamodule = MultiPartExoDataModule(
        frame,
        lookback=2,
        horizon=2,
        freq="daily",
        y_col="y",
        val_ratio=0.8,
        seed=42,
        split_mode="window",
    )
    datamodule.setup()

    assert datamodule.resolved_split_mode == "window"
    assert isinstance(datamodule.train_dataset, Subset)
    assert isinstance(datamodule.val_dataset, Subset)
    full_dataset = datamodule._full_dataset
    assert full_dataset is not None

    train_positions = full_dataset.source_row_positions_for_windows(
        datamodule.train_dataset.indices
    )["A"]
    validation_positions = full_dataset.source_row_positions_for_windows(
        datamodule.val_dataset.indices
    )["A"]
    scope = datamodule.categorical_training_frame(["category"])

    assert scope["category"].to_list() == [
        f"row-{position}"
        for position in train_positions
    ]
    validation_only = set(validation_positions).difference(train_positions)
    assert validation_only
    assert {
        f"row-{position}"
        for position in validation_only
    }.isdisjoint(scope["category"].to_list())

    schema = ExogenousFeatureSchema.from_columns(
        past_cat=["category"],
        future_cat=["category"],
    )
    artifact = datamodule.fit_categorical_vocabulary(schema)
    vocabulary = artifact.vocabulary_for("category")
    assert set(vocabulary.known_values) == set(scope["category"].to_list())
    assert all(
        vocabulary.id_of(f"row-{position}") == 0
        for position in validation_only
    )

    for window_index in datamodule.train_dataset.indices:
        _, start = full_dataset.index_map[int(window_index)]
        assert start + datamodule.lookback in train_positions
        assert start + datamodule.lookback + datamodule.horizon - 1 in train_positions


def test_multi_scope_uses_valid_training_window_rows_from_training_series_only():
    frame = _multi_frame()
    datamodule = MultiPartExoDataModule(
        frame,
        lookback=2,
        horizon=1,
        freq="daily",
        y_col="y",
        val_ratio=0.25,
        seed=7,
        split_mode="multi",
    )
    datamodule.setup()

    assert datamodule.resolved_split_mode == "multi"
    assert isinstance(datamodule.train_dataset, Subset)
    assert isinstance(datamodule.val_dataset, Subset)
    full_dataset = datamodule._full_dataset
    assert full_dataset is not None

    train_ids = {
        full_dataset.index_map[int(index)][0]
        for index in datamodule.train_dataset.indices
    }
    validation_ids = {
        full_dataset.index_map[int(index)][0]
        for index in datamodule.val_dataset.indices
    }
    scope = datamodule.categorical_training_frame(["category"])

    assert set(scope["unique_id"].to_list()) == train_ids
    assert train_ids.isdisjoint(validation_ids)
    assert scope.height == len(train_ids) * 5
    assert {
        f"{uid}-tail"
        for uid in train_ids
    }.isdisjoint(scope["category"].to_list())
    assert {
        f"{uid}-tail"
        for uid in validation_ids
    }.isdisjoint(scope["category"].to_list())

    schema = ExogenousFeatureSchema.from_columns(future_cat=["category"])
    artifact = datamodule.fit_categorical_vocabulary(schema)
    vocabulary = artifact.vocabulary_for("category")
    assert all(
        vocabulary.id_of(f"{uid}-tail") == 0
        for uid in train_ids.union(validation_ids)
    )


def test_multi_split_with_one_series_records_window_fallback_scope():
    frame = _window_frame()
    datamodule = MultiPartExoDataModule(
        frame,
        lookback=2,
        horizon=2,
        freq="daily",
        y_col="y",
        val_ratio=0.8,
        seed=42,
        split_mode="multi",
    )
    datamodule.setup()
    scope = datamodule.categorical_training_frame(["category"])

    assert datamodule.resolved_split_mode == "window"
    assert 0 < scope.height < frame.height


def test_categorical_training_scope_rejects_unknown_or_duplicate_columns():
    datamodule = MultiPartExoDataModule(
        _window_frame(),
        lookback=2,
        horizon=2,
        freq="daily",
        y_col="y",
    )

    with pytest.raises(ValueError, match="missing from the dataframe"):
        datamodule.categorical_training_frame(["missing"])

    with pytest.raises(ValueError, match="duplicates"):
        datamodule.categorical_training_frame(["category", "category"])


def test_setup_fits_and_shares_categorical_vocabulary_after_window_split():
    frame = _window_frame()
    datamodule = MultiPartExoDataModule(
        frame,
        lookback=2,
        horizon=2,
        freq="daily",
        y_col="y",
        past_exo_cat_cols=["category"],
        val_ratio=0.8,
        seed=42,
        split_mode="window",
    )

    assert datamodule.categorical_vocabulary_artifact is None
    assert datamodule.cat_indexers == {}
    datamodule.setup()

    artifact = datamodule.categorical_vocabulary_artifact
    assert artifact is not None
    assert datamodule.categorical_vocabulary_fingerprint == artifact.fingerprint
    assert isinstance(datamodule.train_dataset, Subset)
    assert isinstance(datamodule.val_dataset, Subset)
    assert datamodule.train_dataset.dataset is datamodule._full_dataset
    assert datamodule.val_dataset.dataset is datamodule._full_dataset
    assert datamodule._full_dataset.cat_indexers["category"] is (
        artifact.vocabulary_for("category")
    )

    train_positions = datamodule._full_dataset.source_row_positions_for_windows(
        datamodule.train_dataset.indices
    )["A"]
    validation_positions = datamodule._full_dataset.source_row_positions_for_windows(
        datamodule.val_dataset.indices
    )["A"]
    validation_only = set(validation_positions).difference(train_positions)
    assert validation_only

    vocabulary = artifact.vocabulary_for("category")
    assert set(vocabulary.known_values) == {
        frame["category"][position]
        for position in train_positions
    }
    assert all(
        datamodule.df["category"][position] == 0
        for position in validation_only
    )

    first_train_sample = datamodule.train_dataset[0]
    assert first_train_sample[5].dtype == torch.long
    assert first_train_sample[5].shape == (datamodule.lookback, 1)

    first_fingerprint = datamodule.categorical_vocabulary_fingerprint
    first_train_indices = tuple(datamodule.train_dataset.indices)
    datamodule.setup()
    assert datamodule.categorical_vocabulary_fingerprint == first_fingerprint
    assert tuple(datamodule.train_dataset.indices) == first_train_indices


def test_setup_fits_multi_split_vocabulary_from_training_series_only():
    frame = _multi_frame()
    datamodule = MultiPartExoDataModule(
        frame,
        lookback=2,
        horizon=1,
        freq="daily",
        y_col="y",
        past_exo_cat_cols=["category"],
        val_ratio=0.25,
        seed=7,
        split_mode="multi",
    )
    datamodule.setup()

    assert datamodule.resolved_split_mode == "multi"
    full_dataset = datamodule._full_dataset
    train_ids = {
        full_dataset.index_map[int(index)][0]
        for index in datamodule.train_dataset.indices
    }
    validation_ids = {
        full_dataset.index_map[int(index)][0]
        for index in datamodule.val_dataset.indices
    }
    vocabulary = datamodule.categorical_vocabulary_artifact.vocabulary_for(
        "category"
    )

    assert train_ids.isdisjoint(validation_ids)
    assert all(
        vocabulary.id_of(f"{uid}-{position}") > 0
        for uid in train_ids
        for position in range(5)
    )
    assert all(
        vocabulary.id_of(f"{uid}-{position}") == 0
        for uid in validation_ids
        for position in range(5)
    )
    assert all(
        vocabulary.id_of(f"{uid}-tail") == 0
        for uid in train_ids.union(validation_ids)
    )


def test_future_category_dataset_uses_horizon_slice_and_shared_vocabulary():
    frame = _window_frame()
    datamodule = MultiPartExoDataModule(
        frame,
        lookback=2,
        horizon=2,
        freq="daily",
        y_col="y",
        past_exo_cat_cols=["category"],
        future_exo_cat_cols=["category"],
        val_ratio=0.8,
        seed=42,
        split_mode="window",
    )
    datamodule.setup()

    artifact = datamodule.categorical_vocabulary_artifact
    assert artifact.feature_names == ("category",)
    assert datamodule.exogenous_schema.past_cat_cardinalities == (
        artifact.vocabulary_for("category").cardinality,
    )
    assert datamodule.exogenous_schema.future_cat_cardinalities == (
        artifact.vocabulary_for("category").cardinality,
    )

    full_dataset = datamodule._full_dataset
    train_index = int(datamodule.train_dataset.indices[0])
    _, start = full_dataset.index_map[train_index]
    sample = full_dataset[train_index]
    past_cat = sample[5]
    future_cat = sample[6]

    assert len(sample) == 7
    assert past_cat.dtype == torch.long
    assert future_cat.dtype == torch.long
    assert past_cat.shape == (datamodule.lookback, 1)
    assert future_cat.shape == (datamodule.horizon, 1)
    assert past_cat[:, 0].tolist() == datamodule.df["category"][
        start:start + datamodule.lookback
    ].to_list()
    assert future_cat[:, 0].tolist() == datamodule.df["category"][
        start + datamodule.lookback:
        start + datamodule.lookback + datamodule.horizon
    ].to_list()

    train_future_ids = torch.cat(
        [datamodule.train_dataset[index][6].flatten() for index in range(len(datamodule.train_dataset))]
    )
    val_future_ids = torch.cat(
        [datamodule.val_dataset[index][6].flatten() for index in range(len(datamodule.val_dataset))]
    )
    assert bool(torch.all(train_future_ids > 0))
    assert bool(torch.any(val_future_ids == 0))


def test_future_category_multi_split_maps_validation_series_to_unk():
    datamodule = MultiPartExoDataModule(
        _multi_frame(),
        lookback=2,
        horizon=1,
        freq="daily",
        y_col="y",
        future_exo_cat_cols=["category"],
        val_ratio=0.25,
        seed=7,
        split_mode="multi",
    )
    datamodule.setup()

    train_future_ids = torch.cat(
        [datamodule.train_dataset[index][6].flatten() for index in range(len(datamodule.train_dataset))]
    )
    val_future_ids = torch.cat(
        [datamodule.val_dataset[index][6].flatten() for index in range(len(datamodule.val_dataset))]
    )
    assert bool(torch.all(train_future_ids > 0))
    assert bool(torch.all(val_future_ids == 0))


def test_legacy_build_cat_indexer_from_uses_training_scope_and_keeps_raw_column():
    frame = _window_frame()
    datamodule = MultiPartExoDataModule(
        frame,
        lookback=2,
        horizon=2,
        freq="daily",
        y_col="y",
        build_cat_indexer_from=["category"],
        val_ratio=0.8,
        seed=42,
        split_mode="window",
    )

    assert datamodule.past_exo_cat_cols == ["category_id"]
    assert "category_id" not in datamodule.df.columns
    assert datamodule.cat_indexers == {}
    datamodule.setup()

    artifact = datamodule.categorical_vocabulary_artifact
    assert artifact is not None
    assert artifact.feature_names == ("category_id",)
    assert datamodule.df["category"].to_list() == frame["category"].to_list()
    assert datamodule.df["category_id"].dtype == pl.Int64
    assert datamodule.cat_indexers["category"] is (
        artifact.vocabulary_for("category_id")
    )
    assert datamodule.cat_indexers["category_id"] is (
        artifact.vocabulary_for("category_id")
    )

    train_positions = datamodule._full_dataset.source_row_positions_for_windows(
        datamodule.train_dataset.indices
    )["A"]
    validation_positions = datamodule._full_dataset.source_row_positions_for_windows(
        datamodule.val_dataset.indices
    )["A"]
    validation_only = set(validation_positions).difference(train_positions)
    assert validation_only
    assert all(
        datamodule.df["category_id"][position] == 0
        for position in validation_only
    )


def test_legacy_custom_target_column_is_only_a_feature_name_mapping():
    datamodule = MultiPartExoDataModule(
        _window_frame(),
        lookback=2,
        horizon=2,
        freq="daily",
        y_col="y",
        build_cat_indexer_from=["category"],
        cat_indexer_target_col="category_code",
        val_ratio=0.8,
        seed=42,
    )
    datamodule.setup()

    assert datamodule.past_exo_cat_cols == ["category_code"]
    assert datamodule.categorical_vocabulary_artifact.feature_names == (
        "category_code",
    )
    assert datamodule.df["category_code"].dtype == pl.Int64

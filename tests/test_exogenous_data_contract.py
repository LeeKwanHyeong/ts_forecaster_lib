from __future__ import annotations

import polars as pl
import pytest
import torch

from modeling_module import (
    ExogenousBatch,
    ExogenousFeatureSchema,
    build_exogenous_schema,
)
from modeling_module.api.data import build_datamodule


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 6,
            "date": [20240101 + index for index in range(6)],
            "y": [float(index) for index in range(6)],
            "price": [1.0, 1.0, 1.1, 1.1, 1.2, 1.2],
            "segment": [0, 0, 1, 1, 2, 2],
            "promo": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )


def test_exogenous_feature_schema_preserves_order_and_has_stable_fingerprint():
    schema = ExogenousFeatureSchema.from_columns(
        past_cont=["price", "promo"],
        past_cat=["segment"],
        future_cont=["promo"],
        past_cat_cardinalities=[3],
    )
    same = ExogenousFeatureSchema.from_columns(
        past_cont=["price", "promo"],
        past_cat=["segment"],
        future_cont=["promo"],
        past_cat_cardinalities=[3],
    )
    reordered = ExogenousFeatureSchema.from_columns(
        past_cont=["promo", "price"],
        past_cat=["segment"],
        future_cont=["promo"],
        past_cat_cardinalities=[3],
    )

    assert schema.past_cont_names == ("price", "promo")
    assert schema.fingerprint == same.fingerprint
    assert schema.fingerprint != reordered.fingerprint


def test_exogenous_feature_schema_rejects_ambiguous_feature_identity():
    with pytest.raises(ValueError, match="duplicate feature names"):
        ExogenousFeatureSchema.from_columns(past_cont=["price", "price"])

    with pytest.raises(ValueError, match="categorical and continuous"):
        ExogenousFeatureSchema.from_columns(
            past_cont=["segment"],
            past_cat=["segment"],
        )

    with pytest.raises(ValueError, match="must be empty or match"):
        ExogenousFeatureSchema.from_columns(
            past_cat=["segment"],
            past_cat_cardinalities=[3, 4],
        )


def test_exogenous_batch_validates_shape_dtype_and_exact_schema():
    schema = ExogenousFeatureSchema.from_columns(
        past_cont=["price", "promo"],
        past_cat=["segment"],
        future_cont=["promo"],
    )
    batch = ExogenousBatch(
        past_cont=torch.randn(2, 4, 2),
        past_cat=torch.randint(0, 3, (2, 4, 1)),
        future_cont=torch.randn(2, 2, 1),
    )

    assert batch.validate(batch_size=2, lookback=4, horizon=2, schema=schema) is batch
    assert batch.provided_inputs() == frozenset({"past_cont", "past_cat", "future_cont"})

    with pytest.raises(ValueError, match="past_cont time-axis mismatch"):
        batch.validate(lookback=3)

    with pytest.raises(TypeError, match="past_cat must use an integer dtype"):
        ExogenousBatch(past_cat=torch.randn(2, 4, 1)).validate()

    with pytest.raises(ValueError, match="future_cont is required by the exogenous schema"):
        ExogenousBatch(
            past_cont=batch.past_cont,
            past_cat=batch.past_cat,
        ).validate(schema=schema)

    endogenous_schema = ExogenousFeatureSchema()
    with pytest.raises(ValueError, match="past_cont is not declared"):
        ExogenousBatch(past_cont=torch.randn(2, 4, 1)).validate(
            schema=endogenous_schema
        )


def test_exogenous_batch_normalizes_legacy_future_shape_and_empty_tensors():
    batch = ExogenousBatch.from_legacy(
        past_exo_cont=torch.empty(2, 4, 0),
        future_exo=torch.tensor([[1.0], [2.0]]),
        batch_size=2,
    )

    assert batch.past_cont is None
    assert batch.future_cont is not None
    assert batch.future_cont.shape == (2, 2, 1)
    torch.testing.assert_close(batch.future_cont[0], batch.future_cont[1])
    assert batch.as_legacy_kwargs()["future_exo"] is batch.future_cont


def test_data_layer_builds_and_attaches_exogenous_schema():
    request = {
        "df": _frame(),
        "lookback": 2,
        "horizon": 1,
        "freq": "daily",
        "past_exo_cont_cols": ["price", "promo"],
        "past_exo_cat_cols": ["segment"],
        "future_exo_cont_cols": ["promo"],
    }

    schema = build_exogenous_schema(request)
    datamodule = build_datamodule(request)

    assert schema.past_cont_names == ("price", "promo")
    assert schema.past_cat_names == ("segment",)
    assert schema.future_cont_names == ("promo",)
    assert datamodule.exogenous_schema == schema


def test_data_layer_rejects_missing_exogenous_columns_before_dataset_build():
    with pytest.raises(ValueError, match="missing dataframe columns: unavailable"):
        build_exogenous_schema(
            {
                "df": _frame(),
                "past_exo_cont_cols": ["unavailable"],
            }
        )

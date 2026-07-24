from __future__ import annotations

import polars as pl
import pytest
import torch

from modeling_module import (
    DataRequest,
    ExogenousBatch,
    ExogenousConfig,
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
        future_cat=["segment", "holiday"],
        future_cat_cardinalities=[3, 2],
    )
    same = ExogenousFeatureSchema.from_columns(
        past_cont=["price", "promo"],
        past_cat=["segment"],
        future_cont=["promo"],
        past_cat_cardinalities=[3],
        future_cat=["segment", "holiday"],
        future_cat_cardinalities=[3, 2],
    )
    reordered = ExogenousFeatureSchema.from_columns(
        past_cont=["promo", "price"],
        past_cat=["segment"],
        future_cont=["promo"],
        past_cat_cardinalities=[3],
        future_cat=["segment", "holiday"],
        future_cat_cardinalities=[3, 2],
    )

    assert schema.past_cont_names == ("price", "promo")
    assert schema.future_cat_names == ("segment", "holiday")
    assert schema.future_cat_cardinalities == (3, 2)
    assert schema.fingerprint == same.fingerprint
    assert schema.fingerprint != reordered.fingerprint


def test_exogenous_feature_schema_preserves_legacy_v1_payload_and_fingerprint():
    legacy = ExogenousFeatureSchema(
        past_cont_names=("price", "promo"),
        past_cat_names=("segment",),
        future_cont_names=("promo",),
        past_cat_cardinalities=(3,),
        version=1,
    )
    payload = {
        "version": 1,
        "past_cont_names": ["price", "promo"],
        "past_cat_names": ["segment"],
        "future_cont_names": ["promo"],
        "past_cat_cardinalities": [3],
    }

    assert legacy.to_dict() == payload
    assert legacy.fingerprint == "a2c532cce16281db9a6e5bc1fc9166594a3752d1f8880224d57cef7c072c4ec5"
    assert ExogenousFeatureSchema(**payload) == legacy

    with pytest.raises(ValueError, match="future categorical features require"):
        ExogenousFeatureSchema(
            version=1,
            future_cat_names=("holiday",),
            future_cat_cardinalities=(2,),
        )


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

    with pytest.raises(ValueError, match="future_cat_cardinalities"):
        ExogenousFeatureSchema.from_columns(
            future_cat=["segment"],
            future_cat_cardinalities=[3, 4],
        )

    with pytest.raises(ValueError, match="one cardinality"):
        ExogenousFeatureSchema.from_columns(
            past_cat=["segment"],
            future_cat=["segment"],
            past_cat_cardinalities=[3],
            future_cat_cardinalities=[4],
        )

    with pytest.raises(ValueError, match="categorical and continuous"):
        ExogenousFeatureSchema.from_columns(
            past_cont=["holiday"],
            future_cat=["holiday"],
        )


def test_exogenous_batch_validates_shape_dtype_and_exact_schema():
    schema = ExogenousFeatureSchema.from_columns(
        past_cont=["price", "promo"],
        past_cat=["segment"],
        future_cont=["promo"],
        past_cat_cardinalities=[3],
        future_cat=["segment"],
        future_cat_cardinalities=[3],
    )
    batch = ExogenousBatch(
        past_cont=torch.randn(2, 4, 2),
        past_cat=torch.randint(0, 3, (2, 4, 1)),
        future_cont=torch.randn(2, 2, 1),
        future_cat=torch.randint(0, 3, (2, 2, 1)),
    )

    assert batch.validate(batch_size=2, lookback=4, horizon=2, schema=schema) is batch
    assert batch.provided_inputs() == frozenset(
        {"past_cont", "past_cat", "future_cont", "future_cat"}
    )

    with pytest.raises(ValueError, match="past_cont time-axis mismatch"):
        batch.validate(lookback=3)

    with pytest.raises(TypeError, match="past_cat must use an integer dtype"):
        ExogenousBatch(past_cat=torch.randn(2, 4, 1)).validate()

    with pytest.raises(TypeError, match="future_cat must use an integer dtype"):
        ExogenousBatch(future_cat=torch.randn(2, 2, 1)).validate()

    with pytest.raises(ValueError, match="future_cat time-axis mismatch"):
        ExogenousBatch(future_cat=torch.zeros(2, 3, 1, dtype=torch.long)).validate(
            horizon=2
        )

    with pytest.raises(ValueError, match="future_cont is required by the exogenous schema"):
        ExogenousBatch(
            past_cont=batch.past_cont,
            past_cat=batch.past_cat,
            future_cat=batch.future_cat,
        ).validate(schema=schema)

    endogenous_schema = ExogenousFeatureSchema()
    with pytest.raises(ValueError, match="past_cont is not declared"):
        ExogenousBatch(past_cont=torch.randn(2, 4, 1)).validate(
            schema=endogenous_schema
        )


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (torch.tensor([[[-1]], [[0]]]), "must be non-negative"),
        (torch.tensor([[[3]], [[0]]]), "exceed schema cardinality"),
    ],
)
def test_exogenous_batch_validates_future_category_id_range(
    values: torch.Tensor,
    match: str,
):
    schema = ExogenousFeatureSchema.from_columns(
        future_cat=["segment"],
        future_cat_cardinalities=[3],
    )

    with pytest.raises(ValueError, match=match):
        ExogenousBatch(future_cat=values).validate(schema=schema)


def test_exogenous_batch_normalizes_legacy_future_shape_and_empty_tensors():
    batch = ExogenousBatch.from_legacy(
        past_exo_cont=torch.empty(2, 4, 0),
        future_exo=torch.tensor([[1.0], [2.0]]),
        future_exo_cat=torch.tensor([[0], [1]]),
        batch_size=2,
    )

    assert batch.past_cont is None
    assert batch.future_cont is not None
    assert batch.future_cat is not None
    assert batch.future_cont.shape == (2, 2, 1)
    assert batch.future_cat.shape == (2, 2, 1)
    torch.testing.assert_close(batch.future_cont[0], batch.future_cont[1])
    torch.testing.assert_close(batch.future_cat[0], batch.future_cat[1])
    assert batch.as_legacy_kwargs()["future_exo"] is batch.future_cont
    assert batch.as_legacy_kwargs()["future_exo_cat"] is batch.future_cat


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


def test_data_layer_builds_future_categorical_schema_from_nested_and_flat_configs():
    nested = build_exogenous_schema(
        DataRequest(
            df=_frame(),
            exogenous=ExogenousConfig(
                past_exo_cat_cols=["segment"],
                future_exo_cat_cols=["segment"],
            ),
        )
    )
    flat = build_exogenous_schema(
        {
            "df": _frame(),
            "past_exo_cat_cols": ["segment"],
            "future_exo_cat_cols": ["segment"],
        }
    )
    aliased = build_exogenous_schema(
        {
            "df": _frame(),
            "exogenous": {
                "past_cat": ["segment"],
                "future_cat": ["segment"],
            },
        }
    )

    assert nested.past_cat_names == ("segment",)
    assert nested.future_cat_names == ("segment",)
    assert nested == flat == aliased


def test_data_layer_builds_future_categorical_dataset_tensor_and_schema():
    datamodule = build_datamodule(
        {
            "df": _frame(),
            "lookback": 2,
            "horizon": 1,
            "freq": "daily",
            "val_ratio": 0.0,
            "future_exo_cont_cols": ["promo"],
            "future_exo_cat_cols": ["segment"],
        }
    )
    datamodule.setup()

    sample = datamodule.train_dataset[0]
    assert len(sample) == 7
    assert sample[3].dtype == torch.float32
    assert sample[3].shape == (1, 1)
    assert sample[6].dtype == torch.long
    assert sample[6].shape == (1, 1)
    assert datamodule.categorical_vocabulary_artifact.feature_names == (
        "segment",
    )
    assert datamodule.exogenous_schema.future_cat_names == ("segment",)
    assert datamodule.exogenous_schema.future_cat_cardinalities == (4,)

    loader = datamodule.get_train_loader(
        batch_size=2,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )
    assert loader.exogenous_schema == datamodule.exogenous_schema
    assert (
        loader.categorical_vocabulary_fingerprint
        == datamodule.categorical_vocabulary_fingerprint
    )

    batch = next(
        iter(
            loader
        )
    )
    assert len(batch) == 7
    assert batch[3].shape == (2, 1, 1)
    assert batch[6].dtype == torch.long
    assert batch[6].shape == (2, 1, 1)


def test_data_layer_rejects_missing_exogenous_columns_before_dataset_build():
    with pytest.raises(ValueError, match="missing dataframe columns: unavailable"):
        build_exogenous_schema(
            {
                "df": _frame(),
                "past_exo_cont_cols": ["unavailable"],
            }
        )

    with pytest.raises(ValueError, match="missing dataframe columns: unavailable"):
        build_exogenous_schema(
            {
                "df": _frame(),
                "future_exo_cat_cols": ["unavailable"],
            }
        )

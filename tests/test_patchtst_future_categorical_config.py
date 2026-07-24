from __future__ import annotations

import pytest
import torch

from modeling_module.data_loader.exogenous_contracts import ExogenousFeatureSchema
from modeling_module.models.PatchTST.common.configs import (
    AttentionConfig,
    PatchTSTConfig,
)
from modeling_module.models.PatchTST.supervised.variants import (
    PatchTSTEndogenousModel,
    PatchTSTExogenousModel,
)
from modeling_module.models.PatchTST.supervised.future_categorical import (
    FutureCategoricalEmbedding,
)
from modeling_module.models.model_builder import (
    build_patchTST,
    build_patchTST_exogenous,
    build_patchTST_quantile_exogenous,
)
from modeling_module.training.config import TrainingConfig
from modeling_module.training.model_trainers import total_train
from modeling_module.training.model_trainers.exo_policy import (
    infer_future_cat_cardinalities_from_loader,
    infer_future_exo_spec_from_loader,
)
from modeling_module.utils.checkpoint import build_checkpoint_payload


def _config(**overrides) -> PatchTSTConfig:
    values = {
        "lookback": 8,
        "horizon": 2,
        "patch_len": 4,
        "stride": 2,
        "padding_patch": "end",
        "d_model": 8,
        "d_ff": 16,
        "n_layers": 1,
        "dropout": 0.0,
        "c_in": 1,
        "use_revin": False,
        "attn": AttentionConfig(
            n_heads=2,
            d_model=8,
            attn_dropout=0.0,
            proj_dropout=0.0,
        ),
    }
    values.update(overrides)
    return PatchTSTConfig(**values)


def test_future_categorical_config_defaults_preserve_endogenous_model() -> None:
    config = _config()
    model = build_patchTST(config)

    assert config.future_exo_cat_cardinalities == ()
    assert config.future_exo_cat_embedding_dim == 8
    assert config.future_exo_cat_dim == 0
    assert isinstance(model, PatchTSTEndogenousModel)
    assert model.future_exo_cat_cardinalities == ()
    assert model.future_exo_cat_dim == 0
    assert model.future_exo_cat_embedding_dim == 8
    assert model.future_cat_embedding is None


def test_future_categorical_config_is_normalized_serialized_and_routes_exogenous() -> None:
    endogenous = build_patchTST(_config())
    config = _config(
        future_exo_cat_cardinalities=[4, 7],
        future_exo_cat_embedding_dim=6,
    )
    model = build_patchTST(config)
    checkpoint = build_checkpoint_payload(model, config)

    assert isinstance(model, PatchTSTExogenousModel)
    assert config.future_exo_cat_cardinalities == (4, 7)
    assert config.future_exo_cat_dim == 2
    assert model.future_exo_cat_cardinalities == (4, 7)
    assert model.future_exo_cat_dim == 2
    assert model.future_exo_cat_embedding_dim == 6
    embedding_keys = {
        "future_cat_embedding.tables.0.weight",
        "future_cat_embedding.tables.1.weight",
    }
    fuser_keys = {
        key for key in model.state_dict() if key.startswith("future_fuser.")
    }
    assert fuser_keys
    assert set(model.state_dict()) == (
        set(endogenous.state_dict()) | embedding_keys | fuser_keys
    )
    assert model.state_dict()["future_cat_embedding.tables.0.weight"].shape == (4, 6)
    assert model.state_dict()["future_cat_embedding.tables.1.weight"].shape == (7, 6)
    assert model.future_fuser is not None
    assert model.future_fuser.input_dim == 12
    assert model.future_fuser.future_proj.in_features == 12
    assert checkpoint["cfg_state"]["future_exo_cat_cardinalities"] == [4, 7]
    assert checkpoint["cfg_state"]["future_exo_cat_embedding_dim"] == 6
    assert (embedding_keys | fuser_keys).issubset(checkpoint["state_dict"])

    restored = build_patchTST(config)
    restored.load_state_dict(checkpoint["state_dict"], strict=True)


def test_future_categorical_embedding_uses_independent_tables_and_learnable_unk() -> None:
    embedding = FutureCategoricalEmbedding(
        cardinalities=(3, 5),
        embedding_dim=4,
        horizon=2,
    )
    future_cat = torch.tensor(
        [
            [[0, 0], [1, 3]],
            [[2, 4], [0, 0]],
        ],
        dtype=torch.int16,
    )

    output = embedding(future_cat, batch_size=2)

    assert len(embedding.tables) == 2
    assert embedding.tables[0].num_embeddings == 3
    assert embedding.tables[1].num_embeddings == 5
    assert all(table.padding_idx is None for table in embedding.tables)
    assert embedding.output_dim == 8
    assert output.shape == (2, 2, 8)
    torch.testing.assert_close(
        output[..., :4],
        embedding.tables[0](future_cat[..., 0].long()),
    )
    torch.testing.assert_close(
        output[..., 4:],
        embedding.tables[1](future_cat[..., 1].long()),
    )

    output.sum().backward()
    assert float(embedding.tables[0].weight.grad[0].abs().sum()) > 0.0
    assert float(embedding.tables[1].weight.grad[0].abs().sum()) > 0.0


@pytest.mark.parametrize(
    ("future_cat", "batch_size", "error_type", "message"),
    (
        (
            torch.zeros(2, 2, dtype=torch.long),
            2,
            ValueError,
            "rank 3",
        ),
        (
            torch.zeros(3, 2, 2, dtype=torch.long),
            2,
            ValueError,
            "batch mismatch",
        ),
        (
            torch.zeros(2, 3, 2, dtype=torch.long),
            2,
            ValueError,
            "horizon mismatch",
        ),
        (
            torch.zeros(2, 2, 1, dtype=torch.long),
            2,
            ValueError,
            "feature-width mismatch",
        ),
        (
            torch.zeros(2, 2, 2, dtype=torch.float32),
            2,
            TypeError,
            "integer dtype",
        ),
        (
            torch.tensor(
                [[[-1, 0], [0, 0]], [[0, 0], [0, 0]]],
                dtype=torch.long,
            ),
            2,
            ValueError,
            "feature index 0",
        ),
        (
            torch.tensor(
                [[[0, 5], [0, 0]], [[0, 0], [0, 0]]],
                dtype=torch.long,
            ),
            2,
            ValueError,
            "feature index 1",
        ),
    ),
)
def test_future_categorical_embedding_validates_shape_dtype_and_id_range(
    future_cat,
    batch_size,
    error_type,
    message,
) -> None:
    embedding = FutureCategoricalEmbedding(
        cardinalities=(3, 5),
        embedding_dim=4,
        horizon=2,
    )

    with pytest.raises(error_type, match=message):
        embedding(future_cat, batch_size=batch_size)


@pytest.mark.parametrize(
    "builder",
    (build_patchTST_exogenous, build_patchTST_quantile_exogenous),
)
def test_patchtst_point_and_quantile_register_future_categorical_embeddings(
    builder,
) -> None:
    model = builder(
        _config(
            future_exo_cat_cardinalities=(3, 5),
            future_exo_cat_embedding_dim=4,
        )
    )
    future_cat = torch.tensor(
        [
            [[0, 0], [1, 3]],
            [[2, 4], [0, 0]],
        ],
        dtype=torch.int32,
    )

    output = model.encode_future_categorical(future_cat, batch_size=2)

    assert output is not None
    assert output.shape == (2, 2, 8)
    assert model.future_cat_embedding is not None
    assert model.future_cat_embedding.cardinalities == (3, 5)


@pytest.mark.parametrize(
    "overrides",
    (
        {"future_exo_cat_cardinalities": [3, 0]},
        {"future_exo_cat_cardinalities": [True]},
        {"future_exo_cat_cardinalities": 3},
        {"future_exo_cat_embedding_dim": 0},
        {"future_exo_cat_embedding_dim": 1.5},
    ),
)
def test_future_categorical_config_rejects_invalid_values(overrides) -> None:
    with pytest.raises(ValueError, match="future_exo_cat"):
        _config(**overrides)


def test_model_creation_revalidates_mutated_future_categorical_config() -> None:
    config = _config()
    config.future_exo_cat_cardinalities = (3, -1)

    with pytest.raises(ValueError, match="positive integers"):
        build_patchTST(config)


def test_explicit_endogenous_model_rejects_future_categorical_width() -> None:
    config = _config(future_exo_cat_cardinalities=(3,))

    with pytest.raises(ValueError, match="future_cat=1"):
        PatchTSTEndogenousModel(config)


def test_future_categorical_only_model_requires_runtime_tensor_and_predicts() -> None:
    model = build_patchTST_exogenous(
        _config(future_exo_cat_cardinalities=(3,))
    )
    x = torch.ones(2, 8, 1)

    with pytest.raises(RuntimeError, match="future_exo_cat"):
        model(x)
    invalid = torch.zeros(2, 2, 1, dtype=torch.long)
    invalid[0, 0, 0] = 3
    with pytest.raises(ValueError, match="feature index 0"):
        model(x, future_exo_cat=invalid)

    output = model(
        x,
        future_exo_cat=torch.zeros(2, 2, 1, dtype=torch.long),
    )

    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()


def test_loader_schema_exposes_fitted_future_categorical_cardinalities() -> None:
    class _Loader:
        exogenous_schema = ExogenousFeatureSchema.from_columns(
            future_cat=["promotion_type", "holiday_type"],
            future_cat_cardinalities=[5, 3],
        )

    assert infer_future_cat_cardinalities_from_loader(_Loader()) == (5, 3)


def test_seven_tuple_future_category_is_not_inferred_as_continuous() -> None:
    batch = (
        torch.zeros(2, 8, 1),
        torch.zeros(2, 2),
        ["A", "B"],
        torch.empty(2, 2, 0),
        torch.empty(2, 8, 0),
        torch.empty(2, 8, 0, dtype=torch.long),
        torch.ones(2, 2, 1, dtype=torch.long),
    )

    has_future_cont, future_cont_dim = infer_future_exo_spec_from_loader(
        [batch],
        lookback=8,
        horizon=2,
    )

    assert has_future_cont is True
    assert future_cont_dim == 0


def test_patchtst_runner_binds_loader_cardinalities_and_embedding_override(
    monkeypatch,
) -> None:
    class _Loader:
        exogenous_schema = ExogenousFeatureSchema.from_columns(
            future_cat=["promotion_type", "holiday_type"],
            future_cat_cardinalities=[5, 3],
        )

    captured: dict[str, PatchTSTConfig] = {}

    def _build(config):
        captured["config"] = config
        return torch.nn.Identity()

    monkeypatch.setattr(total_train, "build_patchTST_exogenous", _build)
    monkeypatch.setattr(
        total_train,
        "train_patchtst",
        lambda model, *args, **kwargs: {"model": model, "best_val": 0.0},
    )

    results = {}
    total_train._run_patchtst(
        results=results,
        freq="weekly",
        train_loader=_Loader(),
        val_loader=None,
        save_root=None,
        lookback=8,
        horizon=2,
        future_exo_cb=None,
        exo_dim=0,
        past_cont_dim=0,
        past_cat_dim=0,
        patch_len=4,
        stride=2,
        point_train_cfg=TrainingConfig(device="cpu", lookback=8, horizon=2),
        quantile_train_cfg=TrainingConfig(
            device="cpu",
            lookback=8,
            horizon=2,
        ),
        stages=[],
        device="cpu",
        use_exogenous_mode=True,
        use_ssl_mode="sl_only",
        requested_artifact_keys=["patchtst_exogenous"],
        architecture_override={"future_exo_cat_embedding_dim": 6},
    )

    config = captured["config"]
    assert config.future_exo_cat_cardinalities == (5, 3)
    assert config.future_exo_cat_embedding_dim == 6
    assert results["PatchTST Exogenous"]["model_key"] == "patchtst_exogenous"

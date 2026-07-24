from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from modeling_module.api.data import (
    DataRequest,
    ExogenousConfig,
    build_dataloader,
)
from modeling_module.api.forecast import (
    ForecastRequest,
    ForecastRuntimeConfig,
    forecast,
)
from modeling_module.api.infer import load_predictor
from modeling_module.api.train import (
    ArchitectureConfig,
    ArtifactConfig,
    PatchTSTArchitectureConfig,
    RuntimeConfig,
    SSLConfig,
    TrainerConfig,
    TrainRequest,
    train,
)
from modeling_module.data_loader.categorical_vocabulary import (
    CategoricalVocabularyArtifact,
)
from modeling_module.data_loader.exogenous_contracts import (
    ExogenousFeatureSchema,
)
from modeling_module.models.PatchTST.common.configs import (
    AttentionConfig,
    PatchTSTConfig,
)
from modeling_module.models.model_builder import build_patchTST_exogenous
from modeling_module.utils.checkpoint import save_model


def _checkpoint_config(cardinalities: tuple[int, ...]) -> PatchTSTConfig:
    return PatchTSTConfig(
        lookback=8,
        horizon=2,
        patch_len=4,
        stride=2,
        padding_patch="end",
        d_model=8,
        d_ff=16,
        n_layers=1,
        dropout=0.0,
        c_in=1,
        future_exo_cat_cardinalities=cardinalities,
        future_exo_cat_embedding_dim=4,
        future_exo_fusion_dropout=0.0,
        use_revin=False,
        attn=AttentionConfig(
            n_heads=2,
            d_model=8,
            attn_dropout=0.0,
            proj_dropout=0.0,
        ),
    )


def _vocabulary_contract():
    schema = ExogenousFeatureSchema.from_columns(
        future_cat=["event_type"],
    )
    vocabulary = CategoricalVocabularyArtifact.fit_for_schema(
        schema,
        {
            "event_type": [
                "regular",
                "promotion",
                "regular",
            ]
        },
    )
    return vocabulary.bind_schema(schema), vocabulary


def test_categorical_checkpoint_strict_restore_preserves_prediction(
    tmp_path: Path,
) -> None:
    torch.manual_seed(401)
    schema, vocabulary = _vocabulary_contract()
    config = _checkpoint_config(schema.future_cat_cardinalities)
    model = build_patchTST_exogenous(config).eval()
    x = torch.linspace(-1.0, 1.0, 16).reshape(2, 8, 1)
    future_cat = torch.tensor(
        [
            [[1], [2]],
            [[2], [1]],
        ],
        dtype=torch.long,
    )
    with torch.no_grad():
        expected = model(x, future_exo_cat=future_cat)

    checkpoint_path = tmp_path / "patchtst-categorical.pt"
    save_model(
        model,
        config,
        str(checkpoint_path),
        extra_meta={
            "model_key": "patchtst_exogenous",
            "family_key": "patchtst",
        },
        exogenous_schema=schema,
        categorical_vocabulary_artifact=vocabulary,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    data_artifacts = checkpoint["data_artifacts"]
    assert checkpoint["cfg_state"][
        "future_exo_cat_cardinalities"
    ] == list(schema.future_cat_cardinalities)
    assert any(
        key.startswith("future_cat_embedding.tables.0")
        for key in checkpoint["state_dict"]
    )
    assert data_artifacts["exogenous_schema"] == schema.to_dict()
    assert (
        data_artifacts["categorical_vocabulary"]
        == vocabulary.to_dict()
    )
    assert (
        data_artifacts["categorical_vocabulary_fingerprint"]
        == vocabulary.fingerprint
    )

    predictor = load_predictor(
        str(checkpoint_path),
        device="cpu",
        strict=True,
    )
    assert (
        predictor.categorical_vocabulary_fingerprint
        == vocabulary.fingerprint
    )
    assert (
        predictor.categorical_vocabulary_artifact.to_dict()
        == vocabulary.to_dict()
    )
    assert predictor.exogenous_schema == schema
    with torch.no_grad():
        restored = predictor.model(
            x,
            future_exo_cat=future_cat,
        )
    torch.testing.assert_close(restored, expected, rtol=0.0, atol=0.0)

    public_output = predictor.predict(
        {
            "x": x,
            "future_exo_cat_batch": future_cat,
        },
        horizon=2,
    )
    np.testing.assert_array_equal(
        np.asarray(public_output["point"]),
        expected.numpy().reshape(-1),
    )


def test_categorical_checkpoint_rejects_tampered_fingerprint(
    tmp_path: Path,
) -> None:
    schema, vocabulary = _vocabulary_contract()
    config = _checkpoint_config(schema.future_cat_cardinalities)
    model = build_patchTST_exogenous(config)
    checkpoint_path = tmp_path / "tampered.pt"
    save_model(
        model,
        config,
        str(checkpoint_path),
        extra_meta={"model_key": "patchtst_exogenous"},
        exogenous_schema=schema,
        categorical_vocabulary_artifact=vocabulary,
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint["data_artifacts"][
        "categorical_vocabulary_fingerprint"
    ] = "0" * 64
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(
        ValueError,
        match="vocabulary fingerprint mismatch",
    ):
        load_predictor(str(checkpoint_path), device="cpu", strict=True)


def _training_frame(end_day: int = 14) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["series-1"] * end_day,
            "date": [20240100 + day for day in range(1, end_day + 1)],
            "y": [
                2.0 + 0.2 * day + float(day % 3)
                for day in range(1, end_day + 1)
            ],
            "event_type": [
                "regular" if day % 2 else "promotion"
                for day in range(1, end_day + 1)
            ],
        }
    )


def _training_request(
    *,
    model_key: str,
    save_dir: Path,
) -> TrainRequest:
    return TrainRequest(
        data=DataRequest(
            df=_training_frame(),
            lookback=4,
            horizon=2,
            freq="daily",
            batch_size=4,
            val_ratio=0.25,
            shuffle=False,
            exogenous=ExogenousConfig(
                use_exogenous_mode=True,
                use_past_exogenous=False,
                use_future_exogenous=True,
                future_exo_cat_cols=["event_type"],
            ),
        ),
        models=[model_key],
        trainer=TrainerConfig(
            epochs=1,
            lr=1e-3,
            use_intermittent=False,
            val_use_weights=False,
        ),
        ssl=SSLConfig(mode="sl_only"),
        runtime=RuntimeConfig(device="cpu"),
        artifacts=ArtifactConfig(
            save_dir=str(save_dir),
            auto_save_dir=False,
        ),
        architecture=ArchitectureConfig(
            patchtst=PatchTSTArchitectureConfig(
                patch_len=2,
                stride=1,
                d_model=16,
                n_layers=1,
                d_ff=16,
                dropout=0.0,
                norm="LayerNorm",
                pre_norm=True,
                use_revin=False,
                pe="zeros",
                learn_pe=True,
                padding_patch="end",
                future_exo_cat_embedding_dim=4,
                future_exo_fusion_dropout=0.0,
            )
        ),
    )


def _forecast_request(
    checkpoint_path: str,
    *,
    model_key: str,
    future_events: tuple[str, str],
) -> ForecastRequest:
    history = _training_frame()
    future = pl.DataFrame(
        {
            "unique_id": ["series-1", "series-1"],
            "date": [20240115, 20240116],
            "y": [0.0, 0.0],
            "event_type": list(future_events),
        }
    )
    return ForecastRequest(
        checkpoint_path=checkpoint_path,
        expected_model_key=model_key,
        data=DataRequest(
            df=pl.concat([history, future]),
            lookback=4,
            horizon=2,
            freq="daily",
            exogenous=ExogenousConfig(
                use_exogenous_mode=True,
                use_past_exogenous=False,
                use_future_exogenous=True,
                future_exo_cat_cols=["event_type"],
            ),
        ),
        series_ids=["series-1"],
        forecast_origin=20240115,
        runtime=ForecastRuntimeConfig(
            batch_size=1,
            num_workers=0,
            device="cpu",
            pin_memory=False,
        ),
    )


def _save_categorical_checkpoint(
    path: Path,
    *,
    schema: ExogenousFeatureSchema,
    vocabulary: CategoricalVocabularyArtifact,
) -> None:
    config = _checkpoint_config(
        vocabulary.bind_schema(schema).future_cat_cardinalities
    )
    save_model(
        build_patchTST_exogenous(config),
        config,
        str(path),
        extra_meta={
            "model_key": "patchtst_exogenous",
            "family_key": "patchtst",
        },
        exogenous_schema=vocabulary.bind_schema(schema),
        categorical_vocabulary_artifact=vocabulary,
    )


@pytest.mark.parametrize(
    "model_key",
    (
        "patchtst_exogenous",
        "patchtst_quantile_exogenous",
    ),
)
def test_public_dataframe_categorical_train_save_load_forecast(
    tmp_path: Path,
    model_key: str,
) -> None:
    torch.manual_seed(402)
    np.random.seed(402)
    result = train(
        _training_request(
            model_key=model_key,
            save_dir=tmp_path / model_key,
        )
    )
    assert result.primary_ckpt_path is not None

    checkpoint = torch.load(
        result.primary_ckpt_path,
        map_location="cpu",
        weights_only=False,
    )
    data_artifacts = checkpoint["data_artifacts"]
    fingerprint = data_artifacts[
        "categorical_vocabulary_fingerprint"
    ]
    assert isinstance(fingerprint, str) and len(fingerprint) == 64
    assert checkpoint["cfg_state"][
        "future_exo_cat_cardinalities"
    ] == [3]

    predictor = load_predictor(
        result.primary_ckpt_path,
        device="cpu",
        strict=True,
    )
    assert predictor.categorical_vocabulary_fingerprint == fingerprint
    vocabulary = predictor.categorical_vocabulary_artifact.vocabulary_for(
        "event_type"
    )
    assert vocabulary.id_of("regular") != vocabulary.id_of("promotion")

    regular = forecast(
        _forecast_request(
            result.primary_ckpt_path,
            model_key=model_key,
            future_events=("regular", "regular"),
        )
    )
    promotion = forecast(
        _forecast_request(
            result.primary_ckpt_path,
            model_key=model_key,
            future_events=("promotion", "promotion"),
        )
    )

    assert regular.predictions.height == 2
    assert promotion.predictions.height == 2
    assert regular.predictions["model_key"].unique().to_list() == [
        model_key
    ]
    assert regular.predictions["point"].is_finite().all()
    assert promotion.predictions["point"].is_finite().all()
    assert not np.allclose(
        regular.predictions["point"].to_numpy(),
        promotion.predictions["point"].to_numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

    unseen_request = _forecast_request(
        result.primary_ckpt_path,
        model_key=model_key,
        future_events=("emergency", "emergency"),
    )
    unseen_request.data.categorical_vocabulary_artifact = (
        predictor.categorical_vocabulary_artifact
    )
    unseen_request.data.stage = "inference"
    unseen_request.data.plan_dt = 20240115
    unseen_request.data.series_ids = ["series-1"]
    unseen_request.data.batch_size = 1
    unseen_request.data.num_workers = 0
    unseen_request.data.pin_memory = False
    unseen_batch = next(iter(build_dataloader(unseen_request.data)))
    assert vocabulary.id_of("emergency") == 0
    assert unseen_batch[6].dtype == torch.long
    assert unseen_batch[6].eq(0).all()

    unseen = forecast(unseen_request)
    assert unseen.predictions.height == 2
    assert unseen.predictions["point"].is_finite().all()
    restored_again = load_predictor(
        result.primary_ckpt_path,
        device="cpu",
        strict=True,
    )
    assert restored_again.categorical_vocabulary_fingerprint == fingerprint


def test_forecast_rejects_missing_future_categorical_column(
    tmp_path: Path,
) -> None:
    schema, vocabulary = _vocabulary_contract()
    checkpoint_path = tmp_path / "missing-column.pt"
    _save_categorical_checkpoint(
        checkpoint_path,
        schema=schema,
        vocabulary=vocabulary,
    )
    request = _forecast_request(
        str(checkpoint_path),
        model_key="patchtst_exogenous",
        future_events=("regular", "promotion"),
    )
    request.data.df = request.data.df.drop("event_type")

    with pytest.raises(
        ValueError,
        match=(
            "Exogenous schema references missing dataframe columns: "
            "event_type"
        ),
    ):
        forecast(request)


def test_forecast_rejects_reordered_future_categorical_columns(
    tmp_path: Path,
) -> None:
    schema = ExogenousFeatureSchema.from_columns(
        future_cat=["event_type", "plan_state"],
    )
    vocabulary = CategoricalVocabularyArtifact.fit_for_schema(
        schema,
        {
            "event_type": ["regular", "promotion"],
            "plan_state": ["draft", "confirmed"],
        },
    )
    checkpoint_path = tmp_path / "reordered-columns.pt"
    _save_categorical_checkpoint(
        checkpoint_path,
        schema=schema,
        vocabulary=vocabulary,
    )
    frame = _training_frame().with_columns(
        pl.Series(
            "plan_state",
            [
                "draft" if index % 2 else "confirmed"
                for index in range(1, 15)
            ],
        )
    )
    request = ForecastRequest(
        checkpoint_path=str(checkpoint_path),
        expected_model_key="patchtst_exogenous",
        data=DataRequest(
            df=frame,
            lookback=4,
            horizon=2,
            freq="daily",
            exogenous=ExogenousConfig(
                use_exogenous_mode=True,
                use_future_exogenous=True,
                future_exo_cat_cols=["plan_state", "event_type"],
            ),
        ),
        series_ids=["series-1"],
        forecast_origin=20240113,
        runtime=ForecastRuntimeConfig(
            batch_size=1,
            num_workers=0,
            device="cpu",
            pin_memory=False,
        ),
    )

    with pytest.raises(
        ValueError,
        match="categorical vocabulary feature order",
    ):
        forecast(request)


def test_forecast_rejects_checkpoint_categorical_role_mismatch(
    tmp_path: Path,
) -> None:
    schema, vocabulary = _vocabulary_contract()
    checkpoint_path = tmp_path / "role-mismatch.pt"
    _save_categorical_checkpoint(
        checkpoint_path,
        schema=schema,
        vocabulary=vocabulary,
    )
    request = ForecastRequest(
        checkpoint_path=str(checkpoint_path),
        expected_model_key="patchtst_exogenous",
        data=DataRequest(
            df=_training_frame(),
            lookback=4,
            horizon=2,
            freq="daily",
            exogenous=ExogenousConfig(
                use_exogenous_mode=True,
                use_past_exogenous=True,
                use_future_exogenous=False,
                past_exo_cat_cols=["event_type"],
            ),
        ),
        series_ids=["series-1"],
        forecast_origin=20240113,
        runtime=ForecastRuntimeConfig(
            batch_size=1,
            num_workers=0,
            device="cpu",
            pin_memory=False,
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Forecast request exogenous schema does not match checkpoint "
            "schema: .*past_cat_names.*future_cat_names"
        ),
    ):
        forecast(request)

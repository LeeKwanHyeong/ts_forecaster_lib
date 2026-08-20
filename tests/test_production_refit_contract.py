from __future__ import annotations

import importlib

import pytest
import torch

from modeling_module import (
    ArchitectureConfig,
    ArtifactConfig,
    ExoTSTArchitectureConfig,
    NHITSArchitectureConfig,
    PatchMixerArchitectureConfig,
    PatchTSTArchitectureConfig,
    RuntimeConfig,
    SSLConfig,
    TimeMixerArchitectureConfig,
    TimexerArchitectureConfig,
    TrainRequest,
    TrainerConfig,
    load_predictor,
    train,
)
from modeling_module.data_loader.exogenous_contracts import (
    ExogenousFeatureSchema,
)


PRODUCTION_MODEL_KEYS = (
    "patchtst_base",
    "patchtst_quantile",
    "patchmixer",
    "nhits_base",
    "timemixer",
)
EXOGENOUS_PRODUCTION_MODEL_KEYS = (
    "exotst_base",
    "patchtst_exogenous",
    "timexer_base",
)


def _loader():
    return [
        (
            torch.zeros(2, 14, 1),
            torch.zeros(2, 2),
            ["A", "B"],
        )
    ]


def _exo_loader():
    class Loader(list):
        pass

    loader = Loader(
        [
            (
                torch.zeros(2, 14, 1),
                torch.zeros(2, 2),
                ["A", "B"],
                torch.zeros(2, 2, 1),
                torch.zeros(2, 14, 1),
                torch.zeros(2, 14, 0, dtype=torch.long),
            )
        ]
    )
    loader.exogenous_schema = ExogenousFeatureSchema.from_columns(
        past_cont=("past_calendar",),
        future_cont=("future_calendar",),
    )
    return loader


def _past_only_exo_loader():
    class Loader(list):
        pass

    loader = Loader(
        [
            (
                torch.zeros(2, 14, 1),
                torch.zeros(2, 2),
                ["A", "B"],
                torch.zeros(2, 2, 0),
                torch.zeros(2, 14, 1),
                torch.zeros(2, 14, 0, dtype=torch.long),
            )
        ]
    )
    loader.exogenous_schema = ExogenousFeatureSchema.from_columns(
        past_cont=("past_calendar",),
    )
    return loader


def _architecture(model_key: str) -> ArchitectureConfig:
    if model_key == "exotst_base":
        return ArchitectureConfig(
            exotst=ExoTSTArchitectureConfig(
                patch_len=7,
                stride=3,
                d_model=16,
                n_heads=4,
                d_ff=32,
                dropout=0.0,
                attn_dropout=0.0,
                exo_enc_layers=1,
                fusion_layers=1,
                endo_dec_layers=1,
            )
        )
    if model_key.startswith("patchtst"):
        return ArchitectureConfig(
            patchtst=PatchTSTArchitectureConfig(
                patch_len=7,
                stride=3,
                d_model=16,
                n_layers=1,
                d_ff=32,
                dropout=0.0,
            )
        )
    if model_key == "patchmixer":
        return ArchitectureConfig(
            patchmixer=PatchMixerArchitectureConfig(
                patch_len=7,
                stride=3,
                d_model=4,
                e_layers=1,
                mixer_kernel_size=3,
                f_out=4,
                head_hidden=4,
                dropout=0.0,
                head_dropout=0.0,
                use_revin=False,
                final_nonneg=False,
                expander_n_harmonics=1,
            )
        )
    if model_key == "nhits_base":
        return ArchitectureConfig(
            nhits=NHITSArchitectureConfig(
                stack_types=("identity",),
                n_blocks=(1,),
                n_layers=(2,),
                n_theta_hidden=((8, 8),),
                n_pool_kernel_size=(1,),
                n_freq_downsample=(1,),
                batch_normalization=False,
                dropout_prob_theta=0.0,
                shared_weights=False,
            )
        )
    if model_key == "timemixer":
        return ArchitectureConfig(
            timemixer=TimeMixerArchitectureConfig(
                d_model=4,
                d_ff=8,
                e_layers=1,
                moving_avg=3,
                down_sampling_layers=1,
                down_sampling_window=2,
                dropout=0.0,
                use_norm=True,
            )
        )
    if model_key == "timexer_base":
        return ArchitectureConfig(
            timexer=TimexerArchitectureConfig(
                patch_len=7,
                d_model=4,
                n_heads=1,
                d_ff=8,
                e_layers=1,
                dropout=0.0,
                factor=1,
                activation="gelu",
                use_norm=False,
            )
        )
    raise AssertionError(f"Unexpected production model key: {model_key}")


@pytest.mark.parametrize("model_key", PRODUCTION_MODEL_KEYS)
def test_public_production_refit_accepts_train_only_loader_and_forwards_mode(
    monkeypatch,
    tmp_path,
    model_key,
):
    train_module = importlib.import_module("modeling_module.api.train")
    captured: dict[str, object] = {}

    def fake_run_total_train(train_loader, val_loader, **kwargs):
        captured["train_loader"] = train_loader
        captured["val_loader"] = val_loader
        captured.update(kwargs)
        return {
            "Production Artifact": {
                "ckpt_path": str(tmp_path / f"{model_key}.pt"),
                "model_key": model_key,
                "family_key": model_key.split("_", 1)[0],
                "state_selection": "final_epoch",
            }
        }

    monkeypatch.setattr(train_module, "run_total_train", fake_run_total_train)
    train_loader = _loader()

    result = train(
        TrainRequest(
            train_loader=train_loader,
            models=[model_key],
            freq="daily",
            lookback=14,
            horizon=2,
            trainer=TrainerConfig(
                epochs=3,
                lr=1e-3,
                training_mode="production_refit",
            ),
            ssl=SSLConfig(mode="sl_only"),
            runtime=RuntimeConfig(device="cpu"),
            artifacts=ArtifactConfig(
                save_dir=str(tmp_path),
                auto_save_dir=False,
            ),
        )
    )

    assert captured["train_loader"] is train_loader
    assert captured["val_loader"] is None
    assert captured["training_mode"] == "production_refit"
    assert result.primary_result_name == model_key


def test_public_production_refit_rejects_validation_loader(tmp_path):
    with pytest.raises(ValueError, match="requires `val_loader=None`"):
        train(
            TrainRequest(
                train_loader=_loader(),
                val_loader=_loader(),
                models=["patchtst_base"],
                freq="daily",
                lookback=14,
                horizon=2,
                trainer=TrainerConfig(
                    epochs=1,
                    training_mode="production_refit",
                ),
                artifacts=ArtifactConfig(
                    save_dir=str(tmp_path),
                    auto_save_dir=False,
                ),
            )
        )


def test_public_production_refit_rejects_unapproved_artifact(tmp_path):
    with pytest.raises(
        ValueError,
        match="supports exactly one approved artifact",
    ):
        train(
            TrainRequest(
                train_loader=_loader(),
                models=["patchmixer_exo"],
                freq="daily",
                lookback=14,
                horizon=2,
                trainer=TrainerConfig(
                    epochs=1,
                    training_mode="production_refit",
                ),
                artifacts=ArtifactConfig(
                    save_dir=str(tmp_path),
                    auto_save_dir=False,
                ),
            )
        )


def test_public_production_refit_rejects_multiple_artifacts(tmp_path):
    with pytest.raises(
        ValueError,
        match="supports exactly one approved artifact",
    ):
        train(
            TrainRequest(
                train_loader=_loader(),
                models=["patchtst"],
                freq="daily",
                lookback=14,
                horizon=2,
                trainer=TrainerConfig(
                    epochs=1,
                    training_mode="production_refit",
                ),
                artifacts=ArtifactConfig(
                    save_dir=str(tmp_path),
                    auto_save_dir=False,
                ),
            )
        )


@pytest.mark.parametrize("model_key", PRODUCTION_MODEL_KEYS)
def test_production_refit_saves_final_epoch_checkpoint(tmp_path, model_key):
    train_loader = _loader()
    result = train(
        TrainRequest(
            train_loader=train_loader,
            models=[model_key],
            freq="daily",
            lookback=14,
            horizon=2,
            trainer=TrainerConfig(
                epochs=1,
                lr=1e-3,
                use_intermittent=False,
                training_mode="production_refit",
                random_seed=42,
            ),
            ssl=SSLConfig(mode="sl_only"),
            runtime=RuntimeConfig(device="cpu"),
            artifacts=ArtifactConfig(
                save_dir=str(tmp_path / model_key),
                auto_save_dir=False,
            ),
            architecture=_architecture(model_key),
        )
    )

    assert result.primary_ckpt_path is not None
    checkpoint = torch.load(
        result.primary_ckpt_path,
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["meta"]["training_mode"] == "production_refit"
    assert checkpoint["meta"]["validation_enabled"] is False
    assert checkpoint["meta"]["state_selection"] == "final_epoch"
    assert checkpoint["meta"]["configured_epochs"] == 1
    assert checkpoint["meta"]["completed_epochs"] == 1
    assert checkpoint["meta"]["random_seed"] == 42
    assert checkpoint["meta"]["model_key"] == model_key
    assert checkpoint["meta"]["final_train_loss"] >= 0.0

    predictor = load_predictor(
        result.primary_ckpt_path,
        device="cpu",
        strict=True,
    )
    output = predictor.predict(
        {"x": train_loader[0][0], "part_ids": train_loader[0][2]},
        horizon=2,
    )
    point = torch.as_tensor(output["point"])
    assert point.numel() == 4
    assert torch.isfinite(point).all()
    assert predictor.model_key == model_key


@pytest.mark.parametrize("model_key", EXOGENOUS_PRODUCTION_MODEL_KEYS)
def test_exogenous_production_refit_saves_final_epoch_checkpoint(
    tmp_path,
    model_key,
):
    train_loader = (
        _past_only_exo_loader()
        if model_key == "timexer_base"
        else _exo_loader()
    )
    uses_future_exogenous = model_key != "timexer_base"
    configured_epochs = 2 if model_key == "timexer_base" else 1
    result = train(
        TrainRequest(
            train_loader=train_loader,
            models=[model_key],
            freq="daily",
            lookback=14,
            horizon=2,
            trainer=TrainerConfig(
                epochs=configured_epochs,
                lr=1e-3,
                use_intermittent=False,
                training_mode="production_refit",
                random_seed=42,
            ),
            ssl=SSLConfig(mode="sl_only"),
            runtime=RuntimeConfig(device="cpu"),
            artifacts=ArtifactConfig(
                save_dir=str(tmp_path / model_key),
                auto_save_dir=False,
            ),
            architecture=_architecture(model_key),
            use_exogenous_mode=True,
            use_past_exogenous=True,
            use_future_exogenous=uses_future_exogenous,
        )
    )

    assert result.primary_ckpt_path is not None
    checkpoint = torch.load(
        result.primary_ckpt_path,
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["meta"]["training_mode"] == "production_refit"
    assert checkpoint["meta"]["validation_enabled"] is False
    assert checkpoint["meta"]["state_selection"] == "final_epoch"
    assert checkpoint["meta"]["configured_epochs"] == configured_epochs
    assert checkpoint["meta"]["completed_epochs"] == configured_epochs
    assert checkpoint["meta"]["random_seed"] == 42
    assert checkpoint["meta"]["model_key"] == model_key
    assert checkpoint["meta"]["final_train_loss"] >= 0.0

    predictor = load_predictor(
        result.primary_ckpt_path,
        device="cpu",
        strict=True,
    )
    batch = train_loader[0]
    prediction_payload = {
        "x": batch[0],
        "part_ids": batch[2],
        "past_exo_cont": batch[4],
    }
    if uses_future_exogenous:
        prediction_payload["future_exo"] = batch[3]
    output = predictor.predict(prediction_payload, horizon=2)
    point = torch.as_tensor(output["point"])
    assert point.numel() == 4
    assert torch.isfinite(point).all()
    assert predictor.model_key == model_key

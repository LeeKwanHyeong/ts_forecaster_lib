from __future__ import annotations

import importlib

import pytest
import torch

from modeling_module import (
    ArchitectureConfig,
    ArtifactConfig,
    PatchTSTArchitectureConfig,
    RuntimeConfig,
    SSLConfig,
    TrainRequest,
    TrainerConfig,
    load_predictor,
    train,
)


def _loader():
    return [
        (
            torch.zeros(2, 14, 1),
            torch.zeros(2, 2),
            ["A", "B"],
        )
    ]


def test_public_production_refit_accepts_train_only_loader_and_forwards_mode(
    monkeypatch,
    tmp_path,
):
    train_module = importlib.import_module("modeling_module.api.train")
    captured: dict[str, object] = {}

    def fake_run_total_train(train_loader, val_loader, **kwargs):
        captured["train_loader"] = train_loader
        captured["val_loader"] = val_loader
        captured.update(kwargs)
        return {
            "PatchTST": {
                "ckpt_path": str(tmp_path / "weekly_PatchTST_L14_H2.pt"),
                "model_key": "patchtst_base",
                "family_key": "patchtst",
                "state_selection": "final_epoch",
            }
        }

    monkeypatch.setattr(train_module, "run_total_train", fake_run_total_train)
    train_loader = _loader()

    result = train(
        TrainRequest(
            train_loader=train_loader,
            models=["patchtst_base"],
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
    assert result.primary_result_name == "patchtst_base"


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


def test_public_production_refit_rejects_unverified_model_family(tmp_path):
    with pytest.raises(
        ValueError,
        match="supports exactly one artifact: patchtst_base",
    ):
        train(
            TrainRequest(
                train_loader=_loader(),
                models=["patchmixer"],
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


def test_patchtst_production_refit_saves_final_epoch_checkpoint(tmp_path):
    train_loader = _loader()
    result = train(
        TrainRequest(
            train_loader=train_loader,
            models=["patchtst_base"],
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
                save_dir=str(tmp_path),
                auto_save_dir=False,
            ),
            architecture=ArchitectureConfig(
                patchtst=PatchTSTArchitectureConfig(
                    patch_len=7,
                    stride=3,
                    d_model=32,
                    n_layers=1,
                    d_ff=64,
                    dropout=0.0,
                )
            ),
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

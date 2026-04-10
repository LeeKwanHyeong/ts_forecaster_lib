from __future__ import annotations

import importlib
from datetime import date, timedelta

import polars as pl

import modeling_module as mm
from modeling_module import (
    ArtifactConfig,
    DataRequest,
    DataWindowConfig,
    RuntimeConfig,
    TrainRequest,
    TrainerConfig,
    train,
)


def _make_daily_df(n_rows: int = 24) -> pl.DataFrame:
    rows = []
    start = date(2024, 1, 1)
    for uid in ("A", "B"):
        current = start
        for idx in range(n_rows):
            rows.append(
                {
                    "unique_id": uid,
                    "date": int(current.strftime("%Y%m%d")),
                    "y": float(idx + 1),
                }
            )
            current += timedelta(days=1)
    return pl.DataFrame(rows)


def _make_request(models, tmp_path) -> TrainRequest:
    return TrainRequest(
        data=DataRequest(
            df=_make_daily_df(),
            window=DataWindowConfig(lookback=14, horizon=2, freq="daily"),
        ),
        models=models,
        trainer=TrainerConfig(epochs=1, lr=1e-3),
        runtime=RuntimeConfig(device="cpu"),
        artifacts=ArtifactConfig(save_dir=str(tmp_path), auto_save_dir=False),
    )


def test_public_api_exports_official_surface():
    expected = {
        "train",
        "load_predictor",
        "predict",
        "build_dataset",
        "build_dataloader",
        "TrainRequest",
        "TrainResult",
        "DataRequest",
        "TrainerConfig",
        "SSLConfig",
        "RuntimeConfig",
        "ArtifactConfig",
        "DataWindowConfig",
        "DataColumnConfig",
        "ExogenousConfig",
        "LoaderConfig",
    }

    for name in expected:
        assert hasattr(mm, name), f"missing public API export: {name}"


def test_train_result_populates_primary_fields_for_single_model(monkeypatch, tmp_path):
    train_module = importlib.import_module("modeling_module.api.train")

    def fake_run_total_train(train_loader, val_loader, **kwargs):
        return {
            "PatchTST": {
                "ckpt_path": str(tmp_path / "patchtst.pt"),
                "model_key": "patchtst_base",
                "family_key": "patchtst",
            }
        }

    monkeypatch.setattr(train_module, "run_total_train", fake_run_total_train)

    result = train(_make_request(models=["patchtst_base"], tmp_path=tmp_path))

    assert result.requested_models == ("patchtst_base",)
    assert result.ckpt_paths == {"patchtst_base": str(tmp_path / "patchtst.pt")}
    assert result.primary_result_name == "patchtst_base"
    assert result.primary_ckpt_path == str(tmp_path / "patchtst.pt")
    assert result.best_ckpt_path == str(tmp_path / "patchtst.pt")


def test_train_result_leaves_primary_fields_empty_for_multi_model(monkeypatch, tmp_path):
    train_module = importlib.import_module("modeling_module.api.train")

    def fake_run_total_train(train_loader, val_loader, **kwargs):
        return {
            "PatchTST": {
                "ckpt_path": str(tmp_path / "patchtst.pt"),
                "model_key": "patchtst_base",
                "family_key": "patchtst",
            },
            "Titan": {
                "ckpt_path": str(tmp_path / "titan.pt"),
                "model_key": "titan_base",
                "family_key": "titan",
            },
        }

    monkeypatch.setattr(train_module, "run_total_train", fake_run_total_train)

    result = train(_make_request(models=["patchtst_base", "titan_base"], tmp_path=tmp_path))

    assert result.requested_models == ("patchtst_base", "titan_base")
    assert result.ckpt_paths == {
        "patchtst_base": str(tmp_path / "patchtst.pt"),
        "titan_base": str(tmp_path / "titan.pt"),
    }
    assert result.primary_result_name is None
    assert result.primary_ckpt_path is None
    assert result.best_ckpt_path is None

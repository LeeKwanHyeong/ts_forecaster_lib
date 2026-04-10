from __future__ import annotations

import importlib
from datetime import date, timedelta

import polars as pl

from modeling_module import (
    ArtifactConfig,
    DataColumnConfig,
    DataRequest,
    DataWindowConfig,
    RuntimeConfig,
    TrainRequest,
    TrainerConfig,
    train,
)


def _make_daily_df(n_rows: int = 40) -> pl.DataFrame:
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


def _make_request(model_name: str, tmp_path) -> TrainRequest:
    return TrainRequest(
        data=DataRequest(
            df=_make_daily_df(),
            window=DataWindowConfig(lookback=14, horizon=2, freq="daily"),
            columns=DataColumnConfig(id_col="unique_id", date_col="date", y_col="y"),
        ),
        models=model_name,
        trainer=TrainerConfig(epochs=1, lr=1e-3),
        runtime=RuntimeConfig(device="cpu"),
        artifacts=ArtifactConfig(save_dir=str(tmp_path), auto_save_dir=False),
    )


def test_train_preserves_single_titan_artifact_selection(monkeypatch, tmp_path):
    train_module = importlib.import_module("modeling_module.api.train")
    captured: dict[str, object] = {}

    def fake_run_total_train(train_loader, val_loader, **kwargs):
        captured["models_to_run"] = kwargs["models_to_run"]
        return {
            "Titan LMM": {
                "ckpt_path": str(tmp_path / "titan_lmm.pt"),
                "model_key": "titan_lmm",
                "family_key": "titan",
            }
        }

    monkeypatch.setattr(train_module, "run_total_train", fake_run_total_train)

    result = train(_make_request("titan_lmm", tmp_path))

    assert captured["models_to_run"] == ["titan_lmm"]
    assert result.requested_models == ("titan_lmm",)


def test_train_preserves_single_quantile_artifact_selection(monkeypatch, tmp_path):
    train_module = importlib.import_module("modeling_module.api.train")
    captured: dict[str, object] = {}

    def fake_run_total_train(train_loader, val_loader, **kwargs):
        captured["models_to_run"] = kwargs["models_to_run"]
        return {
            "PatchTST Quantile": {
                "ckpt_path": str(tmp_path / "patchtst_quantile.pt"),
                "model_key": "patchtst_quantile",
                "family_key": "patchtst",
            }
        }

    monkeypatch.setattr(train_module, "run_total_train", fake_run_total_train)

    result = train(_make_request("patchtst_quantile", tmp_path))

    assert captured["models_to_run"] == ["patchtst_quantile"]
    assert result.requested_models == ("patchtst_quantile",)


def test_train_expands_titan_family_to_all_titan_artifacts(monkeypatch, tmp_path):
    train_module = importlib.import_module("modeling_module.api.train")
    captured: dict[str, object] = {}

    def fake_run_total_train(train_loader, val_loader, **kwargs):
        captured["models_to_run"] = kwargs["models_to_run"]
        return {
            "Titan Base": {
                "ckpt_path": str(tmp_path / "titan_base.pt"),
                "model_key": "titan_base",
                "family_key": "titan",
            }
        }

    monkeypatch.setattr(train_module, "run_total_train", fake_run_total_train)

    result = train(_make_request("titan", tmp_path))

    assert captured["models_to_run"] == ["titan_base", "titan_lmm", "titan_seq2seq"]
    assert result.requested_models == ("titan_base", "titan_lmm", "titan_seq2seq")

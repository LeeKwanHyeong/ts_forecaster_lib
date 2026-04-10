from __future__ import annotations

import importlib
from datetime import date, timedelta

import polars as pl
import torch

from modeling_module import (
    ArtifactConfig,
    DataColumnConfig,
    DataRequest,
    DataWindowConfig,
    ExogenousConfig,
    LoaderConfig,
    RuntimeConfig,
    SSLConfig,
    TrainRequest,
    TrainerConfig,
    build_dataloader,
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


def _make_daily_df_with_future_covariates() -> pl.DataFrame:
    rows = []
    start = date(2024, 1, 1)
    values = [float(i) for i in range(1, 11)] + [None, None]
    promo = [float(i % 2) for i in range(10)] + [1.0, 1.0]
    holiday = [0.0] * 10 + [1.0, 1.0]

    for uid in ("A", "B"):
        current = start
        for y, promo_flag, holiday_flag in zip(values, promo, holiday):
            rows.append(
                {
                    "unique_id": uid,
                    "date": int(current.strftime("%Y%m%d")),
                    "y": y,
                    "promo_flag": promo_flag,
                    "holiday_flag": holiday_flag,
                }
            )
            current += timedelta(days=1)
    return pl.DataFrame(rows)


def test_build_dataloader_accepts_nested_data_config_dataclasses():
    loader = build_dataloader(
        DataRequest(
            df=_make_daily_df_with_future_covariates(),
            window=DataWindowConfig(lookback=2, horizon=2, freq="daily"),
            exogenous=ExogenousConfig(future_exo_cont_cols=["promo_flag", "holiday_flag"]),
            loader=LoaderConfig(stage="train", batch_size=1, drop_last=False),
        )
    )

    batch = next(iter(loader))
    assert torch.is_tensor(batch[3])
    assert batch[3].shape == (1, 2, 2)


def test_train_accepts_compact_nested_configs(monkeypatch, tmp_path):
    train_module = importlib.import_module("modeling_module.api.train")
    captured: dict[str, object] = {}

    def fake_run_total_train(train_loader, val_loader, **kwargs):
        captured.update(kwargs)
        return {
            "PatchTST": {
                "ckpt_path": str(tmp_path / "patchtst.pt"),
                "pretrain_ckpt_path": str(tmp_path / "pretrain" / "patchtst_pretrain_best.pt"),
                "model_key": "patchtst_base",
                "family_key": "patchtst",
            }
        }

    monkeypatch.setattr(train_module, "run_total_train", fake_run_total_train)

    req = TrainRequest(
        data=DataRequest(
            df=_make_daily_df(),
            window=DataWindowConfig(lookback=14, horizon=2, freq="daily"),
            columns=DataColumnConfig(id_col="unique_id", date_col="date", y_col="y"),
            loader=LoaderConfig(batch_size=4),
        ),
        models=["patchtst_base"],
        trainer=TrainerConfig(epochs=1, lr=1e-3),
        ssl=SSLConfig(mode="full", pretrain_epochs=2, mask_ratio=0.4),
        runtime=RuntimeConfig(device="cpu"),
        artifacts=ArtifactConfig(save_dir=str(tmp_path), auto_save_dir=False),
    )

    result = train(req)

    assert captured["freq"] == "daily"
    assert captured["lookback"] == 14
    assert captured["horizon"] == 2
    assert captured["device"] == "cpu"
    assert captured["use_ssl_mode"] == "full"
    assert captured["ssl_pretrain_epochs"] == 2
    assert captured["ssl_mask_ratio"] == 0.4
    assert result.requested_models == ("patchtst_base",)
    assert result.ckpt_paths["patchtst_base"].endswith("patchtst.pt")

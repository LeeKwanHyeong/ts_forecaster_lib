from __future__ import annotations

import importlib
from datetime import date, timedelta

import polars as pl
import pytest
import torch

from modeling_module import (
    ArtifactConfig,
    ArchitectureConfig,
    DataColumnConfig,
    DataRequest,
    DataWindowConfig,
    ExogenousConfig,
    LoaderConfig,
    PatchTSTArchitectureConfig,
    RuntimeConfig,
    SELLMArchitectureConfig,
    SSLConfig,
    TimexerArchitectureConfig,
    TitanArchitectureConfig,
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
        ssl=SSLConfig(
            mode="full",
            pretrain_epochs=2,
            pretrain_stride=13,
            mask_ratio=0.4,
        ),
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
    assert captured["ssl_pretrain_stride"] == 13
    assert captured["ssl_mask_ratio"] == 0.4
    assert result.requested_models == ("patchtst_base",)
    assert result.ckpt_paths["patchtst_base"].endswith("patchtst.pt")


def test_train_honors_loader_runtime_kwargs(monkeypatch, tmp_path):
    train_module = importlib.import_module("modeling_module.api.train")
    captured: dict[str, object] = {}

    class FakeDataModule:
        def get_train_loader(
            self,
            batch_size=None,
            shuffle=None,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        ):
            captured["train_loader_kwargs"] = {
                "batch_size": batch_size,
                "shuffle": shuffle,
                "drop_last": drop_last,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "persistent_workers": persistent_workers,
                "prefetch_factor": prefetch_factor,
            }
            return "TRAIN_LOADER"

        def get_val_loader(
            self,
            batch_size=None,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        ):
            captured["val_loader_kwargs"] = {
                "batch_size": batch_size,
                "drop_last": drop_last,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "persistent_workers": persistent_workers,
                "prefetch_factor": prefetch_factor,
            }
            return "VAL_LOADER"

    def fake_build_datamodule(_cfg):
        return FakeDataModule()

    def fake_run_total_train(train_loader, val_loader, **kwargs):
        captured["train_loader"] = train_loader
        captured["val_loader"] = val_loader
        captured["run_kwargs"] = dict(kwargs)
        return {
            "TitanBase": {
                "ckpt_path": str(tmp_path / "titan_base.pt"),
                "model_key": "titan_base",
                "family_key": "titan",
            }
        }

    monkeypatch.setattr(train_module, "build_datamodule", fake_build_datamodule)
    monkeypatch.setattr(train_module, "run_total_train", fake_run_total_train)

    req = TrainRequest(
        data=DataRequest(
            df=_make_daily_df(),
            window=DataWindowConfig(lookback=14, horizon=2, freq="daily"),
            columns=DataColumnConfig(id_col="unique_id", date_col="date", y_col="y"),
            loader=LoaderConfig(
                batch_size=32,
                shuffle=False,
                num_workers=6,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=5,
                drop_last=False,
            ),
        ),
        models=["titan_base"],
        trainer=TrainerConfig(epochs=1, lr=1e-3),
        runtime=RuntimeConfig(device="cpu"),
        artifacts=ArtifactConfig(save_dir=str(tmp_path), auto_save_dir=False),
    )

    result = train(req)

    assert captured["train_loader"] == "TRAIN_LOADER"
    assert captured["val_loader"] == "VAL_LOADER"

    assert captured["train_loader_kwargs"] == {
        "batch_size": 32,
        "shuffle": False,
        "drop_last": False,
        "num_workers": 6,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 5,
    }
    assert captured["val_loader_kwargs"] == {
        "batch_size": 32,
        "drop_last": False,
        "num_workers": 6,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 5,
    }
    assert result.requested_models == ("titan_base",)
    assert result.ckpt_paths["titan_base"].endswith("titan_base.pt")


def test_train_accepts_family_architecture_overrides(monkeypatch, tmp_path):
    train_module = importlib.import_module("modeling_module.api.train")
    captured: dict[str, object] = {}

    def fake_run_total_train(train_loader, val_loader, **kwargs):
        captured.update(kwargs)
        return {
            "PatchTST": {
                "ckpt_path": str(tmp_path / "patchtst.pt"),
                "model_key": "patchtst_base",
                "family_key": "patchtst",
            }
        }

    monkeypatch.setattr(train_module, "run_total_train", fake_run_total_train)

    req = TrainRequest(
        data=DataRequest(
            df=_make_daily_df(),
            window=DataWindowConfig(lookback=14, horizon=2, freq="daily"),
        ),
        models=["patchtst_base"],
        trainer=TrainerConfig(epochs=1, lr=1e-3),
        architecture=ArchitectureConfig(
            patchtst=PatchTSTArchitectureConfig(
                patch_len=12,
                stride=6,
                d_model=384,
                n_layers=5,
            ),
            sellm=SELLMArchitectureConfig(
                architecture_variant="paper_v1",
                llm_source="local",
                llm_local_path="/models/Qwen2-0.5B",
                output_head_mode="zero_inflated_softplus",
                output_head_hidden_dim=16,
            ),
            titan=TitanArchitectureConfig(d_model=512),
        ),
        runtime=RuntimeConfig(device="cpu"),
        artifacts=ArtifactConfig(save_dir=str(tmp_path), auto_save_dir=False),
    )

    train(req)

    assert captured["model_architecture"] == {
        "patchtst": {
            "patch_len": 12,
            "stride": 6,
            "d_model": 384,
            "n_layers": 5,
        },
        "titan": {
            "d_model": 512,
        },
        "sellm": {
            "architecture_variant": "paper_v1",
            "llm_source": "local",
            "llm_local_path": "/models/Qwen2-0.5B",
            "output_head_mode": "zero_inflated_softplus",
            "output_head_hidden_dim": 16,
        },
    }


def test_train_validates_patch_len_from_architecture_override(tmp_path):
    req = TrainRequest(
        data=DataRequest(
            df=_make_daily_df(),
            window=DataWindowConfig(lookback=14, horizon=2, freq="daily"),
        ),
        models=["patchtst_base"],
        trainer=TrainerConfig(epochs=1, lr=1e-3),
        architecture=ArchitectureConfig(
            patchtst=PatchTSTArchitectureConfig(
                patch_len=16,
            )
        ),
        runtime=RuntimeConfig(device="cpu"),
        artifacts=ArtifactConfig(save_dir=str(tmp_path), auto_save_dir=False),
    )

    with pytest.raises(ValueError, match="lookback=14 is too short"):
        train(req)


def test_train_accepts_timexer_architecture_overrides(monkeypatch, tmp_path):
    train_module = importlib.import_module("modeling_module.api.train")
    captured: dict[str, object] = {}

    def fake_run_total_train(train_loader, val_loader, **kwargs):
        captured.update(kwargs)
        return {
            "TimeXer": {
                "ckpt_path": str(tmp_path / "timexer.pt"),
                "model_key": "timexer_base",
                "family_key": "timexer",
            }
        }

    monkeypatch.setattr(train_module, "run_total_train", fake_run_total_train)
    monkeypatch.setattr(train_module, "_validate_training_request", lambda **kwargs: None)

    req = TrainRequest(
        data=DataRequest(
            df=_make_daily_df(),
            window=DataWindowConfig(lookback=14, horizon=2, freq="daily"),
            exogenous=ExogenousConfig(use_exogenous_mode=True, past_exo_cont_cols=["exo"]),
        ),
        models=["timexer_base"],
        trainer=TrainerConfig(epochs=1, lr=1e-3),
        architecture=ArchitectureConfig(
            timexer=TimexerArchitectureConfig(
                patch_len=7,
                d_model=192,
                e_layers=2,
                use_norm=False,
            )
        ),
        runtime=RuntimeConfig(device="cpu"),
        artifacts=ArtifactConfig(save_dir=str(tmp_path), auto_save_dir=False),
    )

    # A tiny exogenous column is enough because validation is stubbed in this normalization test.
    req.data.df = req.data.df.with_columns(pl.lit(1.0).alias("exo"))

    train(req)

    assert captured["model_architecture"] == {
        "timexer": {
            "patch_len": 7,
            "d_model": 192,
            "e_layers": 2,
            "use_norm": False,
        }
    }

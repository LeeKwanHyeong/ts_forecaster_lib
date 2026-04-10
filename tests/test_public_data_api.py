import polars as pl
import torch

from modeling_module import build_dataloader, build_dataset


def _make_daily_df() -> pl.DataFrame:
    rows = []
    for uid in ("A", "B"):
        for idx, y in enumerate([1.0, 2.0, 3.0, 4.0, 5.0], start=1):
            rows.append(
                {
                    "unique_id": uid,
                    "date": 20240100 + idx,
                    "y": y,
                }
            )
    return pl.DataFrame(rows)


def _make_daily_df_with_known_future() -> pl.DataFrame:
    rows = []
    values = [1.0, 2.0, 3.0, 4.0, 5.0, None, None]
    promo = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    holiday = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]

    for uid in ("A", "B"):
        for idx, (y, promo_flag, holiday_flag) in enumerate(zip(values, promo, holiday), start=1):
            rows.append(
                {
                    "unique_id": uid,
                    "date": 20240100 + idx,
                    "y": y,
                    "promo_flag": promo_flag,
                    "holiday_flag": holiday_flag,
                }
            )
    return pl.DataFrame(rows)


def test_build_dataset_smoke():
    ds = build_dataset(
        {
            "df": _make_daily_df(),
            "lookback": 2,
            "horizon": 1,
            "freq": "daily",
            "stage": "train",
        }
    )
    assert len(ds) > 0


def test_build_dataloader_smoke():
    loader = build_dataloader(
        {
            "df": _make_daily_df(),
            "lookback": 2,
            "horizon": 1,
            "freq": "daily",
            "stage": "train",
            "batch_size": 2,
        }
    )
    batch = next(iter(loader))
    assert torch.is_tensor(batch[0])
    assert batch[0].ndim == 3


def test_build_dataloader_with_single_table_future_covariates():
    loader = build_dataloader(
        {
            "df": _make_daily_df_with_known_future(),
            "lookback": 2,
            "horizon": 2,
            "freq": "daily",
            "stage": "train",
            "batch_size": 2,
            "future_exo_cont_cols": ["promo_flag", "holiday_flag"],
        }
    )

    batch = next(iter(loader))
    assert torch.is_tensor(batch[3])
    assert batch[3].shape == (2, 2, 2)


def test_build_inference_dataloader_with_single_table_future_covariates():
    loader = build_dataloader(
        {
            "df": _make_daily_df_with_known_future(),
            "lookback": 2,
            "horizon": 2,
            "freq": "daily",
            "stage": "inference",
            "plan_dt": 20240106,
            "batch_size": 2,
            "future_exo_cont_cols": ["promo_flag", "holiday_flag"],
        }
    )

    batch = next(iter(loader))
    assert torch.is_tensor(batch[3])
    assert batch[3].shape[-1] == 2

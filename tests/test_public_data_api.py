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


def _make_daily_df_with_known_future_categories() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 8,
            "date": [20240101 + index for index in range(8)],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, None, None],
            "event_type": [
                "base",
                "promo",
                "base",
                "promo",
                "base",
                "promo",
                "promo",
                "unseen-event",
            ],
        }
    )


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
    assert len(ds[0]) == 6


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


def test_build_inference_dataset_with_known_and_unknown_future_categories():
    request = {
        "df": _make_daily_df_with_known_future_categories(),
        "lookback": 2,
        "horizon": 2,
        "freq": "daily",
        "stage": "inference",
        "plan_dt": 20240107,
        "val_ratio": 0.0,
        "future_exo_cat_cols": ["event_type"],
    }
    dataset = build_dataset(request)

    sample = dataset[0]
    future_cat = sample[6]
    assert len(sample) == 7
    assert future_cat.dtype == torch.long
    assert future_cat.shape == (2, 1)
    assert int(future_cat[0, 0]) > 0
    assert int(future_cat[1, 0]) == 0

    batch = next(iter(build_dataloader(request)))
    assert len(batch) == 7
    assert batch[6].dtype == torch.long
    assert batch[6].shape == (1, 2, 1)
    assert int(batch[6][0, 0, 0]) > 0
    assert int(batch[6][0, 1, 0]) == 0


def test_build_training_dataloader_with_future_categories():
    batch = next(
        iter(
            build_dataloader(
                {
                    "df": _make_daily_df_with_known_future_categories(),
                    "lookback": 2,
                    "horizon": 2,
                    "freq": "daily",
                    "stage": "train",
                    "batch_size": 2,
                    "shuffle": False,
                    "val_ratio": 0.0,
                    "future_exo_cat_cols": ["event_type"],
                }
            )
        )
    )

    assert len(batch) == 7
    assert batch[6].dtype == torch.long
    assert batch[6].shape == (2, 2, 1)
    assert bool(torch.all(batch[6] > 0))

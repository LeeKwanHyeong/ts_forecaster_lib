import polars as pl
import pytest

from modeling_module import train


def _make_daily_df(n_rows: int = 10) -> pl.DataFrame:
    rows = []
    for uid in ("A", "B"):
        for idx in range(1, n_rows + 1):
            rows.append(
                {
                    "unique_id": uid,
                    "date": 20240100 + idx,
                    "y": float(idx),
                }
            )
    return pl.DataFrame(rows)


def test_train_rejects_patch_models_with_short_lookback(tmp_path):
    with pytest.raises(ValueError, match="lookback=2.*patch_len=14"):
        train(
            {
                "data": {
                    "df": _make_daily_df(),
                    "lookback": 2,
                    "horizon": 1,
                    "freq": "daily",
                    "batch_size": 2,
                },
                "models": ["patchtst_base"],
                "trainer": {"epochs": 1, "lr": 1e-3},
                "device": "cpu",
                "save_dir": str(tmp_path),
                "auto_save_dir": False,
            }
        )


def test_train_rejects_exotst_without_exogenous_mode(tmp_path):
    with pytest.raises(ValueError, match="ExoTST requires use_exogenous_mode=True"):
        train(
            {
                "data": {
                    "df": _make_daily_df(n_rows=30),
                    "lookback": 14,
                    "horizon": 2,
                    "freq": "daily",
                    "batch_size": 2,
                },
                "models": ["exotst_base"],
                "use_exogenous_mode": False,
                "trainer": {"epochs": 1, "lr": 1e-3},
                "device": "cpu",
                "save_dir": str(tmp_path),
                "auto_save_dir": False,
            }
        )

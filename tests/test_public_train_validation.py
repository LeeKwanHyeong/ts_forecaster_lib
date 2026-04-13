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


def _make_daily_df_with_past_exo(n_rows: int = 30) -> pl.DataFrame:
    rows = []
    for uid in ("A", "B"):
        for idx in range(1, n_rows + 1):
            rows.append(
                {
                    "unique_id": uid,
                    "date": 20240100 + idx,
                    "y": float(idx),
                    "exo_hist": float(idx % 3),
                }
            )
    return pl.DataFrame(rows)


def _make_daily_df_with_future_exo(n_rows: int = 30) -> pl.DataFrame:
    rows = []
    for uid in ("A", "B"):
        for idx in range(1, n_rows + 1):
            rows.append(
                {
                    "unique_id": uid,
                    "date": 20240100 + idx,
                    "y": float(idx) if idx <= (n_rows - 2) else None,
                    "exo_hist": float(idx % 3),
                    "promo_flag": float(idx % 2),
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


def test_train_rejects_timexer_without_exogenous_mode(tmp_path):
    with pytest.raises(ValueError, match="TimeXer requires use_exogenous_mode=True"):
        train(
            {
                "data": {
                    "df": _make_daily_df_with_past_exo(),
                    "lookback": 14,
                    "horizon": 2,
                    "freq": "daily",
                    "batch_size": 2,
                    "past_exo_cont_cols": ["exo_hist"],
                },
                "models": ["timexer_base"],
                "use_exogenous_mode": False,
                "trainer": {"epochs": 1, "lr": 1e-3},
                "device": "cpu",
                "save_dir": str(tmp_path),
                "auto_save_dir": False,
            }
        )


def test_train_rejects_timexer_without_past_exogenous(tmp_path):
    with pytest.raises(ValueError, match="TimeXer requires past continuous exogenous inputs"):
        train(
            {
                "data": {
                    "df": _make_daily_df(n_rows=30),
                    "lookback": 14,
                    "horizon": 2,
                    "freq": "daily",
                    "batch_size": 2,
                },
                "models": ["timexer_base"],
                "use_exogenous_mode": True,
                "trainer": {"epochs": 1, "lr": 1e-3},
                "device": "cpu",
                "save_dir": str(tmp_path),
                "auto_save_dir": False,
            }
        )


def test_train_rejects_timexer_with_future_exogenous(tmp_path):
    with pytest.raises(ValueError, match="does not support future exogenous inputs"):
        train(
            {
                "data": {
                    "df": _make_daily_df_with_future_exo(),
                    "lookback": 14,
                    "horizon": 2,
                    "freq": "daily",
                    "batch_size": 2,
                    "past_exo_cont_cols": ["exo_hist"],
                    "future_exo_cont_cols": ["promo_flag"],
                },
                "models": ["timexer_base"],
                "use_exogenous_mode": True,
                "trainer": {"epochs": 1, "lr": 1e-3},
                "device": "cpu",
                "save_dir": str(tmp_path),
                "auto_save_dir": False,
            }
        )

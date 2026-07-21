import importlib

import polars as pl
import pytest

from modeling_module import (
    ArtifactConfig,
    DistributionLoss,
    SSLConfig,
    TrainRequest,
    train,
)


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


def _make_daily_df_with_categorical_exo(n_rows: int = 30) -> pl.DataFrame:
    return _make_daily_df_with_future_exo(n_rows).with_columns(
        pl.Series("segment_id", [idx % 3 for idx in range(2 * n_rows)], dtype=pl.Int64)
    )


@pytest.mark.parametrize("ssl_mode", ["full", "ssl_only"])
def test_train_rejects_ssl_mode_without_artifact_directory_before_data_resolution(
    monkeypatch,
    ssl_mode,
):
    train_module = importlib.import_module("modeling_module.api.train")
    reached_data_resolution = False

    def unexpected_resolve_loaders(payload):
        nonlocal reached_data_resolution
        reached_data_resolution = True
        raise AssertionError("data resolution must not run")

    monkeypatch.setattr(train_module, "_resolve_loaders", unexpected_resolve_loaders)

    request = TrainRequest(
        models=["patchtst_base"],
        ssl=SSLConfig(mode=ssl_mode),
        artifacts=ArtifactConfig(save_dir=None, auto_save_dir=False),
    )

    with pytest.raises(ValueError) as exc_info:
        train(request)

    assert str(exc_info.value) == (
        f"PatchTST SSL mode {ssl_mode!r} requires an artifact `save_dir`. "
        "Provide `artifacts.save_dir` or enable `artifacts.auto_save_dir`."
    )
    assert reached_data_resolution is False


def test_train_allows_sl_only_without_artifact_directory(monkeypatch):
    train_module = importlib.import_module("modeling_module.api.train")
    marker = RuntimeError("data-resolution-reached")

    def stop_at_data_resolution(payload):
        raise marker

    monkeypatch.setattr(train_module, "_resolve_loaders", stop_at_data_resolution)

    request = TrainRequest(
        models=["patchtst_base"],
        ssl=SSLConfig(mode="sl_only"),
        artifacts=ArtifactConfig(save_dir=None, auto_save_dir=False),
    )

    with pytest.raises(RuntimeError, match="data-resolution-reached") as exc_info:
        train(request)

    assert exc_info.value is marker


@pytest.mark.parametrize("ssl_mode", ["full", "ssl_only"])
def test_train_rejects_patchtst_ssl_mode_for_non_patchtst_request_before_data_resolution(
    monkeypatch,
    tmp_path,
    ssl_mode,
):
    train_module = importlib.import_module("modeling_module.api.train")
    reached_data_resolution = False

    def unexpected_resolve_loaders(payload):
        nonlocal reached_data_resolution
        reached_data_resolution = True
        raise AssertionError("data resolution must not run")

    monkeypatch.setattr(train_module, "_resolve_loaders", unexpected_resolve_loaders)

    request = TrainRequest(
        models=["titan_base"],
        ssl=SSLConfig(mode=ssl_mode),
        artifacts=ArtifactConfig(save_dir=str(tmp_path), auto_save_dir=False),
    )

    with pytest.raises(ValueError, match="requires at least one PatchTST artifact.*titan_base"):
        train(request)

    assert reached_data_resolution is False


@pytest.mark.parametrize("ssl_mode", ["full", "ssl_only"])
@pytest.mark.parametrize("missing_save_dir", [None, ""])
def test_internal_total_train_rejects_patchtst_ssl_without_save_dir_before_loader_use(
    ssl_mode,
    missing_save_dir,
):
    total_train_module = importlib.import_module(
        "modeling_module.training.model_trainers.total_train"
    )

    with pytest.raises(ValueError, match="requires an artifact `save_dir`"):
        total_train_module.run_total_train(
            object(),
            object(),
            freq="monthly",
            lookback=2,
            horizon=1,
            device="cpu",
            save_dir=missing_save_dir,
            models_to_run=["patchtst_base"],
            use_ssl_mode=ssl_mode,
        )


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


def test_train_rejects_timexer_with_categorical_exogenous_input(tmp_path):
    with pytest.raises(ValueError, match="TimeXer v1 does not consume past categorical"):
        train(
            {
                "data": {
                    "df": _make_daily_df_with_categorical_exo(),
                    "lookback": 14,
                    "horizon": 2,
                    "freq": "daily",
                    "batch_size": 2,
                    "pin_memory": False,
                    "past_exo_cont_cols": ["exo_hist"],
                    "past_exo_cat_cols": ["segment_id"],
                },
                "models": ["timexer_base"],
                "use_exogenous_mode": True,
                "use_past_exogenous": True,
                "use_future_exogenous": False,
                "trainer": {"epochs": 1, "lr": 1e-3},
                "device": "cpu",
                "save_dir": str(tmp_path / "timexer-categorical"),
                "auto_save_dir": False,
            }
        )


def test_train_rejects_patchmixer_original_distribution_before_data_resolution(
    monkeypatch,
):
    train_module = importlib.import_module("modeling_module.api.train")
    reached_data_resolution = False

    def unexpected_resolve_loaders(payload):
        nonlocal reached_data_resolution
        reached_data_resolution = True
        raise AssertionError("data resolution must not run")

    monkeypatch.setattr(train_module, "_resolve_loaders", unexpected_resolve_loaders)

    with pytest.raises(ValueError, match="patchmixer_original supports point loss only"):
        train(
            TrainRequest(
                models=["patchmixer_original"],
                trainer={"loss": DistributionLoss("Normal")},
                artifacts=ArtifactConfig(save_dir=None, auto_save_dir=False),
            )
        )

    assert reached_data_resolution is False


def test_train_rejects_patchmixer_original_exogenous_inputs_before_training(
    monkeypatch,
    tmp_path,
):
    train_module = importlib.import_module("modeling_module.api.train")
    reached_training = False

    def unexpected_training(*args, **kwargs):
        nonlocal reached_training
        reached_training = True
        raise AssertionError("model construction and training must not run")

    monkeypatch.setattr(train_module, "run_total_train", unexpected_training)

    with pytest.raises(ValueError, match="canonical upstream baseline supports endogenous"):
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
                "models": ["patchmixer_original"],
                "use_exogenous_mode": True,
                "trainer": {"epochs": 1, "lr": 1e-3},
                "device": "cpu",
                "save_dir": str(tmp_path / "patchmixer-original-exogenous"),
                "auto_save_dir": False,
            }
        )

    assert reached_training is False


@pytest.mark.parametrize(
    "model_key",
    [
        "patchtst_exogenous",
        "patchtst_quantile_exogenous",
        "patchmixer_exogenous",
        "patchmixer_quantile_exogenous",
    ],
)
def test_train_rejects_explicit_exogenous_models_without_exogenous_mode(
    monkeypatch,
    tmp_path,
    model_key,
):
    train_module = importlib.import_module("modeling_module.api.train")
    reached_training = False

    def unexpected_training(*args, **kwargs):
        nonlocal reached_training
        reached_training = True
        raise AssertionError("model construction and training must not run")

    monkeypatch.setattr(train_module, "run_total_train", unexpected_training)

    with pytest.raises(ValueError, match="explicit exogenous models require use_exogenous_mode=True"):
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
                "models": [model_key],
                "use_exogenous_mode": False,
                "trainer": {"epochs": 1, "lr": 1e-3},
                "device": "cpu",
                "save_dir": str(tmp_path / model_key),
                "auto_save_dir": False,
            }
        )

    assert reached_training is False


@pytest.mark.parametrize(
    "model_key",
    [
        "patchtst_exogenous",
        "patchtst_quantile_exogenous",
        "patchmixer_exogenous",
        "patchmixer_quantile_exogenous",
    ],
)
def test_train_rejects_explicit_exogenous_models_without_features(
    monkeypatch,
    tmp_path,
    model_key,
):
    train_module = importlib.import_module("modeling_module.api.train")
    reached_training = False

    def unexpected_training(*args, **kwargs):
        nonlocal reached_training
        reached_training = True
        raise AssertionError("model construction and training must not run")

    monkeypatch.setattr(train_module, "run_total_train", unexpected_training)

    with pytest.raises(ValueError, match="at least one past or future exogenous feature is required"):
        train(
            {
                "data": {
                    "df": _make_daily_df(n_rows=30),
                    "lookback": 14,
                    "horizon": 2,
                    "freq": "daily",
                    "batch_size": 2,
                },
                "models": [model_key],
                "use_exogenous_mode": True,
                "trainer": {"epochs": 1, "lr": 1e-3},
                "device": "cpu",
                "save_dir": str(tmp_path / model_key),
                "auto_save_dir": False,
            }
        )

    assert reached_training is False


@pytest.mark.parametrize(
    "model_key",
    ["patchtst_base", "patchmixer_base", "titan_base", "exotst_base"],
)
def test_train_rejects_categorical_exogenous_inputs_before_model_construction(
    monkeypatch,
    tmp_path,
    model_key,
):
    train_module = importlib.import_module("modeling_module.api.train")
    reached_training = False

    def unexpected_training(*args, **kwargs):
        nonlocal reached_training
        reached_training = True
        raise AssertionError("model construction and training must not run")

    monkeypatch.setattr(train_module, "run_total_train", unexpected_training)

    with pytest.raises(
        ValueError,
        match=rf"categorical past exogenous inputs are not supported.*{model_key}",
    ):
        train(
            {
                "data": {
                    "df": _make_daily_df_with_categorical_exo(),
                    "lookback": 14,
                    "horizon": 2,
                    "freq": "daily",
                    "batch_size": 2,
                    "pin_memory": False,
                    "past_exo_cont_cols": ["exo_hist"],
                    "past_exo_cat_cols": ["segment_id"],
                    "future_exo_cont_cols": ["promo_flag"],
                },
                "models": [model_key],
                "use_exogenous_mode": True,
                "use_past_exogenous": True,
                "use_future_exogenous": True,
                "trainer": {"epochs": 1, "lr": 1e-3},
                "device": "cpu",
                "save_dir": str(tmp_path / model_key),
                "auto_save_dir": False,
            }
        )

    assert reached_training is False

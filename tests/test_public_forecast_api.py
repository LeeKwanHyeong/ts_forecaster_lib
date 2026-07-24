"""End-to-end tests for the high-level public anchored forecast API."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import torch
from polars.testing import assert_frame_equal

from modeling_module import (
    DataRequest,
    ExogenousConfig,
    ForecastRequest,
    ForecastResult,
    ForecastRuntimeConfig,
    forecast,
)
from modeling_module.models.PatchTST.common.configs import AttentionConfig, PatchTSTConfig
from modeling_module.models.model_builder import (
    build_patchTST,
    build_patchTST_exogenous,
    build_patchTST_quantile,
)
from modeling_module.utils.checkpoint import save_model


EXPECTED_SCHEMA = pl.Schema(
    {
        "series_id": pl.String,
        "model_key": pl.String,
        "forecast_origin": pl.Int64,
        "horizon_step": pl.Int32,
        "point": pl.Float64,
        "q10": pl.Float64,
        "q50": pl.Float64,
        "q90": pl.Float64,
    }
)


def _make_daily_frame(series_ids: tuple[str, ...] = ("A", "B", "C")) -> pl.DataFrame:
    """Build eight observed daily periods for each requested series."""
    rows: list[dict[str, Any]] = []
    for series_ordinal, series_id in enumerate(series_ids):
        for day in range(1, 9):
            rows.append(
                {
                    "unique_id": series_id,
                    "date": 20240100 + day,
                    "y": float(series_ordinal * 100 + day),
                }
            )
    return pl.DataFrame(rows)


def _make_point_checkpoint(path: Path) -> None:
    """Write a tiny supported point-model checkpoint for API testing."""
    config = PatchTSTConfig(
        lookback=8,
        horizon=2,
        patch_len=4,
        stride=2,
        d_model=16,
        d_ff=32,
        n_layers=1,
        future_exo_dim=0,
        past_exo_cont_dim=0,
        past_exo_cat_dim=0,
        use_exogenous_mode=False,
        use_revin=False,
        attn=AttentionConfig(
            n_heads=4,
            d_model=16,
            attn_dropout=0.0,
            proj_dropout=0.0,
        ),
    )
    model = build_patchTST(config)
    save_model(
        model,
        config,
        str(path),
        extra_meta={"model_key": "patchtst_base", "family_key": "patchtst"},
    )


def _make_quantile_checkpoint(path: Path) -> None:
    """Write a tiny supported quantile-model checkpoint for API testing."""
    config = PatchTSTConfig(
        lookback=8,
        horizon=2,
        patch_len=4,
        stride=2,
        d_model=16,
        d_ff=32,
        n_layers=1,
        future_exo_dim=0,
        past_exo_cont_dim=0,
        past_exo_cat_dim=0,
        use_exogenous_mode=False,
        use_revin=False,
        attn=AttentionConfig(
            n_heads=4,
            d_model=16,
            attn_dropout=0.0,
            proj_dropout=0.0,
        ),
    )
    model = build_patchTST_quantile(config)
    save_model(
        model,
        config,
        str(path),
        extra_meta={"model_key": "patchtst_quantile", "family_key": "patchtst"},
    )


def _make_exogenous_point_checkpoint(path: Path) -> None:
    """Write a deterministic PatchTST checkpoint with both continuous exo paths."""
    torch.manual_seed(20260724)
    config = PatchTSTConfig(
        lookback=8,
        horizon=2,
        patch_len=4,
        stride=2,
        d_model=16,
        d_ff=32,
        n_layers=1,
        dropout=0.0,
        future_exo_dim=1,
        future_exo_fusion_dropout=0.0,
        past_exo_cont_dim=1,
        past_exo_cat_dim=0,
        use_exogenous_mode=True,
        use_revin=False,
        attn=AttentionConfig(
            n_heads=4,
            d_model=16,
            attn_dropout=0.0,
            proj_dropout=0.0,
        ),
    )
    model = build_patchTST_exogenous(config)
    save_model(
        model,
        config,
        str(path),
        extra_meta={
            "model_key": "patchtst_exogenous",
            "family_key": "patchtst",
        },
    )


def _exogenous_point_request(
    checkpoint_path: Path,
    *,
    future_values: tuple[float, float],
) -> ForecastRequest:
    rows = [
        {
            "unique_id": "A",
            "date": 20240100 + day,
            "y": float(day),
            "exo_known": (
                0.25
                if day <= 8
                else float(future_values[day - 9])
            ),
        }
        for day in range(1, 11)
    ]
    return ForecastRequest(
        checkpoint_path=checkpoint_path,
        expected_model_key="patchtst_exogenous",
        data=DataRequest(
            df=pl.DataFrame(rows),
            lookback=8,
            horizon=2,
            freq="daily",
            exogenous=ExogenousConfig(
                use_exogenous_mode=True,
                use_past_exogenous=True,
                use_future_exogenous=True,
                past_exo_cont_cols=["exo_known"],
                future_exo_cont_cols=["exo_known"],
            ),
        ),
        series_ids=["A"],
        forecast_origin=20240109,
        runtime=ForecastRuntimeConfig(
            batch_size=1,
            num_workers=0,
            device="cpu",
            pin_memory=False,
        ),
    )


def _point_request(
    checkpoint_path: Path,
    *,
    batch_size: int,
    expected_model_key: str | None = "patchtst_base",
) -> ForecastRequest:
    """Build a deterministic point-forecast request."""
    return ForecastRequest(
        checkpoint_path=checkpoint_path,
        expected_model_key=expected_model_key,
        data=DataRequest(
            df=_make_daily_frame(),
            lookback=8,
            horizon=2,
            freq="daily",
        ),
        series_ids=["C", "A"],
        forecast_origin=20240109,
        runtime=ForecastRuntimeConfig(
            batch_size=batch_size,
            num_workers=0,
            device="cpu",
            pin_memory=False,
        ),
    )


def test_forecast_point_result_schema_order_and_no_file_output(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "point.pt"
    _make_point_checkpoint(checkpoint_path)
    files_before = set(tmp_path.iterdir())

    result = forecast(_point_request(checkpoint_path, batch_size=2))

    assert isinstance(result, ForecastResult)
    assert result.model_key == "patchtst_base"
    assert result.forecast_origin == 20240109
    assert result.predictions.schema == EXPECTED_SCHEMA
    assert result.predictions.select("series_id", "horizon_step").rows() == [
        ("C", 0),
        ("C", 1),
        ("A", 0),
        ("A", 1),
    ]
    assert result.predictions["point"].null_count() == 0
    assert result.predictions["q10"].null_count() == 4
    assert result.predictions["q50"].null_count() == 4
    assert result.predictions["q90"].null_count() == 4
    assert set(tmp_path.iterdir()) == files_before


def test_forecast_row_identity_and_values_are_batch_size_independent(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "point.pt"
    _make_point_checkpoint(checkpoint_path)

    one = forecast(_point_request(checkpoint_path, batch_size=1)).predictions
    two = forecast(_point_request(checkpoint_path, batch_size=2)).predictions

    assert_frame_equal(one, two, check_exact=False, rtol=1e-6, atol=1e-6)


def test_forecast_patchtst_continuous_future_exogenous_values_are_active(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "patchtst-exogenous.pt"
    _make_exogenous_point_checkpoint(checkpoint_path)

    low = forecast(
        _exogenous_point_request(
            checkpoint_path,
            future_values=(0.0, 0.0),
        )
    ).predictions
    low_repeat = forecast(
        _exogenous_point_request(
            checkpoint_path,
            future_values=(0.0, 0.0),
        )
    ).predictions
    high = forecast(
        _exogenous_point_request(
            checkpoint_path,
            future_values=(1.0, -0.5),
        )
    ).predictions

    assert low.schema == high.schema == EXPECTED_SCHEMA
    assert low["model_key"].to_list() == ["patchtst_exogenous"] * 2
    np.testing.assert_array_equal(low["point"].to_numpy(), low_repeat["point"].to_numpy())
    assert float(np.max(np.abs(low["point"].to_numpy() - high["point"].to_numpy()))) > 1e-6


def test_forecast_rejects_checkpoint_model_key_mismatch(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "point.pt"
    _make_point_checkpoint(checkpoint_path)

    with pytest.raises(ValueError, match="Checkpoint model key mismatch"):
        forecast(
            _point_request(
                checkpoint_path,
                batch_size=2,
                expected_model_key="patchmixer_base",
            )
        )


def test_forecast_rejects_legacy_simple_backend_before_checkpoint_loading() -> None:
    request = ForecastRequest(
        checkpoint_path="unused.pt",
        expected_model_key=None,
        data=DataRequest(
            df=_make_daily_frame(("A",)),
            lookback=8,
            horizon=2,
            freq="daily",
            backend="simple",
        ),
        series_ids=["A"],
        forecast_origin=20240109,
    )

    with pytest.raises(ValueError, match="canonical exo data backend"):
        forecast(request)


def test_forecast_rejects_empty_series_selection_before_checkpoint_loading() -> None:
    request = ForecastRequest(
        checkpoint_path="unused.pt",
        expected_model_key=None,
        data=DataRequest(
            df=_make_daily_frame(("A",)),
            lookback=8,
            horizon=2,
            freq="daily",
        ),
        series_ids=[],
        forecast_origin=20240109,
    )

    with pytest.raises(ValueError, match="series_ids must not be empty"):
        forecast(request)


def test_forecast_quantile_result_uses_q50_as_point(monkeypatch: pytest.MonkeyPatch) -> None:
    forecast_module = importlib.import_module("modeling_module.api.forecast")

    class FakeQuantilePredictor:
        """Minimal predictor that emits the established flattened quantile contract."""

        model_key = "patchtst_quantile"

        def predict(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, np.ndarray]:
            horizon = int(kwargs["horizon"])
            size = len(batch["part_ids"]) * horizon
            q50 = np.arange(size, dtype=np.float64) + 10.0
            return {"q10": q50 - 1.0, "q50": q50, "q90": q50 + 1.0}

    monkeypatch.setattr(forecast_module, "load_predictor", lambda *args, **kwargs: FakeQuantilePredictor())
    request = ForecastRequest(
        checkpoint_path="unused.pt",
        expected_model_key="patchtst_quantile",
        data=DataRequest(
            df=_make_daily_frame(("A", "B")),
            lookback=8,
            horizon=2,
            freq="daily",
        ),
        series_ids=["B", "A"],
        forecast_origin=20240109,
        runtime=ForecastRuntimeConfig(batch_size=1, device="cpu", pin_memory=False),
    )

    result = forecast_module.forecast(request)

    assert result.predictions.schema == EXPECTED_SCHEMA
    assert result.predictions.select("series_id", "horizon_step").rows() == [
        ("B", 0),
        ("B", 1),
        ("A", 0),
        ("A", 1),
    ]
    assert result.predictions["point"].to_list() == result.predictions["q50"].to_list()
    assert result.predictions["q10"].null_count() == 0
    assert result.predictions["q90"].null_count() == 0


def test_forecast_real_quantile_checkpoint_populates_all_quantiles(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "quantile.pt"
    _make_quantile_checkpoint(checkpoint_path)

    result = forecast(
        _point_request(
            checkpoint_path,
            batch_size=2,
            expected_model_key="patchtst_quantile",
        )
    )

    assert result.model_key == "patchtst_quantile"
    assert result.predictions.schema == EXPECTED_SCHEMA
    assert result.predictions["q10"].null_count() == 0
    assert result.predictions["q50"].null_count() == 0
    assert result.predictions["q90"].null_count() == 0
    assert result.predictions["point"].to_list() == result.predictions["q50"].to_list()

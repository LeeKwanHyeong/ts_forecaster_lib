"""MORA shape·시간·채널·0수요 계약. 실제 모델·DB 없이 검증."""

from dataclasses import replace

import numpy as np
import pytest

from modeling_module.forecast_adapters.contracts import ForecastQuery, ForecastFitData
from modeling_module.forecast_adapters.MORA import MORAAdapter, MORAConfig
from modeling_module.forecast_adapters.MORA.fitting import purged_folds


def query():
    return ForecastQuery(np.ones((12, 52, 1)), np.ones((12, 27, 1)), np.arange(12) * 79 + 52,
                          ("demand",), ("part",) * 12, "frozen_model_sha256", "raw_demand")


@pytest.mark.parametrize("field,value", [
    ("history", np.ones((12, 52))), ("history", np.ones((12, 52, 1), dtype=bool)),
    ("history", np.full((12, 52, 1), np.nan)), ("history", np.full((12, 52, 1), np.inf)),
    ("backbone_prediction", np.ones((12, 26, 2))), ("origins", np.arange(12, dtype=float)),
    ("origins", np.zeros(12, dtype=int)), ("channel_names", ()), ("series_ids", ("part",)),
    ("backbone_id", ""), ("scale_id", ""),
])
def test_invalid_queries_are_rejected(field, value):
    with pytest.raises((TypeError, ValueError)):
        replace(query(), **{field: value})


def test_queries_copy_inputs_and_zero_remains_observed():
    values = np.zeros((12, 52, 1))
    prepared = replace(query(), history=values)
    values[:] = 100
    assert np.all(prepared.history == 0) and not prepared.history.flags.writeable
    with pytest.raises(ValueError):
        prepared.history[0, 0, 0] = 3
    assert prepared.take([1, 3]).series_ids == ("part", "part")


@pytest.mark.parametrize("changes", [
    {"lookback": 23}, {"horizon": 3}, {"correction_rank": 27}, {"period": 0},
    {"taper_steps": 1}, {"folds": 2}, {"ridge_strength": 0}, {"ridge_strength": float("nan")},
    {"shrinkage": -1}, {"structured_mix": 2}, {"weight_power": -1}, {"scope": "automatic"},
    {"calibration_groups": 17}, {"horizon": True},
])
def test_invalid_configuration_is_rejected(changes):
    with pytest.raises((TypeError, ValueError)):
        MORAConfig(**changes)


def test_backbone_training_declaration_is_explicit_not_automatically_verified():
    data = ForecastFitData(query(), np.ones((12, 27, 1)), "frozen_pretrained")
    with pytest.raises(ValueError):
        replace(data, backbone_provenance="in_sample_trained")
    with pytest.raises(ValueError, match="SHA"):
        replace(data, backbone_provenance="blocked_oof")
    replace(data, backbone_provenance="blocked_oof", oof_manifest_sha256="a" * 64)
    with pytest.raises(ValueError):
        replace(data, target=np.zeros((12, 26, 1)))


def test_purging_removes_history_and_target_overlap_not_only_equal_origins():
    n, length, horizon = 100, 24, 8
    data = ForecastQuery(np.ones((n, length, 1)), np.ones((n, horizon, 1)),
                         np.arange(n) * 3 + length, ("demand",), ("part",) * n, "model")
    folds = purged_folds(data, 4, 8)
    assert sorted(np.concatenate([held for _, held in folds]).tolist()) == list(range(n))
    for fit, held in folds:
        start = data.origins - length
        end = data.origins + horizon - 1
        assert np.all((end[fit] < start[held].min()) | (start[fit] > end[held].max()))
        assert len(fit) < n - len(held)


def test_dense_short_history_does_not_silently_disable_purging():
    dense = replace(query(), origins=np.arange(12) + 52)
    with pytest.raises(ValueError, match="부족"):
        purged_folds(dense, 4, 8)


def test_fit_and_predict_fail_before_state_exists():
    adapter = MORAAdapter()
    with pytest.raises(RuntimeError):
        adapter.predict(query())
    with pytest.raises(RuntimeError):
        adapter.diagnostics
    with pytest.raises(TypeError):
        adapter.fit(query(), query())

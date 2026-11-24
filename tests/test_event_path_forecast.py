"""Contract and sampling checks for the public event path API."""

import json
import math
from types import SimpleNamespace

import pytest
import torch

import modeling_module.api.event as event_api
from modeling_module.api.event import (
    EVENT_PATH_CONTRACT_ID,
    EventDurationDistribution,
    EventHistory,
    EventModelConfig,
    ForecastEvent,
    NextEventPrediction,
    build_event_model,
    forecast_event_paths,
    list_event_model_keys,
    load_event_checkpoint,
    save_event_checkpoint,
)


def _history():
    return EventHistory(
        torch.tensor([[0, 2], [0, 5]]),
        torch.tensor([[3.0, 4.0], [7.0, 9.0]]),
        torch.ones((2, 2), dtype=torch.bool),
        "week",
    )


def _fake_model(*, max_history_events=2):
    return SimpleNamespace(
        event_config=EventModelConfig("week", max_history_events, 1.2),
        event_model_key="phen_tpp",
    )


def _prediction(history, *, median, scale=0.01, quantity=2.0):
    rows = history.gaps.size(0)
    return NextEventPrediction(
        EventDurationDistribution(
            torch.full((rows,), math.log(median), dtype=torch.float64),
            torch.full((rows,), scale, dtype=torch.float64),
            1.0,
            "week",
        ),
        torch.full((rows,), quantity),
    )


def test_first_event_is_conditioned_per_series_then_history_is_advanced(monkeypatch):
    calls = []

    def predict(_model, history):
        calls.append(history.gaps.clone())
        return _prediction(history, median=3 if len(calls) == 1 else 1)

    monkeypatch.setattr(event_api, "predict_next_event", predict)
    history = _history()
    result = forecast_event_paths(
        _fake_model(), history, series_ids=("late", "recent"),
        elapsed_steps=(2, 0), horizon_steps=3, num_paths=2, seed=17,
    )
    late, recent = result.series
    assert late.paths == ((ForecastEvent(0, 2.0), ForecastEvent(1, 2.0), ForecastEvent(2, 2.0)),) * 2
    assert recent.paths == ((ForecastEvent(2, 2.0),),) * 2
    assert calls[1].tolist() == [[2, 3], [2, 3]]
    assert calls[2].tolist() == [[3, 1], [3, 1]]
    assert history.gaps.tolist() == [[0, 2], [0, 5]]


def test_empty_paths_preserve_tail_probability_after_elapsed_weeks(monkeypatch):
    history = _history()
    prediction = _prediction(history, median=4, scale=0.5)
    monkeypatch.setattr(event_api, "predict_next_event", lambda *_: prediction)
    samples = 4096
    result = forecast_event_paths(
        _fake_model(), history, series_ids=("a", "b"), elapsed_steps=(0, 3),
        horizon_steps=1, num_paths=samples, seed=53,
    )
    assert len(result.series) == 2
    for index, steps in enumerate((0, 3)):
        expected_tail = (
            prediction.duration.log_survival(steps + 1)
            - prediction.duration.log_survival(steps)
        ).exp()[index].item()
        paths = result.series[index].paths
        assert len(paths) == samples
        assert sum(not path for path in paths) / samples == pytest.approx(expected_tail, abs=0.03)
        assert all(path in ((), (ForecastEvent(0, 2.0),)) for path in paths)


def test_rare_condition_does_not_round_away_event_or_move_it_before_origin(monkeypatch):
    monkeypatch.setattr(
        event_api, "predict_next_event", lambda _, history: _prediction(history, median=1, scale=0.01)
    )
    result = forecast_event_paths(
        _fake_model(), _history(), series_ids=("a", "b"), elapsed_steps=(8, 10),
        horizon_steps=1, num_paths=4, seed=39,
    )
    assert all(path == (ForecastEvent(0, 2.0),) for series in result.series for path in series.paths)


@pytest.mark.parametrize("key", list_event_model_keys())
def test_checkpoint_to_repeatable_json_paths_for_each_model(key, tmp_path):
    torch.manual_seed(10)
    config = EventModelConfig("week", 3, 1.2)
    model = build_event_model(key, config).eval()
    checkpoint = tmp_path / "tpp.pt"
    save_event_checkpoint(model, checkpoint)
    kwargs = dict(
        series_ids=("A", "B"), elapsed_steps=(0, 5),
        horizon_steps=26, num_paths=2, seed=123,
    )
    before = torch.random.get_rng_state()
    original = forecast_event_paths(model, _history(), **kwargs)
    assert torch.equal(torch.random.get_rng_state(), before)
    restored = load_event_checkpoint(checkpoint, expected_model_key=key)
    result = forecast_event_paths(restored, _history(), **kwargs)
    assert original == result
    payload = json.loads(json.dumps(result.as_dict(), allow_nan=False))
    assert payload["contract_id"] == EVENT_PATH_CONTRACT_ID
    assert payload["model_key"] == key
    assert payload["horizon_steps"] == 26 and payload["time_unit"] == "week"
    assert [series["series_id"] for series in payload["series"]] == ["A", "B"]
    for series in result.series:
        assert len(series.paths) == 2
        for path in series.paths:
            offsets = [item.offset for item in path]
            assert offsets == sorted(set(offsets))
            assert all(0 <= offset < 26 for offset in offsets)
            assert all(item.quantity > 0 for item in path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"series_ids": ("a", "a")}, "unique"),
        ({"series_ids": ("a",)}, "history row"),
        ({"elapsed_steps": (1,)}, "elapsed count"),
        ({"elapsed_steps": (1, -1)}, "nonnegative integers"),
        ({"horizon_steps": 0}, "horizon_steps"),
        ({"num_paths": 0}, "num_paths"),
        ({"seed": True}, "seed"),
    ],
)
def test_invalid_path_request(overrides, message):
    kwargs = dict(series_ids=("a", "b"), elapsed_steps=(0, 0), horizon_steps=3, num_paths=1, seed=1)
    with pytest.raises(ValueError, match=message):
        forecast_event_paths(_fake_model(), _history(), **{**kwargs, **overrides})


def test_top_coded_checkpoints_remain_available_only_for_one_step():
    model = build_event_model("phen_tpp", EventModelConfig("week", 3, 1.2, top_code=6)).eval()
    assert event_api.predict_next_event(model, _history()).duration.top_code == 6
    with pytest.raises(ValueError, match="uncapped-duration"):
        forecast_event_paths(
            model, _history(), series_ids=("a", "b"), elapsed_steps=(0, 0),
            horizon_steps=4, num_paths=1, seed=2,
        )

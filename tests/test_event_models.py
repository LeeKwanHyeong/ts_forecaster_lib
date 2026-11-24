"""Event registry, checkpoint, and one-step objective regression tests."""

import pytest
import torch

from modeling_module.api.event import (
    EventDurationDistribution,
    EventHistory,
    EventModelConfig,
    build_event_model,
    list_event_model_keys,
    load_event_checkpoint,
    next_event_loss,
    predict_next_event,
    save_event_checkpoint,
)
from modeling_module.models.registry import (
    expand_training_targets,
    list_available_model_keys,
    resolve_artifact_model_key,
)

EVENT_KEYS = ("rmtpp_matched_head", "phen_tpp", "s2p2_matched_head")


def _config(**overrides) -> EventModelConfig:
    return EventModelConfig(**{"time_unit": "week", "max_history_events": 12,
                               "train_log_mean": 1.2, "min_history_events": 2, **overrides})


def _history() -> EventHistory:
    return EventHistory(
        torch.tensor([[0, 4, 5], [1, 8, 0]]),
        torch.tensor([[3, 5, 4], [4, 12, 0]]),
        torch.tensor([[True, True, True], [True, True, False]]),
        "week",
    )


def test_event_keys_are_visible_only_on_event_route() -> None:
    assert list_event_model_keys() == EVENT_KEYS
    assert set(EVENT_KEYS).isdisjoint(list_available_model_keys())
    for key in EVENT_KEYS:
        with pytest.raises(ValueError, match="Unknown artifact model"):
            resolve_artifact_model_key(key)
        with pytest.raises(ValueError, match="Unknown artifact model"):
            expand_training_targets([key])


@pytest.mark.parametrize("key", EVENT_KEYS)
def test_train_one_step_predict_and_restore(key, tmp_path) -> None:
    torch.manual_seed(42)
    model = build_event_model(key, _config())
    history = _history()
    optimizer = torch.optim.AdamW(model.parameters())
    loss = next_event_loss(
        model, history, target_gap=torch.tensor([3, 8]), target_quantity=torch.tensor([6, 2]),
    )
    objective = loss.time_nll.mean() + loss.quantity_log_mse.mean()
    objective.backward()
    assert model.quantity_head.weight.grad.abs().sum() > 0
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    optimizer.step()

    prediction = predict_next_event(model.eval(), history)
    assert prediction.duration.time_unit == "week"
    assert prediction.duration.scale.dtype == torch.float64
    assert (prediction.duration.scale > 0).all()
    assert (prediction.quantity_point > 0).all()
    for tensor in (prediction.duration.location, prediction.duration.scale, prediction.quantity_point):
        assert tensor.shape == (2,)

    path = tmp_path / "event.pt"
    save_event_checkpoint(model, path)
    restored = load_event_checkpoint(path, expected_model_key=key)
    assert restored.event_config == _config()
    reread = predict_next_event(restored, history)
    torch.testing.assert_close(reread.quantity_point, prediction.quantity_point, rtol=0, atol=0)
    torch.testing.assert_close(reread.duration.location, prediction.duration.location, rtol=0, atol=0)
    torch.testing.assert_close(reread.duration.scale, prediction.duration.scale, rtol=0, atol=0)
    with pytest.raises(ValueError, match="different model key"):
        load_event_checkpoint(path, expected_model_key="patchtst_base")


def test_padding_is_not_observed_demand() -> None:
    history = _history()
    dirty = EventHistory(
        history.gaps.float().masked_fill(~history.mask, float("nan")),
        history.quantities.float().masked_fill(~history.mask, float("inf")),
        history.mask, "week",
    )
    for key in EVENT_KEYS:
        model = build_event_model(key, _config()).eval()
        actual, expected = (predict_next_event(model, frame) for frame in (dirty, history))
        torch.testing.assert_close(actual.duration.location, expected.duration.location)
        torch.testing.assert_close(actual.quantity_point, expected.quantity_point)


@pytest.mark.parametrize(("gaps", "quantities", "mask", "unit", "error"), [
    ([1, 0, 2], [1, 2, 3], [True, False, True], "week", "right-padded"),
    ([1, 1, 0], [1, 1, 0], [True, False, False], "week", "Insufficient"),
    ([1, 1, 0], [1, 1, 0], [True, True, False], "day", "time unit"),
    ([1, 1.5, 0], [1, 1, 0], [True, True, False], "week", "integer"),
    ([1, 0, 0], [1, 1, 0], [True, True, False], "week", "integer"),
    ([1, 2, 0], [1, 0, 0], [True, True, False], "week", "positive quantities"),
])
def test_invalid_observations_are_rejected(gaps, quantities, mask, unit, error) -> None:
    event = EventHistory(torch.tensor([gaps]), torch.tensor([quantities]), torch.tensor([mask]), unit)
    with pytest.raises(ValueError, match=error):
        event.validated(_config())


def test_discrete_survival_matches_probability_mass() -> None:
    for top_code in (None, 4):
        law = EventDurationDistribution(
            torch.tensor([0.5], dtype=torch.float64),
            torch.tensor([0.8], dtype=torch.float64),
            1.0, "week", top_code,
        )
        for step in range(1, 5):
            observed = law.log_mass(torch.tensor([step])).exp()
            expected = law.log_survival(step - 1).exp() - law.log_survival(step).exp()
            torch.testing.assert_close(observed, expected)


def test_wrong_checkpoint_format_is_rejected(tmp_path) -> None:
    path = tmp_path / "regular.pt"
    torch.save({"state_dict": {}, "model_key": "patchtst_base"}, path)
    with pytest.raises(ValueError, match="Incompatible event checkpoint"):
        load_event_checkpoint(path)

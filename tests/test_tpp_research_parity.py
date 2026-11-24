"""Optional verification against the pinned independent research implementation."""

import os
from pathlib import Path

import pytest
import torch

from modeling_module.api.event import (
    EventHistory,
    EventModelConfig,
    build_event_model,
    next_event_loss,
    predict_next_event,
)


@pytest.mark.parametrize(("model_key", "research_key"), [
    ("rmtpp_matched_head", "rmtpp"),
    ("phen_tpp", "phen"),
    ("s2p2_matched_head", "s2p2_matched_head"),
])
def test_research_state_outputs_and_gradients_match(model_key, research_key, monkeypatch) -> None:
    research_root = os.environ.get("TPP_RESEARCH_ROOT")
    if not research_root:
        pytest.skip("Set TPP_RESEARCH_ROOT to the pinned paper_research checkout")
    if not (Path(research_root) / "phen" / "registry.py").is_file():
        pytest.fail("TPP_RESEARCH_ROOT does not contain the PHEN research reference")
    monkeypatch.syspath_prepend(research_root)
    from phen.registry import build_model

    torch.manual_seed(42)
    original = build_model(
        research_key, max_seq_len=24, hidden_dim=64, train_log_mean=1.2,
        time_scale=3.0, time_initial_location=-0.2, time_initial_scale=0.68,
        time_observation_contract={
            "mode": "positive_integer_round_clamp_v1", "unit": "week", "top_code": None,
        },
    ).eval()
    torch.manual_seed(42)
    model = build_event_model(
        model_key, EventModelConfig("week", 24, 1.2, 3.0, -0.2, 0.68),
    ).eval()
    assert original.state_dict().keys() == model.state_dict().keys()
    for key, value in original.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    gaps = torch.tensor([[1, 4, 5, 2], [1, 8, 0, 0]])
    qty = torch.tensor([[3., 5., 4., 8.], [4., 12., 0., 0.]])
    mask = gaps > 0
    history = EventHistory(gaps, qty, mask, "week")
    target_gap, target_qty = torch.tensor([3, 8]), torch.tensor([6., 2.])

    with torch.no_grad():
        original.quantity_head.weight.normal_()
        if research_key == "phen":
            original.history_correction.output_projections[0].weight.normal_()
    model.load_state_dict(original.state_dict(), strict=True)
    original_hidden = original.encode(gaps, qty, mask)[torch.arange(2), mask.sum(1) - 1]
    prediction = predict_next_event(model, history)
    torch.testing.assert_close(prediction.quantity_point, original.predict_quantity(original_hidden)[1])
    torch.testing.assert_close(
        prediction.duration.log_mass(target_gap), original.log_observation_dt(original_hidden, target_gap),
    )
    torch.testing.assert_close(
        prediction.duration.time_scale * prediction.duration.location.exp(),
        original.predict_time_median(original_hidden),
    )

    reference = -original.log_observation_dt(original_hidden, target_gap)
    reference += original.quantity_outputs(original_hidden, target_qty)["log_mse"]
    reference.mean().backward()
    result = next_event_loss(model, history, target_gap=target_gap, target_quantity=target_qty)
    (result.time_nll + result.quantity_log_mse).mean().backward()
    for (_, actual), (_, expected) in zip(model.named_parameters(), original.named_parameters(), strict=True):
        if expected.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(actual.grad, expected.grad)

"""Explicit TPP API; separate from regular-window training and inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from modeling_module.models.registry import get_event_model_spec, list_event_model_keys
from modeling_module.models.TPP.contracts import (
    EVENT_CONTRACT_ID,
    EventDurationDistribution,
    EventHistory,
    EventModelConfig,
    NextEventLoss,
    NextEventPrediction,
    _is_real_numeric,
)
from modeling_module.models.TPP.heads import NextEventModel
from modeling_module.models.TPP.rollout import (
    EVENT_PATH_CONTRACT_ID,
    EventPathForecast,
    ForecastEvent,
    SeriesEventPaths,
    rollout_event_paths,
)

EVENT_CHECKPOINT_ID = "modeling-module.event-checkpoint.v1"


def build_event_model(model_key: str, config: EventModelConfig) -> NextEventModel:
    """Create fresh weights. No pretrained downloads or regular-window fallback."""
    if not isinstance(config, EventModelConfig):
        raise TypeError("Expected EventModelConfig")
    spec = get_event_model_spec(model_key)
    model = spec.load_builder()(config)
    model.event_config = config
    model.event_model_key = spec.key
    return model


def _last_states(
    model: NextEventModel, history: EventHistory,
) -> tuple[torch.Tensor, torch.Tensor]:
    gaps, quantities, indices = history.validated(model.event_config)
    device = next(model.parameters()).device
    if gaps.device != device:
        raise ValueError("Event history must be on the model device")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("The float64 event likelihood currently supports CPU and CUDA only")
    time_states, quantity_states = model.encode_task_states(gaps, quantities, history.mask)
    rows = torch.arange(indices.numel(), device=device)
    return time_states[rows, indices], quantity_states[rows, indices]


@torch.inference_mode()
def predict_next_event(model: NextEventModel, history: EventHistory) -> NextEventPrediction:
    """Return a distribution measured from the last observed event (not W0)."""
    if model.training:
        raise ValueError("Call model.eval() before next-event inference")
    time_state, quantity_state = _last_states(model, history)
    location, scale = model.duration_parameters(time_state)
    _, point = model.predict_quantity(quantity_state)
    if not all(bool(torch.isfinite(value).all()) for value in (location, scale, point)):
        raise ValueError("Non-finite next-event prediction")
    config = model.event_config
    return NextEventPrediction(
        EventDurationDistribution(location, scale, config.time_scale, config.time_unit, config.top_code),
        point,
    )


def next_event_loss(
    model: NextEventModel, history: EventHistory, *,
    target_gap: torch.Tensor, target_quantity: torch.Tensor,
) -> NextEventLoss:
    """Score the next observed event without passing its values to the encoder."""
    time_state, quantity_state = _last_states(model, history)
    if not isinstance(target_quantity, torch.Tensor) or (
        target_quantity.shape != time_state.shape[:1] or target_quantity.device != time_state.device
        or not _is_real_numeric(target_quantity) or not bool(torch.isfinite(target_quantity).all())
        or bool((target_quantity <= 0).any())
    ):
        raise ValueError("Expected one finite positive quantity per observed target event")
    time_nll = -model.log_observation_dt(time_state, target_gap)
    quantity = model.quantity_outputs(quantity_state, target_quantity.float())
    return NextEventLoss(time_nll, quantity["log_mse"])


@torch.inference_mode()
def forecast_event_paths(
    model: NextEventModel,
    history: EventHistory,
    *,
    series_ids: Sequence[str],
    elapsed_steps: Sequence[int],
    horizon_steps: int,
    num_paths: int,
    seed: int,
) -> EventPathForecast:
    """Sample in-horizon paths with the first gap conditioned on D > elapsed."""
    config = model.event_config
    if config.top_code is not None:
        raise ValueError("Event paths require an uncapped-duration checkpoint")
    return rollout_event_paths(
        lambda observations: predict_next_event(model, observations),
        history,
        model_key=model.event_model_key,
        max_history_events=config.max_history_events,
        series_ids=series_ids,
        elapsed_steps=elapsed_steps,
        horizon_steps=horizon_steps,
        num_paths=num_paths,
        seed=seed,
    )


def save_event_checkpoint(model: NextEventModel, path: str | Path) -> None:
    """Store model identity, the event config and weights, not a serialized model."""
    torch.save(
        {
            "format": EVENT_CHECKPOINT_ID,
            "input_contract": EVENT_CONTRACT_ID,
            "model_key": model.event_model_key,
            "config": asdict(model.event_config),
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_event_checkpoint(
    path: str | Path, *, expected_model_key: str | None = None,
) -> NextEventModel:
    """Restore to CPU in eval mode; require the exact input/weight contract."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    keys = {"format", "input_contract", "model_key", "config", "state_dict"}
    if (
        not isinstance(payload, Mapping) or set(payload) != keys
        or payload["format"] != EVENT_CHECKPOINT_ID or payload["input_contract"] != EVENT_CONTRACT_ID
        or not isinstance(payload["model_key"], str)
        or not isinstance(payload["config"], Mapping) or not isinstance(payload["state_dict"], Mapping)
    ):
        raise ValueError("Incompatible event checkpoint contract")
    if expected_model_key is not None and payload["model_key"] != expected_model_key:
        raise ValueError("Event checkpoint has a different model key")
    model = build_event_model(payload["model_key"], EventModelConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


__all__ = [
    "EVENT_CHECKPOINT_ID", "EVENT_CONTRACT_ID", "EVENT_PATH_CONTRACT_ID",
    "EventDurationDistribution", "EventHistory", "EventModelConfig",
    "EventPathForecast", "ForecastEvent", "SeriesEventPaths", "forecast_event_paths",
    "NextEventLoss", "NextEventPrediction", "build_event_model",
    "list_event_model_keys", "load_event_checkpoint", "next_event_loss",
    "predict_next_event", "save_event_checkpoint",
]

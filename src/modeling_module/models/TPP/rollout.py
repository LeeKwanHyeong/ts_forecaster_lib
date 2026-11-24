"""Bounded, right-conditioned event paths without calendar dependencies."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import torch

from .contracts import EventHistory, EventTimeUnit, NextEventPrediction

EVENT_PATH_CONTRACT_ID = "modeling-module.event-paths.v1"
_PATH_BATCH_SIZE = 64


@dataclass(frozen=True, slots=True)
class ForecastEvent:
    offset: int
    quantity: float


@dataclass(frozen=True, slots=True)
class SeriesEventPaths:
    series_id: str
    paths: tuple[tuple[ForecastEvent, ...], ...]


@dataclass(frozen=True, slots=True)
class EventPathForecast:
    model_key: str
    time_unit: EventTimeUnit
    horizon_steps: int
    seed: int
    series: tuple[SeriesEventPaths, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready result without tensors or calendar dates."""
        return {
            "contract_id": EVENT_PATH_CONTRACT_ID,
            "model_key": self.model_key,
            "time_unit": self.time_unit,
            "horizon_steps": self.horizon_steps,
            "seed": self.seed,
            "series": [
                {
                    "series_id": item.series_id,
                    "paths": [
                        [{"offset": event.offset, "quantity": event.quantity} for event in path]
                        for path in item.paths
                    ],
                }
                for item in self.series
            ],
        }


@dataclass(slots=True)
class _Path:
    gaps: list[int]
    quantities: list[float]
    offset: int
    elapsed: int
    events: list[ForecastEvent] = field(default_factory=list)

    def append(self, gap: int, quantity: float, max_history_events: int) -> None:
        self.offset += gap
        self.events.append(ForecastEvent(self.offset, quantity))
        self.elapsed = 0
        self.gaps = [*self.gaps, gap][-max_history_events:]
        self.quantities = [*self.quantities, quantity][-max_history_events:]


def rollout_event_paths(
    predict: Callable[[EventHistory], NextEventPrediction],
    history: EventHistory,
    *,
    model_key: str,
    max_history_events: int,
    series_ids: Sequence[str],
    elapsed_steps: Sequence[int],
    horizon_steps: int,
    num_paths: int,
    seed: int,
) -> EventPathForecast:
    ids = tuple(series_ids)
    elapsed = tuple(elapsed_steps)
    if (
        isinstance(series_ids, str) or not ids
        or any(not isinstance(key, str) or not key.strip() for key in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Specify unique nonempty series IDs in batch order")
    if (
        not isinstance(history, EventHistory)
        or not isinstance(history.gaps, torch.Tensor)
        or history.gaps.ndim != 2
    ):
        raise ValueError("Event paths require rank-two event history")
    if len(ids) != history.gaps.size(0) or len(elapsed) != len(ids):
        raise ValueError("Each history row requires one series ID and elapsed count")
    if type(horizon_steps) is not int or horizon_steps < 1:
        raise ValueError("horizon_steps must be a positive integer")
    if type(num_paths) is not int or num_paths < 1:
        raise ValueError("num_paths must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative signed 64-bit integer")
    if any(type(steps) is not int or steps < 0 for steps in elapsed):
        raise ValueError("elapsed_steps must be nonnegative integers")
    if max(elapsed) + horizon_steps >= 2**52:
        raise ValueError("Event steps exceed float64 half-step precision")

    initial = predict(history)
    device = initial.duration.location.device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    lengths = history.mask.sum(dim=1).tolist()
    source_gaps = history.gaps.tolist()
    source_qty = history.quantities.tolist()
    series_paths: list[list[tuple[ForecastEvent, ...]]] = [[] for _ in ids]
    for begin in range(0, len(ids) * num_paths, _PATH_BATCH_SIZE):
        series_indices = [
            index // num_paths
            for index in range(begin, min(begin + _PATH_BATCH_SIZE, len(ids) * num_paths))
        ]
        paths = [
            _Path(
                [int(gap) for gap in source_gaps[index][:lengths[index]]],
                source_qty[index][:lengths[index]],
                -elapsed[index] - 1,
                elapsed[index],
            )
            for index in series_indices
        ]
        indices = torch.tensor(series_indices, dtype=torch.long, device=device)
        active = paths
        prediction = _select_prediction(initial, indices)
        while active:
            sampled = _sample_gaps(prediction, active, horizon_steps, generator)
            quantities = prediction.quantity_point.tolist()
            remaining = []
            for path, gap, quantity in zip(active, sampled, quantities):
                if gap is None:
                    continue
                if not math.isfinite(quantity) or quantity <= 0:
                    raise ValueError("A sampled event requires a finite positive quantity")
                path.append(gap, quantity, max_history_events)
                if path.offset < horizon_steps - 1:
                    remaining.append(path)
            active = remaining
            if active:
                prediction = predict(_pack(active, history.time_unit, device))
        for index, path in zip(series_indices, paths):
            series_paths[index].append(tuple(path.events))

    return EventPathForecast(
        model_key, history.time_unit, horizon_steps, seed,
        tuple(SeriesEventPaths(key, tuple(paths)) for key, paths in zip(ids, series_paths)),
    )


def _select_prediction(
    prediction: NextEventPrediction, indices: torch.Tensor
) -> NextEventPrediction:
    duration = prediction.duration
    return NextEventPrediction(
        replace(duration, location=duration.location[indices], scale=duration.scale[indices]),
        prediction.quantity_point[indices],
    )


def _pack(paths: list[_Path], unit: EventTimeUnit, device: torch.device) -> EventHistory:
    width = max(len(path.gaps) for path in paths)
    padding = [width - len(path.gaps) for path in paths]
    return EventHistory(
        torch.tensor(
            [path.gaps + [0] * pad for path, pad in zip(paths, padding)], device=device,
        ),
        torch.tensor(
            [path.quantities + [0.0] * pad for path, pad in zip(paths, padding)],
            dtype=torch.float32, device=device,
        ),
        torch.tensor(
            [[True] * len(path.gaps) + [False] * pad for path, pad in zip(paths, padding)],
            dtype=torch.bool, device=device,
        ),
        unit,
    )


def _sample_gaps(
    prediction: NextEventPrediction,
    paths: list[_Path],
    horizon: int,
    generator: torch.Generator,
) -> list[int | None]:
    law = prediction.duration
    device = law.location.device
    elapsed = torch.tensor([path.elapsed for path in paths], device=device)
    upper = torch.tensor([horizon - 1 - path.offset for path in paths], device=device)

    def log_survival(steps: torch.Tensor) -> torch.Tensor:
        z = (
            torch.log(steps.to(torch.float64) + 0.5)
            - math.log(law.time_scale) - law.location
        ) / law.scale
        return torch.where(steps == 0, 0.0, torch.special.log_ndtr(-z))

    baseline = log_survival(elapsed)
    if not bool(torch.isfinite(baseline).all()):
        raise ValueError("Unresolved survival probability for the elapsed interval")
    log_draw = torch.log1p(-torch.rand(
        len(paths), generator=generator, dtype=torch.float64, device="cpu",
    )).to(device)
    # Keep the full tail probability; do not renormalize over in-horizon gaps.
    in_horizon = log_draw > log_survival(upper) - baseline
    lower = elapsed + 1
    for _ in range(horizon.bit_length()):
        middle = lower + (upper - lower) // 2
        moves_left = log_draw > log_survival(middle) - baseline
        searching = in_horizon & (lower < upper)
        lower = torch.where(searching & ~moves_left, middle + 1, lower)
        upper = torch.where(searching & moves_left, middle, upper)
    return [gap if keep else None for gap, keep in zip(lower.tolist(), in_horizon.tolist())]

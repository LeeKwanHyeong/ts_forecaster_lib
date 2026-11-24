"""Versioned input and output for one-step event forecasting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from .time_likelihood import positive_integer_log_mass

EVENT_CONTRACT_ID = "modeling-module.next-event.v1"
EventTimeUnit = Literal["hour", "day", "week", "month"]
TIME_UNITS = {"hour", "day", "week", "month"}


@dataclass(frozen=True, slots=True)
class EventModelConfig:
    """Persist training-only initialization values with the resulting weights."""

    time_unit: EventTimeUnit
    max_history_events: int
    train_log_mean: float
    time_scale: float = 1.0
    time_initial_location: float = 0.0
    time_initial_scale: float = 1.0
    time_sigma_floor: float = 0.001
    min_history_events: int = 1
    top_code: int | None = None

    def __post_init__(self) -> None:
        if self.time_unit not in TIME_UNITS:
            raise ValueError("Unsupported event time unit")
        if (
            type(self.max_history_events) is not int or self.max_history_events < 2
            or type(self.min_history_events) is not int
            or not 1 <= self.min_history_events <= self.max_history_events
        ):
            raise ValueError("Invalid minimum or maximum event history length")
        if not all(
            type(value) in (float, int) and math.isfinite(value)
            for value in (self.train_log_mean, self.time_scale, self.time_initial_location,
                          self.time_initial_scale, self.time_sigma_floor)
        ):
            raise ValueError("Event model initialization values must be finite numbers")
        if self.train_log_mean <= 0 or self.time_scale <= 0:
            raise ValueError("Event log quantity mean and time scale must be positive")
        if not 0 < self.time_sigma_floor < self.time_initial_scale:
            raise ValueError("Initial time scale must be greater than its positive floor")
        if self.top_code is not None and (type(self.top_code) is not int or self.top_code < 2):
            raise ValueError("top_code must be an integer >= 2")


def _is_real_numeric(tensor: torch.Tensor) -> bool:
    return tensor.is_floating_point() or tensor.dtype in (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    )


@dataclass(frozen=True, slots=True)
class EventHistory:
    """Right-padded observed events, not regularly spaced demand buckets."""

    gaps: torch.Tensor
    quantities: torch.Tensor
    mask: torch.Tensor
    time_unit: EventTimeUnit

    def validated(self, config: EventModelConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.time_unit != config.time_unit:
            raise ValueError("Event time unit differs from the checkpoint")
        gaps, qty, mask = self.gaps, self.quantities, self.mask
        if not all(isinstance(value, torch.Tensor) for value in (gaps, qty, mask)):
            raise ValueError("Event history needs gap, quantity and mask tensors")
        if (
            gaps.ndim != 2 or qty.shape != gaps.shape or mask.shape != gaps.shape
            or not gaps.size(0) or not 1 <= gaps.size(1) <= config.max_history_events
            or gaps.device != qty.device or gaps.device != mask.device
            or mask.dtype != torch.bool or not _is_real_numeric(gaps) or not _is_real_numeric(qty)
        ):
            raise ValueError("Expected same-device [batch, events] tensors and a boolean mask")
        counts = mask.sum(dim=1)
        if bool((counts < config.min_history_events).any()):
            raise ValueError("Insufficient observed events")
        if bool((mask[:, 1:] & ~mask[:, :-1]).any()):
            raise ValueError("Event history must be right-padded without gaps in the mask")
        gaps = torch.where(mask, gaps, 0)
        qty = torch.where(mask, qty, 0)
        if not torch.isfinite(gaps).all() or not torch.isfinite(qty).all():
            raise ValueError("Event values must be finite")
        # Zero is allowed only as a first-event boundary marker. It is not an
        # observed inter-event gap and must not be supplied as a duration target.
        if bool(((gaps < 0) | (gaps != gaps.round())).any()) or bool(
            ((gaps[:, 1:] == 0) & mask[:, 1:]).any()
        ):
            raise ValueError("Observed inter-event gaps must be positive integer time steps")
        if config.top_code is not None and bool((gaps > config.top_code).any()):
            raise ValueError("Observed gap exceeds top_code")
        if bool(((qty <= 0) & mask).any()):
            raise ValueError("Observed events must have positive quantities")
        return gaps, qty, counts - 1


@dataclass(frozen=True, slots=True)
class EventDurationDistribution:
    """Rounded positive log-normal. The latent median is not a calendar step."""

    location: torch.Tensor
    scale: torch.Tensor
    time_scale: float
    time_unit: EventTimeUnit
    top_code: int | None = None

    def log_mass(self, gap: torch.Tensor) -> torch.Tensor:
        return positive_integer_log_mass(
            gap, self.location, self.scale, time_scale=self.time_scale, top_code=self.top_code,
        )

    def log_survival(self, elapsed_steps: int) -> torch.Tensor:
        """Return log P(observed gap > elapsed_steps). Zero fully elapsed steps has mass one."""
        if type(elapsed_steps) is not int or elapsed_steps < 0:
            raise ValueError("Elapsed time must count complete integer time steps")
        if elapsed_steps == 0:
            return torch.zeros_like(self.location)
        if self.top_code is not None and elapsed_steps >= self.top_code:
            return torch.full_like(self.location, -torch.inf)
        z = (math.log(elapsed_steps + 0.5) - math.log(self.time_scale) - self.location) / self.scale
        return torch.special.log_ndtr(-z)


@dataclass(frozen=True, slots=True)
class NextEventPrediction:
    duration: EventDurationDistribution
    quantity_point: torch.Tensor


@dataclass(frozen=True, slots=True)
class NextEventLoss:
    """One time NLL and log1p-quantity squared error per observed target event."""

    time_nll: torch.Tensor
    quantity_log_mse: torch.Tensor

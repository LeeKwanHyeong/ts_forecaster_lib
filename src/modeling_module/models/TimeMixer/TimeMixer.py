"""Public endogenous TimeMixer tensor boundary."""

from __future__ import annotations

from typing import Any, cast

import torch

from .backbone import TimeMixerBackbone, TimeMixerBackboneConfigLike
from .configs import TimeMixerConfig
from .provenance import (
    TIMEMIXER_UPSTREAM_COMMIT,
    TIMEMIXER_UPSTREAM_REPOSITORY,
)


def _has_nonempty_features(value: Any) -> bool:
    if value is None:
        return False
    if torch.is_tensor(value):
        return value.numel() > 0 and (
            value.ndim == 0 or int(value.shape[-1]) > 0
        )
    return True


class TimeMixerModel(TimeMixerBackbone):
    """Paper-aligned endogenous point model with the library tensor contract."""

    architecture_variant = "endogenous"
    exogenous_fusion_strategy = "none"
    upstream_repository = TIMEMIXER_UPSTREAM_REPOSITORY
    upstream_commit = TIMEMIXER_UPSTREAM_COMMIT

    def __init__(self, config: TimeMixerConfig) -> None:
        if not isinstance(config, TimeMixerConfig):
            raise TypeError(
                "TimeMixerModel requires TimeMixerConfig; use build_timemixer() "
                "for mapping or namespace inputs."
            )
        super().__init__(cast(TimeMixerBackboneConfigLike, config))
        self.cfg = config
        self.config = config
        self.lookback = int(config.lookback)
        self.horizon = int(config.horizon)
        self.y_dim = int(config.y_dim)
        self.future_exo_dim = 0
        self.past_exo_cont_dim = 0
        self.past_exo_cat_dim = 0
        self.loss = config.loss
        self.loss_type = "point"
        self.out_mult = 1
        self.param_names = None

    @classmethod
    def from_config(cls, config: TimeMixerConfig) -> "TimeMixerModel":
        return cls(config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        future_exo: Any = None,
        past_exo_cont: Any = None,
        past_exo_cat: Any = None,
    ) -> torch.Tensor:
        provided = [
            name
            for name, value in (
                ("future_exo", future_exo),
                ("past_exo_cont", past_exo_cont),
                ("past_exo_cat", past_exo_cat),
            )
            if _has_nonempty_features(value)
        ]
        if provided:
            raise RuntimeError(
                "timemixer is endogenous-only; unsupported inputs: "
                + ", ".join(provided)
            )

        if not torch.is_tensor(x):
            raise TypeError(
                "TimeMixerModel expects x to be a floating tensor with shape "
                f"[B,L,1], got {type(x).__name__}."
            )
        if x.ndim != 3:
            raise ValueError(
                "TimeMixerModel expects x with shape [B,L,1], "
                f"got {tuple(x.shape)}."
            )

        batch_size, lookback, channels = (int(value) for value in x.shape)
        if batch_size <= 0:
            raise ValueError("TimeMixerModel requires a non-empty batch.")
        if lookback != self.lookback:
            raise ValueError(
                "TimeMixerModel lookback mismatch: "
                f"got {lookback}, expected {self.lookback}."
            )
        if channels != self.y_dim:
            raise ValueError(
                "TimeMixerModel channel mismatch: "
                f"got {channels}, expected {self.y_dim}."
            )
        if not torch.is_floating_point(x):
            raise TypeError(
                "TimeMixerModel expects a floating input tensor, "
                f"got dtype={x.dtype}."
            )
        if not bool(torch.isfinite(x).all()):
            raise ValueError("TimeMixerModel input must contain only finite values.")

        forecast = super().forward(x)
        expected_shape = (batch_size, self.horizon, self.y_dim)
        if tuple(forecast.shape) != expected_shape:
            raise RuntimeError(
                "TimeMixer backbone violated its output contract: "
                f"got {tuple(forecast.shape)}, expected {expected_shape}."
            )
        return forecast


__all__ = ["TimeMixerModel"]

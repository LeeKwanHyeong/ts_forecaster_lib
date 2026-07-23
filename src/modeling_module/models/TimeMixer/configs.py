"""Configuration contract for the endogenous TimeMixer model family."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

from modeling_module.training.config import TrainingConfig


_SUPPORTED_EMBED_TYPES = {"timeF", "fixed", "learned"}
_SUPPORTED_FREQUENCIES = {"h", "t", "s", "ms", "m", "a", "w", "d", "b"}


def _integral_value(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    return int(value)


def _boolean_value(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean, got {value!r}.")
    return value


@dataclass
class TimeMixerConfig(TrainingConfig):
    """Strict configuration for the first endogenous TimeMixer artifact."""

    y_dim: int = 1
    d_model: int = 16
    d_ff: int = 32
    e_layers: int = 2
    moving_avg: int = 25
    down_sampling_layers: int = 3
    down_sampling_window: int = 2
    down_sampling_method: Literal["avg"] = "avg"
    decomp_method: Literal["moving_avg"] = "moving_avg"
    channel_independence: bool = True
    use_norm: bool = True
    dropout: float = 0.1
    embed: Literal["timeF", "fixed", "learned"] = "timeF"
    freq: str = "h"
    use_future_temporal_feature: bool = False
    use_exogenous_mode: bool = False
    future_exo_dim: int = 0

    def __post_init__(self) -> None:
        self.lookback = _integral_value("lookback", self.lookback)
        self.horizon = _integral_value("horizon", self.horizon)
        self.y_dim = _integral_value("y_dim", self.y_dim)
        self.d_model = _integral_value("d_model", self.d_model)
        self.d_ff = _integral_value("d_ff", self.d_ff)
        self.e_layers = _integral_value("e_layers", self.e_layers)
        self.moving_avg = _integral_value("moving_avg", self.moving_avg)
        self.down_sampling_layers = _integral_value(
            "down_sampling_layers",
            self.down_sampling_layers,
        )
        self.down_sampling_window = _integral_value(
            "down_sampling_window",
            self.down_sampling_window,
        )
        self.future_exo_dim = _integral_value(
            "future_exo_dim",
            self.future_exo_dim,
        )

        positive = {
            "lookback": self.lookback,
            "horizon": self.horizon,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "e_layers": self.e_layers,
            "moving_avg": self.moving_avg,
            "down_sampling_window": self.down_sampling_window,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        if self.y_dim != 1:
            raise ValueError(
                "TimeMixerConfig currently supports exactly one target channel; "
                f"got y_dim={self.y_dim}."
            )
        if self.moving_avg % 2 == 0:
            raise ValueError(
                "moving_avg must be odd so decomposition preserves every scale length."
            )
        if self.down_sampling_layers < 0:
            raise ValueError("down_sampling_layers must be non-negative.")
        if self.down_sampling_layers > 0 and self.down_sampling_window <= 1:
            raise ValueError(
                "down_sampling_window must be greater than 1 when downsampling is enabled."
            )
        if self.scale_lengths[-1] < 1:
            raise ValueError(
                "TimeMixer scale configuration collapses the coarsest sequence: "
                f"lookback={self.lookback}, window={self.down_sampling_window}, "
                f"layers={self.down_sampling_layers}."
            )

        self.channel_independence = _boolean_value(
            "channel_independence",
            self.channel_independence,
        )
        self.use_norm = _boolean_value("use_norm", self.use_norm)
        self.use_future_temporal_feature = _boolean_value(
            "use_future_temporal_feature",
            self.use_future_temporal_feature,
        )
        self.use_exogenous_mode = _boolean_value(
            "use_exogenous_mode",
            self.use_exogenous_mode,
        )
        if not self.channel_independence:
            raise ValueError("TimeMixerConfig requires channel_independence=True.")
        if self.use_future_temporal_feature:
            raise ValueError(
                "TimeMixerConfig does not support future temporal features in v1."
            )
        if self.use_exogenous_mode or self.future_exo_dim != 0:
            raise ValueError(
                "TimeMixerConfig is endogenous-only; exogenous settings must be disabled."
            )

        self.dropout = float(self.dropout)
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}.")
        if self.down_sampling_method != "avg":
            raise ValueError("down_sampling_method must be 'avg'.")
        if self.decomp_method != "moving_avg":
            raise ValueError("decomp_method must be 'moving_avg'.")
        if not isinstance(self.embed, str) or self.embed not in _SUPPORTED_EMBED_TYPES:
            raise ValueError(
                f"embed must be one of {sorted(_SUPPORTED_EMBED_TYPES)}, "
                f"got {self.embed!r}."
            )
        if not isinstance(self.freq, str) or self.freq not in _SUPPORTED_FREQUENCIES:
            raise ValueError(
                f"freq must be one of {sorted(_SUPPORTED_FREQUENCIES)}, "
                f"got {self.freq!r}."
            )

    @property
    def scale_lengths(self) -> tuple[int, ...]:
        return tuple(
            self.lookback // (self.down_sampling_window**index)
            for index in range(self.down_sampling_layers + 1)
        )

    @property
    def task_name(self) -> str:
        return "long_term_forecast"

    @property
    def seq_len(self) -> int:
        return self.lookback

    @property
    def label_len(self) -> int:
        return 0

    @property
    def pred_len(self) -> int:
        return self.horizon

    @property
    def enc_in(self) -> int:
        return self.y_dim

    @property
    def c_out(self) -> int:
        return self.y_dim


__all__ = ["TimeMixerConfig"]

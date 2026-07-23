from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from modeling_module.training.config import TrainingConfig


@dataclass
class NHITSConfig(TrainingConfig):
    """Configuration for the public endogenous N-HiTS point model."""

    y_dim: int = 1
    stack_types: tuple[str, ...] = ("identity", "identity", "identity")
    n_blocks: tuple[int, ...] = (1, 1, 1)
    n_layers: tuple[int, ...] = (2, 2, 2)
    n_theta_hidden: tuple[tuple[int, ...], ...] = (
        (256, 256),
        (256, 256),
        (256, 256),
    )
    n_pool_kernel_size: tuple[int, ...] = (2, 2, 1)
    n_freq_downsample: tuple[int, ...] = (4, 2, 1)
    pooling_mode: Literal["max", "average"] = "max"
    interpolation_mode: str = "linear"
    activation: str = "ReLU"
    initialization: str = "glorot_uniform"
    batch_normalization: bool = False
    dropout_prob_theta: float = 0.0
    shared_weights: bool = False
    use_exogenous_mode: bool = False

    def __post_init__(self) -> None:
        self.stack_types = tuple(str(value) for value in self.stack_types)
        self.n_blocks = tuple(int(value) for value in self.n_blocks)
        self.n_layers = tuple(int(value) for value in self.n_layers)
        self.n_theta_hidden = tuple(
            tuple(int(width) for width in hidden)
            for hidden in self.n_theta_hidden
        )
        self.n_pool_kernel_size = tuple(int(value) for value in self.n_pool_kernel_size)
        self.n_freq_downsample = tuple(int(value) for value in self.n_freq_downsample)

        if int(self.lookback) <= 0:
            raise ValueError(f"lookback must be positive, got {self.lookback}.")
        if int(self.horizon) <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}.")
        if int(self.y_dim) != 1:
            raise ValueError(
                "NHITSConfig currently supports one target channel; "
                f"got y_dim={self.y_dim}."
            )
        if bool(self.use_exogenous_mode):
            raise ValueError("nhits_base is endogenous-only; use_exogenous_mode must be False.")

        stack_count = len(self.stack_types)
        if stack_count == 0:
            raise ValueError("NHITSConfig requires at least one stack.")
        stack_fields = {
            "n_blocks": self.n_blocks,
            "n_layers": self.n_layers,
            "n_theta_hidden": self.n_theta_hidden,
            "n_pool_kernel_size": self.n_pool_kernel_size,
            "n_freq_downsample": self.n_freq_downsample,
        }
        for name, values in stack_fields.items():
            if len(values) != stack_count:
                raise ValueError(
                    f"{name} must contain one value per stack: "
                    f"expected {stack_count}, got {len(values)}."
                )

        if any(stack_type != "identity" for stack_type in self.stack_types):
            raise ValueError("NHITSConfig currently supports only identity stacks.")
        if any(value <= 0 for value in self.n_blocks):
            raise ValueError("n_blocks values must be positive.")
        if any(value <= 0 for value in self.n_layers):
            raise ValueError("n_layers values must be positive.")
        if any(value <= 0 for value in self.n_pool_kernel_size):
            raise ValueError("n_pool_kernel_size values must be positive.")
        if any(value <= 0 for value in self.n_freq_downsample):
            raise ValueError("n_freq_downsample values must be positive.")

        for index, (layer_count, hidden) in enumerate(
            zip(self.n_layers, self.n_theta_hidden)
        ):
            if len(hidden) != layer_count:
                raise ValueError(
                    "n_theta_hidden must provide one width per hidden layer: "
                    f"stack {index} has n_layers={layer_count}, widths={hidden}."
                )
            if any(width <= 0 for width in hidden):
                raise ValueError("n_theta_hidden widths must be positive.")

        if self.pooling_mode not in {"max", "average"}:
            raise ValueError(
                "pooling_mode must be either 'max' or 'average', "
                f"got {self.pooling_mode!r}."
            )
        if self.interpolation_mode not in {"linear", "nearest"} and not str(
            self.interpolation_mode
        ).startswith("cubic-"):
            raise ValueError(
                "interpolation_mode must be 'linear', 'nearest', or 'cubic-<batch_size>', "
                f"got {self.interpolation_mode!r}."
            )
        if str(self.interpolation_mode).startswith("cubic-"):
            try:
                cubic_batch_size = int(str(self.interpolation_mode).rsplit("-", 1)[1])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "cubic interpolation_mode must end with a positive batch size."
                ) from exc
            if cubic_batch_size <= 0:
                raise ValueError(
                    "cubic interpolation_mode must end with a positive batch size."
                )
        supported_activations = {
            "ReLU",
            "Softplus",
            "Tanh",
            "SELU",
            "LeakyReLU",
            "PReLU",
            "Sigmoid",
        }
        if self.activation not in supported_activations:
            raise ValueError(
                f"activation must be one of {sorted(supported_activations)}, "
                f"got {self.activation!r}."
            )
        supported_initializations = {
            "orthogonal",
            "he_uniform",
            "he_normal",
            "glorot_uniform",
            "glorot_normal",
            "lecun_normal",
        }
        if self.initialization not in supported_initializations:
            raise ValueError(
                f"initialization must be one of {sorted(supported_initializations)}, "
                f"got {self.initialization!r}."
            )
        if not 0.0 <= float(self.dropout_prob_theta) < 1.0:
            raise ValueError("dropout_prob_theta must be in [0, 1).")


__all__ = ["NHITSConfig"]

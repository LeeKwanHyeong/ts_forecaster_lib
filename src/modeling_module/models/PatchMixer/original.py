"""Canonical point PatchMixer pinned to the official upstream implementation.

This module adapts the MIT-licensed implementation from
https://github.com/Zeying-Gong/PatchMixer at commit
cfc6c1386e7fe1633f92ef4b258ff1a4649008b4. Class and parameter names inside
the model intentionally follow upstream so its state dict can be loaded
strictly for parity tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from .provenance import PATCHMIXER_UPSTREAM_COMMIT, PATCHMIXER_UPSTREAM_REPOSITORY


_MISSING = object()


def _config_value(config: Any, *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if isinstance(config, Mapping) and name in config:
            return config[name]
        if hasattr(config, name):
            return getattr(config, name)
    if default is not _MISSING:
        return default
    joined = " or ".join(repr(name) for name in names)
    raise ValueError(f"PatchMixerOriginal config requires {joined}.")


@dataclass(frozen=True)
class PatchMixerOriginalConfig:
    """Architecture fields required by the canonical upstream point model."""

    lookback: int
    horizon: int
    enc_in: int = 1
    patch_len: int = 16
    stride: int = 8
    mixer_kernel_size: int = 8
    d_model: int = 256
    e_layers: int = 1
    dropout: float = 0.2
    head_dropout: float = 0.0
    use_revin: bool = True
    revin_affine: bool = True
    revin_subtract_last: bool = False

    def __post_init__(self) -> None:
        positive_fields = {
            "lookback": self.lookback,
            "horizon": self.horizon,
            "enc_in": self.enc_in,
            "patch_len": self.patch_len,
            "stride": self.stride,
            "mixer_kernel_size": self.mixer_kernel_size,
            "d_model": self.d_model,
            "e_layers": self.e_layers,
        }
        for name, value in positive_fields.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.patch_len > self.lookback:
            raise ValueError(
                f"patch_len must not exceed lookback, got {self.patch_len} > {self.lookback}."
            )
        for name, value in (
            ("dropout", self.dropout),
            ("head_dropout", self.head_dropout),
        ):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}.")

    @property
    def seq_len(self) -> int:
        return self.lookback

    @property
    def pred_len(self) -> int:
        return self.horizon

    @classmethod
    def from_config(cls, config: Any) -> "PatchMixerOriginalConfig":
        if isinstance(config, cls):
            return config
        return cls(
            lookback=int(_config_value(config, "lookback", "seq_len")),
            horizon=int(_config_value(config, "horizon", "pred_len")),
            enc_in=int(_config_value(config, "enc_in", default=1)),
            patch_len=int(_config_value(config, "patch_len", default=16)),
            stride=int(_config_value(config, "stride", default=8)),
            mixer_kernel_size=int(
                _config_value(config, "mixer_kernel_size", default=8)
            ),
            d_model=int(_config_value(config, "d_model", default=256)),
            e_layers=int(_config_value(config, "e_layers", default=1)),
            dropout=float(_config_value(config, "dropout", default=0.2)),
            head_dropout=float(_config_value(config, "head_dropout", default=0.0)),
            use_revin=bool(_config_value(config, "use_revin", default=True)),
            revin_affine=bool(
                _config_value(config, "revin_affine", default=True)
            ),
            revin_subtract_last=bool(
                _config_value(config, "revin_subtract_last", default=False)
            ),
        )


class PatchMixerOriginalRevIN(nn.Module):
    """Upstream RevIN math kept local to preserve strict output parity."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine = bool(affine)
        self.subtract_last = bool(subtract_last)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError(f"RevIN mode must be 'norm' or 'denorm', got {mode!r}.")

    def _get_statistics(self, x: torch.Tensor) -> None:
        dims = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dims, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dims, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x - (self.last if self.subtract_last else self.mean)
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + (self.last if self.subtract_last else self.mean)
        return x


class PatchMixerOriginalLayer(nn.Module):
    """Upstream separable convolution over `(patch_num, d_model)`."""

    def __init__(self, dim: int, a: int, kernel_size: int = 8) -> None:
        super().__init__()
        self.Resnet = nn.Sequential(
            nn.Conv1d(
                dim,
                dim,
                kernel_size=kernel_size,
                groups=dim,
                padding="same",
            ),
            nn.GELU(),
            nn.BatchNorm1d(dim),
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv1d(dim, a, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(a),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.Resnet(x)
        return self.Conv_1x1(x)


class PatchMixerOriginalBackbone(nn.Module):
    """Canonical single-scale, channel-independent PatchMixer point backbone."""

    def __init__(self, configs: PatchMixerOriginalConfig) -> None:
        super().__init__()
        self.nvals = int(configs.enc_in)
        self.lookback = int(configs.seq_len)
        self.forecasting = int(configs.pred_len)
        self.patch_size = int(configs.patch_len)
        self.stride = int(configs.stride)
        self.kernel_size = int(configs.mixer_kernel_size)
        self.patch_num = int(
            (self.lookback - self.patch_size) / self.stride + 1
        ) + 1
        self.a = self.patch_num
        self.d_model = int(configs.d_model)
        self.depth = int(configs.e_layers)

        self.PatchMixer_blocks = nn.ModuleList(
            [
                PatchMixerOriginalLayer(
                    dim=self.patch_num,
                    a=self.a,
                    kernel_size=self.kernel_size,
                )
                for _ in range(self.depth)
            ]
        )
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.W_P = nn.Linear(self.patch_size, self.d_model)
        self.head0 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.patch_num * self.d_model, self.forecasting),
            nn.Dropout(float(configs.head_dropout)),
        )
        self.head1 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.a * self.d_model, self.forecasting * 2),
            nn.GELU(),
            nn.Dropout(float(configs.head_dropout)),
            nn.Linear(self.forecasting * 2, self.forecasting),
            nn.Dropout(float(configs.head_dropout)),
        )
        self.dropout = nn.Dropout(float(configs.dropout))
        self.revin = bool(configs.use_revin)
        if self.revin:
            self.revin_layer = PatchMixerOriginalRevIN(
                self.nvals,
                affine=bool(configs.revin_affine),
                subtract_last=bool(configs.revin_subtract_last),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"PatchMixerOriginal expects [B,L,N], got shape {tuple(x.shape)}."
            )
        if x.shape[1] != self.lookback:
            raise ValueError(
                f"PatchMixerOriginal expected lookback={self.lookback}, got {x.shape[1]}."
            )
        if x.shape[2] != self.nvals:
            raise ValueError(
                f"PatchMixerOriginal expected enc_in={self.nvals}, got {x.shape[2]}."
            )

        batch_size, _, nvars = x.shape
        if self.revin:
            x = self.revin_layer(x, "norm")
        x = x.permute(0, 2, 1)
        x = self.padding_patch_layer(x)
        x = x.unfold(
            dimension=-1,
            size=self.patch_size,
            step=self.stride,
        )
        x = self.W_P(x)
        x = x.reshape(batch_size * nvars, x.shape[2], x.shape[3])
        x = self.dropout(x)

        linear_forecast = self.head0(x)
        for block in self.PatchMixer_blocks:
            x = block(x)
        nonlinear_forecast = self.head1(x)
        x = linear_forecast + nonlinear_forecast
        x = x.reshape(batch_size, nvars, -1).permute(0, 2, 1)
        if self.revin:
            x = self.revin_layer(x, "denorm")
        return x


class PatchMixerOriginalModel(nn.Module):
    """Public model wrapper retaining upstream `model.*` state-dict keys."""

    upstream_repository = PATCHMIXER_UPSTREAM_REPOSITORY
    upstream_commit = PATCHMIXER_UPSTREAM_COMMIT
    architecture_variant = "original"

    def __init__(self, configs: Any) -> None:
        super().__init__()
        self.configs = PatchMixerOriginalConfig.from_config(configs)
        self.horizon = self.configs.horizon
        self.future_exo_dim = 0
        self.model = PatchMixerOriginalBackbone(self.configs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

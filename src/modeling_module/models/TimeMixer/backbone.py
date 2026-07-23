"""Forecasting-only backbone adapted from the pinned TimeMixer upstream.

Upstream source: https://github.com/kwuking/TimeMixer at commit
e24610583b36fdd8c76cc17a8df4e65759a5f460. The original source is licensed
under Apache-2.0; see ``LICENSE.upstream`` in this package.

Modifications in this file are limited to typing, naming the root module
``TimeMixerBackbone``, removing non-forecasting and out-of-scope branches, and
supporting the degenerate single-scale case. The supported forecasting graph
retains the upstream module construction order and parameter layout.
"""

from __future__ import annotations

import math
from typing import Protocol, Sequence

import torch
import torch.nn as nn

from .provenance import (
    TIMEMIXER_UPSTREAM_COMMIT,
    TIMEMIXER_UPSTREAM_REPOSITORY,
)


class TimeMixerBackboneConfigLike(Protocol):
    """Attributes consumed by the upstream-compatible numerical core."""

    task_name: str
    seq_len: int
    label_len: int
    pred_len: int
    down_sampling_window: int
    channel_independence: int | bool
    e_layers: int
    moving_avg: int
    enc_in: int
    c_out: int
    use_future_temporal_feature: int | bool
    d_model: int
    d_ff: int
    embed: str
    freq: str
    dropout: float
    use_norm: int | bool
    down_sampling_layers: int
    down_sampling_method: str
    decomp_method: str


class Normalize(nn.Module):
    """Upstream normalization layer used independently at each scale."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
        subtract_last: bool = False,
        non_norm: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError(f"Unsupported normalization mode: {mode!r}.")

    def _get_statistics(self, x: torch.Tensor) -> None:
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class MovingAverage(nn.Module):
    """Moving average with endpoint replication along the time axis."""

    def __init__(self, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class SeriesDecomposition(nn.Module):
    """Split a sequence into seasonal residual and moving-average trend."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moving_mean = self.moving_avg(x)
        return x - moving_mean, moving_mean


class PositionalEmbedding(nn.Module):
    """Sinusoidal buffer retained for upstream state-dict identity."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float()
            * -(math.log(10000.0) / d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    """Circular Conv1d value embedding from the upstream implementation."""

    def __init__(self, c_in: int, d_model: int) -> None:
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int) -> None:
        super().__init__()
        weights = torch.zeros(c_in, d_model).float()
        weights.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float()
            * -(math.log(10000.0) / d_model)
        ).exp()
        weights[:, 0::2] = torch.sin(position * div_term)
        weights[:, 1::2] = torch.cos(
            position * div_term[: weights[:, 1::2].shape[1]]
        )

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(weights, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        embed_type: str = "fixed",
        freq: str = "h",
    ) -> None:
        super().__init__()
        embedding = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = embedding(4, d_model)
        self.hour_embed = embedding(24, d_model)
        self.weekday_embed = embedding(7, d_model)
        self.day_embed = embedding(32, d_model)
        self.month_embed = embedding(13, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        minute_x = (
            self.minute_embed(x[:, :, 4])
            if hasattr(self, "minute_embed")
            else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        embed_type: str = "timeF",
        freq: str = "h",
    ) -> None:
        super().__init__()
        del embed_type
        freq_map = {
            "h": 4,
            "t": 5,
            "s": 6,
            "ms": 7,
            "m": 1,
            "a": 1,
            "w": 2,
            "d": 3,
            "b": 3,
        }
        self.embed = nn.Linear(freq_map[freq], d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class DataEmbeddingWithoutPosition(nn.Module):
    """Value embedding used by TimeMixer; positional state is retained but unused."""

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = "fixed",
        freq: str = "h",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(
                d_model=d_model,
                embed_type=embed_type,
                freq=freq,
            )
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor | None,
        x_mark: torch.Tensor | None,
    ) -> torch.Tensor:
        if x is None and x_mark is not None:
            return self.temporal_embedding(x_mark)
        if x is None:
            raise ValueError("TimeMixer value embedding requires an input tensor.")
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class MultiScaleSeasonMixing(nn.Module):
    """Bottom-up seasonal mixing from fine to coarse scales."""

    def __init__(self, configs: TimeMixerBackboneConfigLike) -> None:
        super().__init__()
        self.down_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        configs.seq_len // (configs.down_sampling_window**index),
                        configs.seq_len
                        // (configs.down_sampling_window ** (index + 1)),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        configs.seq_len
                        // (configs.down_sampling_window ** (index + 1)),
                        configs.seq_len
                        // (configs.down_sampling_window ** (index + 1)),
                    ),
                )
                for index in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(season_list) == 1:
            return [season_list[0].permute(0, 2, 1)]

        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for index in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[index](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if index + 2 <= len(season_list) - 1:
                out_low = season_list[index + 2]
            out_season_list.append(out_high.permute(0, 2, 1))
        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """Top-down trend mixing from coarse to fine scales."""

    def __init__(self, configs: TimeMixerBackboneConfigLike) -> None:
        super().__init__()
        self.up_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        configs.seq_len
                        // (configs.down_sampling_window ** (index + 1)),
                        configs.seq_len // (configs.down_sampling_window**index),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        configs.seq_len // (configs.down_sampling_window**index),
                        configs.seq_len // (configs.down_sampling_window**index),
                    ),
                )
                for index in reversed(range(configs.down_sampling_layers))
            ]
        )

    def forward(self, trend_list: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(trend_list) == 1:
            return [trend_list[0].permute(0, 2, 1)]

        trend_list_reverse = list(reversed(trend_list))
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for index in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[index](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if index + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[index + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    """Past-Decomposable-Mixing block from the forecasting encoder."""

    def __init__(self, configs: TimeMixerBackboneConfigLike) -> None:
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence
        self.decompsition = SeriesDecomposition(configs.moving_avg)
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)
        self.out_cross_layer = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff, configs.d_model),
        )

    def forward(self, x_list: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        length_list = [x.size(1) for x in x_list]
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        out_season_list = self.mixing_multi_scale_season(season_list)
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for original, season, trend, length in zip(
            x_list,
            out_season_list,
            out_trend_list,
            length_list,
        ):
            output = original + self.out_cross_layer(season + trend)
            out_list.append(output[:, :length, :])
        return out_list


class TimeMixerBackbone(nn.Module):
    """Pinned forecasting graph for channel-independent TimeMixer."""

    upstream_repository = TIMEMIXER_UPSTREAM_REPOSITORY
    upstream_commit = TIMEMIXER_UPSTREAM_COMMIT

    def __init__(self, configs: TimeMixerBackboneConfigLike) -> None:
        super().__init__()
        self._validate_supported_scope(configs)
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.pdm_blocks = nn.ModuleList(
            [PastDecomposableMixing(configs) for _ in range(configs.e_layers)]
        )
        self.preprocess = SeriesDecomposition(configs.moving_avg)
        self.enc_in = configs.enc_in
        self.use_future_temporal_feature = configs.use_future_temporal_feature
        self.enc_embedding = DataEmbeddingWithoutPosition(
            1,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.layer = configs.e_layers
        self.normalize_layers = nn.ModuleList(
            [
                Normalize(
                    configs.enc_in,
                    affine=True,
                    non_norm=True if configs.use_norm == 0 else False,
                )
                for _ in range(configs.down_sampling_layers + 1)
            ]
        )
        self.predict_layers = nn.ModuleList(
            [
                nn.Linear(
                    configs.seq_len // (configs.down_sampling_window**index),
                    configs.pred_len,
                )
                for index in range(configs.down_sampling_layers + 1)
            ]
        )
        self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)

    @staticmethod
    def _validate_supported_scope(configs: TimeMixerBackboneConfigLike) -> None:
        if configs.task_name not in {
            "long_term_forecast",
            "short_term_forecast",
        }:
            raise ValueError("TimeMixerBackbone supports forecasting tasks only.")
        if configs.channel_independence != 1:
            raise ValueError(
                "TimeMixerBackbone currently supports channel_independence=1 only."
            )
        if configs.down_sampling_method != "avg":
            raise ValueError(
                "TimeMixerBackbone currently supports average downsampling only."
            )
        if configs.decomp_method != "moving_avg":
            raise ValueError(
                "TimeMixerBackbone currently supports moving-average decomposition only."
            )
        if bool(configs.use_future_temporal_feature):
            raise ValueError(
                "TimeMixerBackbone does not support future temporal features."
            )
        if configs.enc_in != configs.c_out:
            raise ValueError(
                "TimeMixerBackbone requires enc_in and c_out to match for "
                "channel-independent forecasting."
            )

    def _multi_scale_process_inputs(
        self,
        x_enc: torch.Tensor,
    ) -> list[torch.Tensor]:
        down_pool = nn.AvgPool1d(self.configs.down_sampling_window)
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_original = x_enc
        x_enc_sampling_list = [x_enc.permute(0, 2, 1)]

        for _ in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_original)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_original = x_enc_sampling
        return x_enc_sampling_list

    def pre_enc(
        self,
        x_list: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], None]:
        return x_list, None

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        x_enc_list = self._multi_scale_process_inputs(x_enc)
        x_list = []
        batch_size = x_enc.shape[0]
        for index, x in enumerate(x_enc_list):
            batch_size, time_steps, channels = x.size()
            x = self.normalize_layers[index](x, "norm")
            x = (
                x.permute(0, 2, 1)
                .contiguous()
                .reshape(batch_size * channels, time_steps, 1)
            )
            x_list.append(x)

        encoded_list = []
        preprocessed = self.pre_enc(x_list)
        for x in preprocessed[0]:
            encoded_list.append(self.enc_embedding(x, None))

        for block in self.pdm_blocks:
            encoded_list = block(encoded_list)

        decoded_list = self.future_multi_mixing(
            batch_size,
            encoded_list,
            preprocessed,
        )
        output = torch.stack(decoded_list, dim=-1).sum(-1)
        return self.normalize_layers[0](output, "denorm")

    def future_multi_mixing(
        self,
        batch_size: int,
        encoded_list: Sequence[torch.Tensor],
        x_list: tuple[list[torch.Tensor], None],
    ) -> list[torch.Tensor]:
        decoded_list = []
        for index, encoded in zip(range(len(x_list[0])), encoded_list):
            decoded = self.predict_layers[index](
                encoded.permute(0, 2, 1)
            ).permute(0, 2, 1)
            decoded = self.projection_layer(decoded)
            decoded = (
                decoded.reshape(batch_size, self.configs.c_out, self.pred_len)
                .permute(0, 2, 1)
                .contiguous()
            )
            decoded_list.append(decoded)
        return decoded_list

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        return self.forecast(x_enc)


__all__ = [
    "TimeMixerBackbone",
    "TimeMixerBackboneConfigLike",
]

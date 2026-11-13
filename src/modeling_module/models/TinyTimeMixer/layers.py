"""패치·은닉 특징·채널을 각각 혼합하는 TTM 계열 공통 계층."""

from __future__ import annotations

import torch
from torch import nn


class AxisMixer(nn.Module):
    """지정 축에 MLP를 적용하고 잔차 연결로 원래 표현을 보존한다.

    LayerNorm은 은닉 축 D에서 적용하고, MLP 직전에 혼합할 축만 마지막으로
    이동한다. 입력값을 본 뒤 레이어를 만드는 지연 초기화는 사용하지 않는다.
    """

    def __init__(self, size: int, d_model: int, axis: int, expansion: int,
                 dropout: float, gated: bool) -> None:
        super().__init__()
        self.axis = axis
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(size, size) if gated else None
        self.mlp = nn.Sequential(
            nn.Linear(size, size * expansion), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(size * expansion, size), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.norm(x).movedim(self.axis, -1)
        if self.gate is not None:
            # 여기의 게이트는 Transformer의 Q/K/V 어텐션이 아니라 축별 가중치다.
            mixed = mixed * torch.softmax(self.gate(mixed), dim=-1)
        mixed = self.mlp(mixed).movedim(-1, self.axis)
        return x + mixed


class MixerLayer(nn.Module):
    """(B, C, P, D)에서 채널 → 패치 → 은닉 특징 순서로 혼합한다."""

    def __init__(self, channels: int, patches: int, d_model: int, expansion: int,
                 dropout: float, gated: bool, mix_channels: bool) -> None:
        super().__init__()
        self.channel_mixer = (
            AxisMixer(channels, d_model, 1, expansion, dropout, gated)
            if mix_channels else nn.Identity()
        )
        self.patch_mixer = (
            AxisMixer(patches, d_model, 2, expansion, dropout, gated)
            if patches > 1 else nn.Identity()
        )
        self.feature_mixer = AxisMixer(d_model, d_model, 3, expansion, dropout, gated)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_mixer(x)
        x = self.patch_mixer(x)
        return self.feature_mixer(x)


class AdaptivePatchingBlock(nn.Module):
    """은닉 축을 나누어 패치 수를 늘린 뒤, 같은 원소 수로 복원한다.

    factor=2이면 (B,C,P,D) → (B,C,2P,D/2)이다. 원시 수요를 새로
    보간하는 연산이 아니라, 패치 표현의 해상도를 바꾸는 연산이다.
    """

    def __init__(self, channels: int, patches: int, d_model: int, level: int,
                 depth: int, expansion: int, dropout: float, gated: bool,
                 mix_channels: bool) -> None:
        super().__init__()
        self.factor = 2 ** level
        self.layers = nn.Sequential(*[
            MixerLayer(channels, patches * self.factor, d_model // self.factor,
                       expansion, dropout, gated, mix_channels)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, patches, width = x.shape
        split = x.reshape(batch, channels, patches * self.factor, width // self.factor)
        return self.layers(split).reshape(batch, channels, patches, width)


class MixerStack(nn.Module):
    """세분화된 패치에서 원래 해상도로 내려오며 표현을 정제한다."""

    def __init__(self, *, channels: int, patches: int, d_model: int, depth: int,
                 levels: int, expansion: int, dropout: float, gated: bool,
                 mix_channels: bool) -> None:
        super().__init__()
        selected_levels = reversed(range(levels)) if levels else (0,)
        self.blocks = nn.Sequential(*[
            AdaptivePatchingBlock(channels, patches, d_model, level, depth,
                                  expansion, dropout, gated, mix_channels)
            for level in selected_levels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

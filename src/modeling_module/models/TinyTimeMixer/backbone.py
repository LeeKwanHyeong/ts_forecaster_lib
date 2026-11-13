"""정적 특성을 받지 않는 채널 독립 TTM 계열 백본.

정규화 통계는 매 호출의 반환값으로 전달한다. 다른 배치의 통계를 모듈에
저장하지 않으므로, 여러 순전파를 수행한 뒤에도 역정규화 대상이 바뀌지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .configs import TinyTimeMixerConfig
from .layers import MixerStack


@dataclass
class BackboneOutput:
    """백본 표현과 같은 배치에서 계산한 역정규화 정보를 묶는다."""

    hidden_state: torch.Tensor
    patch_values: torch.Tensor
    location: torch.Tensor
    scale: torch.Tensor


class TinyTimeMixerBackbone(nn.Module):
    """수요 정규화 → 패치화 → 선형 투영 → 적응형 Mixer를 수행한다."""

    def __init__(self, cfg: TinyTimeMixerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_projection = nn.Linear(cfg.patch_length, cfg.d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, cfg.d_model))
        self.resolution_embedding = (
            nn.Embedding(cfg.resolution_vocab_size, cfg.d_model)
            if cfg.resolution_vocab_size else None
        )
        self.mixers = MixerStack(
            channels=cfg.num_input_channels, patches=cfg.encoded_patches,
            d_model=cfg.d_model, depth=cfg.num_layers,
            levels=cfg.adaptive_patching_levels, expansion=cfg.expansion_factor,
            dropout=cfg.dropout, gated=cfg.gated_attention, mix_channels=False,
        )

    def observed_mask(self, x: torch.Tensor,
                      supplied: torch.Tensor | None = None) -> torch.Tensor:
        """입력 계약을 검증하고 결측과 실제 관측값을 구분한다."""
        if x.ndim != 3 or x.shape[1:] != (self.cfg.lookback, self.cfg.num_input_channels):
            raise ValueError("x는 (B, lookback, num_input_channels) 형태여야 합니다.")
        if x.shape[0] == 0 or not x.is_floating_point():
            raise ValueError("x는 비어 있지 않은 부동소수점 텐서여야 합니다.")
        if supplied is None:
            return torch.isfinite(x)
        if supplied.shape != x.shape or supplied.dtype != torch.bool or supplied.device != x.device:
            raise ValueError("관측 마스크는 x와 같은 크기·장치의 bool 텐서여야 합니다.")
        if torch.any(supplied & ~torch.isfinite(x)):
            raise ValueError("관측했다고 표시한 값에 NaN 또는 Inf가 있습니다.")
        return supplied

    def normalize(self, x: torch.Tensor, observed: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """관측값만으로 통계를 계산하고, 결측·가림 위치는 0으로 대체한다."""
        counts = observed.sum(dim=1, keepdim=True)
        if torch.any(counts == 0):
            raise ValueError("배치의 각 채널에 최소 한 개의 보이는 관측값이 필요합니다.")
        # 저정밀 입력에서도 평균·분산의 누적 오차를 줄인다.
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        safe = torch.where(observed, work, 0)
        if self.cfg.use_norm:
            location = (safe.sum(dim=1, keepdim=True) / counts).detach()
            centered = torch.where(observed, work - location, 0)
            variance = centered.square().sum(dim=1, keepdim=True) / counts
            scale = torch.sqrt(variance + self.cfg.norm_eps).detach()
        else:
            location = torch.zeros_like(safe[:, :1, :])
            scale = torch.ones_like(location)
        normalized = torch.where(observed, (safe - location) / scale, 0)
        return normalized.to(x.dtype), location, scale

    def patchify(self, values: torch.Tensor) -> torch.Tensor:
        """(B,L,C)를 (B,C,P,패치 길이)로 바꾼다. 앞쪽 패딩은 0이다."""
        values = values.transpose(1, 2)
        if self.cfg.left_padding:
            values = F.pad(values, (self.cfg.left_padding, 0))
        return values.unfold(-1, self.cfg.patch_length, self.cfg.patch_stride)

    def forward(self, x: torch.Tensor, *, past_observed_mask: torch.Tensor | None = None,
                patch_mask: torch.Tensor | None = None,
                freq_token: torch.Tensor | None = None) -> BackboneOutput:
        observed = self.observed_mask(x, past_observed_mask)
        normalized, location, scale = self.normalize(x, observed)
        patches = self.patchify(normalized)
        hidden = self.patch_projection(patches)
        if patch_mask is not None:
            expected = (x.shape[0], self.cfg.num_input_channels, self.cfg.num_patches)
            if (patch_mask.shape != expected or patch_mask.dtype != torch.bool
                    or patch_mask.device != x.device):
                raise ValueError("patch_mask는 (B,C,P) 크기의 같은 장치 bool 텐서여야 합니다.")
            hidden = torch.where(patch_mask.unsqueeze(-1), self.mask_token, hidden)
        if self.resolution_embedding is not None:
            if (freq_token is None or freq_token.shape != (x.shape[0],)
                    or freq_token.dtype not in (torch.int32, torch.int64)
                    or freq_token.device != x.device):
                raise ValueError("freq_token은 배치별 정수 코드 (B,)여야 합니다.")
            if torch.any((freq_token < 0) | (freq_token >= self.cfg.resolution_vocab_size)):
                raise ValueError("freq_token이 설정된 범주 수를 벗어났습니다.")
            prefix = self.resolution_embedding(freq_token.long())[:, None, None, :]
            prefix = prefix.expand(-1, self.cfg.num_input_channels, -1, -1)
            hidden = torch.cat((prefix, hidden), dim=2)
        elif freq_token is not None:
            raise ValueError("주기 토큰을 쓰려면 resolution_vocab_size를 설정해야 합니다.")
        return BackboneOutput(self.mixers(hidden), patches, location, scale)

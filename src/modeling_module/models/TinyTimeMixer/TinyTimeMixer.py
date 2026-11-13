"""직접 초기화하는 TTM 계열 예측 모델.

공식 모델의 래퍼가 아니며 Hugging Face나 네트워크에 접근하지 않는다.
기존 공용 학습 API에는 자동 등록하지 않은 독립 PyTorch 연구 모델이다.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .backbone import BackboneOutput, TinyTimeMixerBackbone
from .configs import TinyTimeMixerConfig
from .decoder import TinyTimeMixerDecoder


def resolve_config(cfg: TinyTimeMixerConfig | Mapping) -> TinyTimeMixerConfig:
    """설정 객체와 저장 후 복원한 사전을 같은 생성 경로로 연결한다."""
    if isinstance(cfg, TinyTimeMixerConfig):
        return cfg
    if isinstance(cfg, Mapping):
        return TinyTimeMixerConfig(**cfg)
    raise TypeError("TinyTimeMixerConfig 또는 설정 사전이 필요합니다.")


class TinyTimeMixerModel(nn.Module):
    """(B,L,C)의 과거 수요와 선택적 정적 코드로 (B,H,C)를 예측한다."""

    def __init__(self, cfg: TinyTimeMixerConfig | Mapping) -> None:
        super().__init__()
        self.cfg = resolve_config(cfg)
        self.lookback = self.cfg.lookback
        self.horizon = self.cfg.horizon
        self.model_name = "TinyTimeMixer"
        self.backbone = TinyTimeMixerBackbone(self.cfg)
        self.decoder = TinyTimeMixerDecoder(self.cfg) if self.cfg.use_decoder else None
        width = self.cfg.decoder_d_model if self.cfg.use_decoder else self.cfg.d_model
        self.head = nn.Sequential(
            nn.Dropout(self.cfg.dropout), nn.Flatten(start_dim=-2),
            nn.Linear(self.cfg.encoded_patches * width, self.cfg.horizon),
        )

    @classmethod
    def from_config(cls, cfg: TinyTimeMixerConfig | Mapping) -> TinyTimeMixerModel:
        return cls(cfg)

    def decode(self, encoded: BackboneOutput,
               static_categorical_values: torch.Tensor | None) -> torch.Tensor:
        if self.decoder is not None:
            return self.decoder(encoded.hidden_state, encoded.patch_values, static_categorical_values)
        if static_categorical_values is not None:
            raise ValueError("디코더가 없는 모델은 정적 입력을 사용할 수 없습니다.")
        return encoded.hidden_state

    def forward(self, x: torch.Tensor, *,
                static_categorical_values: torch.Tensor | None = None,
                past_observed_mask: torch.Tensor | None = None,
                freq_token: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.backbone(x, past_observed_mask=past_observed_mask, freq_token=freq_token)
        decoded = self.decode(encoded, static_categorical_values)
        # 채널마다 같은 예측 헤드를 사용하되, 배치·채널 차원은 합치지 않는다.
        normalized_forecast = self.head(decoded).transpose(1, 2)
        return normalized_forecast * encoded.scale + encoded.location

    def freeze_backbone(self, frozen: bool = True) -> None:
        """백본의 가중치 갱신만 제어한다. train/eval과 드롭아웃은 별도다."""
        self.backbone.requires_grad_(not frozen)

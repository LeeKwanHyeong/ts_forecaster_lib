"""시계열 표현에 정적 범주형 특성을 결합하는 디코더."""

from __future__ import annotations

import torch
from torch import nn

from .configs import TinyTimeMixerConfig
from .layers import MixerStack


class StaticCategoricalEmbedding(nn.Module):
    """각 변수의 범주 사전을 별도 임베딩 테이블로 처리한다.

    범주 ID를 실수로 해석하거나 임의로 잘라 범위에 맞추지 않는다. 미등록
    범주를 쓰려면 데이터 전처리에서 명시적인 UNK 코드를 예약해야 한다.
    """

    def __init__(self, vocab_sizes: tuple[int, ...], d_model: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(size, d_model) for size in vocab_sizes])

    def forward(self, ids: torch.Tensor, *, batch_size: int,
                device: torch.device, patches: int) -> torch.Tensor:
        if ids.shape != (batch_size, len(self.embeddings)):
            raise ValueError("정적 입력은 (B, 정적 변수 수) 형태여야 합니다.")
        if ids.dtype not in (torch.int32, torch.int64) or ids.device != device:
            raise ValueError("정적 입력은 시계열과 같은 장치의 정수 코드여야 합니다.")
        encoded = []
        for index, embedding in enumerate(self.embeddings):
            column = ids[:, index]
            if torch.any((column < 0) | (column >= embedding.num_embeddings)):
                raise ValueError(f"정적 변수 {index}의 코드가 범주 수를 벗어났습니다.")
            encoded.append(embedding(column.long()))
        # 실제 데이터를 52주·26주로 복사하지 않고 잠재 벡터를 패치 축에 펼친다.
        return torch.stack(encoded, dim=1).unsqueeze(2).expand(-1, -1, patches, -1)


class TinyTimeMixerDecoder(nn.Module):
    """정적 벡터를 채널 축에 붙이고 혼합한 뒤 정적 출력 채널을 제거한다."""

    def __init__(self, cfg: TinyTimeMixerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.projection = (
            nn.Linear(cfg.d_model, cfg.decoder_d_model)
            if cfg.d_model != cfg.decoder_d_model else nn.Identity()
        )
        self.raw_projection = (
            nn.Linear(cfg.patch_length, cfg.decoder_d_model)
            if cfg.decoder_raw_residual else None
        )
        static_count = len(cfg.categorical_vocab_size_list)
        self.static_embedding = (
            StaticCategoricalEmbedding(cfg.categorical_vocab_size_list, cfg.decoder_d_model)
            if static_count else None
        )
        self.mixers = MixerStack(
            channels=cfg.num_input_channels + static_count,
            patches=cfg.encoded_patches, d_model=cfg.decoder_d_model,
            depth=cfg.decoder_num_layers, levels=0, expansion=cfg.expansion_factor,
            dropout=cfg.dropout, gated=cfg.gated_attention,
            mix_channels=cfg.decoder_mode == "mix_channel",
        )

    def forward(self, hidden: torch.Tensor, patch_values: torch.Tensor,
                static_categorical_values: torch.Tensor | None = None) -> torch.Tensor:
        decoded = self.projection(hidden)
        if self.raw_projection is not None:
            raw = self.raw_projection(patch_values)
            if self.cfg.resolution_vocab_size:
                # 주기 토큰에는 대응하는 원시 패치가 없으므로 잔차를 0으로 둔다.
                raw = torch.cat((torch.zeros_like(raw[:, :, :1, :]), raw), dim=2)
            decoded = decoded + raw
        if self.static_embedding is not None:
            if static_categorical_values is None:
                raise ValueError("정적 분기가 활성화된 모델에는 static_categorical_values가 필요합니다.")
            static = self.static_embedding(
                static_categorical_values, batch_size=decoded.shape[0],
                device=decoded.device, patches=decoded.shape[2],
            )
            decoded = torch.cat((decoded, static), dim=1)
        elif static_categorical_values is not None:
            raise ValueError("정적 입력을 쓰려면 categorical_vocab_size_list를 설정해야 합니다.")
        decoded = self.mixers(decoded)
        # 법인·부품군의 가상 채널은 예측 대상이 아니다. 시계열 채널만 남긴다.
        return decoded[:, :self.cfg.num_input_channels, :, :]

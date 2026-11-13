"""마스킹 복원으로 TTM 계열 표현을 학습하는 자체 확장.

IBM 공개 가중치의 공식 사전학습 절차를 재현하는 코드가 아니다. 정적 특성을
포함한 복원 손실이 디코더와 백본에 전달되는 과정을 확인하기 위한 학습 경로다.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict

import torch
from torch import nn

from .backbone import TinyTimeMixerBackbone
from .configs import TinyTimeMixerConfig
from .decoder import TinyTimeMixerDecoder
from .TinyTimeMixer import TinyTimeMixerModel, resolve_config


class TinyTimeMixerPretrainModel(nn.Module):
    """숨긴 패치의 관측값만 대상으로 정규화된 복원 오차를 최소화한다.

    겹친 패치에는 같은 원시값이 다시 등장할 수 있으므로 이 구현의 복원
    학습은 비중첩·정확 분할 패치만 허용한다. 예측 모델에는 이 제약이 없다.
    """

    def __init__(self, cfg: TinyTimeMixerConfig | Mapping) -> None:
        super().__init__()
        self.cfg = resolve_config(cfg)
        if (self.cfg.patch_stride != self.cfg.patch_length
                or self.cfg.lookback % self.cfg.patch_length or self.cfg.num_patches < 2):
            raise ValueError("복원 사전학습은 2개 이상의 비중첩·정확 분할 패치를 요구합니다.")
        self.backbone = TinyTimeMixerBackbone(self.cfg)
        self.decoder = TinyTimeMixerDecoder(self.cfg) if self.cfg.use_decoder else None
        width = self.cfg.decoder_d_model if self.cfg.use_decoder else self.cfg.d_model
        self.reconstruction_head = nn.Linear(width, self.cfg.patch_length)

    def make_patch_mask(self, x: torch.Tensor, mask_ratio: float,
                        generator: torch.Generator | None = None) -> torch.Tensor:
        """각 배치·채널에서 최소 한 패치를 숨기고 최소 한 패치를 남긴다."""
        if not math.isfinite(mask_ratio) or not 0 < mask_ratio < 1:
            raise ValueError("mask_ratio는 0보다 크고 1보다 작아야 합니다.")
        patches = self.cfg.num_patches
        count = min(patches - 1, max(1, int(patches * mask_ratio)))
        # 난수 생성기의 장치를 입력과 맞춰 CPU/CUDA 재현성을 명시적으로 제어한다.
        scores = torch.rand(x.shape[0], self.cfg.num_input_channels, patches,
                            device=x.device, generator=generator)
        selected = scores.argsort(dim=-1)[..., :count]
        return torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, selected, True)

    def forward(self, x: torch.Tensor, *,
                static_categorical_values: torch.Tensor | None = None,
                past_observed_mask: torch.Tensor | None = None,
                patch_mask: torch.Tensor | None = None,
                mask_ratio: float = 0.3,
                freq_token: torch.Tensor | None = None,
                generator: torch.Generator | None = None,
                loss_type: str = "mse") -> dict[str, torch.Tensor]:
        observed = self.backbone.observed_mask(x, past_observed_mask)
        if patch_mask is None:
            patch_mask = self.make_patch_mask(x, mask_ratio, generator)
        expected = (x.shape[0], self.cfg.num_input_channels, self.cfg.num_patches)
        if (patch_mask.shape != expected or patch_mask.dtype != torch.bool
                or patch_mask.device != x.device):
            raise ValueError("patch_mask는 (B,C,P) 크기의 같은 장치 bool 텐서여야 합니다.")
        counts = patch_mask.sum(dim=-1)
        if torch.any((counts == 0) | (counts == self.cfg.num_patches)):
            raise ValueError("각 채널에 숨긴 패치와 보이는 패치가 모두 필요합니다.")
        if loss_type not in {"mse", "mae"}:
            raise ValueError("loss_type은 mse 또는 mae여야 합니다.")

        time_mask = patch_mask.repeat_interleave(self.cfg.patch_length, dim=-1).transpose(1, 2)
        visible = observed & ~time_mask
        # 숨긴 값은 평균·분산 계산, 패치 투영, 디코더 원시 잔차에 모두 사용하지 않는다.
        encoded = self.backbone(x, past_observed_mask=visible,
                                patch_mask=patch_mask, freq_token=freq_token)
        if self.decoder is not None:
            decoded = self.decoder(encoded.hidden_state, encoded.patch_values, static_categorical_values)
        else:
            if static_categorical_values is not None:
                raise ValueError("디코더가 없는 모델은 정적 입력을 사용할 수 없습니다.")
            decoded = encoded.hidden_state
        if self.cfg.resolution_vocab_size:
            decoded = decoded[:, :, 1:, :]
        normalized_reconstruction = self.reconstruction_head(decoded)
        normalized_reconstruction = normalized_reconstruction.flatten(start_dim=2).transpose(1, 2)
        normalized_target = torch.where(observed, (x - encoded.location) / encoded.scale, 0)
        scored = time_mask & observed
        if not torch.any(scored):
            raise ValueError("숨긴 위치에 복원 손실을 계산할 실제 관측값이 없습니다.")
        # 인덱싱을 먼저 하여 결측 위치의 NaN이 손실이나 기울기에 섞이지 않게 한다.
        error = normalized_reconstruction[scored] - normalized_target[scored]
        loss = error.square().mean() if loss_type == "mse" else error.abs().mean()
        return {
            "loss": loss,
            "reconstruction": normalized_reconstruction * encoded.scale + encoded.location,
            "mask": scored,
            "patch_mask": patch_mask,
        }

    def transfer_to(self, target: TinyTimeMixerModel, *, freeze_backbone: bool = False) -> None:
        """백본·디코더를 옮기고 예측 헤드는 대상 모델의 초기값으로 남긴다.

        예측 길이는 바꿀 수 있지만 나머지 설정은 같아야 한다. 모든 설정을
        먼저 확인하므로 불일치로 거절된 대상의 일부 가중치가 바뀌지 않는다.
        """
        source_cfg, target_cfg = asdict(self.cfg), asdict(target.cfg)
        source_cfg.pop("horizon")
        target_cfg.pop("horizon")
        if source_cfg != target_cfg:
            raise ValueError("사전학습 전이는 horizon을 제외한 설정이 일치해야 합니다.")
        target.backbone.load_state_dict(self.backbone.state_dict(), strict=True)
        if self.decoder is not None:
            target.decoder.load_state_dict(self.decoder.state_dict(), strict=True)
        target.freeze_backbone(freeze_backbone)

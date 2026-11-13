"""TTM 계열 연구 구현의 입력 계약과 구조 설정.

공식 IBM 체크포인트의 설정을 그대로 읽는 어댑터가 아니다. 이 설정으로
생성하는 모든 가중치는 무작위 초기화되며, 학습 데이터도 사용자가 공급한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TinyTimeMixerConfig:
    """주간 52주 입력과 26주 출력을 기본으로 하는 모델 설정.

    내부 텐서는 (B, C, P, D) 순서다. B는 배치, C는 시계열 채널,
    P는 패치, D는 은닉 차원이며, 법인이나 부품은 일반적으로 B에 배치한다.
    """

    lookback: int = 52
    horizon: int = 26
    num_input_channels: int = 1
    patch_length: int = 4
    patch_stride: int = 4
    d_model: int = 64
    num_layers: int = 2
    expansion_factor: int = 2
    dropout: float = 0.1
    gated_attention: bool = True
    # 2이면 은닉 차원을 나누는 단계 1, 원래 해상도 단계 0을 차례로 적용한다.
    adaptive_patching_levels: int = 2
    use_norm: bool = True
    norm_eps: float = 1e-5
    # 0이면 주기 토큰을 사용하지 않는다. 활성화하면 샘플별 정수 토큰이 필요하다.
    resolution_vocab_size: int = 0

    use_decoder: bool = True
    decoder_d_model: int = 32
    decoder_num_layers: int = 2
    decoder_mode: str = "mix_channel"
    decoder_raw_residual: bool = True
    # 예: (10, 30)이면 법인 10종, 부품군 30종. UNK를 쓰면 그 자리도 포함한다.
    categorical_vocab_size_list: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        positive = (
            "lookback", "horizon", "num_input_channels", "patch_length",
            "patch_stride", "d_model", "num_layers", "expansion_factor",
            "decoder_d_model", "decoder_num_layers",
        )
        for name in positive:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name}은 양의 정수여야 합니다.")
        for name in ("adaptive_patching_levels", "resolution_vocab_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name}은 0 이상의 정수여야 합니다.")
        if self.patch_length > self.lookback:
            raise ValueError("patch_length는 lookback보다 클 수 없습니다.")
        if self.patch_stride > self.patch_length:
            raise ValueError("관측 구간이 누락되지 않도록 patch_stride <= patch_length가 필요합니다.")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout은 0 이상 1 미만이어야 합니다.")
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("norm_eps는 유한한 양수여야 합니다.")
        factor = 2 ** max(0, self.adaptive_patching_levels - 1)
        if self.d_model % factor or self.d_model // factor < 4:
            raise ValueError("적응형 패치의 각 단계에서 d_model은 나누어떨어지고 4 이상이어야 합니다.")
        if self.decoder_mode not in {"common_channel", "mix_channel"}:
            raise ValueError("decoder_mode는 common_channel 또는 mix_channel이어야 합니다.")
        if not isinstance(self.categorical_vocab_size_list, (list, tuple)):
            raise ValueError("categorical_vocab_size_list는 범주 수의 리스트 또는 튜플이어야 합니다.")
        sizes = tuple(self.categorical_vocab_size_list)
        if any(type(size) is not int or size < 1 for size in sizes):
            raise ValueError("각 범주 수는 양의 정수여야 합니다.")
        object.__setattr__(self, "categorical_vocab_size_list", sizes)
        if sizes and (not self.use_decoder or self.decoder_mode != "mix_channel"):
            raise ValueError("정적 특성을 쓰려면 use_decoder=True, decoder_mode='mix_channel'이 필요합니다.")

    @property
    def num_patches(self) -> int:
        """최근 시점까지 포함하기 위해 필요한 패치 수."""
        distance = self.lookback - self.patch_length
        return (distance + self.patch_stride - 1) // self.patch_stride + 1

    @property
    def left_padding(self) -> int:
        """패치 정렬 시 앞쪽에 붙이는 길이. 최근 관측값은 잘라내지 않는다."""
        covered = (self.num_patches - 1) * self.patch_stride + self.patch_length
        return covered - self.lookback

    @property
    def encoded_patches(self) -> int:
        return self.num_patches + int(self.resolution_vocab_size > 0)

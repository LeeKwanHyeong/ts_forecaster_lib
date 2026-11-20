"""공식 체크포인트와 무관한 소형 무작위 초기화 설정."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re


TRAIN_TRANSFORMERS_VERSION = "4.57.6"
DEFAULT_TRAIN_QUANTILES = (0.01,) + tuple(i / 20 for i in range(1, 20)) + (0.99,)


@dataclass(frozen=True)
class Chronos2TrainableConfig:
    """기본 크기는 CPU 계약 검증용이며 운영 모델 크기나 최적값을 의미하지 않음."""

    context_length: int = 52
    horizon: int = 27
    patch_size: int = 16
    d_model: int = 128
    d_ff: int = 256
    num_layers: int = 2
    num_heads: int = 4
    d_kv: int = 32
    dropout_rate: float = 0.1
    initializer_factor: float = 0.05
    quantile_levels: tuple[float, ...] = DEFAULT_TRAIN_QUANTILES
    seed: int = 42
    device: str = "cpu"
    use_arcsinh: bool = True

    def __post_init__(self):
        for name in ("context_length", "horizon", "patch_size", "d_model", "d_ff", "num_layers", "num_heads", "d_kv"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name}은 양의 정수여야 합니다.")
        if self.d_model != self.num_heads * self.d_kv:
            raise ValueError("d_model은 num_heads * d_kv와 일치해야 합니다.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed는 0 이상 2**63 미만의 정수여야 합니다.")
        for name in ("dropout_rate", "initializer_factor"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name}은 유한한 실수여야 합니다.")
        if not 0 <= self.dropout_rate < 1 or self.initializer_factor <= 0:
            raise ValueError("dropout_rate는 [0,1), initializer_factor는 양수여야 합니다.")
        if type(self.use_arcsinh) is not bool:
            raise TypeError("use_arcsinh는 bool이어야 합니다.")
        if not isinstance(self.device, str) or not re.fullmatch(r"cpu|mps|cuda(?::[0-9]+)?", self.device):
            raise ValueError("device는 cpu, mps, cuda 또는 cuda:N이어야 합니다.")
        if not isinstance(self.quantile_levels, (tuple, list)):
            raise TypeError("quantile_levels는 수치 리스트 또는 튜플이어야 합니다.")
        levels = tuple(self.quantile_levels)
        if any(type(q) not in (int, float) or not math.isfinite(q) or not 0 < q < 1 for q in levels):
            raise ValueError("분위수는 0과 1 사이의 유한한 수여야 합니다.")
        if not levels or tuple(sorted(set(levels))) != levels or 0.5 not in levels:
            raise ValueError("분위수는 중복 없이 오름차순이어야 하며 0.5를 포함해야 합니다.")
        object.__setattr__(self, "quantile_levels", levels)

    @property
    def num_output_patches(self) -> int:
        return math.ceil(self.horizon / self.patch_size)

    def native_kwargs(self) -> dict:
        """설정만 생성하며 파일·Hub·기존 모델에 접근하지 않는 변환."""
        return {"d_model": self.d_model, "d_ff": self.d_ff, "num_layers": self.num_layers,
                "num_heads": self.num_heads, "d_kv": self.d_kv, "dropout_rate": self.dropout_rate,
                "initializer_factor": self.initializer_factor, "attn_implementation": "sdpa",
                "chronos_pipeline_class": "Chronos2Pipeline", "architectures": ["Chronos2Model"],
                "chronos_config": {"context_length": self.context_length,
                    "input_patch_size": self.patch_size, "input_patch_stride": self.patch_size,
                    "output_patch_size": self.patch_size, "max_output_patches": self.num_output_patches,
                    "quantiles": list(self.quantile_levels), "use_reg_token": True,
                    "use_arcsinh": self.use_arcsinh, "time_encoding_scale": self.context_length}}

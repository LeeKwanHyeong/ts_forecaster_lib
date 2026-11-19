"""공식 Chronos-2 가중치와 독립 zero-shot 실행 설정."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


OFFICIAL_MODEL_ID = "amazon/chronos-2"
OFFICIAL_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
CHRONOS_PACKAGE_VERSION = "2.3.2"
OFFICIAL_WEIGHTS_SHA256 = "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"
OFFICIAL_CONFIG_SHA256 = "ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031"


@dataclass(frozen=True)
class Chronos2Config:
    """모델 생성만으로 네트워크에 접근하지 않는 추론 전용 설정."""

    model_id: str = OFFICIAL_MODEL_ID
    revision: str = OFFICIAL_REVISION
    model_path: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = True
    expected_weights_sha256: str | None = OFFICIAL_WEIGHTS_SHA256
    expected_config_sha256: str | None = OFFICIAL_CONFIG_SHA256
    context_length: int = 52
    horizon: int = 27
    batch_size: int = 32
    device: str = "cpu"
    dtype: str = "float32"
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9)

    def __post_init__(self) -> None:
        for name in ("context_length", "horizon", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name}은 양의 정수여야 합니다.")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id가 필요합니다.")
        if not isinstance(self.revision, str) or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("revision은 고정된 40자리 커밋 SHA여야 합니다.")
        if type(self.local_files_only) is not bool:
            raise TypeError("local_files_only는 bool이어야 합니다.")
        for name in ("expected_weights_sha256", "expected_config_sha256"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)):
                raise ValueError(f"{name}은 64자리 SHA-256 또는 None이어야 합니다.")
        if not isinstance(self.device, str) or not re.fullmatch(r"cpu|mps|cuda(?::[0-9]+)?", self.device):
            raise ValueError("device는 cpu, mps, cuda 또는 cuda:N이어야 합니다.")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype은 float32, float16, bfloat16 중 하나여야 합니다.")
        for name in ("model_path", "cache_dir"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name}은 경로 문자열이어야 합니다.")
        if not isinstance(self.quantile_levels, (tuple, list)):
            raise TypeError("quantile_levels는 리스트 또는 튜플이어야 합니다.")
        levels = tuple(self.quantile_levels)
        if any(type(q) not in (int, float) or not math.isfinite(q) or not 0 < q < 1 for q in levels):
            raise ValueError("분위수는 0과 1 사이의 유한한 수여야 합니다.")
        if not levels or tuple(sorted(set(levels))) != levels or 0.5 not in levels:
            raise ValueError("분위수는 중복 없이 오름차순이어야 하며 0.5를 포함해야 합니다.")
        object.__setattr__(self, "quantile_levels", levels)

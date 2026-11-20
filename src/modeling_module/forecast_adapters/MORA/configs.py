"""논문 수식의 명시적 실험 설정. 논문의 전체 설정 탐색을 자동 재현하지 않음."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MORAConfig:
    lookback: int = 52
    horizon: int = 27
    period: int = 52
    coarse_max: int = 64
    min_bin_size: int = 4
    correction_rank: int = 16
    taper_steps: int = 4
    ridge_strength: float = 10.0
    shrinkage: float = 0.5
    structured_mix: float = 0.25
    noise_scale: float = 1.0
    weight_power: float = 0.0
    gain_max: float = 1.5
    folds: int = 4
    min_fold_train: int = 8
    calibration_groups: int = 4
    calibration_max: float = 1.5
    calibration_iterations: int = 8
    scope: str = "single_series"

    def __post_init__(self):
        integers = ("lookback", "horizon", "period", "coarse_max", "min_bin_size", "correction_rank",
                    "taper_steps", "folds", "min_fold_train", "calibration_groups", "calibration_iterations")
        for name in integers:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name}은 양의 정수여야 합니다.")
        if self.lookback < 24 or self.horizon < self.min_bin_size or not 1 <= self.correction_rank < self.horizon:
            raise ValueError("v1은 L>=24, H>=min_bin_size, 1<=correction_rank<H가 필요합니다.")
        if not 2 <= self.taper_steps <= self.horizon or self.folds < 3 or self.calibration_groups > self.correction_rank:
            raise ValueError("taper·fold·보정 그룹 수가 유효하지 않습니다.")
        for name in ("ridge_strength", "shrinkage", "structured_mix", "noise_scale", "weight_power", "gain_max", "calibration_max"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name}은 유한한 실수여야 합니다.")
        if self.ridge_strength <= 0 or self.noise_scale <= 0 or self.gain_max <= 0 or self.calibration_max <= 0:
            raise ValueError("정규화 강도·noise_scale·gain 상한은 양수여야 합니다.")
        if not 0 <= self.shrinkage <= 1 or not 0 <= self.structured_mix <= 1 or not 0 <= self.weight_power <= 1:
            raise ValueError("shrinkage·structured_mix·weight_power는 [0,1] 범위여야 합니다.")
        if self.scope not in {"single_series", "pooled_series"}:
            raise ValueError("scope는 single_series 또는 명시적 연구 확장인 pooled_series여야 합니다.")

    @property
    def coarse_count(self):
        return min(self.coarse_max, self.horizon // self.min_bin_size)

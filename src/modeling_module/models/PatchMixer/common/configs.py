from typing import Any, List, Literal, Mapping, Optional, Tuple
from dataclasses import dataclass, field

from modeling_module.training.config import TrainingConfig, DecompositionConfig


_MISSING = object()


def _config_value(config: Any, *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if isinstance(config, Mapping) and name in config:
            return config[name]
        if hasattr(config, name):
            return getattr(config, name)
    if default is not _MISSING:
        return default
    joined = " or ".join(repr(name) for name in names)
    raise ValueError(f"PatchMixerOriginal config requires {joined}.")


@dataclass(frozen=True)
class PatchMixerOriginalConfig:
    """Architecture fields required by the canonical upstream point model."""

    lookback: int
    horizon: int
    enc_in: int = 1
    patch_len: int = 16
    stride: int = 8
    mixer_kernel_size: int = 8
    d_model: int = 256
    e_layers: int = 1
    dropout: float = 0.2
    head_dropout: float = 0.0
    use_revin: bool = True
    revin_affine: bool = True
    revin_subtract_last: bool = False

    def __post_init__(self) -> None:
        positive_fields = {
            "lookback": self.lookback,
            "horizon": self.horizon,
            "enc_in": self.enc_in,
            "patch_len": self.patch_len,
            "stride": self.stride,
            "mixer_kernel_size": self.mixer_kernel_size,
            "d_model": self.d_model,
            "e_layers": self.e_layers,
        }
        for name, value in positive_fields.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.patch_len > self.lookback:
            raise ValueError(
                f"patch_len must not exceed lookback, got {self.patch_len} > {self.lookback}."
            )
        for name, value in (
            ("dropout", self.dropout),
            ("head_dropout", self.head_dropout),
        ):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}.")

    @property
    def seq_len(self) -> int:
        return self.lookback

    @property
    def pred_len(self) -> int:
        return self.horizon

    @classmethod
    def from_config(cls, config: Any) -> "PatchMixerOriginalConfig":
        if isinstance(config, cls):
            return config
        return cls(
            lookback=int(_config_value(config, "lookback", "seq_len")),
            horizon=int(_config_value(config, "horizon", "pred_len")),
            enc_in=int(_config_value(config, "enc_in", default=1)),
            patch_len=int(_config_value(config, "patch_len", default=16)),
            stride=int(_config_value(config, "stride", default=8)),
            mixer_kernel_size=int(
                _config_value(config, "mixer_kernel_size", default=8)
            ),
            d_model=int(_config_value(config, "d_model", default=256)),
            e_layers=int(_config_value(config, "e_layers", default=1)),
            dropout=float(_config_value(config, "dropout", default=0.2)),
            head_dropout=float(_config_value(config, "head_dropout", default=0.0)),
            use_revin=bool(_config_value(config, "use_revin", default=True)),
            revin_affine=bool(
                _config_value(config, "revin_affine", default=True)
            ),
            revin_subtract_last=bool(
                _config_value(config, "revin_subtract_last", default=False)
            ),
        )


# =========================
# PatchMixer 전용 설정
# =========================
@dataclass
class PatchMixerConfig(TrainingConfig):
    """
    PatchMixer 모델 학습 및 구조 설정을 위한 구성 클래스.

    주요 기능:
    - 백본(Backbone) 용량 및 패치(Patch) 구조 정의.
    - 시간적 확장(Temporal Expander) 및 주기성(Seasonality) 하이퍼파라미터 설정.
    - 과거/미래 외생 변수(Exogenous Variables) 및 파트 임베딩 처리.
    - 출력 제약(Non-negative) 및 멀티스케일 옵션 관리.
    """
    # ---------- 입력 데이터 설정 ----------
    enc_in: int = 1  # 입력 채널 수 (단변량 시계열 권장)

    # ---------- 백본(Backbone) 구조 설정 ----------
    d_model: int = 64  # 모델 내부 은닉 차원(Hidden Dimension) 크기
    e_layers: int = 3  # 인코더 레이어 수
    patch_len: int = 12  # 시계열 패치 길이
    stride: int = 8  # 패치 생성을 위한 스트라이드(Stride) 간격
    dropout: float = 0.1  # 드롭아웃 비율
    mixer_kernel_size: int = 5  # Mixer 모듈의 커널 크기

    # ---------- Temporal Expander (시간적 특징 확장) ----------
    f_out: int = 128  # Expander 출력 특성 차원
    expander_season_period: int = 52  # 주 계절성 주기 (예: 주간 데이터 52)
    expander_n_harmonics: int = 8  # 사용할 고조파(Harmonics) 개수
    expander_max_harmonics: int = 16  # 최대 고조파 개수 (호환성 유지용)


    # ---------- 파트(Part) 임베딩 ----------
    use_part_embedding: bool = False  # 파트(ID) 임베딩 사용 여부
    part_vocab_size: int = 0  # 파트 고유 개수 (임베딩 테이블 크기)
    part_embed_dim: int = 16  # 파트 임베딩 벡터 차원

    # ---------- 출력 헤드 및 제약 조건 ----------
    final_nonneg: bool = True  # 추론 결과의 음수 방지(Clamp) 적용 여부
    head_hidden: int = 128  # 출력 헤드 내부 은닉층 크기

    # ---------- EOL (End-of-Life) 편향 ----------
    use_eol_prior: bool = False  # 수명 주기(EOL) 편향 로직 사용 여부
    eol_feature_index: int = 0  # Future Exo 내 EOL 관련 피처 인덱스

    # ---------- 멀티스케일(Multi-scale) 백본 옵션 ----------
    patch_cfgs: tuple = ()  # 다중 스케일 패치 설정 튜플
    per_branch_dim: int = 64  # 브랜치별 차원 크기
    fused_dim: int = 128  # 브랜치 병합 후 차원 크기
    fusion: str = "concat"  # 병합 방식 ('concat' 등)
    head_dropout: float = 0.0  # 헤드 레이어 드롭아웃 비율

    # 호환성 필드 (일부 레거시 구현 지원용)
    season_period: int = 52
    max_harmonics: int = 16

    # 시계열 분해(Decomposition) 설정 (RevIN 등에서 참조 가능)
    decomp: DecompositionConfig = field(default_factory=DecompositionConfig)

    # ---------- 미래 외생 변수 (Future Exogenous) ----------
    future_exo_dim: int = 0  # 미래 외생 변수 차원 (0일 경우 미사용)
    exo_is_normalized_default: bool = False  # 외생 변수 정규화 여부 기본값

    # ---------- 과거 외생 변수 (Past Exogenous) ----------
    # 주의: 체크포인트 저장/로드 시 누락 방지를 위해 타입 어노테이션 필수
    past_exo_mode: str = "z_gate"  # 과거 외생 처리 모드 ('none' | 'z_gate')
    past_exo_cont_dim: int = 0  # 연속형 과거 외생 변수 차원 (B, L, E_p)
    past_exo_cat_dim: int = 0  # 범주형 과거 외생 변수 차원 (B, L, K)
    past_exo_cat_vocab_sizes: Tuple[int, ...] = ()  # 범주형 변수별 어휘(Vocab) 크기
    past_exo_cat_embed_dims: Tuple[int, ...] = ()  # 범주형 변수별 임베딩 차원

    # ---------- 학습 안정화 옵션 ----------
    learn_output_scale: bool = True  # 출력 스케일 파라미터 학습 여부
    learn_dw_gain: bool = True  # Depthwise Conv 이득(Gain) 학습 여부

    use_revin: bool = True
    q_clip_norm = 10.0



# =========================
# 프리셋: 월간/주간
# =========================
@dataclass
class PatchMixerConfigMonthly(PatchMixerConfig):
    """
    월간(Monthly) 데이터 전용 프리셋 설정.
    특징:
    - 계절성 주기 12개월 설정.
    - 월별 Sin/Cos 외생 변수 사용 시 exo_dim=2 권장.
    """
    expander_season_period: int = 12
    expander_n_harmonics: int = 6


@dataclass
class PatchMixerConfigWeekly(PatchMixerConfig):
    """
    주간(Weekly) 데이터 전용 프리셋 설정.
    특징:
    - 계절성 주기 52주 설정.
    - 주차별(Week of Year) Sin/Cos 외생 변수(exo_dim=2) 및 EOL Proxy 사용 적합.
    """
    expander_season_period: int = 52
    expander_n_harmonics: int = 8

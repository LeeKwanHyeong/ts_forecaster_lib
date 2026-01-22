from dataclasses import dataclass, field, replace
from typing import Tuple, Optional, Literal
import torch


@dataclass
class DecompositionConfig:
    """
    시계열 분해(Decomposition) 전처리 및 특징 결합 설정.

    기능:
    - 입력 시계열을 추세(Trend)와 잔차(Residual)로 분리하여 모델 입력으로 활용.
    - 이동평균(MA) 또는 STL 분해 방식 지원.
    """
    type: Optional[Literal["none", "ma", "stl"]] = "none"  # 분해 알고리즘 선택 ('none': 미사용, 'ma': 이동평균, 'stl': STL)
    ma_window: int = 7  # 이동평균 윈도우 크기 (MA 방식 사용 시)
    stl_period: int = 12  # 계절성 주기 (STL 방식 사용 시)
    concat_mode: Literal["concat", "residual_only", "trend_only"] = "concat"  # 분해된 성분의 결합 방식 (원본+분해성분 병합 등)


# ---------------- Spike-aware Loss Configs ---------------- #
@dataclass
class SpikeLossConfig:
    """
    스파이크(급격한 변화) 구간 학습 강화를 위한 손실 함수 설정.

    기능:
    - 이상치(Outlier) 및 스파이크 구간에 가중치를 부여하여 예측 성능 개선.
    - 비대칭(Asymmetric) 손실을 통해 과대/과소 예측 페널티 조절.
    - Huber Loss를 활용한 강건성(Robustness) 확보.
    """
    enabled: bool = True  # 스파이크 특화 손실 함수 사용 여부 활성화
    strategy: Literal['mix', 'direct'] = 'mix'
    # 'mix': WeightedHuber + AsymmetricMSE + Baseline 혼합 사용
    # 'direct': 단일 비대칭 함수 사용

    # 공통 파라미터 (Robustness & Asymmetry)
    huber_delta: float = 2.0  # Huber Loss의 선형/제곱 전환 임계값 (이상치 민감도 제어)
    asym_up_weight: float = 2.0  # 과대 예측(Pred > True) 시 적용할 페널티 가중치
    asym_down_weight: float = 1.0  # 과소 예측(Pred < True) 시 적용할 페널티 가중치

    # Mix 전략 전용 (Spike Detection & Weighting)
    mad_k: float = 3.5  # 스파이크 탐지를 위한 MAD(Median Absolute Deviation) 계수 (높을수록 엄격)
    w_spike: float = 6.0  # 탐지된 스파이크 구간 샘플에 부여할 손실 가중치
    w_norm: float = 1.0  # 일반 구간 샘플 가중치
    w_cap = 12.0  # 가중치의 최대 상한값 (불안정 방지)
    alpha_huber: float = 0.7  # 혼합 손실에서 WeightedHuber Loss의 반영 비율
    beta_asym: float = 0.3  # 혼합 손실에서 Asymmetric MSE의 반영 비율

    # Baseline 혼합 설정 (Stability)
    mix_with_baseline: bool = False  # 기존 기본 손실(MSE/MAE)과의 혼합 여부
    gamma_baseline: float = 0.2  # 기본 손실의 혼합 비율 (mix_with_baseline=True 시)

    def with_enabled(self, flag: bool) -> "SpikeLossConfig":
        """설정의 활성화 상태(enabled)를 변경한 새로운 인스턴스 반환."""
        return replace(self, enabled=bool(flag))


@dataclass
class TrainingConfig:
    """
    모델 학습 전반을 제어하는 하이퍼파라미터 통합 설정 클래스.

    구성:
    - 하드웨어 및 데이터 로더 설정 (Loader).
    - 최적화 및 스케줄링 설정 (Training).
    - 손실 함수 및 가중치 설정 (Loss & Weights).
    - 외생 변수 및 특수 목적(Spike, Intermittent) 처리 설정.
    """
    # ------------Loader (데이터 로딩 및 연산 환경)------------
    device: str = 'cuda' if torch.cuda.is_available() else 'mps'  # 학습 연산 장치 (Mac MPS 지원 포함)
    log_every: int = 100  # 로그 출력 주기 (Step 단위)
    use_amp: bool = torch.cuda.is_available()  # 자동 혼합 정밀도(AMP) 사용 여부
    lookback: int = 54  # 모델 입력(과거) 시퀀스 길이
    horizon: int = 27  # 모델 예측(미래) 시퀀스 길이

    # ------------Training (학습 루프 및 최적화)------------
    epochs: int = 1  # 총 학습 에폭 수
    lr: float = 1e-4  # 초기 학습률 (Learning Rate)
    weight_decay: float = 1e-4  # 가중치 감쇠 (L2 Regularization)
    t_max: int = 10  # CosineAnnealingLR 스케줄러의 주기
    patience: int = 50  # 조기 종료(Early Stopping) 허용 횟수
    max_grad_norm: float = 30.0  # 그라디언트 클리핑 임계값
    amp_device: str = 'cuda'  # AMP 수행 장치 유형

    # ------------Loss (손실 함수 설정)-------------
    loss_mode: Literal['auto', 'point', 'quantile'] = 'auto'  # 손실 계산 모드 선택
    point_loss: Literal['mae', 'mse', 'huber', 'pinball', 'huber_asym', None] = 'mse'  # 점 예측용 손실 함수
    huber_delta: float = 5.0  # Huber Loss의 임계값 (L1/L2 전환점)
    q_star: float = 0.5  # 점 예측 시 타겟 분위수 (Pinball Loss 사용 시)
    use_cost_q_star: bool = False  # 비용 기반 비대칭 가중치(Newsvendor) 사용 여부
    Cu: float = 1.0;
    Co: float = 1.0  # 언더/오버 예측에 대한 비용 가중치 (Cost Under/Over)
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)  # 확률 예측 시 타겟 분위수 목록

    # ------------Intermittent/Horizon Weight (가중치 보정)-------------
    use_intermittent: bool = True  # 간헐적 수요(0이 많은 데이터)에 대한 가중치 보정 활성화
    alpha_zero: float = 1.2  # 0인 구간에 대한 가중치
    alpha_pos: float = 1.0  # 양수 구간에 대한 가중치
    gamma_run: float = 0.6  # 연속된 0 구간(Run-length)에 대한 감쇠 계수
    cap: Optional[float] = None  # 가중치 상한값 (None: 제한 없음)
    use_horizon_decay: bool = False  # 먼 미래 예측 오차에 대한 가중치 감쇠 적용 여부
    tau_h: float = 24.0  # Horizon Decay 반감기 (시간 상수)

    # ------------Validation Weight (검증 설정)-------------
    val_use_weights: bool = False  # 검증 시 학습 가중치(Intermittent 등) 적용 여부 (공정 평가 위해 보통 False)

    # ------------Exogenous Value (외생 변수) --------------
    exo_dim = 2  # 미래 외생 변수 차원
    nonneg_head = False  # 출력단의 비음수(Non-negative) 제약 강제 여부 (Softplus)
    use_exogenous_mode: bool = False  # 외생 변수 모드 사용 여부 마스터 스위치

    # ------------Spike-friendly Loss (스파이크 대응)-------------- #
    spike_loss: SpikeLossConfig = field(default_factory=SpikeLossConfig)  # 스파이크 특화 손실 설정

    lambda_hist_scale: float = 0.1  # 이력 데이터 스케일 규제 강도
    lambda_hist_var: float = 0.03  # 이력 데이터 분산 규제 강도
    hist_window: int = 12  # 규제 계산을 위한 과거 데이터 윈도우 크기

    anchor_last_k: int = 8  # 앵커(기준점)로 사용할 직전 과거 스텝 수
    anchor_weight: float = 0.05  # 앵커 손실(직전 과거와의 연속성 유지) 가중치

    def copy_with(self, **kwargs) -> "TrainingConfig":
        """
        설정 객체 복제 및 일부 필드 수정 유틸리티.
        dataclasses.replace의 래퍼(Wrapper) 역할.
        """
        return replace(self, **kwargs)


@dataclass
class StageConfig:
    """
    다단계 커리큘럼 학습(Curriculum Learning)을 위한 스테이지별 설정 정의.

    기능:
    - 각 학습 단계(Stage)마다 독립적으로 적용할 하이퍼파라미터 지정.
    - 베이스 설정(TrainingConfig)의 특정 필드를 덮어쓰기(Override) 위한 옵션 제공.
    """
    epochs: int  # 해당 스테이지에서 수행할 에폭 수

    # 선택적 오버라이드 옵션 (None일 경우 베이스 설정 유지)
    spike_enabled: Optional[bool] = None  # 스파이크 손실 활성화 여부
    lr: Optional[float] = None  # 학습률(Learning Rate) 변경
    use_horizon_decay: Optional[bool] = None  # Horizon Decay 가중치 사용 여부
    tau_h: Optional[float] = None  # Horizon Decay 반감기
    huber_delta: Optional[float] = None  # Huber Loss 임계값

    # 스파이크 손실 세부 파라미터 튜닝
    w_spike: Optional[float] = None  # 스파이크 구간 가중치
    mad_k: Optional[float] = None  # 스파이크 탐지 민감도
    asym_down_weight: Optional[float] = None  # 과소 예측 페널티 가중치


def apply_stage(base: TrainingConfig, stg: StageConfig) -> TrainingConfig:
    """
    베이스 학습 설정에 현재 스테이지 설정을 병합(Merge) 및 적용.

    기능:
    - 베이스 설정 객체 복제.
    - StageConfig에서 정의된(None이 아닌) 속성들로 값을 갱신.
    - 중첩된 설정(SpikeLossConfig) 내부 속성까지 상세 매핑 수행.
    """
    cfg = replace(base)

    # 일반 학습 설정 오버라이드
    if stg.spike_enabled is not None:
        cfg.spike_loss.enabled = stg.spike_enabled
    if stg.lr is not None:
        cfg.lr = stg.lr
    if stg.use_horizon_decay is not None:
        cfg.use_horizon_decay = stg.use_horizon_decay
    if stg.tau_h is not None:
        cfg.tau_h = stg.tau_h
    if stg.huber_delta is not None:
        cfg.huber_delta = stg.huber_delta

    # 스파이크 손실 내부 설정 오버라이드
    if stg.w_spike is not None:
        cfg.spike_loss.w_spike = stg.w_spike
    if stg.mad_k is not None:
        cfg.spike_loss.mad_k = stg.mad_k
    if stg.asym_down_weight is not None:
        cfg.spike_loss.asym_down_weight = stg.asym_down_weight

    # 현재 스테이지의 에폭 수 설정
    cfg.epochs = stg.epochs
    return cfg
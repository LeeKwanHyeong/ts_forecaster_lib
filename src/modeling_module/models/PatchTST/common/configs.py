from typing import Optional, Literal, Tuple, List
from dataclasses import dataclass, field

from modeling_module.training.config import TrainingConfig, DecompositionConfig


# =========================
# Attention (Self-Attention) 설정
# =========================
@dataclass
class AttentionConfig(TrainingConfig):
    """
    PatchTST 모델의 어텐션 메커니즘을 제어하는 설정 클래스.

    기능:
    - 어텐션 유형(Full vs ProbSparse) 선택.
    - 멀티 헤드 구조 및 차원 설정.
    - 드롭아웃 및 인과적 마스킹(Causal Masking) 제어.
    """
    type: Literal['full', 'probsparse'] = 'full'  # 어텐션 연산 방식 ('full': 표준, 'probsparse': 효율적 희소 어텐션)
    n_heads: int = 16  # 멀티 헤드(Multi-head) 개수 (d_model의 약수 권장)
    d_model: int = 128  # 모델 내부 은닉 차원 (Token Embedding Dimension)
    d_k: Optional[int] = None  # 헤드별 Query/Key 차원 (None: d_model // n_heads 자동 설정)
    d_v: Optional[int] = None  # 헤드별 Value 차원 (None: d_model // n_heads 자동 설정)
    attn_dropout: float = 0.0  # 어텐션 스코어(Score)에 적용할 드롭아웃 비율
    proj_dropout: float = 0.0  # 어텐션 출력 투영(Projection) 시 적용할 드롭아웃 비율
    causal: bool = True  # 인과적 마스킹 사용 여부 (미래 시점 정보 차단)
    residual_logits: bool = True  # Logit 단계의 잔차 연결(RealFormer 스타일) 사용 여부
    output_attention: bool = False  # 어텐션 가중치 맵 반환 여부 (시각화/디버깅용)
    factor: int = 5  # ProbSparse 어텐션의 샘플링 팩터 (Top-k 선택 비율 제어)
    lsa: bool = False  # Local Self-Attention 스케일 파라미터 학습 여부


# =========================
# Head 설정 (출력부)
# =========================
@dataclass
class HeadConfig:
    """
    모델의 최종 출력 헤드(Head) 구조 설정.

    옵션:
    - 'prediction' : 시계열 예측 전용 헤드.
    - 'flatten'    : 평탄화 후 선형 변환 (일반적인 회귀/분류).
    - 'pretrain'   : 자기지도 학습(Self-supervised Learning)용 재구성 헤드.
    """
    type: Literal['prediction', 'flatten', 'pretrain'] = 'flatten'
    individual: bool = False  # 변수별 독립적 헤드(Individual Head) 사용 여부
    head_dropout: float = 0.0  # 헤드 레이어 드롭아웃 비율
    y_range: Optional[Tuple[float, float]] = None  # 출력값 범위 제한 (Sigmoid Scaling) 설정


# =========================
# PatchTST 전역 설정
# =========================
@dataclass
class PatchTSTConfig(TrainingConfig):
    """
    PatchTST 모델의 전체 하이퍼파라미터 통합 구성 클래스.

    주요 구성:
    - 데이터/윈도우: 입력 채널, 패치 길이, 스트라이드 등 데이터 전처리 관련 설정.
    - 인코더/백본: 레이어 수, 차원, 정규화, 위치 임베딩 등 모델 구조 관련 설정.
    - 외생 변수: 과거/미래 외생 변수(Exogenous Variables) 차원 및 임베딩 설정.
    - 서브 모듈: 어텐션, 헤드, 분해(Decomposition) 모듈의 세부 설정 포함.
    """
    # ---------- 데이터/윈도우 설정 ----------
    c_in: int = 1  # 입력 변수(Channel) 개수 (단변량=1, 다변량>1)
    target_dim: int = 1  # 예측 대상 변수 개수 (일반적으로 c_in과 동일하거나 작음)
    patch_len: int = 8  # 패치 길이 (로컬 패턴 학습 단위)
    stride: int = patch_len // 2  # 패치 이동 간격 (Stride) - 중첩 비율 제어
    # lookback: int = 36                # (TrainingConfig 상속) 과거 컨텍스트 길이 (Encoder Input)
    # horizon: int = 48                 # (TrainingConfig 상속) 미래 예측 길이 (Decoder/Head Output)
    padding_patch: Optional[str] = None  # 패딩 전략 ('end': 마지막 패치 보존을 위한 패딩 추가)

    # ---------- 과거 외생 변수 (Past Exogenous) ----------
    d_past_cont: int = 0  # 과거 연속형 외생 변수 개수
    d_past_cat: int = 0  # 과거 범주형 외생 변수 개수
    cat_cardinalities: List[int] = field(default_factory=list)  # 범주형 변수별 카디널리티(고유값 수)
    d_cat_emb: int = 0  # 범주형 변수 임베딩 차원

    # ---------- 미래 외생 변수 (Future Exogenous) ----------
    d_future: int = 0  # 미래 연속형 외생 변수 개수 (예측 헤드에 주입)

    # ---------- 인코더(백본) 구조 설정 ----------
    n_layers: int = 3  # 인코더 블록(Layer) 개수
    d_model: int = 128  # 모델 은닉 차원 (Patch Embedding Dimension)
    d_ff: int = 256  # Feed-Forward Network(FFN) 내부 확장 차원
    norm: str = 'BatchNorm'  # 정규화 방식 ('BatchNorm', 'LayerNorm')
    dropout: float = 0.0  # 인코더 전반에 적용할 드롭아웃 비율
    act: str = 'gelu'  # 활성화 함수 ('gelu', 'relu')
    pre_norm: bool = False  # Pre-Normalization 적용 여부 (True: Pre-LN, False: Post-LN)
    store_attn: bool = False  # 어텐션 맵 저장 여부 (분석용)
    pe: str = 'zeros'  # 위치 임베딩(Positional Embedding) 유형 ('zeros', 'learned' 등)
    learn_pe: bool = True  # 위치 임베딩 파라미터 학습 여부
    use_revin: bool = True  # RevIN(분포 정규화) 모듈 사용 여부
    affine: bool = True  # RevIN 내 Affine 파라미터(Scale/Shift) 학습 여부
    subtract_last: bool = False  # RevIN 적용 시 마지막 시점 값 기준 보정 여부
    verbose: bool = False  # 상세 디버깅 로그 출력 여부

    # ---------- 헤드 기본 설정 ----------
    head_type = 'regression'  # 헤드 작업 유형 (기본: 회귀)
    head_dropout: float = 0.  # 헤드 드롭아웃 (HeadConfig와 중복 가능성 있으나 전역 설정으로 존재)
    individual = False  # 채널별 독립 헤드 사용 여부

    # ---------- 서브 모듈 상세 설정 ----------
    attn: AttentionConfig = field(default_factory=AttentionConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    decomp: DecompositionConfig = field(default_factory=DecompositionConfig)


# =========================
# 프리셋: 월간/주간 권장 윈도우
# =========================
@dataclass
class PatchTSTConfigMonthly(PatchTSTConfig):
    """
    월간(Monthly) 데이터 전용 프리셋 설정.

    권장 사항:
    - Patch Length: 12 (1년 주기)의 약수 또는 배수 권장.
    - Stride: Patch Length의 절반(50% Overlap) 권장.
    """
    # lookback: int = 36  # 예시: 3년
    # horizon: int = 48   # 예시: 4년
    pass


@dataclass
class PatchTSTConfigWeekly(PatchTSTConfig):
    """
    주간(Weekly) 데이터 전용 프리셋 설정.

    권장 사항:
    - Patch Length: 8, 12, 16 등 데이터 특성 및 주기에 맞춰 조정.
    - Stride: Patch Length의 절반 권장.
    """
    # lookback: int = 54  # 예시: 약 1년
    # horizon: int = 27   # 예시: 약 6개월
    pass
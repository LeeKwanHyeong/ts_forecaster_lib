import math

import torch
import torch.nn as nn


def PositionalEncoding(q_len, d_model, normalize=True):
    """
    Sinusoidal(사인/코사인) 기반의 절대 위치 인코딩 행렬 생성.

    기능:
    - Transformer의 Attention 메커니즘이 순서 정보를 인식할 수 있도록 위치별 고유 패턴 주입.
    - 짝수 인덱스는 sin, 홀수 인덱스는 cos 함수를 사용하며 주파수를 달리함.
    - 정규화(Normalize) 옵션 활성화 시, 생성된 인코딩의 분포를 표준화 및 스케일링.

    반환:
        [q_len, d_model] 크기의 위치 인코딩 텐서.
    """
    pe = torch.zeros(q_len, d_model)
    positional = torch.arange(0, q_len).unsqueeze(1)

    # 주파수 계산 (Log Space에서 계산하여 수치 안정성 확보)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

    # 사인/코사인 패턴 적용
    pe[:, 0::2] = torch.sin(positional * div_term)
    pe[:, 1::2] = torch.cos(positional * div_term)

    # 데이터 스케일 맞춤을 위한 정규화 수행
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


SinCosPosEncoding = PositionalEncoding


def positional_encoding(pe, learn_pe, q_len, d_model):
    """
    위치 인코딩 초기화 및 nn.Parameter 변환 팩토리 함수.

    기능:
    - 설정 문자열(pe)에 따라 다양한 초기화 방식(SinCos, Random, Zero 등) 분기.
    - learn_pe 플래그를 통해 위치 인코딩의 학습 가능(Trainable) 여부 제어.
    - nn.Parameter로 래핑하여 모델의 파라미터로 등록 가능하도록 반환.

    Args:
        pe (str | None): 초기화 전략 ('sincos', 'zeros', 'normal', 'uniform', None)
        learn_pe (bool): Gradient 학습 여부 (requires_grad)
        q_len (int): 시퀀스 길이 (Patch 개수)
        d_model (int): 임베딩 차원
    """
    # Positional Encoding 초기화 전략 선택
    match pe:
        case None:
            # None인 경우: 무작위 초기화 후 학습 비활성화 (Base check 용도)
            W_pos = torch.empty((q_len, d_model))
            nn.init.uniform_(W_pos, -0.02, 0.02)
            learn_pe = False
        case 'zero':
            # zero: [q_len, 1] 크기의 브로드캐스팅용 텐서, 무작위 초기화 (-0.02 ~ 0.02)
            W_pos = torch.empty((q_len, 1))
            nn.init.uniform_(W_pos, -0.02, 0.02)
        case 'zeros':
            # zeros: [q_len, d_model] 전체 크기 텐서, 무작위 초기화 (-0.02 ~ 0.02)
            W_pos = torch.empty((q_len, d_model))
            nn.init.uniform_(W_pos, -0.02, 0.02)
        case 'normal' | 'gauss':
            # normal: 정규분포(Mean=0, Std=0.1) 초기화
            W_pos = torch.zeros((q_len, 1))
            torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
        case 'uniform':
            # uniform: 균등분포(0.0 ~ 0.1) 초기화
            W_pos = torch.zeros((q_len, 1))
            nn.init.uniform_(W_pos, a=0.0, b=0.1)
        case 'sincos':
            # sincos: 고정된 Sinusoidal 패턴 사용
            W_pos = PositionalEncoding(q_len, d_model, normalize=True)
        case _:
            raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal'"
                             f"'zeros', 'zero', 'uniform', 'sincos', None.)")

    # 학습 가능 여부를 설정하여 Parameter 반환
    return nn.Parameter(W_pos, requires_grad=learn_pe)
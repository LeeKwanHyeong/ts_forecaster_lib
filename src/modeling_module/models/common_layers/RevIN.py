from __future__ import annotations
import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN) 모듈.
    시계열 데이터의 분포 이동(Distribution Shift) 문제를 완화하기 위해 입력단을 정규화하고,
    출력단을 다시 역정규화하여 원래 스케일로 복원하는 역할 수행.

    특징:
    - use_std: 표준편차를 이용한 스케일링 적용 여부.
    - subtract_last: 평균 대신 마지막 시점 값을 빼는 방식 (비정상성 데이터의 추세 제거에 효과적).
    """

    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            affine: bool = True,
            subtract_last: bool = False,
            use_std: bool = True
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.use_std = use_std

        # 학습 가능한 Affine 파라미터(가중치/편향) 초기화
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        # 정규화 통계량 저장을 위한 버퍼 등록
        # persistent=False: state_dict에 저장되지 않음 (매 배치마다 갱신됨)
        self.register_buffer('last', torch.zeros(1, 1, num_features), persistent=False)
        self.register_buffer('mean', torch.zeros(1, 1, num_features), persistent=False)
        self.register_buffer('std', torch.ones(1, 1, num_features), persistent=False)

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        모드에 따라 정규화 또는 역정규화 수행.
        Args:
            x: 입력 텐서 [Batch, Length, Channel]
            mode: 'norm' | 'denorm'
        """
        if mode == 'norm':
            return self._norm(x)
        elif mode == 'denorm':
            return self._denorm(x)
        else:
            raise NotImplementedError(f"RevIN mode must be 'norm' or 'denorm', got {mode}")

    def _compute_stats(self, x: torch.Tensor):
        """
        입력 배치의 시간 축(Length) 기준 통계량 계산.
        x shape: [Batch, Length, Channel]
        """
        if self.subtract_last:
            # 마지막 시점의 값 추출 (Trend 제거용)
            self.last = x[:, -1:, :]
        else:
            # 시간 축(dim=1) 기준 평균 계산
            self.mean = x.mean(dim=1, keepdim=True).detach()

        # 시간 축(dim=1) 기준 분산 및 표준편차 계산
        var = x.var(dim=1, keepdim=True, unbiased=False).detach()
        self.std = torch.sqrt(var + self.eps)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        정규화(Normalization) 수행.
        Process: (Input - Center) / Scale * Affine_Weight + Affine_Bias
        """
        self._compute_stats(x)

        # Centering: 평균 또는 마지막 값 차감
        if self.subtract_last:
            x_n = x - self.last
        else:
            x_n = x - self.mean

        # Scaling: 표준편차로 나눔
        if self.use_std:
            x_n = x_n / (self.std + self.eps)

        # Affine Transformation: 학습 가능한 파라미터 적용
        if self.affine:
            x_n = x_n * self.affine_weight.view(1, 1, -1) + self.affine_bias.view(1, 1, -1)

        return x_n

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        """
        역정규화(Denormalization) 수행.
        모델의 출력을 원래 데이터 스케일로 복원.
        Process: (Input - Affine_Bias) / Affine_Weight * Scale + Center
        """
        y = x

        # Affine 역연산
        if self.affine:
            y = (y - self.affine_bias.view(1, 1, -1)) / (self.affine_weight.view(1, 1, -1) + self.eps)

        # Scaling 역연산
        if self.use_std:
            y = y * (self.std + self.eps)

        # Centering 역연산
        if self.subtract_last:
            y = y + self.last
        else:
            y = y + self.mean

        return y
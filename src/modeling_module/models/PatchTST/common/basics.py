# basics.py
__all__ = ['Transpose', 'LinBnDrop', 'SigmoidRange', 'sigmoid_range', 'get_activation_fn']

from typing import Optional, Union, Callable
import torch
from torch import nn


class Transpose(nn.Module):
    """
    텐서의 차원 순서를 변경하는 유틸리티 모듈.

    기능:
    - dims 인자 길이에 따라 `transpose`(2개) 또는 `permute`(3개 이상) 연산 자동 선택.
    - `contiguous=True` 설정 시 메모리 연속성이 보장된 텐서 반환.
    """

    def __init__(self, *dims: int, contiguous: bool = False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.dims) == 2:
            y = x.transpose(self.dims[0], self.dims[1])
        else:
            y = x.permute(*self.dims)
        return y.contiguous() if self.contiguous else y


class SigmoidRange(nn.Module):
    """
    Sigmoid 활성화 함수의 출력을 특정 범위 `[low, high]`로 스케일링하는 모듈.
    주로 회귀 문제에서 출력값의 범위를 제한할 때 사용.
    """

    def __init__(self, low: float, high: float):
        super().__init__()
        self.low, self.high = low, high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * (self.high - self.low) + self.low


class LinBnDrop(nn.Sequential):
    """
    Linear, BatchNorm1d, Dropout 레이어를 결합한 순차적 컨테이너(Sequential).

    기능:
    - `lin_first=True`: Linear(+Act) -> BatchNorm -> Dropout 순서 구성.
    - `lin_first=False`: BatchNorm -> Dropout -> Linear(+Act) 순서 구성 (ResNet 스타일).
    - BatchNorm 사용 시 Linear 레이어의 bias 자동 비활성화 처리.
    """

    def __init__(
            self,
            n_in: int,
            n_out: int,
            bn: bool = True,
            p: float = 0.0,
            act: Optional[Union[str, nn.Module, Callable[[], nn.Module], type]] = None,
            lin_first: bool = False,
    ):
        layers: list[nn.Module] = []

        # BN은 Linear 앞/뒤 배치에 따라 입력 채널 수 결정
        if bn:
            bn_features = n_out if lin_first else n_in
            layers.append(nn.BatchNorm1d(bn_features))

        if p and p > 0.0:
            layers.append(nn.Dropout(p))

        # Linear 및 활성화 함수 준비
        lin_and_act: list[nn.Module] = [nn.Linear(n_in, n_out, bias=not bn)]
        act_layer = get_activation_fn(act) if act is not None else None
        if act_layer is not None:
            lin_and_act.append(act_layer)

        # 설정된 순서에 따라 레이어 리스트 결합 및 초기화
        layers = (lin_and_act + layers) if lin_first else (layers + lin_and_act)
        super().__init__(*layers)


def sigmoid_range(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """
    SigmoidRange 클래스의 함수형 구현.
    입력 텐서 값을 `(low, high)` 범위로 변환.
    """
    return torch.sigmoid(x) * (high - low) + low


def get_activation_fn(
        activation: Optional[Union[str, nn.Module, Callable[[], nn.Module], type]]
) -> Optional[nn.Module]:
    """
    다양한 형태의 입력을 받아 적절한 `nn.Module` 활성화 함수 인스턴스 반환.

    지원 포맷:
    - nn.Module 인스턴스: 그대로 반환.
    - 문자열: 'relu', 'gelu' (대소문자 무관).
    - Callable/Class: 인스턴스화하여 반환.
    - None: None 반환.
    """
    if activation is None:
        return None

    if isinstance(activation, nn.Module):
        return activation

    if isinstance(activation, str):
        name = activation.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        # 필요시 확장: 'tanh', 'silu' 등
        raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or provide a callable/nn.Module')

    if callable(activation):
        # 클래스/팩토리 함수인 경우 인스턴스 생성 시도
        try:
            created = activation()
            if isinstance(created, nn.Module):
                return created
            raise TypeError("Callable did not return an nn.Module instance.")
        except TypeError as e:
            raise TypeError("Activation callable must be a zero-arg factory (e.g., nn.ReLU).") from e

    raise TypeError("activation must be a str, nn.Module, callable factory, or None.")
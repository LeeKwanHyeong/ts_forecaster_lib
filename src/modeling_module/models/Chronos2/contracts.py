"""시계열 입력과 예측 결과의 shape 및 결측 처리 계약."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .configs import Chronos2Config


@dataclass(frozen=True)
class Chronos2Forecast:
    """point는 (B,H,1)의 P50, quantiles는 (B,H,Q)의 원 단위 예측."""

    point: torch.Tensor
    quantiles: torch.Tensor
    quantile_levels: tuple[float, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Chronos2FutureFeatures:
    """출력 head 직전의 미래 패치 표현. 이번 구현에서는 추론·관찰 전용."""

    future_tokens: torch.Tensor
    location: torch.Tensor
    scale: torch.Tensor
    num_context_patches: int
    output_patch_size: int
    horizon: int


def _real_tensor(value: Any, name: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"{name}은 실수형 수치 배열이어야 합니다.") from exc
    if tensor.dtype == torch.bool or tensor.is_complex():
        raise TypeError(f"{name}은 bool이나 복소수가 아닌 수치 배열이어야 합니다.")
    # 공식 전처리의 입력은 CPU FP32로 통일하고 호출자의 입력은 수정하지 않음.
    tensor = tensor.detach().to(device="cpu", dtype=torch.float32).clone()
    if torch.isinf(tensor).any():
        raise ValueError(f"{name}에 Inf가 포함돼 있습니다.")
    return tensor


def prepare_inputs(
    x: Any,
    cfg: Chronos2Config,
    *,
    past_covariates: Mapping[str, Any] | None = None,
    future_covariates: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """(B,L,1)과 이름별 (B,T) 외생변수를 공식 입력 사전으로 변환."""
    target = _real_tensor(x, "x")
    if target.ndim != 3 or target.shape[0] < 1 or target.shape[1] < 1 or target.shape[2] != 1:
        raise ValueError("x shape은 단일 수요 채널의 (B,L,1)이어야 합니다.")
    batch, original_length, _ = target.shape
    target = target[:, -cfg.context_length:, 0]
    if (~torch.isnan(target)).sum(dim=1).eq(0).any():
        raise ValueError("잘라낸 context 안에 실제 관측값이 하나 이상 필요합니다.")
    prepared = []
    for name, values in (("past_covariates", past_covariates), ("future_covariates", future_covariates)):
        if values is not None and not isinstance(values, Mapping):
            raise TypeError(f"{name}은 이름과 수치 배열의 Mapping이어야 합니다.")
        encoded = {}
        for key, value in (values or {}).items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("외생변수 이름은 빈 문자열일 수 없습니다.")
            tensor = _real_tensor(value, f"{name}[{key}]")
            length = original_length if name == "past_covariates" else cfg.horizon
            if tensor.shape != (batch, length):
                raise ValueError(f"{name}[{key}] shape은 {(batch, length)}이어야 합니다.")
            if name == "past_covariates":
                tensor = tensor[:, -cfg.context_length:]
                if (~torch.isnan(tensor)).sum(dim=1).eq(0).any():
                    raise ValueError(f"과거 외생변수 {key}에 실제 관측값이 필요합니다.")
            elif torch.isnan(tensor).any():
                raise ValueError("미래 외생변수는 예측 시점에 알려진 유한한 값이어야 합니다.")
            encoded[key] = tensor
        prepared.append(encoded)
    past, future = prepared
    if cfg.batch_size < 1 + len(past):
        raise ValueError("batch_size는 수요 채널과 과거 외생변수 개수의 합 이상이어야 합니다.")
    if not set(future).issubset(past):
        raise ValueError("미래 외생변수는 같은 이름의 과거 외생변수도 필요합니다.")
    # NaN은 결측으로 전달하며 0 수요로 대체하지 않음.
    inputs = [{"target": target[index],
               "past_covariates": {key: value[index] for key, value in past.items()},
               "future_covariates": {key: value[index] for key, value in future.items()}}
              for index in range(batch)]
    return target, inputs

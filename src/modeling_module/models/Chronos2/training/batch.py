"""독립 단변량 수요의 관측 마스크와 native 학습 텐서 계약."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .configs import Chronos2TrainableConfig


def _observations(value: torch.Tensor, mask: torch.Tensor | None, name: str, device):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.ndim != 2:
        raise TypeError(f"{name}은 (N,T) 실수 Tensor여야 합니다.")
    tensor = value.detach().to(device=device, dtype=torch.float32).clone()
    if torch.isinf(tensor).any():
        raise ValueError(f"{name}에 Inf 또는 FP32 범위 초과 수량이 있습니다.")
    if mask is None:
        observed = ~torch.isnan(tensor)
    else:
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != value.shape:
            raise TypeError(f"{name}_mask는 입력과 같은 shape의 bool Tensor여야 합니다.")
        observed = mask.detach().to(device=device).clone()
    if (~torch.isfinite(tensor) & observed).any() or ((tensor < 0) & observed).any():
        raise ValueError(f"{name}의 관측 수량은 유한한 0 이상의 값이어야 합니다.")
    if (~observed.any(dim=1)).any():
        raise ValueError(f"{name}의 각 행에 실제 관측값이 하나 이상 필요합니다.")
    # native 정규화는 context_mask 적용보다 먼저 수행하므로 숨긴 값 자체도 NaN 처리.
    tensor = torch.where(observed, tensor, torch.nan)
    return tensor, observed


@dataclass(frozen=True)
class Chronos2TrainingBatch:
    """1·2단계는 독립 target 행만 지원. covariate 전용 행·공유 그룹은 후속 범위."""

    context: torch.Tensor
    future_target: torch.Tensor
    context_mask: torch.Tensor | None = None
    future_target_mask: torch.Tensor | None = None
    group_ids: torch.Tensor | None = None

    def native_inputs(self, cfg: Chronos2TrainableConfig, device) -> dict[str, torch.Tensor]:
        context, context_mask = _observations(self.context, self.context_mask, "context", device)
        target, target_mask = _observations(self.future_target, self.future_target_mask, "future_target", device)
        if context.shape[0] < 1 or context.shape[1] != cfg.context_length:
            raise ValueError("context는 계약의 (N,context_length)와 정확히 일치해야 합니다.")
        if target.shape != (context.shape[0], cfg.horizon):
            raise ValueError("future_target은 계약의 (N,horizon)과 정확히 일치해야 합니다.")
        groups = self.group_ids
        if groups is None:
            groups = torch.arange(len(context), dtype=torch.long, device=device)
        else:
            if not isinstance(groups, torch.Tensor) or groups.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise TypeError("group_ids는 정수 Tensor여야 합니다.")
            if groups.shape != (len(context),):
                raise ValueError("group_ids shape은 (N,)이어야 합니다.")
            groups = groups.detach().to(device=device, dtype=torch.long).clone()
            if groups.unique().numel() != len(context):
                raise ValueError("현재 단계는 독립 단변량만 지원하므로 group ID는 행마다 달라야 합니다.")
        return {"context": context, "context_mask": context_mask, "future_target": target,
                "future_target_mask": target_mask, "group_ids": groups}


@dataclass(frozen=True)
class Chronos2TrainingOutput:
    """loss는 공식 patch 평균 손실, 예측은 원 단위이며 graph를 유지하는 반환값."""

    loss: torch.Tensor
    point: torch.Tensor
    quantiles: torch.Tensor
    quantile_levels: tuple[float, ...]
    observed_target_points: int
    raw_prediction_length: int
    metadata: dict[str, Any]

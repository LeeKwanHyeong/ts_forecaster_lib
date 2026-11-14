"""학습 실패나 빈 평가를 정상적인 0 loss로 보고하지 않도록 검증."""

from __future__ import annotations

import math

import torch


def require_nonempty_loader(loader, *, context: str) -> None:
    if loader is None:
        raise ValueError(f"{context}: a loader is required.")
    try:
        count = len(loader)
    except TypeError:
        # 길이가 없는 iterable은 실제 처리 배치 수로 확인.
        return
    if count <= 0:
        raise ValueError(f"{context}: loader contains no batches (check drop_last).")


def require_finite_loss(loss, *, context: str) -> None:
    if torch.is_tensor(loss):
        if loss.numel() != 1:
            raise ValueError(f"{context}: loss must be scalar, got {tuple(loss.shape)}.")
        finite = bool(torch.isfinite(loss.detach()).all())
    else:
        finite = math.isfinite(float(loss))
    if not finite:
        raise FloatingPointError(f"{context}: non-finite loss (NaN/Inf); aborting instead of reporting zero.")


def checked_loss_mean(total_scalar, total_tensor, count: int, *, context: str) -> float:
    if count <= 0:
        raise ValueError(f"{context}: no valid batches were processed.")
    total = float(total_scalar)
    if total_tensor is not None:
        total += float(total_tensor.item())
    mean = total / count
    require_finite_loss(mean, context=context)
    return mean

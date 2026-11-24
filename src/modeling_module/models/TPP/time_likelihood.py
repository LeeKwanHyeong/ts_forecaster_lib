"""Differentiable log mass for rounded, positive log-normal durations."""

from __future__ import annotations

import math

import torch


def _log_probability_difference(
    log_larger: torch.Tensor, log_smaller: torch.Tensor
) -> torch.Tensor:
    """Compute a positive probability difference without inactive NaN gradients."""
    delta = log_smaller - log_larger
    if not bool(torch.isfinite(delta).all()) or not bool((delta < 0).all()):
        raise ValueError(
            "Integer observation has unresolved probability at float64 resolution"
        )
    result = torch.empty_like(delta)
    far = delta < -math.log(2.0)
    for mask, use_log1p in ((far, True), (~far, False)):
        indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            continue
        selected = delta[indices]
        # torch.where would evaluate both formulas; the unused log1p branch
        # can round exp(delta) to one and contaminate backward with NaN.
        complement = (
            torch.log1p(-torch.exp(selected))
            if use_log1p
            else torch.log(-torch.expm1(selected))
        )
        result = result.index_copy(
            0, indices, log_larger[indices] + complement
        )
    return result


def positive_integer_log_mass(
    target_dt: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
    *,
    time_scale: float,
    top_code: int | None = None,
) -> torch.Tensor:
    """Return log P(D=target_dt) for T=time_scale*exp(N(location, scale)).

    The observation rule is ``D=max(1, round(T))``, or ``min(top_code, D)``
    when top-coded. Thus code one receives P(T<=1.5), regular codes receive
    P(d-0.5<T<=d+0.5), and the top code receives P(T>top_code-0.5).
    The first bin starts at zero, not 0.5, so the observed law is normalized.

    Inputs are equally shaped, rank-one tensors on one device. Targets must
    be exactly integer-valued and positive. Calculations and results use
    float64 while casts retain gradients to the original parameter dtype.
    Unrepresentable probabilities raise ValueError rather than being clipped.
    """
    tensors = (target_dt, location, scale)
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise ValueError("Targets, location and scale must be tensors")
    if any(value.ndim != 1 for value in tensors):
        raise ValueError("Targets, location and scale must be rank one")
    if target_dt.shape != location.shape or target_dt.shape != scale.shape:
        raise ValueError("Target, location and scale shapes must match")
    if any(value.device != location.device for value in tensors):
        raise ValueError("Targets, location and scale must share a device")
    if target_dt.is_complex() or target_dt.dtype == torch.bool:
        raise ValueError("Duration targets must be real numeric values")
    if not location.is_floating_point() or not scale.is_floating_point():
        raise ValueError("Location and scale must be floating-point tensors")
    if (
        isinstance(time_scale, bool)
        or not isinstance(time_scale, (int, float))
        or not math.isfinite(time_scale)
        or time_scale <= 0
    ):
        raise ValueError("time_scale must be finite and strictly positive")
    if top_code is not None and (
        isinstance(top_code, bool)
        or not isinstance(top_code, int)
        or top_code < 2
    ):
        raise ValueError("top_code must be an integer of at least two")

    target = target_dt.to(dtype=torch.float64)
    mu = location.to(dtype=torch.float64)
    sigma = scale.to(dtype=torch.float64)
    if not bool(torch.isfinite(target).all()) or not bool((target >= 1).all()):
        raise ValueError("Duration targets must be finite and at least one")
    if not bool((target == torch.round(target)).all()):
        raise ValueError("Duration targets must lie exactly on the integer grid")
    if not bool(torch.isfinite(mu).all()) or not bool(torch.isfinite(sigma).all()):
        raise ValueError("Location and scale must be finite")
    if not bool((sigma > 0).all()):
        raise ValueError("Scale must be strictly positive")
    if top_code is not None and not bool((target <= top_code).all()):
        raise ValueError("Duration target exceeds top_code")

    # This also supplies a differentiable empty result for an empty batch.
    result = mu[:0] + sigma[:0] if target.numel() == 0 else torch.empty_like(mu)
    log_time_scale = math.log(time_scale)
    first = target == 1
    top = (
        torch.zeros_like(first)
        if top_code is None
        else target == top_code
    )

    first_indices = torch.nonzero(first, as_tuple=False).squeeze(-1)
    if first_indices.numel():
        z = (
            math.log(1.5) - log_time_scale - mu[first_indices]
        ) / sigma[first_indices]
        if not bool(torch.isfinite(z).all()):
            raise ValueError("Non-finite standardized duration boundary")
        result = result.index_copy(0, first_indices, torch.special.log_ndtr(z))

    top_indices = torch.nonzero(top, as_tuple=False).squeeze(-1)
    if top_indices.numel():
        z = (
            torch.log(target[top_indices] - 0.5)
            - log_time_scale
            - mu[top_indices]
        ) / sigma[top_indices]
        if not bool(torch.isfinite(z).all()):
            raise ValueError("Non-finite standardized duration boundary")
        result = result.index_copy(0, top_indices, torch.special.log_ndtr(-z))

    regular_indices = torch.nonzero(~(first | top), as_tuple=False).squeeze(-1)
    if regular_indices.numel():
        observed = target[regular_indices]
        lower_z = (
            torch.log(observed - 0.5) - log_time_scale - mu[regular_indices]
        ) / sigma[regular_indices]
        upper_z = (
            torch.log(observed + 0.5) - log_time_scale - mu[regular_indices]
        ) / sigma[regular_indices]
        if not bool(torch.isfinite(lower_z).all()) or not bool(
            torch.isfinite(upper_z).all()
        ):
            raise ValueError("Non-finite standardized duration boundary")
        survival = lower_z > 0
        for mask, use_survival in ((survival, True), (~survival, False)):
            indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if indices.numel() == 0:
                continue
            if use_survival:
                larger = torch.special.log_ndtr(-lower_z[indices])
                smaller = torch.special.log_ndtr(-upper_z[indices])
            else:
                larger = torch.special.log_ndtr(upper_z[indices])
                smaller = torch.special.log_ndtr(lower_z[indices])
            result = result.index_copy(
                0,
                regular_indices[indices],
                _log_probability_difference(larger, smaller),
            )
    if not bool(torch.isfinite(result).all()) or not bool((result <= 0).all()):
        raise ValueError("Non-finite or invalid positive-integer log probability")
    return result

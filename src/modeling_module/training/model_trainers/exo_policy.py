"""exo_policy.py

Future-exogenous resolution policy.

This module centralizes how we decide the source of *future* exogenous variables:
- Loader-provided `fe_cont`
- Calendar callback fallback (`compose_exo_calendar_cb`)
- Or none (when `use_exogenous_mode=False`)

Why separate?
- The decision logic is business/pipeline policy, not model logic.
- Keeping it isolated makes total_train/orchestration readable and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable

import torch

from modeling_module.utils.exogenous_utils import compose_exo_calendar_cb

try:
    from .freq_policy import FreqSpec
except Exception:  # pragma: no cover
    from freq_policy import FreqSpec  # type: ignore


@dataclass(frozen=True)
class ExoSpec:
    """Resolved future-exogenous configuration."""
    use_exogenous_mode: bool
    has_loader_future_exo: bool
    loader_exo_dim: int
    exo_dim: int
    future_exo_cb: Optional[Callable]
    source: str  # "none" | "loader" | "callback"


def infer_future_exo_spec_from_loader(loader) -> tuple[bool, int]:
    """Infer whether loader provides `fe_cont` (future exogenous) and its dimension.

    Returns:
        (has_fe, fe_dim)
    """
    try:
        b = next(iter(loader))
        if not isinstance(b, (list, tuple)) or len(b) < 4:
            return (False, 0)

        fe = b[3]
        if fe is None:
            return (False, 0)

        if hasattr(fe, "ndim") and fe.ndim == 3:
            return (True, int(fe.shape[-1]))
        if hasattr(fe, "ndim") and fe.ndim == 2:
            # Defensive: some datasets may provide (H, E)
            return (True, int(fe.shape[-1]))
        return (True, 0)
    except Exception:
        return (False, 0)


def wrap_future_exo_cb(future_exo_cb):
    """Wrap callback to absorb `device=` kwarg and move torch.Tensor output."""
    if future_exo_cb is None:
        return None

    def _wrapped(t0, H, *args, **kwargs):
        device = kwargs.pop("device", None)
        out = future_exo_cb(t0, H)
        if device is not None and isinstance(out, torch.Tensor):
            out = out.to(device)
        return out

    return _wrapped


def resolve_future_exogenous(
    train_loader,
    *,
    freq_spec: FreqSpec,
    use_exogenous_mode: bool,
) -> ExoSpec:
    """Resolve future-exogenous policy in one place.

    Priority (when use_exogenous_mode=True):
      1) Loader-provided future exo (fe_cont)
      2) Calendar callback (compose_exo_calendar_cb)

    When use_exogenous_mode=False:
      - Always ignore future exo, but emit a warning if loader provides it.
    """
    has_fe, fe_dim = infer_future_exo_spec_from_loader(train_loader)
    print(f"[total_train] use_exogenous_mode={use_exogenous_mode} | has_fe={has_fe} fe_dim={fe_dim}")

    if not use_exogenous_mode:
        if has_fe and fe_dim > 0:
            print(f"[total_train][WARN] use_exogenous_mode=False but loader provides fe_cont dim={fe_dim}. Ignoring.")
        return ExoSpec(
            use_exogenous_mode=False,
            has_loader_future_exo=bool(has_fe),
            loader_exo_dim=int(fe_dim),
            exo_dim=0,
            future_exo_cb=None,
            source="none",
        )

    # use_exogenous_mode == True
    if has_fe:
        if fe_dim <= 0:
            raise RuntimeError(
                "[total_train] use_exogenous_mode=True but loader future-exo dim is invalid. "
                f"fe_dim={fe_dim}. Check datamodule wiring / feature selection."
            )
        return ExoSpec(
            use_exogenous_mode=True,
            has_loader_future_exo=True,
            loader_exo_dim=int(fe_dim),
            exo_dim=int(fe_dim),
            future_exo_cb=None,  # loader provides it
            source="loader",
        )

    # fallback: calendar callback
    cb = compose_exo_calendar_cb(date_type=freq_spec.dt_char)
    cb = wrap_future_exo_cb(cb)

    # historical convention in your codebase:
    exo_dim = 4 if freq_spec.freq in ("daily", "hourly") else 2

    return ExoSpec(
        use_exogenous_mode=True,
        has_loader_future_exo=False,
        loader_exo_dim=0,
        exo_dim=int(exo_dim),
        future_exo_cb=cb,
        source="callback",
    )


# Backward-compatible aliases (older codepaths)
_infer_future_exo_spec_from_loader = infer_future_exo_spec_from_loader
_wrap_future_exo_cb = wrap_future_exo_cb
_resolve_future_exogenous = resolve_future_exogenous

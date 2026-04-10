from __future__ import annotations

from typing import Optional

import torch


def _normalize_device(device: str | torch.device) -> str:
    return str(device)


def _device_type(device: str | torch.device) -> str:
    return torch.device(_normalize_device(device)).type


def probe_device(device: str | torch.device) -> tuple[bool, Optional[str]]:
    resolved = _normalize_device(device)
    kind = _device_type(resolved)

    if kind == "cpu":
        return True, None

    if kind == "cuda":
        if not torch.cuda.is_available():
            return False, "CUDA is not available in this PyTorch environment."
    elif kind == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            return False, "MPS is not available in this PyTorch environment."
    else:
        return False, f"Unsupported device type: {kind}"

    try:
        # Run a tiny kernel so we catch runtime incompatibilities such as
        # unsupported GPU architectures, not just availability flags.
        x = torch.tensor([0.0, 1.0], device=resolved)
        _ = torch.isnan(x).any().item()
        _ = (x + 1).sum().item()
        if kind == "cuda":
            torch.cuda.synchronize(torch.device(resolved))
        return True, None
    except Exception as exc:  # pragma: no cover - depends on local runtime
        return False, f"{type(exc).__name__}: {exc}"


def select_default_device() -> tuple[str, Optional[str]]:
    diagnostics: list[str] = []
    for candidate in ("cuda", "mps"):
        ok, reason = probe_device(candidate)
        if ok:
            return candidate, None
        if reason:
            diagnostics.append(f"{candidate}: {reason}")
    return "cpu", "; ".join(diagnostics) if diagnostics else None


def default_device() -> str:
    device, _ = select_default_device()
    return device


def resolve_device(device: str | torch.device | None) -> str:
    if device is None:
        return default_device()

    resolved = _normalize_device(device)
    ok, reason = probe_device(resolved)
    if ok:
        return resolved

    raise RuntimeError(
        f"Requested device `{resolved}` is not usable in this environment. "
        f"{reason or 'Device probe failed.'} "
        "Try `device='cpu'` or install a PyTorch build compatible with this accelerator."
    )

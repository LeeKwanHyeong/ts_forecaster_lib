"""Runtime feature availability derived from the installed package payload."""

from __future__ import annotations

from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SELLM_AVAILABLE = (_PACKAGE_ROOT / "models" / "SELLM").is_dir()


__all__ = ["SELLM_AVAILABLE"]

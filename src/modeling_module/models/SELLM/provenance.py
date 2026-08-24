"""Pinned references used to review the paper-based SELLM implementation."""

from __future__ import annotations

from typing import Final


SELLM_PAPER_TITLE: Final = (
    "Semantic-Enhanced Time-Series Forecasting Via Large Language Models"
)
SELLM_PAPER_URL: Final = "https://arxiv.org/abs/2508.07697"
SELLM_UPSTREAM_REPOSITORY: Final = "https://github.com/LH325/SE-LLM"
SELLM_UPSTREAM_COMMIT: Final = "9fab871b9c4774cd4b58d025de992d55a24c18e7"
SELLM_UPSTREAM_LICENSE: Final[str | None] = None

# These hashes identify review inputs only. No upstream source file is vendored.
SELLM_UPSTREAM_REVIEW_FILES: Final = (
    (
        "models/SELLM.py",
        "e903098beeb7377f56c440f54b34ab81eb7be55483457cc9972ac3d45a2b356f",
    ),
    (
        "models/TimeAdapter.py",
        "098cc86596467801b441a3308736e0cf9fb568f3bdb75a5ba980ba5930953ab9",
    ),
    (
        "models/TSCC.py",
        "01a75eb5c01fb25259dbeec028191728ad35ff4870824eb61925cde0d29a8e52",
    ),
)

__all__ = [
    "SELLM_PAPER_TITLE",
    "SELLM_PAPER_URL",
    "SELLM_UPSTREAM_COMMIT",
    "SELLM_UPSTREAM_LICENSE",
    "SELLM_UPSTREAM_REPOSITORY",
    "SELLM_UPSTREAM_REVIEW_FILES",
]

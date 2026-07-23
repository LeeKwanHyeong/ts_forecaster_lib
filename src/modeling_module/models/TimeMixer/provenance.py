"""Pinned source identities for the planned TimeMixer model family."""

from __future__ import annotations

from typing import Final


TIMEMIXER_PAPER_TITLE: Final = (
    "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting"
)
TIMEMIXER_PAPER_URL: Final = "https://arxiv.org/abs/2405.14616"
TIMEMIXER_UPSTREAM_REPOSITORY: Final = "https://github.com/kwuking/TimeMixer"
TIMEMIXER_UPSTREAM_COMMIT: Final = (
    "e24610583b36fdd8c76cc17a8df4e65759a5f460"
)
TIMEMIXER_UPSTREAM_TREE: Final = "f4e2c914a34e966684d034e3732ef869c18cbc50"
TIMEMIXER_UPSTREAM_MODEL_COMMIT: Final = (
    "38a3507595048d998d12f00d37b66987d03295fc"
)
TIMEMIXER_UPSTREAM_LICENSE: Final = "Apache-2.0"
TIMEMIXER_UPSTREAM_LICENSE_FILE: Final = "LICENSE.upstream"
TIMEMIXER_UPSTREAM_MANIFEST_FILE: Final = "upstream_manifest.json"
TIMEMIXER_UPSTREAM_NOTICE_PRESENT: Final = False

# (upstream path, SHA-256, Git blob, byte size, newline count)
TIMEMIXER_UPSTREAM_FILES: Final = (
    (
        "models/TimeMixer.py",
        "817d62f4aac54c8566e560f6d3785856e31c8ee51460279ee3d0a4823f11d4be",
        "ad1847df86646848cdfecab568eb49f7099ef1ce",
        20_663,
        527,
    ),
    (
        "layers/Embed.py",
        "ab492ea2f68459bbcf3cbffdd1beb75b24d0d70248d017a313a3b470316aaa2b",
        "f558c590b26c7df31942ac12adeae63211df46ca",
        9_579,
        250,
    ),
    (
        "layers/Autoformer_EncDec.py",
        "48745b4bb647355e9845792a855df9c59fd7df7fcc664c765351fec390c4073e",
        "6fce4bcd6b3d3eb00e9bcf5931ed2ee301554f4a",
        6_831,
        203,
    ),
    (
        "layers/StandardNorm.py",
        "cc1c0bc65b7b094bbe83f988fb05b86272a59638c030c3781aa52ce8880379df",
        "c1c9269c0139eaa6cd08de4f55b47c82f64a6f9f",
        2_183,
        67,
    ),
    (
        "LICENSE",
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
        "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64",
        11_357,
        201,
    ),
)

__all__ = [
    "TIMEMIXER_PAPER_TITLE",
    "TIMEMIXER_PAPER_URL",
    "TIMEMIXER_UPSTREAM_COMMIT",
    "TIMEMIXER_UPSTREAM_FILES",
    "TIMEMIXER_UPSTREAM_LICENSE",
    "TIMEMIXER_UPSTREAM_LICENSE_FILE",
    "TIMEMIXER_UPSTREAM_MANIFEST_FILE",
    "TIMEMIXER_UPSTREAM_MODEL_COMMIT",
    "TIMEMIXER_UPSTREAM_NOTICE_PRESENT",
    "TIMEMIXER_UPSTREAM_REPOSITORY",
    "TIMEMIXER_UPSTREAM_TREE",
]

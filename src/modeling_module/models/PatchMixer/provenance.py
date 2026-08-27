"""Pinned source snapshots used for PatchMixer lineage and parity checks."""

from __future__ import annotations

from typing import Final


PATCHMIXER_UPSTREAM_REPOSITORY: Final = "https://github.com/Zeying-Gong/PatchMixer"
PATCHMIXER_UPSTREAM_COMMIT: Final = "cfc6c1386e7fe1633f92ef4b258ff1a4649008b4"
PATCHMIXER_UPSTREAM_MODEL_BLOB: Final = "bf3867109192da6cd8816f4aec8ab0bf16ec80af"
PATCHMIXER_UPSTREAM_LICENSE: Final = "MIT"

PATCHMIXER_ENHANCED_BASELINE_COMMIT: Final = "e53269e8e038a2664a43020587f79303aa2b4ff8"
PATCHMIXER_ENHANCED_SOURCE_BLOBS: Final = (
    (
        "src/modeling_module/models/PatchMixer/PatchMixer.py",
        "97846c17f5101e97308761c9b44e8df03928b374",
    ),
    (
        "src/modeling_module/models/PatchMixer/backbone.py",
        "f225ad28dbadfe5fbc2e18917b58b31b63fe5bc4",
    ),
    (
        "src/modeling_module/models/PatchMixer/common/configs.py",
        "5004e814bb1fc0a751073c4e5e31502cfaaed68f",
    ),
)

# The fixed no-exogenous configuration used for structural cost comparisons.
PATCHMIXER_REFERENCE_CONFIG: Final = (
    ("lookback", 54),
    ("horizon", 27),
    ("enc_in", 1),
    ("patch_len", 12),
    ("stride", 8),
    ("mixer_kernel_size", 5),
    ("d_model", 128),
    ("e_layers", 6),
    ("dropout", 0.1),
    ("head_dropout", 0.02),
    ("f_out", 256),
    ("head_hidden", 256),
)

PATCHMIXER_REFERENCE_PARAMETER_COUNTS: Final = (
    ("original", 76_564),
    ("enhanced", 7_077_643),
)

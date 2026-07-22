"""Pinned source identities for the current PatchTST baseline."""

PATCHTST_PAPER_URL = "https://arxiv.org/abs/2211.14730"
PATCHTST_UPSTREAM_REPOSITORY = "https://github.com/yuqinie98/PatchTST"

# This commit predates the documentation and public-export cleanup. The model
# blobs below are the behavioral baseline and are intentionally unchanged by it.
PATCHTST_BASELINE_COMMIT = "43f5ec8c9cbc89eaed2a28d7fb011d86b5303428"
PATCHTST_BASELINE_BLOBS = {
    "supervised/PatchTST.py": "8fd033e32d2247f6af02442de5c1c4e68deefb8b",
    "supervised/backbone.py": "7104d734acd0f28d26cbbb09a9f129d908b51e44",
    "common/backbone_base.py": "5bb7fd4a42ecb707075cab5301e32e9a90f17a0a",
    "common/configs.py": "90c471a3760867377aa1fe1a4536f708310c8536",
    "supervised/variants.py": "6a580289c172d89957d93eae7371dcbbff869acc",
}

__all__ = [
    "PATCHTST_PAPER_URL",
    "PATCHTST_UPSTREAM_REPOSITORY",
    "PATCHTST_BASELINE_COMMIT",
    "PATCHTST_BASELINE_BLOBS",
]

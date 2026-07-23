from __future__ import annotations

import hashlib
import json
from pathlib import Path

from modeling_module.models.TimeMixer.provenance import (
    TIMEMIXER_PAPER_TITLE,
    TIMEMIXER_PAPER_URL,
    TIMEMIXER_UPSTREAM_COMMIT,
    TIMEMIXER_UPSTREAM_FILES,
    TIMEMIXER_UPSTREAM_LICENSE,
    TIMEMIXER_UPSTREAM_LICENSE_FILE,
    TIMEMIXER_UPSTREAM_MANIFEST_FILE,
    TIMEMIXER_UPSTREAM_MODEL_COMMIT,
    TIMEMIXER_UPSTREAM_NOTICE_PRESENT,
    TIMEMIXER_UPSTREAM_REPOSITORY,
    TIMEMIXER_UPSTREAM_TREE,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "src/modeling_module/models/TimeMixer"
EXPECTED_UPSTREAM = {
    "repository": "https://github.com/kwuking/TimeMixer",
    "commit": "e24610583b36fdd8c76cc17a8df4e65759a5f460",
    "tree": "f4e2c914a34e966684d034e3732ef869c18cbc50",
    "model_last_commit": "38a3507595048d998d12f00d37b66987d03295fc",
}
EXPECTED_FILES = {
    "models/TimeMixer.py": {
        "sha256": "817d62f4aac54c8566e560f6d3785856e31c8ee51460279ee3d0a4823f11d4be",
        "git_blob": "ad1847df86646848cdfecab568eb49f7099ef1ce",
        "bytes": 20_663,
        "lines": 527,
    },
    "layers/Embed.py": {
        "sha256": "ab492ea2f68459bbcf3cbffdd1beb75b24d0d70248d017a313a3b470316aaa2b",
        "git_blob": "f558c590b26c7df31942ac12adeae63211df46ca",
        "bytes": 9_579,
        "lines": 250,
    },
    "layers/Autoformer_EncDec.py": {
        "sha256": "48745b4bb647355e9845792a855df9c59fd7df7fcc664c765351fec390c4073e",
        "git_blob": "6fce4bcd6b3d3eb00e9bcf5931ed2ee301554f4a",
        "bytes": 6_831,
        "lines": 203,
    },
    "layers/StandardNorm.py": {
        "sha256": "cc1c0bc65b7b094bbe83f988fb05b86272a59638c030c3781aa52ce8880379df",
        "git_blob": "c1c9269c0139eaa6cd08de4f55b47c82f64a6f9f",
        "bytes": 2_183,
        "lines": 67,
    },
    "LICENSE": {
        "sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
        "git_blob": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64",
        "bytes": 11_357,
        "lines": 201,
    },
}


def _load_manifest() -> dict:
    return json.loads(
        (PACKAGE_DIR / TIMEMIXER_UPSTREAM_MANIFEST_FILE).read_text(
            encoding="utf-8"
        )
    )


def _git_blob_hash(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def test_timemixer_manifest_matches_frozen_provenance() -> None:
    manifest = _load_manifest()

    assert manifest["schema_version"] == 1
    assert manifest["paper"] == {
        "title": TIMEMIXER_PAPER_TITLE,
        "url": TIMEMIXER_PAPER_URL,
        "arxiv_version": "2405.14616v1",
    }
    assert TIMEMIXER_UPSTREAM_REPOSITORY == EXPECTED_UPSTREAM["repository"]
    assert TIMEMIXER_UPSTREAM_COMMIT == EXPECTED_UPSTREAM["commit"]
    assert TIMEMIXER_UPSTREAM_TREE == EXPECTED_UPSTREAM["tree"]
    assert TIMEMIXER_UPSTREAM_MODEL_COMMIT == EXPECTED_UPSTREAM["model_last_commit"]
    assert manifest["upstream"] == EXPECTED_UPSTREAM
    assert manifest["license"] == {
        "spdx": TIMEMIXER_UPSTREAM_LICENSE,
        "upstream_path": "LICENSE",
        "vendored_path": TIMEMIXER_UPSTREAM_LICENSE_FILE,
        "notice_present": TIMEMIXER_UPSTREAM_NOTICE_PRESENT,
    }

    expected_files = {
        path: {
            "sha256": sha256,
            "git_blob": git_blob,
            "bytes": byte_size,
            "lines": line_count,
        }
        for path, sha256, git_blob, byte_size, line_count in TIMEMIXER_UPSTREAM_FILES
    }
    assert expected_files == EXPECTED_FILES
    assert manifest["files"] == EXPECTED_FILES


def test_timemixer_vendored_license_is_byte_identical_to_pinned_source() -> None:
    manifest = _load_manifest()
    content = (PACKAGE_DIR / TIMEMIXER_UPSTREAM_LICENSE_FILE).read_bytes()
    expected = manifest["files"][manifest["license"]["upstream_path"]]

    assert len(content) == expected["bytes"]
    assert content.count(b"\n") == expected["lines"]
    assert hashlib.sha256(content).hexdigest() == expected["sha256"]
    assert _git_blob_hash(content) == expected["git_blob"]


def test_timemixer_fixture_set_covers_only_reviewed_upstream_files() -> None:
    assert [record[0] for record in TIMEMIXER_UPSTREAM_FILES] == [
        "models/TimeMixer.py",
        "layers/Embed.py",
        "layers/Autoformer_EncDec.py",
        "layers/StandardNorm.py",
        "LICENSE",
    ]

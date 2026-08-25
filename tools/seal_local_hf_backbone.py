#!/usr/bin/env python3
"""Create a deterministic provenance manifest for a local Hugging Face backbone."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.qualify_icl_backbones_5090 import write_backbone_manifest  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--license", dest="license_id", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    model_path = args.model_path.expanduser().resolve()
    manifest = write_backbone_manifest(
        model_path,
        model_id=str(args.model_id),
        revision=str(args.revision),
        license_id=str(args.license_id),
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "manifest": str(model_path / "backbone-manifest.json"),
                "manifest_sha256": manifest["manifest_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

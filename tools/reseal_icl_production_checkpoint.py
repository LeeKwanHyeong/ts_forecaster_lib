#!/usr/bin/env python3
"""Correct ICL production metadata without changing learned parameters."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


CONTRACT = "modeling_module.icl_production_metadata_correction.v1"


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and raw values deterministically."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state_dict entry {name!r} is not a tensor.")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_canonical_json(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def synchronize_production_config(
    checkpoint: Mapping[str, Any],
    *,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
) -> dict[str, Any]:
    """Return a checkpoint copy whose restore config matches sealed refit metadata."""

    payload = dict(checkpoint)
    meta = payload.get("meta")
    if not isinstance(meta, Mapping):
        raise ValueError("Checkpoint meta is missing.")
    required = {
        "model_key": "autotimes_base",
        "training_mode": "production_refit",
        "validation_enabled": False,
        "state_selection": "final_epoch",
    }
    for key, expected in required.items():
        if meta.get(key) != expected:
            raise ValueError(
                f"Checkpoint meta {key!r} must be {expected!r}, got {meta.get(key)!r}."
            )
    random_seed = meta.get("random_seed")
    epochs = meta.get("epochs")
    if not isinstance(random_seed, int):
        raise ValueError("Checkpoint meta random_seed must be an integer.")
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("Checkpoint meta epochs must be a positive integer.")

    config = payload.get("config") or payload.get("cfg_state")
    if not isinstance(config, Mapping):
        raise ValueError("Checkpoint restore config is missing.")
    corrected_config = dict(config)
    corrected_config.update(
        {
            "epochs": epochs,
            "lr": float(learning_rate),
            "weight_decay": float(weight_decay),
            "max_grad_norm": float(max_grad_norm),
            "training_mode": "production_refit",
            "random_seed": random_seed,
        }
    )
    payload["config"] = corrected_config
    payload["cfg_state"] = dict(corrected_config)
    return payload


def reseal_checkpoint(
    source: Path,
    destination: Path,
    *,
    source_receipt: Path,
    receipt_path: Path,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
) -> dict[str, Any]:
    """Write a corrected checkpoint and a sealed correction receipt."""

    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    source_receipt = source_receipt.expanduser().resolve()
    receipt_path = receipt_path.expanduser().resolve()
    if source == destination:
        raise ValueError("Source checkpoint must be preserved at a different path.")
    receipt = json.loads(source_receipt.read_text(encoding="utf-8"))
    expected_source_sha = receipt.get("checkpoint", {}).get("sha256")
    actual_source_sha = _file_sha256(source)
    if expected_source_sha != actual_source_sha:
        raise ValueError("Source checkpoint SHA256 does not match its production receipt.")

    original = torch.load(source, map_location="cpu", weights_only=False)
    original_state_hash = state_dict_sha256(original["state_dict"])
    corrected = synchronize_production_config(
        original,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
    )
    corrected_meta = dict(corrected["meta"])
    corrected_meta["metadata_correction"] = {
        "contract": CONTRACT,
        "corrected_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint_sha256": actual_source_sha,
        "updated_config_fields": [
            "epochs",
            "lr",
            "max_grad_norm",
            "random_seed",
            "training_mode",
            "weight_decay",
        ],
    }
    corrected["meta"] = corrected_meta

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(corrected, destination)
    restored = torch.load(destination, map_location="cpu", weights_only=False)
    restored_state_hash = state_dict_sha256(restored["state_dict"])
    if restored_state_hash != original_state_hash:
        raise RuntimeError("State dict changed while correcting checkpoint metadata.")

    result = {
        "contract": CONTRACT,
        "status": "PASS",
        "source_checkpoint": {
            "path": str(source),
            "sha256": actual_source_sha,
        },
        "corrected_checkpoint": {
            "path": str(destination),
            "sha256": _file_sha256(destination),
            "state_dict_sha256": restored_state_hash,
        },
        "state_dict_unchanged": True,
        "corrected_config": {
            key: restored["config"][key]
            for key in (
                "epochs",
                "lr",
                "weight_decay",
                "max_grad_norm",
                "training_mode",
                "random_seed",
            )
        },
    }
    result["receipt_sha256"] = _payload_sha256(result)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--destination-checkpoint", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = reseal_checkpoint(
        args.source_checkpoint,
        args.destination_checkpoint,
        source_receipt=args.source_receipt,
        receipt_path=args.receipt,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()

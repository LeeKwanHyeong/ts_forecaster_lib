#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from modeling_module import load_predictor


def _config_value(config: Any, name: str, default: int) -> int:
    if isinstance(config, Mapping):
        return int(config.get(name, default))
    return int(getattr(config, name, default))


def _output_summary(output: Any) -> dict[str, Any]:
    if torch.is_tensor(output):
        tensors = {"output": output}
    elif isinstance(output, Mapping):
        tensors = {key: value for key, value in output.items() if torch.is_tensor(value)}
    else:
        raise TypeError(f"Unsupported model output type: {type(output).__name__}")
    return {
        key: {
            "shape": list(value.shape),
            "finite": bool(torch.isfinite(value).all().item()),
        }
        for key, value in tensors.items()
    }


def validate_checkpoint(path: Path, *, device: str) -> dict[str, Any]:
    predictor = load_predictor(str(path), device=device, strict=True)
    lookback = _config_value(predictor.config, "lookback", 104)
    channels = _config_value(predictor.config, "enc_in", 1)
    x = torch.randn(2, lookback, channels, device=device)
    with torch.no_grad():
        output = predictor.model(x)
    return {
        "path": str(path),
        "model_key": predictor.model_key,
        "model_class": type(predictor.model).__name__,
        "state_dict_entries": len(predictor.model.state_dict()),
        "output": _output_summary(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict-load PatchMixer checkpoints and run a finite forward pass."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    results = [validate_checkpoint(path, device=args.device) for path in args.paths]
    print(json.dumps(results, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

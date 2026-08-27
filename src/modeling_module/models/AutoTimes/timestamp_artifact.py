from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class TimestampEmbeddingArtifact:
    """Verified timestamp embeddings consumed by AutoTimes numeric tokens."""

    tensor: torch.Tensor
    path: Path
    sha256: str

    @classmethod
    def load(cls, path: str | Path, expected_sha256: str) -> "TimestampEmbeddingArtifact":
        artifact_path = Path(path).expanduser().resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Timestamp artifact does not exist: {artifact_path}")
        actual_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if actual_sha256 != str(expected_sha256).lower():
            raise ValueError(
                "Timestamp artifact SHA256 mismatch: "
                f"expected={expected_sha256}, actual={actual_sha256}."
            )
        payload: Any = torch.load(artifact_path, map_location="cpu", weights_only=False)
        if isinstance(payload, Mapping):
            payload = payload.get("timestamp_embeddings")
        if not torch.is_tensor(payload):
            raise TypeError("Timestamp artifact must contain a tensor or timestamp_embeddings tensor.")
        tensor = payload.detach().to(dtype=torch.float32, device="cpu")
        if tensor.ndim not in (2, 3, 4):
            raise ValueError(
                "Timestamp embeddings must have shape [N,D], [B,N,D], or [B,C,N,D]."
            )
        if tensor.shape[-1] <= 0 or tensor.shape[-2] <= 0:
            raise ValueError("Timestamp embeddings cannot contain empty token or hidden dimensions.")
        if not torch.isfinite(tensor).all():
            raise ValueError("Timestamp embeddings must contain finite values only.")
        return cls(tensor=tensor, path=artifact_path, sha256=actual_sha256)

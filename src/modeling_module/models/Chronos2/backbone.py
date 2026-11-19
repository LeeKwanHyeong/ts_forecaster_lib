"""공식 Chronos2Pipeline과 내부 수치 backbone을 격리하는 어댑터.

공식 구현을 복사하거나 변경하지 않고 고정 버전 패키지를 사용한다.
다운로드·모델 로딩은 load 또는 실제 예측을 요청한 시점에만 수행한다.
"""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .configs import CHRONOS_PACKAGE_VERSION, Chronos2Config
from .contracts import Chronos2FutureFeatures


class Chronos2Backbone(nn.Module):
    """가중치를 동결한 공식 모델. 기존 CommonTrainer의 학습 대상과 분리."""

    def __init__(self, cfg: Chronos2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self._pipeline: Any = None
        self._model: nn.Module | None = None
        self.resolved_model_path: str | None = None
        self.weights_sha256: str | None = None
        self.config_sha256: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def train(self, mode: bool = True) -> Chronos2Backbone:
        # 상위 모델의 train() 호출로 공식 backbone의 dropout이 켜지는 상황 차단.
        return super().train(False)

    def load(self) -> Chronos2Backbone:
        if self.is_loaded:
            return self
        try:
            installed = importlib.metadata.version("chronos-forecasting")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ImportError("modeling-module[chronos2] 선택 의존성을 설치해야 합니다.") from exc
        if installed != CHRONOS_PACKAGE_VERSION:
            raise RuntimeError(f"검증한 chronos-forecasting=={CHRONOS_PACKAGE_VERSION}가 필요합니다: {installed}")
        if self.cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA가 없어 요청한 장치를 사용할 수 없습니다.")
        if self.cfg.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS가 없어 요청한 장치를 사용할 수 없습니다.")
        if self.cfg.model_path is not None:
            path = Path(self.cfg.model_path).expanduser().resolve()
        else:
            from huggingface_hub import snapshot_download

            # revision은 고정 SHA이며 기본 설정은 로컬 캐시만 허용.
            path = Path(snapshot_download(
                repo_id=self.cfg.model_id, revision=self.cfg.revision,
                cache_dir=self.cfg.cache_dir, local_files_only=self.cfg.local_files_only,
                allow_patterns=["config.json", "model.safetensors"], token=False,
            ))
        for filename in ("config.json", "model.safetensors"):
            if not (path / filename).is_file():
                raise FileNotFoundError(path / filename)
        config_bytes = (path / "config.json").read_bytes()
        config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        if self.cfg.expected_config_sha256 is not None and config_sha256 != self.cfg.expected_config_sha256:
            raise ValueError("설정 SHA-256이 지정한 공식 파일과 일치하지 않습니다.")
        config = json.loads(config_bytes)
        if config.get("chronos_pipeline_class") != "Chronos2Pipeline":
            raise ValueError("Chronos-2 전용 체크포인트가 아닙니다.")
        digest = hashlib.sha256()
        with (path / "model.safetensors").open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        weights_sha256 = digest.hexdigest()
        if self.cfg.expected_weights_sha256 is not None and weights_sha256 != self.cfg.expected_weights_sha256:
            raise ValueError("가중치 SHA-256이 지정한 공식 파일과 일치하지 않습니다.")
        from chronos import Chronos2Pipeline

        pipeline = Chronos2Pipeline.from_pretrained(
            str(path), device_map=self.cfg.device, dtype=getattr(torch, self.cfg.dtype),
            local_files_only=True, trust_remote_code=False, use_safetensors=True,
        )
        if self.cfg.context_length > pipeline.model_context_length:
            raise ValueError("context_length가 공식 모델의 최대 입력 길이를 초과합니다.")
        if self.cfg.horizon > pipeline.model_prediction_length:
            raise ValueError("horizon이 공식 모델의 직접 예측 범위를 초과합니다.")
        if min(self.cfg.quantile_levels) < min(pipeline.quantiles) or max(self.cfg.quantile_levels) > max(pipeline.quantiles):
            raise ValueError("요청한 분위수가 공식 모델의 학습 분위수 범위를 벗어납니다.")
        self._model = pipeline.model.requires_grad_(False).eval()
        self._pipeline = pipeline
        self.resolved_model_path = str(path)
        self.weights_sha256 = weights_sha256
        self.config_sha256 = config_sha256
        self.train(False)
        return self

    @torch.inference_mode()
    def predict_quantiles(self, inputs: list[dict[str, Any]]) -> torch.Tensor:
        self.load()
        self._model.eval()
        quantiles, _ = self._pipeline.predict_quantiles(
            inputs=inputs, prediction_length=self.cfg.horizon,
            quantile_levels=list(self.cfg.quantile_levels), batch_size=self.cfg.batch_size,
            context_length=self.cfg.context_length, cross_learning=False, limit_prediction_length=True,
        )
        expected = (len(inputs), 1, self.cfg.horizon, len(self.cfg.quantile_levels))
        values = torch.stack(quantiles).detach().to(device="cpu", dtype=torch.float32)
        if tuple(values.shape) != expected or not torch.isfinite(values).all():
            raise RuntimeError(f"공식 모델의 출력 shape 또는 유한성 오류: {tuple(values.shape)}, expected={expected}")
        return values[:, 0]

    @torch.inference_mode()
    def encode_future_tokens(self, context: torch.Tensor) -> Chronos2FutureFeatures:
        """외생변수 없는 (B,L)의 미래 패치 표현을 추출하는 관찰용 경로."""
        self.load()
        self._model.eval()
        patch_size = self._pipeline.model_output_patch_size
        count = math.ceil(self.cfg.horizon / patch_size)
        context = context.to(device=self._model.device, dtype=torch.float32)
        encoded, (location, scale), _, context_patches = self._model.encode(
            context=context, num_output_patches=count,
        )
        # 마지막 count개가 예측 구간의 토큰이며, 과거 패치 토큰과 구분.
        return Chronos2FutureFeatures(
            future_tokens=encoded.last_hidden_state[:, -count:].detach().cpu(),
            location=location.detach().cpu(), scale=scale.detach().cpu(),
            num_context_patches=context_patches, output_patch_size=patch_size, horizon=self.cfg.horizon,
        )

    def provenance(self) -> dict[str, Any]:
        parameter = next(self._model.parameters()) if self._model is not None else None
        return {
            "model_id": self.cfg.model_id, "declared_revision": self.cfg.revision,
            "model_path": self.resolved_model_path, "chronos_forecasting": CHRONOS_PACKAGE_VERSION,
            "point_method": "median", "cross_learning": False, "training_performed": False,
            "weights_sha256": self.weights_sha256,
            "weights_verified": self.weights_sha256 is not None and self.weights_sha256 == self.cfg.expected_weights_sha256,
            "config_sha256": self.config_sha256,
            "config_verified": self.config_sha256 is not None and self.config_sha256 == self.cfg.expected_config_sha256,
            "device": str(parameter.device) if parameter is not None else self.cfg.device,
            "dtype": str(parameter.dtype) if parameter is not None else self.cfg.dtype,
        }

"""내부에 포함한 공식 연산 소스로 새 가중치를 만드는 backbone. 동결 추론과 분리."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.metadata

import torch
from torch import nn

from .configs import CHRONOS_PACKAGE_VERSION
from .training.configs import Chronos2TrainableConfig, TRAIN_TRANSFORMERS_VERSION
from .training.batch import Chronos2TrainingBatch, Chronos2TrainingOutput
from .training.contracts import manifest_sha256


def _check_dependencies():
    for name in ("transformers", "einops", "packaging"):
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ImportError("내부 소스 모델에는 modeling-module[chronos2-native] 선택 의존성이 필요합니다.") from exc
        if name == "transformers" and installed != TRAIN_TRANSFORMERS_VERSION:
            raise RuntimeError(f"검증한 transformers=={TRAIN_TRANSFORMERS_VERSION}가 필요합니다: {installed}")
    from packaging.version import Version
    if not Version("0.7") <= Version(importlib.metadata.version("einops")) < Version("1"):
        raise RuntimeError("내부 소스 모델에는 einops>=0.7,<1이 필요합니다.")
    if not Version("2.2") <= Version(torch.__version__) < Version("3"):
        raise RuntimeError("내부 소스 모델에는 torch>=2.2,<3이 필요합니다.")


def _parameters_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(f"{name}:{tuple(tensor.shape)}:float32\n".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class Chronos2TrainableBackbone(nn.Module):
    """생성 시 실제 파라미터를 준비하므로 곧바로 optimizer에 전달 가능한 모델.

    import만으로 optional 패키지를 읽지 않으며, 생성해도 체크포인트·Hub에
    접근하지 않는다. 학습 실행·데이터 표집·저장은 다음 단계의 별도 책임.
    """

    def __init__(self, cfg: Chronos2TrainableConfig | None = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else Chronos2TrainableConfig()
        if not isinstance(self.cfg, Chronos2TrainableConfig):
            raise TypeError("cfg는 Chronos2TrainableConfig여야 합니다.")
        _check_dependencies()
        if self.cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA가 없어 요청한 장치를 사용할 수 없습니다.")
        if self.cfg.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS가 없어 요청한 장치를 사용할 수 없습니다.")
        from .native.config import Chronos2CoreConfig
        from .native.model import Chronos2Model as NativeChronos2Model

        # CPU 생성기만 잠시 사용하므로 호출자의 CPU/CUDA seed를 전역 변경하지 않음.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(self.cfg.seed)
            native_cfg = Chronos2CoreConfig(**self.cfg.native_kwargs())
            self.native_model = NativeChronos2Model(native_cfg).to(dtype=torch.float32)
        self.initial_parameters_sha256 = _parameters_sha256(self.native_model)
        self.native_model.to(self.cfg.device).requires_grad_(True).train()

    def provenance(self) -> dict:
        return {"model_kind": "chronos2_scratch", "initialization_source": "random",
                "pretrained_weights_loaded": False, "optimizer_managed_externally": True,
                "initial_parameters_sha256": self.initial_parameters_sha256,
                "architecture_sha256": manifest_sha256(self.cfg.native_kwargs()),
                "chronos_forecasting": CHRONOS_PACKAGE_VERSION, "transformers": TRAIN_TRANSFORMERS_VERSION,
                "implementation_source": "vendored", "chronos_package_required": False,
                "implementation_module": type(self.native_model).__module__,
                "upstream_commit": "0aba28e9360b6355d5a919596c90b16c25939f02",
                "config": asdict(self.cfg),
                "parameters": sum(p.numel() for p in self.native_model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.native_model.parameters() if p.requires_grad)}

    def forward(self, batch: Chronos2TrainingBatch) -> Chronos2TrainingOutput:
        if not isinstance(batch, Chronos2TrainingBatch):
            raise TypeError("학습 forward에는 Chronos2TrainingBatch가 필요합니다.")
        inputs = batch.native_inputs(self.cfg, self.native_model.device)
        result = self.native_model(**inputs, num_output_patches=self.cfg.num_output_patches)
        loss = result.loss
        if loss is None or loss.ndim != 0 or not torch.isfinite(loss):
            raise RuntimeError("공식 forward가 유한한 scalar loss를 반환하지 않았습니다.")
        raw = result.quantile_preds
        expected = (len(inputs["context"]), len(self.cfg.quantile_levels), self.cfg.num_output_patches * self.cfg.patch_size)
        if raw is None or tuple(raw.shape) != expected or not torch.isfinite(raw).all():
            raise RuntimeError("공식 forward의 분위수 shape 또는 유한성 오류입니다.")
        if self.training and torch.is_grad_enabled() and not loss.requires_grad:
            raise RuntimeError("학습 모드에서 손실의 역전파 경로가 끊겼습니다.")
        # H27의 남는 5개 위치는 native loss에서 mask=0, 사용자 예측에서도 제외.
        quantiles = raw[:, :, :self.cfg.horizon].transpose(1, 2).contiguous()
        median = self.cfg.quantile_levels.index(0.5)
        return Chronos2TrainingOutput(
            loss=loss, point=quantiles[:, :, median:median + 1], quantiles=quantiles,
            quantile_levels=self.cfg.quantile_levels,
            observed_target_points=int(inputs["future_target_mask"].sum().item()),
            raw_prediction_length=expected[-1], metadata=self.provenance())

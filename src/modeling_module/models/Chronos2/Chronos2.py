"""기존 모델과 독립된 Chronos-2 예측 인터페이스."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .backbone import Chronos2Backbone
from .configs import Chronos2Config
from .contracts import Chronos2Forecast, Chronos2FutureFeatures, prepare_inputs


class Chronos2Model(nn.Module):
    """(B,L,1)을 받아 기본 27주 예측과 분위수를 반환하는 zero-shot 모델.

    predict는 상세 결과, forward는 기존 PyTorch 모델처럼 점 예측 tensor 반환.
    점 예측은 공식 학습 분위수의 중앙값이며 산술 평균과 구분한다.
    """

    def __init__(self, cfg: Chronos2Config | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = Chronos2Config()
        elif isinstance(cfg, Mapping):
            cfg = Chronos2Config(**cfg)
        if not isinstance(cfg, Chronos2Config):
            raise TypeError("Chronos2Config 또는 설정 사전이 필요합니다.")
        self.cfg = cfg
        self.model_name = "Chronos2"
        self.horizon = cfg.horizon
        self.backbone = Chronos2Backbone(cfg)

    def load(self) -> Chronos2Model:
        self.backbone.load()
        return self

    def predict(self, x: Any, *, past_covariates=None, future_covariates=None) -> Chronos2Forecast:
        context, inputs = prepare_inputs(x, self.cfg, past_covariates=past_covariates, future_covariates=future_covariates)
        values = self.backbone.predict_quantiles(inputs)
        median_index = self.cfg.quantile_levels.index(0.5)
        return Chronos2Forecast(
            point=values[:, :, median_index:median_index + 1], quantiles=values,
            quantile_levels=self.cfg.quantile_levels,
            metadata={**self.backbone.provenance(), "context_length_used": context.shape[1]},
        )

    def forward(self, x: Any, *, past_covariates=None, future_covariates=None) -> torch.Tensor:
        return self.predict(x, past_covariates=past_covariates, future_covariates=future_covariates).point

    def encode_future_tokens(self, x: Any) -> Chronos2FutureFeatures:
        """ReasonCast 연결 전, 수치 backbone의 미래 표현을 확인하는 경로."""
        context, _ = prepare_inputs(x, self.cfg)
        return self.backbone.encode_future_tokens(context)

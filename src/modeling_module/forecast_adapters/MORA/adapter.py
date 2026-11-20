"""기존 모델 밖에서 과거 입력과 점예측을 받는 독립 MORA 연결 모듈."""

from copy import deepcopy

import numpy as np

from ..contracts import ForecastQuery, ForecastFitData, AdaptedForecast
from .configs import MORAConfig
from .fitting import fit_state, predict_coefficients, canonical_state
from .native.basis import reconstruct


class MORAAdapter:
    def __init__(self, config=None):
        self._config = config if config is not None else MORAConfig()
        if not isinstance(self.config, MORAConfig):
            raise TypeError("config는 MORAConfig여야 합니다.")
        self._state = None

    @property
    def config(self):
        return self._config

    def _check_query(self, query):
        if not isinstance(query, ForecastQuery):
            raise TypeError("ForecastQuery가 필요합니다.")
        if query.history.shape[1] != self.config.lookback or query.backbone_prediction.shape[1] != self.config.horizon:
            raise ValueError("입력 길이가 MORA 설정과 다릅니다.")
        if self.config.scope == "single_series" and len(set(query.series_ids)) != 1:
            raise ValueError("single_series 설정에는 하나의 시계열 ID만 허용됩니다.")

    def fit(self, train, validation):
        if not isinstance(train, ForecastFitData) or not isinstance(validation, ForecastFitData):
            raise TypeError("fit에는 학습·검증 ForecastFitData가 필요합니다.")
        self._check_query(train.query)
        self._check_query(validation.query)
        if train.query.signature != validation.query.signature:
            raise ValueError("학습·검증의 기준 모델·채널 순서·단위·shape가 다릅니다.")
        if self.config.scope == "single_series" and set(train.query.series_ids) != set(validation.query.series_ids):
            raise ValueError("부품별 보정기의 학습·검증 시계열 ID가 다릅니다.")
        if train.query.origins.max() + self.config.horizon - 1 >= validation.query.origins.min():
            raise ValueError("학습 정답과 검증 정답 기간이 겹칩니다.")
        # 계산이 끝나기 전에는 기존 적합 상태를 덮어쓰지 않음.
        state = fit_state(train, validation, self.config)
        self._state = canonical_state(state)
        return self

    def predict(self, query):
        """정답 인자를 받지 않으며 fit 이후의 시점만 기본 허용. 파라미터 갱신 없음."""
        self._check_query(query)
        if self._state is None:
            raise RuntimeError("MORAAdapter.fit 또는 load를 먼저 실행해야 합니다.")
        state = self._state
        if query.signature != state.metadata["signature"]:
            raise ValueError("보정기와 기준 모델·채널 순서·단위 계약이 다릅니다.")
        if self.config.scope == "single_series" and set(query.series_ids) != set(state.metadata["fit_series_ids"]):
            raise ValueError("다른 부품에 부품별 보정기를 적용할 수 없습니다.")
        if query.origins.min() <= state.metadata["calibration_target_end"]:
            raise ValueError("공개 predict는 calibration 정답 종료 이후의 query만 허용합니다.")
        first, second = predict_coefficients(state, query.history, query.backbone_prediction, query.origins, self.config)
        correction = reconstruct((first + second) * state.calibration, state.basis)
        point = query.backbone_prediction + correction
        if not np.isfinite(point).all():
            raise RuntimeError("보정 결과가 유한하지 않습니다.")
        return AdaptedForecast(point, query.backbone_prediction.copy(), correction,
                               {"adapter": "MORA", "implementation": "paper_independent_v1", "output_kind": "corrected_point",
                                "quantiles_calibrated": False, "backbone_id": query.backbone_id, "scope": self.config.scope})

    @property
    def diagnostics(self):
        if self._state is None:
            raise RuntimeError("적합된 상태가 없습니다.")
        return deepcopy(self._state.metadata)

    def save(self, path):
        from .checkpoints import save_adapter
        return save_adapter(self, path)

    @classmethod
    def load(cls, path):
        from .checkpoints import load_adapter
        return load_adapter(path)

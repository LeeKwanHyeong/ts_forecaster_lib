"""Eq. (4)~(6), (26): 외삽과 absolute/last-centered ridge의 미래 구간 평균 추정."""

from dataclasses import dataclass

import numpy as np

from .basis import coarse_values
from .features import compact_features


def extrapolated(history, operator, period):
    horizon = operator.shape[1]
    latest = np.repeat(history[:, -1:, :], horizon, axis=1)
    width = min(period, history.shape[1])
    repeated = history[:, -width:, :][:, np.arange(horizon) % width, :]
    return np.concatenate([coarse_values(latest, operator), coarse_values(repeated, operator)], -1)


@dataclass
class RidgeView:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    weight: np.ndarray

    @classmethod
    def fit(cls, features, target, strength):
        mean, scale = features.mean(0), np.maximum(features.std(0), 1e-5)
        y_mean, y_scale = target.mean(0), np.maximum(target.std(0), 1e-4)
        x, y = (features - mean) / scale, (target - y_mean) / y_scale
        weights = []
        for channel in range(x.shape[1]):
            design = x[:, channel]
            weights.append(np.linalg.solve(design.T @ design + strength * np.eye(design.shape[1]),
                                           design.T @ y[:, channel]))
        return cls(mean, scale, y_mean, y_scale, np.stack(weights))

    def predict(self, features):
        normalized = (features - self.feature_mean) / self.feature_scale
        return np.einsum("ncd,cdq->ncq", normalized, self.weight) * self.target_scale + self.target_mean


@dataclass
class CoarseEstimator:
    absolute: RidgeView
    relative: RidgeView
    period: int

    @classmethod
    def fit(cls, history, origins, target, operator, period, strength):
        latest = history[:, -1:, :]
        coarse = coarse_values(target, operator)
        absolute = RidgeView.fit(compact_features(history, origins, period), coarse, strength)
        relative = RidgeView.fit(compact_features(history - latest, origins, period),
                                 coarse - latest[:, 0, :, None], strength)
        return cls(absolute, relative, period)

    def predict(self, history, origins):
        latest = history[:, -1:, :]
        absolute = self.absolute.predict(compact_features(history, origins, self.period))
        relative = self.relative.predict(compact_features(history - latest, origins, self.period))
        return 0.5 * (absolute + relative + latest[:, 0, :, None])

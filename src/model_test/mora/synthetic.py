"""관측 이력만 사용하는 고정 기준 예측과 연속 시점의 합성 학습·검증·시험 창."""

import numpy as np

from modeling_module.forecast_adapters.contracts import ForecastQuery, ForecastFitData
from modeling_module.forecast_adapters.MORA import MORAConfig


def make_synthetic(config=None, *, seed=42, train_count=96, validation_count=24, test_count=24, channels=2):
    config = config if config is not None else MORAConfig(period=13, correction_rank=12)
    if min(train_count, validation_count, test_count, channels) < 1:
        raise ValueError("합성 표본 수와 채널 수는 양수여야 합니다.")
    n = train_count + validation_count + test_count
    stride = config.lookback + config.horizon
    origins = config.lookback + np.arange(n, dtype=np.int64) * stride
    time = np.arange(int(origins[-1] + config.horizon), dtype=np.float64)
    generator = np.random.default_rng(seed)
    values = np.stack([60 + 15 * channel + 0.001 * time
                       + 6 * np.sin(2 * np.pi * time / config.period + channel * 0.4)
                       + 2 * np.cos(2 * np.pi * time / (4 * config.period))
                       + generator.normal(0, 0.15, len(time)) for channel in range(channels)], axis=-1)
    history = np.stack([values[origin - config.lookback:origin] for origin in origins])
    target = np.stack([values[origin:origin + config.horizon] for origin in origins])
    # 정답은 읽지 않는 고정 baseline. 일정한 과소 예측을 의도적으로 넣은 계산 검증 자료.
    recent = history[:, -min(config.period, config.lookback):].mean(axis=1, keepdims=True)
    base = np.repeat(recent - 8.0, config.horizon, axis=1)
    query = ForecastQuery(history, base, origins, tuple(f"signal_{i}" for i in range(channels)),
                          ("synthetic_continuous_series",) * n, "synthetic_history_mean_minus_8_v1", "synthetic_units")
    train = np.arange(train_count)
    validation = np.arange(train_count, train_count + validation_count)
    test = np.arange(train_count + validation_count, n)
    return (ForecastFitData(query.take(train), target[train], "synthetic_history_only"),
            ForecastFitData(query.take(validation), target[validation], "synthetic_history_only"),
            ForecastFitData(query.take(test), target[test], "synthetic_history_only"))


def point_metrics(target, prediction):
    error = np.asarray(target) - np.asarray(prediction)
    return {"mse": float(np.mean(error**2)), "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2)))}

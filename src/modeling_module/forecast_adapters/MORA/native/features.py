"""부록 C.1의 compact 78개 과거 특징. 미래 정답을 인자로 받지 않는 함수."""

import numpy as np


def _slope(values):
    time = np.arange(values.shape[-1], dtype=np.float64)
    time -= time.mean()
    return np.sum(values * time, axis=-1) / np.sum(time**2)


def _block_means(values, count=12):
    return np.stack([values[..., indices].mean(-1) for indices in np.array_split(np.arange(values.shape[-1]), count)], -1)


def compact_features(history, origins, period):
    values = np.asarray(history, dtype=np.float64).transpose(0, 2, 1)
    n, channels, length = values.shape
    if length < 24:
        raise ValueError("compact 특징에는 최소 24개 관측 위치가 필요합니다.")
    positions = np.arange(length) + 0.5
    cosine = np.sqrt(2.0 / length) * np.cos(np.pi * np.arange(16)[:, None] * positions / length)
    cosine[0] /= np.sqrt(2.0)
    dct = values @ cosine.T
    differences = np.diff(values, axis=-1)
    statistics = np.stack([values.mean(-1), values.std(-1), values[..., 0], values[..., -1],
                           values[..., -1] - values[..., 0], _slope(values),
                           differences.mean(-1), differences.std(-1)], -1)
    recent = values[..., -24:]
    recent_stats = []
    for width in (8, 16, 24):
        window = values[..., -width:]
        recent_stats.extend([window.mean(-1), window.std(-1), _slope(window), window[..., -1] - window[..., 0]])
    # 최근 24개 관측의 차분은 23개. 빈 구간 없이 12개로 나누는 명시적 구현 선택.
    dynamics = np.concatenate([_block_means(recent), _block_means(np.diff(recent, axis=-1)),
                               np.stack(recent_stats, -1)], -1)
    time = np.stack([function(2 * np.pi * origins / p) for p in (period, period / 2, 7 * period)
                     for function in (np.sin, np.cos)], -1)
    result = np.concatenate([dct, statistics, _block_means(values), dynamics,
                             np.broadcast_to(time[:, None, :], (n, channels, 6))], -1)
    if result.shape[-1] != 78 or not np.isfinite(result).all():
        raise ValueError("특징 차원 또는 유한성 오류입니다.")
    return result

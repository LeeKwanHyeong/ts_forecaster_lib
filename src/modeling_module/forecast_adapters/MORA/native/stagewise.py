"""Eq. (35)~(38)의 비중심 2차 모멘트와 reliability gain. 역행렬 직접 계산 없음."""

import numpy as np


def _shrink(matrix, amount):
    return (1 - amount) * matrix + amount * np.diag(np.diag(matrix))


def fit_mapping(discrepancy, target, geometry, config):
    n, channels, width = discrepancy.shape
    result = []
    for channel in range(channels):
        d, z = discrepancy[:, channel], target[:, channel]
        rms = np.sqrt(np.mean(d**2, axis=1))
        floor = max(1e-5, float(np.quantile(rms, 0.1)))
        weight = np.maximum(rms, floor) ** (-2 * config.weight_power)
        eta = d - z @ geometry.T
        zz = _shrink(z.T @ (weight[:, None] * z) / n, config.shrinkage)
        noise = _shrink(eta.T @ (weight[:, None] * eta) / n, config.shrinkage)
        dd = d.T @ (weight[:, None] * d) / n
        zd = z.T @ (weight[:, None] * d) / n
        mix = config.structured_mix
        covariance = (1 - mix) * (geometry @ zz @ geometry.T + config.noise_scale * noise) + mix * dd
        numerator = (1 - mix) * zz @ geometry.T + mix * zd
        epsilon = 1e-6 * max(float(np.trace(covariance)) / width, 1e-12)
        result.append(np.linalg.solve(covariance + epsilon * np.eye(width), numerator.T).T)
    mapping = np.stack(result)
    if not np.isfinite(mapping).all():
        raise ValueError("잔차 매핑이 유한하지 않습니다.")
    return mapping


def apply_mapping(discrepancy, mapping):
    return np.einsum("ncp,ckp->nck", discrepancy, mapping)


def reliability(prediction, target, blocks, maximum, conservative=False):
    def ratio(indices):
        numerator = np.sum(prediction[indices] * target[indices], axis=(0, 1))
        denominator = np.sum(prediction[indices] ** 2, axis=(0, 1))
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)
    gain = ratio(np.arange(len(target)))
    if conservative:
        gain = np.minimum(gain, np.min([ratio(block) for block in blocks], axis=0))
    return np.clip(gain, 0, maximum)

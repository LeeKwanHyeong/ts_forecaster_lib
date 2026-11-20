"""Eq. (3), (31)~(34): 구간 평균과 첫 예측 위치를 보존하는 보정 공간."""

import numpy as np


def coarse_operator(horizon, count):
    if type(horizon) is not int or type(count) is not int or not 1 <= count <= horizon:
        raise ValueError("1<=구간 수<=horizon이 필요합니다.")
    operator = np.zeros((count, horizon), dtype=np.float64)
    for row, indices in enumerate(np.array_split(np.arange(horizon), count)):
        operator[row, indices] = 1.0 / len(indices)
    return operator


def tapered_basis(horizon, rank, taper_steps):
    if not 1 <= rank < horizon or not 2 <= taper_steps <= horizon:
        raise ValueError("보정 기저의 rank와 taper 길이가 유효하지 않습니다.")
    positions = np.arange(horizon, dtype=np.float64)[:, None] + 0.5
    modes = np.arange(rank, dtype=np.float64)[None, :]
    cosine = np.sqrt(2.0 / horizon) * np.cos(np.pi * positions * modes / horizon)
    cosine[:, 0] /= np.sqrt(2.0)
    taper = np.ones(horizon)
    taper[:taper_steps] = np.sin(np.pi / 2 * np.arange(taper_steps) / (taper_steps - 1)) ** 2
    # 첫 행을 제외한 공간에서 QR 후 0행을 붙여 수치 오차로 인한 W0 변경도 방지.
    tail, triangular = np.linalg.qr((cosine * taper[:, None])[1:], mode="reduced")
    signs = np.where(np.diag(triangular) < 0, -1.0, 1.0)
    return np.vstack([np.zeros((1, rank)), tail * signs])


def coarse_values(values, operator):
    return np.einsum("qh,nhc->ncq", operator, values)


def coefficients(values, basis):
    return np.einsum("hk,nhc->nck", basis, values)


def reconstruct(values, basis):
    return np.einsum("hk,nck->nhc", basis, values)

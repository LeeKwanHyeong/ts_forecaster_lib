"""Eq. (39)의 검증 전용 그룹별 scale. 제한된 IRLS와 box-constrained least squares."""

import numpy as np
from scipy.optimize import lsq_linear

def calibrate(base, target, coefficient, basis, config, origins):
    order = np.argsort(origins, kind="stable")
    blocks = np.array_split(order, 4)
    if any(len(block) == 0 for block in blocks):
        raise ValueError("calibration에는 최소 4개 query가 필요합니다.")
    residual = target - base
    mse_weight = np.zeros(len(base))
    mae_weight = np.zeros(len(base))
    errors = []
    for block in blocks:
        mse = float(np.mean(residual[block] ** 2))
        mae = float(np.mean(np.abs(residual[block])))
        errors.append(mae)
        mse_weight[block] = 0.1 / (4 * len(block) * base.shape[1] * base.shape[2] * max(mse, 1e-12))
        mae_weight[block] = 0.9 / (4 * len(block) * base.shape[1] * base.shape[2] * max(mae, 1e-12))
    epsilon = max(1e-3 * float(np.mean(errors)), 1e-7)
    groups = np.array_split(np.arange(basis.shape[1]), config.calibration_groups)
    scales, details = [], []
    for channel in range(base.shape[2]):
        design = np.stack([np.einsum("hk,nk->nh", basis[:, group], coefficient[:, channel, group])
                           for group in groups], -1).reshape(-1, len(groups))
        y = residual[:, :, channel].reshape(-1)
        wm = np.repeat(mse_weight, base.shape[1])
        wa = np.repeat(mae_weight, base.shape[1])

        def objective(values):
            error = design @ values - y
            return float(np.sum(wm * error**2 + wa * np.sqrt(error**2 + epsilon**2)))

        values = np.full(len(groups), min(1.0, config.calibration_max))
        best = np.zeros(len(groups))
        if objective(values) < objective(best):
            best = values.copy()
        for _ in range(config.calibration_iterations):
            error = design @ values - y
            weight = wm + wa / (2 * np.sqrt(error**2 + epsilon**2))
            root_weight = np.sqrt(weight)
            solution = lsq_linear(design * root_weight[:, None], y * root_weight,
                                   bounds=(0, config.calibration_max), tol=1e-10, max_iter=200)
            if not solution.success or not np.isfinite(solution.x).all():
                raise RuntimeError("검증 calibration 최적화가 수렴하지 않았습니다.")
            updated = solution.x
            if objective(updated) < objective(best):
                best = updated.copy()
            change = np.max(np.abs(updated - values))
            values = updated
            if change < 1e-5:
                break
        expanded = np.empty(basis.shape[1])
        for group, value in zip(groups, best):
            expanded[group] = value
        scales.append(expanded)
        details.append({"channel": channel, "uncorrected_objective": objective(np.zeros(len(groups))),
                        "calibrated_objective": objective(best), "group_scales": best.tolist()})
    return np.stack(scales), details

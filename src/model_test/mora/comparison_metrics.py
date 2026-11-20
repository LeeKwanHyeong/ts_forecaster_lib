"""동일 부품·주차의 보정 전후 점예측 지표. 0수요 분모를 임의 보정하지 않음."""

import numpy as np


def point_metrics(target, prediction):
    y, p = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 3 or len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("유한한 동일 shape (N,H,C) 정답·예측이 필요합니다.")
    error = p - y
    absolute = np.abs(error)
    denominator = float(np.abs(y).sum())
    part_total = np.abs(y).sum(axis=(1, 2))
    positive = part_total > 0
    zeros = ~positive
    return {"mae": float(absolute.mean()), "mse": float(np.mean(error**2)),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "wape": float(absolute.sum() / denominator) if denominator else None,
            "macro_part_mae": float(absolute.mean(axis=(1, 2)).mean()),
            "macro_part_rmse": float(np.sqrt(np.mean(error**2, axis=(1, 2))).mean()),
            "macro_part_wape_positive_only": float(np.mean(absolute.sum(axis=(1, 2))[positive] / part_total[positive])) if positive.any() else None,
            "zero_total_parts": int(zeros.sum()),
            "zero_total_parts_mae": float(absolute[zeros].mean()) if zeros.any() else None,
            "negative_predictions": int((p < 0).sum()), "mean_signed_error": float(error.mean()),
            "target_total": float(y.sum()), "prediction_total": float(p.sum())}


def paired_comparison(target, baseline, corrected, seed=42, bootstrap_draws=2000):
    """부품을 한 묶음으로 재표집. 27개 주차를 독립 표본으로 가정하지 않음."""
    y = np.asarray(target, dtype=np.float64)
    before, after = point_metrics(y, baseline), point_metrics(y, corrected)
    baseline_mae = np.mean(np.abs(np.asarray(baseline) - y), axis=(1, 2))
    corrected_mae = np.mean(np.abs(np.asarray(corrected) - y), axis=(1, 2))
    delta = corrected_mae - baseline_mae
    tolerance = 1e-10
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(bootstrap_draws)])
    return {"baseline": before, "mora": after,
            "mae_reduction_ratio": 1 - after["mae"] / before["mae"] if before["mae"] else None,
            "rmse_reduction_ratio": 1 - after["rmse"] / before["rmse"] if before["rmse"] else None,
            "part_mae": {"improved": int((delta < -tolerance).sum()), "unchanged": int((np.abs(delta) <= tolerance).sum()),
                         "worsened": int((delta > tolerance).sum()), "tolerance": tolerance},
            "paired_part_bootstrap": {"draws": bootstrap_draws, "seed": seed, "statistic": "mora_minus_baseline_mae",
                                      "mean_delta": float(delta.mean()), "percentile_95_ci": np.quantile(means, [.025, .975]).tolist()},
            "horizon": [{"week": h + 1, "baseline_mae": float(np.abs(baseline[:, h] - y[:, h]).mean()),
                         "mora_mae": float(np.abs(corrected[:, h] - y[:, h]).mean())} for h in range(y.shape[1])]}

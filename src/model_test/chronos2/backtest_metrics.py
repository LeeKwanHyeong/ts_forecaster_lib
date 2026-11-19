"""동일 부품·시점의 27주 오차 집계. 무수요 정답도 MAE에서 제외하지 않음."""

from __future__ import annotations

import numpy as np
import polars as pl

from .balanced_sampling import HORIZON


def pair_metrics(cohort: list[dict], actual: np.ndarray, prediction: np.ndarray, model: str) -> list[dict]:
    if actual.shape != prediction.shape or actual.shape != (len(cohort), HORIZON):
        raise ValueError("정답과 예측은 동일한 (표본 수, 27) 배열이어야 합니다.")
    if not np.isfinite(actual).all() or not np.isfinite(prediction).all():
        raise ValueError("정답과 예측은 모두 유한해야 합니다.")
    result = []
    for row, truth, pred in zip(cohort, actual, prediction):
        error = pred.astype(np.float64) - truth
        result.append({**row, "model": model, "points": HORIZON,
                       "actual_sum": float(truth.sum()), "abs_error": float(np.abs(error).sum()),
                       "squared_error": float(np.square(error).sum()), "signed_error": float(error.sum()),
                       "mae": float(np.abs(error).mean()), "zero_actual_points": int((truth == 0).sum()),
                       "negative_prediction_points": int((pred < 0).sum())})
    return result


def aggregate(records: pl.DataFrame, group_by: list[str]) -> list[dict]:
    result = records.group_by(group_by).agg(
        pl.len().alias("pairs"), pl.col("oper_part_no").n_unique().alias("unique_parts"),
        *[pl.col(name).sum() for name in ("points", "actual_sum", "abs_error", "squared_error",
                                         "signed_error", "zero_actual_points", "negative_prediction_points")],
        (pl.col("actual_sum") == 0).sum().alias("zero_actual_pairs"),
    ).with_columns(
        (pl.col("abs_error") / pl.col("points")).alias("mae"),
        (pl.col("squared_error") / pl.col("points")).sqrt().alias("rmse"),
        (pl.col("signed_error") / pl.col("points")).alias("bias_per_week"),
        pl.when(pl.col("actual_sum") > 0)
        .then(100 * pl.col("abs_error") / pl.col("actual_sum")).otherwise(None).alias("wape_pct"),
    ).sort(group_by)
    return result.to_dicts()


def paired_comparison(records: pl.DataFrame) -> dict:
    """한 행은 같은 부품·예측 시작 주차. MAE가 작은 모델의 승수 집계."""
    keys = ["panel", "origin", "oper_part_no", "pattern"]
    chronos = records.filter(pl.col("model") == "chronos2").select(*keys, pl.col("mae").alias("chronos_mae"))
    naive = records.filter(pl.col("model") == "naive").select(*keys, pl.col("mae").alias("naive_mae"))
    paired = chronos.join(naive, on=keys, validate="1:1")
    if paired.height != chronos.height or paired.height != naive.height:
        raise ValueError("두 모델의 평가 표본이 일치하지 않습니다.")
    delta = paired["chronos_mae"].to_numpy() - paired["naive_mae"].to_numpy()
    tie = np.isclose(paired["chronos_mae"].to_numpy(), paired["naive_mae"].to_numpy(), rtol=1e-7, atol=1e-8)
    return {"pairs": paired.height, "chronos_wins": int(((delta < 0) & ~tie).sum()),
            "naive_wins": int(((delta > 0) & ~tie).sum()), "ties": int(tie.sum()),
            "tie_rule": "MAE np.isclose(rtol=1e-7, atol=1e-8)"}


def cluster_bootstrap(records: pl.DataFrame, seed: int, repetitions: int = 2000) -> dict:
    """동일 부품의 여러 주차를 함께 재표집하여 pooled WAPE 차이의 구간 추정.

    차이=Chronos2-Naive, 음수일수록 Chronos2 우세. 달력 주차 간 공통 충격은
    이 방식으로 제거되지 않으며 모집단 전체나 미래 성능의 보장 구간이 아님.
    """
    if type(repetitions) is not int or repetitions < 2:
        raise ValueError("bootstrap repetitions는 2 이상이어야 합니다.")
    parts = records.group_by("oper_part_no").agg(
        pl.col("abs_error").filter(pl.col("model") == "chronos2").sum().alias("chronos_error"),
        pl.col("abs_error").filter(pl.col("model") == "naive").sum().alias("naive_error"),
        pl.col("actual_sum").filter(pl.col("model") == "chronos2").sum().alias("denominator"),
    ).sort("oper_part_no")
    if parts.height == 0:
        raise ValueError("bootstrap 표본이 없습니다.")
    values = parts.select("chronos_error", "naive_error", "denominator").to_numpy()
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        totals = values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
        if totals[2] > 0:
            differences.append(100 * (totals[0] - totals[1]) / totals[2])
    bounds = np.quantile(differences, [0.025, 0.975]).tolist() if differences else [None, None]
    return {"unit": "oper_part_no (all sampled origins retained)", "unique_parts": parts.height,
            "repetitions": repetitions, "valid_repetitions": len(differences),
            "delta_pooled_wape_pp_95pct": bounds, "seed": seed}


def panel_summary(records: pl.DataFrame, seed: int, repetitions: int = 2000) -> dict:
    overall = aggregate(records, ["model"])
    strata = aggregate(records, ["origin", "pattern", "model"])
    macro = {}
    for model in ("chronos2", "naive"):
        buckets = [row["wape_pct"] for row in strata if row["model"] == model]
        valid = [value for value in buckets if value is not None]
        macro[model] = {"equal_origin_pattern_wape_pct": float(np.mean(valid)) if valid else None,
                        "valid_buckets": len(valid), "undefined_zero_demand_buckets": len(buckets) - len(valid)}
    return {"pooled": overall, "macro": macro, "paired_mae": paired_comparison(records),
            "bootstrap": cluster_bootstrap(records, seed, repetitions)}

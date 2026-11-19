"""저장된 예측 CSV를 평가 코드와 독립된 연산으로 재검산하는 도구.

선정·오차 집계 함수를 재사용하지 않고 원 CSV, 입력 배열, 원천 parquet를 대조.
추가 모델 호출이나 DB 변경 없이 증적의 행수·동일 조건·산식을 검증.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual, expected, message: str) -> None:
    if actual is None or expected is None:
        require(actual is None and expected is None, message)
    else:
        require(bool(np.isclose(actual, expected, rtol=1e-9, atol=1e-8)), message)


def audit(output: Path) -> dict:
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "selection_manifest.json").read_text(encoding="utf-8"))
    cohort = pl.read_csv(output / "cohort_manifest.csv")
    raw = pl.read_csv(output / "forecast_27_weeks.csv")
    require(digest(Path(summary["source"])) == summary["source_sha256"], "원천 파일 해시 불일치")
    require(digest(output / "cohort_manifest.csv") == summary["cohort_sha256"], "선정 표본 변경")
    require(digest(output / "forecast_27_weeks.csv") == summary["forecast_csv_sha256"], "원 예측 변경")
    require(manifest["source_sha256"] == summary["source_sha256"], "선정 단계 원천 불일치")
    require(summary["trainable_parameters"] == 0 and not summary["training_performed"], "추가 학습 감지")
    require(summary["provenance"]["weights_verified"] and summary["provenance"]["config_verified"], "공식 가중치 검증 실패")
    keys = ["panel", "origin", "oper_part_no", "pattern"]
    require(cohort.unique(subset=keys).height == cohort.height, "중복 표본")
    require(raw.height == cohort.height * 27 * 2, "예측 행수 불일치")
    require(raw.unique(subset=keys + ["model", "horizon_step"]).height == raw.height, "중복 예측")
    require(set(raw["model"]) == {"chronos2", "naive"}, "모델 목록 불일치")
    require(raw.filter(~pl.col("actual").is_finite() | ~pl.col("prediction").is_finite()
                       | pl.col("actual").is_null() | pl.col("prediction").is_null()).height == 0, "비유한 평가값")
    require(raw.select(keys).unique().sort(keys).equals(cohort.select(keys).sort(keys)), "평가 표본과 manifest 불일치")
    group = raw.group_by(keys + ["model"]).agg(pl.len().alias("n"),
        pl.col("horizon_step").n_unique().alias("steps"), pl.col("horizon_step").min().alias("min"),
        pl.col("horizon_step").max().alias("max"))
    require(group.filter((pl.col("n") != 27) | (pl.col("steps") != 27) | (pl.col("min") != 0) | (pl.col("max") != 26)).height == 0,
            "일부 표본의 27주 범위 불일치")
    identity = raw.group_by(keys + ["horizon_step"]).agg(pl.col("actual").n_unique().alias("actuals"),
                                     pl.col("model").n_unique().alias("models"))
    require(identity.filter((pl.col("actuals") != 1) | (pl.col("models") != 2)).height == 0, "두 모델의 정답 불일치")
    for selected in manifest["selection_audit"]:
        rows = cohort.filter((pl.col("panel") == selected["panel"]) & (pl.col("origin") == selected["origin"]))
        panel = next(p for p in manifest["panels"] if p["name"] == selected["panel"])
        for pattern in panel["patterns"]:
            require(rows.filter(pl.col("pattern") == pattern).height == selected["actual_per_pattern"], "유형 균형 위반")
    # 원천에서 같은 ID/주차를 직접 대조. 모델 입력의 float32 변환만 동일하게 적용.
    source = pl.scan_parquet(summary["source"]).filter(pl.col("oper_part_no").is_in(cohort["oper_part_no"].unique().to_list())).collect()
    source_keys = source.select("oper_part_no", "demand_dt")
    require(source_keys.unique().height == source.height, "원천 선택 부품의 중복 행")
    lookup = {(part, week): value for part, week, value in source.select("oper_part_no", "demand_dt", "demand_qty").iter_rows()}
    for item in summary["inputs"]:
        panel, origin = item["panel"], item["origin"]
        rows = cohort.filter((pl.col("panel") == panel) & (pl.col("origin") == origin)).to_dicts()
        monday = date.fromisocalendar(origin // 100, origin % 100, 1)
        weeks = []
        for step in range(-52, 27):
            iso = (monday + timedelta(weeks=step)).isocalendar()
            weeks.append(iso.year * 100 + iso.week)
        expected = np.asarray([[lookup[(row["oper_part_no"], week)] for week in weeks] for row in rows], dtype=float)
        with np.load(output / f"arrays_{panel}_{origin}.npz") as arrays:
            np.testing.assert_array_equal(arrays["history"], expected[:, :52].astype(np.float32))
            np.testing.assert_array_equal(arrays["actual"], expected[:, 52:])
            require(hashlib.sha256(arrays["history"].tobytes()).hexdigest() == item["input_sha256"], "입력 배열 해시 불일치")
            np.testing.assert_array_equal(arrays["naive"], np.repeat(arrays["history"][:, -1:], 27, axis=1))
            for i, row in enumerate(rows):
                positive = expected[i, :52][expected[i, :52] > 0]
                adi = 52 / len(positive)
                cv2 = np.var(positive, ddof=1) / np.mean(positive) ** 2
                pattern = ("smooth" if cv2 < 0.49 else "erratic") if adi < 1.32 else ("intermittent" if cv2 < 0.49 else "lumpy")
                require(row["pattern"] == pattern, "독립 계산 수요 유형 불일치")
                close(row["adi"], adi, "ADI 불일치")
                close(row["cv2"], cv2, "CV² 불일치")
                for model in ("chronos2", "naive"):
                    record = raw.filter((pl.col("panel") == panel) & (pl.col("origin") == origin)
                                        & (pl.col("oper_part_no") == row["oper_part_no"]) & (pl.col("model") == model)).sort("horizon_step")
                    np.testing.assert_array_equal(record["forecast_week"].to_numpy(), weeks[52:])
                    np.testing.assert_allclose(record["prediction"].to_numpy(), arrays[model][i], rtol=0, atol=1e-12)
                    np.testing.assert_array_equal(record["actual"].to_numpy(), expected[i, 52:])
    # 평가 모듈을 import하지 않고 CSV에서 오차를 새로 계산.
    raw = raw.with_columns((pl.col("prediction") - pl.col("actual")).alias("error"))
    def recalculate(columns):
        return raw.group_by(columns).agg(pl.len().alias("points"), pl.col("actual").sum().alias("actual_sum"),
            pl.col("error").abs().sum().alias("abs_error"), pl.col("error").pow(2).sum().alias("squared_error"),
            pl.col("error").sum().alias("signed_error"), pl.col("error").abs().mean().alias("mae"),
            (pl.col("actual") == 0).sum().alias("zero_actual_points"),
            (pl.col("prediction") < 0).sum().alias("negative_prediction_points"),
            pl.col("error").pow(2).mean().sqrt().alias("rmse"))
    def compare_table(actual, expected, columns):
        actual_map = {tuple(row[key] for key in columns): row for row in actual.to_dicts()}
        require(len(actual_map) == expected.height, "재집계 그룹 수 불일치")
        for row in expected.to_dicts():
            other = actual_map[tuple(row[key] for key in columns)]
            for metric in ("points", "actual_sum", "abs_error", "squared_error", "signed_error", "mae",
                           "zero_actual_points", "negative_prediction_points"):
                close(other[metric], row[metric], f"독립 재집계 {metric} 불일치")
            if "rmse" in row:
                close(other["rmse"], row["rmse"], "RMSE 불일치")
            if "wape_pct" in row:
                wape = 100 * other["abs_error"] / other["actual_sum"] if other["actual_sum"] else None
                close(wape, row["wape_pct"], "WAPE 불일치")
    compare_table(recalculate(keys + ["model"]), pl.read_csv(output / "pair_metrics.csv"), keys + ["model"])
    for suffix, columns in (("pattern", ["panel", "pattern", "model"]), ("origin", ["panel", "origin", "model"]),
                             ("origin_pattern", ["panel", "origin", "pattern", "model"])):
        compare_table(recalculate(columns), pl.read_csv(output / f"metrics_by_{suffix}.csv"), columns)
    pooled = recalculate(["panel", "model"])
    diagnostics = {}
    for panel, result in summary["panels"].items():
        expected = pl.DataFrame(result["pooled"]).with_columns(pl.lit(panel).alias("panel"))
        compare_table(pooled.filter(pl.col("panel") == panel), expected, ["panel", "model"])
        table = pl.read_csv(output / "pair_metrics.csv").filter(pl.col("panel") == panel)
        chrono = table.filter(pl.col("model") == "chronos2").sort("origin", "oper_part_no")
        naive = table.filter(pl.col("model") == "naive").sort("origin", "oper_part_no")
        delta = chrono["mae"].to_numpy() - naive["mae"].to_numpy()
        tie = np.isclose(chrono["mae"].to_numpy(), naive["mae"].to_numpy(), rtol=1e-7, atol=1e-8)
        require(int(((delta < 0) & ~tie).sum()) == result["paired_mae"]["chronos_wins"], "승수 불일치")
        require(int(tie.sum()) == result["paired_mae"]["ties"], "동률 불일치")
        buckets = pl.read_csv(output / "metrics_by_origin_pattern.csv").filter(pl.col("panel") == panel)
        for model in ("chronos2", "naive"):
            values = buckets.filter(pl.col("model") == model)["wape_pct"].drop_nulls()
            close(values.mean(), result["macro"][model]["equal_origin_pattern_wape_pct"], "macro WAPE 불일치")
            require(len(values) == result["macro"][model]["valid_buckets"], "정의 가능한 셀 수 불일치")
        diagnostics[panel] = {"total_pairs": chrono.height,
            "all_zero_future_pairs": chrono.filter(pl.col("actual_sum") == 0).height,
            "future_zero_fraction": float(chrono["zero_actual_points"].sum() / chrono["points"].sum()),
            "negative_chronos_fraction": float(chrono["negative_prediction_points"].sum() / chrono["points"].sum()),
            "by_future_actual_regime": []}
        # 아래 구분은 사후 원인 점검 전용이며 표본 선정이나 주 평가 지표에는 사용하지 않음.
        for label, mask in (("all_zero", chrono["actual_sum"].to_numpy() == 0),
                            ("positive_total", chrono["actual_sum"].to_numpy() > 0)):
            diagnostics[panel]["by_future_actual_regime"].append({"regime": label, "pairs": int(mask.sum()),
                "chronos_wins": int(((delta < 0) & ~tie & mask).sum()), "naive_wins": int(((delta > 0) & ~tie & mask).sum()),
                "chronos_mae": float(chrono["mae"].to_numpy()[mask].mean()) if mask.any() else None,
                "naive_mae": float(naive["mae"].to_numpy()[mask].mean()) if mask.any() else None})
    result = {"status": "passed", "cohort_pairs": cohort.height, "forecast_rows": raw.height,
        "source_sha256": summary["source_sha256"], "forecast_csv_sha256": summary["forecast_csv_sha256"],
        "checks": ["frozen_source_and_manifests", "balanced_patterns_at_each_origin", "identical_h27_targets",
                   "independent_iso_calendar", "source_to_input_arrays", "independent_adi_cv2",
                   "last_observation_naive", "raw_csv_to_arrays", "all_pair_and_group_metrics", "pooled_and_macro", "paired_wins"],
        "diagnostics": diagnostics}
    (output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.output_dir.expanduser().resolve()), ensure_ascii=False, indent=2))

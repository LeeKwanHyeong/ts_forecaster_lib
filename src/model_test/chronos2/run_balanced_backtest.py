"""수요 유형 균등 표본과 여러 예측 시작 주차에서 Chronos2/Naive H27 비교.

실행 예: python -m model_test.chronos2.run_balanced_backtest --help
모든 표본을 미래 정답 조회 전에 저장하고, 공식 고정 가중치를 오프라인에서 사용.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import time

import numpy as np
import polars as pl
import torch

from modeling_module.models.Chronos2 import Chronos2Config, Chronos2Model
from .backtest_metrics import aggregate, pair_metrics, panel_summary
from .balanced_sampling import (ADI_THRESHOLD, CONTEXT, CV2_THRESHOLD, DEFAULT_PANELS, HORIZON,
    choose_cohort, last_value_naive, load_selected_window, profile_history, source_scan, week_sequence)


LOGGER = logging.getLogger(__name__)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def raw_forecast_rows(cohort, actual, predictions):
    for model, prediction in predictions.items():
        for i, row in enumerate(cohort):
            for step, week in enumerate(week_sequence(row["origin"], 0, HORIZON)):
                yield {"panel": row["panel"], "origin": row["origin"], "pattern": row["pattern"],
                       "oper_part_no": row["oper_part_no"], "model": model,
                       "horizon_step": step, "forecast_week": week,
                       "actual": float(actual[i, step]), "prediction": float(prediction[i, step])}


def render_report(summary: dict) -> str:
    lines = ["# Chronos2 수요 유형 균등 H27 백테스트", "", "## 평가 조건", "",
        "- 입력: 예측 시작 주차 직전 52주 수요, 평가: 시작 주차 포함 27주",
        "- Chronos2: 공식 고정 가중치, 추가 학습 없음, P50 예측, 외생변수 없음, 음수 예측 보정 없음",
        "- Naive: 같은 입력의 마지막 관측값을 27주 반복",
        "- 유형: ADI=52/양의 수요 주수, CV²=양의 수요 표본분산/평균²(ddof=1)",
        "- 경계: ADI 1.32, CV² 0.49. 무수요·양의 수요 1회·잘못된 이력은 별도 제외",
        "- 빈도·변동성 4유형 기준이며 추세·계절성까지 균등화한 실험은 아님",
        "- 각 시점의 유형별 후보에서 비복원 무작위 추출. 미래 정답이나 모델 오차에 따른 표본 교체 없음",
        "- pooled WAPE는 수요량 가중, macro WAPE는 시작 주차×유형 셀의 WAPE를 동일 가중 평균", ""]
    for name, result in summary["panels"].items():
        lines += [f"## {name}", "", "| 모델 | 부품·시점 쌍 | 고유 부품 | MAE | RMSE | pooled WAPE | macro WAPE |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in result["pooled"]:
            macro = result["macro"][row["model"]]["equal_origin_pattern_wape_pct"]
            wape = row["wape_pct"]
            lines.append(f"| {row['model']} | {row['pairs']} | {row['unique_parts']} | {row['mae']:.4f} | {row['rmse']:.4f} | {wape:.4f}% | {macro:.4f}% |"
                         if wape is not None and macro is not None else f"| {row['model']} | {row['pairs']} | {row['unique_parts']} | {row['mae']:.4f} | {row['rmse']:.4f} | 정의 불가 | 정의 불가 |")
        wins = result["paired_mae"]
        lines += ["", f"- 쌍별 MAE: Chronos2 {wins['chronos_wins']}승 / Naive {wins['naive_wins']}승 / 동률 {wins['ties']}건",
                  f"- 부품 단위 bootstrap 95% 구간(Chronos2−Naive pooled WAPE, %p): {result['bootstrap']['delta_pooled_wape_pp_95pct']}", ""]
    lines += ["## 시점별 표본 수", "", "| 패널 | 시작 주차 | 유형당 표본 | 상태 |", "|---|---:|---:|---|"]
    for row in summary["selection_audit"]:
        lines.append(f"| {row['panel']} | {row['origin']} | {row['actual_per_pattern']} | {row['status']} |")
    lines += ["", "## 해석 범위", "",
        "- 장기 2유형 패널은 27주 간격으로 평가 실적이 겹치지 않도록 구성",
        "- 최근 4유형 패널은 4주 간격으로 평가 실적이 겹치며, 두 패널 사이에도 중복 가능. 패널 간 합산 금지",
        "- bootstrap은 부품별 반복 관측을 함께 재표집하나 주차 공통 충격과 겹치는 평가 기간의 의존성을 모두 해소하지 못함",
        "- 52주 연속 이력을 가진 부품에 한정한 비교. 신규·이력 부족·완전 무수요 부품으로 일반화 불가",
        "- 로컬 스냅샷의 생산/합성 출처는 미확인. 공식 사전학습 데이터와의 중복 여부도 미확인",
        "- 유형별 후보가 적은 시점은 표본 수를 동일하게 축소. 특정 유형이 없으면 해당 시점 전체 제외",
        "- 원천 데이터 변경 없음, DB 접근·쓰기 없음", "",
        "## 산출물", "", "- cohort_manifest.csv: 미래 정답 조회 전에 확정한 표본",
        "- candidate_profile_*.csv / selection_manifest.json: 후보 수·품질 제외·선정 설정",
        "- forecast_27_weeks.csv: 두 모델의 동일 실적 및 보정 없는 예측",
        "- pair_metrics.csv / metrics_by_*.csv: 표본별·유형별·시점별 오차",
        "- arrays_*.npz: 입력·정답·두 모델 예측",
        "- summary.json: 파일 해시·실행 설정·가중치 검증·구간 추정", ""]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict:
    source = args.source_parquet.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"기존 결과 덮어쓰기 금지: {output}")
    if args.threads < 1 or not 1 <= args.per_pattern <= 200 or args.bootstrap_repetitions < 2:
        raise ValueError("threads>=1, per-pattern 1~200, bootstrap-repetitions>=2 필요")
    source_digest = file_sha256(source)
    output.mkdir(parents=True, exist_ok=False)
    try:
        selection, audits = [], []
        profiles = {}
        # 모든 시점의 표본 확정이 끝나기 전에는 미래 정답이나 모델 결과를 조회하지 않음.
        for panel in DEFAULT_PANELS:
            for origin in panel.origins:
                if origin not in profiles:
                    profiles[origin] = profile_history(source, origin)
                    profiles[origin].write_csv(output / f"candidate_profile_{origin}.csv")
                chosen, audit = choose_cohort(profiles[origin], panel, origin, args.per_pattern, args.seed)
                selection.extend(chosen)
                audits.append(audit)
                LOGGER.info("선정 %s %s: 유형당 %s / 후보 %s", panel.name, origin,
                            audit["actual_per_pattern"], audit["candidate_counts"])
        if not selection:
            raise ValueError("선정된 표본이 없습니다.")
        cohort = pl.DataFrame(selection)
        cohort.write_csv(output / "cohort_manifest.csv")
        policy = {"context_length": CONTEXT, "horizon": HORIZON, "seed": args.seed,
                  "per_pattern_target": args.per_pattern, "adi_threshold": ADI_THRESHOLD,
                  "cv2_threshold": CV2_THRESHOLD, "positive_variance_ddof": 1,
                  "selection_uses_future_actuals": False, "source_sha256": source_digest,
                  "panels": [asdict(panel) for panel in DEFAULT_PANELS], "selection_audit": audits}
        write_json(output / "selection_manifest.json", policy)
        torch.set_num_threads(args.threads)
        cfg = Chronos2Config(model_path=str(args.model_path.expanduser().resolve()),
                            local_files_only=True, context_length=CONTEXT, horizon=HORIZON,
                            batch_size=args.batch_size, device=args.device)
        model = Chronos2Model(cfg)
        start = time.perf_counter()
        model.load()
        load_seconds = time.perf_counter() - start
        pairs, raw, timings, inputs = [], [], [], []
        for panel in DEFAULT_PANELS:
            for origin in panel.origins:
                chosen = [row for row in selection if row["panel"] == panel.name and row["origin"] == origin]
                if not chosen:
                    continue
                ids = [row["oper_part_no"] for row in chosen]
                history = load_selected_window(source, ids, origin, future=False)
                # 동일 입력을 보장하기 위해 두 모델 모두 float32 양자화된 52주 이력 사용.
                x = history.astype(np.float32)
                actual = load_selected_window(source, ids, origin, future=True)
                start = time.perf_counter()
                result = model.predict(x[:, :, None])
                elapsed = time.perf_counter() - start
                predictions = {"chronos2": result.point.numpy()[:, :, 0], "naive": last_value_naive(x)}
                for name, prediction in predictions.items():
                    pairs.extend(pair_metrics(chosen, actual, prediction, name))
                raw.extend(raw_forecast_rows(chosen, actual, predictions))
                np.savez_compressed(output / f"arrays_{panel.name}_{origin}.npz", history=x, actual=actual,
                                    chronos2=predictions["chronos2"], naive=predictions["naive"])
                inputs.append({"panel": panel.name, "origin": origin, "pairs": len(ids),
                               "input_sha256": hashlib.sha256(x.tobytes()).hexdigest()})
                timings.append({"panel": panel.name, "origin": origin, "seconds": elapsed})
                LOGGER.info("예측 완료 %s %s: %s개 × 27주, %.2f초", panel.name, origin, len(ids), elapsed)
        records = pl.DataFrame(pairs)
        records.write_csv(output / "pair_metrics.csv")
        pl.DataFrame(raw).write_csv(output / "forecast_27_weeks.csv")
        for suffix, keys in (("pattern", ["panel", "pattern", "model"]),
                             ("origin", ["panel", "origin", "model"]),
                             ("origin_pattern", ["panel", "origin", "pattern", "model"])):
            pl.DataFrame(aggregate(records, keys)).write_csv(output / f"metrics_by_{suffix}.csv")
        panels = {panel.name: panel_summary(records.filter(pl.col("panel") == panel.name),
                    args.seed, args.bootstrap_repetitions) for panel in DEFAULT_PANELS
                    if records.filter(pl.col("panel") == panel.name).height}
        unchanged = file_sha256(source) == source_digest
        if not unchanged:
            raise RuntimeError("실행 중 원천 데이터 변경 감지. 결과 확정 금지.")
        dataset = source_scan(source).select(pl.len().alias("rows"), pl.col("oper_part_no").n_unique().alias("parts"),
                            pl.col("demand_dt").min().alias("min_week"), pl.col("demand_dt").max().alias("max_week")).collect().row(0, named=True)
        summary = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(source), "source_kind": "local_snapshot_production_provenance_unverified",
            "source_sha256": source_digest, "source_unchanged": unchanged, "dataset": dataset,
            "config": asdict(cfg), "provenance": model.backbone.provenance(),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "cohort_sha256": file_sha256(output / "cohort_manifest.csv"), "selection_audit": audits,
            "panels": panels, "inputs": inputs, "load_seconds": load_seconds, "inference_timings": timings,
            "forecast_csv_sha256": file_sha256(output / "forecast_27_weeks.csv"),
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "polars", "transformers", "chronos-forecasting")},
            "future_actuals_used_as_input": False, "predictions_clipped": False,
            "cross_learning": False, "training_performed": False,
            "macro_weighting": "equal weight per origin x pattern cell; undefined denominators counted",
        }
        write_json(output / "summary.json", summary)
        (output / "report.md").write_text(render_report(summary), encoding="utf-8")
        return summary
    except Exception as exc:
        # 중간 실패를 성공으로 오인하지 않도록 원래 오류를 다시 올리며 상태를 별도 보존.
        write_json(output / "failure.json", {"status": "failed", "error_type": type(exc).__name__, "message": str(exc)})
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-pattern", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run(build_parser().parse_args())
    print(json.dumps(report["panels"], ensure_ascii=False, indent=2, allow_nan=False))

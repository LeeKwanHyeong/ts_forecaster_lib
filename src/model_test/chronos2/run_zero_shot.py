"""27주 예측을 수행하고 원 예측·분위수·입력 분리 증적을 저장하는 실행기.

기본 입력은 공개가 가능한 합성 시계열이며 외부 서비스와 DB 쓰기는 없다.
로컬 parquet를 명시하면 학습 시점 이전 이력만으로 최대 16개 부품 선택.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, timedelta
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import polars as pl
import torch

from modeling_module.models.Chronos2 import Chronos2Config, Chronos2Model


def week_id(day: date) -> int:
    year, week, _ = day.isocalendar()
    return year * 100 + week


def forecast_weeks(plan_week: int, horizon: int = 27) -> list[int]:
    start = date.fromisocalendar(plan_week // 100, plan_week % 100, 1)
    return [week_id(start + timedelta(weeks=step)) for step in range(horizon)]


def synthetic_input(context_length: int):
    """추세·계절·간헐 패턴 생성. 미래 정답은 평가에만 사용하는 별도 배열."""
    count = context_length + 27
    t = np.arange(count, dtype=np.float32)
    demand = np.stack([
        20 + 8 * np.sin(2 * np.pi * t / 52) + 2 * np.cos(2 * np.pi * t / 13),
        5 + 0.3 * t + 2 * np.sin(2 * np.pi * t / 13),
        np.where(t % 7 == 0, 12 + t % 5, 0),
    ]).astype("float32")
    plan = week_id(date.fromisocalendar(2025, 1, 1) + timedelta(weeks=context_length))
    return ["synthetic_seasonal", "synthetic_trend", "synthetic_intermittent"], demand[:, :context_length, None], demand[:, context_length:], plan


def parquet_input(source: Path, plan_week: int, context_length: int, max_parts: int):
    """미래 실적을 보지 않고 연속 이력을 가진 부품을 선택하는 제한 조회."""
    if not 1 <= max_parts <= 16:
        raise ValueError("로컬 점검은 max_parts 1~16개만 지원합니다.")
    origin = date.fromisocalendar(plan_week // 100, plan_week % 100, 1)
    past_weeks = [week_id(origin - timedelta(weeks=i)) for i in reversed(range(1, context_length + 1))]
    future_weeks = forecast_weeks(plan_week)
    scan = pl.scan_parquet(source).select("oper_part_no", "demand_dt", "demand_qty").filter(pl.col("oper_part_no").is_not_null())
    history = scan.filter(pl.col("demand_dt").is_in(past_weeks))
    candidates = history.group_by("oper_part_no").agg(
        pl.len().alias("rows"), pl.col("demand_dt").n_unique().alias("weeks"),
        pl.col("demand_qty").is_finite().all().alias("finite"),
        pl.col("demand_qty").is_null().any().alias("nulls"),
        (pl.col("demand_qty") > 0).any().alias("has_demand"),
    ).filter((pl.col("rows") == context_length) & (pl.col("weeks") == context_length)
             & pl.col("finite") & ~pl.col("nulls") & pl.col("has_demand"))
    ids = candidates.sort("oper_part_no").head(max_parts).collect()["oper_part_no"].to_list()
    if not ids:
        raise ValueError("연속 과거 관측과 양의 수요를 가진 부품이 없습니다.")
    selected = scan.filter(pl.col("oper_part_no").is_in(ids) & pl.col("demand_dt").is_in(past_weeks + future_weeks)).collect()
    if selected.select(pl.struct("oper_part_no", "demand_dt").n_unique()).item() != selected.height:
        raise ValueError("선택한 부품의 주차별 중복 행이 있습니다.")
    if selected.filter(pl.col("demand_qty").is_not_null() & (~pl.col("demand_qty").is_finite() | (pl.col("demand_qty") < 0))).height:
        raise ValueError("선택한 수요 수량에 비유한 값 또는 음수가 있습니다.")
    contexts, actuals = [], []
    for part in ids:
        values = {row["demand_dt"]: row["demand_qty"] for row in selected.filter(pl.col("oper_part_no") == part).to_dicts()}
        contexts.append([values[week] for week in past_weeks])
        # 미관측 미래 실적은 NaN으로 보존하며 예측 입력에 전달하지 않음.
        actuals.append([np.nan if values.get(week) is None else values[week] for week in future_weeks])
    return list(map(str, ids)), np.asarray(contexts, dtype="float32")[:, :, None], np.asarray(actuals, dtype="float32"), plan_week


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    known = np.isfinite(actual)
    if not known.any():
        return {"observed_points": 0, "mae": None, "wape_pct": None}
    error = np.abs(actual[known] - prediction[known])
    denominator = np.abs(actual[known]).sum()
    return {"observed_points": int(known.sum()), "mae": float(error.mean()),
            "wape_pct": float(100 * error.sum() / denominator) if denominator > 0 else None}


def run(args: argparse.Namespace) -> dict:
    for name in ("model_path", "cache_dir", "source_parquet", "output_dir"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.threads < 1:
        raise ValueError("threads는 양의 정수여야 합니다.")
    if args.output_dir.exists():
        raise FileExistsError(f"기존 실행 결과는 덮어쓰지 않습니다: {args.output_dir}")
    torch.set_num_threads(args.threads)
    if args.source_parquet is None:
        ids, x, truth, plan_week = synthetic_input(args.context_length)
        source_kind = "synthetic_patterns_not_customer_data"
    else:
        if args.plan_week is None:
            raise ValueError("로컬 parquet 사용 시 --plan-week가 필요합니다.")
        ids, x, truth, plan_week = parquet_input(args.source_parquet, args.plan_week, args.context_length, args.max_parts)
        source_kind = "local_snapshot_production_provenance_unverified"
    cfg = Chronos2Config(model_path=str(args.model_path.resolve()) if args.model_path else None,
                        cache_dir=str(args.cache_dir.resolve()) if args.cache_dir else None,
                        local_files_only=not args.allow_download, device=args.device,
                        context_length=args.context_length, horizon=27)
    model = Chronos2Model(cfg)
    start = time.perf_counter()
    model.load()
    load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    forecast = model.predict(x)
    inference_seconds = time.perf_counter() - start
    features = model.encode_future_tokens(x)
    variants = {"endogenous": forecast}
    if args.with_calendar:
        origin = date.fromisocalendar(plan_week // 100, plan_week % 100, 1)
        # 알려진 달력만으로 생성하며 미래 수요를 외생변수로 사용하지 않음.
        elapsed = np.asarray([(origin + timedelta(weeks=i) - date(2000, 1, 3)).days
                              for i in range(-x.shape[1], 27)], dtype=np.float32)
        values = np.sin(2 * np.pi * elapsed / 365.2425)
        past = np.repeat(values[None, :x.shape[1]], len(ids), axis=0)
        future = np.repeat(values[None, x.shape[1]:], len(ids), axis=0)
        variants["calendar_covariate"] = model.predict(x, past_covariates={"sin_annual": past}, future_covariates={"sin_annual": future})
    rows = []
    weeks = forecast_weeks(plan_week)
    scores = {}
    for name, result in variants.items():
        scores[name] = metrics(truth, result.point.numpy()[:, :, 0])
        for index, part in enumerate(ids):
            for step, week in enumerate(weeks):
                rows.append({"variant": name, "oper_part_no": part, "plan_week": plan_week,
                    "forecast_week": week, "horizon_step": step, "prediction": float(result.point[index, step, 0]),
                    "p10": float(result.quantiles[index, step, 0]), "p50": float(result.quantiles[index, step, 1]),
                    "p90": float(result.quantiles[index, step, 2]),
                    "actual": float(truth[index, step]) if np.isfinite(truth[index, step]) else None})
    naive = np.repeat(x[:, -1, 0:1], 27, axis=1)
    summary = {
        "config": asdict(cfg), "provenance": forecast.metadata, "source_kind": source_kind,
        "source": str(args.source_parquet.resolve()) if args.source_parquet else None,
        "series_ids": ids, "plan_week": plan_week, "forecast_weeks": weeks,
        "point_shape": list(forecast.point.shape), "quantile_shape": list(forecast.quantiles.shape),
        "future_token_shape": list(features.future_tokens.shape),
        "input_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "load_seconds": load_seconds, "endogenous_inference_seconds": inference_seconds,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "metrics": scores, "last_value_baseline": metrics(truth, naive),
        "future_actuals_used_as_model_input": False,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "polars", "transformers", "chronos-forecasting")},
        "caveat": "실행 검증과 제한된 시점의 지표이며 고객 운영 성능 개선 또는 사전학습 데이터와의 비중복을 보장하지 않는 범위",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pl.DataFrame(rows).write_csv(args.output_dir / "forecast_27_weeks.csv")
    np.savez_compressed(args.output_dir / "arrays.npz", context=x, actual=truth,
                        prediction=forecast.point.numpy(), quantiles=forecast.quantiles.numpy())
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-parquet", type=Path)
    parser.add_argument("--plan-week", type=int)
    parser.add_argument("--max-parts", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=52)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--with-calendar", action="store_true")
    return parser


if __name__ == "__main__":
    result = run(build_parser().parse_args())
    print(json.dumps({key: result[key] for key in ("source_kind", "point_shape", "metrics", "endogenous_inference_seconds")}, ensure_ascii=False, indent=2))

"""수정 전후 실험이 공유하는 데이터 계약과 원 단위 평가 지표."""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np


SETTINGS = {
    "lookback": 52, "horizon": 27, "history_weeks": 260,
    "window_step": 13, "batch_size": 64, "parts": 64,
    "pretrain_epochs": 50, "warmup_epochs": 25, "spike_epochs": 5,
    "lr": 1e-4, "spike_lr": 5e-5, "weight_decay": 1e-4,
    "mask_ratio": 0.3, "patch_len": 8, "patch_stride": 4,
    "d_model": 64, "d_ff": 128, "n_layers": 2, "n_heads": 4,
    "dropout": 0.1, "norm": "LayerNorm", "device": "cpu",
    "seeds": [42, 142, 242], "sensitivity_lr": 1e-3,
    "future_features": ["sin_annual", "cos_annual", "sin_semi", "cos_semi"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def week_date(week: int) -> date:
    return date.fromisocalendar(int(week) // 100, int(week) % 100, 1)


def week_id(day: date) -> int:
    year, week, _ = day.isocalendar()
    return year * 100 + week


def calendar_features(weeks: np.ndarray) -> np.ndarray:
    # 고정 기준일의 날짜 차이만 사용. 수요, 전체 데이터 길이, 실행 주차와 무관.
    elapsed = np.array([(week_date(int(w)) - date(2000, 1, 3)).days for w in weeks])
    phase = 2 * np.pi * elapsed / 365.2425
    return np.stack([np.sin(phase), np.cos(phase), np.sin(2 * phase), np.cos(2 * phase)], axis=-1).astype("float32")


def scores(actual, prediction) -> dict:
    y, p = np.asarray(actual, dtype="float64"), np.asarray(prediction, dtype="float64")
    if y.size == 0 or y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("평가값은 같은 shape의 유한한 수여야 함")
    error = p - y
    denominator = float(np.abs(y).sum())
    return {
        "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
        "wape_pct": float(100 * np.abs(error).sum() / denominator) if denominator else None,
        "bias_pct": float(100 * error.sum() / denominator) if denominator else None,
        "negative_prediction_pct": float(100 * (p < 0).mean()), "n": int(y.size),
    }


def prepare(source: Path, output: Path, exogenous: Path | None = None) -> dict:
    """미래 실적을 보지 않고 표본을 정하고, 세 실행 구간을 한 번만 생성."""
    import polars as pl

    output.mkdir(parents=True, exist_ok=False)
    raw = pl.read_parquet(source, columns=["oper_part_no", "demand_dt", "demand_qty"])
    if raw.select(pl.struct("oper_part_no", "demand_dt").n_unique()).item() != raw.height:
        raise ValueError("부품-주차 중복")
    qty = raw["demand_qty"].to_numpy()
    if not np.isfinite(qty).all() or (qty < 0).any():
        raise ValueError("수요에 결측, 음수 또는 비유한 값 존재")
    s = SETTINGS
    end = week_date(raw["demand_dt"].max())
    weeks = np.array([week_id(end - timedelta(weeks=i)) for i in reversed(range(s["history_weeks"]))])
    train_end = len(weeks) - 2 * s["horizon"]
    candidates, excluded = [], 0
    tail = raw.filter(pl.col("demand_dt") >= int(weeks[0])).sort("oper_part_no", "demand_dt")
    for frame in tail.partition_by("oper_part_no", maintain_order=True):
        if not np.array_equal(frame["demand_dt"].to_numpy(), weeks):
            excluded += 1
            continue
        values = frame["demand_qty"].to_numpy().astype("float32")
        training = values[:train_end]
        candidates.append({"part": frame["oper_part_no"][0], "values": values,
                           "mean": float(training.mean()), "zero": float((training == 0).mean())})
    if len(candidates) < s["parts"]:
        raise ValueError("연속 이력을 갖춘 부품 수 부족")

    # 수요 규모 4구간 x 0 수요 비율 2구간. 구간 정의와 추출 모두 학습 이력만 사용.
    candidates.sort(key=lambda row: (row["mean"], row["part"]))
    rng = np.random.default_rng(20261115)
    selected = []
    for level, indices in enumerate(np.array_split(np.arange(len(candidates)), 4)):
        ranked = sorted(indices, key=lambda i: (candidates[int(i)]["zero"], candidates[int(i)]["part"]))
        for zero_bin, bucket in enumerate(np.array_split(ranked, 2)):
            for index in rng.choice(bucket, size=8, replace=False):
                row = dict(candidates[int(index)])
                row["stratum"] = f"level{level + 1}_zero{zero_bin + 1}"
                selected.append(row)
    selected.sort(key=lambda row: row["part"])
    values = np.stack([row["values"] for row in selected])
    parts = np.array([row["part"] for row in selected])
    origins = np.arange(s["lookback"], train_end - s["horizon"] + 1, s["window_step"])
    if origins[-1] != train_end - s["horizon"]:
        origins = np.append(origins, train_end - s["horizon"])
    features = calendar_features(weeks)
    arrays = {"parts": parts, "weeks": weeks, "values": values, "features": features}
    for split, points in {"train": origins, "val": [train_end], "test": [train_end + s["horizon"]]}.items():
        xs, ys, fs, ps = [], [], [], []
        for part_index in range(len(parts)):
            for origin in points:
                xs.append(values[part_index, origin - s["lookback"]:origin, None])
                ys.append(values[part_index, origin:origin + s["horizon"], None])
                fs.append(features[origin:origin + s["horizon"]])
                ps.append(part_index)
        arrays.update({f"{split}_x": np.stack(xs), f"{split}_y": np.stack(ys),
                       f"{split}_f": np.stack(fs), f"{split}_part": np.array(ps)})
    # 학습 정답, 검증 정답, 최종 테스트 정답의 시간 구간을 서로 분리.
    assert int(origins.max() + s["horizon"]) <= train_end
    assert train_end + 2 * s["horizon"] == len(weeks)
    np.savez_compressed(output / "dataset.npz", **arrays)
    cohort = [{"oper_part_no": r["part"], "train_mean": r["mean"], "train_zero_ratio": r["zero"], "stratum": r["stratum"]} for r in selected]
    pl.DataFrame(cohort).write_csv(output / "cohort.csv")
    selected_rows = tail.filter(pl.col("oper_part_no").is_in(parts.tolist()))
    selected_rows.write_parquet(output / "cohort.parquet")
    manifest = {"settings": s, "source": str(source.resolve()), "source_sha256": sha256(source),
                "source_rows": raw.height, "source_parts": raw["oper_part_no"].n_unique(),
                "eligible_parts": len(candidates), "incomplete_history_excluded": excluded,
                "data_provenance": "existing local snapshot; production/synthetic provenance not verified",
                "selected_parts": parts.tolist(), "dataset_sha256": sha256(output / "dataset.npz"),
                "splits": {"train_end": int(weeks[train_end - 1]), "val_start": int(weeks[train_end]),
                           "val_end": int(weeks[train_end + s["horizon"] - 1]), "test_start": int(weeks[train_end + s["horizon"]]), "test_end": int(weeks[-1])},
                "sample_counts": {split: len(arrays[f"{split}_x"]) for split in ("train", "val", "test")}}
    if exogenous is not None:
        exo_parts = pl.scan_parquet(exogenous).select("oper_part_no").unique().collect()["oper_part_no"]
        manifest["existing_exogenous"] = {"path": str(exogenous.resolve()), "sha256": sha256(exogenous),
                                          "matching_part_count": len(set(raw["oper_part_no"]) & set(exo_parts)), "used": False}
    save_json(output / "manifest.json", manifest)
    return manifest

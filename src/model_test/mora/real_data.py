"""기존 Chronos2 분할을 복원하고 수요량과 무관하게 L52/H27 창을 고정하는 도구."""

from collections import Counter
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from modeling_module.models.Chronos2.training.contracts import (
    Chronos2DataContract, Chronos2SplitContract, ForecastPartition, SeriesKey, manifest_sha256,
)


EPOCH = date(1970, 1, 5)  # ISO 주차를 연속 정수로 변환할 기준 월요일.
SPLITS = ("train", "validation", "test")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def week_step(week):
    value = int(week)
    return (date.fromisocalendar(value // 100, value % 100, 1) - EPOCH).days // 7


def step_week(step):
    year, week, _ = (EPOCH + timedelta(weeks=int(step))).isocalendar()
    return year * 100 + week


def restore_contract(path):
    """기존 배정 결과를 검증해 복원. 재분할이나 신규 무작위 배정은 수행하지 않음."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest_sha256(payload) != manifest.get("manifest_sha256"):
        raise ValueError("기존 split manifest의 SHA-256이 일치하지 않습니다.")
    source = Chronos2DataContract(**manifest["source"])
    source.assert_semantics_confirmed()
    values = dict(manifest["split"])
    for name in SPLITS:
        part = values[name]
        values[name] = ForecastPartition(tuple(SeriesKey(tuple(key["values"])) for key in part["series"]),
                                        part["target_start"], part["target_end"])
    split = Chronos2SplitContract(**values)
    if source.series_columns != ("oper_part_no",) or source.zero_semantics != "observed_zero":
        raise ValueError("이 비교는 부품 번호 단일 키와 실제 관측된 0수요 계약만 지원합니다.")
    if (split.context_length, split.horizon) != (52, 27):
        raise ValueError("기존 L52/H27 조건과 다릅니다.")
    return manifest, source, split


def identity_calendar(source_path):
    """창 선택 단계에서는 수요량 칼럼을 아예 읽지 않음."""
    frame = pl.read_parquet(source_path, columns=["oper_part_no", "demand_dt"])
    if any(frame.null_count().row(0)):
        raise ValueError("부품 번호 또는 주차에 결측이 있습니다.")
    if frame.schema != pl.Schema({"oper_part_no": pl.String, "demand_dt": pl.Int64}):
        raise TypeError("원천 식별자·주차 스키마가 기존 계약과 다릅니다.")
    if frame.select(pl.struct("oper_part_no", "demand_dt").is_duplicated().any()).item():
        raise ValueError("부품·주차 중복이 있습니다. 임의 집계하지 않습니다.")
    if frame.select(pl.col("oper_part_no").str.strip_chars().eq("").any()).item():
        raise ValueError("부품 번호가 비어 있습니다.")
    dates = {week: week_step(week) for week in frame["demand_dt"].unique().to_list()}
    return frame.with_columns(pl.col("demand_dt").replace_strict(dates).alias("step")).sort("oper_part_no", "step")


def plan_windows(index, split, train_windows_per_part=4):
    """ID·달력만으로 적합 창을 선택. 미래 수요량과 예측 오차로 표본을 고르지 않음."""
    if type(train_windows_per_part) is not int or train_windows_per_part < 1:
        raise ValueError("부품당 학습 창 수는 양의 정수여야 합니다.")
    by_part = {key[0]: group["step"].to_numpy() for key, group in index.partition_by("oper_part_no", as_dict=True).items()}
    assigned = {key.values[0] for name in SPLITS for key in getattr(split, name).series}
    if assigned != set(by_part):
        raise ValueError("원천 부품 목록과 기존 분할에 배정된 부품 목록이 다릅니다.")
    length, horizon = split.context_length, split.horizon
    rows, excluded, audit = [], [], {}
    for name in SPLITS:
        part = getattr(split, name)
        first, last = week_step(part.target_start), week_step(part.target_end)
        eligible_count = 0
        selected_counts = []
        for key in sorted(part.series):
            part_id = key.values[0]
            weeks = by_part[part_id]
            positions = np.arange(length, len(weeks) - horizon + 1)
            candidates = [int(weeks[pos]) for pos in positions
                          if first <= weeks[pos] <= last - horizon + 1
                          and weeks[pos + horizon - 1] - weeks[pos - length] == length + horizon - 1]
            eligible_count += len(candidates)
            if not candidates:
                excluded.append({"split": name, "oper_part_no": part_id, "reason": "no_complete_L52_H27_in_allowed_dates"})
                selected_counts.append(0)
                continue
            if name == "train":
                candidates = sorted(candidates, key=lambda origin: manifest_sha256(
                    {"seed": split.seed, "part": part_id, "origin_step": origin, "policy": "id_date_only_v1"}
                ))[:train_windows_per_part]
            selected_counts.append(len(candidates))
            for origin in candidates:
                rows.append({"split": name, "oper_part_no": part_id, "origin": step_week(origin),
                             "origin_step": origin, "context_start": step_week(origin - length),
                             "target_end": step_week(origin + horizon - 1)})
        audit[name] = {"assigned_parts": len(part.series), "eligible_parts": sum(count > 0 for count in selected_counts),
                       "eligible_windows": eligible_count, "selected_windows": sum(selected_counts),
                       "selected_windows_per_part": dict(sorted(Counter(selected_counts).items())),
                       "target_start": part.target_start, "target_end": part.target_end}
    return sorted(rows, key=lambda row: (SPLITS.index(row["split"]), row["origin_step"], row["oper_part_no"])), excluded, audit


def materialize(source_path, rows, length=52, horizon=27):
    """선택이 끝난 창만 추출. 결측·음수·비유한 값을 0으로 대체하지 않음."""
    if not rows:
        raise ValueError("対象窓がありません。")
    # 부품과 기간을 먼저 제한하여 시험 정답 읽기를 적합 이후로 분리.
    ids = sorted({row["oper_part_no"] for row in rows})
    first = min(row["context_start"] for row in rows)
    last = max(row["target_end"] for row in rows)
    frame = pl.scan_parquet(source_path).filter(pl.col("oper_part_no").is_in(ids),
        pl.col("demand_dt").is_between(first, last)).select(
        "oper_part_no", "demand_dt", "demand_qty").collect()
    by_part = {}
    for key, group in frame.partition_by("oper_part_no", as_dict=True).items():
        by_part[key[0]] = dict(zip(group["demand_dt"].to_list(), group["demand_qty"].to_list()))
    history = np.empty((len(rows), length, 1), dtype=np.float32)
    target = np.empty((len(rows), horizon, 1), dtype=np.float64)
    for n, row in enumerate(rows):
        source = by_part[row["oper_part_no"]]
        weeks = [step_week(row["origin_step"] + offset) for offset in range(-length, horizon)]
        values = np.asarray([source.get(week, np.nan) for week in weeks], dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"선택한 창에 결측·음수·Inf가 있습니다: {row['oper_part_no']} / {row['origin']}")
        history[n, :, 0], target[n, :, 0] = values[:length], values[length:]
    if not np.isfinite(history).all():
        raise ValueError("FP32 입력 범위를 벗어나는 수요입니다.")
    return history, target

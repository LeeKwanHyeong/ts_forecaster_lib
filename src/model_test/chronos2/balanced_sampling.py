"""미래 실적을 보지 않고 수요 유형별 표본을 확정하는 백테스트 준비 도구."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
from pathlib import Path

import numpy as np
import polars as pl


HORIZON = 27
CONTEXT = 52
PATTERNS = ("smooth", "erratic", "intermittent", "lumpy")
ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49


def week_sequence(origin: int, start: int, count: int) -> list[int]:
    """YYYYWW를 정수로 더하지 않고 ISO 달력으로 연말과 53주차를 처리."""
    monday = date.fromisocalendar(origin // 100, origin % 100, 1)
    days = [monday + timedelta(weeks=start + step) for step in range(count)]
    return [day.isocalendar().year * 100 + day.isocalendar().week for day in days]


@dataclass(frozen=True)
class Panel:
    name: str
    origins: tuple[int, ...]
    patterns: tuple[str, ...]


DEFAULT_PANELS = (
    Panel("long_history_two_patterns", (202137, 202212, 202239, 202314, 202341, 202416, 202443, 202518),
          ("smooth", "intermittent")),
    Panel("recent_four_patterns", (202502, 202506, 202510, 202514, 202518), PATTERNS),
)


def source_scan(source: Path) -> pl.LazyFrame:
    return pl.scan_parquet(source).select("oper_part_no", "demand_dt", "demand_qty")


def profile_history(source: Path, origin: int) -> pl.DataFrame:
    """직전 52주만 조회. 중복·누락·비유한 수량은 별도 사유로 제외.

    ADI는 관측 주수/양의 수요 주수, CV²는 양의 수요의 표본분산/평균².
    무수요 및 양의 수요 1회는 변동성을 정의할 수 없어 4유형에 강제 편입하지 않음.
    """
    weeks = week_sequence(origin, -CONTEXT, CONTEXT)
    positive = pl.col("demand_qty").filter(pl.col("demand_qty") > 0)
    stats = (source_scan(source).filter(pl.col("demand_dt").is_in(weeks))
        .sort("oper_part_no", "demand_dt")
        .group_by("oper_part_no").agg(
            pl.len().alias("rows"), pl.col("demand_dt").n_unique().alias("weeks"),
            pl.col("demand_qty").is_null().any().alias("has_null"),
            (~pl.col("demand_qty").is_finite()).fill_null(True).any().alias("nonfinite"),
            (pl.col("demand_qty") < 0).any().alias("negative"),
            (pl.col("demand_qty") > 0).sum().alias("positive_weeks"),
            positive.mean().alias("positive_mean"), positive.var(ddof=1).alias("positive_variance"),
        ).with_columns(
            (CONTEXT / pl.col("positive_weeks").cast(pl.Float64)).alias("adi"),
            (pl.col("positive_variance") / pl.col("positive_mean").pow(2)).alias("cv2"),
        ).with_columns(
            pl.when(pl.col("oper_part_no").is_null()).then(pl.lit("null_part_id"))
            .when(pl.col("rows") != pl.col("weeks")).then(pl.lit("duplicate_week"))
            .when(pl.col("weeks") != CONTEXT).then(pl.lit("incomplete_history"))
            .when(pl.col("has_null") | pl.col("nonfinite") | pl.col("negative"))
            .then(pl.lit("invalid_quantity"))
            .when(pl.col("positive_weeks") == 0).then(pl.lit("no_demand"))
            .when(pl.col("positive_weeks") == 1).then(pl.lit("single_event"))
            .otherwise(pl.lit("eligible")).alias("eligibility"),
        ).with_columns(
            pl.when(pl.col("eligibility") != "eligible").then(pl.lit(None, dtype=pl.String))
            .when(pl.col("adi") < ADI_THRESHOLD)
            .then(pl.when(pl.col("cv2") < CV2_THRESHOLD).then(pl.lit("smooth")).otherwise(pl.lit("erratic")))
            .otherwise(pl.when(pl.col("cv2") < CV2_THRESHOLD).then(pl.lit("intermittent")).otherwise(pl.lit("lumpy")))
            .alias("pattern")
        ).sort("oper_part_no").collect())
    # 무수요의 ADI=inf와 미정의 CV²는 JSON/CSV에서 빈 값으로 보존.
    return stats.with_columns([
        pl.when(pl.col(name).is_finite()).then(pl.col(name)).otherwise(None).alias(name)
        for name in ("adi", "cv2", "positive_mean", "positive_variance")
    ])


def choose_cohort(profile: pl.DataFrame, panel: Panel, origin: int, per_pattern: int, seed: int):
    """부품 순서는 난수 재현용일 뿐 선택 기준이 아님. 유형 내 비복원 추출."""
    if type(per_pattern) is not int or not 1 <= per_pattern <= 200:
        raise ValueError("per_pattern은 1~200 사이여야 합니다.")
    if not panel.patterns or len(set(panel.patterns)) != len(panel.patterns) or set(panel.patterns) - set(PATTERNS):
        raise ValueError("중복 없는 유효한 수요 유형이 필요합니다.")
    candidates = {pattern: profile.filter((pl.col("eligibility") == "eligible") & (pl.col("pattern") == pattern))
                  .sort("oper_part_no") for pattern in panel.patterns}
    counts = {pattern: frame.height for pattern, frame in candidates.items()}
    quota = min(per_pattern, min(counts.values()))
    audit = {"panel": panel.name, "origin": origin, "candidate_counts": counts,
             "target_per_pattern": per_pattern, "actual_per_pattern": quota,
             "status": "selected" if quota else "skipped_missing_pattern",
             "quality_counts": {str(row["eligibility"]): row["len"] for row in
                                profile.group_by("eligibility").len().to_dicts()}}
    selected = []
    for pattern, frame in candidates.items():
        # Python의 실행별 hash() 대신 SHA-256으로 난수 시드를 고정.
        digest = hashlib.sha256(f"{seed}:{panel.name}:{origin}:{pattern}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        for index in rng.choice(frame.height, size=quota, replace=False):
            row = frame.row(int(index), named=True)
            selected.append({"panel": panel.name, "origin": origin, "pattern": pattern,
                             "oper_part_no": str(row["oper_part_no"]), "adi": float(row["adi"]),
                             "cv2": float(row["cv2"]), "positive_weeks": int(row["positive_weeks"])})
    return selected, audit


def load_selected_window(source: Path, ids: list[str], origin: int, *, future: bool) -> np.ndarray:
    """선정 이후 정답을 별도 조회. 정답 누락 시 교체 표본을 뽑지 않고 실행 실패."""
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("중복 없는 부품 목록이 필요합니다.")
    weeks = week_sequence(origin, 0 if future else -CONTEXT, HORIZON if future else CONTEXT)
    data = (source_scan(source).filter(pl.col("oper_part_no").is_in(ids) & pl.col("demand_dt").is_in(weeks))
            .collect())
    lookup = {}
    for part, week, quantity in data.iter_rows():
        key = (part, week)
        if key in lookup:
            raise ValueError(f"선택 표본의 중복 주차: {part}/{week}")
        if quantity is None or not np.isfinite(quantity) or quantity < 0:
            raise ValueError(f"선택 표본의 잘못된 수량: {part}/{week}")
        lookup[key] = quantity
    if len(lookup) != len(ids) * len(weeks):
        raise ValueError("선택 표본의 이력 또는 27주 실적 누락. 미래 정답 기준 재추출 금지.")
    # 평가 정답은 float64 원 수량을 유지하고 모델 입력만 호출부에서 float32로 변환.
    return np.asarray([[lookup[(part, week)] for week in weeks] for part in ids], dtype=np.float64)


def last_value_naive(history: np.ndarray) -> np.ndarray:
    if history.ndim != 2 or history.shape[1] != CONTEXT or not np.isfinite(history).all():
        raise ValueError("Naive 입력은 유한한 (부품 수, 52) 배열이어야 합니다.")
    return np.repeat(history[:, -1:], HORIZON, axis=1)

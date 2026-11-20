"""수요의 의미와 부품·정답 기간을 시간 창 생성 전에 고정하는 계약."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Iterable, Literal

from modeling_module.data_loader.temporal import add_period, normalize_period_key


SplitName = Literal["train", "validation", "test"]


def _name(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}은 비어 있지 않은 문자열이어야 합니다.")


def _week(value: int) -> int:
    if type(value) is not int:
        raise TypeError("주차는 ISO YYYYWW 정수여야 합니다.")
    return normalize_period_key(value, "weekly")


def manifest_sha256(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, order=True)
class SeriesKey:
    """부품 번호 또는 회사·법인·사이트를 포함한 복합 식별값. 숫자로 강제 변환하지 않음."""

    values: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.values, (tuple, list)) or not self.values:
            raise ValueError("SeriesKey에는 하나 이상의 식별값이 필요합니다.")
        for value in self.values:
            _name(value, "식별값")
        object.__setattr__(self, "values", tuple(self.values))


@dataclass(frozen=True)
class Chronos2DataContract:
    """선언된 원천 계약이며 수량 품질 검사나 실제 학습 실행 승인을 대신하지 않음."""

    source_id: str
    source_sha256: str
    series_columns: tuple[str, ...] = ("oper_part_no",)
    time_column: str = "demand_dt"
    target_column: str = "demand_qty"
    provenance: Literal["unverified", "synthetic", "user_provided"] = "unverified"
    zero_semantics: Literal["unconfirmed", "observed_zero", "explicit_mask"] = "unconfirmed"
    observed_mask_column: str | None = None

    def __post_init__(self):
        _name(self.source_id, "source_id")
        if not isinstance(self.source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ValueError("source_sha256은 64자리 소문자 SHA-256이어야 합니다.")
        if not isinstance(self.series_columns, (tuple, list)) or not self.series_columns:
            raise ValueError("series_columns가 필요합니다.")
        columns = tuple(self.series_columns) + (self.time_column, self.target_column)
        if self.observed_mask_column is not None:
            columns += (self.observed_mask_column,)
        for column in columns:
            _name(column, "칼럼명")
        if len(set(columns)) != len(columns):
            raise ValueError("식별·주차·수량·마스크 칼럼은 중복될 수 없습니다.")
        if self.provenance not in {"unverified", "synthetic", "user_provided"}:
            raise ValueError("지원하지 않는 원천 출처입니다.")
        if self.zero_semantics not in {"unconfirmed", "observed_zero", "explicit_mask"}:
            raise ValueError("지원하지 않는 0수요 의미입니다.")
        if self.zero_semantics == "explicit_mask" and self.observed_mask_column is None:
            raise ValueError("0이 결측 대체값일 수 있으면 실제 관측 여부 마스크 칼럼이 필요합니다.")
        object.__setattr__(self, "series_columns", tuple(self.series_columns))

    def assert_semantics_confirmed(self) -> None:
        if self.provenance == "unverified" or self.zero_semantics == "unconfirmed":
            raise ValueError("출처와 0수요 의미를 확인하기 전에는 실제 데이터 학습을 진행할 수 없습니다.")


@dataclass(frozen=True)
class ForecastPartition:
    """시계열 ID와 정답 허용 기간. 입력 context는 이 기간보다 앞설 수 있음."""

    series: tuple[SeriesKey, ...]
    target_start: int
    target_end: int

    def __post_init__(self):
        if not isinstance(self.series, (tuple, list)) or not self.series or any(not isinstance(key, SeriesKey) for key in self.series):
            raise ValueError("분할에는 하나 이상의 SeriesKey가 필요합니다.")
        if len(set(self.series)) != len(self.series):
            raise ValueError("분할 내부에 중복 식별자가 있습니다.")
        if _week(self.target_start) > _week(self.target_end):
            raise ValueError("정답 시작 주차는 종료 주차보다 늦을 수 없습니다.")
        object.__setattr__(self, "series", tuple(self.series))


@dataclass(frozen=True)
class ForecastWindow:
    series: SeriesKey
    origin: int

    def __post_init__(self):
        if not isinstance(self.series, SeriesKey):
            raise TypeError("series는 SeriesKey여야 합니다.")
        _week(self.origin)

    def context_weeks(self, length: int) -> tuple[int, ...]:
        if type(length) is not int or length < 1:
            raise ValueError("context 길이는 양의 정수여야 합니다.")
        return tuple(add_period(self.origin, offset, "weekly") for offset in range(-length, 0))

    def target_weeks(self, horizon: int) -> tuple[int, ...]:
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon은 양의 정수여야 합니다.")
        return tuple(add_period(self.origin, offset, "weekly") for offset in range(horizon))


@dataclass(frozen=True)
class Chronos2SplitContract:
    """1차 평가는 부품·정답 기간 모두 분리하는 엄격한 cold-part/forward-time 계약."""

    train: ForecastPartition
    validation: ForecastPartition
    test: ForecastPartition
    context_length: int = 52
    horizon: int = 27
    seed: int = 42

    def __post_init__(self):
        for name in ("context_length", "horizon"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name}은 양의 정수여야 합니다.")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed는 0 이상의 정수여야 합니다.")
        parts = (self.train, self.validation, self.test)
        if any(not isinstance(part, ForecastPartition) for part in parts):
            raise TypeError("분할은 ForecastPartition이어야 합니다.")
        keys = [set(part.series) for part in parts]
        if any(keys[i] & keys[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("학습·검증·시험 사이에 부품 ID가 겹칩니다.")
        if not (self.train.target_end < self.validation.target_start <= self.validation.target_end
                < self.test.target_start):
            raise ValueError("학습·검증·시험의 정답 기간은 순서대로 분리해야 합니다.")
        for part in parts:
            if add_period(part.target_start, self.horizon - 1, "weekly") > part.target_end:
                raise ValueError("각 정답 기간은 적어도 horizon 길이를 포함해야 합니다.")

    def validate_window(self, split: SplitName, window: ForecastWindow) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split은 train, validation, test 중 하나여야 합니다.")
        if not isinstance(window, ForecastWindow):
            raise TypeError("window는 ForecastWindow여야 합니다.")
        part = getattr(self, split)
        if window.series not in part.series:
            raise ValueError("해당 분할에 속하지 않은 부품입니다.")
        last = add_period(window.origin, self.horizon - 1, "weekly")
        if window.origin < part.target_start or last > part.target_end:
            raise ValueError("예측 정답이 허용 기간을 벗어나 cutoff를 침범합니다.")

    def validate_windows(self, assignments: Iterable[tuple[SplitName, ForecastWindow]]) -> None:
        seen = set()
        for split, window in assignments:
            self.validate_window(split, window)
            if (split, window) in seen:
                raise ValueError("동일한 부품·예측 시작 주차가 중복되었습니다.")
            seen.add((split, window))

    def manifest(self, source: Chronos2DataContract) -> dict:
        if not isinstance(source, Chronos2DataContract):
            raise TypeError("source는 Chronos2DataContract여야 합니다.")
        width = len(source.series_columns)
        if any(len(key.values) != width for part in (self.train, self.validation, self.test) for key in part.series):
            raise ValueError("복합 식별값 수와 series_columns 수가 다릅니다.")
        result = {"schema_version": 1, "source": asdict(source), "split": asdict(self),
                  "holdout_level": "series", "normalization_scope": "context_only",
                  "pattern_classification_scope": "context_only", "missing_fill": "never_implicit_zero",
                  "semantics_confirmed": source.provenance != "unverified" and source.zero_semantics != "unconfirmed",
                  "quantity_and_window_quality_validated": False, "training_executed": False}
        return {**result, "manifest_sha256": manifest_sha256(result)}


def make_split_contract(source: Chronos2DataContract, series: Iterable[SeriesKey], first_week: int,
                        last_week: int, *, excluded_from_test: Iterable[SeriesKey] = (), seed: int = 42,
                        context_length: int = 52, horizon: int = 27) -> Chronos2SplitContract:
    """ID hash만으로 70:15:15 배정. 시험 제외 부품을 전체 학습 풀에서 버리지 않음."""
    if type(seed) is not int or seed < 0 or type(horizon) is not int or horizon < 1:
        raise ValueError("seed와 horizon이 올바르지 않습니다.")
    if not isinstance(source, Chronos2DataContract):
        raise TypeError("source는 Chronos2DataContract여야 합니다.")
    keys = tuple(series)
    if any(not isinstance(key, SeriesKey) for key in keys):
        raise TypeError("모든 식별자는 SeriesKey여야 합니다.")
    keys = tuple(sorted(keys))
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("식별자는 비어 있거나 중복될 수 없습니다.")
    _week(first_week)
    _week(last_week)
    excluded = set(excluded_from_test)
    if any(not isinstance(key, SeriesKey) or len(key.values) != len(source.series_columns) for key in excluded):
        raise ValueError("시험 제외 목록도 원천과 동일한 SeriesKey 형식이어야 합니다.")
    n_test = len(keys) * 15 // 100
    n_val = len(keys) * 15 // 100
    if min(n_test, n_val) < 1:
        raise ValueError("70:15:15 분할을 구성할 부품 수가 부족합니다.")
    def ranking(key, role):
        return manifest_sha256({"seed": seed, "role": role, "series": key.values})
    candidates = sorted((key for key in keys if key not in excluded), key=lambda key: ranking(key, "test"))
    if len(candidates) < n_test:
        raise ValueError("기존 평가 부품을 제외한 새 시험 후보 수가 부족합니다.")
    test_keys = set(candidates[:n_test])
    remaining = sorted((key for key in keys if key not in test_keys), key=lambda key: ranking(key, "validation"))
    val_keys = set(remaining[:n_val])
    train_keys = set(keys) - test_keys - val_keys
    test_start = add_period(last_week, -(horizon - 1), "weekly")
    val_end = add_period(test_start, -1, "weekly")
    val_start = add_period(val_end, -(horizon - 1), "weekly")
    result = Chronos2SplitContract(
        train=ForecastPartition(tuple(sorted(train_keys)), first_week, add_period(val_start, -1, "weekly")),
        validation=ForecastPartition(tuple(sorted(val_keys)), val_start, val_end),
        test=ForecastPartition(tuple(sorted(test_keys)), test_start, last_week),
        context_length=context_length, horizon=horizon, seed=seed)
    result.manifest(source)
    return result

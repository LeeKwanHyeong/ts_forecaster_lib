"""Public contracts for lifecycle-anchored forecasting inputs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from numbers import Real
from typing import Any, Iterable, Mapping, Optional


LIFECYCLE_INPUT_CONTRACT_ID = "modeling-module-lifecycle-input-v1"
LIFECYCLE_INPUT_CONTRACT_VERSION = 1
LTB_OBSERVED_MONTHS = 12
LTB_FORECAST_MONTHS = 72
LTB_TOTAL_MONTHS = 84


class LifecycleValidationError(ValueError):
    """Raised when one lifecycle input violates the frozen public contract."""


class LifecycleSamplePurpose(StrEnum):
    """Whether a sample carries labels or represents an operating forecast."""

    TRAINING = "training"
    INFERENCE = "inference"


def _normalize_names(values: Iterable[str], *, field_name: str) -> tuple[str, ...]:
    names = tuple(str(value).strip() for value in values)
    if any(not name for name in names):
        raise LifecycleValidationError(f"{field_name} cannot contain empty names")
    if len(set(names)) != len(names):
        raise LifecycleValidationError(f"{field_name} contains duplicate names")
    return names


def _require_month_start(value: object, *, field_name: str) -> date:
    if isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a date, not datetime")
    if not isinstance(value, date):
        raise TypeError(f"{field_name} must be a date")
    if value.day != 1:
        raise LifecycleValidationError(f"{field_name} must be the first day of a month")
    return value


def add_calendar_months(value: date, months: int) -> date:
    """Add whole calendar months to a canonical month-start date."""

    month_start = _require_month_start(value, field_name="value")
    if isinstance(months, bool) or not isinstance(months, int):
        raise TypeError("months must be an integer")
    month_index = month_start.year * 12 + month_start.month - 1 + months
    if month_index < 12:
        raise LifecycleValidationError("resulting month is outside the supported date range")
    year, zero_based_month = divmod(month_index, 12)
    return date(year, zero_based_month + 1, 1)


@dataclass(frozen=True, slots=True)
class LifecycleWindowSpec:
    """Frozen LTB window: M0..M11 observed and M12..M83 forecast."""

    observed_months: int = LTB_OBSERVED_MONTHS
    forecast_months: int = LTB_FORECAST_MONTHS
    total_months: int = LTB_TOTAL_MONTHS
    version: int = LIFECYCLE_INPUT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        values = (
            self.observed_months,
            self.forecast_months,
            self.total_months,
            self.version,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("lifecycle window values must be integers")
        if self.version != LIFECYCLE_INPUT_CONTRACT_VERSION:
            raise LifecycleValidationError(
                f"unsupported lifecycle contract version={self.version}"
            )
        if (
            self.observed_months,
            self.forecast_months,
            self.total_months,
        ) != (LTB_OBSERVED_MONTHS, LTB_FORECAST_MONTHS, LTB_TOTAL_MONTHS):
            raise LifecycleValidationError(
                "lifecycle input v1 is fixed to 12 observed, 72 forecast, "
                "and 84 total months"
            )

    @property
    def forecast_start_index(self) -> int:
        return self.observed_months

    @property
    def forecast_end_index(self) -> int:
        return self.total_months - 1

    def to_dict(self) -> dict[str, int | str]:
        return {
            "contract_id": LIFECYCLE_INPUT_CONTRACT_ID,
            "version": self.version,
            "observed_months": self.observed_months,
            "forecast_months": self.forecast_months,
            "total_months": self.total_months,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LifecycleWindowSpec":
        if not isinstance(payload, Mapping):
            raise TypeError("lifecycle window payload must be a mapping")
        expected = {
            "contract_id",
            "version",
            "observed_months",
            "forecast_months",
            "total_months",
        }
        if set(payload) != expected:
            raise LifecycleValidationError(
                "lifecycle window payload has an invalid schema"
            )
        if payload["contract_id"] != LIFECYCLE_INPUT_CONTRACT_ID:
            raise LifecycleValidationError(
                "unsupported lifecycle input contract ID"
            )
        return cls(
            version=int(payload["version"]),
            observed_months=int(payload["observed_months"]),
            forecast_months=int(payload["forecast_months"]),
            total_months=int(payload["total_months"]),
        )


@dataclass(frozen=True, slots=True)
class LifecycleFeatureSchema:
    """Ordered feature roles known at the LTB forecast origin.

    Observed features cover M0..M11. Known-future features cover M12..M83
    and must be values genuinely available at the forecast origin. Actual
    future sales, failures, and demand must never be placed in those roles.
    """

    static_cont_names: tuple[str, ...] = ()
    static_cat_names: tuple[str, ...] = ()
    observed_cont_names: tuple[str, ...] = ()
    observed_cat_names: tuple[str, ...] = ()
    known_future_cont_names: tuple[str, ...] = ()
    known_future_cat_names: tuple[str, ...] = ()
    version: int = LIFECYCLE_INPUT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.version != LIFECYCLE_INPUT_CONTRACT_VERSION:
            raise LifecycleValidationError(
                f"unsupported lifecycle feature schema version={self.version}"
            )
        normalized = {
            field_name: _normalize_names(getattr(self, field_name), field_name=field_name)
            for field_name in (
                "static_cont_names",
                "static_cat_names",
                "observed_cont_names",
                "observed_cat_names",
                "known_future_cont_names",
                "known_future_cat_names",
            )
        }
        for field_name, value in normalized.items():
            object.__setattr__(self, field_name, value)

        continuous = set(self.static_cont_names).union(
            self.observed_cont_names,
            self.known_future_cont_names,
        )
        categorical = set(self.static_cat_names).union(
            self.observed_cat_names,
            self.known_future_cat_names,
        )
        overlap = sorted(continuous.intersection(categorical))
        if overlap:
            raise LifecycleValidationError(
                "features cannot be both continuous and categorical: "
                + ", ".join(overlap)
            )

        static = set(self.static_cont_names).union(self.static_cat_names)
        temporal = set(self.observed_cont_names).union(
            self.observed_cat_names,
            self.known_future_cont_names,
            self.known_future_cat_names,
        )
        static_temporal_overlap = sorted(static.intersection(temporal))
        if static_temporal_overlap:
            raise LifecycleValidationError(
                "features cannot be both static and temporal: "
                + ", ".join(static_temporal_overlap)
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LifecycleFeatureSchema":
        if not isinstance(payload, Mapping):
            raise TypeError("lifecycle feature schema payload must be a mapping")
        allowed = {
            "version",
            "static_cont_names",
            "static_cat_names",
            "observed_cont_names",
            "observed_cat_names",
            "known_future_cont_names",
            "known_future_cat_names",
        }
        unexpected = sorted(set(payload).difference(allowed))
        if unexpected:
            raise LifecycleValidationError(
                "unsupported lifecycle feature schema fields: " + ", ".join(unexpected)
            )
        return cls(
            version=int(payload.get("version", LIFECYCLE_INPUT_CONTRACT_VERSION)),
            static_cont_names=tuple(payload.get("static_cont_names", ())),
            static_cat_names=tuple(payload.get("static_cat_names", ())),
            observed_cont_names=tuple(payload.get("observed_cont_names", ())),
            observed_cat_names=tuple(payload.get("observed_cat_names", ())),
            known_future_cont_names=tuple(payload.get("known_future_cont_names", ())),
            known_future_cat_names=tuple(payload.get("known_future_cat_names", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "static_cont_names": list(self.static_cont_names),
            "static_cat_names": list(self.static_cat_names),
            "observed_cont_names": list(self.observed_cont_names),
            "observed_cat_names": list(self.observed_cat_names),
            "known_future_cont_names": list(self.known_future_cont_names),
            "known_future_cat_names": list(self.known_future_cat_names),
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _normalize_target(values: Iterable[Real], *, field_name: str, size: int) -> tuple[float, ...]:
    normalized = tuple(values)
    if len(normalized) != size:
        raise LifecycleValidationError(
            f"{field_name} must contain exactly {size} monthly values"
        )
    output: list[float] = []
    for value in normalized:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{field_name} values must be real numbers")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise LifecycleValidationError(
                f"{field_name} values must be finite and non-negative"
            )
        output.append(number)
    return tuple(output)


def _normalize_cont_row(values: Iterable[Optional[Real]], *, field_name: str) -> tuple[Optional[float], ...]:
    output: list[Optional[float]] = []
    for value in values:
        if value is None:
            output.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{field_name} values must be real numbers or None")
        number = float(value)
        if not math.isfinite(number):
            raise LifecycleValidationError(f"{field_name} values must be finite or None")
        output.append(number)
    return tuple(output)


def _normalize_matrix(
    values: Iterable[Iterable[Any]],
    *,
    field_name: str,
    steps: int,
    width: int,
    continuous: bool,
) -> tuple[tuple[Any, ...], ...]:
    rows = tuple(tuple(row) for row in values)
    if width == 0:
        if rows:
            raise LifecycleValidationError(
                f"{field_name} must be empty when its schema has no features"
            )
        return ()
    if len(rows) != steps:
        raise LifecycleValidationError(f"{field_name} must contain exactly {steps} rows")
    if any(len(row) != width for row in rows):
        raise LifecycleValidationError(
            f"{field_name} rows must contain exactly {width} features"
        )
    if continuous:
        return tuple(
            _normalize_cont_row(row, field_name=field_name)
            for row in rows
        )
    return rows


@dataclass(frozen=True, slots=True)
class LifecycleSample:
    """One leakage-safe lifecycle sample before categorical encoding."""

    sample_id: str
    purpose: LifecycleSamplePurpose
    lifecycle_start_month: date
    source_cutoff_month: date
    observed_target: tuple[float, ...]
    future_target: Optional[tuple[float, ...]] = None
    feature_schema: LifecycleFeatureSchema = field(default_factory=LifecycleFeatureSchema)
    static_cont: tuple[Optional[float], ...] = ()
    static_cat: tuple[Any, ...] = ()
    observed_cont: tuple[tuple[Optional[float], ...], ...] = ()
    observed_cat: tuple[tuple[Any, ...], ...] = ()
    known_future_cont: tuple[tuple[Optional[float], ...], ...] = ()
    known_future_cat: tuple[tuple[Any, ...], ...] = ()
    window: LifecycleWindowSpec = field(default_factory=LifecycleWindowSpec)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise LifecycleValidationError("sample_id must be non-empty text")
        if not isinstance(self.purpose, LifecycleSamplePurpose):
            raise TypeError("purpose must be LifecycleSamplePurpose")
        if not isinstance(self.feature_schema, LifecycleFeatureSchema):
            raise TypeError("feature_schema must be LifecycleFeatureSchema")
        if not isinstance(self.window, LifecycleWindowSpec):
            raise TypeError("window must be LifecycleWindowSpec")

        start = _require_month_start(
            self.lifecycle_start_month,
            field_name="lifecycle_start_month",
        )
        cutoff = _require_month_start(
            self.source_cutoff_month,
            field_name="source_cutoff_month",
        )
        observation_end = add_calendar_months(start, self.window.observed_months - 1)
        lifecycle_end = add_calendar_months(start, self.window.total_months - 1)
        if cutoff < observation_end:
            raise LifecycleValidationError(
                "source_cutoff_month does not cover all 12 observed months"
            )

        observed_target = _normalize_target(
            self.observed_target,
            field_name="observed_target",
            size=self.window.observed_months,
        )
        object.__setattr__(self, "observed_target", observed_target)

        if self.purpose is LifecycleSamplePurpose.TRAINING:
            if self.future_target is None:
                raise LifecycleValidationError("training samples require future_target")
            if cutoff < lifecycle_end:
                raise LifecycleValidationError(
                    "training source_cutoff_month must cover the full 84-month lifecycle"
                )
            future_target = _normalize_target(
                self.future_target,
                field_name="future_target",
                size=self.window.forecast_months,
            )
            object.__setattr__(self, "future_target", future_target)
        elif self.future_target is not None:
            raise LifecycleValidationError(
                "inference samples must not contain future_target"
            )

        static_cont = _normalize_cont_row(self.static_cont, field_name="static_cont")
        if len(static_cont) != len(self.feature_schema.static_cont_names):
            raise LifecycleValidationError(
                "static_cont width does not match feature_schema"
            )
        static_cat = tuple(self.static_cat)
        if len(static_cat) != len(self.feature_schema.static_cat_names):
            raise LifecycleValidationError(
                "static_cat width does not match feature_schema"
            )
        object.__setattr__(self, "static_cont", static_cont)
        object.__setattr__(self, "static_cat", static_cat)

        matrix_specs = (
            ("observed_cont", self.window.observed_months, len(self.feature_schema.observed_cont_names), True),
            ("observed_cat", self.window.observed_months, len(self.feature_schema.observed_cat_names), False),
            ("known_future_cont", self.window.forecast_months, len(self.feature_schema.known_future_cont_names), True),
            ("known_future_cat", self.window.forecast_months, len(self.feature_schema.known_future_cat_names), False),
        )
        for field_name, steps, width, continuous in matrix_specs:
            normalized = _normalize_matrix(
                getattr(self, field_name),
                field_name=field_name,
                steps=steps,
                width=width,
                continuous=continuous,
            )
            object.__setattr__(self, field_name, normalized)

    @property
    def observation_end_month(self) -> date:
        return add_calendar_months(
            self.lifecycle_start_month,
            self.window.observed_months - 1,
        )

    @property
    def forecast_start_month(self) -> date:
        return add_calendar_months(
            self.lifecycle_start_month,
            self.window.observed_months,
        )

    @property
    def lifecycle_end_month(self) -> date:
        return add_calendar_months(
            self.lifecycle_start_month,
            self.window.total_months - 1,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible lifecycle request item."""

        return {
            "contract_id": LIFECYCLE_INPUT_CONTRACT_ID,
            "sample_id": self.sample_id,
            "purpose": self.purpose.value,
            "lifecycle_start_month": self.lifecycle_start_month.isoformat(),
            "source_cutoff_month": self.source_cutoff_month.isoformat(),
            "observed_target": list(self.observed_target),
            "future_target": (
                None if self.future_target is None else list(self.future_target)
            ),
            "feature_schema": self.feature_schema.to_dict(),
            "static_cont": list(self.static_cont),
            "static_cat": list(self.static_cat),
            "observed_cont": [list(row) for row in self.observed_cont],
            "observed_cat": [list(row) for row in self.observed_cat],
            "known_future_cont": [
                list(row) for row in self.known_future_cont
            ],
            "known_future_cat": [
                list(row) for row in self.known_future_cat
            ],
            "window": self.window.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LifecycleSample":
        if not isinstance(payload, Mapping):
            raise TypeError("lifecycle sample payload must be a mapping")
        expected = {
            "contract_id",
            "sample_id",
            "purpose",
            "lifecycle_start_month",
            "source_cutoff_month",
            "observed_target",
            "future_target",
            "feature_schema",
            "static_cont",
            "static_cat",
            "observed_cont",
            "observed_cat",
            "known_future_cont",
            "known_future_cat",
            "window",
        }
        if set(payload) != expected:
            raise LifecycleValidationError(
                "lifecycle sample payload has an invalid schema"
            )
        if payload["contract_id"] != LIFECYCLE_INPUT_CONTRACT_ID:
            raise LifecycleValidationError(
                "unsupported lifecycle input contract ID"
            )
        future = payload["future_target"]
        return cls(
            sample_id=str(payload["sample_id"]),
            purpose=LifecycleSamplePurpose(str(payload["purpose"])),
            lifecycle_start_month=date.fromisoformat(
                str(payload["lifecycle_start_month"])
            ),
            source_cutoff_month=date.fromisoformat(
                str(payload["source_cutoff_month"])
            ),
            observed_target=tuple(payload["observed_target"]),
            future_target=None if future is None else tuple(future),
            feature_schema=LifecycleFeatureSchema.from_dict(
                payload["feature_schema"]
            ),
            static_cont=tuple(payload["static_cont"]),
            static_cat=tuple(payload["static_cat"]),
            observed_cont=tuple(
                tuple(row) for row in payload["observed_cont"]
            ),
            observed_cat=tuple(
                tuple(row) for row in payload["observed_cat"]
            ),
            known_future_cont=tuple(
                tuple(row) for row in payload["known_future_cont"]
            ),
            known_future_cat=tuple(
                tuple(row) for row in payload["known_future_cat"]
            ),
            window=LifecycleWindowSpec.from_dict(payload["window"]),
        )


__all__ = [
    "LIFECYCLE_INPUT_CONTRACT_ID",
    "LIFECYCLE_INPUT_CONTRACT_VERSION",
    "LTB_FORECAST_MONTHS",
    "LTB_OBSERVED_MONTHS",
    "LTB_TOTAL_MONTHS",
    "LifecycleFeatureSchema",
    "LifecycleSample",
    "LifecycleSamplePurpose",
    "LifecycleValidationError",
    "LifecycleWindowSpec",
    "add_calendar_months",
]

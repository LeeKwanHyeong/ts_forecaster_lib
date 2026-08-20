"""Deterministic weekly calendar features shared by governed training runs."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from typing import Final, Sequence

import polars as pl

from modeling_module.data_loader.temporal import normalize_period_key


WEEKLY_CALENDAR_CONTRACT_VERSION: Final = "2.0.0"
WEEKLY_CALENDAR_CONTINUOUS_FEATURES: Final = (
    "sin_annual",
    "cos_annual",
    "sin_semi",
    "cos_semi",
    "sin_quarter",
    "cos_quarter",
    "week_of_year",
    "peak_season_flag",
    "is_year_start",
    "is_year_end",
    "is_q_start",
    "is_q_end",
)


def weekly_calendar_schema_fingerprint(
    feature_columns: Sequence[str] = WEEKLY_CALENDAR_CONTINUOUS_FEATURES,
) -> str:
    """Return the stable identity of the ordered calendar feature contract."""

    columns = _validate_feature_columns(feature_columns)
    payload = {
        "contract_version": WEEKLY_CALENDAR_CONTRACT_VERSION,
        "frequency": "weekly",
        "iso_week_denominators": [52, 26, 13],
        "ordered_continuous_columns": list(columns),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def attach_weekly_calendar_features(
    frame: pl.DataFrame,
    *,
    date_column: str,
    feature_columns: Sequence[str] = WEEKLY_CALENDAR_CONTINUOUS_FEATURES,
) -> pl.DataFrame:
    """Attach Demand Engine-compatible ISO calendar features to a weekly frame.

    The input date column uses the canonical ``YYYYWW`` integer key. Feature
    values are calculated once per distinct week and joined back to the source
    frame, so large multi-series datasets do not repeat Python date conversion.
    """

    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"frame must be a polars DataFrame, got {type(frame)!r}.")
    if frame.is_empty():
        raise ValueError("calendar feature input cannot be empty.")
    if date_column not in frame.columns:
        raise ValueError(
            f"calendar feature input is missing date column {date_column!r}."
        )

    columns = _validate_feature_columns(feature_columns)
    conflicts = tuple(column for column in columns if column in frame.columns)
    if conflicts:
        raise ValueError(
            "calendar feature columns already exist in the input frame: "
            + ", ".join(conflicts)
        )

    raw_weeks = frame[date_column]
    if raw_weeks.null_count() > 0:
        raise ValueError("calendar feature date column cannot contain null values.")

    weeks = sorted(
        normalize_period_key(int(raw_week), "weekly")
        for raw_week in raw_weeks.unique().to_list()
    )
    rows: list[dict[str, float | int]] = []
    two_pi = 2.0 * math.pi
    for week_key in weeks:
        iso_year, iso_week = divmod(week_key, 100)
        week_date = date.fromisocalendar(iso_year, iso_week, 1)
        month = week_date.month
        values = {
            "sin_annual": math.sin(two_pi * iso_week / 52),
            "cos_annual": math.cos(two_pi * iso_week / 52),
            "sin_semi": math.sin(two_pi * iso_week / 26),
            "cos_semi": math.cos(two_pi * iso_week / 26),
            "sin_quarter": math.sin(two_pi * iso_week / 13),
            "cos_quarter": math.cos(two_pi * iso_week / 13),
            "week_of_year": float(iso_week),
            "peak_season_flag": float(month in {11, 12, 1, 2}),
            "is_year_start": float(month <= 2),
            "is_year_end": float(month >= 11),
            "is_q_start": float(month in {1, 4, 7, 10}),
            "is_q_end": float(month in {3, 6, 9, 12}),
        }
        rows.append(
            {
                date_column: week_key,
                **{column: values[column] for column in columns},
            }
        )

    lookup = pl.DataFrame(rows).with_columns(
        pl.col(date_column).cast(pl.Int64),
        *[pl.col(column).cast(pl.Float64) for column in columns],
    )
    order_column = "__calendar_source_order"
    while order_column in frame.columns:
        order_column = f"_{order_column}"
    return (
        frame.with_row_index(order_column)
        .with_columns(pl.col(date_column).cast(pl.Int64))
        .join(lookup, on=date_column, how="left", validate="m:1")
        .sort(order_column)
        .drop(order_column)
    )


def _validate_feature_columns(feature_columns: Sequence[str]) -> tuple[str, ...]:
    columns = tuple(str(column).strip() for column in feature_columns)
    if not columns or any(not column for column in columns):
        raise ValueError("calendar feature columns must be non-empty.")
    if len(set(columns)) != len(columns):
        raise ValueError("calendar feature columns must be unique and ordered.")
    unsupported = tuple(
        column
        for column in columns
        if column not in WEEKLY_CALENDAR_CONTINUOUS_FEATURES
    )
    if unsupported:
        raise ValueError(
            "unsupported weekly calendar feature columns: "
            + ", ".join(unsupported)
        )
    return columns


__all__ = [
    "WEEKLY_CALENDAR_CONTRACT_VERSION",
    "WEEKLY_CALENDAR_CONTINUOUS_FEATURES",
    "attach_weekly_calendar_features",
    "weekly_calendar_schema_fingerprint",
]

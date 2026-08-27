"""Private canonical-period utilities for time-series data loaders."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl


SUPPORTED_FREQUENCIES = frozenset({"weekly", "monthly", "daily", "hourly"})


def normalize_period_key(value: Any, freq: str) -> int:
    """Normalize a date-like scalar to the canonical integer key for ``freq``.

    Weekly keys use ISO week-year/week. Integer inputs are validated against
    the corresponding calendar representation instead of accepted by shape.

    Args:
        value: A Python/Polars date-like scalar or canonical integer key.
        freq: One of ``weekly``, ``monthly``, ``daily``, or ``hourly``.

    Returns:
        A validated canonical integer period key.

    Raises:
        TypeError: If the value representation is unsupported.
        ValueError: If the frequency or calendar value is invalid.
    """
    normalized_freq = str(freq).strip().lower()
    if normalized_freq not in SUPPORTED_FREQUENCIES:
        raise ValueError(f"Unsupported frequency: {freq!r}")
    if value is None:
        raise ValueError("Temporal values cannot be null.")

    if isinstance(value, datetime):
        temporal_value: date | datetime = value
    elif isinstance(value, date):
        temporal_value = value
    elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return _validate_integer_key(int(value), normalized_freq)
    else:
        raise TypeError(f"Unsupported temporal value type: {type(value)!r}")

    if normalized_freq == "weekly":
        iso = temporal_value.isocalendar()
        return int(iso.year) * 100 + int(iso.week)
    if normalized_freq == "monthly":
        return temporal_value.year * 100 + temporal_value.month
    if normalized_freq == "daily":
        return temporal_value.year * 10_000 + temporal_value.month * 100 + temporal_value.day
    if not isinstance(temporal_value, datetime):
        raise TypeError("hourly frequency requires datetime or YYYYMMDDHH input")
    return (
        temporal_value.year * 1_000_000
        + temporal_value.month * 10_000
        + temporal_value.day * 100
        + temporal_value.hour
    )


def _validate_integer_key(value: int, freq: str) -> int:
    """Validate and return a canonical integer period key."""
    raw = str(value)
    try:
        if freq == "weekly":
            if len(raw) != 6:
                raise ValueError("weekly keys must use YYYYWW")
            year, week = divmod(value, 100)
            date.fromisocalendar(year, week, 1)
        elif freq == "monthly":
            if len(raw) != 6:
                raise ValueError("monthly keys must use YYYYMM")
            year, month = divmod(value, 100)
            date(year, month, 1)
        elif freq == "daily":
            if len(raw) != 8:
                raise ValueError("daily keys must use YYYYMMDD")
            datetime.strptime(raw, "%Y%m%d")
        else:
            if len(raw) != 10:
                raise ValueError("hourly keys must use YYYYMMDDHH")
            datetime.strptime(raw, "%Y%m%d%H")
    except ValueError as exc:
        raise ValueError(f"Invalid {freq} period key: {value!r}") from exc
    return value


def normalize_temporal_frame(df: pl.DataFrame, date_col: str, freq: str) -> pl.DataFrame:
    """Return a frame whose temporal column contains canonical ``pl.Int64`` keys."""
    if date_col not in df.columns:
        raise KeyError(f"date_col='{date_col}' not found in df.columns")
    keys = [normalize_period_key(value, freq) for value in df[date_col].to_list()]
    return df.with_columns(pl.Series(date_col, keys, dtype=pl.Int64))


def add_period(period_key: int, amount: int, freq: str) -> int:
    """Shift a validated canonical period key by ``amount`` periods."""
    key = normalize_period_key(period_key, freq)
    if freq == "weekly":
        year, week = divmod(key, 100)
        shifted = date.fromisocalendar(year, week, 1) + timedelta(weeks=amount)
        iso = shifted.isocalendar()
        return int(iso.year) * 100 + int(iso.week)
    if freq == "monthly":
        year, month = divmod(key, 100)
        month_index = year * 12 + month - 1 + amount
        shifted_year, shifted_month_index = divmod(month_index, 12)
        return shifted_year * 100 + shifted_month_index + 1
    if freq == "daily":
        shifted = datetime.strptime(str(key), "%Y%m%d") + timedelta(days=amount)
        return int(shifted.strftime("%Y%m%d"))
    shifted = datetime.strptime(str(key), "%Y%m%d%H") + timedelta(hours=amount)
    return int(shifted.strftime("%Y%m%d%H"))


def lookback_periods(origin: date | datetime | int, length: int, freq: str) -> np.ndarray:
    """Return the ``length`` canonical periods immediately before ``origin``."""
    if length < 0:
        raise ValueError("lookback length must be non-negative")
    origin_key = normalize_period_key(origin, freq)
    return np.asarray(
        [add_period(origin_key, offset, freq) for offset in range(-length, 0)],
        dtype=np.int64,
    )

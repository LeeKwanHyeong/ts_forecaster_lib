"""High-level, result-returning public anchored forecast API."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np
import polars as pl

from modeling_module._internal.data_runtime import normalize_period_key
from modeling_module._internal.inference_runtime import _unpack_batch_for_export
from modeling_module.api.data import DataRequest, _materialize_payload, build_dataloader
from modeling_module.api.infer import LoadedPredictor, load_predictor


_FORECAST_SCHEMA = {
    "series_id": pl.String,
    "model_key": pl.String,
    "forecast_origin": pl.Int64,
    "horizon_step": pl.Int32,
    "point": pl.Float64,
    "q10": pl.Float64,
    "q50": pl.Float64,
    "q90": pl.Float64,
}


@dataclass(frozen=True)
class ForecastRuntimeConfig:
    """Runtime controls for one high-level forecast call.

    Args:
        batch_size: Number of series evaluated per model call.
        num_workers: Number of PyTorch DataLoader workers.
        device: Explicit runtime device, or ``None`` for library resolution.
        pin_memory: Whether the DataLoader uses pinned host memory.
        persistent_workers: Keep workers alive while iterating when enabled.
        prefetch_factor: Number of batches prefetched by each worker.
    """

    batch_size: int = 64
    num_workers: int = 0
    device: str | None = None
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2


@dataclass(frozen=True)
class ForecastRequest:
    """Complete request for checkpoint-backed anchored inference.

    Args:
        checkpoint_path: Path to one supported model checkpoint.
        expected_model_key: Optional exact artifact-key safety check.
        data: Public data request containing the long table and window config.
        series_ids: Ordered series selection, or ``None`` for all series.
        forecast_origin: W0/M0/daily/hourly origin represented as date-like or
            canonical integer input.
        runtime: Batch, worker, and device controls.
        unknown_series_policy: Whether unknown requested IDs fail or are ignored.
    """

    checkpoint_path: str | Path
    expected_model_key: str | None
    data: DataRequest
    series_ids: Sequence[str] | None
    forecast_origin: date | datetime | int
    runtime: ForecastRuntimeConfig = field(default_factory=ForecastRuntimeConfig)
    unknown_series_policy: Literal["error", "ignore"] = "error"


@dataclass(frozen=True)
class ForecastResult:
    """Normalized forecast rows and resolved checkpoint identity."""

    predictions: pl.DataFrame
    model_key: str
    forecast_origin: int


def _as_flat_float_array(value: Any, *, name: str, expected_size: int) -> np.ndarray:
    """Convert one predictor output to a validated flat float64 array."""
    if value is None:
        raise ValueError(f"Prediction output is missing required field {name!r}.")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != expected_size:
        raise ValueError(
            f"Prediction output {name!r} has {array.size} values; expected {expected_size}."
        )
    return array


def _optional_output_array(
    output: Mapping[str, Any],
    name: str,
    *,
    expected_size: int,
) -> Optional[np.ndarray]:
    """Return a validated optional predictor output array."""
    value = output.get(name)
    if value is None:
        return None
    return _as_flat_float_array(value, name=name, expected_size=expected_size)


def _empty_predictions() -> pl.DataFrame:
    """Build an empty result frame with the frozen ordered schema."""
    return pl.DataFrame(schema=_FORECAST_SCHEMA)


def _validate_request(request: ForecastRequest) -> dict[str, Any]:
    """Validate a public request and return its materialized data payload."""
    if not isinstance(request, ForecastRequest):
        raise TypeError(f"request must be ForecastRequest, got {type(request)!r}")
    if not isinstance(request.data, DataRequest):
        raise TypeError(f"request.data must be DataRequest, got {type(request.data)!r}")
    if not isinstance(request.runtime, ForecastRuntimeConfig):
        raise TypeError(
            "request.runtime must be ForecastRuntimeConfig, "
            f"got {type(request.runtime)!r}"
        )
    if request.runtime.batch_size <= 0:
        raise ValueError("runtime.batch_size must be positive")
    if request.runtime.num_workers < 0:
        raise ValueError("runtime.num_workers must be non-negative")
    if request.runtime.prefetch_factor <= 0:
        raise ValueError("runtime.prefetch_factor must be positive")
    if request.unknown_series_policy not in {"error", "ignore"}:
        raise ValueError("unknown_series_policy must be 'error' or 'ignore'")
    if isinstance(request.series_ids, (str, bytes)):
        raise TypeError("series_ids must be a sequence of IDs, not a string")
    if request.series_ids is not None and len(request.series_ids) == 0:
        raise ValueError("series_ids must not be empty; use None to select all series")

    payload = _materialize_payload(request.data)
    if payload.get("lookback") is None or payload.get("horizon") is None:
        raise ValueError("request.data must define both lookback and horizon")
    if int(payload["lookback"]) <= 0 or int(payload["horizon"]) <= 0:
        raise ValueError("lookback and horizon must be positive")
    backend = str(payload.get("backend") or "exo").strip().lower()
    if backend != "exo":
        raise ValueError("forecast() supports the canonical exo data backend only")
    return payload


def _validate_model_key(predictor: LoadedPredictor, expected_model_key: str | None) -> None:
    """Fail fast when a checkpoint does not match its expected artifact key."""
    if expected_model_key is None:
        return
    if predictor.model_key != str(expected_model_key):
        raise ValueError(
            "Checkpoint model key mismatch: "
            f"expected {expected_model_key!r}, got {predictor.model_key!r}."
        )


def _validate_checkpoint_exogenous_schema(
    predictor: LoadedPredictor,
    loader: Any,
) -> None:
    """Require inference feature roles and order to match the saved checkpoint."""
    expected = getattr(predictor, "exogenous_schema", None)
    if expected is None:
        return

    actual = getattr(loader, "exogenous_schema", None)
    if actual is None:
        raise ValueError(
            "Forecast request did not produce an exogenous schema, but the "
            "checkpoint requires one."
        )

    fields = (
        "past_cont_names",
        "past_cat_names",
        "future_cont_names",
        "future_cat_names",
        "past_cat_cardinalities",
        "future_cat_cardinalities",
    )
    mismatches = [
        f"{field}: expected {getattr(expected, field)!r}, "
        f"got {getattr(actual, field)!r}"
        for field in fields
        if getattr(expected, field) != getattr(actual, field)
    ]
    if mismatches:
        raise ValueError(
            "Forecast request exogenous schema does not match checkpoint "
            "schema: "
            + "; ".join(mismatches)
        )


def forecast(request: ForecastRequest) -> ForecastResult:
    """Run deterministic anchored inference without writing files.

    Args:
        request: Checkpoint, data, series selection, origin, and runtime config.

    Returns:
        A ``ForecastResult`` whose rows follow the frozen public Polars schema.

    Raises:
        TypeError: If request objects do not use the public dataclasses.
        ValueError: If the request, checkpoint identity, or output shape is invalid.
    """
    payload = _validate_request(request)
    freq = str(payload.get("freq", "weekly")).strip().lower()
    horizon = int(payload["horizon"])
    origin = normalize_period_key(request.forecast_origin, freq)

    predictor = load_predictor(
        str(request.checkpoint_path),
        device=request.runtime.device,
    )
    _validate_model_key(predictor, request.expected_model_key)

    loader_payload = dict(payload)
    loader_payload.update(
        {
            "stage": "inference",
            "plan_dt": request.forecast_origin,
            "series_ids": request.series_ids,
            "unknown_series_policy": request.unknown_series_policy,
            "batch_size": request.runtime.batch_size,
            "num_workers": request.runtime.num_workers,
            "pin_memory": request.runtime.pin_memory,
            "persistent_workers": request.runtime.persistent_workers,
            "prefetch_factor": request.runtime.prefetch_factor,
            "drop_last": False,
        }
    )
    categorical_vocabulary_artifact = getattr(
        predictor,
        "categorical_vocabulary_artifact",
        None,
    )
    if categorical_vocabulary_artifact is not None:
        loader_payload["categorical_vocabulary_artifact"] = (
            categorical_vocabulary_artifact
        )
    loader = build_dataloader(loader_payload)
    _validate_checkpoint_exogenous_schema(predictor, loader)

    rows: list[dict[str, Any]] = []
    series_ordinal = 0
    for batch in loader:
        unpacked = _unpack_batch_for_export(batch)
        raw_series_ids = unpacked.get("part_ids")
        if raw_series_ids is None:
            raise ValueError("Anchored inference batch is missing series identifiers.")
        series_ids = [str(value) for value in raw_series_ids]
        batch_size = len(series_ids)
        expected_size = batch_size * horizon

        output = predictor.predict(
            {
                "x": unpacked["x"],
                "part_ids": series_ids,
                "future_exo_batch": unpacked.get("future_exo"),
                "future_exo_cat_batch": unpacked.get(
                    "future_exo_cat"
                ),
                "past_exo_cont": unpacked.get("past_exo_cont"),
                "past_exo_cat": unpacked.get("past_exo_cat"),
            },
            horizon=horizon,
            device=request.runtime.device,
        )
        if not isinstance(output, Mapping):
            raise TypeError(f"Predictor output must be a mapping, got {type(output)!r}")

        q10 = _optional_output_array(output, "q10", expected_size=expected_size)
        q50 = _optional_output_array(output, "q50", expected_size=expected_size)
        q90 = _optional_output_array(output, "q90", expected_size=expected_size)
        quantile_presence = [value is not None for value in (q10, q50, q90)]
        if any(quantile_presence) and not all(quantile_presence):
            raise ValueError("Predictor output must provide q10, q50, and q90 together.")
        point_value = output.get("point")
        if point_value is None:
            point_value = output.get("q50")
        point = _as_flat_float_array(point_value, name="point", expected_size=expected_size)

        for batch_index, series_id in enumerate(series_ids):
            offset = batch_index * horizon
            for step in range(horizon):
                flat_index = offset + step
                rows.append(
                    {
                        "_series_ordinal": series_ordinal + batch_index,
                        "series_id": series_id,
                        "model_key": predictor.model_key,
                        "forecast_origin": origin,
                        "horizon_step": step,
                        "point": float(point[flat_index]),
                        "q10": None if q10 is None else float(q10[flat_index]),
                        "q50": None if q50 is None else float(q50[flat_index]),
                        "q90": None if q90 is None else float(q90[flat_index]),
                    }
                )
        series_ordinal += batch_size

    if not rows:
        predictions = _empty_predictions()
    else:
        predictions = (
            pl.DataFrame(rows)
            .sort("_series_ordinal", "horizon_step")
            .drop("_series_ordinal")
            .cast(_FORECAST_SCHEMA)
            .select(list(_FORECAST_SCHEMA))
        )
    return ForecastResult(
        predictions=predictions,
        model_key=predictor.model_key,
        forecast_origin=origin,
    )


__all__ = [
    "ForecastRequest",
    "ForecastResult",
    "ForecastRuntimeConfig",
    "forecast",
]

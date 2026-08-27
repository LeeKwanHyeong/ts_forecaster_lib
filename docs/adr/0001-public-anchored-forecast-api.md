# ADR 0001: Public Anchored Forecast API

- Status: Accepted
- Date: 2026-07-21
- Accepted: 2026-07-21
- Decision owners: `ts_forecaster_lib` maintainers
- Consumer reviewed: `DSIODemandEngine` (read-only)

## Contract identity

- Contract ID: `modeling-module.public-anchored-forecast`
- Contract version: `1.0.0`
- Contract file: `docs/contracts/public_forecast_contract.v1.json`
- Contract SHA-256: `07e8d2d825929bd9882d413c32faf76108b3f5e0d147d6a628575e0ebda563bd`
- Seal: SHA-256 of sorted compact ASCII canonical JSON after removing only
  the top-level `contract_sha256` field

## Context

`DSIODemandEngine` currently performs part-level anchored inference by importing
`modeling_module.data_loader`, `modeling_module.models`,
`modeling_module.training`, and `modeling_module.utils`. Those modules are
implementation details. The stable package surfaces are `modeling_module` and
`modeling_module.api`.

The exogenous DataModule is duplicated in a monolithic, capitalized module and a
lowercase modular implementation. The public data runtime currently selects the
capitalized implementation. Its anchored loader does not accept a series
selection or forward inference loader options. Both implementations compare a
date column converted to its physical integer representation with semantic
`YYYYWW`/`YYYYMM` keys. Consequently, `pl.Date` and `pl.Datetime` inputs can
produce an all-missing anchored window.

The library must own checkpoint restoration, temporal normalization, anchored
window construction, series selection, model execution, and forecast result
normalization. It must not own database access, environment loading, Consumer
paths, or persistence policy.

## Decision

### Public boundary

Add the following names to both stable surfaces:

```python
@dataclass(frozen=True)
class ForecastRuntimeConfig:
    batch_size: int = 64
    num_workers: int = 0
    device: str | None = None
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2


@dataclass(frozen=True)
class ForecastRequest:
    checkpoint_path: str | Path
    expected_model_key: str | None
    data: DataRequest
    series_ids: Sequence[str] | None
    forecast_origin: date | datetime | int
    runtime: ForecastRuntimeConfig = field(default_factory=ForecastRuntimeConfig)
    unknown_series_policy: Literal["error", "ignore"] = "error"


@dataclass(frozen=True)
class ForecastResult:
    predictions: pl.DataFrame
    model_key: str
    forecast_origin: int


def forecast(request: ForecastRequest) -> ForecastResult:
    ...
```

`DataRequest` remains the owner of lookback, horizon, frequency, column names,
missing-value policy, and exogenous configuration. The new request does not
duplicate those fields.

`forecast()` returns data and does not write files. Database, `.env`, path
selection, and Parquet persistence remain Consumer responsibilities.

### Temporal semantics

All supported input representations are converted to a canonical period key
before matching:

| Frequency | Canonical key | Accepted representations |
|---|---:|---|
| weekly | ISO `YYYYWW` | `pl.Date`, `pl.Datetime`, valid `YYYYWW` integer |
| monthly | `YYYYMM` | `pl.Date`, `pl.Datetime`, valid `YYYYMM` integer |
| daily | `YYYYMMDD` | `pl.Date`, `pl.Datetime`, valid `YYYYMMDD` integer |
| hourly | `YYYYMMDDHH` | `pl.Datetime`, valid `YYYYMMDDHH` integer |

Weekly years are ISO week-years, not Gregorian calendar years. ISO Week 53 is
accepted only when it exists. A weekly origin is W0 and a monthly origin is M0:
the origin is `horizon_step == 0`. A lookback of L contains exactly the L
periods immediately preceding the origin, ordered oldest to newest.

For timezone-aware datetimes, the represented timezone is preserved while the
calendar period is derived; inference does not silently convert to UTC first.
Ambiguous or invalid integer encodings fail validation.

### Series selection

- `series_ids is None` selects all available series in canonical string order.
- An empty `series_ids` sequence raises `ValueError`.
- Duplicate requested IDs are de-duplicated by first occurrence.
- Explicit selections preserve request order.
- Unknown IDs raise by default.
- `unknown_series_policy="ignore"` permits partial selection, but a selection
  with no known IDs still raises.

These rules prevent an empty or stale execution plan from being mistaken for a
successful all-series forecast.

### Missing values

The existing `fill_missing="ffill"` default is unchanged. A Consumer that
requires zero filling must explicitly pass `fill_missing="zero"` through the
public exogenous/data configuration.

### Result contract and ordering

`ForecastResult.predictions` is a Polars DataFrame with this minimum schema:

| Column | Type | Meaning |
|---|---|---|
| `series_id` | `pl.String` | Generic series identifier |
| `model_key` | `pl.String` | Validated checkpoint artifact key |
| `forecast_origin` | `pl.Int64` | Canonical frequency-specific origin |
| `horizon_step` | `pl.Int32` | Zero-based forecast step |
| `point` | `pl.Float64` | Point forecast |
| `q10` | `pl.Float64` | Nullable 10th percentile |
| `q50` | `pl.Float64` | Nullable median |
| `q90` | `pl.Float64` | Nullable 90th percentile |

Point-only models retain nullable quantile columns. If a quantile model has no
separate point output, q50 is used as point. Final rows are ordered by the
resolved series ordinal and then `horizon_step`. Batch boundaries must not
affect row identity or order.

### Implementation ownership

The lowercase modular DataModule becomes the authority after its capability
gaps are closed and characterized. Dataset and collate logic stay in focused
lowercase modules. The capitalized legacy module becomes a thin compatibility
re-export. Concrete DataModule and Dataset classes remain private and are not
exported from the stable API.

Existing `load_predictor()`, `predict()`, `build_dataset()`, and
`build_dataloader()` remain available. If `forecast_to_parquet()` must remain,
it becomes a compatibility adapter around the result-returning inference path;
it is not the new public contract.

## Alternatives considered

### Keep the capitalized monolith authoritative and add only a facade

This minimizes immediate movement but retains duplicated Dataset/collate logic,
increases change amplification for temporal fixes, and leaves module ownership
unclear. Rejected as the long-term direction.

### Expose the DataModule publicly

This would require Consumers to understand batch tuples, collate behavior,
checkpoint/model coupling, and ordering rules. It would freeze implementation
details into the compatibility surface. Rejected.

### Copy the historical Consumer implementation

The historical implementation combines DSI terminology, persistence policy,
model builders, and local filtering. Copying it would reproduce the boundary
violation and known temporal weaknesses. Rejected.

## Consequences

- Consumers receive one generic request/result integration point.
- DSI-specific names such as `KH`, `oper_part_no`, and `part_ids` do not enter
  the public API.
- The library carries explicit validation and deterministic ordering work.
- Nullable quantile columns slightly widen point-only results but make schemas
  stable across model families.
- The modularization step requires regression coverage for features currently
  present only in the capitalized implementation.

## Validation

Before implementation is accepted, tests must cover public imports, `pl.Date`
weekly windows, ISO year boundaries and Week 53, monthly Date/integer
equivalence, series selection policies, loader forwarding, point/quantile
schemas, zero-based horizons, batch-size-independent ordering, existing
inference helpers, and the complete existing public API suite.

## Non-goals

- Reading from or writing to a database
- Loading Consumer `.env` files
- Discovering DSI checkpoint or data paths
- Defining Parquet naming, partitioning, or retention policy
- Modifying `DSIODemandEngine` in this decision unit

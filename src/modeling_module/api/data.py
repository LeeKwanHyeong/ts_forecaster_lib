from __future__ import annotations

from dataclasses import MISSING, dataclass, fields, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence

import polars as pl

from modeling_module._internal.data_runtime import (
    ExogenousBatch,
    ExogenousFeatureSchema,
    MultiPartDataModule,
    MultiPartExoDataModule,
)


def _is_default_value(value: Any, default_value: Any) -> bool:
    if default_value is MISSING:
        return False
    if default_value is None:
        return value is None
    try:
        return value == default_value
    except Exception:
        return False


@dataclass
class DataWindowConfig:
    """
    Rolling window definition used by the public data API.

    Parameters
    - `lookback`: Number of historical steps fed to the model.
    - `horizon`: Number of future steps predicted by the model.
    - `freq`: One of `weekly`, `monthly`, `daily`, `hourly`.
    """
    lookback: Optional[int] = None
    horizon: Optional[int] = None
    freq: str = "weekly"


@dataclass
class DataColumnConfig:
    """
    Column names for long-table inputs.

    Parameters
    - `id_col`: Series identifier column.
    - `date_col`: Time index column.
    - `y_col`: Target column.
    """
    id_col: str = "unique_id"
    date_col: str = "date"
    y_col: str = "y"


@dataclass
class ExogenousConfig:
    """
    Exogenous feature configuration for one-table or callback-based setups.

    Parameters
    - `use_exogenous_mode`: Enable exogenous-aware model / loader paths.
    - `use_past_exogenous`: Whether training should consume historical exogenous inputs.
    - `use_future_exogenous`: Whether training should consume future exogenous inputs.
    - `past_exo_cont_cols`: Continuous covariates sliced from the lookback window.
    - `past_exo_cat_cols`: Categorical covariates sliced from the lookback window.
    - `future_exo_cont_cols`: Known future covariates sliced from the horizon window.
    - `fill_missing`: Missing-value policy forwarded to the exogenous datamodule.
    - `target_back_steps`: Backward steps used by categorical indexer helpers.
    - `future_exo_cb`: Optional callback that generates future exogenous tensors.
    - `part_future_exo_fn`: Optional batch callback that builds part-specific future exogenous tensors
      from `(uid_list, start_idxs, horizon)`.
    - `date_indexer`: Optional external date indexer used by callback-based workflows.
    - `build_cat_indexer_from`: Source columns for categorical index construction.
    - `cat_indexer_target_col`: Target column for categorical index construction.
    """
    use_exogenous_mode: Optional[bool] = None
    use_past_exogenous: Optional[bool] = None
    use_future_exogenous: Optional[bool] = None
    past_exo_cont_cols: Optional[list[str]] = None
    past_exo_cat_cols: Optional[list[str]] = None
    future_exo_cont_cols: Optional[list[str]] = None
    fill_missing: str = "ffill"
    target_back_steps: int = 100
    future_exo_cb: Any = None
    part_future_exo_fn: Any = None
    date_indexer: Any = None
    build_cat_indexer_from: Optional[list[str]] = None
    cat_indexer_target_col: Optional[str] = None


@dataclass
class LoaderConfig:
    """
    Dataloader controls used by `build_dataset(...)` and `build_dataloader(...)`.

    Parameters
    - `batch_size`: Batch size for train/val/inference loaders.
    - `val_ratio`: Validation split ratio when the datamodule creates the split internally.
    - `shuffle`: Shuffle training batches.
    - `seed`: Random seed for split and shuffling helpers.
    - `stage`: One of `train`, `val`, `infer` / `inference` / `predict`.
    - `plan_dt`: Anchor date for inference dataset construction on the exogenous backend.
    - `split_mode`: Internal split mode for the exogenous backend.
    - `is_running`: Weekly/monthly toggle used by the simple backend.
    - `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`, `drop_last`:
      forwarded to the underlying PyTorch dataloader where supported.
    """
    batch_size: int = 64
    val_ratio: float = 0.2
    shuffle: bool = True
    seed: int = 42
    stage: str = "train"
    plan_dt: Optional[date | datetime | int] = None
    split_mode: str = "window"
    is_running: Optional[bool] = None
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: Optional[bool] = None


@dataclass
class DataRequest:
    """
    Preferred public request object for `build_dataset(...)` and `build_dataloader(...)`.

    Recommended usage is to provide nested configs:
    - `window`: `DataWindowConfig`
    - `columns`: `DataColumnConfig`
    - `exogenous`: `ExogenousConfig`
    - `loader`: `LoaderConfig`

    Flat aliases are still accepted for backward compatibility, but new code should prefer
    the nested dataclass-style configuration.
    """
    data: Any = None
    df: Any = None
    path: Optional[str] = None

    window: Optional[DataWindowConfig | Mapping[str, Any]] = None
    columns: Optional[DataColumnConfig | Mapping[str, Any]] = None
    exogenous: Optional[ExogenousConfig | Mapping[str, Any]] = None
    loader: Optional[LoaderConfig | Mapping[str, Any]] = None

    lookback: Optional[int] = None
    horizon: Optional[int] = None
    freq: str = "weekly"
    batch_size: int = 64
    val_ratio: float = 0.2
    shuffle: bool = True
    seed: int = 42

    use_exogenous_mode: bool = False
    use_past_exogenous: Optional[bool] = None
    use_future_exogenous: Optional[bool] = None
    backend: Optional[str] = None
    stage: str = "train"
    plan_dt: Optional[date | datetime | int] = None
    series_ids: Optional[Sequence[str]] = None
    unknown_series_policy: Literal["error", "ignore"] = "error"

    id_col: str = "unique_id"
    date_col: str = "date"
    y_col: str = "y"
    past_exo_cont_cols: Optional[list[str]] = None
    past_exo_cat_cols: Optional[list[str]] = None
    future_exo_cont_cols: Optional[list[str]] = None
    fill_missing: str = "ffill"
    target_back_steps: int = 100
    future_exo_cb: Any = None
    part_future_exo_fn: Any = None
    date_indexer: Any = None
    build_cat_indexer_from: Optional[list[str]] = None
    cat_indexer_target_col: Optional[str] = None
    split_mode: str = "window"

    is_running: Optional[bool] = None
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: Optional[bool] = None


def _request_to_dict(cfg: DataRequest | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, DataRequest):
        out: dict[str, Any] = {}
        for field in fields(DataRequest):
            value = getattr(cfg, field.name)
            has_default = field.default is not MISSING
            default_value = field.default if has_default else MISSING
            if value is not None and not _is_default_value(value, default_value):
                out[field.name] = value
        return out

    if isinstance(cfg, Mapping):
        return dict(cfg)

    raise TypeError(f"Unsupported data request type: {type(cfg)}")


def _coerce_mapping(value: Any) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if is_dataclass(value):
        out: dict[str, Any] = {}
        for field_ in fields(value):
            item = getattr(value, field_.name)
            has_default = field_.default is not MISSING
            default_value = field_.default if has_default else MISSING
            if item is not None and not _is_default_value(item, default_value):
                out[field_.name] = item
        return out
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _merge_section(
    payload: dict[str, Any],
    section_name: str,
    key_mapping: Mapping[str, str],
) -> None:
    section = _coerce_mapping(payload.get(section_name))
    if not section:
        return

    for raw_key, value in section.items():
        target_key = key_mapping.get(raw_key, raw_key)
        payload.setdefault(target_key, value)


def _materialize_payload(cfg: DataRequest | Mapping[str, Any]) -> dict[str, Any]:
    payload = _request_to_dict(cfg)

    nested = payload.get("data")
    if isinstance(nested, Mapping):
        for key, value in dict(nested).items():
            payload.setdefault(key, value)
    elif is_dataclass(nested):
        for key, value in (_coerce_mapping(nested) or {}).items():
            payload.setdefault(key, value)
    elif nested is not None:
        payload.setdefault("df", nested)

    _merge_section(
        payload,
        "window",
        {
            "lookback": "lookback",
            "horizon": "horizon",
            "freq": "freq",
        },
    )
    _merge_section(
        payload,
        "columns",
        {
            "id": "id_col",
            "date": "date_col",
            "target": "y_col",
            "id_col": "id_col",
            "date_col": "date_col",
            "y_col": "y_col",
        },
    )
    _merge_section(
        payload,
        "exogenous",
        {
            "use_mode": "use_exogenous_mode",
            "use_exogenous_mode": "use_exogenous_mode",
            "use_past_exogenous": "use_past_exogenous",
            "use_future_exogenous": "use_future_exogenous",
            "past_cont": "past_exo_cont_cols",
            "past_cat": "past_exo_cat_cols",
            "future_cont": "future_exo_cont_cols",
            "past_exo_cont_cols": "past_exo_cont_cols",
            "past_exo_cat_cols": "past_exo_cat_cols",
            "future_exo_cont_cols": "future_exo_cont_cols",
            "fill_missing": "fill_missing",
            "target_back_steps": "target_back_steps",
            "future_exo_cb": "future_exo_cb",
            "part_future_exo_fn": "part_future_exo_fn",
            "date_indexer": "date_indexer",
            "build_cat_indexer_from": "build_cat_indexer_from",
            "cat_indexer_target_col": "cat_indexer_target_col",
        },
    )
    _merge_section(
        payload,
        "loader",
        {
            "batch_size": "batch_size",
            "val_ratio": "val_ratio",
            "shuffle": "shuffle",
            "seed": "seed",
            "stage": "stage",
            "plan_dt": "plan_dt",
            "split_mode": "split_mode",
            "is_running": "is_running",
            "num_workers": "num_workers",
            "pin_memory": "pin_memory",
            "persistent_workers": "persistent_workers",
            "prefetch_factor": "prefetch_factor",
            "drop_last": "drop_last",
        },
    )

    return payload


def _load_dataframe(payload: Mapping[str, Any]) -> pl.DataFrame:
    df_like = payload.get("df")
    if df_like is None:
        df_like = payload.get("data")
    if df_like is None:
        df_like = payload.get("path")

    if isinstance(df_like, pl.DataFrame):
        return df_like

    if isinstance(df_like, pl.LazyFrame):
        return df_like.collect()

    if isinstance(df_like, (str, Path)):
        path = Path(df_like)
        suffix = path.suffix.lower()

        if suffix == ".parquet":
            return pl.read_parquet(path)
        if suffix in {".csv", ".txt"}:
            return pl.read_csv(path)

        raise ValueError(f"Unsupported data file format: {path}")

    raise ValueError("`df`, `data`, or `path` must provide a Polars DataFrame or a parquet/csv path.")


def _normalize_backend(payload: Mapping[str, Any]) -> str:
    requested = str(payload.get("backend") or "").strip().lower()
    if requested:
        if requested not in {"exo", "simple"}:
            raise ValueError("backend must be one of {'exo', 'simple'}")
        return requested

    # library API에서는 exo backend를 기본값으로 사용한다.
    # 이유:
    # - daily/hourly 지원
    # - custom id/date/y column 지원
    # - exogenous OFF인 경우도 자연스럽게 동작
    return "exo"


def _build_exogenous_schema_from_payload(
    payload: Mapping[str, Any],
    *,
    df: Optional[pl.DataFrame] = None,
) -> ExogenousFeatureSchema:
    schema = ExogenousFeatureSchema.from_columns(
        past_cont=payload.get("past_exo_cont_cols"),
        past_cat=payload.get("past_exo_cat_cols"),
        future_cont=payload.get("future_exo_cont_cols"),
    )
    if df is not None:
        requested = set(schema.past_cont_names)
        requested.update(schema.past_cat_names)
        requested.update(schema.future_cont_names)
        missing = sorted(requested.difference(df.columns))
        if missing:
            raise ValueError(
                "Exogenous schema references missing dataframe columns: "
                + ", ".join(missing)
            )
    return schema


def build_exogenous_schema(
    cfg: DataRequest | Mapping[str, Any],
) -> ExogenousFeatureSchema:
    """Resolve and validate the ordered exogenous feature schema for a data request."""
    payload = _materialize_payload(cfg)
    df = _load_dataframe(payload)
    return _build_exogenous_schema_from_payload(payload, df=df)


def build_datamodule(cfg: DataRequest | Mapping[str, Any]) -> Any:
    """
    Build the internal datamodule used by the library data API.

    The preferred input is `DataRequest(...)`, although mappings are still accepted for
    backward compatibility.
    """
    payload = _materialize_payload(cfg)
    df = _load_dataframe(payload)
    exogenous_schema = _build_exogenous_schema_from_payload(payload, df=df)

    lookback = payload.get("lookback")
    horizon = payload.get("horizon")
    if lookback is None or horizon is None:
        raise ValueError("Both `lookback` and `horizon` are required to build data.")

    backend = _normalize_backend(payload)

    if backend == "simple":
        is_running = payload.get("is_running")
        if is_running is None:
            freq = str(payload.get("freq", "weekly")).strip().lower()
            if freq == "weekly":
                is_running = True
            elif freq == "monthly":
                is_running = False
            else:
                raise ValueError("simple backend supports only weekly/monthly. Use backend='exo' for daily/hourly.")

        datamodule = MultiPartDataModule(
            df=df,
            lookback=int(lookback),
            horizon=int(horizon),
            is_running=bool(is_running),
            batch_size=int(payload.get("batch_size", 64)),
            val_ratio=float(payload.get("val_ratio", 0.2)),
            shuffle=bool(payload.get("shuffle", True)),
            seed=int(payload.get("seed", 42)),
        )
        datamodule.exogenous_schema = exogenous_schema
        return datamodule

    datamodule = MultiPartExoDataModule(
        df=df,
        lookback=int(lookback),
        horizon=int(horizon),
        freq=str(payload.get("freq", "weekly")).strip().lower(),
        batch_size=int(payload.get("batch_size", 64)),
        val_ratio=float(payload.get("val_ratio", 0.2)),
        shuffle=bool(payload.get("shuffle", True)),
        seed=int(payload.get("seed", 42)),
        id_col=str(payload.get("id_col", "unique_id")),
        date_col=str(payload.get("date_col", "date")),
        y_col=str(payload.get("y_col", "y")),
        past_exo_cont_cols=payload.get("past_exo_cont_cols"),
        past_exo_cat_cols=payload.get("past_exo_cat_cols"),
        future_exo_cont_cols=payload.get("future_exo_cont_cols"),
        fill_missing=str(payload.get("fill_missing", "ffill")),
        target_back_steps=int(payload.get("target_back_steps", 100)),
        future_exo_cb=payload.get("future_exo_cb"),
        part_future_exo_fn=payload.get("part_future_exo_fn"),
        date_indexer=payload.get("date_indexer"),
        build_cat_indexer_from=payload.get("build_cat_indexer_from"),
        cat_indexer_target_col=payload.get("cat_indexer_target_col"),
        split_mode=str(payload.get("split_mode", "window")),
    )
    datamodule.exogenous_schema = exogenous_schema
    return datamodule


def build_dataset(cfg: DataRequest | Mapping[str, Any]) -> Any:
    """
    Build a dataset for `train`, `val`, or `inference` stage inspection.

    This is mainly intended for manual checks, notebooks, and debugging before full training.
    """
    payload = _materialize_payload(cfg)
    stage = str(payload.get("stage", "train")).strip().lower()
    datamodule = build_datamodule(payload)

    if stage == "train":
        datamodule.setup()
        return datamodule.train_dataset

    if stage == "val":
        datamodule.setup()
        return datamodule.val_dataset

    if stage in {"infer", "inference", "predict"}:
        plan_dt = payload.get("plan_dt")
        if plan_dt is not None:
            return build_dataloader(payload).dataset
        if hasattr(datamodule, "get_inference_loader"):
            return datamodule.get_inference_loader().dataset
        raise ValueError("`plan_dt` is required for inference dataset with the exo backend.")

    raise ValueError(f"Unsupported stage={stage!r}. Use one of train/val/inference.")


def build_dataloader(cfg: DataRequest | Mapping[str, Any]) -> Any:
    """
    Build a public dataloader from a `DataRequest` or compatible mapping.

    This is the recommended way to inspect batch shapes or supply prebuilt loaders to `train(...)`.
    """
    payload = _materialize_payload(cfg)
    stage = str(payload.get("stage", "train")).strip().lower()
    datamodule = build_datamodule(payload)

    loader_kwargs = {
        "batch_size": payload.get("batch_size"),
        "drop_last": payload.get("drop_last"),
        "num_workers": int(payload.get("num_workers", 0)),
        "pin_memory": bool(payload.get("pin_memory", True)),
        "persistent_workers": bool(payload.get("persistent_workers", True)),
        "prefetch_factor": int(payload.get("prefetch_factor", 2)),
    }
    loader_kwargs = {k: v for k, v in loader_kwargs.items() if v is not None}

    if stage == "train":
        if isinstance(datamodule, MultiPartExoDataModule):
            loader_kwargs["shuffle"] = bool(payload.get("shuffle", True))
            return datamodule.get_train_loader(**loader_kwargs)
        return datamodule.get_train_loader()

    if stage == "val":
        if isinstance(datamodule, MultiPartExoDataModule):
            return datamodule.get_val_loader(**loader_kwargs)
        return datamodule.get_val_loader()

    if stage in {"infer", "inference", "predict"}:
        plan_dt = payload.get("plan_dt")
        if plan_dt is not None:
            if isinstance(datamodule, MultiPartExoDataModule):
                return datamodule.get_inference_loader_at_plan(
                    plan_dt,
                    series_ids=payload.get("series_ids"),
                    unknown_series_policy=str(payload.get("unknown_series_policy", "error")),
                    **loader_kwargs,
                )
            return datamodule.get_inference_loader_at_plan(int(plan_dt))
        if hasattr(datamodule, "get_inference_loader"):
            return datamodule.get_inference_loader()
        raise ValueError("`plan_dt` is required for inference dataloader with the exo backend.")

    raise ValueError(f"Unsupported stage={stage!r}. Use one of train/val/inference.")


__all__ = [
    "DataColumnConfig",
    "DataRequest",
    "DataWindowConfig",
    "ExogenousBatch",
    "ExogenousConfig",
    "ExogenousFeatureSchema",
    "LoaderConfig",
    "build_dataloader",
    "build_dataset",
    "build_exogenous_schema",
]

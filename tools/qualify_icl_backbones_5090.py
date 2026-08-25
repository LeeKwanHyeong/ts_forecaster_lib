#!/usr/bin/env python3
"""Qualify real AutoTimes and SELLM backbones on sealed H26/H27 ICL episodes."""

from __future__ import annotations

import argparse
import calendar
import gc
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
import torch


ROOT: Final = Path(__file__).resolve().parents[1]
SRC_ROOT: Final = ROOT / "src"
for path in (ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from modeling_module.api.icl import (  # noqa: E402
    ICLForecastRequest,
    ICLForecastRuntimeConfig,
    forecast_icl,
)
from modeling_module.data_loader import ICLEpisodeDataModule  # noqa: E402
from modeling_module.icl import (  # noqa: E402
    EndogenousICLBuilderConfig,
    ExogenousICLBuilderConfig,
    ExogenousICLDatasetBuilder,
    ICLSplit,
    ICLTrainerConfig,
    write_icl_episode_artifact,
)
from modeling_module.models.AutoTimes import AutoTimesConfig, AutoTimesModel  # noqa: E402
from modeling_module.models.SELLM.SELLM import SELLMModel  # noqa: E402
from modeling_module.models.SELLM.configs import SELLMConfig  # noqa: E402
from modeling_module.training.model_trainers.autotimes_train import (  # noqa: E402
    train_autotimes_icl,
)
from modeling_module.training.model_trainers.sellm_train import (  # noqa: E402
    train_sellm_icl,
)
from modeling_module.utils.checkpoint import save_model  # noqa: E402


RECEIPT_CONTRACT: Final = "modeling_module.icl_backbone_qualification.v2"
CALENDAR_SOURCE_REVISION: Final = "deterministic-iso-calendar-v1"
OPERATION_PART_SNAPSHOT_CONTRACT: Final = "demand-engine-operation-part-snapshot-v1"
OPERATION_PART_SNAPSHOT_COLUMNS: Final = (
    "site_cd",
    "oper_part_no",
    "demand_start_dt",
    "demand_end_dt",
    "warranty",
)
APPROVED_EXOGENOUS_FEATURES: Final = (
    "sin_annual",
    "cos_annual",
    "sin_semi",
    "cos_semi",
    "sin_quarter",
    "cos_quarter",
    "week_of_year_norm",
    "peak_season_flag",
    "is_year_start",
    "is_year_end",
    "is_q_start",
    "is_q_end",
    "lifecycle_pre_launch_flag",
    "lifecycle_active_flag",
    "lifecycle_service_ended_flag",
    "lifecycle_age_years",
    "lifecycle_remaining_years",
    "post_lifecycle_years",
    "warranty_years",
    "warranty_active_flag",
    "weeks_to_warranty_end_years",
    "weeks_since_warranty_end_years",
    "lifecycle_source_observed_flag",
)
MODEL_KEYS: Final = ("autotimes_base", "sellm_base")


class QualificationError(RuntimeError):
    """Raised when an input or result violates the qualification contract."""


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit(explicit: str | None = None) -> str:
    value = str(explicit or "").strip()
    if value:
        if len(value) < 7 or any(character not in "0123456789abcdef" for character in value):
            raise QualificationError("source-commit must be one lowercase Git SHA.")
        return value
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise QualificationError(
            "A source commit is required when the qualification package has no .git directory."
        ) from exc


def _code_contract_sha256() -> str:
    roots = (
        SRC_ROOT / "modeling_module" / "icl",
        SRC_ROOT / "modeling_module" / "models" / "AutoTimes",
    )
    explicit = (
        SRC_ROOT / "modeling_module" / "api" / "icl.py",
        SRC_ROOT / "modeling_module" / "data_loader" / "icl_episode_data_module.py",
        SRC_ROOT / "modeling_module" / "models" / "SELLM" / "SELLM.py",
        SRC_ROOT / "modeling_module" / "models" / "SELLM" / "backbone.py",
        SRC_ROOT / "modeling_module" / "models" / "SELLM" / "configs.py",
        SRC_ROOT / "modeling_module" / "training" / "model_trainers" / "autotimes_train.py",
        SRC_ROOT / "modeling_module" / "training" / "model_trainers" / "sellm_train.py",
        Path(__file__).resolve(),
    )
    files = list(explicit)
    for root in roots:
        files.extend(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    payload = [
        {
            "path": str(path.relative_to(ROOT)),
            "sha256": _file_sha256(path),
        }
        for path in sorted(set(files))
    ]
    return _sha256_payload(payload)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_source_contract(manifest_path: Path, target_path: Path) -> dict[str, Any]:
    envelope = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise QualificationError("Input manifest payload is missing.")
    if envelope.get("payload_sha256") != _sha256_payload(payload):
        raise QualificationError("Input manifest seal mismatch.")
    artifact = payload.get("artifact") or {}
    observed_sha = _file_sha256(target_path)
    if artifact.get("sha256") != observed_sha:
        raise QualificationError("Target Parquet SHA256 differs from the input manifest.")
    source_revision = str(payload.get("source_bundle_sha256") or "").strip()
    if len(source_revision) != 64:
        raise QualificationError("Input manifest has no governed source bundle SHA256.")
    return {
        "source_revision": source_revision,
        "target_sha256": observed_sha,
        "manifest_sha256": _file_sha256(manifest_path),
        "source_max_week": int((payload.get("dataset") or {}).get("maximum_week")),
        "site_cd": str((payload.get("training_contract") or {}).get("site_cd") or ""),
    }


def _load_operation_part_source(
    manifest_path: Path,
    snapshot_path: Path,
    *,
    expected_site_cd: str,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise QualificationError("Operation Part manifest must be one JSON object.")
    manifest = dict(raw)
    seal = manifest.pop("manifest_sha256", None)
    if seal != _sha256_payload(manifest):
        raise QualificationError("Operation Part manifest seal mismatch.")
    if manifest.get("contract_id") != OPERATION_PART_SNAPSHOT_CONTRACT:
        raise QualificationError("Operation Part snapshot contract is unsupported.")
    if manifest.get("contract_version") != "1.0.0":
        raise QualificationError("Operation Part snapshot version is unsupported.")
    scope = manifest.get("scope")
    if not isinstance(scope, dict) or str(scope.get("site_cd") or "") != expected_site_cd:
        raise QualificationError("Operation Part scope differs from the demand artifact.")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        raise QualificationError("Operation Part artifact contract is missing.")
    if artifact.get("columns") != list(OPERATION_PART_SNAPSHOT_COLUMNS):
        raise QualificationError("Operation Part snapshot columns differ from the contract.")

    frame = (
        pl.read_parquet(snapshot_path, columns=list(OPERATION_PART_SNAPSHOT_COLUMNS))
        .select(
            pl.col("site_cd").cast(pl.String),
            pl.col("oper_part_no").cast(pl.String),
            pl.col("demand_start_dt").cast(pl.Int64, strict=False),
            pl.col("demand_end_dt").cast(pl.Int64, strict=False),
            pl.col("warranty").cast(pl.Int16, strict=False),
        )
        .sort("site_cd", "oper_part_no")
    )
    if frame.height != int(artifact.get("row_count") or 0):
        raise QualificationError("Operation Part row count differs from the manifest.")
    if frame.get_column("oper_part_no").n_unique() != int(artifact.get("part_count") or 0):
        raise QualificationError("Operation Part part count differs from the manifest.")
    if frame.select("site_cd", "oper_part_no").unique().height != frame.height:
        raise QualificationError("Operation Part snapshot contains duplicate identities.")
    if sum(frame.null_count().row(0)):
        raise QualificationError("Operation Part snapshot contains null source values.")
    if set(frame.get_column("site_cd").unique().to_list()) != {expected_site_cd}:
        raise QualificationError("Operation Part rows differ from the approved Site.")
    content_sha256 = _sha256_payload(
        frame.select(OPERATION_PART_SNAPSHOT_COLUMNS).to_dicts()
    )
    if content_sha256 != artifact.get("content_sha256"):
        raise QualificationError("Operation Part content hash differs from the manifest.")
    invalid_warranty = frame.filter(~pl.col("warranty").is_in([12, 24, 36, 48]))
    if not invalid_warranty.is_empty():
        raise QualificationError("Operation Part warranty is outside the approved scale.")
    for value in frame.select("demand_start_dt", "demand_end_dt").iter_columns():
        for week in value.to_list():
            _iso_week_monday(int(week))
    return frame, {
        "source_id": str(manifest.get("source_id") or ""),
        "source_revision": str(manifest.get("source_revision") or ""),
        "source_manifest_sha256": str(seal),
        "feature_schema_version": str(manifest.get("feature_schema_version") or ""),
        "snapshot_content_sha256": content_sha256,
        "snapshot_file_sha256": _file_sha256(snapshot_path),
        "row_count": frame.height,
        "part_count": frame.get_column("oper_part_no").n_unique(),
        "scope": scope,
    }


def _load_backbone_contract(llm_local_path: Path) -> dict[str, Any]:
    required = ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")
    files: dict[str, dict[str, Any]] = {}
    for name in required:
        path = llm_local_path / name
        if not path.is_file():
            raise QualificationError(f"Backbone file is missing: {name}.")
        files[name] = {"sha256": _file_sha256(path), "size_bytes": path.stat().st_size}
    config = json.loads((llm_local_path / "config.json").read_text(encoding="utf-8"))
    return {
        "model_type": str(config.get("model_type") or ""),
        "hidden_size": int(config.get("hidden_size") or 0),
        "frozen": True,
        "files": files,
        "contract_sha256": _sha256_payload(files),
    }


def _is_contiguous(weeks: list[int]) -> bool:
    ordinals = [date.fromisocalendar(value // 100, value % 100, 1).toordinal() for value in weeks]
    return all(current - previous == 7 for previous, current in zip(ordinals, ordinals[1:]))


def _iso_week_monday(value: int) -> date:
    try:
        return date.fromisocalendar(int(value) // 100, int(value) % 100, 1)
    except (TypeError, ValueError) as exc:
        raise QualificationError(f"Invalid approved ISO week: {value!r}.") from exc


def _add_months(value: date, months: int) -> date:
    total_months = value.year * 12 + value.month - 1 + int(months)
    year, zero_based_month = divmod(total_months, 12)
    month = zero_based_month + 1
    return date(year, month, min(value.day, calendar.monthrange(year, month)[1]))


def _select_series(frame: pl.DataFrame, *, count: int, minimum_rows: int) -> pl.DataFrame:
    weekly = (
        frame.select("oper_part_no", "demand_dt", "demand_qty")
        .group_by("oper_part_no", "demand_dt")
        .agg(pl.col("demand_qty").sum())
        .sort("oper_part_no", "demand_dt")
    )
    candidates = (
        weekly.group_by("oper_part_no")
        .agg(pl.len().alias("row_count"))
        .filter(pl.col("row_count") >= int(minimum_rows))
        .sort(["row_count", "oper_part_no"], descending=[True, False])
    )
    selected: list[str] = []
    for part_no in candidates["oper_part_no"].to_list():
        weeks = weekly.filter(pl.col("oper_part_no") == part_no)["demand_dt"].to_list()
        if _is_contiguous([int(value) for value in weeks]):
            selected.append(str(part_no))
        if len(selected) == int(count):
            break
    if len(selected) != int(count):
        raise QualificationError(
            f"Only {len(selected)} series satisfy the continuous-history contract; "
            f"required={count}."
        )
    selected_weekly = weekly.filter(pl.col("oper_part_no").is_in(selected))
    boundaries = selected_weekly.group_by("oper_part_no").agg(
        pl.col("demand_dt").min().alias("start_week"),
        pl.col("demand_dt").max().alias("end_week"),
    )
    common_start = int(boundaries["start_week"].max())
    common_end = int(boundaries["end_week"].min())
    aligned = selected_weekly.filter(
        pl.col("demand_dt").is_between(common_start, common_end, closed="both")
    ).sort("oper_part_no", "demand_dt")
    aligned_counts = aligned.group_by("oper_part_no").agg(pl.len().alias("row_count"))
    if (
        aligned_counts.height != int(count)
        or int(aligned_counts["row_count"].min()) < int(minimum_rows)
        or int(aligned_counts["row_count"].n_unique()) != 1
    ):
        raise QualificationError(
            "Selected series do not share enough aligned continuous history for the "
            f"global temporal split; required_rows={minimum_rows}."
        )
    return aligned


def _minimum_contiguous_rows(
    *,
    horizon: int,
    stride: int,
    lookback: int = 52,
    seasonal_period: int = 52,
    validation_episodes: int = 1,
    test_episodes: int = 1,
) -> int:
    """Return the shortest history that yields non-overlapping train/val/test targets."""

    if min(horizon, stride, lookback, seasonal_period) <= 0:
        raise ValueError("ICL history dimensions must be positive.")
    if validation_episodes < 0 or test_episodes < 0:
        raise ValueError("ICL holdout episode counts must be non-negative.")
    total = int(lookback) + int(horizon)
    query_start = 0
    while True:
        historical_start = query_start - total
        if historical_start >= 0:
            seasonal_target_start = query_start + int(lookback) - int(seasonal_period)
            while seasonal_target_start + int(horizon) > historical_start:
                seasonal_target_start -= int(seasonal_period)
            if seasonal_target_start - int(lookback) >= 0:
                break
        query_start += int(stride)

    non_overlapping_gap = math.ceil(int(horizon) / int(stride)) * int(stride)
    latest_query_start = query_start + (
        int(validation_episodes) + int(test_episodes)
    ) * non_overlapping_gap
    return latest_query_start + total


def _add_approved_exogenous_features(
    frame: pl.DataFrame,
    operation_parts: pl.DataFrame,
) -> pl.DataFrame:
    joined = frame.join(
        operation_parts.select(
            "oper_part_no", "demand_start_dt", "demand_end_dt", "warranty"
        ),
        on="oper_part_no",
        how="left",
        validate="m:1",
    )
    if sum(
        joined.select("demand_start_dt", "demand_end_dt", "warranty")
        .null_count()
        .row(0)
    ):
        raise QualificationError("Selected demand series lack approved lifecycle coverage.")
    feature_rows: list[dict[str, float]] = []
    for row in joined.iter_rows(named=True):
        current = _iso_week_monday(int(row["demand_dt"]))
        start = _iso_week_monday(int(row["demand_start_dt"]))
        end = _iso_week_monday(int(row["demand_end_dt"]))
        if start >= end:
            raise QualificationError("Lifecycle start must precede lifecycle end.")
        warranty_months = int(row["warranty"])
        warranty_end = _add_months(start, warranty_months)
        week = int(current.isocalendar().week)
        month = current.month
        two_pi = 2.0 * math.pi
        before_start = current < start
        before_end = current < end
        before_warranty_end = current < warranty_end
        feature_rows.append(
            {
                "sin_annual": math.sin(two_pi * week / 52.0),
                "cos_annual": math.cos(two_pi * week / 52.0),
                "sin_semi": math.sin(two_pi * week / 26.0),
                "cos_semi": math.cos(two_pi * week / 26.0),
                "sin_quarter": math.sin(two_pi * week / 13.0),
                "cos_quarter": math.cos(two_pi * week / 13.0),
                "week_of_year_norm": week / 52.0,
                "peak_season_flag": float(month in {11, 12, 1, 2}),
                "is_year_start": float(month <= 2),
                "is_year_end": float(month >= 11),
                "is_q_start": float(month in {1, 4, 7, 10}),
                "is_q_end": float(month in {3, 6, 9, 12}),
                "lifecycle_pre_launch_flag": float(before_start),
                "lifecycle_active_flag": float(not before_start and before_end),
                "lifecycle_service_ended_flag": float(not before_end),
                "lifecycle_age_years": max((current - start).days // 7, 0) / 52.0,
                "lifecycle_remaining_years": max((end - current).days // 7, 0) / 52.0,
                "post_lifecycle_years": max((current - end).days // 7, 0) / 52.0,
                "warranty_years": warranty_months / 12.0,
                "warranty_active_flag": float(not before_start and before_warranty_end),
                "weeks_to_warranty_end_years": (
                    max((warranty_end - current).days // 7, 0) / 52.0
                    if not before_start and before_warranty_end
                    else 0.0
                ),
                "weeks_since_warranty_end_years": (
                    max((current - warranty_end).days // 7, 0) / 52.0
                    if not before_warranty_end
                    else 0.0
                ),
                "lifecycle_source_observed_flag": 1.0,
            }
        )
    return joined.drop("demand_start_dt", "demand_end_dt", "warranty").hstack(
        pl.DataFrame(feature_rows)
    )


def prepare_bundles(
    *,
    target_path: Path,
    source_revision: str,
    output_root: Path,
    horizons: tuple[int, ...],
    sample_series: int,
    stride: int,
    operation_parts: pl.DataFrame,
    exogenous_source_revision: str,
) -> dict[int, Any]:
    frame = pl.read_parquet(
        target_path,
        columns=["oper_part_no", "demand_dt", "demand_qty"],
    )
    minimum_rows = max(
        _minimum_contiguous_rows(
            horizon=int(horizon),
            stride=int(stride),
            validation_episodes=(1 if int(horizon) == 26 else 0),
            test_episodes=1,
        )
        for horizon in horizons
    )
    selected = _select_series(
        frame,
        count=sample_series,
        minimum_rows=minimum_rows,
    )
    selected = _add_approved_exogenous_features(selected, operation_parts)
    bundles: dict[int, Any] = {}
    for horizon in horizons:
        validation_episodes = 1 if int(horizon) == 26 else 0
        builder = ExogenousICLDatasetBuilder(
            ExogenousICLBuilderConfig(
                episode=EndogenousICLBuilderConfig(
                    lookback=52,
                    horizon=int(horizon),
                    window_stride=int(stride),
                    seasonal_period=52,
                    validation_episodes_per_series=validation_episodes,
                    test_episodes_per_series=1,
                ),
                past_feature_cols=APPROVED_EXOGENOUS_FEATURES,
                future_feature_cols=APPROVED_EXOGENOUS_FEATURES,
            )
        )
        bundle = builder.build(
            selected,
            source_revision=source_revision,
            exogenous_source_revision=exogenous_source_revision,
        )
        artifact_dir = output_root / f"h{horizon}" / "episodes"
        write_icl_episode_artifact(bundle, artifact_dir)
        bundles[int(horizon)] = bundle
    return bundles


def _split_target_contract(bundle) -> dict[str, dict[str, int]]:
    ranges: dict[str, dict[str, int]] = {}
    populated: list[tuple[ICLSplit, tuple[Any, ...]]] = []
    for split in ICLSplit:
        episodes = bundle.for_split(split)
        if not episodes:
            continue
        populated.append((split, episodes))
        ranges[split.value] = {
            "episode_count": len(episodes),
            "target_start_week": min(item.query_target.start_week for item in episodes),
            "target_end_week": max(item.query_target.end_week for item in episodes),
        }
    for (left_split, left), (right_split, right) in zip(populated, populated[1:]):
        left_end = max(item.query_target.end_week for item in left)
        right_start = min(item.query_target.start_week for item in right)
        if left_end >= right_start:
            raise QualificationError(
                "ICL split target windows overlap: "
                f"{left_split.value} ends {left_end}, "
                f"{right_split.value} starts {right_start}."
            )
    return ranges


def _model_config(
    model_key: str,
    *,
    horizon: int,
    llm_local_path: Path,
    schema_hash: str,
    past_exogenous_dim: int,
    future_exogenous_dim: int,
):
    common = {
        "lookback": 52,
        "horizon": int(horizon),
        "y_dim": 1,
        "icl_enabled": True,
        "icl_past_exogenous_dim": int(past_exogenous_dim),
        "icl_future_exogenous_dim": int(future_exogenous_dim),
        "icl_exogenous_schema_hash": schema_hash,
        "use_exogenous_mode": False,
        "use_intermittent": False,
    }
    if model_key == "autotimes_base":
        return AutoTimesConfig(
            **common,
            token_len=2,
            backbone_type="gpt2",
            llm_source="local",
            llm_local_path=str(llm_local_path),
            freeze_llm=True,
            mlp_hidden_dim=256,
            mlp_hidden_layers=1,
            dropout=0.0,
            mix_timestamp_embeddings=False,
        )
    if model_key == "sellm_base":
        return SELLMConfig(
            **common,
            architecture_variant="paper_v1",
            token_len=13,
            use_pretrained_llm=True,
            llm_source="local",
            llm_local_path=str(llm_local_path),
            freeze_llm=True,
            use_time_adapter=False,
            semantic_vocab_size=8,
            semantic_top_k=4,
            tscc_latent_dim=8,
            tscc_hidden_dim=64,
            mlp_hidden_dim=256,
            dropout=0.0,
            head_hidden_dim=128,
            use_norm=True,
        )
    raise QualificationError(f"Unsupported qualification model: {model_key}.")


def _build_model(model_key: str, config):
    if model_key == "autotimes_base":
        return AutoTimesModel(config)
    if model_key == "sellm_base":
        return SELLMModel(config)
    raise QualificationError(f"Unsupported qualification model: {model_key}.")


def _fit(model_key: str, model, module: ICLEpisodeDataModule, config: ICLTrainerConfig):
    train_loader = module.loader(ICLSplit.TRAIN, shuffle=False)
    validation_loader = (
        module.loader(ICLSplit.VALIDATION, shuffle=False)
        if module.bundle.for_split(ICLSplit.VALIDATION)
        else None
    )
    if model_key == "autotimes_base":
        return train_autotimes_icl(
            model,
            train_loader,
            validation_loader,
            trainer_config=config,
        )
    return train_sellm_icl(
        model,
        train_loader,
        validation_loader,
        trainer_config=config,
    )


def _accuracy(predictions: pl.DataFrame, bundle) -> dict[str, float | int]:
    actual_rows: list[dict[str, Any]] = []
    for episode in bundle.for_split(ICLSplit.TEST):
        for step, values in enumerate(episode.query_target.target):
            actual_rows.append(
                {
                    "episode_id": episode.episode_id,
                    "horizon_step": step,
                    "actual": float(values[0]),
                }
            )
    joined = predictions.join(
        pl.DataFrame(actual_rows),
        on=["episode_id", "horizon_step"],
        how="inner",
        validate="1:1",
    )
    if joined.height != predictions.height:
        raise QualificationError("Prediction rows do not match sealed query targets.")
    absolute_error = (pl.col("point") - pl.col("actual")).abs()
    totals = joined.select(
        pl.len().alias("points"),
        absolute_error.mean().alias("mae"),
        absolute_error.sum().alias("absolute_error_sum"),
        pl.col("actual").abs().sum().alias("actual_abs_sum"),
    ).row(0, named=True)
    denominator = float(totals["actual_abs_sum"])
    return {
        "points": int(totals["points"]),
        "mae": float(totals["mae"]),
        "wape": float(totals["absolute_error_sum"]) / denominator if denominator else 0.0,
    }


def _parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def qualify_one(
    *,
    model_key: str,
    horizon: int,
    bundle,
    artifact_dir: Path,
    output_dir: Path,
    llm_local_path: Path,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    schema = bundle.manifest.exogenous_schema
    if schema is None:
        raise QualificationError("Qualification requires a sealed exogenous schema.")
    _seed_all(seed)
    model_config = _model_config(
        model_key,
        horizon=horizon,
        llm_local_path=llm_local_path,
        schema_hash=schema.fingerprint,
        past_exogenous_dim=len(schema.past_feature_names),
        future_exogenous_dim=len(schema.future_feature_names),
    )
    load_started = time.perf_counter()
    model = _build_model(model_key, model_config)
    backbone_load_seconds = time.perf_counter() - load_started
    counts = _parameter_counts(model)
    module = ICLEpisodeDataModule(bundle, batch_size=batch_size, seed=seed)

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    training_started = time.perf_counter()
    result = _fit(
        model_key,
        model,
        module,
        ICLTrainerConfig(
            epochs=epochs,
            lr=1e-3,
            weight_decay=0.0,
            device=device,
            max_grad_norm=1.0,
        ),
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_started
    final_train_loss = float(result.final_train_loss)
    best_validation_loss = (
        None
        if result.best_validation_loss is None
        else float(result.best_validation_loss)
    )
    memory = {
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated() / (1024**2)
            if device.startswith("cuda")
            else 0.0
        ),
        "peak_reserved_mib": (
            torch.cuda.max_memory_reserved() / (1024**2)
            if device.startswith("cuda")
            else 0.0
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / f"{model_key}_L52_H{horizon}.pt"
    result.model.to("cpu")
    save_model(
        result.model,
        result.model.cfg,
        str(checkpoint_path),
        extra_meta={
            "model_key": model_key,
            "family_key": model_key.removesuffix("_base"),
            "qualification_contract": RECEIPT_CONTRACT,
            "episode_manifest_hash": bundle.manifest.manifest_hash,
            "random_seed": seed,
        },
    )
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    del result, model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    request = ICLForecastRequest(
        checkpoint_path=checkpoint_path,
        episode_artifact_dir=artifact_dir,
        expected_model_key=model_key,
        runtime=ICLForecastRuntimeConfig(batch_size=batch_size, device=device),
    )
    first = forecast_icl(request)
    second = forecast_icl(request)
    first_points = first.predictions.sort("episode_id", "horizon_step")["point"]
    second_points = second.predictions.sort("episode_id", "horizon_step")["point"]
    max_reload_delta = float((first_points - second_points).abs().max() or 0.0)
    if max_reload_delta > 1e-6:
        raise QualificationError(
            f"Checkpoint reload prediction drifted by {max_reload_delta}."
        )
    return {
        "model_key": model_key,
        "horizon": int(horizon),
        "backbone": {
            "type": "Qwen2-0.5B",
            "local_path": str(llm_local_path),
            "load_seconds": backbone_load_seconds,
            "frozen": True,
        },
        "parameters": counts,
        "training": {
            "epochs": int(epochs),
            "seconds": training_seconds,
            "final_train_loss": final_train_loss,
            "best_validation_loss": best_validation_loss,
            **memory,
        },
        "accuracy": _accuracy(first.predictions, bundle),
        "checkpoint": {
            "filename": checkpoint_path.name,
            "sha256": checkpoint_sha256,
            "reload_max_abs_delta": max_reload_delta,
        },
    }


def run_qualification(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise QualificationError(f"Output root already exists: {output_root}")
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise QualificationError("Real backbone qualification requires CUDA.")
    device_name = torch.cuda.get_device_name(0)
    if args.expected_device and device_name != args.expected_device:
        raise QualificationError(
            f"Expected {args.expected_device!r}, observed {device_name!r}."
        )
    target_path = args.target_source.expanduser().resolve()
    manifest_path = args.input_manifest.expanduser().resolve()
    llm_local_path = args.llm_local_path.expanduser().resolve()
    if not llm_local_path.is_dir():
        raise QualificationError(f"LLM directory is missing: {llm_local_path}")
    source = _load_source_contract(manifest_path, target_path)
    operation_parts, exogenous_source = _load_operation_part_source(
        args.operation_part_manifest.expanduser().resolve(),
        args.operation_part_source.expanduser().resolve(),
        expected_site_cd=source["site_cd"],
    )
    exogenous_source_revision = _sha256_payload(
        {
            "calendar_source_revision": CALENDAR_SOURCE_REVISION,
            "operation_part_source": exogenous_source,
            "feature_names": APPROVED_EXOGENOUS_FEATURES,
        }
    )
    backbone = _load_backbone_contract(llm_local_path)
    output_root.mkdir(parents=True)
    horizons = tuple(int(value) for value in args.horizons)
    if 26 not in horizons:
        raise QualificationError("H26 must be present as the primary operating horizon.")
    bundles = prepare_bundles(
        target_path=target_path,
        source_revision=source["source_revision"],
        output_root=output_root,
        horizons=horizons,
        sample_series=int(args.sample_series),
        stride=int(args.stride),
        operation_parts=operation_parts,
        exogenous_source_revision=exogenous_source_revision,
    )
    results: list[dict[str, Any]] = []
    for horizon in horizons:
        bundle = bundles[horizon]
        for model_key in MODEL_KEYS:
            results.append(
                qualify_one(
                    model_key=model_key,
                    horizon=horizon,
                    bundle=bundle,
                    artifact_dir=output_root / f"h{horizon}" / "episodes",
                    output_dir=output_root / f"h{horizon}" / model_key,
                    llm_local_path=llm_local_path,
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                    seed=int(args.seed),
                    device=str(args.device),
                )
            )
    receipt = {
        "contract": RECEIPT_CONTRACT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _source_commit(args.source_commit),
        "code_contract_sha256": _code_contract_sha256(),
        "device": {
            "name": device_name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "input": source,
        "exogenous_input": {
            **exogenous_source,
            "calendar_source_revision": CALENDAR_SOURCE_REVISION,
            "combined_source_revision": exogenous_source_revision,
            "feature_names": list(APPROVED_EXOGENOUS_FEATURES),
        },
        "backbone": backbone,
        "qualification": {
            "status": "PASS",
            "sample_series": int(args.sample_series),
            "horizons": list(horizons),
            "primary_horizon": 26,
            "diagnostic_horizons": [value for value in horizons if value != 26],
            "lookback": 52,
            "stride": int(args.stride),
            "seed": int(args.seed),
            "exogenous_source_revision": exogenous_source_revision,
        },
        "episodes": {
            str(horizon): {
                "role": "operating" if horizon == 26 else "boundary_diagnostic",
                "validation_enabled": horizon == 26,
                "manifest_hash": bundles[horizon].manifest.manifest_hash,
                "split_counts": dict(bundles[horizon].manifest.split_counts),
                "split_target_ranges": _split_target_contract(bundles[horizon]),
            }
            for horizon in horizons
        },
        "results": results,
    }
    receipt["receipt_sha256"] = _sha256_payload(receipt)
    (output_root / "qualification-receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-source", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--llm-local-path", type=Path, required=True)
    parser.add_argument("--operation-part-source", type=Path, required=True)
    parser.add_argument("--operation-part-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[26, 27])
    parser.add_argument("--sample-series", type=int, default=4)
    parser.add_argument("--stride", type=int, default=26)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-device", default="NVIDIA GeForce RTX 5090")
    parser.add_argument("--source-commit")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if int(args.sample_series) <= 0 or int(args.epochs) <= 0 or int(args.batch_size) <= 0:
        raise QualificationError("sample-series, epochs, and batch-size must be positive.")
    receipt = run_qualification(args)
    print(
        json.dumps(
            {
                "status": "PASS",
                "receipt_sha256": receipt["receipt_sha256"],
                "results": len(receipt["results"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

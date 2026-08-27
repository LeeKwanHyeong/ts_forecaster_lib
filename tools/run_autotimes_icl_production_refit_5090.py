#!/usr/bin/env python3
"""Run and seal the governed AutoTimes ICL L52/H26 production refit."""

from __future__ import annotations

import argparse
import gc
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
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

from modeling_module.api.infer import load_predictor  # noqa: E402
from modeling_module.data_loader import ICLEpisodeDataModule  # noqa: E402
from modeling_module.icl import (  # noqa: E402
    AutoTimesICLAdapter,
    EndogenousICLBuilderConfig,
    ExogenousICLBuilderConfig,
    ExogenousICLDatasetBuilder,
    ICLSplit,
    ICLTrainerConfig,
    SELLMICLAdapter,
    save_icl_production_checkpoint,
    write_icl_episode_artifact,
)
from modeling_module.training.model_trainers.autotimes_train import (  # noqa: E402
    train_autotimes_icl,
)
from modeling_module.training.model_trainers.sellm_train import (  # noqa: E402
    train_sellm_icl,
)
from tools.qualify_icl_backbones_5090 import (  # noqa: E402
    APPROVED_EXOGENOUS_FEATURES,
    CALENDAR_SOURCE_REVISION,
    QualificationError,
    _add_approved_exogenous_features,
    _build_model,
    _file_sha256,
    _is_contiguous,
    _load_backbone_contract,
    _load_operation_part_source,
    _load_source_contract,
    _minimum_contiguous_rows,
    _model_config,
    _sha256_payload,
)


LOOKBACK: Final = 52
HORIZON: Final = 26
TRAIN_END_WEEK: Final = 202509
FORECAST_ORIGIN: Final = 202510
STRIDE: Final = 26
SEED: Final = 42
BATCH_SIZE: Final = 4
EPOCHS: Final = 5
LEARNING_RATE: Final = 1e-3
MODEL_KEY: Final = "autotimes_base"
CHECKPOINT_FILENAME: Final = "weekly_AutoTimesBase_L52_H26.pt"
RECEIPT_CONTRACT: Final = "modeling_module.autotimes_icl_production_refit.v1"


@dataclass(frozen=True)
class ICLProductionRefitPolicy:
    model_key: str
    checkpoint_filename: str
    receipt_contract: str
    batch_size: int
    epochs: int
    learning_rate: float
    semantic_vocab_size: int | None = None


AUTOTIMES_POLICY: Final = ICLProductionRefitPolicy(
    model_key=MODEL_KEY,
    checkpoint_filename=CHECKPOINT_FILENAME,
    receipt_contract=RECEIPT_CONTRACT,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
)


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _complete_training_series(
    frame: pl.DataFrame,
    *,
    cutoff: int = TRAIN_END_WEEK,
    stride: int = STRIDE,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Select every continuous series and align its last target to the cutoff."""

    weekly = (
        frame.filter(pl.col("demand_dt") <= int(cutoff))
        .select("oper_part_no", "demand_dt", "demand_qty")
        .group_by("oper_part_no", "demand_dt")
        .agg(pl.col("demand_qty").sum())
        .sort("oper_part_no", "demand_dt")
    )
    minimum_rows = _minimum_contiguous_rows(
        horizon=HORIZON,
        stride=int(stride),
        lookback=LOOKBACK,
        seasonal_period=52,
        validation_episodes=0,
        test_episodes=0,
    )
    selected: list[pl.DataFrame] = []
    excluded: dict[str, int] = {
        "does_not_reach_cutoff": 0,
        "insufficient_history": 0,
        "non_contiguous": 0,
    }
    trimmed_rows = 0
    for group in weekly.partition_by("oper_part_no", maintain_order=True):
        weeks = [int(value) for value in group["demand_dt"].to_list()]
        if not weeks or weeks[-1] != int(cutoff):
            excluded["does_not_reach_cutoff"] += 1
            continue
        if len(weeks) < minimum_rows:
            excluded["insufficient_history"] += 1
            continue
        if not _is_contiguous(weeks):
            excluded["non_contiguous"] += 1
            continue
        latest_query_start = len(weeks) - (LOOKBACK + HORIZON)
        trim = latest_query_start % int(stride)
        aligned = group.slice(trim)
        if aligned.height < minimum_rows:
            excluded["insufficient_history"] += 1
            continue
        selected.append(aligned)
        trimmed_rows += int(trim)
    if not selected:
        raise QualificationError("No series satisfy the AutoTimes refit contract.")
    output = pl.concat(selected).sort("oper_part_no", "demand_dt")
    return output, {
        "source_series_count": weekly["oper_part_no"].n_unique(),
        "eligible_series_count": output["oper_part_no"].n_unique(),
        "minimum_rows": minimum_rows,
        "alignment_trimmed_rows": trimmed_rows,
        "excluded_series": excluded,
        "minimum_week": int(output["demand_dt"].min()),
        "maximum_week": int(output["demand_dt"].max()),
    }


def _build_training_bundle(
    frame: pl.DataFrame,
    *,
    source_revision: str,
    exogenous_source_revision: str,
):
    builder = ExogenousICLDatasetBuilder(
        ExogenousICLBuilderConfig(
            episode=EndogenousICLBuilderConfig(
                lookback=LOOKBACK,
                horizon=HORIZON,
                window_stride=STRIDE,
                seasonal_period=52,
                validation_episodes_per_series=0,
                test_episodes_per_series=0,
            ),
            past_feature_cols=APPROVED_EXOGENOUS_FEATURES,
            future_feature_cols=APPROVED_EXOGENOUS_FEATURES,
        )
    )
    bundle = builder.build(
        frame,
        source_revision=source_revision,
        exogenous_source_revision=exogenous_source_revision,
    )
    if bundle.manifest.split_counts != {
        "train": len(bundle.episodes),
        "validation": 0,
        "test": 0,
    }:
        raise QualificationError("AutoTimes refit bundle must be train-only.")
    return bundle


def _cuda_measurement() -> tuple[float, dict[str, Any]]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return time.perf_counter(), {}


def _finish_cuda_measurement(started: float) -> dict[str, Any]:
    torch.cuda.synchronize()
    mib = 1024.0 * 1024.0
    return {
        "seconds": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / mib,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / mib,
    }


def _train_model_icl(
    policy: ICLProductionRefitPolicy,
    model,
    module: ICLEpisodeDataModule,
    trainer_config: ICLTrainerConfig,
):
    train_loader = module.loader(ICLSplit.TRAIN, shuffle=False)
    if policy.model_key == "autotimes_base":
        return train_autotimes_icl(
            model,
            train_loader,
            trainer_config=trainer_config,
        )
    if policy.model_key == "sellm_base":
        return train_sellm_icl(
            model,
            train_loader,
            trainer_config=trainer_config,
        )
    raise QualificationError(f"Unsupported ICL production model: {policy.model_key}.")


def _forward_icl_canary(policy: ICLProductionRefitPolicy, model, batch):
    if policy.model_key == "autotimes_base":
        inputs = AutoTimesICLAdapter().adapt(batch)
        return model.forward_icl(
            inputs.packed_context,
            prompt_mask=inputs.prompt_mask,
            packed_exogenous=inputs.packed_exogenous,
            query_target_exogenous=inputs.query_target_exogenous,
        )
    if policy.model_key == "sellm_base":
        inputs = SELLMICLAdapter().adapt(batch)
        return model.forward_icl(
            demonstration_contexts=inputs.demonstration_contexts,
            demonstration_targets=inputs.demonstration_targets,
            query_context=inputs.query_context,
            prompt_mask=inputs.prompt_mask,
            demonstration_context_exogenous=inputs.demonstration_context_exogenous,
            demonstration_target_exogenous=inputs.demonstration_target_exogenous,
            query_context_exogenous=inputs.query_context_exogenous,
            query_target_exogenous=inputs.query_target_exogenous,
        )
    raise QualificationError(f"Unsupported ICL production model: {policy.model_key}.")


def _run_refit(
    args: argparse.Namespace,
    policy: ICLProductionRefitPolicy,
) -> dict[str, Any]:
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise QualificationError(f"Output root already exists: {output_root}")
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise QualificationError("ICL production refit requires CUDA.")
    device_name = torch.cuda.get_device_name(0)
    if args.expected_device and device_name != args.expected_device:
        raise QualificationError(
            f"Expected {args.expected_device!r}, observed {device_name!r}."
        )

    target_path = args.target_source.expanduser().resolve()
    source = _load_source_contract(
        args.input_manifest.expanduser().resolve(),
        target_path,
    )
    if int(source["source_max_week"]) != TRAIN_END_WEEK:
        raise QualificationError("ICL source maximum week must be 202509.")
    operation_parts, exogenous_source = _load_operation_part_source(
        args.operation_part_manifest.expanduser().resolve(),
        args.operation_part_source.expanduser().resolve(),
        expected_site_cd=source["site_cd"],
    )
    backbone = _load_backbone_contract(args.llm_local_path.expanduser().resolve())
    if (
        backbone.get("model_id") != "Qwen/Qwen2-0.5B"
        or backbone.get("revision")
        != "91d2aff3f957f99e4c74c962f2f408dcc88a18d8"
    ):
        raise QualificationError("ICL refit requires the sealed Qwen2-0.5B revision.")

    frame = pl.read_parquet(
        target_path,
        columns=["oper_part_no", "demand_dt", "demand_qty"],
    )
    selected, selection = _complete_training_series(frame)
    selected = _add_approved_exogenous_features(selected, operation_parts)
    exogenous_source_revision = _sha256_payload(
        {
            "calendar_source_revision": CALENDAR_SOURCE_REVISION,
            "operation_part_source": exogenous_source,
            "feature_names": APPROVED_EXOGENOUS_FEATURES,
        }
    )
    bundle = _build_training_bundle(
        selected,
        source_revision=source["source_revision"],
        exogenous_source_revision=exogenous_source_revision,
    )
    if bundle.manifest.series_count != selection["eligible_series_count"]:
        raise QualificationError("Eligible series count changed while building Episodes.")

    output_root.mkdir(parents=True)
    status_path = output_root / "production-refit-status.txt"
    status_path.write_text("RUNNING current=episode_artifact\n", encoding="ascii")
    artifact_dir = output_root / "episodes"
    write_icl_episode_artifact(bundle, artifact_dir)
    preflight = {
        "contract": policy.receipt_contract,
        "status": "PREFLIGHT_PASS",
        "source_commit": _source_commit(),
        "source": source,
        "target_path": str(target_path),
        "selection": selection,
        "episode_manifest_hash": bundle.manifest.manifest_hash,
        "episode_count": len(bundle.episodes),
        "exogenous_source": exogenous_source,
        "exogenous_source_revision": exogenous_source_revision,
        "backbone": backbone,
        "training": {
            "model_key": policy.model_key,
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "train_end_week": TRAIN_END_WEEK,
            "forecast_origin": FORECAST_ORIGIN,
            "stride": STRIDE,
            "seed": SEED,
            "batch_size": policy.batch_size,
            "epochs": policy.epochs,
            "learning_rate": policy.learning_rate,
            "semantic_vocab_size": policy.semantic_vocab_size,
            "training_mode": "production_refit",
            "validation_enabled": False,
            "state_selection": "final_epoch",
        },
    }
    preflight["preflight_sha256"] = _sha256_payload(preflight)
    (output_root / "production-refit-data-manifest.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.preflight_only:
        status_path.write_text("PREFLIGHT_PASS\n", encoding="ascii")
        return preflight

    try:
        _seed_all(SEED)
        schema = bundle.manifest.exogenous_schema
        if schema is None:
            raise QualificationError("ICL refit requires exogenous Episodes.")
        model_config = _model_config(
            policy.model_key,
            horizon=HORIZON,
            llm_local_path=args.llm_local_path.expanduser().resolve(),
            schema_hash=schema.fingerprint,
            past_exogenous_dim=len(schema.past_feature_names),
            future_exogenous_dim=len(schema.future_feature_names),
        )
        model_config = replace(
            model_config,
            llm_revision=str(backbone["revision"]),
        )
        if policy.semantic_vocab_size is not None:
            model_config = replace(
                model_config,
                semantic_vocab_size=int(policy.semantic_vocab_size),
            )
        model = _build_model(policy.model_key, model_config)
        module = ICLEpisodeDataModule(
            bundle,
            batch_size=policy.batch_size,
            seed=SEED,
        )
        trainer_config = ICLTrainerConfig(
            epochs=policy.epochs,
            lr=policy.learning_rate,
            weight_decay=0.0,
            device=str(args.device),
            max_grad_norm=1.0,
            training_mode="production_refit",
        )
        status_path.write_text("RUNNING current=training epoch=0\n", encoding="ascii")
        started, _ = _cuda_measurement()
        result = _train_model_icl(
            policy,
            model,
            module,
            trainer_config,
        )
        training_runtime = _finish_cuda_measurement(started)
        checkpoint_path = save_icl_production_checkpoint(
            result,
            output_root / policy.checkpoint_filename,
            model_key=policy.model_key,
            bundle=bundle,
            trainer_config=trainer_config,
            random_seed=SEED,
            data_cutoff=TRAIN_END_WEEK,
            eligible_series_count=int(selection["eligible_series_count"]),
            backbone_contract=backbone,
        )
        del result, model
        gc.collect()
        torch.cuda.empty_cache()

        started, _ = _cuda_measurement()
        predictor = load_predictor(
            str(checkpoint_path),
            device=str(args.device),
            strict=True,
            config_overrides={
                "llm_local_path": args.llm_local_path.expanduser().resolve()
            },
        )
        load_runtime = _finish_cuda_measurement(started)
        batch = next(iter(module.loader(ICLSplit.TRAIN, shuffle=False))).to(
            str(args.device)
        )
        started, _ = _cuda_measurement()
        with torch.inference_mode():
            output = _forward_icl_canary(policy, predictor.model, batch)
        inference_runtime = _finish_cuda_measurement(started)
        if output.shape != batch.query_target.shape or not torch.isfinite(output).all():
            raise QualificationError("ICL strict-load canary output is invalid.")

        receipt = {
            **preflight,
            "status": "PASS",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": _file_sha256(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
                "strict_load": True,
            },
            "runtime": {
                "device": device_name,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "training": training_runtime,
                "strict_load": load_runtime,
                "canary_inference": inference_runtime,
            },
            "canary": {
                "batch_size": int(output.shape[0]),
                "shape": list(output.shape),
                "nonfinite_count": int((~torch.isfinite(output)).sum().item()),
                "raw_negative_count": int((output < 0).sum().item()),
            },
        }
        receipt.pop("preflight_sha256", None)
        receipt["receipt_sha256"] = _sha256_payload(receipt)
        (output_root / "production-refit-receipt.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        status_path.write_text("PASS\n", encoding="ascii")
        return receipt
    except Exception as exc:
        status_path.write_text(
            f"FAIL type={type(exc).__name__} message={exc}\n",
            encoding="utf-8",
        )
        raise


def run_refit(args: argparse.Namespace) -> dict[str, Any]:
    return _run_refit(args, AUTOTIMES_POLICY)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-source", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--operation-part-source", type=Path, required=True)
    parser.add_argument("--operation-part-manifest", type=Path, required=True)
    parser.add_argument("--llm-local-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-device", default="NVIDIA GeForce RTX 5090")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> None:
    receipt = run_refit(_parser().parse_args())
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "eligible_series_count": receipt["selection"][
                    "eligible_series_count"
                ],
                "receipt_sha256": receipt.get("receipt_sha256")
                or receipt.get("preflight_sha256"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

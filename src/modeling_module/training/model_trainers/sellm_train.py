from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Optional

import torch
import torch.nn.functional as F

from modeling_module.training.adapters import DefaultAdapter
from modeling_module.training.config import StageConfig, TrainingConfig, apply_stage
from modeling_module.training.engine import CommonTrainer
from modeling_module.training.model_trainers.amp_policy import amp_type_set
from modeling_module.training.model_trainers.loss_policy import infer_loss_mode
from modeling_module.training.model_trainers.spike_policy import maybe_make_spike_loader


def _sellm_icl_forward(model, batch):
    from modeling_module.icl.model_adapters import SELLMICLAdapter

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


def train_sellm_icl(
    model,
    train_loader,
    val_loader=None,
    *,
    trainer_config=None,
):
    """Train the SELLM semantic prompt encoder from sealed ICL episodes."""

    from modeling_module.icl.training import ICLTrainerConfig, fit_icl_model

    if not bool(getattr(getattr(model, "cfg", None), "icl_enabled", False)):
        raise ValueError("SELLM ICL training requires cfg.icl_enabled=True.")
    manifest = getattr(getattr(train_loader, "dataset", None), "manifest", None)
    configured_hash = getattr(model.cfg, "icl_exogenous_schema_hash", None)
    actual_schema = getattr(manifest, "exogenous_schema", None)
    actual_hash = None if actual_schema is None else actual_schema.fingerprint
    if configured_hash != actual_hash:
        raise ValueError("SELLM ICL checkpoint and Episode exogenous schema hash differ.")
    result = fit_icl_model(
        model,
        train_loader,
        val_loader,
        forward=_sellm_icl_forward,
        config=trainer_config or ICLTrainerConfig(),
    )
    fit_sellm_validation_scalar_calibration(
        result.model,
        val_loader,
        device=(trainer_config or ICLTrainerConfig()).device,
    )
    return result


def _calibration_fingerprint(rows: list[dict[str, object]]) -> str:
    payload = {
        "contract": "modeling_module.sellm_validation_scalar_source.v1",
        "source_split": "validation",
        "episodes": sorted(rows, key=lambda item: str(item["episode_id"])),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fit_sellm_validation_scalar_calibration(
    model,
    validation_loader,
    *,
    device: str | torch.device,
) -> dict[str, object] | None:
    """Fit one global demand-scale multiplier from validation episodes only."""

    cfg = getattr(model, "cfg", None)
    mode = str(getattr(cfg, "output_calibration_mode", "none"))
    if mode == "none":
        return None
    if mode != "validation_scalar":
        raise ValueError(f"Unsupported SELLM output calibration mode: {mode!r}.")
    if validation_loader is None:
        raise ValueError("SELLM validation-scalar calibration requires a validation loader.")
    if bool(getattr(cfg, "output_calibration_fitted", False)):
        raise ValueError("SELLM output calibration is already fitted and cannot be refitted.")

    from modeling_module.data_loader.icl_episode_data_module import ICLBatch

    runtime_device = torch.device(device)
    model = model.to(runtime_device)
    model.eval()
    prediction_sum = 0.0
    target_sum = 0.0
    target_absolute_sum = 0.0
    point_count = 0
    source_rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(validation_loader, start=1):
            if not isinstance(batch, ICLBatch):
                raise TypeError(
                    "SELLM output calibration requires ICLEpisodeDataModule batches."
                )
            if not batch.splits or any(
                str(split) != "validation" for split in batch.splits
            ):
                raise ValueError(
                    "SELLM output calibration accepts validation episodes only; "
                    f"batch_index={batch_index}, splits={sorted(set(batch.splits))}."
                )
            for row_index, (episode_id, series_id, origin_week) in enumerate(zip(
                batch.episode_ids,
                batch.series_ids,
                batch.origin_weeks.tolist(),
            )):
                target_payload = json.dumps(
                    batch.query_target[row_index].tolist(),
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                source_rows.append(
                    {
                        "episode_id": str(episode_id),
                        "series_id": str(series_id),
                        "origin_week": int(origin_week),
                        "target_sha256": hashlib.sha256(target_payload).hexdigest(),
                    }
                )
            runtime_batch = batch.to(runtime_device)
            prediction = _sellm_icl_forward(model, runtime_batch)
            target = runtime_batch.query_target
            if prediction.shape != target.shape:
                raise ValueError(
                    "SELLM calibration prediction shape must match its validation target."
                )
            if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
                raise ValueError("SELLM calibration tensors must contain finite values only.")
            if bool((prediction < 0.0).any()) or bool((target < 0.0).any()):
                raise ValueError(
                    "SELLM validation-scalar calibration requires non-negative "
                    "predictions and demand targets."
                )
            prediction_sum += float(prediction.double().sum())
            target_sum += float(target.double().sum())
            target_absolute_sum += float(target.double().abs().sum())
            point_count += int(target.numel())

    if not source_rows or point_count == 0:
        raise ValueError("SELLM output calibration received an empty validation loader.")
    episode_ids = [str(item["episode_id"]) for item in source_rows]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("SELLM output calibration validation episodes must be unique.")
    if prediction_sum <= 0.0 or target_absolute_sum <= 0.0:
        raise ValueError(
            "SELLM output calibration requires positive prediction and target totals."
        )

    raw_scale = target_sum / prediction_sum
    min_scale = float(cfg.output_calibration_min_scale)
    max_scale = float(cfg.output_calibration_max_scale)
    applied_scale = min(max(raw_scale, min_scale), max_scale)
    source_fingerprint = _calibration_fingerprint(source_rows)
    model.seal_output_calibration(
        scale=applied_scale,
        source_fingerprint=source_fingerprint,
    )
    before_bias = (prediction_sum - target_sum) / target_absolute_sum
    after_bias = (prediction_sum * applied_scale - target_sum) / target_absolute_sum
    result = {
        **model.output_calibration_contract(),
        "episode_count": len(source_rows),
        "point_count": point_count,
        "raw_scale": raw_scale,
        "clipped_to_bounds": applied_scale != raw_scale,
        "validation_bias_before": before_bias,
        "validation_bias_after": after_bias,
    }
    model.output_calibration_fit_stats = result
    return result


class SELLMAdapter(DefaultAdapter):
    def reg_loss(self, model):
        reg_fn = getattr(model, "reg_loss", None)
        if callable(reg_fn):
            return reg_fn()
        return None


def sellm_negative_output_penalty(
    prediction: torch.Tensor,
    *,
    weight: float,
) -> torch.Tensor:
    """Penalize negative point forecasts in demand coordinates."""

    if not torch.is_tensor(prediction):
        raise TypeError("SELLM negative-output penalty requires a tensor prediction.")
    if prediction.ndim not in (2, 3):
        raise ValueError(
            "SELLM point prediction must be rank 2 or 3, "
            f"got shape={tuple(prediction.shape)}"
        )
    penalty_weight = float(weight)
    if not math.isfinite(penalty_weight) or penalty_weight <= 0.0:
        raise ValueError(
            "SELLM negative-output penalty weight must be finite and > 0, "
            f"got {weight!r}"
        )
    return penalty_weight * F.relu(-prediction).square().mean()


def make_sellm_negative_output_penalty(model, *, loss_mode: str):
    """Build the training-only penalty hook, preserving the zero-weight baseline."""

    cfg = getattr(model, "cfg", None)
    weight = float(getattr(cfg, "negative_output_penalty_weight", 0.0))
    if weight == 0.0:
        return None
    if loss_mode != "point":
        raise ValueError(
            "negative_output_penalty_weight is supported only for SELLM "
            f"point training, got loss_mode={loss_mode!r}"
        )

    def penalty(_x, prediction, _cfg):
        return sellm_negative_output_penalty(prediction, weight=weight)

    return penalty


def train_sellm(
    model,
    train_loader,
    val_loader,
    device,
    *,
    stages: list[StageConfig] | None = None,
    train_cfg: Optional[TrainingConfig] = None,
):
    """Train SELLM with the shared trainer."""

    assert train_cfg is not None, "train_cfg is required."

    loss_mode = infer_loss_mode(train_cfg)
    if loss_mode != "point":
        raise NotImplementedError(f"[train_sellm] SELLM v1 supports only point loss, got {loss_mode!r}.")
    negative_output_penalty_fn = make_sellm_negative_output_penalty(
        model,
        loss_mode=loss_mode,
    )

    amp_device, amp_enabled, amp_dtype = amp_type_set(train_cfg)
    autocast_input = dict(device_type=amp_device, enabled=amp_enabled, dtype=amp_dtype)

    if not stages:
        stages = [StageConfig(epochs=train_cfg.epochs, spike_enabled=train_cfg.spike_loss.enabled)]

    adapter = SELLMAdapter()
    is_production_refit = (
        getattr(train_cfg, "training_mode", "qualification")
        == "production_refit"
    )
    if is_production_refit and val_loader is not None:
        raise ValueError("production_refit requires val_loader=None.")

    best = None
    global_best_loss = float("inf")
    global_best_state = (
        None if is_production_refit else copy.deepcopy(model.state_dict())
    )
    global_best_cfg = train_cfg
    total_epochs_completed = 0

    for i, stg in enumerate(stages, 1):
        cfg_i = apply_stage(train_cfg, stg)
        print(f"\n[train_sellm] ===== Stage {i}/{len(stages)} =====")
        print(f"  - spike: {'ON' if cfg_i.spike_loss.enabled else 'OFF'}")
        print(f"  - epochs: {cfg_i.epochs} | lr={cfg_i.lr} | horizon_decay={cfg_i.use_horizon_decay}")
        from modeling_module.training.model_trainers.cfg_policy import dump_cfg

        dump_cfg(cfg_i, name="sellm_train")
        tl_i = maybe_make_spike_loader(train_loader, enable=cfg_i.spike_loss.enabled)

        trainer = CommonTrainer(
            cfg=cfg_i,
            adapter=adapter,
            logger=print,
            metrics_fn=None,
            future_exo_cb=None,
            autocast_input=autocast_input,
            extra_loss_fn=None,
            training_only_extra_loss_fn=negative_output_penalty_fn,
            use_exogenous_mode=bool(getattr(train_cfg, "use_exogenous_mode", False)),
            device=device,
        )
        model = trainer.fit(model, tl_i, val_loader, tta_steps=0)
        total_epochs_completed += int(getattr(trainer, "epochs_completed_", 0))
        if is_production_refit:
            best = {
                "model": model,
                "cfg": cfg_i,
                "best_val_loss": None,
                "final_train_loss": float(
                    getattr(trainer, "final_train_loss_", float("nan"))
                ),
                "epochs_completed": total_epochs_completed,
                "state_selection": "final_epoch",
            }
            continue

        stage_best_loss = float(getattr(trainer, "best_loss_", float("inf")))
        if stage_best_loss < global_best_loss:
            global_best_loss = stage_best_loss
            global_best_state = copy.deepcopy(model.state_dict())
            global_best_cfg = cfg_i
        best = {"model": model, "cfg": cfg_i, "best_val_loss": stage_best_loss}

    if not is_production_refit:
        assert global_best_state is not None
        model.load_state_dict(global_best_state)
        best = {
            "model": model,
            "cfg": global_best_cfg,
            "best_val_loss": global_best_loss,
        }
    assert best is not None
    return best


__all__ = [
    "SELLMAdapter",
    "fit_sellm_validation_scalar_calibration",
    "make_sellm_negative_output_penalty",
    "sellm_negative_output_penalty",
    "train_sellm",
    "train_sellm_icl",
]

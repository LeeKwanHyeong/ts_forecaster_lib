from __future__ import annotations

import copy
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
    return fit_icl_model(
        model,
        train_loader,
        val_loader,
        forward=_sellm_icl_forward,
        config=trainer_config or ICLTrainerConfig(),
    )


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
    "make_sellm_negative_output_penalty",
    "sellm_negative_output_penalty",
    "train_sellm",
    "train_sellm_icl",
]

from __future__ import annotations

import copy
from typing import Optional

from modeling_module.training.adapters import DefaultAdapter
from modeling_module.training.config import StageConfig, TrainingConfig, apply_stage
from modeling_module.training.engine import CommonTrainer
from modeling_module.training.model_trainers.amp_policy import amp_type_set
from modeling_module.training.model_trainers.loss_policy import infer_loss_mode
from modeling_module.training.model_trainers.spike_policy import maybe_make_spike_loader


def _autotimes_icl_forward(model, batch):
    from modeling_module.icl.model_adapters import AutoTimesICLAdapter

    inputs = AutoTimesICLAdapter().adapt(batch)
    return model.forward_icl(
        inputs.packed_context,
        prompt_mask=inputs.prompt_mask,
        packed_exogenous=inputs.packed_exogenous,
        query_target_exogenous=inputs.query_target_exogenous,
    )


def train_autotimes_icl(
    model,
    train_loader,
    val_loader=None,
    *,
    trainer_config=None,
):
    """Train an ICL-enabled AutoTimes checkpoint from sealed episode batches."""

    from modeling_module.icl.training import ICLTrainerConfig, fit_icl_model

    if not bool(getattr(getattr(model, "cfg", None), "icl_enabled", False)):
        raise ValueError("AutoTimes ICL training requires cfg.icl_enabled=True.")
    manifest = getattr(getattr(train_loader, "dataset", None), "manifest", None)
    configured_hash = getattr(model.cfg, "icl_exogenous_schema_hash", None)
    actual_schema = getattr(manifest, "exogenous_schema", None)
    actual_hash = None if actual_schema is None else actual_schema.fingerprint
    if configured_hash != actual_hash:
        raise ValueError(
            "AutoTimes ICL checkpoint and Episode exogenous schema hash differ."
        )
    return fit_icl_model(
        model,
        train_loader,
        val_loader,
        forward=_autotimes_icl_forward,
        config=trainer_config or ICLTrainerConfig(),
    )


def train_autotimes(
    model,
    train_loader,
    val_loader,
    device,
    *,
    stages: list[StageConfig] | None = None,
    train_cfg: Optional[TrainingConfig] = None,
):
    """Train AutoTimes token adapters while keeping the LLM backbone frozen."""

    assert train_cfg is not None, "train_cfg is required."
    if bool(getattr(train_cfg, "use_exogenous_mode", False)):
        raise RuntimeError("[train_autotimes] autotimes_base supports endogenous inputs only.")
    loss_mode = infer_loss_mode(train_cfg)
    if loss_mode != "point":
        raise NotImplementedError(
            f"[train_autotimes] autotimes_base supports point loss only, got {loss_mode!r}."
        )

    amp_device, amp_enabled, amp_dtype = amp_type_set(train_cfg)
    autocast_input = dict(device_type=amp_device, enabled=amp_enabled, dtype=amp_dtype)
    if not stages:
        stages = [StageConfig(epochs=train_cfg.epochs, spike_enabled=train_cfg.spike_loss.enabled)]

    is_production_refit = getattr(train_cfg, "training_mode", "qualification") == "production_refit"
    if is_production_refit and val_loader is not None:
        raise ValueError("production_refit requires val_loader=None.")

    global_best_loss = float("inf")
    global_best_state = None if is_production_refit else copy.deepcopy(model.state_dict())
    global_best_cfg = train_cfg
    result = None
    total_epochs_completed = 0

    for index, stage in enumerate(stages, 1):
        cfg_i = apply_stage(train_cfg, stage)
        print(f"\n[train_autotimes] ===== Stage {index}/{len(stages)} =====")
        stage_loader = maybe_make_spike_loader(train_loader, enable=cfg_i.spike_loss.enabled)
        trainer = CommonTrainer(
            cfg=cfg_i,
            adapter=DefaultAdapter(),
            logger=print,
            metrics_fn=None,
            future_exo_cb=None,
            autocast_input=autocast_input,
            extra_loss_fn=None,
            use_exogenous_mode=False,
            device=device,
        )
        model = trainer.fit(model, stage_loader, val_loader, tta_steps=0)
        total_epochs_completed += int(getattr(trainer, "epochs_completed_", 0))
        if is_production_refit:
            result = {
                "model": model,
                "cfg": cfg_i,
                "best_val_loss": None,
                "final_train_loss": float(getattr(trainer, "final_train_loss_", float("nan"))),
                "epochs_completed": total_epochs_completed,
                "state_selection": "final_epoch",
            }
            continue

        stage_best_loss = float(getattr(trainer, "best_loss_", float("inf")))
        if stage_best_loss < global_best_loss:
            global_best_loss = stage_best_loss
            global_best_state = copy.deepcopy(model.state_dict())
            global_best_cfg = cfg_i

    if is_production_refit:
        assert result is not None
        return result

    assert global_best_state is not None
    model.load_state_dict(global_best_state)
    return {"model": model, "cfg": global_best_cfg, "best_val_loss": global_best_loss}


__all__ = ["train_autotimes", "train_autotimes_icl"]

from __future__ import annotations

import copy
from typing import Optional

from modeling_module.training.adapters import DefaultAdapter
from modeling_module.training.config import StageConfig, TrainingConfig, apply_stage
from modeling_module.training.engine import CommonTrainer
from modeling_module.training.model_trainers.amp_policy import amp_type_set
from modeling_module.training.model_trainers.loss_policy import infer_loss_mode
from modeling_module.training.model_trainers.spike_policy import maybe_make_spike_loader


def train_nhits(
    model,
    train_loader,
    val_loader,
    device,
    *,
    stages: list[StageConfig] | None = None,
    train_cfg: Optional[TrainingConfig] = None,
):
    """Train the public endogenous, point-only N-HiTS artifact."""

    assert train_cfg is not None, "train_cfg is required."
    if bool(getattr(train_cfg, "use_exogenous_mode", False)):
        raise RuntimeError("[train_nhits] nhits_base supports endogenous inputs only.")

    loss_mode = infer_loss_mode(train_cfg)
    if loss_mode != "point":
        raise NotImplementedError(
            f"[train_nhits] nhits_base supports only point loss, got loss_mode={loss_mode!r}."
        )

    amp_device, amp_enabled, amp_dtype = amp_type_set(train_cfg)
    autocast_input = dict(
        device_type=amp_device,
        enabled=amp_enabled,
        dtype=amp_dtype,
    )
    if not stages:
        stages = [
            StageConfig(
                epochs=train_cfg.epochs,
                spike_enabled=train_cfg.spike_loss.enabled,
            )
        ]

    adapter = DefaultAdapter()
    global_best_loss = float("inf")
    global_best_state = copy.deepcopy(model.state_dict())
    global_best_cfg = train_cfg

    for index, stage in enumerate(stages, 1):
        cfg_i = apply_stage(train_cfg, stage)
        print(f"\n[train_nhits] ===== Stage {index}/{len(stages)} =====")
        print(f"  - spike: {'ON' if cfg_i.spike_loss.enabled else 'OFF'}")
        print(
            f"  - epochs: {cfg_i.epochs} | lr={cfg_i.lr} | "
            f"horizon_decay={cfg_i.use_horizon_decay}"
        )

        from modeling_module.training.model_trainers.cfg_policy import dump_cfg

        dump_cfg(cfg_i, name="nhits_train")
        stage_loader = maybe_make_spike_loader(
            train_loader,
            enable=cfg_i.spike_loss.enabled,
        )
        trainer = CommonTrainer(
            cfg=cfg_i,
            adapter=adapter,
            logger=print,
            metrics_fn=None,
            future_exo_cb=None,
            autocast_input=autocast_input,
            extra_loss_fn=None,
            use_exogenous_mode=False,
            device=device,
        )
        model = trainer.fit(model, stage_loader, val_loader, tta_steps=0)
        stage_best_loss = float(getattr(trainer, "best_loss_", float("inf")))
        if stage_best_loss < global_best_loss:
            global_best_loss = stage_best_loss
            global_best_state = copy.deepcopy(model.state_dict())
            global_best_cfg = cfg_i

    model.load_state_dict(global_best_state)
    return {
        "model": model,
        "cfg": global_best_cfg,
        "best_val_loss": global_best_loss,
    }


__all__ = ["train_nhits"]

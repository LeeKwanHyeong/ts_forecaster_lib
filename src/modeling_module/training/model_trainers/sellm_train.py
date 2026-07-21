from __future__ import annotations

import copy
from typing import Optional

from modeling_module.training.adapters import DefaultAdapter
from modeling_module.training.config import StageConfig, TrainingConfig, apply_stage
from modeling_module.training.engine import CommonTrainer
from modeling_module.training.model_trainers.amp_policy import amp_type_set
from modeling_module.training.model_trainers.loss_policy import infer_loss_mode
from modeling_module.training.model_trainers.spike_policy import maybe_make_spike_loader


class SELLMAdapter(DefaultAdapter):
    def reg_loss(self, model):
        reg_fn = getattr(model, "reg_loss", None)
        if callable(reg_fn):
            return reg_fn()
        return None


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

    amp_device, amp_enabled, amp_dtype = amp_type_set(train_cfg)
    autocast_input = dict(device_type=amp_device, enabled=amp_enabled, dtype=amp_dtype)

    if not stages:
        stages = [StageConfig(epochs=train_cfg.epochs, spike_enabled=train_cfg.spike_loss.enabled)]

    adapter = SELLMAdapter()
    best = None
    global_best_loss = float("inf")
    global_best_state = copy.deepcopy(model.state_dict())
    global_best_cfg = train_cfg

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
            use_exogenous_mode=bool(getattr(train_cfg, "use_exogenous_mode", False)),
            device=device,
        )
        model = trainer.fit(model, tl_i, val_loader, tta_steps=0)
        stage_best_loss = float(getattr(trainer, "best_loss_", float("inf")))
        if stage_best_loss < global_best_loss:
            global_best_loss = stage_best_loss
            global_best_state = copy.deepcopy(model.state_dict())
            global_best_cfg = cfg_i
        best = {"model": model, "cfg": cfg_i, "best_val_loss": stage_best_loss}

    model.load_state_dict(global_best_state)
    best = {"model": model, "cfg": global_best_cfg, "best_val_loss": global_best_loss}
    return best

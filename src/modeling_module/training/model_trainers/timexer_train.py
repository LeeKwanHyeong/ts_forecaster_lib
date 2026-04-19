from __future__ import annotations

import copy
from typing import Optional

from modeling_module.training.adapters import DefaultAdapter
from modeling_module.training.config import StageConfig, TrainingConfig, apply_stage
from modeling_module.training.engine import CommonTrainer
from modeling_module.training.model_trainers.amp_policy import amp_type_set
from modeling_module.training.model_trainers.loss_policy import infer_loss_mode
from modeling_module.training.model_trainers.spike_policy import maybe_make_spike_loader


def train_timexer(
    model,
    train_loader,
    val_loader,
    device,
    *,
    stages: list[StageConfig] | None = None,
    train_cfg: Optional[TrainingConfig] = None,
):
    """
    Train TimeXer with the library's shared trainer.

    TimeXer v1 is intentionally strict:
    - point forecasting only
    - historical continuous exogenous inputs only
    """

    assert train_cfg is not None, "train_cfg is required."

    if not bool(getattr(train_cfg, "use_exogenous_mode", True)):
        raise RuntimeError("[train_timexer] TimeXer requires use_exogenous_mode=True.")

    loss_mode = infer_loss_mode(train_cfg)
    if loss_mode != "point":
        raise NotImplementedError(
            f"[train_timexer] TimeXer v1 supports only point loss, got loss_mode={loss_mode!r}."
        )

    amp_device, amp_enabled, amp_dtype = amp_type_set(train_cfg)
    autocast_input = dict(device_type=amp_device, enabled=amp_enabled, dtype=amp_dtype)

    if not stages:
        stages = [StageConfig(epochs=train_cfg.epochs, spike_enabled=train_cfg.spike_loss.enabled)]

    adapter = DefaultAdapter()
    best = None
    global_best_loss = float("inf")
    global_best_state = copy.deepcopy(model.state_dict())
    global_best_cfg = train_cfg

    for i, stg in enumerate(stages, 1):
        cfg_i = apply_stage(train_cfg, stg)
        print(f"\n[train_timexer] ===== Stage {i}/{len(stages)} =====")
        print(f"  - spike: {'ON' if cfg_i.spike_loss.enabled else 'OFF'}")
        print(f"  - epochs: {cfg_i.epochs} | lr={cfg_i.lr} | horizon_decay={cfg_i.use_horizon_decay}")
        from modeling_module.training.model_trainers.cfg_policy import dump_cfg

        dump_cfg(cfg_i, name="timexer_train")
        tl_i = maybe_make_spike_loader(train_loader, enable=cfg_i.spike_loss.enabled)

        trainer = CommonTrainer(
            cfg=cfg_i,
            adapter=adapter,
            logger=print,
            metrics_fn=None,
            future_exo_cb=None,
            autocast_input=autocast_input,
            extra_loss_fn=None,
            use_exogenous_mode=True,
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

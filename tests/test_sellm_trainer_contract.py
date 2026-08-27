from __future__ import annotations

import torch

from modeling_module.api.train import TrainerConfig, _normalize_payload
from modeling_module.models.SELLM.training_contract import SELLM_TRAINER_CONTRACT
from modeling_module.training.config import TrainingConfig
from modeling_module.training.model_trainers.amp_policy import amp_type_set
from modeling_module.training.model_trainers.total_train import (
    _build_common_train_configs,
    _training_checkpoint_meta,
)
from modeling_module.training.optim import build_optimizer_and_scheduler


def test_sellm_shared_trainer_contract_is_explicit_and_stable():
    assert SELLM_TRAINER_CONTRACT.as_metadata() == {
        "optimizer": "adamw",
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "lr_scheduler": "constant",
        "t_max": 6,
        "use_amp": False,
        "amp_dtype": "fp32",
        "loss": "mae",
        "max_grad_norm": 30.0,
    }


def test_public_trainer_payload_preserves_shared_optimization_fields():
    trainer = TrainerConfig(**SELLM_TRAINER_CONTRACT.trainer_kwargs())
    payload = _normalize_payload({"trainer": trainer})

    assert payload["base_lr"] == SELLM_TRAINER_CONTRACT.learning_rate
    for key, value in SELLM_TRAINER_CONTRACT.trainer_kwargs().items():
        if key == "lr":
            continue
        assert payload[key] == value


def test_common_training_config_and_checkpoint_meta_record_contract():
    point_cfg, _quantile_cfg, stages = _build_common_train_configs(
        device="cpu",
        lookback=52,
        horizon=26,
        warmup_epochs=6,
        spike_epochs=0,
        base_lr=SELLM_TRAINER_CONTRACT.learning_rate,
        loss_point=None,
        loss_quantile=None,
        use_exogenous_mode=False,
        training_mode="qualification",
        random_seed=42,
        **{
            key: value
            for key, value in SELLM_TRAINER_CONTRACT.trainer_kwargs().items()
            if key != "lr"
        },
    )

    assert point_cfg.weight_decay == SELLM_TRAINER_CONTRACT.weight_decay
    assert point_cfg.lr_scheduler == "constant"
    assert point_cfg.use_amp is False
    assert point_cfg.amp_dtype == "fp32"
    assert point_cfg.max_grad_norm == 30.0
    meta = _training_checkpoint_meta(point_cfg, stages, {})
    assert meta["optimizer"] == "adamw"
    assert meta["learning_rate"] == 1e-4
    assert meta["weight_decay"] == 1e-2
    assert meta["lr_scheduler"] == "constant"
    assert meta["use_amp"] is False
    assert meta["amp_dtype"] == "fp32"
    assert meta["loss"] == "mae"
    assert meta["max_grad_norm"] == 30.0


def test_constant_scheduler_keeps_learning_rate_unchanged():
    model = torch.nn.Linear(2, 1)
    cfg = TrainingConfig(
        device="cpu",
        lr=1e-4,
        weight_decay=1e-2,
        lr_scheduler="constant",
        t_max=6,
    )
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)

    for _ in range(6):
        optimizer.step()
        scheduler.step()
        assert scheduler.get_last_lr() == [1e-4]


def test_amp_policy_respects_explicit_disable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    cfg = TrainingConfig(
        amp_device="cuda",
        use_amp=False,
        amp_dtype="fp32",
    )

    device, enabled, dtype = amp_type_set(cfg)

    assert device == "cuda"
    assert enabled is False
    assert dtype is torch.float32

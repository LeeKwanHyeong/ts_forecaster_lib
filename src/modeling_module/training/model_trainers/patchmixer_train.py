# patchmixer_train_fixed.py
#
# Fixes:
# 1) Do NOT delete/rebuild model.exo_head when future exogenous comes from the DataLoader (future_exo_cb=None).
#    The loader-based mode is controlled by PatchMixerConfig(exo_dim=E) and the model should be constructed
#    with that E so that optimizer captures exo_head parameters.
# 2) Only infer exo_dim and (re)build exo_head in the callback-based mode (future_exo_cb is not None).
# 3) _ensure_model_exo_head(exo_dim<=0) no longer deletes exo_head (avoids accidental removal).

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Optional, Callable

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, WeightedRandomSampler

from modeling_module.training.adapters import PatchMixerAdapter, DefaultAdapter
from modeling_module.training.config import TrainingConfig, StageConfig, apply_stage
from modeling_module.training.engine import CommonTrainer


def _dump_cfg(cfg):
    data = asdict(cfg) if is_dataclass(cfg) else cfg.__dict__
    print("[train_patchmixer] Effective TrainingConfig:")
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _infer_exo_dim_from_cb(future_exo_cb, horizon: int, device: str = "cpu") -> int:
    """Infer E from callback output. Expected shape: (H, E) tensor-like."""
    if future_exo_cb is None:
        return 0
    fe = future_exo_cb(0, horizon, device=device)  # (H,E) expected
    if isinstance(fe, torch.Tensor):
        return int(fe.size(-1))
    try:
        return int(fe.shape[-1])
    except Exception:
        return 0


def _ensure_model_exo_head(model, exo_dim: int):
    """(Re)build model.exo_head only when we are in callback-based exo mode.

    IMPORTANT:
      - If exo_dim <= 0: do nothing (do NOT delete exo_head).
        Disabling exogenous should be done via model/config construction (exo_dim=0).
    """
    if not hasattr(model, "exo_dim"):
        return model

    if exo_dim <= 0:
        # Do not touch model (prevents accidental removal when loader provides future_exo).
        return model

    current = int(getattr(model, "exo_dim", 0))
    has_head = getattr(model, "exo_head", None) is not None

    if current == exo_dim and has_head:
        return model

    model.exo_head = nn.Sequential(
        nn.Linear(exo_dim, 64),
        nn.GELU(),
        nn.Linear(64, 1),
    )
    model.exo_dim = int(exo_dim)
    print(f"[train_patchmixer] exo_head rebuilt with exo_dim={exo_dim}")
    return model


def _maybe_make_spike_loader(train_loader: DataLoader, enable: bool) -> DataLoader:
    """Simple spike oversampling for stage-2 if dataset exposes `sample_is_spike`."""
    if (not enable) or (not hasattr(train_loader.dataset, "sample_is_spike")):
        return train_loader

    import numpy as np

    m = np.asarray(train_loader.dataset.sample_is_spike, dtype=bool)
    w = np.where(m, 3.0, 1.0).astype("float32")
    sampler = WeightedRandomSampler(weights=w, num_samples=len(w), replacement=True)

    return DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        sampler=sampler,
        num_workers=train_loader.num_workers,
        pin_memory=getattr(train_loader, "pin_memory", True),
        drop_last=getattr(train_loader, "drop_last", False),
        collate_fn=getattr(train_loader, "collate_fn", None),
    )


def train_patchmixer(
    model,
    train_loader,
    val_loader,
    *,
    stages: list[StageConfig] | None = None,
    train_cfg: Optional[TrainingConfig] = None,
    # exogenous
    future_exo_cb: Optional[Callable[[int, int], "torch.Tensor"]] = None,
    exo_is_normalized: bool = False,
):
    """PatchMixer trainer entry.

    Modes:
      1) Callback-based future exo: future_exo_cb != None
         - Infer E from callback and ensure model.exo_head is built BEFORE optimizer creation.
      2) Loader-based future exo: future_exo_cb == None
         - DataLoader must provide future_exo (e.g., fe_cont) and model must be constructed with configs.exo_dim=E.
         - We DO NOT touch exo_head here.
    """
    assert train_cfg is not None, "train_cfg는 필수입니다."

    # (A) exo_head 사전 세팅은 'callback 모드'에서만 수행
    if future_exo_cb is not None:
        horizon = getattr(model, "horizon", None) or getattr(train_cfg, "horizon", None)
        if horizon is None:
            raise ValueError("horizon을 model 또는 train_cfg에서 찾을 수 없습니다.")

        E = _infer_exo_dim_from_cb(future_exo_cb, int(horizon), device="cpu")
        model = _ensure_model_exo_head(model, E)
        print(
            "[EXO-setup] (callback) "
            f"inferred E={E}, model.exo_dim={getattr(model, 'exo_dim', None)}, "
            f"has_head={getattr(model, 'exo_head', None) is not None}"
        )
    else:
        print(
            "[EXO-setup] (loader) future_exo_cb=None → skip exo_head setup. "
            f"model.exo_dim={getattr(model, 'exo_dim', None)}, "
            f"has_head={getattr(model, 'exo_head', None) is not None}"
        )

    # AMP 설정(bf16 기본)
    amp_device = getattr(train_cfg, "amp_device", "cuda")
    amp_enabled = (amp_device == "cuda" and torch.cuda.is_available())
    amp_dtype_str = getattr(train_cfg, "amp_dtype", "bf16")
    if isinstance(amp_dtype_str, torch.dtype):
        amp_dtype = amp_dtype_str
    else:
        s = str(amp_dtype_str).lower()
        if s in ("bf16", "bfloat16"):
            amp_dtype = torch.bfloat16
        elif s in ("fp16", "float16", "half"):
            amp_dtype = torch.float16
        elif s in ("fp32", "float32"):
            amp_dtype = torch.float32
        else:
            amp_dtype = torch.bfloat16
    autocast_input = dict(device_type=amp_device, enabled=amp_enabled, dtype=amp_dtype)

    # Adapter
    adapter = PatchMixerAdapter() if PatchMixerAdapter else DefaultAdapter()

    # stages
    if not stages or len(stages) == 0:
        stages = [StageConfig(epochs=train_cfg.epochs, spike_enabled=train_cfg.spike_loss.enabled)]

    best = None
    for i, stg in enumerate(stages, 1):
        cfg_i = apply_stage(train_cfg, stg)
        print(f"\n[train_patchmixer] ===== Stage {i}/{len(stages)} =====")
        print(f"  - spike: {'ON' if cfg_i.spike_loss.enabled else 'OFF'}")
        print(f"  - epochs: {cfg_i.epochs} | lr={cfg_i.lr} | horizon_decay={cfg_i.use_horizon_decay}")
        _dump_cfg(cfg_i)

        tl_i = _maybe_make_spike_loader(train_loader, enable=cfg_i.spike_loss.enabled)

        trainer = CommonTrainer(
            cfg=cfg_i,
            adapter=adapter,
            logger=print,
            metrics_fn=None,
            future_exo_cb=future_exo_cb,
            autocast_input=autocast_input,
            extra_loss_fn=None,
        )
        model = trainer.fit(model, tl_i, val_loader, tta_steps=0)
        best = {"model": model, "cfg": cfg_i}

    print(
        f"[EXO-train] model.exo_dim={getattr(model, 'exo_dim', 0)}  "
        f"future_exo_cb? {future_exo_cb is not None}  "
        f"exo_is_normalized={exo_is_normalized}"
    )
    return best

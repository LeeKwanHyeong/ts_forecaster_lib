from __future__ import annotations

import copy
import json
from dataclasses import asdict, is_dataclass
from typing import Optional, Callable

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, WeightedRandomSampler

from modeling_module.training.adapters import (
    PatchMixerAdapter,
    PatchMixerOriginalAdapter,
)
from modeling_module.training.config import TrainingConfig, StageConfig, apply_stage
from modeling_module.training.engine import CommonTrainer
from modeling_module.training.model_trainers.amp_policy import amp_type_set
from modeling_module.training.model_trainers.exo_policy import infer_exo_dim_from_cb
from modeling_module.training.model_trainers.spike_policy import maybe_make_spike_loader


def _ensure_patchmixer_exo_head(model, exo_dim: int):
    """
    모델의 외생 변수 처리용 헤드(exo_head)를 동적으로 생성 또는 갱신.

    기능:
    - Callback 모드일 때만 호출되어야 함.
    - exo_dim이 0 이하이면 기존 구성을 유지(삭제 방지).
    - 기존 차원과 다를 경우 새로운 MLP 헤드로 교체.
    """
    if not hasattr(model, "exo_dim"):
        return model

    # 외생 변수가 없다고 명시된 경우 모델 변경 없이 반환
    if exo_dim <= 0:
        return model

    current = int(getattr(model, "exo_dim", 0))
    has_head = getattr(model, "exo_head", None) is not None

    # 이미 적절한 헤드가 존재하면 스킵
    if current == exo_dim and has_head:
        return model

    # 새로운 헤드 생성 (Linear -> GELU -> Linear)
    model.exo_head = nn.Sequential(
        nn.Linear(exo_dim, 64),
        nn.GELU(),
        nn.Linear(64, 1),
    )
    model.future_exo_dim = int(exo_dim)
    print(f"[train_patchmixer] exo_head rebuilt with exo_dim={exo_dim}")
    return model

def train_patchmixer(
        model,
        train_loader,
        val_loader,
        *,
        stages: list[StageConfig] | None = None,
        train_cfg: Optional[TrainingConfig] = None,
        # 외생 변수 관련 인자
        future_exo_cb: Optional[Callable[[int, int], "torch.Tensor"]] = None,
        device
):
    """
    PatchMixer 모델 학습 진입점(Entry Point).

    기능:
    - 외생 변수 처리 모드(Callback vs Loader)에 따른 모델 헤드 설정.
    - AMP(Automatic Mixed Precision) 환경 구성.
    - 다단계(Multi-stage) 학습 루프 실행 (Spike Loss, LR 등 단계별 변경).

    Exo Mode:
    1. Callback Mode (future_exo_cb != None):
       - 콜백으로부터 차원(E)을 추론하여 모델의 exo_head를 동적으로 생성/갱신.
    2. Loader Mode (future_exo_cb == None):
       - DataLoader가 이미 외생 변수를 제공한다고 가정.
       - 모델의 exo_head 설정을 건드리지 않음.
    """
    assert train_cfg is not None, "train_cfg는 필수입니다."
    use_exogenous_mode = getattr(train_cfg, 'use_exogenous_mode', True)
    exo_is_normalized = getattr(train_cfg, 'exo_is_normalized', True)
    is_original = getattr(model, "architecture_variant", None) == "original"
    if is_original and (bool(use_exogenous_mode) or future_exo_cb is not None):
        raise RuntimeError(
            "[train_patchmixer] PatchMixerOriginal supports endogenous-only training."
        )
    # 1. 외생 변수 헤드 설정 (Callback 모드일 경우에만 동적 처리)
    if future_exo_cb is not None:
        horizon = getattr(model, "horizon", None) or getattr(train_cfg, "horizon", None)
        if horizon is None:
            raise ValueError("horizon을 model 또는 train_cfg에서 찾을 수 없습니다.")

        # 콜백을 통해 차원 추론 후 헤드 구성
        E = infer_exo_dim_from_cb(future_exo_cb, int(horizon), device="cpu")
        model = _ensure_patchmixer_exo_head(model, E)
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

    # 2. AMP (Mixed Precision) 설정
    amp_device, amp_enabled, amp_dtype = amp_type_set(train_cfg)
    autocast_input = dict(device_type=amp_device, enabled=amp_enabled, dtype=amp_dtype)

    # 3. 모델 어댑터 초기화 (입/출력 형식 변환용)
    adapter = PatchMixerOriginalAdapter() if is_original else PatchMixerAdapter()

    # 4. 학습 스테이지 설정
    # 별도 스테이지가 없으면 단일 스테이지로 구성
    if not stages or len(stages) == 0:
        stages = [StageConfig(epochs=train_cfg.epochs, spike_enabled=train_cfg.spike_loss.enabled)]

    best = None
    global_best_loss = float("inf")
    global_best_state = copy.deepcopy(model.state_dict())
    global_best_cfg = train_cfg

    # 5. 스테이지별 학습 루프 실행
    for i, stg in enumerate(stages, 1):
        # 현재 스테이지 설정 적용
        cfg_i = apply_stage(train_cfg, stg)
        print(f"\n[train_patchmixer] ===== Stage {i}/{len(stages)} =====")
        print(f"  - spike: {'ON' if cfg_i.spike_loss.enabled else 'OFF'}")
        print(f"  - epochs: {cfg_i.epochs} | lr={cfg_i.lr} | horizon_decay={cfg_i.use_horizon_decay}")
        from modeling_module.training.model_trainers.cfg_policy import dump_cfg
        dump_cfg(cfg_i, name = 'patchmixer_train')

        # Spike Loss 활성화 시 전용 로더 생성
        tl_i = maybe_make_spike_loader(train_loader, enable=cfg_i.spike_loss.enabled)

        # 트레이너 초기화 및 학습 수행
        trainer = CommonTrainer(
            cfg=cfg_i,
            adapter=adapter,
            logger=print,
            metrics_fn=None,
            future_exo_cb=future_exo_cb,
            autocast_input=autocast_input,
            extra_loss_fn=None,
            use_exogenous_mode=use_exogenous_mode,
            device = device
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

    # 학습 완료 상태 로그
    print(
        f"[EXO-train] model.exo_dim={getattr(model, 'exo_dim', 0)}  "
        f"future_exo_cb? {future_exo_cb is not None}  "
        f"exo_is_normalized={exo_is_normalized}"
    )
    return best

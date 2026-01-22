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
    """
    현재 학습 설정(TrainingConfig)을 JSON 형태로 출력.
    디버깅 및 로깅 용도.
    """
    data = asdict(cfg) if is_dataclass(cfg) else cfg.__dict__
    print("[train_patchmixer] Effective TrainingConfig:")
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _infer_exo_dim_from_cb(future_exo_cb, horizon: int, device: str = "cpu") -> int:
    """
    콜백 함수(future_exo_cb)의 출력으로부터 외생 변수의 차원(E) 추론.

    Args:
        future_exo_cb: (start_idx, horizon, device) -> Tensor[H, E]
    Returns:
        E (int): 외생 변수 차원 (없으면 0)
    """
    if future_exo_cb is None:
        return 0
    # 더미 호출을 통해 차원 확인
    fe = future_exo_cb(0, horizon, device=device)  # (H, E) expected

    if isinstance(fe, torch.Tensor):
        return int(fe.size(-1))
    try:
        return int(fe.shape[-1])
    except Exception:
        return 0


def _ensure_model_exo_head(model, exo_dim: int):
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
    model.exo_dim = int(exo_dim)
    print(f"[train_patchmixer] exo_head rebuilt with exo_dim={exo_dim}")
    return model


def _maybe_make_spike_loader(train_loader: DataLoader, enable: bool) -> DataLoader:
    """
    Spike Loss 활성화 시, 스파이크 샘플에 가중치를 부여한 DataLoader 생성.

    기능:
    - 데이터셋이 `sample_is_spike` 속성을 가질 때만 동작.
    - 스파이크 샘플에 3.0배 가중치를 부여하여 WeightedRandomSampler 적용.
    """
    if (not enable) or (not hasattr(train_loader.dataset, "sample_is_spike")):
        return train_loader

    import numpy as np

    # 샘플 가중치 계산 (Spike: 3.0, Normal: 1.0)
    m = np.asarray(train_loader.dataset.sample_is_spike, dtype=bool)
    w = np.where(m, 3.0, 1.0).astype("float32")
    sampler = WeightedRandomSampler(weights=w, num_samples=len(w), replacement=True)

    # 샘플러가 적용된 새로운 DataLoader 반환
    return DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        sampler=sampler,  # 셔플 대신 샘플러 사용
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
        # 외생 변수 관련 인자
        future_exo_cb: Optional[Callable[[int, int], "torch.Tensor"]] = None,
        exo_is_normalized: bool = False,
        use_exogenous_mode: bool = True

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

    # 1. 외생 변수 헤드 설정 (Callback 모드일 경우에만 동적 처리)
    if future_exo_cb is not None:
        horizon = getattr(model, "horizon", None) or getattr(train_cfg, "horizon", None)
        if horizon is None:
            raise ValueError("horizon을 model 또는 train_cfg에서 찾을 수 없습니다.")

        # 콜백을 통해 차원 추론 후 헤드 구성
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

    # 2. AMP (Mixed Precision) 설정
    amp_device = getattr(train_cfg, "amp_device", "cuda")
    amp_enabled = (amp_device == "cuda" and torch.cuda.is_available())
    amp_dtype_str = getattr(train_cfg, "amp_dtype", "bf16")

    # dtype 파싱 및 설정
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

    # 3. 모델 어댑터 초기화 (입/출력 형식 변환용)
    adapter = PatchMixerAdapter() if PatchMixerAdapter else DefaultAdapter()

    # 4. 학습 스테이지 설정
    # 별도 스테이지가 없으면 단일 스테이지로 구성
    if not stages or len(stages) == 0:
        stages = [StageConfig(epochs=train_cfg.epochs, spike_enabled=train_cfg.spike_loss.enabled)]

    best = None

    # 5. 스테이지별 학습 루프 실행
    for i, stg in enumerate(stages, 1):
        # 현재 스테이지 설정 적용
        cfg_i = apply_stage(train_cfg, stg)
        print(f"\n[train_patchmixer] ===== Stage {i}/{len(stages)} =====")
        print(f"  - spike: {'ON' if cfg_i.spike_loss.enabled else 'OFF'}")
        print(f"  - epochs: {cfg_i.epochs} | lr={cfg_i.lr} | horizon_decay={cfg_i.use_horizon_decay}")
        _dump_cfg(cfg_i)

        # Spike Loss 활성화 시 전용 로더 생성
        tl_i = _maybe_make_spike_loader(train_loader, enable=cfg_i.spike_loss.enabled)

        # 트레이너 초기화 및 학습 수행
        trainer = CommonTrainer(
            cfg=cfg_i,
            adapter=adapter,
            logger=print,
            metrics_fn=None,
            future_exo_cb=future_exo_cb,
            autocast_input=autocast_input,
            extra_loss_fn=None,
            use_exogenous_mode=use_exogenous_mode

        )
        model = trainer.fit(model, tl_i, val_loader, tta_steps=0)
        best = {"model": model, "cfg": cfg_i}

    # 학습 완료 상태 로그
    print(
        f"[EXO-train] model.exo_dim={getattr(model, 'exo_dim', 0)}  "
        f"future_exo_cb? {future_exo_cb is not None}  "
        f"exo_is_normalized={exo_is_normalized}"
    )
    return best
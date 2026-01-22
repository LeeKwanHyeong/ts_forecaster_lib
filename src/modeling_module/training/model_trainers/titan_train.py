from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from modeling_module.training.adapters import TitanAdapter, DefaultAdapter
from modeling_module.training.config import TrainingConfig, StageConfig, apply_stage
from modeling_module.training.engine import CommonTrainer
from modeling_module.utils.exogenous_utils import calendar_sin_cos


def _dump_cfg(cfg):
    """
    현재 적용된 학습 설정(TrainingConfig)을 JSON 형식으로 출력하여 로깅함.
    """
    data = asdict(cfg) if is_dataclass(cfg) else cfg.__dict__
    print("[train_titan] Effective TrainingConfig:")
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _infer_exo_dim_from_cb(future_exo_cb, horizon: int, device: str = "cpu") -> int:
    """
    콜백 함수(future_exo_cb) 실행을 통해 미래 외생 변수(Future Exo)의 차원(E)을 추론함.
    반환값은 (Horizon, Exo_Dim) 텐서의 마지막 차원 크기임.
    """
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
    """
    콜백 기반 외생 변수 모드일 때, 추론된 차원에 맞춰 모델의 `exo_head`를 동적으로 생성 또는 갱신함.

    특징:
    - exo_dim이 0 이하일 경우 모델을 변경하지 않음 (의도치 않은 삭제 방지).
    - 기존 차원과 다를 경우 MLP(Linear-GELU-Linear) 구조의 헤드를 재생성함.
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
    print(f"[train_titan] exo_head rebuilt with exo_dim={exo_dim}")
    return model


def _maybe_make_spike_loader(train_loader: DataLoader, enable: bool) -> DataLoader:
    """
    스파이크(Spike, 급격한 변화) 구간의 학습 강화를 위한 가중 샘플링(Weighted Random Sampling) 데이터 로더 생성.

    기능:
    - 데이터셋에 `sample_is_spike` 정보가 있을 경우에만 동작함.
    - 일반 샘플 대비 스파이크 샘플에 3.0배 높은 가중치를 부여하여 오버샘플링함.
    """
    if (not enable) or (not hasattr(train_loader.dataset, "sample_is_spike")):
        return train_loader

    import numpy as np
    m = np.asarray(train_loader.dataset.sample_is_spike, dtype=bool)
    w = np.where(m, 3.0, 1.0).astype("float32")  # 스파이크:기본 = 3:1
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


def train_titan(
        model,
        train_loader,
        val_loader,
        *,
        stages: list[StageConfig] | None = None,
        train_cfg: Optional[TrainingConfig] = None,
        future_exo_cb=None,
        use_exogenous_mode: bool = True
):
    """
    Titan 모델의 학습 파이프라인 실행 (Runner).

    주요 기능:
    - 외생 변수(Exo) 설정 자동화 및 모델 헤드 동적 구성.
    - AMP(Automatic Mixed Precision) 환경 설정.
    - 다단계(Multi-stage) 커리큘럼 학습 지원 (각 스테이지별 LR, Epoch, Spike Loss 적용).
    """
    assert train_cfg is not None, "train_cfg는 필수입니다."

    print(f'future_exo_cb : {future_exo_cb is not None}')

    # 1. 외생 변수 설정 및 헤드 갱신
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

    # 2. AMP (Mixed Precision) 설정

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

    # 3. 어댑터 초기화 (입출력 인터페이스 변환용)
    adapter = TitanAdapter() if TitanAdapter else DefaultAdapter()

    # 4. 스테이지 구성 (기본 단일 스테이지)
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

        # Spike Loss 설정에 따른 데이터 로더 생성
        tl_i = _maybe_make_spike_loader(train_loader, enable=cfg_i.spike_loss.enabled)

        # CommonTrainer를 통한 학습 수행
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
        model = trainer.fit(model, tl_i, val_loader, tta_steps=2)
        best = {"model": model, "cfg": cfg_i}

    return best
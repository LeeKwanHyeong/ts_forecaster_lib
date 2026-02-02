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
from modeling_module.training.model_trainers.exo_policy import infer_exo_dim_from_cb
from modeling_module.training.model_trainers.spike_policy import maybe_make_spike_loader
from modeling_module.utils.exogenous_utils import calendar_sin_cos


def _ensure_titan_exo_head(model, exo_dim: int):
    """콜백 기반 외생 변수 모드일 때, 추론된 차원에 맞춰 Titan의 decoder exogenous projection을 갱신함.

    Titan 계열은 `model.exo_head`를 사용하지 않고, `TitanDecoder.exo_proj`를 통해
    future_exo ([B,H,E])를 d_model로 투영하여 디코더 입력에 더합니다.

    특징:
    - exo_dim <= 0이면 변경하지 않음 (의도치 않은 삭제 방지)
    - 기존 exo_dim과 다를 경우 decoder.exo_proj를 재생성
    """
    if exo_dim <= 0:
        return model

    # model-level meta
    if hasattr(model, "exo_dim"):
        model.exo_dim = int(exo_dim)
    if hasattr(model, "use_exogenous_mode"):
        model.use_exogenous_mode = True

    dec = getattr(model, "decoder", None)
    if dec is None:
        return model

    d_model = getattr(model, "d_model", None)
    if d_model is None:
        qe = getattr(dec, "query_embed", None)
        if isinstance(qe, torch.Tensor):
            d_model = int(qe.size(-1))
        else:
            return model

    current = int(getattr(dec, "exo_dim", 0) or 0)
    has_proj = getattr(dec, "exo_proj", None) is not None
    if current == int(exo_dim) and has_proj:
        return model

    device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")
    dec.exo_dim = int(exo_dim)
    dec.exo_proj = nn.Linear(int(exo_dim), int(d_model)).to(device)
    print(f"[train_titan] decoder.exo_proj rebuilt: exo_dim={exo_dim} -> d_model={d_model}")
    return model


def train_titan(
        model,
        train_loader,
        val_loader,
        *,
        stages: list[StageConfig] | None = None,
        train_cfg: Optional[TrainingConfig] = None,
        future_exo_cb=None,
        use_exogenous_mode: bool = True,
        device
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

        E = infer_exo_dim_from_cb(future_exo_cb, int(horizon), device="cpu")
        model = _ensure_titan_exo_head(model, E)
        print(
            "[EXO-setup] (callback) "
            f"inferred E={E}, model.exo_dim={getattr(model, 'exo_dim', None)}, "
            f"has_exo_proj={getattr(getattr(model, 'decoder', None), 'exo_proj', None) is not None}"
        )
    else:
        print(
            "[EXO-setup] (loader) future_exo_cb=None → skip decoder.exo_proj setup. "
            f"model.exo_dim={getattr(model, 'exo_dim', None)}, "
            f"has_exo_proj={getattr(getattr(model, 'decoder', None), 'exo_proj', None) is not None}"
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
        print(f"\n[train_titan] ===== Stage {i}/{len(stages)} =====")
        print(f"  - spike: {'ON' if cfg_i.spike_loss.enabled else 'OFF'}")
        print(f"  - epochs: {cfg_i.epochs} | lr={cfg_i.lr} | horizon_decay={cfg_i.use_horizon_decay}")
        from modeling_module.training.model_trainers.cfg_policy import dump_cfg
        dump_cfg(cfg = cfg_i, name = 'titan_train')

        # Spike Loss 설정에 따른 데이터 로더 생성
        tl_i = maybe_make_spike_loader(train_loader, enable=cfg_i.spike_loss.enabled)

        # CommonTrainer를 통한 학습 수행
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
        model = trainer.fit(model, tl_i, val_loader, tta_steps=2)
        best = {"model": model, "cfg": cfg_i}

    return best
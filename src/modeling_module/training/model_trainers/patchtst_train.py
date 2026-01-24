# modeling_module/training/model_trainers/patchtst_train.py
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Optional, Callable

import torch
import torch.nn as nn
from modeling_module.models.PatchTST.supervised.PatchTST import PointHeadWithExo, QuantileHeadWithExo, DistHeadWithExo
from torch.utils.data import DataLoader, WeightedRandomSampler

from modeling_module.training.adapters import DefaultAdapter
from modeling_module.training.config import TrainingConfig, StageConfig, apply_stage
from modeling_module.training.engine import CommonTrainer
from modeling_module.utils.exogenous_utils import calendar_sin_cos

# PatchTST 내부 head 재구성에 필요
from modeling_module.models.PatchTST.common.patching import compute_patch_num


def _dump_cfg(cfg):
    """학습 설정(Config) 내용 출력."""
    data = asdict(cfg) if is_dataclass(cfg) else cfg.__dict__
    print("[train_patchtst] Effective TrainingConfig:")
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _infer_exo_dim_from_cb(future_exo_cb, horizon: int, device: str = "cpu") -> int:
    """
    콜백 함수 실행을 통한 미래 외생 변수 차원(E) 추론.
    반환: (H, E) 텐서의 마지막 차원.
    """
    if future_exo_cb is None:
        return 0
    fe = future_exo_cb(0, horizon, device=device)  # (H,E) 또는 (B,H,E) 류를 가정
    if isinstance(fe, torch.Tensor):
        return int(fe.size(-1))
    try:
        return int(fe.shape[-1])
    except Exception:
        return 0


def _infer_exo_dim_from_loader(train_loader: DataLoader) -> int:
    """
    데이터 로더의 첫 배치 검사를 통한 미래 외생 변수 차원 추론.
    배치 구조: (x, y, part_ids, fe_cont, ...) 가정.
    """
    try:
        batch = next(iter(train_loader))
    except Exception:
        return 0

    # (x, y, part_ids, fe_cont, pe_cont, pe_cat) 형태라고 가정
    if not isinstance(batch, (list, tuple)) or len(batch) < 4:
        return 0

    fe_cont = batch[3]
    if torch.is_tensor(fe_cont):
        return int(fe_cont.size(-1))
    return 0


def _pick_future_exo_cb(model, user_cb: Optional[Callable]) -> Optional[Callable]:
    """
    사용할 미래 외생 변수 생성 콜백 결정.
    우선순위: 사용자 지정 콜백 > 모델 설정(d_future > 0) 시 캘린더 콜백 > None.
    """
    # 사용자 콜백이 우선
    if user_cb is not None:
        return user_cb

    # 모델 cfg에 d_future가 0보다 크면 캘린더 기본 주입 (필요 시 프로젝트 정책에 맞게 조정)
    cfg = getattr(model, "cfg", None) or getattr(model, "config", None)
    d_future = int(getattr(cfg, "d_future", 0)) if cfg is not None else 0
    return calendar_sin_cos if d_future > 0 else None


def _ensure_patchtst_future_head(model, exo_dim: int, *, loss_mode: str = "point"):
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        return model

    current = int(getattr(cfg, "d_future", 0))
    cfg.d_future = int(exo_dim) if exo_dim > 0 else 0

    patch_num = compute_patch_num(cfg.lookback, cfg.patch_len, cfg.stride, cfg.padding_patch)

    # ---- dist 우선 처리 ----
    if loss_mode == "dist":
        # 이미 dist head면 유지(단, d_future 변경 반영 필요 시 rebuild)
        model.head = DistHeadWithExo(
            d_model=cfg.d_model,
            horizon=cfg.horizon,
            d_future=int(cfg.d_future),
            act=getattr(cfg, "act", "gelu"),
        )
        print(f"[train_patchtst] dist head rebuilt: d_future {current} -> {cfg.d_future}")
        return model

    # ---- 기존 quantile / point ----
    if getattr(model, "is_quantile", False):
        model.head = QuantileHeadWithExo(
            d_model=cfg.d_model,
            horizon=cfg.horizon,
            d_future=int(cfg.d_future),
            quantiles=getattr(cfg, "quantiles", (0.1, 0.5, 0.9)),
            hidden=getattr(cfg, "q_hidden", 128),
            monotonic=True,
        )
        print(f"[train_patchtst] quantile head rebuilt: d_future {current} -> {cfg.d_future}")
    else:
        model.head = PointHeadWithExo(
            d_model=cfg.d_model,
            horizon=cfg.horizon,
            d_future=int(cfg.d_future),
            patch_num=patch_num,
            agg=getattr(model.head, "agg", "mean"),
        )
        print(f"[train_patchtst] point head rebuilt: d_future {current} -> {cfg.d_future}")

    return model


def _maybe_make_spike_loader(train_loader: DataLoader, enable: bool) -> DataLoader:
    """
    Spike Loss 활성화 시 스파이크 샘플에 가중치를 부여한 DataLoader 생성.
    """
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


def train_patchtst(
        model,
        train_loader,
        val_loader,
        *,
        stages: list[StageConfig] | None = None,
        train_cfg: Optional[TrainingConfig] = None,
        future_exo_cb: Optional[Callable] = None,
        exo_is_normalized: bool = True,
        use_exogenous_mode: bool = True
):
    """
    PatchTST 모델 학습 진입점(Entry Point).

    기능:
    - 외생 변수(Exogenous Variable) 차원 자동 추론 및 헤드 조정.
    - AMP(Automatic Mixed Precision) 환경 구성.
    - CommonTrainer를 이용한 스테이지별(Stage-wise) 학습 루프 실행.
    """
    assert train_cfg is not None, "train_cfg는 필수입니다."

    # 1) exo 콜백 결정
    future_exo_cb = _pick_future_exo_cb(model, future_exo_cb)

    # 2) exo_dim 추론 후 head 보정 (PatchMixer와 동일한 전략)
    horizon = getattr(model, "horizon", None) or getattr(getattr(model, "cfg", None), "horizon", None) or getattr(
        train_cfg, "horizon", None)
    if horizon is None:
        raise ValueError("horizon을 model/cfg/train_cfg에서 찾을 수 없습니다.")

    E_loader = _infer_exo_dim_from_loader(train_loader)
    E_cb = _infer_exo_dim_from_cb(future_exo_cb, horizon, device="cpu")

    # 실제 학습 입력 기준으로 head를 맞추는 것이 안전
    E = E_loader if E_loader > 0 else E_cb

    loss_mode = str(getattr(train_cfg, "loss_mode", "auto")).lower()

    # --- ADD: auto 해석 ---
    if loss_mode in ("auto", "infer"):
        loss_obj = getattr(train_cfg, "loss", None)
        loss_name = getattr(loss_obj, "__class__", type("x", (object,), {})).__name__

        if (loss_name == "DistributionLoss") or bool(getattr(loss_obj, "is_distribution_output", False)):
            loss_mode = "dist"
        elif loss_name in ("MQLoss", "QuantileLoss") or bool(getattr(model, "is_quantile", False)):
            loss_mode = "quantile"
        else:
            loss_mode = "point"
    print(f'[train_patchtst] loss_mode: {loss_mode}')
    if use_exogenous_mode:
        print(f'[train_patchtst] exogenous_mode: {use_exogenous_mode}')
        model = _ensure_patchtst_future_head(model, E, loss_mode = loss_mode)

    # loader가 fe_cont를 주는 경우, 자동 CB는 끄는 쪽이 안전 (중복/불일치 방지)
    if E_loader > 0 and future_exo_cb is not None:
        future_exo_cb = None
        print(f"[train_patchtst] loader provides fe_cont(E={E_loader}), so future_exo_cb disabled.")

    # 3) AMP 설정 (Titan/PM과 동일 패턴)
    amp_device = getattr(train_cfg, "amp_device", "cuda")
    amp_enabled = (amp_device == "cuda" and torch.cuda.is_available())
    amp_dtype_str = str(getattr(train_cfg, "amp_dtype", "bf16")).lower()
    if amp_dtype_str in ("bf16", "bfloat16"):
        amp_dtype = torch.bfloat16
    elif amp_dtype_str in ("fp16", "float16", "half"):
        amp_dtype = torch.float16
    elif amp_dtype_str in ("fp32", "float32"):
        amp_dtype = torch.float32f
    else:
        amp_dtype = torch.bfloat16

    autocast_input = dict(device_type=amp_device, enabled=amp_enabled, dtype=amp_dtype)

    # 4) stages 구성 (기본 1 스테이지)
    if not stages or len(stages) == 0:
        stages = [StageConfig(epochs=train_cfg.epochs, spike_enabled=train_cfg.spike_loss.enabled)]

    adapter = DefaultAdapter()

    best = None
    for i, stg in enumerate(stages, 1):
        # 스테이지별 설정 적용
        cfg_i = apply_stage(train_cfg, stg)
        print(f"\n[train_patchtst] ===== Stage {i}/{len(stages)} =====")
        print(f"  - spike: {'ON' if cfg_i.spike_loss.enabled else 'OFF'}")
        print(f"  - epochs: {cfg_i.epochs} | lr={cfg_i.lr} | horizon_decay={cfg_i.use_horizon_decay}")
        _dump_cfg(cfg_i)

        tl_i = _maybe_make_spike_loader(train_loader, enable=cfg_i.spike_loss.enabled)

        # 트레이너 초기화 및 학습 수행
        trainer = CommonTrainer(
            cfg=cfg_i,
            adapter=adapter,
            future_exo_cb=future_exo_cb,
            logger=print,
            autocast_input=autocast_input,
            extra_loss_fn=None,
            use_exogenous_mode=use_exogenous_mode
        )
        model = trainer.fit(model, tl_i, val_loader, tta_steps=0)
        best = {"model": model, "cfg": cfg_i}

    print(
        f"[EXO-train] inferred E={E} | future_exo_cb? {future_exo_cb is not None} | exo_is_normalized={exo_is_normalized}")
    return best
"""total_train.py

High-level training runner for the LTB forecasting engine.

This module orchestrates training for multiple model families (PatchTST, Titan, PatchMixer)
across different frequencies (hourly/daily/weekly/monthly).

Key design points
-----------------
1) Exogenous feature wiring
   - Two supported sources for *future* exogenous features:
     (A) Loader-provided: batch[3] == fe_cont with shape (B, H, E)
     (B) Callback-provided: future_exo_cb(t0, H) -> (H, E)
   - When use_exogenous_mode=True:
     * If loader provides fe_cont, we prefer (A) and disable the callback to avoid dim mismatch.
     * If loader does not provide fe_cont, we fall back to the calendar callback.
   - When use_exogenous_mode=False: exogenous path is disabled for all models.

2) Fail-fast validation
   - If the loader provides fe_cont but E==0 while exogenous mode is enabled, we raise an error
     because models configured with d_future>0 will fail in forward.

3) Save layout
   - Checkpoints are stored under save_dir/<freq>/... with model-specific prefixes.

Operational guidance
-------------------
- If you see an error like:
    "future_exo last-dim(D)=0 != d_future=2"
  then your model was built with exo_dim=2, but the inference/train batch provides fe_cont with E=0
  (or you did not provide a future_exo callback/batch). Fix by aligning exo_dim with actual future exo.
"""

from typing import Dict, Tuple, Optional, Iterable, List, Callable
from pathlib import Path
from dataclasses import asdict, is_dataclass, replace

import numpy as np
import torch

from modeling_module.models.PatchMixer.common.configs import (
    PatchMixerConfigMonthly,
    PatchMixerConfig,
    PatchMixerConfigWeekly,
)
# PatchMixerConfigDaily, Hourly는 별도 클래스 없이 PatchMixerConfig + kwargs로 처리

from modeling_module.models.PatchTST.common.configs import PatchTSTConfig
from modeling_module.models.PatchTST.self_supervised.PatchTST import PatchTSTPretrainModel
from modeling_module.models.Titan import TitanBaseModel
from modeling_module.models.Titan.common.configs import TitanConfig
from modeling_module.models.model_builder import (
    build_titan_base,
    build_titan_lmm,
    build_titan_seq2seq,
    build_patch_mixer_base,
    build_patch_mixer_quantile,
    build_patchTST_base,
    build_patchTST_quantile,
)
from modeling_module.training.config import SpikeLossConfig, TrainingConfig, StageConfig
from modeling_module.training.metrics import quantile_metrics
from modeling_module.training.model_trainers.patchmixer_train import train_patchmixer
from modeling_module.training.model_trainers.patchtst_finetune import train_patchtst_finetune
from modeling_module.training.model_trainers.patchtst_pretrain import train_patchtst_pretrain
from modeling_module.training.model_trainers.patchtst_train import train_patchtst
from modeling_module.training.model_trainers.titan_train import train_titan
from modeling_module.utils.exogenous_utils import compose_exo_calendar_cb
from modeling_module.utils.metrics import smape, rmse, mae


# ===================== 공통 유틸 =====================

def _get_part_vocab_size_from_loader(loader) -> int:
    try:
        return len(getattr(loader.dataset, "part_vocab", {}))
    except Exception:
        return 0




def _infer_future_exo_spec_from_loader(loader) -> tuple[bool, int]:
    """Infer whether the loader provides future exogenous features (fe_cont).

    Expected batch format (MultiPartExoDataModule convention):
        (x, y, uid, fe_cont, pe_cont, pe_cat)

    Returns
    -------
    has_fe: bool
        True if the batch includes a non-None fe_cont tensor.
    fe_dim: int
        The last-dimension of fe_cont if present, otherwise 0.

    Notes
    -----
    - has_fe=False means "loader does not provide future exo" (we may rely on future_exo_cb).
    - has_fe=True & fe_dim==0 means "loader provides fe_cont but its feature dimension is 0",
      which is almost always a configuration error when exogenous mode is enabled.
    """
    try:
        b = next(iter(loader))
        if not isinstance(b, (list, tuple)) or len(b) < 4:
            return (False, 0)
        fe = b[3]
        if fe is None:
            return (False, 0)
        if hasattr(fe, 'ndim') and fe.ndim == 3:
            return (True, int(fe.shape[-1]))
        if hasattr(fe, 'ndim') and fe.ndim == 2:
            # (H, E) 형태로 제공되는 경우도 방어
            return (True, int(fe.shape[-1]))
        return (True, 0)
    except Exception:
        return (False, 0)
def save_model(model: torch.nn.Module, cfg, path: str) -> None:
    path = str(path)
    state = {
        "model_state": model.state_dict(),
        "model_class": model.__class__.__name__,
    }
    if cfg is not None:
        if is_dataclass(cfg):
            state["config"] = asdict(cfg)
        else:
            cfg_dict = getattr(cfg, "__dict__", None)
            if cfg_dict is not None:
                state["config"] = dict(cfg_dict)
            else:
                state["config"] = cfg
    torch.save(state, path)
    print(f'{model} save success! {path}')


def _make_ckpt_path(
        save_dir: Path,
        freq: str,
        model_name: str,
        lookback: int,
        horizon: int,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{freq}_{model_name}_L{lookback}_H{horizon}.pt"
    # fname = f"{model_name}.pt"
    return save_dir / fname


def _build_common_train_configs(
        *,
        device: str,
        lookback: int,
        horizon: int,
        freq: str = 'weekly',
        warmup_epochs: Optional[int] = None,
        spike_epochs: Optional[int] = None,
        base_lr: Optional[float] = None,
) -> Tuple[TrainingConfig, TrainingConfig, SpikeLossConfig, Tuple[StageConfig, StageConfig]]:


    # 주기별 학습률/에폭 튜닝
    if warmup_epochs is None:
        if freq == 'hourly':
            warmup_epochs = 20
        elif freq == 'daily':
            warmup_epochs = 30
        else:
            warmup_epochs = 10

    if spike_epochs is None:
        if freq == 'hourly':
            spike_epochs = 50
        elif freq == 'daily':
            spike_epochs = 70
        else:
            spike_epochs = 10

    if base_lr is None:
        if freq == 'hourly':
            base_lr = 1e-4
        elif freq == 'daily':
            base_lr = 2e-4
        else:
            base_lr = 3e-4

    # 1) 2-stage 학습 스케줄
    stg_warmup = StageConfig(
        epochs=warmup_epochs,
        spike_enabled=False,
        lr=base_lr,
        use_horizon_decay=False,
    )

    stg_spike = StageConfig(
        epochs=spike_epochs,
        spike_enabled=True,
        lr=base_lr / 3,
        use_horizon_decay=True,
        tau_h=0.85,
    )
    stages = (stg_warmup, stg_spike)

    # 2) Spike loss 공통 설정
    spike_cfg = SpikeLossConfig(
        enabled=True,
        strategy='mix',
        huber_delta=0.6,
        asym_up_weight=1.0,
        asym_down_weight=2.0,
        mad_k=1.5,
        w_spike=4.0,
        w_norm=1.0,
        alpha_huber=0.6,
        beta_asym=0.4,
        mix_with_baseline=False,
        gamma_baseline=0.0,
    )

    # 3) Point용 공통 TrainingConfig
    point_train_cfg = TrainingConfig(
        device=device,
        lookback=lookback,
        horizon=horizon,
        lr=base_lr,
        weight_decay=1e-3,
        t_max=40,
        patience=100,
        loss_mode='point',
        point_loss='huber',
        huber_delta=0.8,
        use_intermittent=True,
        alpha_zero=3.0,
        alpha_pos=1.0,
        gamma_run=0.3,
        use_horizon_decay=False,
        tau_h=0.85,
        val_use_weights=False,
        spike_loss=spike_cfg,
        max_grad_norm=30.0,
    )

    # 4) Quantile용 공통 TrainingConfig
    quantile_train_cfg = TrainingConfig(
        device=device,
        lookback=lookback,
        horizon=horizon,
        lr=base_lr,
        weight_decay=1e-4,
        t_max=40,
        patience=20,
        loss_mode='quantile',
        quantiles=(0.1, 0.5, 0.9),
        use_intermittent=True,
        alpha_zero=1.2,
        alpha_pos=1.0,
        gamma_run=0.6,
        use_horizon_decay=False,
        tau_h=0.85,
        val_use_weights=False,
        spike_loss=spike_cfg,
        max_grad_norm=30.0,
    )

    return point_train_cfg, quantile_train_cfg, spike_cfg, stages


def _norm_list(xs: Optional[Iterable[str]]) -> List[str]:
    if xs is None:
        return []
    return [str(x).strip().lower() for x in xs if str(x).strip()]

def _contains(xs: List[str], key: str) -> bool:
    return key.lower() in xs


# ---------------------------------------------------------------------
# 모델별 러너: “(freq 공통 계산 결과) + loaders”만 받아서 results를 채움
# ---------------------------------------------------------------------
def _run_patchtst(
        *,
        results: Dict[str, Dict],
        freq: str,
        train_loader, val_loader,
        save_root,
        lookback: int, horizon: int,
        future_exo_cb,
        exo_dim: int,
        patch_len: int, stride: int,
        point_train_cfg, quantile_train_cfg,
        stages,
        device: str,
        use_exogenous_mode: bool = True,
        use_ssl_pretrain: bool = False,
        ssl_pretrain_epochs: int = 10,
        ssl_mask_ratio: float = 0.3,
        ssl_loss_type: str = "mse",
        ssl_freeze_encoder_before_ft: bool = False,
):
    pt_kwargs = dict(
        device=device,
        lookback=lookback,
        horizon=horizon,
        c_in=1,
        d_model=256,
        n_layers=3,
        patch_len=patch_len,
        stride=stride,
        d_future=exo_dim,     # future exo dim (A0=0, A1=2, A2=3 ...)
        use_revin=True,
    )

    # >>> PATCH START: infer past_exo dims from loader batch and inject into pt_kwargs
    # 목적: backbone init 시점에 d_past_cont/d_past_cat가 0으로 고정되는 문제 해결
    # 전제: loader batch tuple = (x, y, uid, fe, pe_cont, pe_cat)
    d_past_cont = 0
    d_past_cat = 0
    try:
        b = next(iter(train_loader))
        if isinstance(b, (list, tuple)) and len(b) >= 6:
            # x, y, uid, fe, pe_cont, pe_cat
            pe_cont = b[4]
            pe_cat = b[5]

            if pe_cont is not None and hasattr(pe_cont, "ndim") and pe_cont.ndim == 3:
                d_past_cont = int(pe_cont.shape[-1])
            if pe_cat is not None and hasattr(pe_cat, "ndim") and pe_cat.ndim == 3:
                d_past_cat = int(pe_cat.shape[-1])
        else:
            print(f"[DBG-pt_kwargs] unexpected batch format: type={type(b)} len={len(b) if hasattr(b,'__len__') else 'NA'}")
    except Exception as e:
        print(f"[DBG-pt_kwargs] failed to infer past_exo dims: {repr(e)}")
        d_past_cont, d_past_cat = 0, 0

    # 안전: use_exogenous_mode=False면 future exo 차원도 강제로 0으로
    if not use_exogenous_mode:
        pt_kwargs["d_future"] = 0

    pt_kwargs["d_past_cont"] = d_past_cont
    pt_kwargs["d_past_cat"] = d_past_cat

    print(
        f"[DBG-pt_kwargs] use_exogenous_mode={use_exogenous_mode} | "
        f"d_future={pt_kwargs['d_future']} | d_past_cont={d_past_cont} | d_past_cat={d_past_cat}"
    )
    # <<< PATCH END


    # ---------------------------
    # (NEW) 0) self-supervised pretrain (optional)
    # ---------------------------
    pretrain_ckpt_path = None
    if use_ssl_pretrain and save_root is not None:
        pretrain_dir = Path(save_root) / "pretrain"
        pretrain_dir.mkdir(parents=True, exist_ok=True)
        pretrain_ckpt_path = str(pretrain_dir / "patchtst_pretrain_best.pt")

        pt_pre_cfg = PatchTSTConfig(**pt_kwargs, loss_mode="point", point_loss="mse")
        pre_model = PatchTSTPretrainModel(cfg=pt_pre_cfg)

        pre_train_cfg = point_train_cfg
        pre_stages = [StageConfig(epochs=ssl_pretrain_epochs, lr=point_train_cfg.lr, spike_enabled=False)]

        print(f"[SSL] PatchTST Pretrain ({freq.capitalize()})")
        _ = train_patchtst_pretrain(
            pre_model,
            train_loader,
            val_loader,
            train_cfg=pre_train_cfg,
            stages=pre_stages,
            mask_ratio=ssl_mask_ratio,
            loss_type=ssl_loss_type,
            save_dir=str(pretrain_dir),
            ckpt_name="patchtst_pretrain_best.pt",
        )

    # ---------------------------
    # 1) PatchTST Base (supervised)
    # ---------------------------
    pt_base_cfg = PatchTSTConfig(**pt_kwargs, loss_mode='point', point_loss='huber')
    pt_base = build_patchTST_base(pt_base_cfg)

    print(f'PatchTST Base ({freq.capitalize()})')
    if pretrain_ckpt_path is not None:
        best_pt_base = train_patchtst_finetune(
            pt_base, train_loader, val_loader,
            train_cfg=point_train_cfg, stages=list(stages),
            future_exo_cb=future_exo_cb,
            exo_is_normalized=True,
            pretrain_ckpt_path=pretrain_ckpt_path,
            load_strict=False,
            freeze_encoder_before_ft=ssl_freeze_encoder_before_ft,
        )
    else:
        best_pt_base = train_patchtst(
            pt_base, train_loader, val_loader,
            train_cfg=point_train_cfg, stages=list(stages),
            future_exo_cb=future_exo_cb,
            use_exogenous_mode=use_exogenous_mode
        )

    if save_root:
        print('save_root:: ', save_root)
        ckpt_path = _make_ckpt_path(save_root, freq, "PatchTSTBase", lookback, horizon)
        save_model(pt_base, pt_base_cfg, ckpt_path)
        best_pt_base["ckpt_path"] = str(ckpt_path)
        if pretrain_ckpt_path is not None:
            best_pt_base["pretrain_ckpt_path"] = str(pretrain_ckpt_path)
    results['PatchTST Base'] = best_pt_base

    # ---------------------------
    # 2) PatchTST Quantile (supervised)
    # ---------------------------
    pt_q_cfg = PatchTSTConfig(**pt_kwargs, loss_mode='quantile', quantiles=(0.1, 0.5, 0.9))
    pt_q = build_patchTST_quantile(pt_q_cfg)

    print(f'PatchTST Quantile ({freq.capitalize()})')
    if pretrain_ckpt_path is not None:
        best_pt_q = train_patchtst_finetune(
            pt_q, train_loader, val_loader,
            train_cfg=quantile_train_cfg, stages=list(stages),
            future_exo_cb=future_exo_cb,
            exo_is_normalized=True,
            pretrain_ckpt_path=pretrain_ckpt_path,
            load_strict=False,
            freeze_encoder_before_ft=ssl_freeze_encoder_before_ft,
        )
    else:
        best_pt_q = train_patchtst(
            pt_q, train_loader, val_loader,
            train_cfg=quantile_train_cfg, stages=list(stages),
            future_exo_cb=future_exo_cb,
            exo_is_normalized=True,
            use_exogenous_mode=use_exogenous_mode
        )

    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, "PatchTSTQuantile", lookback, horizon)
        save_model(pt_q, pt_q_cfg, ckpt_path)
        best_pt_q["ckpt_path"] = str(ckpt_path)
        if pretrain_ckpt_path is not None:
            best_pt_q["pretrain_ckpt_path"] = str(pretrain_ckpt_path)
    results['PatchTST Quantile'] = best_pt_q

def _run_titan(
        *,
        results: Dict[str, Dict],
        freq: str,
        train_loader, val_loader,
        save_root,
        lookback: int, horizon: int,
        use_exogenous_mode: bool,
        future_exo_cb,
        exo_dim: int,
        patch_len: int, stride: int,
        point_train_cfg, quantile_train_cfg,
        stages,
        device: str,
):
    # --------------------------------------------------
    # 2) Titan
    # --------------------------------------------------
    d_past_cont = 0
    d_past_cat = 0
    try:
        b = next(iter(train_loader))
        if isinstance(b, (list, tuple)) and len(b) >= 6:
            pe_cont = b[4]  # [B,L,Dc]
            pe_cat = b[5]  # [B,L,K]
            if pe_cont is not None and hasattr(pe_cont, "ndim") and pe_cont.ndim == 3:
                d_past_cont = int(pe_cont.shape[-1])
            if pe_cat is not None and hasattr(pe_cat, "ndim") and pe_cat.ndim == 3:
                d_past_cat = int(pe_cat.shape[-1])
    except Exception as e:
        print(f"[DBG-ti_kwargs] failed to infer past_exo dims: {repr(e)}")
        d_past_cont, d_past_cat = 0, 0

    # cat embedding spec (기본값)
    cat_vocab_sizes = tuple([512] * d_past_cat)
    cat_embed_dims = tuple([16] * d_past_cat)
    past_dim_total = d_past_cont + sum(cat_embed_dims)

    # TitanConfig
    ti_config = TitanConfig(
        lookback=lookback,
        horizon=horizon,

        # (중요) Titan encoder input_dim 확장: target(1) + past_exo
        input_dim=1 + past_dim_total,

        d_model=256,
        n_layers=3,
        n_heads=4,
        d_ff=512,
        dropout=0.1,
        contextual_mem_size=256,
        persistent_mem_size=64,

        # Exogenous handling (future)
        use_exogenous_mode= use_exogenous_mode,
        exo_dim=(int(exo_dim) if use_exogenous_mode else 0),

        # Past exo spec
        past_exo_cont_dim=d_past_cont,
        past_exo_cat_dim=d_past_cat,
        past_exo_cat_vocab_sizes=cat_vocab_sizes,
        past_exo_cat_embed_dims=cat_embed_dims,
        past_exo_mode="concat",

        final_clamp_nonneg=True,
        use_revin=True,
        revin_use_std=True,
        revin_subtract_last=False,
        revin_affine=True,
        use_lmm=False,
    )
    if freq == 'hourly':
        ti_config.contextual_mem_size = 512

    # Titan Base
    ti_base = build_titan_base(ti_config)

    best_ti_base = train_titan(
        ti_base, train_loader, val_loader,
        train_cfg=point_train_cfg, stages=list(stages),
        future_exo_cb=(future_exo_cb if use_exogenous_mode else None),
        use_exogenous_mode=use_exogenous_mode,
    )
    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, "TitanBase", lookback, horizon)
        save_model(ti_base, ti_config, ckpt_path)
        best_ti_base["ckpt_path"] = str(ckpt_path)
    results['Titan Base'] = best_ti_base

    # Titan LMM
    ti_config_lmm = replace(ti_config, use_lmm=True)
    ti_lmm = build_titan_lmm(ti_config_lmm)
    print(f'Titan LMM ({freq.capitalize()})')
    best_ti_lmm = train_titan(
        ti_lmm, train_loader, val_loader,
        train_cfg=point_train_cfg, stages=list(stages),
        future_exo_cb=(future_exo_cb if use_exogenous_mode else None),
        use_exogenous_mode=use_exogenous_mode,

    )
    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, "TitanLMM", lookback, horizon)
        save_model(ti_lmm, ti_config_lmm, ckpt_path)
        best_ti_lmm["ckpt_path"] = str(ckpt_path)
    results['Titan LMM'] = best_ti_lmm

    # Titan Seq2Seq
    ti_seq2seq = build_titan_seq2seq(ti_config)
    print(f'Titan Seq2Seq ({freq.capitalize()})')
    best_ti_s2s = train_titan(
        ti_seq2seq, train_loader, val_loader,
        train_cfg=point_train_cfg, stages=list(stages),
        future_exo_cb=(future_exo_cb if use_exogenous_mode else None),
        use_exogenous_mode=use_exogenous_mode,

    )
    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, "TitanSeq2Seq", lookback, horizon)
        save_model(ti_seq2seq, ti_config, ckpt_path)
        best_ti_s2s["ckpt_path"] = str(ckpt_path)
    results['Titan Seq2Seq'] = best_ti_s2s


def _run_patchmixer(
        *,
        results: Dict[str, Dict],
        freq: str,
        train_loader, val_loader,
        save_root,
        lookback: int, horizon: int,
        future_exo_cb,
        exo_dim: int,
        patch_len: int, stride: int,
        use_exogenous_mode: bool,
        point_train_cfg, quantile_train_cfg,
        stages,
        device: str,
):
    # --------------------------------------------------
    # 1) PatchMixer
    # --------------------------------------------------
    # 주기별 Patch Len / Stride 추천값
    if freq == 'hourly':
        patch_len, stride = 24, 12
        season_period = 24
    elif freq == 'daily':
        patch_len, stride = 14, 7
        season_period = 7
    elif freq == 'weekly':
        patch_len, stride = 12, 8
        season_period = 52
    else:  # monthly
        patch_len, stride = 6, 3
        season_period = 12
    # Config kwargs 구성
    pm_kwargs = dict(
        lookback=lookback,
        horizon=horizon,
        device=device,
        enc_in=1,
        d_model=64,
        e_layers=3,
        patch_len=patch_len,
        stride=stride,
        f_out=128,
        head_hidden=128,
        exo_dim=exo_dim,
        use_part_embedding=True,
        part_vocab_size=_get_part_vocab_size_from_loader(train_loader),
        part_embed_dim=16,
        final_nonneg=True,
        use_eol_prior=False,
        # exo_is_normalized_default=(freq != 'monthly'),  # 월간만 False 권장
        exo_is_normalized_default=False,  # 월간만 False 권장
        expander_season_period=season_period,
        expander_n_harmonics=min(season_period // 2, 16),
    )

    d_past_cont = 0
    d_past_cat = 0
    try:
        b = next(iter(train_loader))
        if isinstance(b, (list, tuple)) and len(b) >= 6:
            # x, y, uid, fe, pe_cont, pe_cat
            pe_cont = b[4]
            pe_cat = b[5]

            if pe_cont is not None and hasattr(pe_cont, "ndim") and pe_cont.ndim == 3:
                d_past_cont = int(pe_cont.shape[-1])
            if pe_cat is not None and hasattr(pe_cat, "ndim") and pe_cat.ndim == 3:
                d_past_cat = int(pe_cat.shape[-1])
        else:
            print(
                f"[DBG-pt_kwargs] unexpected batch format: type={type(b)} len={len(b) if hasattr(b, '__len__') else 'NA'}")
    except Exception as e:
        print(f"[DBG-pt_kwargs] failed to infer past_exo dims: {repr(e)}")
        d_past_cont, d_past_cat = 0, 0

    # Base
    pm_base_cfg = PatchMixerConfig(**pm_kwargs, loss_mode='point', point_loss='huber', head_dropout=0.05)
    pm_base_cfg.learn_output_scale = False  # 초기 스케일 안정화
    pm_base_cfg.learn_dw_gain = False  # 초기 스케일 안정화
    pm_base_cfg.exo_is_normalized_default = False  # future_exo를 표준화했다는 전제
    pm_base_cfg.past_exo_mode = "z_gate"
    pm_base_cfg.past_exo_cont_dim = d_past_cont
    pm_base_cfg.past_exo_cat_dim = d_past_cat
    pm_base_cfg.past_exo_cat_vocab_sizes = (512, 128)  # K개
    pm_base_cfg.past_exo_cat_embed_dims = (16, 16)  # K개
    pm_base_model = build_patch_mixer_base(pm_base_cfg)


    print(f'PatchMixer Base ({freq.capitalize()})')
    best_pm_base = train_patchmixer(
        pm_base_model, train_loader, val_loader,
        train_cfg=point_train_cfg, stages=list(stages),
        future_exo_cb=(future_exo_cb if use_exogenous_mode else None),
        exo_is_normalized=pm_base_cfg.exo_is_normalized_default,
        use_exogenous_mode=use_exogenous_mode,

    )
    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, "PatchMixerBase", lookback, horizon)
        save_model(pm_base_model, pm_base_cfg, ckpt_path)
        best_pm_base["ckpt_path"] = str(ckpt_path)
    results['PatchMixer Base'] = best_pm_base

    # Quantile
    pm_q_cfg = PatchMixerConfig(**pm_kwargs, loss_mode='quantile', quantiles=(0.1, 0.5, 0.9), head_dropout=0.02)
    pm_q_cfg.learn_output_scale = False
    pm_q_cfg.learn_dw_gain = False
    pm_q_cfg.exo_is_normalized_default = False
    pm_base_cfg.past_exo_mode = "z_gate"
    pm_base_cfg.past_exo_cont_dim = d_past_cont
    pm_base_cfg.past_exo_cat_dim = d_past_cat
    pm_base_cfg.past_exo_cat_vocab_sizes = (512, 128)  # K개
    pm_base_cfg.past_exo_cat_embed_dims = (16, 16)  # K개
    pm_q_model = build_patch_mixer_quantile(pm_q_cfg)

    print(f'PatchMixer Quantile ({freq.capitalize()})')
    best_pm_q = train_patchmixer(
        pm_q_model, train_loader, val_loader,
        train_cfg=quantile_train_cfg, stages=list(stages),
        future_exo_cb=(future_exo_cb if use_exogenous_mode else None),
        exo_is_normalized=pm_q_cfg.exo_is_normalized_default,
        use_exogenous_mode=use_exogenous_mode,

    )
    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, "PatchMixerQuantile", lookback, horizon)
        save_model(pm_q_model, pm_q_cfg, ckpt_path)
        best_pm_q["ckpt_path"] = str(ckpt_path)
    results['PatchMixer Quantile'] = best_pm_q

MODEL_REGISTRY: Dict[str, Callable] = {
    "patchtst": _run_patchtst,
    "titan": _run_titan,
    "patchmixer": _run_patchmixer,
}

# ===================== GENERIC RUNNER (통합) =====================
def _run_total_train_generic(
    train_loader,
    val_loader,
    device: str,
    lookback: int,
    horizon: int,
    freq: str,
    save_dir: Optional[str],
    *,
    use_exogenous_mode: Optional[bool] = False,
    models_to_run: Optional[Iterable[str]] = None,
    warmup_epochs: Optional[int] = None,
    spike_epochs: Optional[int] = None,
    base_lr: Optional[float] = None,


    # PatchTST 전용 Property
    use_ssl_pretrain: bool = False,
    ssl_pretrain_epochs: int = 10,
    ssl_mask_ratio: float = 0.3,
    ssl_loss_type: str = "mse",
    ssl_freeze_encoder_before_ft: bool = False
):
    save_root = Path(save_dir) if save_dir is not None else None

    point_train_cfg, quantile_train_cfg, spike_cfg, stages = _build_common_train_configs(
        device=device, lookback=lookback, horizon=horizon, freq=freq,
        warmup_epochs= warmup_epochs, spike_epochs = spike_epochs, base_lr = base_lr
    )

    date_type_map = {"weekly": "W", "monthly": "M", "daily": "D", "hourly": "H"}
    dt_char = date_type_map.get(freq, "W")

    # --------------------------------------------------------------
    # Exogenous feature policy (TRAIN)
    # --------------------------------------------------------------
    # We support two ways of supplying "future exogenous" features:
    #   (A) Loader-provided: batch[3] == fe_cont with shape (B, H, E)
    #   (B) Callback-provided: future_exo_cb(t0, H) -> (H, E)
    #
    # IMPORTANT:
    # - If loader provides fe_cont, we prefer (A) and set future_exo_cb=None
    #   to avoid inconsistent dim between loader and calendar callback.
    # - If loader does NOT provide fe_cont, we use (B) when use_exogenous_mode=True.
    # - If loader provides fe_cont but E==0, that is treated as a configuration error.
    has_fe, fe_dim = _infer_future_exo_spec_from_loader(train_loader)

    if use_exogenous_mode:
        if has_fe:
            if fe_dim <= 0:
                raise RuntimeError(
                    f"[total_train] use_exogenous_mode=True but loader fe_cont dim is {fe_dim}. "
                    f"Check feature selection / exogenous datamodule wiring."
                )
            # Prefer loader-provided exo
            future_exo_cb = None
            exo_dim = int(fe_dim)
            print(f"[total_train] future exo from loader: fe_dim={exo_dim} (freq={freq})")
        else:
            # Loader does not provide fe_cont -> use calendar callback
            future_exo_cb = compose_exo_calendar_cb(date_type=dt_char)
            exo_dim = 4 if freq in ("daily", "hourly") else 2
            print(f"[total_train] future exo from callback: exo_dim={exo_dim} (freq={freq}, dt={dt_char})")
    else:
        # Exogenous disabled
        future_exo_cb = None
        exo_dim = 0
        if has_fe and fe_dim > 0:
            print(
                f"[total_train][WARN] use_exogenous_mode=False but loader provides fe_cont dim={fe_dim}. "
                f"Ignoring future exo."
            )

    # freq별 patch_len/stride/season_period
    if freq == "hourly":
        patch_len, stride, season_period = 24, 12, 24
    elif freq == "daily":
        patch_len, stride, season_period = 14, 7, 7
    elif freq == "weekly":
        patch_len, stride, season_period = 12, 8, 52
    else:
        patch_len, stride, season_period = 6, 3, 12

    # NEW: 선택값 정규화
    selected = _norm_list(models_to_run)
    if not selected:
        selected = ["patchtst"]  # 기본값(원하시면 전체로 바꿔도 됨)

    # 검증
    unknown = [m for m in selected if m not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown models_to_run: {unknown}. allowed={list(MODEL_REGISTRY.keys())}")

    results: Dict[str, Dict] = {}

    # 선택된 모델만 실행
    for m in selected:
        print(f"\n[total_train] === RUN: {m} ({freq}) ===")
        kwargs = dict(
            results=results,
            freq=freq,
            train_loader=train_loader,
            val_loader=val_loader,
            save_root=save_root,
            lookback=lookback,
            horizon=horizon,
            future_exo_cb=(future_exo_cb if use_exogenous_mode else None),
            exo_dim=exo_dim,
            patch_len=patch_len,
            stride=stride,
            point_train_cfg=point_train_cfg,
            quantile_train_cfg=quantile_train_cfg,
            stages=stages,
            device=device,
            use_exogenous_mode = use_exogenous_mode,
        )

        if m == "patchtst":
            kwargs.update(dict(
                use_ssl_pretrain=use_ssl_pretrain,
                ssl_pretrain_epochs=ssl_pretrain_epochs,
                ssl_mask_ratio=ssl_mask_ratio,
                ssl_loss_type=ssl_loss_type,
                ssl_freeze_encoder_before_ft=ssl_freeze_encoder_before_ft,
            ))

        MODEL_REGISTRY[m](**kwargs)

    return results


# ===================== EXPORTED FUNCTIONS =====================

def run_total_train_weekly(
        train_loader,
        val_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        *,
        lookback,
        horizon,
        warmup_epochs = None,
        spike_epochs = None,
        base_lr = None,
        save_dir=None,
        use_exogenous_mode: bool = False,
        models_to_run=None,
        use_ssl_pretrain: bool = False,
        ssl_pretrain_epochs: int = 10,
        ssl_mask_ratio: float = 0.3,
        ssl_loss_type: str = "mse",
        ssl_freeze_encoder_before_ft: bool = False
):
    return _run_total_train_generic(
        train_loader,
        val_loader,
        device,
        lookback,
        horizon,
        'weekly',
        save_dir,
        use_exogenous_mode=use_exogenous_mode,
        warmup_epochs= warmup_epochs,
        spike_epochs = spike_epochs,
        base_lr = base_lr,
        models_to_run=models_to_run,
        use_ssl_pretrain=use_ssl_pretrain,
        ssl_pretrain_epochs=ssl_pretrain_epochs,
        ssl_mask_ratio=ssl_mask_ratio,
        ssl_loss_type=ssl_loss_type,
        ssl_freeze_encoder_before_ft=ssl_freeze_encoder_before_ft,
    )


def run_total_train_monthly(
        train_loader,
        val_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        *,
        lookback,
        horizon,
        warmup_epochs = None,
        spike_epochs = None,
        base_lr = None,
        save_dir=None,
        models_to_run=None,
        use_ssl_pretrain: bool = False,
        ssl_pretrain_epochs: int = 10,
        ssl_mask_ratio: float = 0.3,
        ssl_loss_type: str = "mse",
        ssl_freeze_encoder_before_ft: bool = False
):
    return _run_total_train_generic(
        train_loader,
        val_loader,
        device,
        lookback,
        horizon,
        'monthly',
        save_dir,
        warmup_epochs= warmup_epochs,
        spike_epochs = spike_epochs,
        base_lr = base_lr,
        models_to_run=models_to_run,
        use_ssl_pretrain=use_ssl_pretrain,
        ssl_pretrain_epochs=ssl_pretrain_epochs,
        ssl_mask_ratio=ssl_mask_ratio,
        ssl_loss_type=ssl_loss_type,
        ssl_freeze_encoder_before_ft=ssl_freeze_encoder_before_ft,
    )


def run_total_train_daily(
        train_loader,
        val_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        *,
        lookback,
        horizon,
        warmup_epochs = None,
        spike_epochs = None,
        base_lr = None,
        save_dir=None,
        models_to_run=None,
        use_ssl_pretrain: bool = False,
        ssl_pretrain_epochs: int = 10,
        ssl_mask_ratio: float = 0.3,
        ssl_loss_type: str = "mse",
        ssl_freeze_encoder_before_ft: bool = False
):
    return _run_total_train_generic(
        train_loader,
        val_loader,
        device,
        lookback,
        horizon,
        'daily',
        save_dir,
        warmup_epochs= warmup_epochs,
        spike_epochs = spike_epochs,
        base_lr = base_lr,
        models_to_run=models_to_run,
        use_ssl_pretrain=use_ssl_pretrain,
        ssl_pretrain_epochs=ssl_pretrain_epochs,
        ssl_mask_ratio=ssl_mask_ratio,
        ssl_loss_type=ssl_loss_type,
        ssl_freeze_encoder_before_ft=ssl_freeze_encoder_before_ft,
    )


def run_total_train_hourly(
        train_loader,
        val_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        *,
        lookback,
        horizon,
        warmup_epochs = None,
        spike_epochs = None,
        base_lr = None,
        save_dir=None,
        models_to_run=None,
        use_ssl_pretrain: bool = False,
        ssl_pretrain_epochs: int = 10,
        ssl_mask_ratio: float = 0.3,
        ssl_loss_type: str = "mse",
        ssl_freeze_encoder_before_ft: bool = False
):
    return _run_total_train_generic(
        train_loader,
        val_loader,
        device,
        lookback,
        horizon,
        'hourly',
        save_dir,
        warmup_epochs= warmup_epochs,
        spike_epochs = spike_epochs,
        base_lr = base_lr,
        models_to_run=models_to_run,
        use_ssl_pretrain=use_ssl_pretrain,
        ssl_pretrain_epochs=ssl_pretrain_epochs,
        ssl_mask_ratio=ssl_mask_ratio,
        ssl_loss_type=ssl_loss_type,
        ssl_freeze_encoder_before_ft=ssl_freeze_encoder_before_ft,
    )


def summarize_metrics(results: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, float]]:
    table: Dict[str, Dict[str, float]] = {}
    for name, res in results.items():
        y = res['y_true'].reshape(-1)
        yhat = res['y_pred'].reshape(-1)

        row: Dict[str, float] = {
            'MAE': mae(y, yhat),
            'RMSE': rmse(y, yhat),
            'SMAPE': smape(y, yhat),
        }
        if res.get('q_pred') is not None and 0.1 in res['q_pred'] and 0.9 in res['q_pred']:
            result = quantile_metrics(y, yhat)
            row['converage_per_q'] = result['coverage_per_q']
            row['i80_cov'] = result['i80_cov']
            row['i80_wid'] = result['i80_wid']
        table[name] = row
    return table
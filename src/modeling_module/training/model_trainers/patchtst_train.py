# modeling_module/training/model_trainers/patchtst_train.py
from __future__ import annotations

import copy
from typing import Optional, Callable

# PatchTST 내부 head 재구성에 필요
from modeling_module.models.PatchTST.common.patching import compute_patch_num
from modeling_module.models.PatchTST.heads.distribution_head import DistHeadWithExo
from modeling_module.models.PatchTST.heads.point_head import PointHeadWithExo
from modeling_module.models.PatchTST.heads.quantile_head import QuantileHeadWithExo
from modeling_module.training.adapters import PatchTSTAdapter
from modeling_module.training.config import TrainingConfig, StageConfig, apply_stage
from modeling_module.training.engine import CommonTrainer
from modeling_module.training.model_trainers.amp_policy import amp_type_set
from modeling_module.training.model_trainers.exo_policy import infer_future_exo_spec_from_loader, infer_exo_dim_from_cb
from modeling_module.training.model_trainers.loss_policy import infer_loss_mode
from modeling_module.training.model_trainers.spike_policy import maybe_make_spike_loader


def _ensure_patchtst_future_head(model, exo_dim: int, *, loss_mode: str = "point"):
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        return model

    current = int(getattr(cfg, "future_exo_dim", getattr(cfg, "d_future", 0)))
    cfg.future_exo_dim = int(exo_dim) if exo_dim > 0 else 0
    cfg.d_future = int(cfg.future_exo_dim)
    head_future_dim = 0

    patch_num = compute_patch_num(cfg.lookback, cfg.patch_len, cfg.stride, cfg.padding_patch)

    if hasattr(model, "_rebuild_future_exo_path"):
        model._rebuild_future_exo_path(int(cfg.future_exo_dim))

    # ---- dist 우선 처리 ----
    if loss_mode == "dist":
        out_mult = int(cfg.loss.outputsize_multiplier) if hasattr(cfg.loss, 'outputsize_multiplier') else 2
        # 이미 dist head면 유지(단, d_future 변경 반영 필요 시 rebuild)
        model.head = DistHeadWithExo(
            d_model=cfg.d_model,
            horizon=cfg.horizon,
            d_future=int(head_future_dim),
            act=getattr(cfg, "act", "gelu"),
            out_mult=out_mult
        )
        print(
            f"[train_patchtst] dist head rebuilt: future_exo_dim {current} -> {cfg.future_exo_dim} "
            f"(head_d_future={head_future_dim})"
        )
        return model

    # ---- 기존 quantile / point ----
    if getattr(model, "is_quantile", False):
        model.head = QuantileHeadWithExo(
            d_model=cfg.d_model,
            horizon=cfg.horizon,
            d_future=int(head_future_dim),
            quantiles=getattr(cfg, "quantiles", (0.1, 0.5, 0.9)),
            hidden=getattr(cfg, "q_hidden", 128),
            monotonic=True,
        )
        print(
            f"[train_patchtst] quantile head rebuilt: future_exo_dim {current} -> {cfg.future_exo_dim} "
            f"(head_d_future={head_future_dim})"
        )
    else:
        model.head = PointHeadWithExo(
            d_model=cfg.d_model,
            horizon=cfg.horizon,
            d_future=int(head_future_dim),
            patch_num=patch_num,
            agg=getattr(model.head, "agg", "mean"),
        )
        print(
            f"[train_patchtst] point head rebuilt: future_exo_dim {current} -> {cfg.future_exo_dim} "
            f"(head_d_future={head_future_dim})"
        )

    return model



def train_patchtst(
        model,
        train_loader,
        val_loader,
        device,
        *,
        stages: list[StageConfig] | None = None,
        train_cfg: Optional[TrainingConfig] = None,
        future_exo_cb: Optional[Callable] = None,
):
    """
    PatchTST 모델 학습 진입점(Entry Point).

    기능:
    - 외생 변수(Exogenous Variable) 차원 자동 추론 및 헤드 조정.
    - AMP(Automatic Mixed Precision) 환경 구성.
    - CommonTrainer를 이용한 스테이지별(Stage-wise) 학습 루프 실행.
    """
    assert train_cfg is not None, "train_cfg는 필수입니다."
    use_exogenous_mode = getattr(train_cfg, 'use_exogenous_mode', True)
    exo_is_normalized = getattr(train_cfg, 'exo_is_normalized', False)
    # ----------
    # (1) exo dim inference
    # ----------
    horizon = getattr(model, 'horizon', None) or getattr(train_cfg, 'horizon', None)
    if future_exo_cb is not None:
        if horizon is None:
            raise ValueError('horizon을 model 또는 train_cfg에서 찾을 수 없습니다.')


    E_loader = infer_future_exo_spec_from_loader(train_loader)[1]
    E_cb = infer_exo_dim_from_cb(future_exo_cb, horizon, device="cpu")

    # 실제 학습 입력 기준으로 head를 맞추는 것이 안전
    E = E_loader if E_loader > 0 else E_cb

    loss_mode = infer_loss_mode(train_cfg)
    print(f'[train_patchtst] loss_mode: {loss_mode}')


    if not use_exogenous_mode:
        if future_exo_cb is not None:
            print("[train_patchtst][WARN] use_exogenous_mode=False so future_exo_cb is force-disabled.")
        future_exo_cb = None
        E_loader, E_cb, E = 0, 0, 0

        model = _ensure_patchtst_future_head(model, 0, loss_mode=loss_mode)
    else:
        horizon = getattr(model, "horizon", None) or getattr(train_cfg, "horizon", None)
        if future_exo_cb is not None and horizon is None:
            raise ValueError("horizon을 model 또는 train_cfg에서 찾을 수 없습니다.")

        E_loader = infer_future_exo_spec_from_loader(train_loader)[1]
        E_cb = infer_exo_dim_from_cb(future_exo_cb, horizon, device="cpu")
        E = int(E_loader) if int(E_loader) > 0 else int(E_cb)

        model = _ensure_patchtst_future_head(model, E, loss_mode=loss_mode)

        # loader가 fe_cont를 주면 callback 끄기(중복 방지)
        if int(E_loader) > 0 and future_exo_cb is not None:
            future_exo_cb = None
            print(f"[train_patchtst] loader provides fe_cont(E={E_loader}), so future_exo_cb disabled.")

    print(
        f"[EXO-train] inferred E={E} | future_exo_cb? {future_exo_cb is not None} | exo_is_normalized={exo_is_normalized}")

    # # loader가 fe_cont를 주는 경우, 자동 CB는 끄는 쪽이 안전 (중복/불일치 방지)
    # if E_loader > 0 and future_exo_cb is not None:
    #     future_exo_cb = None
    #     print(f"[train_patchtst] loader provides fe_cont(E={E_loader}), so future_exo_cb disabled.")

    # 3) AMP 설정 (Titan/PM과 동일 패턴)
    amp_device, amp_enabled, amp_dtype = amp_type_set(train_cfg)

    autocast_input = dict(device_type=amp_device, enabled=amp_enabled, dtype=amp_dtype)

    # 4) stages 구성 (기본 1 스테이지)
    if not stages or len(stages) == 0:
        stages = [StageConfig(epochs=train_cfg.epochs, spike_enabled=train_cfg.spike_loss.enabled)]

    adapter = PatchTSTAdapter()

    is_production_refit = (
        getattr(train_cfg, "training_mode", "qualification")
        == "production_refit"
    )
    if is_production_refit and val_loader is not None:
        raise ValueError("production_refit requires val_loader=None.")

    best = None
    global_best_loss = float("inf")
    global_best_state = (
        None if is_production_refit else copy.deepcopy(model.state_dict())
    )
    global_best_cfg = train_cfg
    total_epochs_completed = 0
    for i, stg in enumerate(stages, 1):
        # 스테이지별 설정 적용
        cfg_i = apply_stage(train_cfg, stg)
        print(f"\n[train_patchtst] ===== Stage {i}/{len(stages)} =====")
        print(f"  - spike: {'ON' if cfg_i.spike_loss.enabled else 'OFF'}")
        print(f"  - epochs: {cfg_i.epochs} | lr={cfg_i.lr} | horizon_decay={cfg_i.use_horizon_decay}")
        from modeling_module.training.model_trainers.cfg_policy import dump_cfg
        dump_cfg(cfg_i, name = 'patchtst_train')

        tl_i = maybe_make_spike_loader(train_loader, enable=cfg_i.spike_loss.enabled)

        # 트레이너 초기화 및 학습 수행
        trainer = CommonTrainer(
            cfg=cfg_i,
            adapter=adapter,
            future_exo_cb=future_exo_cb,
            logger=print,
            autocast_input=autocast_input,
            extra_loss_fn=None,
            use_exogenous_mode=use_exogenous_mode,
            device = device
        )
        model = trainer.fit(model, tl_i, val_loader, tta_steps=0)
        total_epochs_completed += int(getattr(trainer, "epochs_completed_", 0))
        if is_production_refit:
            best = {
                "model": model,
                "cfg": cfg_i,
                "best_val_loss": None,
                "final_train_loss": float(
                    getattr(trainer, "final_train_loss_", float("nan"))
                ),
                "epochs_completed": total_epochs_completed,
                "state_selection": "final_epoch",
            }
            continue

        stage_best_loss = float(getattr(trainer, "best_loss_", float("inf")))
        if stage_best_loss < global_best_loss:
            global_best_loss = stage_best_loss
            global_best_state = copy.deepcopy(model.state_dict())
            global_best_cfg = cfg_i
        best = {"model": model, "cfg": cfg_i, "best_val_loss": stage_best_loss}

    if not is_production_refit:
        assert global_best_state is not None
        model.load_state_dict(global_best_state)
        best = {"model": model, "cfg": global_best_cfg, "best_val_loss": global_best_loss}

    print(
        f"[EXO-train] inferred E={E} | future_exo_cb? {future_exo_cb is not None} | exo_is_normalized={exo_is_normalized}")
    return best

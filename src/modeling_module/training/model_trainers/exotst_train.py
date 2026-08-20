import copy
from contextlib import contextmanager
from typing import Optional, Callable

import torch
import torch.nn as nn

from modeling_module.models.ExoTST.backbone import HorizonDistMLPHead, HorizonMLPHead
from modeling_module.training.adapters import DefaultAdapter
from modeling_module.training.config import TrainingConfig, StageConfig, apply_stage
from modeling_module.training.engine import CommonTrainer
from modeling_module.training.model_trainers.amp_policy import amp_type_set
from modeling_module.training.model_trainers.loss_policy import infer_loss_mode
from modeling_module.training.model_trainers.spike_policy import maybe_make_spike_loader
from modeling_module.training.model_trainers.exo_policy import (
    ExoSpec,
    infer_future_exo_spec_from_loader,
    infer_past_exo_dim_from_loader_for_exotst,
    infer_exo_dim_from_cb,
    wrap_future_exo_cb,
)

def _ensure_exotst_loss_head(model, train_cfg: TrainingConfig):
    """
        ExoTST는 head 출력 shape가 loss에 종속되므로,
        train_cfg.loss_mode(또는 auto 추론)에 맞춰 model.head를 동기화합니다.

        가정:
          - model.cfg 존재
          - model.ny, model.horizon, model.y_dim 존재
          - dist의 경우: HorizonDistMLPHead를 사용 (B,H,out_mult) 형태
        """
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        return

    loss_mode = infer_loss_mode(train_cfg)
    loss_obj = getattr(train_cfg, "loss", None) or getattr(cfg, "loss", None)

    # cfg.loss도 trainer loss와 맞춰두는 편이 디버깅/재현성 측면에서 안전
    if loss_obj is not None:
        cfg.loss = loss_obj
        model.loss = loss_obj

    if loss_mode == "quantile":
        raise NotImplementedError(
            "[train_exotst] quantile loss_mode requested, "
            "but ExoTST quantile head is not implemented yet."
        )

    if loss_mode == "dist":
        # DistributionLoss 컨벤션: outputsize_multiplier, param_names
        out_mult = int(getattr(loss_obj, "outputsize_multiplier", 2)) if loss_obj is not None else 2
        param_names = list(getattr(loss_obj, "param_names", [])) if loss_obj is not None else None

        model.loss_type = "distribution"
        model.out_mult = out_mult
        model.param_names = param_names

        # dist는 현재 ExoTST 구현상 y_dim==1만 지원하는 쪽이 안전
        if int(getattr(cfg, "y_dim", getattr(model, "y_dim", 1))) != 1:
            raise RuntimeError("[train_exotst] dist mode currently requires y_dim==1 for ExoTST.")

        model.head = HorizonDistMLPHead(
            ny=int(getattr(model, "ny")),
            d_model=int(getattr(cfg, "d_model")),
            horizon=int(getattr(cfg, "horizon", getattr(model, "horizon"))),
            y_dim=int(getattr(cfg, "y_dim", getattr(model, "y_dim"))),
            out_mult=out_mult,
            dropout=float(getattr(cfg, "dropout", 0.0)),
        )
        print(f"[train_exotst] dist head rebuilt: out_mult={out_mult}, param_names={param_names}")
        return

    # point
    model.loss_type = "point"
    model.out_mult = 1
    model.param_names = None
    model.head = HorizonMLPHead(
        ny=int(getattr(model, "ny")),
        d_model=int(getattr(cfg, "d_model")),
        horizon=int(getattr(cfg, "horizon", getattr(model, "horizon"))),
        y_dim=int(getattr(cfg, "y_dim", getattr(model, "y_dim"))),
        dropout=float(getattr(cfg, "dropout", 0.0)),
    )
    print("[train_exotst] point head rebuilt")

def _ensure_exotst_exo_dims(model, e_past: int, e_future: int) -> None:
    """
    ExoTST forward에서 use_past/use_future 판단에 cfg.exo_dim_past/exo_dim_future를 사용하므로,
    실제 입력 차원 기준으로 cfg를 동기화합니다.
    """
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        return

    # ExoTST는 논문 정렬 기준으로 past+future 둘 다 필요
    cfg.use_past_exo = True
    cfg.use_future_exo = True
    cfg.exo_dim_past = int(e_past)
    cfg.exo_dim_future = int(e_future)

    print(f"[train_exotst] cfg.exo_dim_past={cfg.exo_dim_past}, cfg.exo_dim_future={cfg.exo_dim_future}")

@contextmanager
def sdp_math_only():
    if torch.cuda.is_available() and hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
        # with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            yield
    else:
        yield


def train_exotst(
    model,
    train_loader,
    val_loader,
    device,
    *,
    stages=None,
    train_cfg=None,
    future_exo_cb: Optional[Callable] = None,

    exo_spec: Optional[ExoSpec] = None,

    lookback: Optional[int] = None,
):
    """
    ExoTST 학습 진입점.

    정책(= exo_policy 정렬):
    - 미래 exo: loader(fe_cont) 우선, 없으면 future_exo_cb로 제공
    - 과거 exo: loader(pe_cont)에서만 받는 것을 기본 가정
    - ExoTST는 past+future 모두 필요 → 하나라도 없으면 fail-fast
    """
    assert train_cfg is not None, "train_cfg는 필수입니다."

    use_exogenous_mode = bool(getattr(train_cfg, "use_exogenous_mode", True))
    exo_is_normalized = bool(getattr(train_cfg, "exo_is_normalized", True))

    # ExoTST는 구조적으로 exo가 필수이므로 OFF면 에러로 고정(권장)
    if not use_exogenous_mode:
        raise RuntimeError("[train_exotst] ExoTST는 use_exogenous_mode=True가 필수입니다.")

    horizon = int(getattr(model, "horizon", None) or getattr(train_cfg, "horizon", None) or 0)
    if horizon <= 0:
        raise RuntimeError("[train_exotst] horizon을 모델 또는 train_cfg에서 얻지 못했습니다.")

    # lookback(검증 정확도↑): 주입 안되면 train_cfg.lookback을 사용
    if lookback is None:
        lookback = int(getattr(train_cfg, "lookback", 0) or 0) or None

    # ----------------------------------------------------------
    # (1) exo dim inference (exo_policy 기준)
    # ----------------------------------------------------------
    if exo_spec is not None:
        # total_train에서 이미 resolve_exogenous()로 결정된 값을 그대로 사용
        E_future_loader = int(exo_spec.loader_exo_dim) if exo_spec.has_loader_future_exo else 0
        E_future = int(exo_spec.exo_dim)
        E_past_cont = int(exo_spec.past_cont_dim)
        E_past_cat = int(exo_spec.past_cat_dim)
        # source가 loader면 cb는 반드시 None이 맞음(중복/불일치 방지)
        future_exo_cb_eff = None if exo_spec.source == "loader" else exo_spec.future_exo_cb or future_exo_cb
    else:
        # 1) loader에서 future exo dim 추론
        has_fe, fe_dim = infer_future_exo_spec_from_loader(
            train_loader, lookback=lookback, horizon=horizon
        )
        E_future_loader = int(fe_dim) if has_fe else 0

        # 2) loader에서 past exo dim 추론 (cont/cat)
        past_cont_dim, past_cat_dim = infer_past_exo_dim_from_loader_for_exotst(
            train_loader, lookback=lookback, horizon=horizon
        )
        E_past_cont = int(past_cont_dim)
        E_past_cat = int(past_cat_dim)

        # 3) loader가 미래 exo를 주면 callback은 끄는 것이 안전
        future_exo_cb_eff = future_exo_cb
        if E_future_loader > 0 and future_exo_cb_eff is not None:
            future_exo_cb_eff = None
            print(f"[train_exotst] loader provides fe_cont(E_future={E_future_loader}), so future_exo_cb disabled.")

        # 4) loader가 미래 exo를 안 주면 cb dim 추론
        if E_future_loader > 0:
            E_future = E_future_loader
        else:
            cb_wrapped = wrap_future_exo_cb(future_exo_cb_eff)
            E_future_cb = int(infer_exo_dim_from_cb(cb_wrapped, horizon, device="cpu"))
            E_future = E_future_cb

    # ----------------------------------------------------------
    # (2) fail-fast (ExoTST 구현은 past+future 둘다 필요)
    # ----------------------------------------------------------
    if E_past_cont <= 0:
        raise RuntimeError(
            "[train_exotst] past_exo_cont(pe_cont) dim == 0 입니다. "
            "ExoTST는 과거 외생변수가 필수입니다. DataLoader wiring을 확인하세요."
        )

    if E_future <= 0:
        raise RuntimeError(
            "[train_exotst] future_exo dim == 0 입니다. "
            "ExoTST는 미래 외생변수가 필수입니다. "
            "DataLoader(fe_cont) 또는 future_exo_cb를 제공하세요."
        )

    if E_past_cat > 0:
        # ExoTST가 cat을 직접 안 쓰면 경고만(또는 여기서 에러로 강제)
        print(f"[train_exotst][WARN] past_exo_cat dim={E_past_cat} detected. "
              f"ExoTST가 cat을 직접 지원하지 않으면 pe_cont로 인코딩해서 넣는 것을 권장합니다.")

    # cfg 동기화 (forward의 use_past/use_future gating에 필요)
    _ensure_exotst_exo_dims(model, E_past_cont, E_future)

    # loss_mode에 따른 head 동기화
    _ensure_exotst_loss_head(model, train_cfg)

    print(
        f"[EXO-setup] E_past_cont={E_past_cont} | E_future={E_future} | "
        f"future_exo_cb? {future_exo_cb_eff is not None} | exo_is_normalized={exo_is_normalized}"
    )

    # ----------------------------------------------------------
    # (3) 이하 AMP / stages / trainer loop는 기존 그대로
    # ----------------------------------------------------------
    amp_device, amp_enabled, amp_dtype = amp_type_set(train_cfg)
    autocast_input = dict(device_type=amp_device, enabled=amp_enabled, dtype=amp_dtype)

    if not stages or len(stages) == 0:
        stages = [StageConfig(epochs=train_cfg.epochs, spike_enabled=train_cfg.spike_loss.enabled)]

    adapter = DefaultAdapter()
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
        cfg_i = apply_stage(train_cfg, stg)
        print(f"\n[train_exotst] ===== Stage {i}/{len(stages)} =====")
        print(f"  - spike: {'ON' if cfg_i.spike_loss.enabled else 'OFF'}")
        print(f"  - epochs: {cfg_i.epochs} | lr={cfg_i.lr} | horizon_decay={cfg_i.use_horizon_decay}")

        from modeling_module.training.model_trainers.cfg_policy import dump_cfg
        dump_cfg(cfg_i, name="exotst_train")

        tl_i = maybe_make_spike_loader(train_loader, enable=cfg_i.spike_loss.enabled)

        trainer = CommonTrainer(
            cfg=cfg_i,
            adapter=adapter,
            future_exo_cb=future_exo_cb_eff,
            logger=print,
            metrics_fn=None,
            autocast_input=autocast_input,
            extra_loss_fn=None,
            use_exogenous_mode=use_exogenous_mode,
            device=device,
        )

        with sdp_math_only():
            model = trainer.fit(model, tl_i, val_loader, tta_steps=0)
        total_epochs_completed += int(
            getattr(trainer, "epochs_completed_", 0)
        )
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
        best = {
            "model": model,
            "cfg": global_best_cfg,
            "best_val_loss": global_best_loss,
        }

    assert best is not None
    return best

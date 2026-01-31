import os
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Iterable, List, Callable, Any, Literal, Union, Tuple

import torch
import torch.nn as nn

from modeling_module.models.PatchMixer.common.configs import PatchMixerConfig
from modeling_module.models.PatchTST.common.configs import PatchTSTConfig
from modeling_module.models.PatchTST.self_supervised.PatchTST import PatchTSTPretrainModel
from modeling_module.models.Titan.common.configs import TitanConfig
from modeling_module.models.model_builder import (
    build_titan_base,
    build_titan_lmm,
    build_titan_seq2seq,
    build_patch_mixer_quantile,
    build_patchTST,
    build_patchTST_quantile,
    build_patch_mixer,
)
from modeling_module.training.config import SpikeLossConfig, TrainingConfig, StageConfig
from modeling_module.training.model_losses.loss_module import (
    MAE,
    Huber,
    QuantileLoss,
    MQLoss,
    DistributionLoss,
)
from modeling_module.training.model_trainers.patchmixer_train import train_patchmixer
from modeling_module.training.model_trainers.patchtst_finetune import train_patchtst_finetune
from modeling_module.training.model_trainers.patchtst_pretrain import train_patchtst_pretrain
from modeling_module.training.model_trainers.patchtst_train import train_patchtst
from modeling_module.training.model_trainers.titan_train import train_titan
from modeling_module.utils.exogenous_utils import compose_exo_calendar_cb


SSLMode = Literal["ssl_only", "full", "sl_only"]


# =============================================================================
# Loss routing helpers (권장안 2)
# =============================================================================

def _extract_state_dict(ckpt_obj) -> Dict[str, torch.Tensor]:
    """
    다양한 체크포인트 포맷을 안전하게 처리:
    - state_dict, model_state_dict, model, net 등 흔한 키를 우선 탐색
    - 이미 state_dict 형태면 그대로 반환
    """
    if ckpt_obj is None:
        return {}

    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            v = ckpt_obj.get(k, None)
            if isinstance(v, dict):
                return v
        # dict 자체가 state_dict인 케이스
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj

    # 예상 밖 포맷이면 빈 dict
    return {}

def _strip_common_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    DataParallel/Lightning 등에서 흔한 prefix 제거.
    """
    out = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        out[nk] = v
    return out

def _apply_key_mapping(
    sd: Dict[str, torch.Tensor],
    mapping_rules: Iterable[Tuple[str, str]],
) -> Dict[str, torch.Tensor]:
    """
    prefix 기반 key mapping.
    예: ("encoder.", "backbone.")
    """
    out = {}
    for k, v in sd.items():
        nk = k
        for src, dst in mapping_rules:
            if nk.startswith(src):
                nk = dst + nk[len(src):]
                break
        out[nk] = v
    return out

def _filter_state_dict_for_model(
    model: torch.nn.Module,
    sd: Dict[str, torch.Tensor],
    include_prefixes: Tuple[str, ...] = ("backbone.", "revin_layer."),
    exclude_prefixes: Tuple[str, ...] = ("head.",),
    enforce_shape_match: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    - include_prefixes에 해당하는 키만 선택 (encoder-only)
    - exclude_prefixes는 반드시 제외 (head 제외)
    - model에 실제 존재하는 키만 선택
    - shape mismatch는 제거(enforce_shape_match=True)
    """
    model_sd = model.state_dict()
    filtered = {}

    for k, v in sd.items():
        if include_prefixes and not any(k.startswith(p) for p in include_prefixes):
            continue
        if exclude_prefixes and any(k.startswith(p) for p in exclude_prefixes):
            continue
        if k not in model_sd:
            continue
        if enforce_shape_match and tuple(v.shape) != tuple(model_sd[k].shape):
            continue
        filtered[k] = v

    return filtered


def _infer_dist_spec(loss_obj):
    """
    DistributionLoss(distribution=..., ...)에서 out_mul, param_names를 최대한 robust하게 추론.
    - Nixtla DistributionLoss: outputsize_multiplier 속성이 존재하는 케이스가 많음
    - distribution 문자열로 fallback
    """
    # 1) multiplier 우선
    out_mul = int(getattr(loss_obj, "outputsize_multiplier", 0) or 0)

    distr = getattr(loss_obj, "distribution", None)
    if isinstance(distr, str):
        distr_name = distr
    else:
        distr_name = None

    # 2) param_names가 있으면 그대로 사용
    pn = getattr(loss_obj, "param_names", None)
    if pn is not None:
        try:
            param_names = list(pn)  # tuple/list 모두 처리
        except Exception:
            param_names = None
    else:
        param_names = None

    # 3) fallback: distribution 기반
    if out_mul <= 0:
        if (distr_name or "").lower() in ("studentt", "student_t", "student-t"):
            out_mul = 3
        else:
            out_mul = 2

    if param_names is None:
        if (distr_name or "").lower() in ("studentt", "student_t", "student-t"):
            # 기본 관례: [df_raw, loc, scale_raw]
            param_names = ["df", "loc", "scale"]
        else:
            # 기본 관례: [loc, scale_raw]
            param_names = ["loc", "scale"]

    return out_mul, param_names, distr_name

def load_pretrained_encoder_only(
    model: torch.nn.Module,
    ckpt_path: str,
    *,
    include_prefixes: Tuple[str, ...] = ("backbone.", "revin_layer."),
    exclude_prefixes: Tuple[str, ...] = ("head.",),
    mapping_rules: Iterable[Tuple[str, str]] = (("encoder.", "backbone."),),
    strict: bool = False,
) -> Dict[str, int]:
    """
    QuantileModel에 'encoder(backbone)만' 로드.
    - head는 항상 제외
    - mapping + shape match + 존재 키만 로드
    """
    obj = torch.load(ckpt_path, map_location="cpu")
    sd = _extract_state_dict(obj)
    sd = _strip_common_prefixes(sd)
    sd = _apply_key_mapping(sd, mapping_rules)

    enc_sd = _filter_state_dict_for_model(
        model,
        sd,
        include_prefixes=include_prefixes,
        exclude_prefixes=exclude_prefixes,
        enforce_shape_match=True,
    )

    missing, unexpected = model.load_state_dict(enc_sd, strict=strict)

    # 참고용 통계 반환
    return {
        "loaded": len(enc_sd),
        "missing": len(missing),
        "unexpected": len(unexpected),
    }


class QuantileAsPointLoss(nn.Module):
    """Quantile model output에서 특정 quantile(q_star)만 뽑아서 point loss를 적용하는 래퍼.

    - 목적: 사용자가 loss_quantile로 Huber/MAE 같은 point loss를 넣더라도,
            quantile head의 출력([B,H,Q] 또는 [B,Q,H])에서 q_star(기본 0.5)만 추출해 학습 가능하게 함.
    - 주의: 실제로 quantile 전체를 학습시키려면 MQLoss(quantiles=...)를 쓰는 것이 정석입니다.
    """

    def __init__(self, base_loss: nn.Module, quantiles: Iterable[float], q_star: float = 0.5):
        super().__init__()
        self.base_loss = base_loss
        self.quantiles = tuple(float(q) for q in quantiles)
        self.q_star = float(q_star)

        if self.q_star not in self.quantiles:
            raise ValueError(f"q_star={self.q_star} must be in quantiles={self.quantiles}")

        self._q_idx = self.quantiles.index(self.q_star)

        # forward signature compat flags (used by infer_supervised_mode)
        self.is_distribution_output = False

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor, *, mask=None, y_insample=None):
        # y: [B,H,1] (권장), y_hat: [B,H,Q] 또는 [B,Q,H] 또는 [B,H,1,Q]
        if y_hat is None:
            raise ValueError("y_hat is None")

        # normalize y_hat -> [B,H,Q]
        if y_hat.dim() == 4:
            # [B,H,1,Q] or [B,1,H,Q] etc -> squeeze N dim if 1
            if y_hat.shape[2] == 1:
                y_hat3 = y_hat.squeeze(2)
            else:
                # if N>1, take first target channel by default
                y_hat3 = y_hat[:, :, 0, :]
        elif y_hat.dim() == 3:
            y_hat3 = y_hat
        else:
            raise ValueError(f"QuantileAsPointLoss expects 3D/4D y_hat, got {tuple(y_hat.shape)}")

        # [B,Q,H] -> [B,H,Q]
        if y_hat3.shape[1] != y.shape[1] and y_hat3.shape[2] == y.shape[1]:
            y_hat3 = y_hat3.permute(0, 2, 1).contiguous()

        y_hat_med = y_hat3[:, :, self._q_idx].unsqueeze(-1)  # [B,H,1]

        # base_loss signature: (y, y_hat, mask=, y_insample=) 형태가 많음
        try:
            return self.base_loss(y, y_hat_med, mask=mask, y_insample=y_insample)
        except TypeError:
            return self.base_loss(y, y_hat_med, mask=mask)


def infer_supervised_mode(loss_obj) -> str:
    """Infer supervised head/loss mode from loss object."""
    if loss_obj is None:
        return "point"
    if bool(getattr(loss_obj, "is_distribution_output", False)) or (loss_obj.__class__.__name__ == "DistributionLoss"):
        return "dist"
    if loss_obj.__class__.__name__ in ("QuantileLoss", "MQLoss", "QuantileAsPointLoss", "MultiQuantilePinball"):
        return "quantile"
    return "point"


def default_loss_point():
    return MAE()


def default_loss_quantile(quantiles=(0.1, 0.5, 0.9)):
    return MQLoss(quantiles=list(quantiles))


def coerce_quantile_loss(loss_quantile: Optional[nn.Module], *, quantiles=(0.1, 0.5, 0.9)) -> nn.Module:
    """loss_quantile이 None 또는 point loss일 때도 quantile 학습이 가능하도록 보정."""
    print(f'[coerce_quantile_loss]: {loss_quantile.__class__.__name__}')
    if loss_quantile is None:
        return default_loss_quantile(quantiles)

    # 정석: multi-quantile loss
    if loss_quantile.__class__.__name__ in (
            "MQLoss",
            "QuantileLoss",
            "MultiQuantileLoss",
            "MultiQuantilePinball",
    ):
        return loss_quantile
    # 그 외는 point loss라고 보고 median(q_star)에만 적용
    return QuantileAsPointLoss(base_loss=loss_quantile, quantiles=quantiles, q_star=0.5)


# =============================================================================
# Misc utils
# =============================================================================
def _validate_ssl_mode(use_ssl_mode: str) -> str:
    """SSL 모드 문자열 유효성 검증."""
    m = str(use_ssl_mode).strip().lower()
    if m not in ("ssl_only", "full", "sl_only"):
        raise ValueError(f"use_ssl_mode must be one of ['ssl_only','full','sl_only'], got={use_ssl_mode!r}")
    return m


def _get_part_vocab_size_from_loader(loader) -> int:
    """데이터셋의 파트(ID) 어휘 크기 조회."""
    try:
        return len(getattr(loader.dataset, "part_vocab", {}))
    except Exception:
        return 0


def save_model(model: torch.nn.Module, cfg, path: str) -> None:
    """모델의 가중치(State Dict) 및 설정(Config) 저장."""
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
            state["config"] = dict(cfg_dict) if cfg_dict is not None else cfg
    torch.save(state, path)
    print(f"{model} save success! {path}")


def _make_ckpt_path(save_dir: Path, freq: str, model_name: str, lookback: int, horizon: int) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{freq}_{model_name}_L{lookback}_H{horizon}.pt"
    return save_dir / fname


def _build_common_train_configs(
    *,
    device: str,
    lookback: int,
    horizon: int,
    warmup_epochs: Optional[int],
    spike_epochs: Optional[int],
    base_lr: Optional[float],
    loss_point: Optional[nn.Module],
    loss_quantile: Optional[nn.Module],
    use_exogenous_mode: bool,
    quantiles=(0.1, 0.5, 0.9),
    use_intermittent: bool,
    val_use_weights: bool
):
    """Build common TrainingConfig + StageConfig.

    - point_train_cfg.loss 는 loss_point(또는 기본 MAE/DistributionLoss)를 사용
    - quantile_train_cfg.loss 는 loss_quantile(또는 기본 MQLoss)을 사용
    """
    base_lr = float(base_lr) if base_lr is not None else 1e-4
    print(f'[build_common_train_configs] loss_quantile:: {loss_quantile}')

    loss_point_obj = loss_point if loss_point is not None else default_loss_point()
    loss_quantile_obj = coerce_quantile_loss(loss_quantile, quantiles=quantiles)

    print(f'[build_common_train_configs] loss_quantile_obj:: {loss_quantile_obj}')

    point_train_cfg = TrainingConfig(
        device=device,
        lookback=lookback,
        horizon=horizon,
        epochs=0,  # stage-driven
        lr=base_lr,
        weight_decay=1e-3,
        t_max=40,
        patience=100,
        max_grad_norm=30.0,
        amp_device="cuda",
        use_exogenous_mode=bool(use_exogenous_mode),
        loss=loss_point_obj,

        # baseline & weights
        huber_delta=0.8,
        q_star=0.5,
        use_cost_q_star=False,
        Cu=2.0,
        Co=1.0,

        use_intermittent=use_intermittent,
        alpha_zero=0.3,
        alpha_pos=1.0,
        gamma_run=0.5,

        use_horizon_decay=False,
        tau_h=1.0,
        val_use_weights=val_use_weights,

        spike_loss=SpikeLossConfig(
            enabled=False,
            strategy="mix",
            huber_delta=0.8,
            mad_k=3.5,
            w_spike=6.0,
            w_norm=1.0,
            alpha_huber=1.0,
            beta_asym=1.0,
            asym_up_weight=2.0,
            asym_down_weight=1.0,
            mix_with_baseline=False,
            gamma_baseline=0.1,
        ),
    )

    quantile_train_cfg = TrainingConfig(
        device=device,
        lookback=lookback,
        horizon=horizon,
        epochs=0,
        lr=base_lr,
        weight_decay=1e-3,
        t_max=40,
        patience=100,
        max_grad_norm=30.0,
        amp_device="cuda",
        use_exogenous_mode=bool(use_exogenous_mode),
        loss=loss_quantile_obj,

        huber_delta=0.8,
        q_star=0.5,
        use_cost_q_star=False,
        Cu=2.0,
        Co=1.0,

        use_intermittent=use_intermittent,
        alpha_zero=0.3,
        alpha_pos=1.0,
        gamma_run=0.5,

        use_horizon_decay=False,
        tau_h=1.0,
        val_use_weights=val_use_weights,

        spike_loss=SpikeLossConfig(enabled=False),
    )

    spike_cfg = SpikeLossConfig(
        enabled=True,
        strategy="mix",
        huber_delta=0.8,
        mad_k=3.5,
        w_spike=6.0,
        w_norm=1.0,
        alpha_huber=1.0,
        beta_asym=1.0,
        asym_up_weight=2.0,
        asym_down_weight=1.0,
        mix_with_baseline=False,
        gamma_baseline=0.1,
    )

    stages: list[StageConfig] = []
    if warmup_epochs and int(warmup_epochs) > 0:
        stages.append(StageConfig(epochs=int(warmup_epochs), lr=base_lr, spike_enabled=False, use_horizon_decay=False))
    if spike_epochs and int(spike_epochs) > 0:
        stages.append(StageConfig(epochs=int(spike_epochs), lr=base_lr, spike_enabled=True, use_horizon_decay=True))
    if not stages:
        stages.append(StageConfig(epochs=1, lr=base_lr, spike_enabled=False, use_horizon_decay=False))

    return point_train_cfg, quantile_train_cfg, spike_cfg, stages


def _norm_list(xs: Optional[Iterable[str]]) -> List[str]:
    if xs is None:
        return []
    return [str(x).strip().lower() for x in xs if str(x).strip()]


# =============================================================================
# Model runners
# =============================================================================
def _run_patchtst(
    *,
    results: Dict[str, Dict],
    freq: str,
    train_loader,
    val_loader,
    save_root,
    lookback: int,
    horizon: int,
    future_exo_cb,
    exo_dim: int,
    patch_len: int,
    stride: int,
    point_train_cfg,
    quantile_train_cfg,
    stages,
    device: str,
    loss_point: Optional[nn.Module] = None,
    loss_quantile: Optional[nn.Module] = None,
    use_exogenous_mode: bool = True,
    use_ssl_mode: SSLMode = "sl_only",
    ssl_pretrain_epochs: int = 10,
    ssl_mask_ratio: float = 0.3,
    ssl_loss_type: str = "mse",
    ssl_freeze_encoder_before_ft: bool = False,
    ssl_pretrained_ckpt_path: Optional[str] = None,
):
    """PatchTST 모델 학습 파이프라인 실행."""
    use_ssl_mode = _validate_ssl_mode(use_ssl_mode)

    # ------------------------------------------------------------
    # 1) PatchTST 공통 설정 구성
    # ------------------------------------------------------------
    pt_kwargs = dict(
        device=device,
        lookback=lookback,
        horizon=horizon,
        c_in=1,

        patch_len=patch_len,
        stride=stride,
        padding_patch='end',

        d_future=exo_dim,

        d_model=256,
        n_layers=4,
        d_ff=1024,

        norm="LayerNorm",
        pre_norm=True,
        dropout=0.1,
        act="gelu",

        use_revin=True,

        pe="sincos",
        learn_pe=True,

    )

    # ------------------------------------------------------------
    # 2) 외부 사전학습 체크포인트 확인
    # ------------------------------------------------------------
    pretrain_ckpt_path = None
    if ssl_pretrained_ckpt_path:
        if not os.path.exists(ssl_pretrained_ckpt_path):
            raise FileNotFoundError(ssl_pretrained_ckpt_path)
        pretrain_ckpt_path = str(ssl_pretrained_ckpt_path)
        print(f"[SSL] use external pretrained ckpt: {pretrain_ckpt_path}")

    # ------------------------------------------------------------
    # 3) SSL 사전학습 실행 (Optional)
    # ------------------------------------------------------------
    if (use_ssl_mode in ("ssl_only", "full")) and (pretrain_ckpt_path is None) and (save_root is not None):
        pretrain_dir = Path(save_root) / "pretrain"
        pretrain_dir.mkdir(parents=True, exist_ok=True)
        pretrain_ckpt_path = str(pretrain_dir / "patchtst_pretrain_best.pt")

        # SSL은 y-only로 (d_future=0) 권장
        pt_pre_kwargs = dict(pt_kwargs)
        pt_pre_kwargs["d_future"] = 0

        pt_pre_cfg = PatchTSTConfig(**pt_pre_kwargs)
        pre_model = PatchTSTPretrainModel(cfg=pt_pre_cfg)

        pre_train_cfg = point_train_cfg
        pre_stages = [StageConfig(epochs=ssl_pretrain_epochs, lr=point_train_cfg.lr, spike_enabled=False)]

        print(f"[SSL] PatchTST Pretrain ({freq.capitalize()}) -> {pretrain_ckpt_path}")
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
            device = device
        )

    if use_ssl_mode == "ssl_only":
        results["PatchTST SSL"] = {
            "pretrain_ckpt_path": pretrain_ckpt_path,
            "note": "use_ssl_mode='ssl_only' 이므로 supervised(point/dist/quantile) 학습은 수행하지 않음",
        }
        return

    # ============================================================
    # 4) 지도학습 - Base (Point or Dist)
    # ============================================================
    loss_point_obj = loss_point if loss_point is not None else point_train_cfg.loss
    mode = infer_supervised_mode(loss_point_obj)  # "point" | "dist"
    if mode == "dist":
        out_mul, param_names, distr_name = _infer_dist_spec(loss_point_obj)
    else:
        out_mul, param_names, distr_name = 1, None, None

    pt_train_cfg = PatchTSTConfig(**pt_kwargs,
                                  loss = loss_point_obj,
                                  loss_mode = mode,
                                  out_mul = out_mul,
                                  param_names = param_names,
                                  dist_name = distr_name
                                  )

    pt_base = build_patchTST(pt_train_cfg)
    name_base = 'PatchTST'

    print(f"{name_base} ({freq.capitalize()})")

    if (use_ssl_mode == "full") and (pretrain_ckpt_path is not None):
        best_pt_base = train_patchtst_finetune(
            pt_base,
            train_loader,
            val_loader,
            train_cfg=point_train_cfg,
            stages=list(stages),
            future_exo_cb=future_exo_cb,
            exo_is_normalized=True,
            pretrain_ckpt_path=pretrain_ckpt_path,
            load_strict=False,
            freeze_encoder_before_ft=ssl_freeze_encoder_before_ft,
            device = device
        )
    else:
        best_pt_base = train_patchtst(
            pt_base,
            train_loader,
            val_loader,
            train_cfg=point_train_cfg,
            stages=list(stages),
            future_exo_cb=future_exo_cb,
            use_exogenous_mode=use_exogenous_mode,
            device = device
        )

    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, name_base.replace(" ", ""), lookback, horizon)
        save_model(pt_base, pt_train_cfg, ckpt_path)
        best_pt_base["ckpt_path"] = str(ckpt_path)
        if (use_ssl_mode == "full") and (pretrain_ckpt_path is not None):
            best_pt_base["pretrain_ckpt_path"] = str(pretrain_ckpt_path)
    results[name_base] = best_pt_base

    # ============================================================
    # 5) 지도학습 - Quantile Model
    # ============================================================
    quantiles = (0.1, 0.5, 0.9)
    loss_q_obj = coerce_quantile_loss(loss_quantile, quantiles=quantiles)
    quantile_train_cfg = replace(quantile_train_cfg, loss=loss_q_obj)

    pt_q_cfg = PatchTSTConfig(**pt_kwargs, quantiles=quantiles, loss=loss_q_obj)
    pt_q = build_patchTST_quantile(pt_q_cfg)

    print(f"PatchTST Quantile ({freq.capitalize()})")

    if (use_ssl_mode == "full") and (pretrain_ckpt_path is not None):
        best_pt_q = train_patchtst_finetune(
            pt_q,
            train_loader,
            val_loader,
            train_cfg=quantile_train_cfg,
            stages=list(stages),
            future_exo_cb=future_exo_cb,
            exo_is_normalized=True,
            pretrain_ckpt_path=pretrain_ckpt_path,
            load_strict=False,  # head mismatch를 허용 (매우 중요)
            freeze_encoder_before_ft=ssl_freeze_encoder_before_ft,
            device = device
        )
    else:
        best_pt_q = train_patchtst(
            pt_q,
            train_loader,
            val_loader,
            train_cfg=quantile_train_cfg,
            stages=list(stages),
            future_exo_cb=future_exo_cb,
            use_exogenous_mode=use_exogenous_mode,
            device = device
        )

    if save_root:
        ckpt_path_q = _make_ckpt_path(save_root, freq, "PatchTSTQuantile", lookback, horizon)
        save_model(pt_q, pt_q_cfg, ckpt_path_q)
        best_pt_q["ckpt_path"] = str(ckpt_path_q)

        if (use_ssl_mode == "full") and (pretrain_ckpt_path is not None):
            best_pt_q["pretrain_ckpt_path"] = str(pretrain_ckpt_path)

    results["PatchTST Quantile"] = best_pt_q


def _run_titan(
    *,
    results: Dict[str, Dict],
    freq: str,
    train_loader,
    val_loader,
    save_root,
    lookback: int,
    horizon: int,
    use_exogenous_mode: bool,
    future_exo_cb,
    exo_dim: int,
    loss_point: Optional[nn.Module] = None,
    point_train_cfg,
    stages,
    device: str,
):
    """Titan 계열 모델(Base, LMM, Seq2Seq) 학습 실행."""
    loss_point_obj = loss_point if loss_point is not None else point_train_cfg.loss
    mode = infer_supervised_mode(loss_point_obj)

    # past exo dims
    d_past_cont = 0
    d_past_cat = 0
    try:
        b = next(iter(train_loader))
        if isinstance(b, (list, tuple)) and len(b) >= 6:
            pe_cont = b[4]
            pe_cat = b[5]
            if pe_cont is not None and getattr(pe_cont, "ndim", 0) == 3:
                d_past_cont = int(pe_cont.shape[-1])
            if pe_cat is not None and getattr(pe_cat, "ndim", 0) == 3:
                d_past_cat = int(pe_cat.shape[-1])
    except Exception as e:
        print(f"[DBG-ti_kwargs] failed to infer past_exo dims: {repr(e)}")
        d_past_cont, d_past_cat = 0, 0

    cat_embed_dims = tuple([16] * d_past_cat)
    d_past_cont + sum(cat_embed_dims)

    ti_config = TitanConfig(
        lookback=lookback,
        horizon=horizon,
        d_model=256,
        n_layers=3,
        n_heads=4,
        d_ff=4 * 256,
        dropout=0.1,
        contextual_mem_size=256,
        persistent_mem_size=64,
        exo_dim=(int(exo_dim) if use_exogenous_mode else 0),
        past_exo_cont_dim=d_past_cont,
        use_revin=True,
        final_clamp_nonneg=False,

    )
    if freq == "hourly":
        ti_config.contextual_mem_size = 512

    name_suffix = " Dist" if mode == "dist" else ""

    loss_point_obj = loss_point if loss_point is not None else point_train_cfg.loss
    mode = infer_supervised_mode(loss_point_obj)
    out_mult = int(getattr(loss_point_obj, 'outputsize_multiplier', 2)) if mode == 'dist' else 1
    param_names = getattr(loss_point_obj, 'param_names', None) if mode == 'dist' else None

    print(f'[_run_titan] mode: {mode} out_mult: {out_mult}, param_names: {param_names}')

    name_lmm = f"Titan LMM{name_suffix}"
    ckpt_name_lmm = "TitanLMMDist" if mode == "dist" else "TitanLMM"
    print(f"{name_lmm} ({freq.capitalize()})")
    ti_lmm = build_titan_lmm(ti_config, out_mult=out_mult, param_names=param_names)
    best_ti_lmm = train_titan(
        ti_lmm,
        train_loader,
        val_loader,
        device = device,
        train_cfg=point_train_cfg,
        stages=list(stages),
        future_exo_cb=(future_exo_cb if use_exogenous_mode else None),
        use_exogenous_mode=use_exogenous_mode,
    )
    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, ckpt_name_lmm, lookback, horizon)
        save_model(model = ti_lmm, cfg = ti_config, path = ckpt_path)
        best_ti_lmm["ckpt_path"] = str(ckpt_path)
    results[name_lmm] = best_ti_lmm

    name_s2s = f"Titan Seq2Seq{name_suffix}"
    ckpt_name_s2s = "TitanSeq2SeqDist" if mode == "dist" else "TitanSeq2Seq"
    print(f"{name_s2s} ({freq.capitalize()})")
    ti_seq2seq = build_titan_seq2seq(ti_config, out_mult = out_mult, param_names = param_names)
    best_ti_s2s = train_titan(
        ti_seq2seq,
        train_loader,
        val_loader,
        device = device,
        train_cfg=point_train_cfg,
        stages=list(stages),
        future_exo_cb=(future_exo_cb if use_exogenous_mode else None),
        use_exogenous_mode=use_exogenous_mode,
    )
    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, ckpt_name_s2s, lookback, horizon)
        save_model(ti_seq2seq, ti_config, ckpt_path)
        best_ti_s2s["ckpt_path"] = str(ckpt_path)
    results[name_s2s] = best_ti_s2s


def _run_patchmixer(
    *,
    results: Dict[str, Dict],
    freq: str,
    train_loader,
    val_loader,
    save_root,
    lookback: int,
    horizon: int,
    future_exo_cb,
    exo_dim: int,
    patch_len: int,
    stride: int,
    season_period: int,
    loss_point: Optional[nn.Module] = None,
    loss_quantile: Optional[nn.Module] = None,
    loss: Optional[nn.Module] = None,
    use_exogenous_mode: bool = True,
    point_train_cfg=None,
    quantile_train_cfg=None,
    stages=None,
    device: str = "cuda",
):
    """
    PatchMixer 모델(Base/Dist, Quantile) 학습 실행.

    - Point(Base): out_mult=1 -> (B,H)
    - Dist(Base):  out_mult>1 -> (B,H,out_mult) packed (DistributionLoss가 기대)
    - Quantile:    기존 quantile head 유지
    """

    # ------------------------------------------------------------------
    # 0) loss object 결정
    # ------------------------------------------------------------------
    if loss is not None:
        loss_point_obj = loss
    else:
        loss_point_obj = (
            loss_point
            if loss_point is not None
            else (point_train_cfg.loss if point_train_cfg else default_loss_point())
        )

    # quantile loss 구성
    quantiles = (0.1, 0.5, 0.9)
    loss_q_obj = coerce_quantile_loss(loss_quantile, quantiles=quantiles)
    if quantile_train_cfg is not None:
        quantile_train_cfg = replace(quantile_train_cfg, loss=loss_q_obj)

    # ------------------------------------------------------------------
    # 1) supervised mode(dist/point) + dist spec(out_mult/param_names) 추론
    # ------------------------------------------------------------------
    mode = infer_supervised_mode(loss_point_obj)  # "point" | "dist"


    if mode == "dist":
        out_mul, param_names, distr_name = _infer_dist_spec(loss_point_obj)
    else:
        out_mul, param_names, distr_name = 1, None, None


    # ------------------------------------------------------------------
    # 3) past exo dims inference (loader batch에서 추론)
    # ------------------------------------------------------------------
    d_past_cont = 0
    d_past_cat = 0
    try:
        b = next(iter(train_loader))
        if isinstance(b, (list, tuple)) and len(b) >= 6:
            pe_cont = b[4]
            pe_cat = b[5]
            if pe_cont is not None and getattr(pe_cont, "ndim", 0) == 3:
                d_past_cont = int(pe_cont.shape[-1])
            if pe_cat is not None and getattr(pe_cat, "ndim", 0) == 3:
                d_past_cat = int(pe_cat.shape[-1])
    except Exception as e:
        print(f"[DBG-pm_kwargs] failed to infer past_exo dims: {repr(e)}")
        d_past_cont, d_past_cat = 0, 0

    # ------------------------------------------------------------------
    # 2) PatchMixerConfig 공통 kwargs
    #    - 여기서 out_mult/param_names를 cfg에 심어두면 저장/로드에도 유리
    # ------------------------------------------------------------------
    pm_kwargs = dict(
        lookback=lookback,
        horizon=horizon,
        device=device,
        enc_in=1,
        d_model=128,
        e_layers=6,
        patch_len=patch_len,
        stride=stride,
        # f_out=128,
        f_out = 256,
        head_hidden=256,
        exo_dim=exo_dim,
        use_part_embedding=False,
        part_vocab_size=_get_part_vocab_size_from_loader(train_loader),
        part_embed_dim=16,
        final_nonneg=True,
        use_eol_prior=False,
        exo_is_normalized_default=True,
        expander_season_period=season_period,
        expander_n_harmonics=min(season_period // 2, 24),
        quantiles=quantiles,
        loss=loss_point_obj,
        use_revin=True,
        learn_output_scale = True,
        learn_dw_gain = True,
        past_exo_mode = 'z_gate',
        past_exo_cont_dim = d_past_cont,
        past_exo_cat_dim = d_past_cat,
        past_exo_cat_vocab_sizes = (512, 128),
        past_exo_cat_embed_dims = (16, 16),
        out_mul = int(out_mul),
        dist_name = distr_name,
        param_names = param_names,
        head_dropout = 0.02
    )


    # ------------------------------------------------------------------
    # 4) Base/Dist 모델 학습
    # ------------------------------------------------------------------
    pm_base_cfg = PatchMixerConfig(**pm_kwargs)

    # Stabilization Options
    pm_base_cfg.loss = loss_point_obj
    pm_base_model = build_patch_mixer(pm_base_cfg)

    best_pm_base = train_patchmixer(
        pm_base_model,
        train_loader,
        val_loader,
        device=device,
        train_cfg=point_train_cfg,
        stages=list(stages),
        future_exo_cb=(future_exo_cb if use_exogenous_mode else None),
        exo_is_normalized=pm_base_cfg.exo_is_normalized_default,
        use_exogenous_mode=use_exogenous_mode,
    )

    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, "PatchMixer", lookback, horizon)
        save_model(pm_base_model, pm_base_cfg, ckpt_path)
        best_pm_base["ckpt_path"] = str(ckpt_path)

    results["PatchMixer"] = best_pm_base

    # ------------------------------------------------------------------
    # 5) Quantile 모델 학습
    # ------------------------------------------------------------------
    pm_q_cfg = PatchMixerConfig(**pm_kwargs)
    pm_q_cfg.loss = loss_q_obj

    pm_q_model = build_patch_mixer_quantile(pm_q_cfg)
    print(f"PatchMixer Quantile ({freq.capitalize()})")

    best_pm_q = train_patchmixer(
        pm_q_model,
        train_loader,
        val_loader,
        device=device,
        train_cfg=quantile_train_cfg,
        stages=list(stages),
        future_exo_cb=(future_exo_cb if use_exogenous_mode else None),
        exo_is_normalized=pm_q_cfg.exo_is_normalized_default,
        use_exogenous_mode=use_exogenous_mode,
    )

    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, "PatchMixerQuantile", lookback, horizon)
        save_model(pm_q_model, pm_q_cfg, ckpt_path)
        best_pm_q["ckpt_path"] = str(ckpt_path)

    results["PatchMixer Quantile"] = best_pm_q


MODEL_REGISTRY: Dict[str, Callable] = {
    "patchtst": _run_patchtst,
    "titan": _run_titan,
    "patchmixer": _run_patchmixer,
}


# =============================================================================
# Orchestration (modular)
# =============================================================================

# Frequency/Exogenous policies are split into dedicated modules for readability.
try:
    from .freq_policy import FreqSpec, get_freq_spec
    from .exo_policy import ExoSpec, resolve_future_exogenous
except Exception:  # pragma: no cover
    from freq_policy import FreqSpec, get_freq_spec  # type: ignore
    from exo_policy import ExoSpec, resolve_future_exogenous  # type: ignore

# =============================================================================
def _validate_models_to_run(models_to_run: Optional[Iterable[str]]) -> List[str]:
    """Normalize & validate model list."""
    selected = _norm_list(models_to_run)
    if not selected:
        selected = ["patchtst"]

    unknown = [m for m in selected if m not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown models_to_run={unknown}. allowed={list(MODEL_REGISTRY.keys())}")
    return selected


def _build_common_kwargs(
    *,
    results: Dict[str, Dict],
    freq_spec: FreqSpec,
    exo_spec: ExoSpec,
    train_loader,
    val_loader,
    save_root: Optional[Path],
    lookback: int,
    horizon: int,
    point_train_cfg: TrainingConfig,
    quantile_train_cfg: TrainingConfig,
    stages: List[StageConfig],
    device: str,
    loss_point: Optional[nn.Module],
    loss_quantile: Optional[nn.Module],
) -> Dict[str, Any]:
    """Kwargs shared across all model runners."""
    return dict(
        results=results,
        freq=freq_spec.freq,
        train_loader=train_loader,
        val_loader=val_loader,
        save_root=save_root,
        lookback=lookback,
        horizon=horizon,
        future_exo_cb=(exo_spec.future_exo_cb if exo_spec.use_exogenous_mode else None),
        exo_dim=exo_spec.exo_dim,
        point_train_cfg=point_train_cfg,
        quantile_train_cfg=quantile_train_cfg,
        stages=stages,
        device=device,
        use_exogenous_mode=exo_spec.use_exogenous_mode,
        loss_point=loss_point,
        loss_quantile=loss_quantile,
    )


def run_total_train(
    train_loader,
    val_loader,
    *,
    freq: str,
    lookback: int,
    horizon: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    warmup_epochs: Optional[int] = None,
    spike_epochs: Optional[int] = None,
    base_lr: Optional[float] = None,
    save_dir: Optional[str] = None,
    use_exogenous_mode: bool = False,
    models_to_run: Optional[Iterable[str]] = None,
    # loss routing (recommended)
    loss_point: Optional[nn.Module] = None,
    loss_quantile: Optional[nn.Module] = None,
    # backward compat
    loss: Optional[nn.Module] = None,
    # PatchTST SSL
    use_ssl_mode: SSLMode = "sl_only",
    ssl_pretrain_epochs: int = 10,
    ssl_mask_ratio: float = 0.3,
    ssl_loss_type: str = "mse",
    ssl_freeze_encoder_before_ft: bool = False,
    ssl_pretrained_ckpt_path: Optional[str] = None,
    # weights policy
    use_intermittent: bool = True,
    val_use_weights: bool = True,
) -> Dict[str, Dict]:
    """
    Unified training entrypoint.

    Notes
    - This function keeps the public behavior of the older *_run_total_train_generic,
      but isolates policies into:
        - frequency policy (patch/stride/season_period)
        - exogenous resolution (loader vs callback)
        - common TrainingConfig/stage building
        - per-model kwargs composition
    """
    freq_spec = get_freq_spec(freq)
    save_root = Path(save_dir) if save_dir is not None else None

    # backward-compat: loss -> point loss
    if loss_point is None and loss is not None:
        loss_point = loss

    # training configs + stages
    point_train_cfg, quantile_train_cfg, _spike_cfg, stages = _build_common_train_configs(
        device=device,
        lookback=lookback,
        horizon=horizon,
        warmup_epochs=warmup_epochs,
        spike_epochs=spike_epochs,
        base_lr=base_lr,
        loss_point=loss_point,
        loss_quantile=loss_quantile,
        use_exogenous_mode=bool(use_exogenous_mode),
        quantiles=(0.1, 0.5, 0.9),
        use_intermittent=use_intermittent,
        val_use_weights=val_use_weights,
    )

    # exogenous policy
    exo_spec = resolve_future_exogenous(
        train_loader,
        freq_spec=freq_spec,
        use_exogenous_mode=bool(use_exogenous_mode),
    )
    print(f"[total_train] future exo source={exo_spec.source} exo_dim={exo_spec.exo_dim} (freq={freq_spec.freq})")

    # model selection
    selected = _validate_models_to_run(models_to_run)

    # run
    results: Dict[str, Dict] = {}
    base_kwargs = _build_common_kwargs(
        results=results,
        freq_spec=freq_spec,
        exo_spec=exo_spec,
        train_loader=train_loader,
        val_loader=val_loader,
        save_root=save_root,
        lookback=lookback,
        horizon=horizon,
        point_train_cfg=point_train_cfg,
        quantile_train_cfg=quantile_train_cfg,
        stages=stages,
        device=device,
        loss_point=loss_point,
        loss_quantile=loss_quantile,
    )

    for m in selected:
        print(f"\n[total_train] === RUN: {m} ({freq_spec.freq}) ===")

        kwargs = dict(base_kwargs)

        # per-model extras
        if m == "patchtst":
            kwargs.update(
                dict(
                    patch_len=freq_spec.patch_len,
                    stride=freq_spec.stride,
                    use_ssl_mode=use_ssl_mode,
                    ssl_pretrain_epochs=ssl_pretrain_epochs,
                    ssl_mask_ratio=ssl_mask_ratio,
                    ssl_loss_type=ssl_loss_type,
                    ssl_freeze_encoder_before_ft=ssl_freeze_encoder_before_ft,
                    ssl_pretrained_ckpt_path=ssl_pretrained_ckpt_path,
                )
            )
        elif m == "patchmixer":
            kwargs.update(
                dict(
                    patch_len=freq_spec.patch_len,
                    stride=freq_spec.stride,
                    season_period=freq_spec.season_period,
                )
            )
        elif m == "titan":
            # titan runner does not use quantile configs
            kwargs.pop("quantile_train_cfg", None)
            kwargs.pop("loss_quantile", None)

        MODEL_REGISTRY[m](**kwargs)

    return results


# backward compatible alias (older code may import this symbol)
_run_total_train_generic = run_total_train


# =============================================================================
# Exported wrappers (backward compatible)
# =============================================================================

def run_total_train_weekly(
    train_loader,
    val_loader,
    device="cuda" if torch.cuda.is_available() else "cpu",
    *,
    lookback,
    horizon,
    warmup_epochs=None,
    spike_epochs=None,
    base_lr=None,
    save_dir=None,
    use_exogenous_mode: bool = False,
    models_to_run=None,
    loss_point: Optional[nn.Module] = None,
    loss_quantile: Optional[nn.Module] = None,
    loss: Optional[nn.Module] = None,
    use_ssl_mode: SSLMode = "sl_only",
    ssl_pretrain_epochs: int = 10,
    ssl_mask_ratio: float = 0.3,
    ssl_loss_type: str = "mse",
    ssl_freeze_encoder_before_ft: bool = False,
    ssl_pretrained_ckpt_path: Optional[str] = None,
    use_intermittent: bool = True,
    val_use_weights: bool = True,
):
    return run_total_train(
        train_loader,
        val_loader,
        freq="weekly",
        lookback=lookback,
        horizon=horizon,
        device=device,
        warmup_epochs=warmup_epochs,
        spike_epochs=spike_epochs,
        base_lr=base_lr,
        save_dir=save_dir,
        use_exogenous_mode=use_exogenous_mode,
        models_to_run=models_to_run,
        loss_point=loss_point,
        loss_quantile=loss_quantile,
        loss=loss,
        use_ssl_mode=use_ssl_mode,
        ssl_pretrain_epochs=ssl_pretrain_epochs,
        ssl_mask_ratio=ssl_mask_ratio,
        ssl_loss_type=ssl_loss_type,
        ssl_freeze_encoder_before_ft=ssl_freeze_encoder_before_ft,
        ssl_pretrained_ckpt_path=ssl_pretrained_ckpt_path,
        use_intermittent=use_intermittent,
        val_use_weights=val_use_weights,
    )


def run_total_train_monthly(
    train_loader,
    val_loader,
    device="cuda" if torch.cuda.is_available() else "cpu",
    *,
    lookback,
    horizon,
    warmup_epochs=None,
    spike_epochs=None,
    base_lr=None,
    save_dir=None,
    use_exogenous_mode: bool = False,
    models_to_run=None,
    loss_point: Optional[nn.Module] = None,
    loss_quantile: Optional[nn.Module] = None,
    loss: Optional[nn.Module] = None,
    use_ssl_mode: SSLMode = "sl_only",
    ssl_pretrain_epochs: int = 10,
    ssl_mask_ratio: float = 0.3,
    ssl_loss_type: str = "mse",
    ssl_freeze_encoder_before_ft: bool = False,
    ssl_pretrained_ckpt_path: Optional[str] = None,
    use_intermittent: bool = True,
    val_use_weights: bool = True,
):
    return run_total_train(
        train_loader,
        val_loader,
        freq="monthly",
        lookback=lookback,
        horizon=horizon,
        device=device,
        warmup_epochs=warmup_epochs,
        spike_epochs=spike_epochs,
        base_lr=base_lr,
        save_dir=save_dir,
        use_exogenous_mode=use_exogenous_mode,
        models_to_run=models_to_run,
        loss_point=loss_point,
        loss_quantile=loss_quantile,
        loss=loss,
        use_ssl_mode=use_ssl_mode,
        ssl_pretrain_epochs=ssl_pretrain_epochs,
        ssl_mask_ratio=ssl_mask_ratio,
        ssl_loss_type=ssl_loss_type,
        ssl_freeze_encoder_before_ft=ssl_freeze_encoder_before_ft,
        ssl_pretrained_ckpt_path=ssl_pretrained_ckpt_path,
        use_intermittent=use_intermittent,
        val_use_weights=val_use_weights,
    )


def run_total_train_daily(
    train_loader,
    val_loader,
    device="cuda" if torch.cuda.is_available() else "cpu",
    *,
    lookback,
    horizon,
    warmup_epochs=None,
    spike_epochs=None,
    base_lr=None,
    save_dir=None,
    use_exogenous_mode: bool = False,
    models_to_run=None,
    loss_point: Optional[nn.Module] = None,
    loss_quantile: Optional[nn.Module] = None,
    loss: Optional[nn.Module] = None,
    use_ssl_mode: SSLMode = "sl_only",
    ssl_pretrain_epochs: int = 10,
    ssl_mask_ratio: float = 0.3,
    ssl_loss_type: str = "mse",
    ssl_freeze_encoder_before_ft: bool = False,
    ssl_pretrained_ckpt_path: Optional[str] = None,
    use_intermittent: bool = True,
    val_use_weights: bool = True,
):
    return run_total_train(
        train_loader,
        val_loader,
        freq="daily",
        lookback=lookback,
        horizon=horizon,
        device=device,
        warmup_epochs=warmup_epochs,
        spike_epochs=spike_epochs,
        base_lr=base_lr,
        save_dir=save_dir,
        use_exogenous_mode=use_exogenous_mode,
        models_to_run=models_to_run,
        loss_point=loss_point,
        loss_quantile=loss_quantile,
        loss=loss,
        use_ssl_mode=use_ssl_mode,
        ssl_pretrain_epochs=ssl_pretrain_epochs,
        ssl_mask_ratio=ssl_mask_ratio,
        ssl_loss_type=ssl_loss_type,
        ssl_freeze_encoder_before_ft=ssl_freeze_encoder_before_ft,
        ssl_pretrained_ckpt_path=ssl_pretrained_ckpt_path,
        use_intermittent=use_intermittent,
        val_use_weights=val_use_weights,
    )


def run_total_train_hourly(
    train_loader,
    val_loader,
    device="cuda" if torch.cuda.is_available() else "cpu",
    *,
    lookback,
    horizon,
    warmup_epochs=None,
    spike_epochs=None,
    base_lr=None,
    save_dir=None,
    use_exogenous_mode: bool = False,
    models_to_run=None,
    loss_point: Optional[nn.Module] = None,
    loss_quantile: Optional[nn.Module] = None,
    loss: Optional[nn.Module] = None,
    use_ssl_mode: SSLMode = "sl_only",
    ssl_pretrain_epochs: int = 10,
    ssl_mask_ratio: float = 0.3,
    ssl_loss_type: str = "mse",
    ssl_freeze_encoder_before_ft: bool = False,
    ssl_pretrained_ckpt_path: Optional[str] = None,
    use_intermittent: bool = True,
    val_use_weights: bool = True,
):
    return run_total_train(
        train_loader,
        val_loader,
        freq="hourly",
        lookback=lookback,
        horizon=horizon,
        device=device,
        warmup_epochs=warmup_epochs,
        spike_epochs=spike_epochs,
        base_lr=base_lr,
        save_dir=save_dir,
        use_exogenous_mode=use_exogenous_mode,
        models_to_run=models_to_run,
        loss_point=loss_point,
        loss_quantile=loss_quantile,
        loss=loss,
        use_ssl_mode=use_ssl_mode,
        ssl_pretrain_epochs=ssl_pretrain_epochs,
        ssl_mask_ratio=ssl_mask_ratio,
        ssl_loss_type=ssl_loss_type,
        ssl_freeze_encoder_before_ft=ssl_freeze_encoder_before_ft,
        ssl_pretrained_ckpt_path=ssl_pretrained_ckpt_path,
        use_intermittent=use_intermittent,
        val_use_weights=val_use_weights,
    )
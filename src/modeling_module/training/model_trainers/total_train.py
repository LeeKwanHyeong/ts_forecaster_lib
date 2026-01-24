"""total_train.py

High-level training runner for the LTB forecasting engine.

기능:
1) 다양한 모델(PatchTST, Titan, PatchMixer)과 주기(Hourly/Daily/Weekly/Monthly)에 대한 통합 학습 실행.
2) 외생 변수(Exogenous Variable) 주입 방식(Loader vs Callback) 자동 제어.
3) 2-Stage 학습(Warmup -> Spike Loss) 및 Self-Supervised Learning(SSL) 파이프라인 오케스트레이션.

[Refactor - 권장안 2]
- Point/Dist(분포) 학습과 Quantile 학습의 Loss를 분리하여 주입:
    - loss_point: Point 또는 Distribution 학습용 (예: MAE, Huber, DistributionLoss)
    - loss_quantile: Quantile 학습용 (예: MQLoss, QuantileLoss, 또는 PointLoss를 median(q=0.5)에 적용하는 Wrapper)
- 단일 loss 인자로 모든 모델/헤드를 자동 처리하려는 시도는,
  (1) 모델 출력 스키마(예: [B,H,2], [B,Q,H], dict) 불일치,
  (2) loss 별 기대 입력 텐서 차원 불일치
  로 인해 디버깅 비용이 급격히 증가하므로, runner 레벨에서 명시적으로 분리합니다.
"""

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
    build_patch_mixer_base,
    build_patch_mixer_quantile,
    build_patchTST_base,
    build_patchTST_quantile,
    build_patchTST_dist,
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

def load_pretrained_encoder_only(
    model: torch.nn.Module,
    ckpt_path: str,
    *,
    include_prefixes: Tuple[str, ...] = ("backbone.", "revin_layer."),
    exclude_prefixes: Tuple[str, ...] = ("head.",),
    mapping_rules: Iterable[Tuple[str, str]] = (
        # (B)까지 커버: 혹시 ckpt가 encoder.*로 저장된 경우
        ("encoder.", "backbone."),
        # 필요시 추가: ("backbone.", "backbone.") 같은 것은 의미 없어서 생략
    ),
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
    if loss_obj.__class__.__name__ in ("QuantileLoss", "MQLoss", "QuantileAsPointLoss"):
        return "quantile"
    return "point"


def default_loss_point():
    return MAE()


def default_loss_quantile(quantiles=(0.1, 0.5, 0.9)):
    return MQLoss(quantiles=list(quantiles))


def coerce_quantile_loss(loss_quantile: Optional[nn.Module], *, quantiles=(0.1, 0.5, 0.9)) -> nn.Module:
    """loss_quantile이 None 또는 point loss일 때도 quantile 학습이 가능하도록 보정."""
    if loss_quantile is None:
        return default_loss_quantile(quantiles)

    # 정석: multi-quantile loss
    if loss_quantile.__class__.__name__ in ("MQLoss",):
        return loss_quantile
    if loss_quantile.__class__.__name__ in ("QuantileLoss",):
        # 단일 q 학습용. 모델이 multi-quantile 출력이면 불완전하지만 허용
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


def _infer_future_exo_spec_from_loader(loader) -> tuple[bool, int]:
    """
    데이터 로더로부터 미래 외생 변수(fe_cont) 제공 여부 및 차원 추론.

    반환:
        (has_fe, fe_dim)
    """
    try:
        b = next(iter(loader))
        if not isinstance(b, (list, tuple)) or len(b) < 4:
            return (False, 0)
        fe = b[3]
        if fe is None:
            return (False, 0)
        if hasattr(fe, "ndim") and fe.ndim == 3:
            return (True, int(fe.shape[-1]))
        if hasattr(fe, "ndim") and fe.ndim == 2:
            # (H, E) 형태로 제공되는 경우도 방어
            return (True, int(fe.shape[-1]))
        return (True, 0)
    except Exception:
        return (False, 0)


def _wrap_future_exo_cb(future_exo_cb):
    """미래 외생 변수 콜백 함수 래핑 (device keyword 흡수)."""
    if future_exo_cb is None:
        return None

    def _wrapped(t0, H, *args, **kwargs):
        device = kwargs.pop("device", None)
        out = future_exo_cb(t0, H)
        if device is not None and isinstance(out, torch.Tensor):
            out = out.to(device)
        return out

    return _wrapped


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
):
    """Build common TrainingConfig + StageConfig.

    - point_train_cfg.loss 는 loss_point(또는 기본 MAE/DistributionLoss)를 사용
    - quantile_train_cfg.loss 는 loss_quantile(또는 기본 MQLoss)을 사용
    """
    base_lr = float(base_lr) if base_lr is not None else 1e-4

    loss_point_obj = loss_point if loss_point is not None else default_loss_point()
    loss_quantile_obj = coerce_quantile_loss(loss_quantile, quantiles=quantiles)

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

        use_intermittent=True,
        alpha_zero=0.3,
        alpha_pos=1.0,
        gamma_run=0.5,

        use_horizon_decay=False,
        tau_h=1.0,
        val_use_weights=True,

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

        use_intermittent=True,
        alpha_zero=0.3,
        alpha_pos=1.0,
        gamma_run=0.5,

        use_horizon_decay=False,
        tau_h=1.0,
        val_use_weights=True,

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
        d_model=256,
        n_layers=3,
        patch_len=patch_len,
        stride=stride,
        d_future=exo_dim,
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
    mode = infer_supervised_mode(loss_point_obj)

    pt_base_cfg = PatchTSTConfig(**pt_kwargs, loss=loss_point_obj, loss_mode=("dist" if mode == "dist" else "point"))
    print(f'[run_patchtst] mode:: {mode}')
    if mode == "dist":
        pt_base = build_patchTST_dist(pt_base_cfg)
        name_base = "PatchTST Dist"
    else:
        pt_base = build_patchTST_base(pt_base_cfg)
        name_base = "PatchTST Base"

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
        )

    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, name_base.replace(" ", ""), lookback, horizon)
        save_model(pt_base, pt_base_cfg, ckpt_path)
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

    # ---- [핵심] Quantile도 pretrain_ckpt_path 적용 (encoder만 안전 로드) ----
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
        )

    if save_root:
        ckpt_path_q = _make_ckpt_path(save_root, freq, "PatchTSTQuantile", lookback, horizon)
        save_model(pt_q, pt_q_cfg, ckpt_path_q)
        best_pt_q["ckpt_path"] = str(ckpt_path_q)

        # (선택) 어떤 pretrain을 썼는지 결과에 기록
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
    patch_len: int,
    stride: int,
    point_train_cfg,
    stages,
    device: str,
):
    """Titan 계열 모델(Base, LMM, Seq2Seq) 학습 실행."""
    loss_point_obj = loss_point if loss_point is not None else point_train_cfg.loss

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

    cat_vocab_sizes = tuple([512] * d_past_cat)
    cat_embed_dims = tuple([16] * d_past_cat)
    past_dim_total = d_past_cont + sum(cat_embed_dims)

    ti_config = TitanConfig(
        lookback=lookback,
        horizon=horizon,
        input_dim=1 + past_dim_total,
        d_model=256,
        n_layers=3,
        n_heads=4,
        d_ff=512,
        dropout=0.1,
        contextual_mem_size=256,
        persistent_mem_size=64,
        use_exogenous_mode=use_exogenous_mode,
        exo_dim=(int(exo_dim) if use_exogenous_mode else 0),
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
        loss=loss_point_obj,
    )
    if freq == "hourly":
        ti_config.contextual_mem_size = 512

    ti_base = build_titan_base(ti_config)
    print(f"Titan Base ({freq.capitalize()})")
    best_ti_base = train_titan(
        ti_base,
        train_loader,
        val_loader,
        train_cfg=point_train_cfg,
        stages=list(stages),
        future_exo_cb=(future_exo_cb if use_exogenous_mode else None),
        use_exogenous_mode=use_exogenous_mode,
    )
    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, "TitanBase", lookback, horizon)
        save_model(ti_base, ti_config, ckpt_path)
        best_ti_base["ckpt_path"] = str(ckpt_path)
    results["Titan Base"] = best_ti_base

    ti_config_lmm = replace(ti_config, use_lmm=True)
    ti_lmm = build_titan_lmm(ti_config_lmm)
    print(f"Titan LMM ({freq.capitalize()})")
    best_ti_lmm = train_titan(
        ti_lmm,
        train_loader,
        val_loader,
        train_cfg=point_train_cfg,
        stages=list(stages),
        future_exo_cb=(future_exo_cb if use_exogenous_mode else None),
        use_exogenous_mode=use_exogenous_mode,
    )
    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, "TitanLMM", lookback, horizon)
        save_model(ti_lmm, ti_config_lmm, ckpt_path)
        best_ti_lmm["ckpt_path"] = str(ckpt_path)
    results["Titan LMM"] = best_ti_lmm

    ti_seq2seq = build_titan_seq2seq(ti_config)
    print(f"Titan Seq2Seq ({freq.capitalize()})")
    best_ti_s2s = train_titan(
        ti_seq2seq,
        train_loader,
        val_loader,
        train_cfg=point_train_cfg,
        stages=list(stages),
        future_exo_cb=(future_exo_cb if use_exogenous_mode else None),
        use_exogenous_mode=use_exogenous_mode,
    )
    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, "TitanSeq2Seq", lookback, horizon)
        save_model(ti_seq2seq, ti_config, ckpt_path)
        best_ti_s2s["ckpt_path"] = str(ckpt_path)
    results["Titan Seq2Seq"] = best_ti_s2s


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
    loss_point: Optional[nn.Module] = None,
    loss_quantile: Optional[nn.Module] = None,
    use_exogenous_mode: bool = True,
    point_train_cfg=None,
    quantile_train_cfg=None,
    stages=None,
    device: str = "cuda",
):
    """PatchMixer 모델(Base, Quantile) 학습 실행."""
    loss_point_obj = loss_point if loss_point is not None else (point_train_cfg.loss if point_train_cfg else default_loss_point())
    quantiles = (0.1, 0.5, 0.9)
    loss_q_obj = coerce_quantile_loss(loss_quantile, quantiles=quantiles)
    if quantile_train_cfg is not None:
        quantile_train_cfg = replace(quantile_train_cfg, loss=loss_q_obj)

    # freq별 patch 추천값
    if freq == "hourly":
        patch_len, stride, season_period = 24, 12, 24
    elif freq == "daily":
        patch_len, stride, season_period = 14, 7, 7
    elif freq == "weekly":
        patch_len, stride, season_period = 12, 8, 52
    else:
        patch_len, stride, season_period = 6, 3, 12

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
        exo_is_normalized_default=False,
        expander_season_period=season_period,
        expander_n_harmonics=min(season_period // 2, 16),
        quantiles=quantiles,
        loss=loss_point_obj,
    )

    # past exo dims inference
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

    pm_base_cfg = PatchMixerConfig(**pm_kwargs, head_dropout=0.05)
    pm_base_cfg.learn_output_scale = False
    pm_base_cfg.learn_dw_gain = False
    pm_base_cfg.exo_is_normalized_default = False
    pm_base_cfg.past_exo_mode = "z_gate"
    pm_base_cfg.past_exo_cont_dim = d_past_cont
    pm_base_cfg.past_exo_cat_dim = d_past_cat
    pm_base_cfg.past_exo_cat_vocab_sizes = (512, 128)
    pm_base_cfg.past_exo_cat_embed_dims = (16, 16)

    pm_base_model = build_patch_mixer_base(pm_base_cfg)
    print(f"PatchMixer Base ({freq.capitalize()})")
    best_pm_base = train_patchmixer(
        pm_base_model,
        train_loader,
        val_loader,
        train_cfg=point_train_cfg,
        stages=list(stages),
        future_exo_cb=(future_exo_cb if use_exogenous_mode else None),
        exo_is_normalized=pm_base_cfg.exo_is_normalized_default,
        use_exogenous_mode=use_exogenous_mode,
    )
    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, "PatchMixerBase", lookback, horizon)
        save_model(pm_base_model, pm_base_cfg, ckpt_path)
        best_pm_base["ckpt_path"] = str(ckpt_path)
    results["PatchMixer Base"] = best_pm_base

    pm_q_cfg = PatchMixerConfig(**pm_kwargs, quantiles=quantiles, head_dropout=0.02)
    pm_q_cfg.loss = loss_q_obj
    pm_q_cfg.learn_output_scale = False
    pm_q_cfg.learn_dw_gain = False
    pm_q_cfg.exo_is_normalized_default = False
    pm_q_cfg.past_exo_mode = "z_gate"
    pm_q_cfg.past_exo_cont_dim = d_past_cont
    pm_q_cfg.past_exo_cat_dim = d_past_cat
    pm_q_cfg.past_exo_cat_vocab_sizes = (512, 128)
    pm_q_cfg.past_exo_cat_embed_dims = (16, 16)

    pm_q_model = build_patch_mixer_quantile(pm_q_cfg)
    print(f"PatchMixer Quantile ({freq.capitalize()})")
    best_pm_q = train_patchmixer(
        pm_q_model,
        train_loader,
        val_loader,
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
# Generic runner
# =============================================================================

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
    # 권장안 2: loss 분리
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
):
    """전체 학습 프로세스 오케스트레이션 (Generic Runner)."""
    save_root = Path(save_dir) if save_dir is not None else None

    # backward-compat: loss만 넘어오면 point loss로 취급
    if loss_point is None and loss is not None:
        loss_point = loss

    point_train_cfg, quantile_train_cfg, spike_cfg, stages = _build_common_train_configs(
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
    )

    date_type_map = {"weekly": "W", "monthly": "M", "daily": "D", "hourly": "H"}
    dt_char = date_type_map.get(freq, "W")

    # Exogenous policy
    has_fe, fe_dim = _infer_future_exo_spec_from_loader(train_loader)
    print(f"[total_train] use_exogenous_mode: {use_exogenous_mode} has_fe: {has_fe}, fe_dim: {fe_dim}")

    if use_exogenous_mode:
        if has_fe:
            if fe_dim <= 0:
                raise RuntimeError(
                    f"[total_train] use_exogenous_mode=True but loader fe_cont dim is {fe_dim}. "
                    f"Check feature selection / exogenous datamodule wiring."
                )
            future_exo_cb = None
            exo_dim = int(fe_dim)
            print(f"[total_train] future exo from loader: fe_dim={exo_dim} (freq={freq})")
        else:
            future_exo_cb = compose_exo_calendar_cb(date_type=dt_char)
            future_exo_cb = _wrap_future_exo_cb(future_exo_cb)
            exo_dim = 4 if freq in ("daily", "hourly") else 2
            print(f"[total_train] future exo from callback: exo_dim={exo_dim} (freq={freq}, dt={dt_char})")
    else:
        future_exo_cb = None
        exo_dim = 0
        if has_fe and fe_dim > 0:
            print(
                f"[total_train][WARN] use_exogenous_mode=False but loader provides fe_cont dim={fe_dim}. "
                f"Ignoring future exo."
            )

    # freq별 patch_len/stride
    if freq == "hourly":
        patch_len, stride = 24, 12
    elif freq == "daily":
        patch_len, stride = 14, 7
    elif freq == "weekly":
        patch_len, stride = 12, 8
    else:
        patch_len, stride = 6, 3

    selected = _norm_list(models_to_run)
    if not selected:
        selected = ["patchtst"]

    unknown = [m for m in selected if m not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown models_to_run: {unknown}. allowed={list(MODEL_REGISTRY.keys())}")

    results: Dict[str, Dict] = {}

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
            use_exogenous_mode=bool(use_exogenous_mode),
            loss_point=loss_point,
            loss_quantile=loss_quantile,
        )

        if m == "patchtst":
            kwargs.update(
                dict(
                    use_ssl_mode=use_ssl_mode,
                    ssl_pretrain_epochs=ssl_pretrain_epochs,
                    ssl_mask_ratio=ssl_mask_ratio,
                    ssl_loss_type=ssl_loss_type,
                    ssl_freeze_encoder_before_ft=ssl_freeze_encoder_before_ft,
                    ssl_pretrained_ckpt_path=ssl_pretrained_ckpt_path,
                )
            )

        if m == "titan":
            # titan runner does not use quantile_train_cfg
            kwargs.pop("quantile_train_cfg", None)
            kwargs.pop("loss_quantile", None)

        MODEL_REGISTRY[m](**kwargs)

    return results


# =============================================================================
# Exported wrappers
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
    # 권장안 2
    loss_point: Optional[nn.Module] = None,
    loss_quantile: Optional[nn.Module] = None,
    # backward compat
    loss: Optional[nn.Module] = None,
    use_ssl_mode: SSLMode = "sl_only",
    ssl_pretrain_epochs: int = 10,
    ssl_mask_ratio: float = 0.3,
    ssl_loss_type: str = "mse",
    ssl_freeze_encoder_before_ft: bool = False,
    ssl_pretrained_ckpt_path: Optional[str] = None,
):
    return _run_total_train_generic(
        train_loader,
        val_loader,
        device,
        lookback,
        horizon,
        "weekly",
        save_dir,
        use_exogenous_mode=use_exogenous_mode,
        warmup_epochs=warmup_epochs,
        spike_epochs=spike_epochs,
        base_lr=base_lr,
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
):
    return _run_total_train_generic(
        train_loader,
        val_loader,
        device,
        lookback,
        horizon,
        "monthly",
        save_dir,
        warmup_epochs=warmup_epochs,
        spike_epochs=spike_epochs,
        base_lr=base_lr,
        models_to_run=models_to_run,
        use_exogenous_mode=use_exogenous_mode,
        loss_point=loss_point,
        loss_quantile=loss_quantile,
        loss=loss,
        use_ssl_mode=use_ssl_mode,
        ssl_pretrain_epochs=ssl_pretrain_epochs,
        ssl_mask_ratio=ssl_mask_ratio,
        ssl_loss_type=ssl_loss_type,
        ssl_freeze_encoder_before_ft=ssl_freeze_encoder_before_ft,
        ssl_pretrained_ckpt_path=ssl_pretrained_ckpt_path,
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
):
    return _run_total_train_generic(
        train_loader,
        val_loader,
        device,
        lookback,
        horizon,
        "daily",
        save_dir,
        warmup_epochs=warmup_epochs,
        spike_epochs=spike_epochs,
        base_lr=base_lr,
        models_to_run=models_to_run,
        use_exogenous_mode=use_exogenous_mode,
        loss_point=loss_point,
        loss_quantile=loss_quantile,
        loss=loss,
        use_ssl_mode=use_ssl_mode,
        ssl_pretrain_epochs=ssl_pretrain_epochs,
        ssl_mask_ratio=ssl_mask_ratio,
        ssl_loss_type=ssl_loss_type,
        ssl_freeze_encoder_before_ft=ssl_freeze_encoder_before_ft,
        ssl_pretrained_ckpt_path=ssl_pretrained_ckpt_path,
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
):
    return _run_total_train_generic(
        train_loader,
        val_loader,
        device,
        lookback,
        horizon,
        "hourly",
        save_dir,
        warmup_epochs=warmup_epochs,
        spike_epochs=spike_epochs,
        base_lr=base_lr,
        models_to_run=models_to_run,
        use_exogenous_mode=use_exogenous_mode,
        loss_point=loss_point,
        loss_quantile=loss_quantile,
        loss=loss,
        use_ssl_mode=use_ssl_mode,
        ssl_pretrain_epochs=ssl_pretrain_epochs,
        ssl_mask_ratio=ssl_mask_ratio,
        ssl_loss_type=ssl_loss_type,
        ssl_freeze_encoder_before_ft=ssl_freeze_encoder_before_ft,
        ssl_pretrained_ckpt_path=ssl_pretrained_ckpt_path,
    )

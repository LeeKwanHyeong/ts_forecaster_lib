from __future__ import annotations

import gc
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Optional, Iterable, List, Callable, Any, Literal, Tuple, Union, Mapping

import torch
import torch.nn as nn

from modeling_module.models.registry import (
    expand_training_targets,
    filter_targets_for_family,
    ordered_training_families_for_targets,
    resolve_artifact_model_key,
)
from modeling_module.models.PatchMixer.common.configs import (
    PatchMixerConfig,
    PatchMixerExogenousConfig,
)
from modeling_module.models.PatchTST.common.configs import PatchTSTConfig
from modeling_module.models.PatchTST.self_supervised.PatchTST import PatchTSTPretrainModel
from modeling_module.models.NHITS.configs import NHITSConfig
from modeling_module.models.SELLM.configs import SELLMConfig
from modeling_module.models.TimeMixer.configs import TimeMixerConfig
from modeling_module.models.TimeXer.configs import TimeXerConfig
from modeling_module.models.Titan.common.configs import TitanConfig
from modeling_module.models.model_builder import (
    build_titan_base,
    build_titan_lmm,
    build_titan_seq2seq,
    build_patch_mixer_exogenous,
    build_patch_mixer,
    build_patchTST,
    build_patchTST_exogenous,
    build_patchTST_quantile,
    build_patchTST_quantile_exogenous,
    build_exotst,
    build_nhits,
    build_timemixer,
    build_timexer,
    build_sellm,
)
from modeling_module.training.config import SpikeLossConfig, TrainingConfig, StageConfig
from modeling_module.training.model_losses.loss_module import (
    MAE,
    Huber,
    QuantileLoss,
    MQLoss,
    DistributionLoss,
)
from modeling_module.utils.checkpoint import save_model as save_checkpoint
from modeling_module.training.model_trainers.patchmixer_train import train_patchmixer
from modeling_module.training.model_trainers.patchtst_finetune import train_patchtst_finetune
from modeling_module.training.model_trainers.patchtst_pretrain import train_patchtst_pretrain
from modeling_module.training.model_trainers.patchtst_train import train_patchtst
from modeling_module.training.model_trainers.sellm_train import train_sellm
from modeling_module.training.model_trainers.timemixer_train import train_timemixer
from modeling_module.training.model_trainers.timexer_train import train_timexer
from modeling_module.training.model_trainers.nhits_train import train_nhits
from modeling_module.training.model_trainers.titan_train import train_titan
from .exotst_train import train_exotst
from ...models.ExoTST.configs import ExoTSTConfig

SSLMode = Literal["ssl_only", "full", "sl_only", "off"]

# =============================================================================
# Policy imports (freq + exo)
# =============================================================================
try:
    from .freq_policy import FreqSpec, get_freq_spec
except Exception:  # pragma: no cover
    from freq_policy import FreqSpec, get_freq_spec  # type: ignore

# exo_policy: prefer resolve_exogenous; fallback to resolve_future_exogenous for compatibility
try:
    from .exo_policy import ExoSpec, resolve_exogenous
except Exception:  # pragma: no cover
    try:
        from exo_policy import ExoSpec, resolve_exogenous  # type: ignore
    except Exception:  # pragma: no cover
        from .exo_policy import ExoSpec, resolve_future_exogenous as resolve_exogenous  # type: ignore


# =============================================================================
# Loss routing helpers
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
        for k in ["state_dict", "model_state_dict", "model", "net", "weights", "model_state"]:
            v = ckpt_obj.get(k, None)
            if isinstance(v, dict):
                return v
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    return {}


def _strip_common_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """DataParallel/Lightning 등에서 흔한 prefix 제거."""
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
    """prefix 기반 key mapping. 예: ("encoder.", "backbone.")"""
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
    DistributionLoss(distribution=..., ...)에서 out_mul, param_names를 robust하게 추론.
    """
    out_mul = int(getattr(loss_obj, "outputsize_multiplier", 0) or 0)

    distr = getattr(loss_obj, "distribution", None)
    distr_name = distr if isinstance(distr, str) else None

    pn = getattr(loss_obj, "param_names", None)
    if pn is not None:
        try:
            param_names = list(pn)
        except Exception:
            param_names = None
    else:
        param_names = None

    if out_mul <= 0:
        if (distr_name or "").lower() in ("studentt", "student_t", "student-t"):
            out_mul = 3
        else:
            out_mul = 2

    if param_names is None:
        if (distr_name or "").lower() in ("studentt", "student_t", "student-t"):
            param_names = ["df", "loc", "scale"]
        else:
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
    encoder(backbone)만 로드.
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
    return {"loaded": len(enc_sd), "missing": len(missing), "unexpected": len(unexpected)}


class QuantileAsPointLoss(nn.Module):
    """
    Quantile output에서 특정 quantile(q_star, 기본 0.5)만 뽑아서 point loss 적용.
    - loss_quantile에 Huber/MAE 등 point loss가 들어와도 학습 가능하게 함.
    """
    def __init__(self, base_loss: nn.Module, quantiles: Iterable[float], q_star: float = 0.5):
        super().__init__()
        self.base_loss = base_loss
        self.quantiles = tuple(float(q) for q in quantiles)
        self.q_star = float(q_star)

        if self.q_star not in self.quantiles:
            raise ValueError(f"q_star={self.q_star} must be in quantiles={self.quantiles}")
        self._q_idx = self.quantiles.index(self.q_star)

        # infer_supervised_mode compat
        self.is_distribution_output = False

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor, *, mask=None, y_insample=None):
        if y_hat is None:
            raise ValueError("y_hat is None")

        # normalize y_hat -> [B,H,Q]
        if y_hat.dim() == 4:
            if y_hat.shape[2] == 1:
                y_hat3 = y_hat.squeeze(2)
            else:
                y_hat3 = y_hat[:, :, 0, :]
        elif y_hat.dim() == 3:
            y_hat3 = y_hat
        else:
            raise ValueError(f"QuantileAsPointLoss expects 3D/4D y_hat, got {tuple(y_hat.shape)}")

        # [B,Q,H] -> [B,H,Q]
        if y_hat3.shape[1] != y.shape[1] and y_hat3.shape[2] == y.shape[1]:
            y_hat3 = y_hat3.permute(0, 2, 1).contiguous()

        y_hat_med = y_hat3[:, :, self._q_idx].unsqueeze(-1)  # [B,H,1]

        try:
            return self.base_loss(y, y_hat_med, mask=mask, y_insample=y_insample)
        except TypeError:
            return self.base_loss(y, y_hat_med, mask=mask)


def infer_supervised_mode(loss_obj) -> str:
    """loss object로부터 supervised 모드(point/dist/quantile) 추론."""
    if loss_obj is None:
        return "point"
    if bool(getattr(loss_obj, "is_distribution_output", False)) or (loss_obj.__class__.__name__ == "DistributionLoss"):
        return "dist"
    if loss_obj.__class__.__name__ in ("QuantileLoss", "MQLoss", "QuantileAsPointLoss", "MultiQuantilePinball"):
        return "quantile"
    return "point"


def _family_architecture_override(
    model_architecture: Optional[Mapping[str, Mapping[str, Any]]],
    family: str,
) -> Dict[str, Any]:
    if not model_architecture:
        return {}
    section = model_architecture.get(family)
    if not section:
        return {}
    return {key: value for key, value in dict(section).items() if value is not None}


def default_loss_point():
    return MAE()


def default_loss_quantile(quantiles=(0.1, 0.5, 0.9)):
    return MQLoss(quantiles=list(quantiles))


def coerce_quantile_loss(loss_quantile: Optional[nn.Module], *, quantiles=(0.1, 0.5, 0.9)) -> nn.Module:
    """loss_quantile이 None 또는 point loss여도 quantile 학습이 가능하도록 보정."""
    if loss_quantile is None:
        return default_loss_quantile(quantiles)

    if loss_quantile.__class__.__name__ in (
        "MQLoss",
        "QuantileLoss",
        "MultiQuantileLoss",
        "MultiQuantilePinball",
    ):
        return loss_quantile

    return QuantileAsPointLoss(base_loss=loss_quantile, quantiles=quantiles, q_star=0.5)


# =============================================================================
# Misc utils
# =============================================================================

def _validate_ssl_mode(use_ssl_mode: str) -> SSLMode:
    m = str(use_ssl_mode).strip().lower()
    if m == "off":
        m = "sl_only"
    if m not in ("ssl_only", "full", "sl_only"):
        raise ValueError(
            f"use_ssl_mode must be one of ['ssl_only','full','sl_only','off'], got={use_ssl_mode!r}"
        )
    return m  # type: ignore[return-value]


def _get_part_vocab_size_from_loader(loader) -> int:
    """데이터셋의 파트(ID) 어휘 크기 조회."""
    try:
        return len(getattr(loader.dataset, "part_vocab", {}))
    except Exception:
        return 0


def save_model(
    model: torch.nn.Module,
    cfg,
    path: Union[str, Path],
    *,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """학습기 내부 저장은 utils.checkpoint 포맷을 공식 포맷으로 사용한다."""
    save_checkpoint(model, cfg, str(path), extra_meta=extra_meta)


def _make_ckpt_path(save_dir: Path, freq: str, model_name: str, lookback: int, horizon: int) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{freq}_{model_name}_L{lookback}_H{horizon}.pt"
    return save_dir / fname


def _requested_target_set(requested_artifact_keys: Optional[Iterable[str]]) -> Optional[set[str]]:
    if not requested_artifact_keys:
        return None
    return {resolve_artifact_model_key(key) for key in requested_artifact_keys}


def _wants_artifact(requested_artifact_keys: Optional[set[str]], artifact_key: str) -> bool:
    return requested_artifact_keys is None or artifact_key in requested_artifact_keys


def _store_result(
    results: Dict[str, Dict],
    *,
    result_name: str,
    best: Dict[str, Any],
    model_key: Optional[str] = None,
    family_key: Optional[str] = None,
) -> None:
    record = dict(best)
    model_obj = record.pop("model", None)
    if model_obj is not None and isinstance(model_obj, torch.nn.Module):
        # Sequential family training should not keep previous GPU models alive.
        try:
            model_obj.to("cpu")
        except Exception:
            pass
        del model_obj
    if model_key is not None:
        record.setdefault("model_key", model_key)
    if family_key is not None:
        record.setdefault("family_key", family_key)
    record.setdefault("display_name", result_name)
    results[result_name] = record
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    use_intermittent: bool = True,
    val_use_weights: bool = True,
):
    """
    공통 TrainingConfig + StageConfig 생성.
    - point_train_cfg.loss: loss_point(또는 기본 MAE)
    - quantile_train_cfg.loss: loss_quantile(또는 기본 MQLoss)
    """
    base_lr = float(base_lr) if base_lr is not None else 1e-4

    loss_point_obj = loss_point if loss_point is not None else default_loss_point()
    loss_quantile_obj = coerce_quantile_loss(loss_quantile, quantiles=quantiles)

    point_train_cfg = TrainingConfig(
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
        loss=loss_point_obj,

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

    stages: List[StageConfig] = []
    if warmup_epochs and int(warmup_epochs) > 0:
        stages.append(StageConfig(epochs=int(warmup_epochs), lr=base_lr, spike_enabled=False, use_horizon_decay=False))
    if spike_epochs and int(spike_epochs) > 0:
        stages.append(StageConfig(epochs=int(spike_epochs), lr=base_lr, spike_enabled=True, use_horizon_decay=True))
    if not stages:
        stages.append(StageConfig(epochs=1, lr=base_lr, spike_enabled=False, use_horizon_decay=False))

    return point_train_cfg, quantile_train_cfg, stages


def _norm_list(xs: Optional[Iterable[str]]) -> List[str]:
    if xs is None:
        return []
    return [str(x).strip().lower() for x in xs if str(x).strip()]


def _infer_mode_and_dist(loss_obj) -> tuple[str, int, Optional[List[str]], Optional[str]]:
    """
    loss_obj -> (mode, out_mul, param_names, dist_name)
    """
    mode = infer_supervised_mode(loss_obj)
    if mode == "dist":
        out_mul, param_names, dist_name = _infer_dist_spec(loss_obj)
        return mode, int(out_mul), list(param_names) if param_names is not None else None, dist_name
    return mode, 1, None, None


# =============================================================================
# Model runners (Contract: exo dims are provided by exo_policy)
# =============================================================================

def _run_nhits(
    *,
    results: Dict[str, Dict],
    freq: str,
    train_loader,
    val_loader,
    point_train_cfg: TrainingConfig,
    stages: List[StageConfig],
    device: str,
    lookback: int,
    horizon: int,
    use_exogenous_mode: bool,
    save_root: Optional[Path] = None,
    requested_artifact_keys: Optional[Iterable[str]] = None,
    architecture_override: Optional[Mapping[str, Any]] = None,
    **kwargs,
):
    """Run the public endogenous, point-only N-HiTS artifact."""

    requested = _requested_target_set(requested_artifact_keys)
    if not _wants_artifact(requested, "nhits_base"):
        return
    if use_exogenous_mode:
        raise RuntimeError("[total_train] nhits_base supports endogenous inputs only.")

    loss_obj = getattr(point_train_cfg, "loss", None)
    mode = infer_supervised_mode(loss_obj)
    if mode != "point":
        raise NotImplementedError(
            f"[total_train] nhits_base supports only point mode, got {mode!r}."
        )

    cfg_kwargs = asdict(point_train_cfg)
    cfg_kwargs["loss"] = loss_obj
    cfg_kwargs.update(y_dim=1, use_exogenous_mode=False)
    if architecture_override:
        cfg_kwargs.update(
            {
                key: value
                for key, value in dict(architecture_override).items()
                if value is not None
            }
        )

    nhits_cfg = NHITSConfig(**cfg_kwargs)
    nhits_model = build_nhits(nhits_cfg).to(device)
    nhits_train_cfg = replace(point_train_cfg, use_exogenous_mode=False)

    print(f"N-HiTS ({freq.capitalize()}) mode=point")
    best = train_nhits(
        model=nhits_model,
        train_loader=train_loader,
        val_loader=val_loader,
        stages=list(stages),
        train_cfg=nhits_train_cfg,
        device=device,
    )

    if save_root:
        ckpt_path = _make_ckpt_path(
            save_root,
            freq,
            "NHITSBase",
            lookback,
            horizon,
        )
        save_model(
            nhits_model,
            nhits_cfg,
            ckpt_path,
            extra_meta={"model_key": "nhits_base", "family_key": "nhits"},
        )
        best["ckpt_path"] = str(ckpt_path)

    _store_result(
        results,
        result_name="N-HiTS",
        best=best,
        model_key="nhits_base",
        family_key="nhits",
    )


def _run_timemixer(
    *,
    results: Dict[str, Dict],
    freq: str,
    train_loader,
    val_loader,
    point_train_cfg: TrainingConfig,
    stages: List[StageConfig],
    device: str,
    lookback: int,
    horizon: int,
    use_exogenous_mode: bool,
    exo_dim: int,
    future_exo_cb: Optional[Callable],
    past_cont_dim: int,
    past_cat_dim: int,
    save_root: Optional[Path] = None,
    requested_artifact_keys: Optional[Iterable[str]] = None,
    architecture_override: Optional[Mapping[str, Any]] = None,
    **kwargs,
):
    """Run the endogenous, point-only TimeMixer artifact."""

    requested = _requested_target_set(requested_artifact_keys)
    if not _wants_artifact(requested, "timemixer"):
        return
    if (
        use_exogenous_mode
        or int(exo_dim) > 0
        or future_exo_cb is not None
        or int(past_cont_dim) > 0
        or int(past_cat_dim) > 0
    ):
        raise RuntimeError("[total_train] TimeMixer supports endogenous inputs only.")

    loss_obj = getattr(point_train_cfg, "loss", None)
    mode = infer_supervised_mode(loss_obj)
    if mode != "point":
        raise NotImplementedError(
            f"[total_train] TimeMixer supports only point mode, got {mode!r}."
        )

    cfg_kwargs = asdict(point_train_cfg)
    cfg_kwargs["loss"] = loss_obj
    cfg_kwargs.update(
        y_dim=1,
        use_exogenous_mode=False,
        future_exo_dim=0,
    )
    if architecture_override:
        cfg_kwargs.update(
            {
                key: value
                for key, value in dict(architecture_override).items()
                if value is not None
            }
        )

    timemixer_cfg = TimeMixerConfig(**cfg_kwargs)
    timemixer_model = build_timemixer(timemixer_cfg).to(device)
    timemixer_train_cfg = replace(point_train_cfg, use_exogenous_mode=False)

    print(f"TimeMixer ({freq.capitalize()}) mode=point")
    best = train_timemixer(
        model=timemixer_model,
        train_loader=train_loader,
        val_loader=val_loader,
        stages=list(stages),
        train_cfg=timemixer_train_cfg,
        device=device,
    )

    if save_root:
        ckpt_path = _make_ckpt_path(
            save_root,
            freq,
            "TimeMixer",
            lookback,
            horizon,
        )
        save_model(
            timemixer_model,
            timemixer_cfg,
            ckpt_path,
            extra_meta={"model_key": "timemixer", "family_key": "timemixer"},
        )
        best["ckpt_path"] = str(ckpt_path)

    _store_result(
        results,
        result_name="TimeMixer",
        best=best,
        model_key="timemixer",
        family_key="timemixer",
    )


def _run_exotst(
    *,
    results: Dict[str, Dict],
    freq: str,
    train_loader,
    val_loader,
    point_train_cfg: TrainingConfig,
    stages: List[StageConfig],
    device: str,
    lookback: int,
    horizon: int,
    patch_len: int,
    stride: int,
    use_exogenous_mode: bool,
    exo_dim: int,
    future_exo_cb: Optional[Callable],
    past_cont_dim: int,
    past_cat_dim: int,
    save_root: Optional[Path] = None,
    requested_artifact_keys: Optional[Iterable[str]] = None,
    architecture_override: Optional[Mapping[str, Any]] = None,
    **kwargs,
):
    """
    ExoTST runner:
      - past + future exo 필수.
      - exo_policy에서 past_cont_dim/exo_dim을 확정했으므로 여기서는 '검증'만 수행.
    """
    requested = _requested_target_set(requested_artifact_keys)
    if not _wants_artifact(requested, "exotst_base"):
        return

    if not use_exogenous_mode:
        raise RuntimeError("[total_train] ExoTST requires use_exogenous_mode=True (needs past+future exo).")
    if int(exo_dim) <= 0 and future_exo_cb is None:
        raise RuntimeError("[total_train] ExoTST requires future exogenous (loader fe_cont dim > 0 or future_exo_cb).")
    if int(past_cont_dim) <= 0:
        raise RuntimeError("[total_train] ExoTST requires past_exo_cont from loader (pe_cont dim > 0).")
    if int(past_cat_dim) > 0:
        print(f"[WARN] ExoTST past_exo_cat detected (d_past_cat={past_cat_dim}). "
              f"If ExoTSTConfig does not support cats, please encode cats into cont.")

    loss_obj = getattr(point_train_cfg, "loss", None)
    mode, out_mul, param_names, dist_name = _infer_mode_and_dist(loss_obj)
    if mode == "quantile":
        raise NotImplementedError("[total_train] ExoTST quantile trainer is not implemented yet.")
    head_type = "dist" if mode == "dist" else "point"

    cfg_kwargs = asdict(point_train_cfg)
    cfg_kwargs["loss"] = loss_obj
    cfg_kwargs.update(
        dict(
            y_dim=1,
            patch_len=patch_len,
            stride=stride,
            use_past_exo=True,
            use_future_exo=True,
            exo_dim_past=int(past_cont_dim),
            exo_dim_future=max(int(exo_dim), 0),
            exo_nan_policy="zero+indicator",
            head_type=head_type,
            loss_mode=mode,
            out_mul=int(out_mul),
            param_names=param_names,
            dist_name=dist_name,
            strict_shape=True,
            subtract_last=True,
        )
    )
    if architecture_override:
        cfg_kwargs.update({key: value for key, value in dict(architecture_override).items() if value is not None})

    exotst_cfg = ExoTSTConfig(**cfg_kwargs)
    exotst_model = build_exotst(exotst_cfg).to(device)

    print(f"ExoTST ({freq.capitalize()}) head_type={head_type}")
    best = train_exotst(
        model=exotst_model,
        train_loader=train_loader,
        val_loader=val_loader,
        stages=list(stages),
        train_cfg=point_train_cfg,
        device=device,
        future_exo_cb=future_exo_cb,
    )

    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, "ExoTSTBase", lookback, horizon)
        save_model(
            exotst_model,
            exotst_cfg,
            ckpt_path,
            extra_meta={"model_key": "exotst_base", "family_key": "exotst"},
        )
        best["ckpt_path"] = str(ckpt_path)

    _store_result(results, result_name="ExoTST", best=best, model_key="exotst_base", family_key="exotst")


def _run_timexer(
    *,
    results: Dict[str, Dict],
    freq: str,
    train_loader,
    val_loader,
    point_train_cfg: TrainingConfig,
    stages: List[StageConfig],
    device: str,
    lookback: int,
    horizon: int,
    patch_len: int,
    use_exogenous_mode: bool,
    exo_dim: int,
    future_exo_cb: Optional[Callable],
    past_cont_dim: int,
    past_cat_dim: int,
    save_root: Optional[Path] = None,
    requested_artifact_keys: Optional[Iterable[str]] = None,
    architecture_override: Optional[Mapping[str, Any]] = None,
    **kwargs,
):
    """
    TimeXer runner aligned to the paper contract.

    v1 assumptions:
    - historical continuous exogenous inputs are required
    - future exogenous inputs are intentionally rejected
    - point forecasting only
    """

    requested = _requested_target_set(requested_artifact_keys)
    if not _wants_artifact(requested, "timexer_base"):
        return

    if not use_exogenous_mode:
        raise RuntimeError("[total_train] TimeXer requires use_exogenous_mode=True.")
    if int(past_cont_dim) <= 0:
        raise RuntimeError("[total_train] TimeXer requires past_exo_cont from loader (pe_cont dim > 0).")
    if int(exo_dim) > 0 or future_exo_cb is not None:
        raise RuntimeError("[total_train] TimeXer v1 does not support future exogenous inputs.")
    if int(past_cat_dim) > 0:
        raise RuntimeError(
            "[total_train] TimeXer v1 supports only past continuous exogenous inputs. "
            "Encode categorical exogenous features into continuous channels first."
        )

    loss_obj = getattr(point_train_cfg, "loss", None)
    mode = infer_supervised_mode(loss_obj)
    if mode != "point":
        raise NotImplementedError(f"[total_train] TimeXer v1 supports only point mode, got {mode!r}.")

    cfg_kwargs = asdict(point_train_cfg)
    cfg_kwargs["loss"] = loss_obj
    cfg_kwargs.update(
        dict(
            y_dim=1,
            past_exo_cont_dim=int(past_cont_dim),
            patch_len=patch_len,
            use_norm=True,
        )
    )
    if architecture_override:
        cfg_kwargs.update({key: value for key, value in dict(architecture_override).items() if value is not None})

    timexer_cfg = TimeXerConfig(**cfg_kwargs)
    timexer_model = build_timexer(timexer_cfg).to(device)

    print(f"TimeXer ({freq.capitalize()})")
    best = train_timexer(
        model=timexer_model,
        train_loader=train_loader,
        val_loader=val_loader,
        stages=list(stages),
        train_cfg=point_train_cfg,
        device=device,
    )

    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, "TimeXerBase", lookback, horizon)
        save_model(
            timexer_model,
            timexer_cfg,
            ckpt_path,
            extra_meta={"model_key": "timexer_base", "family_key": "timexer"},
        )
        best["ckpt_path"] = str(ckpt_path)

    _store_result(results, result_name="TimeXer", best=best, model_key="timexer_base", family_key="timexer")


def _run_sellm(
    *,
    results: Dict[str, Dict],
    freq: str,
    train_loader,
    val_loader,
    point_train_cfg: TrainingConfig,
    stages: List[StageConfig],
    device: str,
    lookback: int,
    horizon: int,
    patch_len: int,
    use_exogenous_mode: bool,
    exo_dim: int,
    future_exo_cb: Optional[Callable],
    past_cont_dim: int,
    past_cat_dim: int,
    save_root: Optional[Path] = None,
    requested_artifact_keys: Optional[Iterable[str]] = None,
    architecture_override: Optional[Mapping[str, Any]] = None,
    **kwargs,
):
    """SELLM runner for semantic-enhanced point forecasting."""

    requested = _requested_target_set(requested_artifact_keys)
    if not _wants_artifact(requested, "sellm_base"):
        return

    if int(past_cat_dim) > 0:
        print(
            "[WARN] SELLM v1 ignores categorical past exogenous inputs. "
            "Encode them as continuous/future features if they should affect the forecast."
        )
    if use_exogenous_mode and int(past_cont_dim) > 0 and int(exo_dim) <= 0 and future_exo_cb is None:
        print(
            "[WARN] SELLM v1 uses future continuous exogenous inputs only; "
            "past_exo_cont will be ignored for this artifact."
        )

    loss_obj = getattr(point_train_cfg, "loss", None)
    mode = infer_supervised_mode(loss_obj)
    if mode != "point":
        raise NotImplementedError(f"[total_train] SELLM v1 supports only point mode, got {mode!r}.")

    cfg_kwargs = asdict(point_train_cfg)
    cfg_kwargs["loss"] = loss_obj
    cfg_kwargs.update(
        dict(
            y_dim=1,
            future_exo_dim=(max(int(exo_dim), 0) if use_exogenous_mode else 0),
            token_len=max(int(patch_len), 1),
            use_exogenous_mode=bool(use_exogenous_mode),
        )
    )
    if architecture_override:
        cfg_kwargs.update({key: value for key, value in dict(architecture_override).items() if value is not None})

    sellm_cfg = SELLMConfig(**cfg_kwargs)
    sellm_model = build_sellm(sellm_cfg).to(device)

    llm_runtime = sellm_cfg.llm_source if sellm_cfg.use_pretrained_llm else "fallback"
    print(
        f"SELLM ({freq.capitalize()}) use_pretrained_llm={sellm_cfg.use_pretrained_llm} "
        f"llm_runtime={llm_runtime}"
    )
    best = train_sellm(
        model=sellm_model,
        train_loader=train_loader,
        val_loader=val_loader,
        stages=list(stages),
        train_cfg=point_train_cfg,
        device=device,
    )

    if save_root:
        ckpt_path = _make_ckpt_path(save_root, freq, "SELLMBase", lookback, horizon)
        save_model(
            sellm_model,
            sellm_cfg,
            ckpt_path,
            extra_meta={"model_key": "sellm_base", "family_key": "sellm"},
        )
        best["ckpt_path"] = str(ckpt_path)

    _store_result(results, result_name="SELLM", best=best, model_key="sellm_base", family_key="sellm")


def _run_patchtst(
    *,
    results: Dict[str, Dict],
    freq: str,
    train_loader,
    val_loader,
    save_root: Optional[Path],
    lookback: int,
    horizon: int,
    future_exo_cb: Optional[Callable],
    exo_dim: int,
    past_cont_dim: int,
    past_cat_dim: int,
    patch_len: int,
    stride: int,
    point_train_cfg: TrainingConfig,
    quantile_train_cfg: TrainingConfig,
    stages: List[StageConfig],
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
    requested_artifact_keys: Optional[Iterable[str]] = None,
    architecture_override: Optional[Mapping[str, Any]] = None,
):
    """
    PatchTST 학습 파이프라인.
    - 중요: exo dim/past dim은 exo_policy에서 확정된 값을 그대로 사용.
    - SSL pretrain은 y-only로 강제(d_future=0, d_past_cont=0, d_past_cat=0)하여 shape mismatch 방지.
    """
    use_ssl_mode = _validate_ssl_mode(use_ssl_mode)
    if use_ssl_mode in ("ssl_only", "full") and save_root is None:
        raise ValueError(
            f"PatchTST SSL mode {use_ssl_mode!r} requires an artifact `save_dir`. "
            "Provide `save_dir` before starting PatchTST SSL training."
        )
    requested = _requested_target_set(requested_artifact_keys)
    run_explicit_exogenous = requested is not None and "patchtst_exogenous" in requested
    run_explicit_quantile_exogenous = (
        requested is not None and "patchtst_quantile_exogenous" in requested
    )
    run_base = _wants_artifact(requested, "patchtst_base") or run_explicit_exogenous
    run_quantile = (
        _wants_artifact(requested, "patchtst_quantile")
        or run_explicit_quantile_exogenous
    )

    if requested is not None and {
        "patchtst_base",
        "patchtst_exogenous",
    }.issubset(requested):
        raise ValueError("Request either patchtst_base or patchtst_exogenous, not both.")
    if requested is not None and {
        "patchtst_quantile",
        "patchtst_quantile_exogenous",
    }.issubset(requested):
        raise ValueError(
            "Request either patchtst_quantile or patchtst_quantile_exogenous, not both."
        )
    if (run_explicit_exogenous or run_explicit_quantile_exogenous) and not use_exogenous_mode:
        raise ValueError("Explicit PatchTST exogenous variants require use_exogenous_mode=True.")
    if (run_explicit_exogenous or run_explicit_quantile_exogenous) and not any(
        (int(exo_dim), int(past_cont_dim), int(past_cat_dim))
    ):
        raise ValueError("Explicit PatchTST exogenous variants require configured exogenous features.")

    # ------------------------------------------------------------
    # 1) PatchTST common kwargs
    # ------------------------------------------------------------
    pt_kwargs = dict(
        device=device,
        lookback=lookback,
        horizon=horizon,
        c_in=1,
        patch_len=patch_len,
        stride=stride,
        padding_patch="end",
        future_exo_dim=int(exo_dim) if use_exogenous_mode else 0,
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
    if architecture_override:
        pt_kwargs.update({key: value for key, value in dict(architecture_override).items() if value is not None})

    if use_exogenous_mode:
        pt_kwargs.update(
            dict(
                past_exo_cont_dim=int(past_cont_dim),
                past_exo_cat_dim=int(past_cat_dim),
                # cat 미사용이면 []/0 유지
                cat_cardinalities=[],
                d_cat_emb=0,
            )
        )
    else:
        # exo OFF 불변식
        pt_kwargs.update(dict(past_exo_cont_dim=0, past_exo_cat_dim=0, cat_cardinalities=[], d_cat_emb=0))

    # ------------------------------------------------------------
    # 2) external pretrained ckpt path(optional)
    # ------------------------------------------------------------
    pretrain_ckpt_path = None
    if ssl_pretrained_ckpt_path:
        if not os.path.exists(ssl_pretrained_ckpt_path):
            raise FileNotFoundError(ssl_pretrained_ckpt_path)
        pretrain_ckpt_path = str(ssl_pretrained_ckpt_path)
        print(f"[SSL] use external pretrained ckpt: {pretrain_ckpt_path}")

    # ------------------------------------------------------------
    # 3) SSL pretrain (Optional)
    # ------------------------------------------------------------
    if (use_ssl_mode in ("ssl_only", "full")) and (pretrain_ckpt_path is None) and (save_root is not None):
        pretrain_dir = Path(save_root) / "pretrain"
        pretrain_dir.mkdir(parents=True, exist_ok=True)
        pretrain_ckpt_path = str(pretrain_dir / "patchtst_pretrain_best.pt")

        pt_pre_kwargs = dict(pt_kwargs)
        pt_pre_kwargs["future_exo_dim"] = 0
        pt_pre_kwargs["past_exo_cont_dim"] = 0
        pt_pre_kwargs["past_exo_cat_dim"] = 0

        pt_pre_cfg = PatchTSTConfig(**pt_pre_kwargs)
        pre_model = PatchTSTPretrainModel(cfg=pt_pre_cfg)

        pre_stages = [StageConfig(epochs=int(ssl_pretrain_epochs), lr=float(point_train_cfg.lr), spike_enabled=False)]
        print(f"[SSL] PatchTST Pretrain ({freq.capitalize()}) -> {pretrain_ckpt_path}")

        _ = train_patchtst_pretrain(
            pre_model,
            train_loader,
            val_loader,
            train_cfg=point_train_cfg,
            stages=pre_stages,
            mask_ratio=float(ssl_mask_ratio),
            loss_type=str(ssl_loss_type),
            save_dir=str(pretrain_dir),
            ckpt_name="patchtst_pretrain_best.pt",
            device=device,
        )

    if use_ssl_mode == "ssl_only":
        results["PatchTST SSL"] = {
            "pretrain_ckpt_path": pretrain_ckpt_path,
            "note": "use_ssl_mode='ssl_only' 이므로 supervised(point/dist/quantile) 학습은 수행하지 않음",
        }
        return

    # ============================================================
    # 4) Supervised - Base (Point or Dist)
    # ============================================================
    loss_point_obj = loss_point if loss_point is not None else point_train_cfg.loss
    mode, out_mul, param_names, dist_name = _infer_mode_and_dist(loss_point_obj)

    # future_exo_cb는 exo ON일 때만 의미 있음
    _fcb = future_exo_cb if use_exogenous_mode else None
    print(f"[PatchTST][EXO] use_exo={use_exogenous_mode} future_exo_dim={pt_kwargs.get('future_exo_dim')} "
          f"past_exo_cont_dim={pt_kwargs.get('past_exo_cont_dim')} future_cb={_fcb is not None} "
          f"past_exo_cat_dim={pt_kwargs.get('past_exo_cat_dim')}")

    if run_base:
        point_model_key = "patchtst_exogenous" if run_explicit_exogenous else "patchtst_base"
        pt_train_cfg = PatchTSTConfig(
            **pt_kwargs,
            loss=loss_point_obj,
            loss_mode=mode,
            out_mul=int(out_mul),
            param_names=param_names,
            dist_name=dist_name,
        )

        point_builder = build_patchTST_exogenous if run_explicit_exogenous else build_patchTST
        pt_base = point_builder(pt_train_cfg)
        name_base = "PatchTST Exogenous" if run_explicit_exogenous else "PatchTST"
        print(f"{name_base} ({freq.capitalize()})")

        if (use_ssl_mode == "full") and (pretrain_ckpt_path is not None):
            best_pt_base = train_patchtst_finetune(
                pt_base,
                train_loader,
                val_loader,
                train_cfg=point_train_cfg,
                stages=list(stages),
                future_exo_cb=_fcb,
                pretrain_ckpt_path=pretrain_ckpt_path,
                load_strict=False,
                freeze_encoder_before_ft=bool(ssl_freeze_encoder_before_ft),
                device=device,
            )
        else:
            best_pt_base = train_patchtst(
                pt_base,
                train_loader,
                val_loader,
                train_cfg=point_train_cfg,
                stages=list(stages),
                future_exo_cb=_fcb,
                device=device,
            )

        if save_root:
            artifact_name = "PatchTSTExogenous" if run_explicit_exogenous else "PatchTST"
            ckpt_path = _make_ckpt_path(save_root, freq, artifact_name, lookback, horizon)
            save_model(
                pt_base,
                pt_train_cfg,
                ckpt_path,
                extra_meta={"model_key": point_model_key, "family_key": "patchtst"},
            )
            best_pt_base["ckpt_path"] = str(ckpt_path)
            if (use_ssl_mode == "full") and (pretrain_ckpt_path is not None):
                best_pt_base["pretrain_ckpt_path"] = str(pretrain_ckpt_path)
        _store_result(
            results,
            result_name=name_base,
            best=best_pt_base,
            model_key=point_model_key,
            family_key="patchtst",
        )

    # ============================================================
    # 5) Supervised - Quantile
    # ============================================================
    if run_quantile:
        quantile_model_key = (
            "patchtst_quantile_exogenous"
            if run_explicit_quantile_exogenous
            else "patchtst_quantile"
        )
        quantiles = (0.1, 0.5, 0.9)
        loss_q_obj = coerce_quantile_loss(loss_quantile, quantiles=quantiles)
        quantile_train_cfg = replace(quantile_train_cfg, loss=loss_q_obj)

        pt_q_cfg = PatchTSTConfig(**pt_kwargs, quantiles=quantiles, loss=loss_q_obj)
        quantile_builder = (
            build_patchTST_quantile_exogenous
            if run_explicit_quantile_exogenous
            else build_patchTST_quantile
        )
        pt_q = quantile_builder(pt_q_cfg)
        quantile_name = (
            "PatchTST Quantile Exogenous"
            if run_explicit_quantile_exogenous
            else "PatchTST Quantile"
        )
        print(f"{quantile_name} ({freq.capitalize()})")

        if (use_ssl_mode == "full") and (pretrain_ckpt_path is not None):
            best_pt_q = train_patchtst_finetune(
                pt_q,
                train_loader,
                val_loader,
                train_cfg=quantile_train_cfg,
                stages=list(stages),
                future_exo_cb=_fcb,
                pretrain_ckpt_path=pretrain_ckpt_path,
                load_strict=False,  # head mismatch 허용
                freeze_encoder_before_ft=bool(ssl_freeze_encoder_before_ft),
                device=device,
            )
        else:
            best_pt_q = train_patchtst(
                pt_q,
                train_loader,
                val_loader,
                train_cfg=quantile_train_cfg,
                stages=list(stages),
                future_exo_cb=_fcb,
                device=device,
            )

        if save_root:
            artifact_name = (
                "PatchTSTQuantileExogenous"
                if run_explicit_quantile_exogenous
                else "PatchTSTQuantile"
            )
            ckpt_path_q = _make_ckpt_path(save_root, freq, artifact_name, lookback, horizon)
            save_model(
                pt_q,
                pt_q_cfg,
                ckpt_path_q,
                extra_meta={"model_key": quantile_model_key, "family_key": "patchtst"},
            )
            best_pt_q["ckpt_path"] = str(ckpt_path_q)
            if (use_ssl_mode == "full") and (pretrain_ckpt_path is not None):
                best_pt_q["pretrain_ckpt_path"] = str(pretrain_ckpt_path)

        _store_result(
            results,
            result_name=quantile_name,
            best=best_pt_q,
            model_key=quantile_model_key,
            family_key="patchtst",
        )


def _run_titan(
    *,
    results: Dict[str, Dict],
    freq: str,
    train_loader,
    val_loader,
    save_root: Optional[Path],
    lookback: int,
    horizon: int,
    use_exogenous_mode: bool,
    future_exo_cb: Optional[Callable],
    exo_dim: int,
    past_cont_dim: int,
    past_cat_dim: int,
    loss_point: Optional[nn.Module] = None,
    point_train_cfg,
    stages,
    device: str,
    requested_artifact_keys: Optional[Iterable[str]] = None,
    architecture_override: Optional[Mapping[str, Any]] = None,
):
    """
    Titan runner (LMM + Seq2Seq)
    - 현재 구현에서는 past_exo_cat을 사용하지 않는다고 가정 (필요 시 Titan 모델 확장)
    """

    loss_point_obj = loss_point if loss_point is not None else point_train_cfg.loss
    mode, out_mul, param_names, dist_name = _infer_mode_and_dist(loss_point_obj)
    name_suffix = " Dist" if mode == "dist" else ""
    requested = _requested_target_set(requested_artifact_keys)
    run_base = _wants_artifact(requested, "titan_base")
    run_lmm = _wants_artifact(requested, "titan_lmm")
    run_seq2seq = _wants_artifact(requested, "titan_seq2seq")
    print(f"[_run_titan] mode={mode} out_mul={out_mul} param_names={param_names}")

    # -------------------------
    # 1) cat 비활성 (현재 runner 정책)
    # -------------------------
    if int(past_cat_dim) > 0:
        print(f"[WARN] Titan runner ignores past_exo_cat (past_cat_dim={past_cat_dim}). Force to 0.")
    past_cat_dim = 0

    # -------------------------
    # 2) EXO dim 최종 결정 (한 번만)
    # -------------------------
    if use_exogenous_mode:
        future_exo_dim = int(exo_dim)
        past_exo_cont_dim = int(past_cont_dim)
        past_exo_cat_dim = 0
    else:
        future_exo_dim = 0
        past_exo_cont_dim = 0
        past_exo_cat_dim = 0

    # future_exo_dim == 0이면 cb도 꺼야 안전
    _fcb = future_exo_cb if (use_exogenous_mode and future_exo_dim > 0) else None

    # -------------------------
    # 3) TitanConfig 생성
    # -------------------------
    ti_kwargs = dict(
        lookback=lookback,
        horizon=horizon,
        d_model=256,
        n_layers=3,
        n_heads=4,
        d_ff=4 * 256,
        dropout=0.1,
        contextual_mem_size=(512 if freq == "hourly" else 256),
        persistent_mem_size=64,

        # EXO
        future_exo_dim=future_exo_dim,
        past_exo_cont_dim=past_exo_cont_dim,
        past_exo_cat_dim=past_exo_cat_dim,
        past_exo_cat_embed_dim=None,

        use_revin=True,
        final_clamp_nonneg=False,
        loss=loss_point_obj,
        loss_mode=mode,
        out_mul=int(out_mul),
        param_names=param_names,
        dist_name=dist_name,
    )
    if architecture_override:
        ti_kwargs.update({key: value for key, value in dict(architecture_override).items() if value is not None})

    print(
        f"[Titan][EXO] use_exo={use_exogenous_mode} "
        f"future_exo_dim={ti_kwargs['future_exo_dim']} "
        f"past_exo_cont_dim={ti_kwargs['past_exo_cont_dim']} "
        f"past_exo_cat_dim={ti_kwargs['past_exo_cat_dim']} "
        f"future_cb={_fcb is not None}"
    )

    ti_config = TitanConfig(**ti_kwargs)

    # 불변식 강제 (A0 같은 케이스에서 실수 방지)
    if not use_exogenous_mode:
        assert ti_config.future_exo_dim == 0
        assert ti_config.past_exo_cont_dim == 0
        assert ti_config.past_exo_cat_dim == 0

    if run_base:
        name_base = f"Titan Base{name_suffix}"
        ckpt_name_base = "TitanBaseDist" if mode == "dist" else "TitanBase"
        print(f"{name_base} ({freq.capitalize()})")

        ti_base = build_titan_base(
            ti_config,
            out_mult=(out_mul if mode == "dist" else 1),
            param_names=param_names,
        )

        best_ti_base = train_titan(
            ti_base,
            train_loader,
            val_loader,
            device=device,
            train_cfg=point_train_cfg,
            stages=list(stages),
            future_exo_cb=_fcb,
        )

        if save_root:
            ckpt_path = _make_ckpt_path(save_root, freq, ckpt_name_base, lookback, horizon)
            save_model(
                ti_base,
                ti_config,
                ckpt_path,
                extra_meta={"model_key": "titan_base", "family_key": "titan"},
            )
            best_ti_base["ckpt_path"] = str(ckpt_path)

        _store_result(
            results,
            result_name=name_base,
            best=best_ti_base,
            model_key="titan_base",
            family_key="titan",
        )

    if run_lmm:
        name_lmm = f"Titan LMM{name_suffix}"
        ckpt_name_lmm = "TitanLMMDist" if mode == "dist" else "TitanLMM"
        print(f"{name_lmm} ({freq.capitalize()})")

        ti_lmm = build_titan_lmm(
            ti_config,
            out_mult=(out_mul if mode == "dist" else 1),
            param_names=param_names
        )

        best_ti_lmm = train_titan(
            ti_lmm,
            train_loader,
            val_loader,
            device=device,
            train_cfg=point_train_cfg,
            stages=list(stages),
            future_exo_cb=_fcb,
        )

        if save_root:
            ckpt_path = _make_ckpt_path(save_root, freq, ckpt_name_lmm, lookback, horizon)
            save_model(
                ti_lmm,
                ti_config,
                ckpt_path,
                extra_meta={"model_key": "titan_lmm", "family_key": "titan"},
            )
            best_ti_lmm["ckpt_path"] = str(ckpt_path)

        _store_result(
            results,
            result_name=name_lmm,
            best=best_ti_lmm,
            model_key="titan_lmm",
            family_key="titan",
        )

    if run_seq2seq:
        name_s2s = f"Titan Seq2Seq{name_suffix}"
        ckpt_name_s2s = "TitanSeq2SeqDist" if mode == "dist" else "TitanSeq2Seq"
        print(f"{name_s2s} ({freq.capitalize()})")

        ti_s2s = build_titan_seq2seq(
            ti_config,
            out_mult=(out_mul if mode == "dist" else 1),
            param_names=param_names
        )

        best_ti_s2s = train_titan(
            ti_s2s,
            train_loader,
            val_loader,
            device=device,
            train_cfg=point_train_cfg,
            stages=list(stages),
            future_exo_cb=_fcb,
        )

        print("[DEBUG] ti_config.past_exo_cont_dim =", ti_config.past_exo_cont_dim)
        print("[DEBUG] ti_config.future_exo_dim    =", ti_config.future_exo_dim)

        if save_root:
            ckpt_path = _make_ckpt_path(save_root, freq, ckpt_name_s2s, lookback, horizon)
            save_model(
                ti_s2s,
                ti_config,
                ckpt_path,
                extra_meta={"model_key": "titan_seq2seq", "family_key": "titan"},
            )
            best_ti_s2s["ckpt_path"] = str(ckpt_path)

        _store_result(
            results,
            result_name=name_s2s,
            best=best_ti_s2s,
            model_key="titan_seq2seq",
            family_key="titan",
        )

def _run_patchmixer(
    *,
    results: Dict[str, Dict],
    freq: str,
    train_loader,
    val_loader,
    save_root: Optional[Path],
    lookback: int,
    horizon: int,
    future_exo_cb: Optional[Callable],
    exo_dim: int,
    past_cont_dim: int,
    past_cat_dim: int,
    patch_len: int,
    stride: int,
    season_period: int,
    loss_point: Optional[nn.Module] = None,
    loss_quantile: Optional[nn.Module] = None,
    loss: Optional[nn.Module] = None,  # backward compat
    use_exogenous_mode: bool = True,
    point_train_cfg: TrainingConfig = None,  # type: ignore[assignment]
    quantile_train_cfg: TrainingConfig = None,  # type: ignore[assignment]
    stages: List[StageConfig] = None,  # type: ignore[assignment]
    device: str = "cuda",
    requested_artifact_keys: Optional[Iterable[str]] = None,
    architecture_override: Optional[Mapping[str, Any]] = None,
):
    """Train the paper endogenous model or the dedicated exogenous point model."""
    if stages is None:
        stages = [StageConfig(epochs=1, lr=1e-4, spike_enabled=False)]

    loss_point_obj = loss if loss is not None else (loss_point if loss_point is not None else point_train_cfg.loss)
    mode, _, _, _ = _infer_mode_and_dist(loss_point_obj)
    if mode != "point":
        raise NotImplementedError(
            "PatchMixer public training supports point loss only; retired distribution "
            "and quantile artifacts remain load-only."
        )

    requested = _requested_target_set(requested_artifact_keys)
    run_endogenous = _wants_artifact(requested, "patchmixer")
    run_exogenous = _wants_artifact(requested, "patchmixer_exo")

    if run_exogenous and not use_exogenous_mode:
        raise ValueError("patchmixer_exo requires use_exogenous_mode=True.")
    if run_exogenous and not any(
        (int(exo_dim), int(past_cont_dim), int(past_cat_dim))
    ):
        raise ValueError("patchmixer_exo requires configured exogenous features.")
    if run_endogenous and use_exogenous_mode:
        raise ValueError("patchmixer is endogenous-only; set use_exogenous_mode=False.")

    # cat embedding 정책 (데이터 메타로 대체 권장)
    if use_exogenous_mode and int(past_cat_dim) > 0:
        past_cat_vocab_sizes = tuple([512] * int(past_cat_dim))
        past_cat_embed_dims = tuple([16] * int(past_cat_dim))
    else:
        past_cat_vocab_sizes = ()
        past_cat_embed_dims = ()



    exogenous_kwargs = dict(
        lookback=lookback,
        horizon=horizon,
        device=device,
        enc_in=1,
        d_model=128,
        e_layers=6,
        mixer_kernel_size=5,
        patch_len=patch_len,
        stride=stride,
        dropout=0.1,
        f_out=256,
        head_hidden=256,

        # exo (SSOT)
        future_exo_dim=(int(exo_dim) if use_exogenous_mode else 0),
        past_exo_cont_dim=(int(past_cont_dim) if use_exogenous_mode else 0),
        past_exo_cat_dim=(int(past_cat_dim) if use_exogenous_mode else 0),
        past_exo_cat_vocab_sizes=past_cat_vocab_sizes,
        past_exo_cat_embed_dims=past_cat_embed_dims,

        use_part_embedding=False,
        part_vocab_size=_get_part_vocab_size_from_loader(train_loader),
        part_embed_dim=16,

        final_nonneg=True,
        use_revin=True,
        exo_is_normalized_default=True,

        expander_season_period=int(season_period),
        expander_n_harmonics=min(int(season_period) // 2, 24),

        out_mul=1,
        param_names=None,

        head_dropout=0.02,
        learn_output_scale=True,
        learn_dw_gain=True,
        past_exo_mode="z_gate",
    )
    if architecture_override:
        exogenous_kwargs.update(
            {key: value for key, value in dict(architecture_override).items() if value is not None}
        )

    _fcb = future_exo_cb if use_exogenous_mode else None
    if use_exogenous_mode:
        exogenous_kwargs.update(
            dict(
                past_exo_cont_dim=int(past_cont_dim),
                past_exo_cat_dim=int(past_cat_dim),
                # cat 미사용이면 []/0 유지
                past_exo_cat_vocab_sizes=(),
                past_exo_cat_embed_dims=(),
            )
        )
    else:
        # exo OFF 불변식
        exogenous_kwargs.update(dict(future_exo_dim=0,
                                     past_exo_cont_dim=0,
                                     past_exo_cat_dim=0,
                                     past_exo_cat_vocab_sizes=(),
                                     past_exo_cat_embed_dims=()))

    print(f"[PatchMixer][EXO] use_exo={use_exogenous_mode} future_exo_dim={exogenous_kwargs.get('future_exo_dim')} "
          f"past_exo_cont_dim={exogenous_kwargs.get('past_exo_cont_dim')} future_cb={_fcb is not None} "
          f"past_exo_cat_dim={exogenous_kwargs.get('past_exo_cat_dim')}")

    if run_endogenous:
        endogenous_source = dict(exogenous_kwargs)
        if architecture_override:
            endogenous_source.update(dict(architecture_override))
        pm_cfg = PatchMixerConfig.from_config(endogenous_source)
        pm_model = build_patch_mixer(pm_cfg)
        endogenous_train_cfg = replace(point_train_cfg, use_exogenous_mode=False)

        print(f"PatchMixer ({freq.capitalize()}) mode=point")
        best_pm = train_patchmixer(
            pm_model,
            train_loader,
            val_loader,
            device=device,
            train_cfg=endogenous_train_cfg,
            stages=list(stages),
            future_exo_cb=None,
        )
        if save_root:
            ckpt_path = _make_ckpt_path(save_root, freq, "PatchMixer", lookback, horizon)
            save_model(
                pm_model,
                pm_cfg,
                ckpt_path,
                extra_meta={"model_key": "patchmixer", "family_key": "patchmixer"},
            )
            best_pm["ckpt_path"] = str(ckpt_path)
        _store_result(
            results,
            result_name="PatchMixer",
            best=best_pm,
            model_key="patchmixer",
            family_key="patchmixer",
        )

    if run_exogenous:
        pm_exo_cfg = PatchMixerExogenousConfig(**exogenous_kwargs)
        pm_exo_cfg.loss = loss_point_obj
        pm_exo_model = build_patch_mixer_exogenous(pm_exo_cfg)

        print(f"PatchMixer Exogenous ({freq.capitalize()}) mode=point")
        best_pm_exo = train_patchmixer(
            pm_exo_model,
            train_loader,
            val_loader,
            device=device,
            train_cfg=point_train_cfg,
            stages=list(stages),
            future_exo_cb=_fcb,
        )
        if save_root:
            ckpt_path = _make_ckpt_path(save_root, freq, "PatchMixerExogenous", lookback, horizon)
            save_model(
                pm_exo_model,
                pm_exo_cfg,
                ckpt_path,
                extra_meta={"model_key": "patchmixer_exo", "family_key": "patchmixer"},
            )
            best_pm_exo["ckpt_path"] = str(ckpt_path)
        _store_result(
            results,
            result_name="PatchMixer Exogenous",
            best=best_pm_exo,
            model_key="patchmixer_exo",
            family_key="patchmixer",
        )


MODEL_REGISTRY: Dict[str, Callable] = {
    "patchtst": _run_patchtst,
    "titan": _run_titan,
    "patchmixer": _run_patchmixer,
    "exotst": _run_exotst,
    "nhits": _run_nhits,
    "timemixer": _run_timemixer,
    "timexer": _run_timexer,
    "sellm": _run_sellm,
}


# =============================================================================
# Orchestration
# =============================================================================

def _resolve_requested_artifact_keys(models_to_run: Optional[Iterable[str]]) -> List[str]:
    return expand_training_targets(models_to_run)


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
    """
    모든 runner가 공유하는 kwargs.
    - exo_spec으로부터 exo_dim/future_exo_cb/past dims를 주입 (SSOT)
    """
    point_train_cfg = replace(point_train_cfg, use_exogenous_mode=bool(exo_spec.use_exogenous_mode))
    quantile_train_cfg = replace(quantile_train_cfg, use_exogenous_mode=bool(exo_spec.use_exogenous_mode))

    return dict(
        results=results,
        freq=freq_spec.freq,
        train_loader=train_loader,
        val_loader=val_loader,
        save_root=save_root,
        lookback=lookback,
        horizon=horizon,

        use_exogenous_mode=exo_spec.use_exogenous_mode,
        exo_dim=int(exo_spec.exo_dim),
        future_exo_cb=(exo_spec.future_exo_cb if exo_spec.use_exogenous_mode else None),
        past_cont_dim=int(exo_spec.past_cont_dim),
        past_cat_dim=int(exo_spec.past_cat_dim),

        point_train_cfg=point_train_cfg,
        quantile_train_cfg=quantile_train_cfg,
        stages=stages,
        device=device,
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
    use_past_exogenous: bool = True,
    use_future_exogenous: bool = True,
    models_to_run: Optional[Iterable[str]] = None,
    model_architecture: Optional[Mapping[str, Mapping[str, Any]]] = None,

    # loss routing
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

    Flow
      1) freq_policy -> patch/stride/season_period
      2) common train cfg + stages
      3) exo_policy.resolve_exogenous -> exo_spec (future + past, individual toggles)
      4) run selected model runners with common kwargs
    """
    freq_spec = get_freq_spec(freq)
    save_root = Path(save_dir) if save_dir is not None and str(save_dir).strip() else None

    selected_artifact_keys = _resolve_requested_artifact_keys(models_to_run)
    selected_families = ordered_training_families_for_targets(selected_artifact_keys)
    use_ssl_mode = _validate_ssl_mode(use_ssl_mode)
    if use_ssl_mode in ("ssl_only", "full"):
        if "patchtst" not in selected_families:
            requested = ", ".join(selected_artifact_keys)
            raise ValueError(
                f"PatchTST SSL mode {use_ssl_mode!r} requires at least one PatchTST artifact. "
                f"Requested models: {requested}."
            )
        if save_root is None:
            raise ValueError(
                f"PatchTST SSL mode {use_ssl_mode!r} requires an artifact `save_dir`. "
                "Provide `save_dir` before starting PatchTST SSL training."
            )

    # backward-compat: loss -> point loss
    if loss_point is None and loss is not None:
        loss_point = loss

    # training configs + stages
    point_train_cfg, quantile_train_cfg, stages = _build_common_train_configs(
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
        use_intermittent=bool(use_intermittent),
        val_use_weights=bool(val_use_weights),
    )

    # Decide family routing early so pure TimeXer runs can force past-only semantics.
    timexer_only = bool(selected_families) and set(selected_families) == {"timexer"}
    effective_use_future_exogenous = bool(use_future_exogenous)
    if timexer_only and effective_use_future_exogenous:
        print("[total_train][INFO] timexer-only run forces use_future_exogenous=False.")
        effective_use_future_exogenous = False

    # exogenous SSOT
    exo_spec = resolve_exogenous(
        train_loader,
        freq_spec=freq_spec,
        use_exogenous_mode=bool(use_exogenous_mode),
        use_past_exogenous=bool(use_past_exogenous),
        use_future_exogenous=bool(effective_use_future_exogenous),
        lookback=lookback,
        horizon=horizon,
        allow_past_only=False,
    )
    print(f"[total_train][EXO] use_exo={exo_spec.use_exogenous_mode} "
          f"source={exo_spec.source} exo_dim={exo_spec.exo_dim} "
          f"future_cb={(exo_spec.future_exo_cb is not None)} "
          f"past_cont={exo_spec.past_cont_dim} past_cat={exo_spec.past_cat_dim}")


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

    for m in selected_families:
        family_targets = filter_targets_for_family(selected_artifact_keys, m)
        print(f"\n[total_train] === RUN: {m} ({freq_spec.freq}) targets={family_targets} ===")
        kwargs = dict(base_kwargs)
        kwargs["requested_artifact_keys"] = family_targets
        kwargs["architecture_override"] = _family_architecture_override(model_architecture, m)

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
        elif m == "exotst":
            kwargs.update(dict(patch_len=freq_spec.patch_len, stride=freq_spec.stride))
        elif m == "timexer":
            # TimeXer v1 intentionally ignores the library's future-exo fallback callback.
            kwargs.update(dict(patch_len=freq_spec.patch_len, exo_dim=0, future_exo_cb=None))
        elif m == "sellm":
            kwargs.update(dict(patch_len=freq_spec.patch_len))

        MODEL_REGISTRY[m](**kwargs)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# backward compatible alias
_run_total_train_generic = run_total_train


# =============================================================================
# Convenience wrappers
# =============================================================================

def run_total_train_weekly(
    train_loader,
    val_loader,
    device="cuda" if torch.cuda.is_available() else "cpu",
    *,
    lookback: int,
    horizon: int,
    warmup_epochs=None,
    spike_epochs=None,
    base_lr=None,
    save_dir=None,
    use_exogenous_mode: bool = False,
    use_past_exogenous: bool = True,
    use_future_exogenous: bool = True,
    models_to_run=None,
    model_architecture: Optional[Mapping[str, Mapping[str, Any]]] = None,
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
        use_past_exogenous=use_past_exogenous,
        use_future_exogenous=use_future_exogenous,
        models_to_run=models_to_run,
        model_architecture=model_architecture,
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
    lookback: int,
    horizon: int,
    warmup_epochs=None,
    spike_epochs=None,
    base_lr=None,
    save_dir=None,
    use_exogenous_mode: bool = False,
    use_past_exogenous: bool = True,
    use_future_exogenous: bool = True,
    models_to_run=None,
    model_architecture: Optional[Mapping[str, Mapping[str, Any]]] = None,
    loss_point: Optional[nn.Module] = None,
    loss_quantile: Optional[nn.Module] = None,
    loss: Optional[nn.Module] = None,
    use_ssl_mode: SSLMode = "sl_only",
    ssl_pretrain_epochs: int = 2,
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
        use_past_exogenous=use_past_exogenous,
        use_future_exogenous=use_future_exogenous,
        models_to_run=models_to_run,
        model_architecture=model_architecture,
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
    lookback: int,
    horizon: int,
    warmup_epochs=None,
    spike_epochs=None,
    base_lr=None,
    save_dir=None,
    use_exogenous_mode: bool = False,
    use_past_exogenous: bool = True,
    use_future_exogenous: bool = True,
    models_to_run=None,
    model_architecture: Optional[Mapping[str, Mapping[str, Any]]] = None,
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
        use_past_exogenous=use_past_exogenous,
        use_future_exogenous=use_future_exogenous,
        models_to_run=models_to_run,
        model_architecture=model_architecture,
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
    lookback: int,
    horizon: int,
    warmup_epochs=None,
    spike_epochs=None,
    base_lr=None,
    save_dir=None,
    use_exogenous_mode: bool = False,
    use_past_exogenous: bool = True,
    use_future_exogenous: bool = True,
    models_to_run=None,
    model_architecture: Optional[Mapping[str, Mapping[str, Any]]] = None,
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
        use_past_exogenous=use_past_exogenous,
        use_future_exogenous=use_future_exogenous,
        models_to_run=models_to_run,
        model_architecture=model_architecture,
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

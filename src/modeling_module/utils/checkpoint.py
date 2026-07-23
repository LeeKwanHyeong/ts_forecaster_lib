import os
import json
import glob
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Callable, Optional, Any, Mapping

import torch
from dataclasses import asdict, is_dataclass

from modeling_module.models.PatchMixer.common.configs import (
    PatchMixerConfig,
    PatchMixerExogenousConfig,
    PatchMixerConfigMonthly,
    PatchMixerConfigWeekly,
)
from modeling_module.models.PatchTST.common.configs import (
    PatchTSTConfigMonthly,
    PatchTSTConfig,
    HeadConfig,
    AttentionConfig,
)
from modeling_module.models.TimeXer.configs import TimeXerConfig
from modeling_module.models.TimeMixer.configs import TimeMixerConfig
from modeling_module.models.NHITS.configs import NHITSConfig
from modeling_module.models.Titan.common.configs import TitanConfig
from modeling_module.training.config import DecompositionConfig


# ------------------------------------------------------------------
# 0) (선택) 옛 포맷 지원용: config dict → config 객체 복원 함수들
# ------------------------------------------------------------------
def _rebuild_patchtst(cfgd: dict):
    cfgd = dict(cfgd)
    if "attn" in cfgd and isinstance(cfgd["attn"], dict):
        cfgd["attn"] = AttentionConfig(**cfgd["attn"])
    if "head" in cfgd and isinstance(cfgd["head"], dict):
        cfgd["head"] = HeadConfig(**cfgd["head"])
    if "decomp" in cfgd and isinstance(cfgd["decomp"], dict):
        cfgd["decomp"] = DecompositionConfig(**cfgd["decomp"])
    return PatchTSTConfig(**cfgd)


def _rebuild_patchmixer_monthly(cfgd: dict):
    return PatchMixerConfigMonthly(**cfgd)


def _rebuild_patchmixer_weekly(cfgd: dict):
    return PatchMixerConfigWeekly(**cfgd)


def _rebuild_patchmixer_original(cfgd: dict):
    return PatchMixerConfig.from_config(cfgd)


def _rebuild_patchmixer_exogenous(cfgd: dict):
    return PatchMixerExogenousConfig(**cfgd)


def _rebuild_titan(cfgd: dict):
    return TitanConfig(**cfgd)


def _rebuild_timexer(cfgd: dict):
    return TimeXerConfig(**cfgd)


def _rebuild_nhits(cfgd: dict):
    return NHITSConfig(**cfgd)


def _rebuild_timemixer(cfgd: dict):
    return TimeMixerConfig(**cfgd)


_REBUILDERS_BY_CLS = {
    # PatchTST
    "PatchTSTConfig": _rebuild_patchtst,
    "PatchTSTConfigMonthly": lambda d: PatchTSTConfigMonthly(**d),
    # PatchMixer
    "PatchMixerConfigMonthly": _rebuild_patchmixer_monthly,
    "PatchMixerConfigWeekly": _rebuild_patchmixer_weekly,
    "PatchMixerExogenousConfig": _rebuild_patchmixer_exogenous,
    "PatchMixerOriginalConfig": _rebuild_patchmixer_original,
    # Titan
    "TitanConfig": _rebuild_titan,
    # TimeXer
    "TimeXerConfig": _rebuild_timexer,
    "NHITSConfig": _rebuild_nhits,
    "TimeMixerConfig": _rebuild_timemixer,
}

# -----------------------------
# helpers: primitive sanitize
# -----------------------------
_PRIMITIVE_TYPES = (str, int, float, bool, type(None))
CHECKPOINT_FORMAT_VERSION = "modeling_module.ckpt.v3"
TRAINING_MANIFEST_VERSION = "modeling_module.training.v1"
_DISTRIBUTION_LOSS_SPEC_TYPE = "modeling_module.DistributionLoss"
_DISTRIBUTION_LOSS_SPEC_VERSION = 1
_DISTRIBUTION_CONTRACTS = {
    "Normal": (2, ["-loc", "-scale"]),
    "StudentT": (3, ["-df", "-loc", "-scale"]),
}

def _is_primitive(x: Any) -> bool:
    return isinstance(x, _PRIMITIVE_TYPES)

def _sanitize(obj: Any, *, max_depth: int = 8, _depth: int = 0) -> Any:
    """
    cfg_state 안에 들어있는 값들을 'pickle-free' 하게 정리.
    - dict/list/tuple 재귀 처리
    - nn.Module / 함수 / 클래스 / 기타 객체는 문자열로 강등 또는 제거
    """
    if _depth >= max_depth:
        return str(obj)

    if _is_primitive(obj):
        return obj

    # torch tensors -> list (원하면 제거로 바꿔도 됨)
    if torch.is_tensor(obj):
        try:
            return obj.detach().cpu().tolist()
        except Exception:
            return str(obj)

    # dict
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            # key는 문자열화
            ks = k if isinstance(k, str) else str(k)
            out[ks] = _sanitize(v, max_depth=max_depth, _depth=_depth + 1)
        return out

    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v, max_depth=max_depth, _depth=_depth + 1) for v in obj]

    # set
    if isinstance(obj, set):
        return [_sanitize(v, max_depth=max_depth, _depth=_depth + 1) for v in sorted(list(obj), key=str)]

    # nn.Module 등: 클래스명으로만
    if isinstance(obj, torch.nn.Module):
        return {"__type__": obj.__class__.__name__}

    # callable / function / class
    if callable(obj):
        name = getattr(obj, "__name__", obj.__class__.__name__)
        mod = getattr(obj, "__module__", "")
        return {"__callable__": f"{mod}.{name}".strip(".")}

    # fallback: 문자열로 강등
    return str(obj)


def _canonical_distribution_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = "".join(ch for ch in str(value).casefold() if ch.isalnum())
    aliases = {
        "normal": "Normal",
        "studentt": "StudentT",
    }
    return aliases.get(normalized)


def _distribution_loss_to_spec(value: Any) -> Optional[dict[str, Any]]:
    """Serialize the supported distribution criteria without pickling an nn.Module."""
    try:
        from modeling_module.training.model_losses.loss_module import DistributionLoss
    except Exception:
        return None

    if not isinstance(value, DistributionLoss):
        return None

    distribution = _canonical_distribution_name(getattr(value, "distribution", None))
    if distribution not in _DISTRIBUTION_CONTRACTS:
        return None

    distribution_kwargs = dict(getattr(value, "distribution_kwargs", {}) or {})
    if set(distribution_kwargs) - {"validate_args"}:
        return None

    out_mult, param_names = _DISTRIBUTION_CONTRACTS[distribution]
    quantiles = getattr(value, "quantiles", None)
    horizon_weight = getattr(value, "horizon_weight", None)
    return {
        "__type__": _DISTRIBUTION_LOSS_SPEC_TYPE,
        "version": _DISTRIBUTION_LOSS_SPEC_VERSION,
        "distribution": distribution,
        "quantiles": _sanitize(quantiles),
        "output_names": _sanitize(list(getattr(value, "output_names", []))),
        "num_samples": int(getattr(value, "num_samples", 1000)),
        "return_params": bool(getattr(value, "return_params", False)),
        "horizon_weight": _sanitize(horizon_weight),
        "distribution_kwargs": _sanitize(distribution_kwargs),
        "contract": {
            "out_mult": out_mult,
            "param_names": list(param_names),
        },
    }


def _distribution_loss_from_spec(spec: Mapping[str, Any]):
    if spec.get("__type__") != _DISTRIBUTION_LOSS_SPEC_TYPE:
        return None
    if int(spec.get("version", -1)) != _DISTRIBUTION_LOSS_SPEC_VERSION:
        raise ValueError(f"Unsupported DistributionLoss spec version: {spec.get('version')!r}")

    distribution = _canonical_distribution_name(spec.get("distribution"))
    if distribution not in _DISTRIBUTION_CONTRACTS:
        raise ValueError(f"Unsupported checkpoint distribution: {spec.get('distribution')!r}")

    expected_out_mult, expected_param_names = _DISTRIBUTION_CONTRACTS[distribution]
    contract = dict(spec.get("contract", {}) or {})
    if contract:
        stored_out_mult = int(contract.get("out_mult", expected_out_mult))
        stored_param_names = list(contract.get("param_names", expected_param_names))
        if stored_out_mult != expected_out_mult or stored_param_names != expected_param_names:
            raise ValueError(
                "DistributionLoss checkpoint contract does not match its distribution: "
                f"distribution={distribution!r}, out_mult={stored_out_mult}, "
                f"param_names={stored_param_names!r}"
            )

    quantiles = spec.get("quantiles")
    if not isinstance(quantiles, (list, tuple)) or not quantiles:
        raise ValueError("DistributionLoss checkpoint spec must contain non-empty quantiles.")
    try:
        quantile_tensor = torch.as_tensor(quantiles, dtype=torch.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("DistributionLoss checkpoint quantiles must be numeric.") from exc
    if quantile_tensor.ndim != 1 or not torch.isfinite(quantile_tensor).all():
        raise ValueError("DistributionLoss checkpoint quantiles must be a finite 1D sequence.")
    if not ((quantile_tensor >= 0.0) & (quantile_tensor <= 1.0)).all():
        raise ValueError("DistributionLoss checkpoint quantiles must be in [0, 1].")

    num_samples = int(spec.get("num_samples", 1000))
    if num_samples <= 0:
        raise ValueError("DistributionLoss checkpoint num_samples must be positive.")

    return_params = bool(spec.get("return_params", False))
    if "output_names" in spec:
        output_names = spec["output_names"]
        expected_name_count = 1 + len(quantiles) + (len(expected_param_names) if return_params else 0)
        if not isinstance(output_names, (list, tuple)) or len(output_names) != expected_name_count:
            raise ValueError(
                "DistributionLoss checkpoint output_names length does not match its quantile contract."
            )

    distribution_kwargs = dict(spec.get("distribution_kwargs", {}) or {})
    unknown_kwargs = set(distribution_kwargs) - {"validate_args"}
    if unknown_kwargs:
        raise ValueError(f"Unsupported DistributionLoss kwargs in checkpoint: {sorted(unknown_kwargs)!r}")
    validate_args = distribution_kwargs.get("validate_args")
    if validate_args is not None and not isinstance(validate_args, bool):
        raise ValueError("DistributionLoss validate_args must be bool or null.")

    horizon_weight = spec.get("horizon_weight")
    if horizon_weight is not None:
        try:
            horizon_weight = torch.as_tensor(horizon_weight, dtype=torch.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("DistributionLoss horizon_weight must be numeric.") from exc
        if horizon_weight.ndim != 1 or horizon_weight.numel() == 0 or not torch.isfinite(horizon_weight).all():
            raise ValueError("DistributionLoss horizon_weight must be a non-empty finite 1D sequence.")

    from modeling_module.training.model_losses.loss_module import DistributionLoss

    loss = DistributionLoss(
        distribution=distribution,
        quantiles=list(quantiles),
        num_samples=num_samples,
        return_params=return_params,
        horizon_weight=horizon_weight,
        **distribution_kwargs,
    )

    # Explicit quantile construction sorts values. Restore the original order and names
    # because level-based DistributionLoss instances may use a different order.
    loss.quantiles = torch.nn.Parameter(
        quantile_tensor.to(dtype=loss.quantiles.dtype),
        requires_grad=False,
    )
    if "output_names" in spec:
        loss.output_names = list(spec["output_names"])

    if loss.outputsize_multiplier != expected_out_mult or list(loss.param_names) != expected_param_names:
        raise ValueError(f"Rebuilt DistributionLoss contract mismatch for {distribution!r}.")
    return loss


def _drop_or_stringify_loss(cfg_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    cfg_state에 loss/criterion 같은 필드가 있으면 pickle 이슈 방지를 위해 문자열화.
    (원하면 완전히 drop 해도 됨)
    """
    for k in ("loss", "loss_fn", "criterion", "loss_point", "loss_quantile"):
        if k in cfg_state and cfg_state[k] is not None:
            v = cfg_state[k]
            loss_spec = _distribution_loss_to_spec(v)
            if loss_spec is not None:
                cfg_state[k] = loss_spec
            elif isinstance(v, str):
                cfg_state[k] = v
            # dict 형태로 들어온 경우도 있으니 방어
            elif isinstance(v, dict):
                if v.get("__type__") == _DISTRIBUTION_LOSS_SPEC_TYPE:
                    cfg_state[k] = v
                else:
                    cfg_state[k] = v.get("__type__", "loss")
            else:
                cfg_state[k] = getattr(v, "__class__", type(v)).__name__
    return cfg_state


def sanitize_for_storage(obj: Any) -> Any:
    return _sanitize(obj)


def _cfg_to_primitive_state(cfg: Any) -> tuple[dict[str, Any], Optional[str]]:
    if cfg is None:
        return {}, None

    if is_dataclass(cfg):
        raw = asdict(cfg)
        cfg_cls = type(cfg).__name__
    else:
        raw = dict(getattr(cfg, "__dict__", {}) or {})
        cfg_cls = type(cfg).__name__ if cfg is not None else None

    raw = _drop_or_stringify_loss(raw)
    return _sanitize(raw), cfg_cls


def _read_config_value(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _write_config_value(cfg: Any, key: str, value: Any) -> None:
    if isinstance(cfg, Mapping):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


def _config_has_key(cfg: Any, key: str) -> bool:
    if isinstance(cfg, Mapping):
        return key in cfg
    return hasattr(cfg, key)


def _build_output_spec(model: torch.nn.Module, cfg: Any) -> dict[str, Any]:
    loss = getattr(model, "loss", None)
    if loss is None:
        loss = _read_config_value(cfg, "loss")

    out_mult = getattr(model, "out_mult", None)
    if out_mult is None:
        out_mult = getattr(model, "out_mul", None)
    if out_mult is None:
        out_mult = _read_config_value(cfg, "out_mul", 1)
    out_mult = int(out_mult or 1)

    param_names = getattr(model, "param_names", None)
    if param_names is None:
        param_names = _read_config_value(cfg, "param_names")
    param_names = list(param_names) if param_names is not None else None

    is_quantile = bool(getattr(model, "is_quantile", False))
    distribution = _canonical_distribution_name(getattr(loss, "distribution", None))
    if distribution is None and param_names is not None:
        for candidate, (candidate_out_mult, candidate_params) in _DISTRIBUTION_CONTRACTS.items():
            if out_mult == candidate_out_mult and param_names == candidate_params:
                distribution = candidate
                break

    if is_quantile:
        mode = "quantile"
    elif distribution is not None:
        mode = "distribution"
    else:
        mode = "point"

    spec = {
        "mode": mode,
        "distribution": distribution,
        "out_mult": out_mult,
        "param_names": param_names,
    }
    loss_spec = _distribution_loss_to_spec(loss)
    if loss_spec is not None:
        spec["loss"] = loss_spec
    return _sanitize(spec)


def build_checkpoint_payload(
    model,
    cfg,
    *,
    extra_meta: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    cfg_state, cfg_cls = _cfg_to_primitive_state(cfg)

    meta = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "torch_version": torch.__version__,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    for attr in (
        "architecture_variant",
        "exogenous_fusion_strategy",
        "upstream_repository",
        "upstream_commit",
    ):
        value = getattr(model, attr, None)
        if value is not None:
            meta[attr] = _sanitize(value)
    if extra_meta:
        meta.update(_sanitize(dict(extra_meta)))

    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "config": cfg_state,
        "cfg_state": cfg_state,
        "cfg_cls": cfg_cls,
        "model_class": model.__class__.__name__,
        "output_spec": _build_output_spec(model, cfg),
        "state_dict": model.state_dict(),
        "meta": meta,
    }


def save_model(model, cfg, path: str, *, extra_meta: Optional[Mapping[str, Any]] = None):
    """
    안전한 단일 모델 저장.
    """
    ckpt = build_checkpoint_payload(model, cfg, extra_meta=extra_meta)

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    print(f"[save] model saved to: {path}")


def save_json_config(cfg, path: str):
    """
    config를 json으로 별도 저장(옵션)
    """
    data, _ = _cfg_to_primitive_state(cfg)
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[save] config saved to: {path}")


def summarize_training_results(results: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    for name, info in results.items():
        if not isinstance(info, Mapping):
            summary[str(name)] = _sanitize(info)
            continue

        item: dict[str, Any] = {}
        for key, value in info.items():
            if key in {"model", "cfg"}:
                continue

            if key.endswith("_path") and value is not None:
                item[key] = str(value)
                continue

            if _is_primitive(value):
                item[key] = value
                continue

            if torch.is_tensor(value):
                item[key] = _sanitize(value)
                continue

            item[key] = _sanitize(value)

        summary[str(name)] = item

    return summary


def save_training_manifest(
    save_dir: str | Path,
    *,
    request: Optional[Mapping[str, Any]] = None,
    results: Optional[Mapping[str, Any]] = None,
    extra_meta: Optional[Mapping[str, Any]] = None,
    filename: str = "training_manifest.json",
) -> str:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format_version": TRAINING_MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if request is not None:
        manifest["request"] = _sanitize(dict(request))
    if results is not None:
        manifest["results"] = summarize_training_results(results)
    if extra_meta is not None:
        manifest["meta"] = _sanitize(dict(extra_meta))

    path = save_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[save] training manifest saved to: {path}")
    return str(path)


# ------------------------------------------------------------------
# 2) 로딩 유틸 (호환 확장 버전)
# ------------------------------------------------------------------
def _canonical_model_key(name: str) -> str:
    """
    ckpt의 model_class / 파일명 / builders key를 최대한 같은 key로 정규화.
    """
    s = str(name).strip()
    if not s:
        return ""

    try:
        from modeling_module.models.registry import (
            TRAINING_FAMILY_DEFAULTS,
            resolve_artifact_model_key,
            resolve_training_request_key,
        )

        try:
            return resolve_artifact_model_key(s)
        except ValueError:
            request_key = resolve_training_request_key(s)
            if request_key in TRAINING_FAMILY_DEFAULTS:
                return TRAINING_FAMILY_DEFAULTS[request_key][0]
            return request_key
    except Exception:
        pass

    sl = s.lower()

    # 이미 builders가 snake_case로 들어오는 경우
    if sl in {
        "patchmixer_base", "patchmixer_exogenous", "patchmixer_original",
        "patchmixer_quantile", "patchmixer_quantile_exogenous", "patchmixer_dist",
        "titan_base", "titan_lmm", "titan_seq2seq",
        "patchtst_base", "patchtst_exogenous", "patchtst_quantile",
        "patchtst_quantile_exogenous", "exotst_base", "timexer_base",
    }:
        return sl

    # 클래스/별칭 정규화
    if "patchtst" in sl and "quant" in sl and ("exogenous" in sl or "exo" in sl):
        return "patchtst_quantile_exogenous"
    if "patchtst" in sl and ("exogenous" in sl or "exo" in sl):
        return "patchtst_exogenous"
    if "patchtst" in sl and "quant" in sl:
        return "patchtst_quantile"
    if "patchtst" in sl and ("base" in sl or "point" in sl):
        return "patchtst_base"
    if "patchtst" in sl and "dist" in sl:
        return "patchtst_dist"

    if "patchmixer" in sl and ("original" in sl or "canonical" in sl or "upstream" in sl):
        return "patchmixer_original"
    if "patchmixer" in sl and "quant" in sl and ("exogenous" in sl or "exo" in sl):
        return "patchmixer_quantile_exogenous"
    if "patchmixer" in sl and ("exogenous" in sl or "exo" in sl):
        return "patchmixer_exogenous"
    if "patchmixer" in sl and "quant" in sl:
        return "patchmixer_quantile"
    if "patchmixer" in sl:
        return 'patchmixer'

    if "titan" in sl and "lmm" in sl:
        return "titan_lmm"
    if "titan" in sl and "seq" in sl:
        return "titan_seq2seq"
    if "titan" in sl:
        return "titan_base"

    if 'exotst' in sl and ('base' in sl):
        return 'exotst_base'

    if "timexer" in sl:
        return "timexer_base"

    # fallback
    return sl


def _norm_search_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _candidate_ckpt_names(name_key: str) -> list[str]:
    candidates = [str(name_key)]

    try:
        from modeling_module.models.registry import get_model_spec

        spec = get_model_spec(name_key)
        candidates.extend(spec.checkpoint_aliases)
        candidates.extend(spec.class_names)
        candidates.extend(spec.aliases)
        candidates.append(spec.label)
    except Exception:
        pass

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def _find_ckpt_path(save_dir: str, name_key: str) -> Optional[str]:
    """
    기존: {name}.pt만 찾던 방식을 확장.
    1) save_dir/{name}.pt
    2) save_dir/**/*{name}*.pt (예: hourly_PatchTSTBase_L52_H27.pt)
    """
    candidate_names = _candidate_ckpt_names(name_key)
    for candidate in candidate_names:
        exact = os.path.join(save_dir, f"{candidate}.pt")
        if os.path.exists(exact):
            return exact

    # 패턴 탐색
    pats = [os.path.join(save_dir, f"*{candidate}*.pt") for candidate in candidate_names]
    pats.append(os.path.join(save_dir, "*.pt"))
    cand = []
    for p in pats:
        cand.extend(glob.glob(p))

    if not cand:
        return None

    # 가장 그럴듯한 후보를 우선: name_key 포함 + 짧은 파일명 우선
    needles = {_norm_search_name(candidate) for candidate in candidate_names}

    def score(path: str):
        base = os.path.basename(path)
        base_norm = _norm_search_name(base)
        contains = any(needle and needle in base_norm for needle in needles)
        return (0 if contains else 1, len(base))

    cand = sorted(set(cand), key=score)
    return cand[0]


def _checkpoint_identity(ckpt: Mapping[str, Any]) -> str:
    meta = ckpt.get("meta", {})
    if isinstance(meta, Mapping) and meta.get("model_key"):
        return str(meta["model_key"]).casefold()
    return str(ckpt.get("model_class", "")).casefold()


def _checkpoint_state_dict_or_empty(ckpt: Mapping[str, Any]) -> Mapping[str, Any]:
    state = ckpt.get("model_state")
    if not isinstance(state, Mapping):
        state = ckpt.get("state_dict")
    return state if isinstance(state, Mapping) else {}


def _infer_legacy_out_mult(
    ckpt: Mapping[str, Any],
    cfg_state: Mapping[str, Any],
) -> Optional[int]:
    identity = _checkpoint_identity(ckpt)
    if "quantile" in identity:
        return None

    state = _checkpoint_state_dict_or_empty(ckpt)
    horizon = int(cfg_state.get("horizon", 0) or 0)

    if "patchtst" in identity and "head.net.2.weight" in state and horizon > 0:
        return int(state["head.net.2.weight"].shape[0]) // horizon
    if "patchmixer" in identity and "head.2.weight" in state:
        return int(state["head.2.weight"].shape[0])
    if "titan" in identity and "head.weight" in state:
        return int(state["head.weight"].shape[0])
    if "exotst" in identity and "head.fc.weight" in state and horizon > 0:
        return int(state["head.fc.weight"].shape[0]) // horizon

    configured = cfg_state.get("out_mul")
    return int(configured) if configured is not None else None


def _normalize_output_spec(raw_spec: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(raw_spec.get("mode", "point")).casefold()
    if mode == "dist":
        mode = "distribution"
    if mode not in {"point", "quantile", "distribution"}:
        raise ValueError(f"Unknown checkpoint output mode: {raw_spec.get('mode')!r}")

    out_mult = int(raw_spec.get("out_mult", 1) or 1)
    if out_mult <= 0:
        raise ValueError(f"Checkpoint out_mult must be positive, got {out_mult}.")
    param_names = raw_spec.get("param_names")
    param_names = list(param_names) if param_names is not None else None
    distribution = _canonical_distribution_name(raw_spec.get("distribution"))

    if mode == "distribution":
        if distribution is None:
            for candidate, (candidate_out_mult, candidate_params) in _DISTRIBUTION_CONTRACTS.items():
                if out_mult == candidate_out_mult and (
                    param_names is None or param_names == candidate_params
                ):
                    distribution = candidate
                    break
        if distribution not in _DISTRIBUTION_CONTRACTS:
            raise ValueError(f"Unknown distribution output contract: {dict(raw_spec)!r}")

        expected_out_mult, expected_param_names = _DISTRIBUTION_CONTRACTS[distribution]
        if param_names is None:
            param_names = list(expected_param_names)
        if out_mult != expected_out_mult or param_names != expected_param_names:
            raise ValueError(
                "Checkpoint output contract is inconsistent: "
                f"distribution={distribution!r}, out_mult={out_mult}, param_names={param_names!r}"
            )

    normalized = {
        "mode": mode,
        "distribution": distribution,
        "out_mult": out_mult,
        "param_names": param_names,
    }
    if isinstance(raw_spec.get("loss"), Mapping):
        normalized["loss"] = dict(raw_spec["loss"])
    return normalized


def _legacy_output_spec(
    ckpt: Mapping[str, Any],
    cfg_state: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    identity = _checkpoint_identity(ckpt)
    if "quantile" in identity or not any(
        family in identity for family in ("patchtst", "patchmixer", "titan", "exotst")
    ):
        return None

    configured_loss = cfg_state.get("loss")
    configured_dist_name = _canonical_distribution_name(cfg_state.get("dist_name"))
    configured_params = cfg_state.get("param_names")
    configured_params = list(configured_params) if configured_params is not None else None
    configured_loss_name = str(configured_loss).casefold()
    out_mult = _infer_legacy_out_mult(ckpt, cfg_state)

    if isinstance(configured_loss, Mapping):
        inferred_loss = _distribution_loss_from_spec(configured_loss)
        configured_distribution = _canonical_distribution_name(getattr(inferred_loss, "distribution", None))
    else:
        configured_distribution = configured_dist_name if configured_loss_name == "distributionloss" else None

    if configured_distribution is None and configured_params is not None:
        for candidate, (candidate_out_mult, candidate_param_names) in _DISTRIBUTION_CONTRACTS.items():
            if configured_params == candidate_param_names:
                configured_distribution = candidate
                break

    if configured_distribution is None and configured_loss_name != "distributionloss":
        if out_mult in (2, 3):
            raise ValueError(
                "Legacy checkpoint has a distribution-shaped head but no persisted "
                "distribution metadata; refusing an ambiguous restore."
            )
        return None

    if out_mult not in (2, 3):
        raise ValueError(
            "Legacy checkpoint declares DistributionLoss but its saved head does not "
            f"encode a supported distribution contract: inferred_out_mult={out_mult!r}."
        )

    distribution = configured_distribution or ("Normal" if out_mult == 2 else "StudentT")
    expected_out_mult, expected_param_names = _DISTRIBUTION_CONTRACTS[distribution]
    if out_mult != expected_out_mult:
        raise ValueError(
            "Legacy checkpoint distribution metadata conflicts with its saved head shape: "
            f"distribution={distribution!r}, inferred_out_mult={out_mult}"
        )
    if configured_params is not None and configured_params != expected_param_names:
        raise ValueError(
            "Legacy checkpoint has conflicting distribution metadata: "
            f"out_mult={out_mult}, param_names={configured_params!r}"
        )

    state = _checkpoint_state_dict_or_empty(ckpt)
    saved_quantiles = state.get("loss.quantiles")
    if torch.is_tensor(saved_quantiles):
        quantiles = saved_quantiles.detach().cpu().tolist()
    else:
        configured_quantiles = cfg_state.get("quantiles", (0.1, 0.5, 0.9))
        quantiles = list(configured_quantiles)

    warnings.warn(
        "Restoring a legacy distribution checkpoint by its saved head shape. "
        "Re-save the artifact to persist the exact v3 output contract.",
        RuntimeWarning,
        stacklevel=3,
    )
    return {
        "mode": "distribution",
        "distribution": distribution,
        "out_mult": expected_out_mult,
        "param_names": list(expected_param_names),
        "loss": {
            "__type__": _DISTRIBUTION_LOSS_SPEC_TYPE,
            "version": _DISTRIBUTION_LOSS_SPEC_VERSION,
            "distribution": distribution,
            "quantiles": quantiles,
            "num_samples": 1000,
            "return_params": False,
            "horizon_weight": None,
            "distribution_kwargs": {},
            "contract": {
                "out_mult": expected_out_mult,
                "param_names": list(expected_param_names),
            },
        },
    }


def _prepare_config_for_restore(ckpt: Mapping[str, Any], cfg: Any) -> Any:
    cfg_state = dict(cfg) if isinstance(cfg, Mapping) else cfg
    cfg_view = cfg_state if isinstance(cfg_state, Mapping) else dict(vars(cfg_state))
    raw_output_spec = ckpt.get("output_spec")
    if isinstance(raw_output_spec, Mapping):
        output_spec = _normalize_output_spec(raw_output_spec)
    else:
        output_spec = _legacy_output_spec(ckpt, cfg_view)

    restored_loss = None
    configured_loss = _read_config_value(cfg_state, "loss")
    if isinstance(configured_loss, Mapping):
        restored_loss = _distribution_loss_from_spec(configured_loss)

    if output_spec is None:
        if restored_loss is not None:
            _write_config_value(cfg_state, "loss", restored_loss)
        return cfg_state

    output_loss_spec = output_spec.get("loss")
    if isinstance(output_loss_spec, Mapping):
        restored_loss = _distribution_loss_from_spec(output_loss_spec)

    if output_spec["mode"] == "distribution":
        if restored_loss is None:
            raise ValueError("Distribution checkpoint is missing a restorable loss specification.")

        distribution = output_spec["distribution"]
        _write_config_value(cfg_state, "loss", restored_loss)
        _write_config_value(cfg_state, "loss_mode", "dist")
        _write_config_value(cfg_state, "out_mul", output_spec["out_mult"])
        _write_config_value(cfg_state, "param_names", list(output_spec["param_names"]))
        _write_config_value(cfg_state, "dist_name", "studentt" if distribution == "StudentT" else "normal")
        if "exotst" in _checkpoint_identity(ckpt) or _config_has_key(cfg_state, "head_type"):
            _write_config_value(cfg_state, "head_type", "dist")
    elif restored_loss is not None:
        _write_config_value(cfg_state, "loss", restored_loss)

    return cfg_state


def _extract_cfg_obj(ckpt: dict) -> Any:
    """
    ckpt에서 config 객체/딕트를 추출한다.
    우선순위:
      1) 회사식 신규: "config"
      2) 본 파일 save_model: "cfg"
      3) 구버전: "config"
      4) cfg_state/cfg_cls로 rebuild
    """
    # 회사식 신규 포맷 / v2+ 포맷
    if "config" in ckpt:
        return _prepare_config_for_restore(ckpt, ckpt["config"])

    # save_model 포맷
    if "cfg" in ckpt:
        return _prepare_config_for_restore(ckpt, ckpt["cfg"])

    # 마지막: cfg_state + cfg_cls로 rebuild
    cfg_state = ckpt.get("cfg_state", None)
    cfg_cls = ckpt.get("cfg_cls", None)
    if cfg_state is not None and cfg_cls is not None:
        prepared_cfg_state = _prepare_config_for_restore(ckpt, cfg_state)
        if cfg_cls == "PatchMixerConfig":
            try:
                from modeling_module.models.registry import infer_artifact_model_key_from_checkpoint

                model_key = infer_artifact_model_key_from_checkpoint(ckpt)
            except ValueError:
                model_key = None
            if model_key == "patchmixer":
                return _rebuild_patchmixer_original(prepared_cfg_state)
            return _rebuild_patchmixer_exogenous(prepared_cfg_state)
        rb = _REBUILDERS_BY_CLS.get(cfg_cls, None)
        if rb is not None:
            return rb(prepared_cfg_state)
        # re-builder가 없으면 dict로라도 반환
        return prepared_cfg_state

    return None


def _extract_state_dict(ckpt: dict) -> dict:
    """
    ckpt에서 state_dict 추출 (새/구 포맷 모두).
    우선순위:
      1) 회사식 신규: "model_state"
      2) save_model: "state_dict"
      3) 구버전: "state_dict" / "model_state"
    """
    if "model_state" in ckpt:
        return ckpt["model_state"]
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    # fallback
    raise ValueError(f"[load_model_dict] No model_state/state_dict in ckpt. keys={list(ckpt.keys())}")


def _drop_revin_buffers(sd: dict) -> dict:
    # RevIN 통계/버퍼는 mismatch 가능성이 높고 추론 시 재계산되므로 drop 권장
    for k in ["revin_layer.mean", "revin_layer.std"]:
        if k in sd:
            sd.pop(k, None)
    return sd


def _partial_load_with_shape_filter(model: torch.nn.Module, sd: dict):
    own = model.state_dict()
    filtered = {}
    skipped = []

    for k, v in sd.items():
        if k not in own:
            continue
        if hasattr(v, "shape") and hasattr(own[k], "shape") and tuple(v.shape) != tuple(own[k].shape):
            skipped.append(f"{k} (ckpt {tuple(v.shape)} vs model {tuple(own[k].shape)})")
            continue
        filtered[k] = v

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return missing, unexpected, skipped


def load_model_dict(
    save_dir: str,
    builders: Dict[str, Callable],
    device: str = "cpu",
    strict: bool = False,
    *,
    prefer_ckpt_model_class: bool = True,
    drop_revin_stats: bool = True,
    allow_partial_load: bool = True,
):
    """
    호환 로더.

    - save_dir: ckpt 디렉터리
    - builders: {"patchtst_base": build_patchTST_base, ...}
      build_fn(cfg_or_dict) -> nn.Module 이어야 함.

    지원 ckpt 포맷:
      A) 회사식 신규:
         {"model_state": ..., "model_class": "...", "config": ...}
      B) save_model 신규:
         {"state_dict": ..., "cfg": ..., "cfg_state": ..., "cfg_cls": ...}
      C) 구버전 혼합:
         {"model_state": ..., "config": ...} 등
    """
    models = {}

    for builder_key, build_fn in builders.items():
        print(f"[load_model_dict] Building {builder_key}")
        canonical_key = _canonical_model_key(builder_key)
        print(f"[load_model_dict] Canonical {canonical_key}")
        path = _find_ckpt_path(save_dir, canonical_key)

        if path is None or (not os.path.exists(path)):
            print(f"[warn] checkpoint not found for '{builder_key}' (canonical='{canonical_key}') in: {save_dir}")
            continue

        print(f"[load] {builder_key} ← {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # 1) 어떤 builder를 쓸지 결정 (회사식 ckpt는 model_class가 있을 수 있음)
        ckpt_model_class = ckpt.get("model_class", None)
        ckpt_key = _canonical_model_key(ckpt_model_class) if ckpt_model_class else None

        # prefer_ckpt_model_class=True이면 ckpt의 model_class를 우선 사용
        selected_key = canonical_key
        if prefer_ckpt_model_class and ckpt_key and (ckpt_key in builders):
            selected_key = ckpt_key
            build_fn = builders[ckpt_key]

        # 2) config 추출
        cfg_obj = _extract_cfg_obj(ckpt)
        if cfg_obj is None:
            raise ValueError(
                f"[load_model_dict] No config info found in {path}. keys={list(ckpt.keys())}"
            )

        # 3) 모델 build
        print(f'[checkpoint] cfg_obj={cfg_obj}')
        model = build_fn(cfg_obj)
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"builder for '{selected_key}' must return nn.Module, got {type(model)}. build_fn={build_fn}"
            )

        # 4) state_dict 추출
        sd = _extract_state_dict(ckpt)
        sd = dict(sd)  # 안전 복사

        # 5) PatchTST RevIN 통계 drop(옵션)
        if drop_revin_stats and "patchtst" in selected_key:
            sd = _drop_revin_buffers(sd)

        # 6) 로드
        try:
            missing, unexpected = model.load_state_dict(sd, strict=strict)
            if missing or unexpected:
                print(f"[load][{selected_key}] missing={len(missing)} unexpected={len(unexpected)}")
                print("  missing sample:", list(missing)[:5])
                print("  unexpected sample:", list(unexpected)[:5])
        except RuntimeError as e:
            if not allow_partial_load:
                raise

            print(f"[info][{selected_key}] strict load failed -> partial load with shape filter")
            # shape mismatch 제거 후 partial 로드
            missing, unexpected, skipped = _partial_load_with_shape_filter(model, sd)
            print(f"[info][{selected_key}] partial load done | missing={len(missing)} unexpected={len(unexpected)}")
            if skipped:
                print(f"[warn][{selected_key}] skipped shape-mismatch keys (sample):")
                for sk in skipped[:10]:
                    print("  -", sk)

        model.to(device).eval()
        models[selected_key] = model

    return models

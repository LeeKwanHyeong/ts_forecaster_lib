# model_io.py
import os
import json
import glob
from typing import Dict, Callable, Optional, Any

import torch
from dataclasses import asdict, is_dataclass

from modeling_module.models.PatchMixer.common.configs import (
    PatchMixerConfigMonthly,
    PatchMixerConfigWeekly,
)
from modeling_module.models.PatchTST.common.configs import (
    PatchTSTConfigMonthly,
    PatchTSTConfig,
    HeadConfig,
    AttentionConfig,
)
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


def _rebuild_titan(cfgd: dict):
    return TitanConfig(**cfgd)


_REBUILDERS_BY_CLS = {
    # PatchTST
    "PatchTSTConfig": _rebuild_patchtst,
    "PatchTSTConfigMonthly": lambda d: PatchTSTConfigMonthly(**d),
    # PatchMixer
    "PatchMixerConfigMonthly": _rebuild_patchmixer_monthly,
    "PatchMixerConfigWeekly": _rebuild_patchmixer_weekly,
    # Titan
    "TitanConfig": _rebuild_titan,
}


# ------------------------------------------------------------------
# 1) 저장 유틸
#    (참고) 현재 load는 다음 ckpt 형태 모두 호환:
#      - 신규(회사식): {"model_state","model_class","config"}
#      - 신규(본 파일 save_model): {"state_dict","cfg","cfg_state","cfg_cls"}
#      - 구버전: {"model_state","config"} 또는 {"state_dict","cfg_state","cfg_cls"} 등
# ------------------------------------------------------------------
# def save_model(model, cfg, path: str):
#     """
#     단일 모델 저장.
#
#     ckpt 포맷:
#       {
#         "cfg":        cfg 객체 (pickle),
#         "cfg_state":  dict(asdict(cfg)) or cfg.__dict__,
#         "cfg_cls":    cfg 클래스 이름(str),
#         "state_dict": model.state_dict(),
#       }
#     """
#     if is_dataclass(cfg):
#         cfg_state = asdict(cfg)
#     else:
#         cfg_state = getattr(cfg, "__dict__", None)
#
#     ckpt = {
#         "cfg": cfg,
#         "cfg_state": cfg_state,
#         "cfg_cls": type(cfg).__name__,
#         "state_dict": model.state_dict(),
#     }
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     torch.save(ckpt, path)
#     print(f"[save] model saved to: {path}")


# -----------------------------
# helpers: primitive sanitize
# -----------------------------
_PRIMITIVE_TYPES = (str, int, float, bool, type(None))

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


def _drop_or_stringify_loss(cfg_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    cfg_state에 loss/criterion 같은 필드가 있으면 pickle 이슈 방지를 위해 문자열화.
    (원하면 완전히 drop 해도 됨)
    """
    for k in ("loss", "loss_fn", "criterion", "loss_point", "loss_quantile"):
        if k in cfg_state and cfg_state[k] is not None:
            v = cfg_state[k]
            # dict 형태로 들어온 경우도 있으니 방어
            if isinstance(v, dict):
                cfg_state[k] = v.get("__type__", "loss")
            else:
                cfg_state[k] = getattr(v, "__class__", type(v)).__name__
    return cfg_state


def save_model(model, cfg, path: str):
    """
    안전한 단일 모델 저장.
    - cfg 객체를 그대로 저장하지 않음 (pickle 차단)
    - cfg_state는 primitive-only로 sanitize
    """
    if is_dataclass(cfg):
        raw = asdict(cfg)
    else:
        raw = dict(getattr(cfg, "__dict__", {}) or {})

    # 1) loss 등 pickle 위험 필드 문자열화
    raw = _drop_or_stringify_loss(raw)

    # 2) 전체 sanitize (dict/list 재귀)
    cfg_state = _sanitize(raw)

    ckpt = {
        "cfg_state": cfg_state,                 # primitive-only
        "cfg_cls": type(cfg).__name__,          # class name only
        "model_class": model.__class__.__name__,# optional but useful
        "state_dict": model.state_dict(),       # weights only
        "meta": {
            "torch_version": torch.__version__,
        }
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)
    print(f"[save] model saved to: {path}")


def save_json_config(cfg, path: str):
    """
    config를 json으로 별도 저장(옵션)
    """
    if is_dataclass(cfg):
        data = asdict(cfg)
    else:
        data = getattr(cfg, "__dict__", None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[save] config saved to: {path}")


# ------------------------------------------------------------------
# 2) 로딩 유틸 (호환 확장 버전)
# ------------------------------------------------------------------
def _canonical_model_key(name: str) -> str:
    """
    ckpt의 model_class / 파일명 / builders key를 최대한 같은 key로 정규화.
    """
    s = str(name).strip()
    sl = s.lower()

    # 이미 builders가 snake_case로 들어오는 경우
    if sl in {
        "patchmixer_base", "patchmixer_quantile",
        "titan_base", "titan_lmm", "titan_seq2seq",
        "patchtst_base", "patchtst_quantile",
    }:
        return sl

    # 클래스/별칭 정규화
    if "patchtst" in sl and "quant" in sl:
        return "patchtst_quantile"
    if "patchtst" in sl and ("base" in sl or "point" in sl):
        return "patchtst_base"
    if "patchmixer" in sl and "quant" in sl:
        return "patchmixer_quantile"
    if "patchmixer" in sl:
        return "patchmixer_base"
    if "titan" in sl and "lmm" in sl:
        return "titan_lmm"
    if "titan" in sl and "seq" in sl:
        return "titan_seq2seq"
    if "titan" in sl:
        return "titan_base"

    # fallback
    return sl


def _find_ckpt_path(save_dir: str, name_key: str) -> Optional[str]:
    """
    기존: {name}.pt만 찾던 방식을 확장.
    1) save_dir/{name}.pt
    2) save_dir/**/*{name}*.pt (예: hourly_PatchTSTBase_L52_H27.pt)
    """
    exact = os.path.join(save_dir, f"{name_key}.pt")
    if os.path.exists(exact):
        return exact

    # 패턴 탐색
    pats = [
        os.path.join(save_dir, f"*{name_key}*.pt"),
        os.path.join(save_dir, f"*{name_key.replace('_', '')}*.pt"),
        os.path.join(save_dir, "*.pt"),
    ]
    cand = []
    for p in pats:
        cand.extend(glob.glob(p))

    if not cand:
        return None

    # 가장 그럴듯한 후보를 우선: name_key 포함 + 짧은 파일명 우선
    def score(path: str):
        base = os.path.basename(path).lower()
        contains = (name_key in base) or (name_key.replace("_", "") in base)
        return (0 if contains else 1, len(base))

    cand = sorted(set(cand), key=score)
    return cand[0]


def _extract_cfg_obj(ckpt: dict) -> Any:
    """
    ckpt에서 config 객체/딕트를 추출한다.
    우선순위:
      1) 회사식 신규: "config"
      2) 본 파일 save_model: "cfg"
      3) 구버전: "config"
      4) cfg_state/cfg_cls로 rebuild
    """
    # 회사식 신규 포맷
    if "config" in ckpt:
        return ckpt["config"]

    # save_model 포맷
    if "cfg" in ckpt:
        return ckpt["cfg"]

    # 마지막: cfg_state + cfg_cls로 rebuild
    cfg_state = ckpt.get("cfg_state", None)
    cfg_cls = ckpt.get("cfg_cls", None)
    if cfg_state is not None and cfg_cls is not None:
        rb = _REBUILDERS_BY_CLS.get(cfg_cls, None)
        if rb is not None:
            return rb(cfg_state)
        # re-builder가 없으면 dict로라도 반환
        return cfg_state

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
        canonical_key = _canonical_model_key(builder_key)
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

from dataclasses import fields, is_dataclass, asdict
from typing import Union, Any, Optional, Mapping

from modeling_module.models.ExoTST.configs import ExoTSTConfig
from modeling_module.models.PatchMixer.PatchMixer import PatchMixerModel, PatchMixerQuantileModel
from modeling_module.models.PatchMixer.common.configs import PatchMixerConfig
from modeling_module.models.PatchTST.common.configs import PatchTSTConfig
from modeling_module.models.Titan import TitanBaseModel, TitanLMMModel, TitanSeq2SeqModel
from modeling_module.models.Titan.common.configs import TitanConfig


# -----------------------------
# PatchMixer: dict → PatchMixerConfig
# -----------------------------
def _ensure_patchmixer_config(cfg: Any):
    """
    load_model_dict에서 cfg가 dict로 들어오는 케이스를 흡수.
    - dict -> PatchMixerConfig(**dict)
    - dataclass -> 그대로
    """
    # 프로젝트 경로에 맞춰 import 경로만 유지해 주세요.
    from modeling_module.models.PatchMixer.common.configs import PatchMixerConfig

    if isinstance(cfg, PatchMixerConfig):
        return cfg
    if is_dataclass(cfg):
        # PatchMixerConfig가 dataclass라면 여기로 들어올 수 있음
        return cfg
    if isinstance(cfg, dict):
        return PatchMixerConfig(**cfg)

    raise TypeError(f"[build_patch_mixer] unsupported cfg type: {type(cfg)}")

# def build_patch_mixer(cfg: PatchMixerConfig, *, out_mult: int = 1, param_names = None):
#     cfg = _ensure_patchmixer_config(cfg)
#     return PatchMixerModel(cfg, out_mult=out_mult, param_names=param_names)
def build_patch_mixer(cfg: PatchMixerConfig):
    cfg = _ensure_patchmixer_config(cfg)
    return PatchMixerModel(cfg)

def build_patch_mixer_quantile(cfg):
    """
    기존 quantile builder는 그대로 유지 (QuantileModel이 별도 class인 구조)
    """
    from modeling_module.models.PatchMixer.PatchMixer import PatchMixerQuantileModel
    cfg = _ensure_patchmixer_config(cfg)
    return PatchMixerQuantileModel(cfg)


# -----------------------------
# Titan: dict/Namespace → TitanConfig
# -----------------------------
def _ensure_titan_config(cfg: Union[TitanConfig, dict, Any]) -> TitanConfig:
    """
    Titan 설정 객체 변환 및 필드 유효성 검증.

    기능:
    - Dict 또는 Namespace 입력을 TitanConfig로 변환.
    - TitanConfig 정의에 없는 불필요한 키(Legacy params) 필터링.
    - 구버전 파라미터(Alias) 매핑 처리.
    """
    if isinstance(cfg, TitanConfig):
        return cfg

    if isinstance(cfg, dict):
        d = dict(cfg)
    elif hasattr(cfg, "__dict__"):
        d = dict(cfg.__dict__)
    else:
        raise TypeError(f"Unsupported cfg type: {type(cfg)}")

    allowed = {f.name for f in fields(TitanConfig)}
    d = {k: v for k, v in d.items() if k in allowed}

    return TitanConfig(**d)


def build_titan_base(cfg: TitanConfig, *, out_mult: int = 1, param_names=None):
    cfg = _ensure_titan_config(cfg)
    return TitanBaseModel(cfg, out_mult=out_mult, param_names=param_names)


def build_titan_lmm(cfg: TitanConfig, *, out_mult: int = 1, param_names=None):
    cfg = _ensure_titan_config(cfg)
    return TitanLMMModel(cfg, out_mult=out_mult, param_names=param_names)


def build_titan_seq2seq(cfg: TitanConfig, *, out_mult: int = 1, param_names=None):
    cfg = _ensure_titan_config(cfg)
    return TitanSeq2SeqModel(cfg, out_mult=out_mult, param_names=param_names)



# -----------------------------
# PatchTST: dict/Namespace → PatchTSTConfig
# -----------------------------
def _ensure_patchtst_config(cfg: Union[PatchTSTConfig, dict, Any]) -> PatchTSTConfig:
    """
    PatchTST 설정 객체 변환 및 중첩 구조 처리.

    기능:
    - Dict/Namespace를 PatchTSTConfig로 변환.
    - 내부의 중첩된 설정(Attn, Head, Decomp)이 dict일 경우 해당 Config 객체로 재귀적 변환.
    """
    if isinstance(cfg, PatchTSTConfig):
        return cfg

    if isinstance(cfg, dict):
        cfgd = dict(cfg)
        try:
            from modeling_module.models.PatchTST.common.configs import (
                AttentionConfig,
                HeadConfig,
                DecompositionConfig,
            )

            if "attn" in cfgd and isinstance(cfgd["attn"], dict):
                cfgd["attn"] = AttentionConfig(**cfgd["attn"])
            if "head" in cfgd and isinstance(cfgd["head"], dict):
                cfgd["head"] = HeadConfig(**cfgd["head"])
            if "decomp" in cfgd and isinstance(cfgd["decomp"], dict):
                cfgd["decomp"] = DecompositionConfig(**cfgd["decomp"])
        except Exception:
            pass

        return PatchTSTConfig(**cfgd)

    if hasattr(cfg, "__dict__"):
        return PatchTSTConfig(**cfg.__dict__)

    raise TypeError(f"Unsupported config type for PatchTST: {type(cfg)}")


def build_patchTST(cfg):
    """PatchTST 점 예측(Point) 모델 인스턴스 생성."""
    cfg = _ensure_patchtst_config(cfg)
    from modeling_module.models.PatchTST.supervised.PatchTST import PatchTSTModel
    return PatchTSTModel.from_config(cfg)


def build_patchTST_quantile(cfg):
    """PatchTST 분위수 예측(Quantile) 모델 인스턴스 생성."""
    from modeling_module.models.PatchTST.supervised.PatchTST import PatchTSTQuantileModel
    cfg = _ensure_patchtst_config(cfg)
    return PatchTSTQuantileModel.from_config(cfg)

def _ensure_exotst_config(cfg: Union[ExoTSTConfig, dict, Any]) -> ExoTSTConfig:
    """
    Normalize various config inputs into ExoTSTConfig.

    Supported inputs:
      - ExoTSTConfig: returned as-is
      - dict / Mapping: ExoTSTConfig(**dict) with key normalization
      - dataclass / Any object: try asdict(), then vars()

    Also:
      - handles alias keys (e.g., "seq_len" -> "lookback", "pred_len" -> "horizon")
      - normalizes head_type/loss_mode conventions
      - basic validation for critical fields
    """
    if isinstance(cfg, ExoTSTConfig):
        out = cfg
    else:
        # 1) dict-like
        if isinstance(cfg, Mapping):
            d = dict(cfg)
        # 2) dataclass-like
        elif is_dataclass(cfg):
            d = asdict(cfg)
        # 3) generic object (TrainingConfig 등)
        else:
            try:
                d = dict(vars(cfg))
            except Exception as e:
                raise TypeError(f"Unsupported cfg type for ExoTSTConfig: {type(cfg)}") from e

        # -------------------------
        # Key normalization (aliases)
        # -------------------------
        # common time-series naming
        alias_map = {
            "seq_len": "lookback",
            "context_length": "lookback",
            "input_len": "lookback",
            "look_back": "lookback",
            "pred_len": "horizon",
            "prediction_length": "horizon",
            "output_len": "horizon",
            "target_dim": "y_dim",
            "c_in": "y_dim",
            "d_model": "d_model",
            "n_head": "n_heads",
            "nhead": "n_heads",
            "ff_dim": "d_ff",
            "dropout_rate": "dropout",
        }
        for k, v in list(d.items()):
            if k in alias_map and alias_map[k] not in d:
                d[alias_map[k]] = v

        # -------------------------
        # head_type / loss_mode normalization
        # -------------------------
        # build_exotst(... out_mult/param_names ...) 호출부에서 head_type을 결정하려는 경우가 많아서,
        # cfg에 head_type이 없고 loss_mode만 있는 경우 head_type으로 매핑해줌.
        loss_mode = str(d.get("loss_mode", "")).lower() if d.get("loss_mode") is not None else ""
        if "head_type" not in d and loss_mode:
            if loss_mode in ("dist", "distribution"):
                d["head_type"] = "dist"
            elif loss_mode in ("point", "mse", "mae", "huber"):
                d["head_type"] = "point"
            # quantile은 ExoTST에서 아직 head가 없으면 builder 단계에서 막는 편이 안전
            elif loss_mode in ("quantile", "mq"):
                d["head_type"] = "quantile"

        # 기본값 강제(프로젝트에서 실수 잦은 부분)
        d.setdefault("strict_shape", True)
        d.setdefault("exo_nan_policy", "zero+indicator")

        # 실제 Config로 변환
        try:
            out = ExoTSTConfig(**d)
        except TypeError as e:
            # 어떤 키가 문제인지 디버깅하기 쉽게 메시지 보강
            allowed = set(ExoTSTConfig.__annotations__.keys())
            extra = sorted(set(d.keys()) - allowed)
            raise TypeError(
                f"Failed to build ExoTSTConfig from input. "
                f"Extra keys not in ExoTSTConfig: {extra}"
            ) from e

    # -------------------------
    # Basic validation (fail-fast)
    # -------------------------
    if int(out.lookback) <= 0:
        raise ValueError(f"lookback must be > 0, got {out.lookback}")
    if int(out.horizon) <= 0:
        raise ValueError(f"horizon must be > 0, got {out.horizon}")
    if int(out.patch_len) <= 0:
        raise ValueError(f"patch_len must be > 0, got {out.patch_len}")
    if int(out.stride) <= 0:
        raise ValueError(f"stride must be > 0, got {out.stride}")
    if int(out.d_model) <= 0:
        raise ValueError(f"d_model must be > 0, got {out.d_model}")
    if int(out.n_heads) <= 0:
        raise ValueError(f"n_heads must be > 0, got {out.n_heads}")

    if out.exo_nan_policy not in ("zero", "zero+indicator"):
        raise ValueError(f"exo_nan_policy must be 'zero' or 'zero+indicator', got {out.exo_nan_policy}")

    if getattr(out, "exo_memory_mode", "all") not in ("all", "agg"):
        raise ValueError(f"exo_memory_mode must be 'all' or 'agg', got {getattr(out, 'exo_memory_mode', None)}")

    head_type = getattr(out, "head_type", "point")
    if head_type not in ("point", "dist", "quantile"):
        raise ValueError(f"head_type must be one of ('point','dist','quantile'), got {head_type}")

    return out

def build_exotst(cfg):
    """ExoTST 점 예측(Point) 모델 인스턴스 생성."""
    cfg = _ensure_exotst_config(cfg)
    from modeling_module.models.ExoTST.ExoTST import ExoTST
    return ExoTST.from_config(cfg)
from dataclasses import fields, is_dataclass
from typing import Union, Any, Optional

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


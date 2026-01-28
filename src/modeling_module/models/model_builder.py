from dataclasses import fields
from typing import Union, Any, Optional

from modeling_module.models.PatchMixer.common.configs import PatchMixerConfig
from modeling_module.models.PatchTST.common.configs import PatchTSTConfig
from modeling_module.models.Titan.common.configs import TitanConfig


# -----------------------------
# PatchMixer: dict → PatchMixerConfig
# -----------------------------
def _ensure_patchmixer_config(cfg: Union[PatchMixerConfig, dict, Any]) -> PatchMixerConfig:
    """
    입력 설정을 PatchMixerConfig 객체로 변환 및 타입 보장.

    기능:
    - 구(旧) 버전 체크포인트(dict 형태) 로드 시 호환성 지원.
    - Dict, Namespace 등 다양한 입력 형식을 표준 Config 객체로 래핑.
    """
    if isinstance(cfg, PatchMixerConfig):
        return cfg
    if isinstance(cfg, dict):
        return PatchMixerConfig(**cfg)
    # 예: argparse.Namespace 등일 경우
    if hasattr(cfg, "__dict__"):
        return PatchMixerConfig(**cfg.__dict__)
    raise TypeError(f"Unsupported config type for PatchMixer: {type(cfg)}")


def build_patch_mixer_base(cfg):
    """PatchMixer 점 예측(Base/Point) 모델 인스턴스 생성."""
    from modeling_module.models.PatchMixer.PatchMixer import BaseModel
    cfg = _ensure_patchmixer_config(cfg)
    return BaseModel(cfg)

def build_patch_mixer_dist(cfg):
    from modeling_module.models.PatchMixer.PatchMixer import DistModel
    cfg = _ensure_patchmixer_config(cfg)
    return DistModel(cfg)


def build_patch_mixer_quantile(cfg):
    """PatchMixer 분위수 예측(Quantile) 모델 인스턴스 생성."""
    from modeling_module.models.PatchMixer.PatchMixer import QuantileModel
    cfg = _ensure_patchmixer_config(cfg)
    return QuantileModel(cfg)


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

    # 1) dict 변환
    if isinstance(cfg, dict):
        d = dict(cfg)
    elif hasattr(cfg, "__dict__"):
        d = dict(cfg.__dict__)
    else:
        raise TypeError(f"Unsupported cfg type: {type(cfg)}")

    # 2) 파라미터 매핑 및 Alias 처리
    # (TitanConfig 필드명 변경 시 호환성 유지 로직)
    if "use_exogenous_mode" in d and "use_exogenous" not in d:
        # 필요 시 구버전 키를 신버전 키로 매핑
        pass

    # 3) 유효 필드 필터링
    # Config 클래스에 정의된 필드만 남겨 초기화 오류 방지
    allowed = {f.name for f in fields(TitanConfig)}
    d = {k: v for k, v in d.items() if k in allowed}

    return TitanConfig(**d)


def build_titan_base(cfg):
    """Titan 기본 모델(BaseModel) 인스턴스 생성."""
    from modeling_module.models.Titan.Titans import TitanBaseModel
    cfg = _ensure_titan_config(cfg)
    return TitanBaseModel.from_config(cfg)


def build_titan_lmm(cfg):
    """Titan LMM(Large Multi-modal Model) 변형 모델 생성."""
    from modeling_module.models.Titan.Titans import TitanLMMModel
    cfg = _ensure_titan_config(cfg)
    return TitanLMMModel.from_config(cfg)


def build_titan_seq2seq(cfg):
    """Titan Seq2Seq 변형 모델 생성."""
    from modeling_module.models.Titan.Titans import TitanSeq2SeqModel
    cfg = _ensure_titan_config(cfg)
    return TitanSeq2SeqModel.from_config(cfg)


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
        # 중첩된(Nested) Config 객체 변환 시도
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
            # 변환 실패 시 원본 dict 구조 유지하며 Config 생성 시도
            pass

        return PatchTSTConfig(**cfgd)

    if hasattr(cfg, "__dict__"):
        return PatchTSTConfig(**cfg.__dict__)

    raise TypeError(f"Unsupported config type for PatchTST: {type(cfg)}")


def build_patchTST_base(cfg):
    """PatchTST 점 예측(Point) 모델 인스턴스 생성."""
    from modeling_module.models.PatchTST.supervised.PatchTST import PatchTSTPointModel
    cfg = _ensure_patchtst_config(cfg)
    return PatchTSTPointModel.from_config(cfg)


def build_patchTST_quantile(cfg):
    """PatchTST 분위수 예측(Quantile) 모델 인스턴스 생성."""
    from modeling_module.models.PatchTST.supervised.PatchTST import PatchTSTQuantileModel
    cfg = _ensure_patchtst_config(cfg)
    return PatchTSTQuantileModel.from_config(cfg)

def build_patchTST_dist(cfg, *, min_scale: Optional[float] = None):
    """PatchTST 분포 예측(Distribution) 모델 인스턴스 생성.

    반환 pred는 텐서가 아니라 dict 형태로:
      - pred["loc"]: (B, H)
      - pred["scale"]: (B, H)  (항상 양수; LossComputer의 dist loss가 이를 사용)
    """
    from modeling_module.models.PatchTST.supervised.PatchTST import PatchTSTDistModel
    cfg = _ensure_patchtst_config(cfg)

    if min_scale is None:
        # cfg에 dist_min_scale이 없을 수 있으므로 안전하게 기본값 사용
        min_scale = float(getattr(cfg, "dist_min_scale", 1e-3))

    return PatchTSTDistModel(cfg=cfg, min_scale=float(min_scale))

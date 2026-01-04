'''
* 사용법 *
from modeling_module.models import build_model
model = build_model("Titan_LMM", cfg)
'''

# src/modeling_module/models/__init__.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict

__all__ = [
    # unified entrypoint
    "build_model",
    "MODEL_BUILDERS",
    "list_available_models",

    # explicit builders (stable public surface)
    "build_patch_mixer_base",
    "build_patch_mixer_quantile",
    "build_titan_base",
    "build_titan_lmm",
    "build_titan_seq2seq",
    "build_patchTST_base",
    "build_patchTST_quantile",
]

if TYPE_CHECKING:
    from .model_builder import (
        build_patch_mixer_base,
        build_patch_mixer_quantile,
        build_titan_base,
        build_titan_lmm,
        build_titan_seq2seq,
        build_patchTST_base,
        build_patchTST_quantile,
    )

# Lazy import map: import modeling_module.models 시점에 heavy import 방지
_LAZY = {
    "build_patch_mixer_base": (".model_builder", "build_patch_mixer_base"),
    "build_patch_mixer_quantile": (".model_builder", "build_patch_mixer_quantile"),
    "build_titan_base": (".model_builder", "build_titan_base"),
    "build_titan_lmm": (".model_builder", "build_titan_lmm"),
    "build_titan_seq2seq": (".model_builder", "build_titan_seq2seq"),
    "build_patchTST_base": (".model_builder", "build_patchTST_base"),
    "build_patchTST_quantile": (".model_builder", "build_patchTST_quantile"),
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_path, attr = _LAZY[name]
    from importlib import import_module

    mod = import_module(module_path, package=__name__)
    value = getattr(mod, attr)
    globals()[name] = value  # cache
    return value


def _resolve_builders() -> Dict[str, Callable[..., Any]]:
    """
    실제로 builder dict가 필요할 때만 model_builder import를 트리거.
    """
    # 여기서 getattr을 쓰면 __getattr__을 통해 lazy 로딩됨
    return {
        # 이름은 운영/학습 코드에서 쓰는 canonical key로 고정 추천
        "PatchMixer_Base": build_patch_mixer_base,          # noqa: F821
        "PatchMixer_Quantile": build_patch_mixer_quantile,  # noqa: F821
        "Titan_Base": build_titan_base,                     # noqa: F821
        "Titan_LMM": build_titan_lmm,                       # noqa: F821
        "Titan_Seq2Seq": build_titan_seq2seq,               # noqa: F821
        "PatchTST_Base": build_patchTST_base,               # noqa: F821
        "PatchTST_Quantile": build_patchTST_quantile,       # noqa: F821
    }


# 필요 시 외부에서 참조 가능한 “등록표”를 제공하되,
# import 시점에 즉시 로딩하지 않도록 property-like 함수로 운영하는 것을 권장합니다.
def list_available_models() -> list[str]:
    return sorted(_resolve_builders().keys())


def build_model(name: str, cfg: Any) -> Any:
    """
    학습 시작점에서 쓰게 될 범용 엔트리포인트.
    - name: "Titan_LMM" 등
    - cfg: dataclass/dict/namespace 등을 model_builder에서 호환 처리
    """
    builders = _resolve_builders()
    if name not in builders:
        available = ", ".join(sorted(builders.keys()))
        raise KeyError(f"Unknown model '{name}'. Available: [{available}]")
    return builders[name](cfg)


# (옵션) dict가 필요하면 외부에서 호출하도록 제공
def MODEL_BUILDERS() -> Dict[str, Callable[..., Any]]:
    return _resolve_builders()
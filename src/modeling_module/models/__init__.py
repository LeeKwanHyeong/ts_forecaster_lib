'''
* 사용법 *
from modeling_module.models import build_model
model = build_model("titan_lmm", cfg)
'''

# src/modeling_module/models/__init__.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict

from .registry import (
    build_model,
    get_model_builders,
    get_patchmixer_default_model_key,
    get_patchtst_default_model_key,
    list_available_model_keys,
)

__all__ = [
    # unified entrypoint
    "build_model",
    "MODEL_BUILDERS",
    "list_available_models",
    "get_patchmixer_default_model_key",
    "get_patchtst_default_model_key",

    # explicit builders (stable public surface)
    "build_patch_mixer",
    "build_patch_mixer_exogenous",
    "build_titan_base",
    "build_titan_lmm",
    "build_titan_seq2seq",
    "build_patchTST",
    "build_patchTST_exogenous",
    "build_patchTST_quantile",
    "build_patchTST_quantile_exogenous",
    "build_exotst",
    "build_nhits",
    "build_timexer",
    "build_sellm",
    "PatchMixerConfig",
    "PatchMixerExogenousConfig",
]

if TYPE_CHECKING:
    from .model_builder import (
        build_exotst,
        build_nhits,
        build_sellm,
        build_patch_mixer,
        build_patch_mixer_exogenous,
        build_titan_base,
        build_titan_lmm,
        build_titan_seq2seq,
        build_patchTST,
        build_patchTST_exogenous,
        build_patchTST_quantile,
        build_patchTST_quantile_exogenous,
        build_timexer,
    )
    from .PatchMixer.common.configs import PatchMixerConfig, PatchMixerExogenousConfig

# Lazy import map: import modeling_module.models 시점에 heavy import 방지
_LAZY = {
    "build_patch_mixer": (".model_builder", "build_patch_mixer"),
    "build_patch_mixer_exogenous": (".model_builder", "build_patch_mixer_exogenous"),
    "build_titan_base": (".model_builder", "build_titan_base"),
    "build_titan_lmm": (".model_builder", "build_titan_lmm"),
    "build_titan_seq2seq": (".model_builder", "build_titan_seq2seq"),
    "build_patchTST": (".model_builder", "build_patchTST"),
    "build_patchTST_exogenous": (".model_builder", "build_patchTST_exogenous"),
    "build_patchTST_quantile": (".model_builder", "build_patchTST_quantile"),
    "build_patchTST_quantile_exogenous": (".model_builder", "build_patchTST_quantile_exogenous"),
    "build_exotst": (".model_builder", "build_exotst"),
    "build_nhits": (".model_builder", "build_nhits"),
    "build_timexer": (".model_builder", "build_timexer"),
    "build_sellm": (".model_builder", "build_sellm"),
    "PatchMixerConfig": (
        ".PatchMixer.common.configs",
        "PatchMixerConfig",
    ),
    "PatchMixerExogenousConfig": (
        ".PatchMixer.common.configs",
        "PatchMixerExogenousConfig",
    ),
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


def list_available_models() -> list[str]:
    return list_available_model_keys()


# (옵션) dict가 필요하면 외부에서 호출하도록 제공
def MODEL_BUILDERS() -> Dict[str, Callable[..., Any]]:
    return get_model_builders()

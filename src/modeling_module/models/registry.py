from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional


def _norm_name(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    builder_module: str
    builder_attr: str
    label: str
    aliases: tuple[str, ...] = ()
    class_names: tuple[str, ...] = ()
    checkpoint_aliases: tuple[str, ...] = ()
    trainable: bool = True
    included_in_family: bool = True

    def load_builder(self) -> Callable[..., Any]:
        module = import_module(self.builder_module)
        return getattr(module, self.builder_attr)


MODEL_SPECS: dict[str, ModelSpec] = {
    "patchtst_base": ModelSpec(
        key="patchtst_base",
        family="patchtst",
        builder_module="modeling_module.models.model_builder",
        builder_attr="build_patchTST",
        label="PatchTST Base",
        aliases=("patchtstbase", "patchtstpoint", "patchtstdist", "patchtst"),
        class_names=("PatchTSTModel",),
        checkpoint_aliases=("PatchTST", "PatchTSTBase", "PatchTSTDist"),
    ),
    "patchtst_quantile": ModelSpec(
        key="patchtst_quantile",
        family="patchtst",
        builder_module="modeling_module.models.model_builder",
        builder_attr="build_patchTST_quantile",
        label="PatchTST Quantile",
        aliases=("patchtstquantile", "patchtstq"),
        class_names=("PatchTSTQuantileModel",),
        checkpoint_aliases=("PatchTSTQuantile",),
    ),
    "patchmixer_base": ModelSpec(
        key="patchmixer_base",
        family="patchmixer",
        builder_module="modeling_module.models.model_builder",
        builder_attr="build_patch_mixer",
        label="PatchMixer Base",
        aliases=("patchmixerbase", "patchmixerdist", "patchmixer"),
        class_names=("PatchMixerModel", "PatchMixerPointModel", "PatchMixerDistributionModel"),
        checkpoint_aliases=("PatchMixer", "PatchMixerBase", "PatchMixerDist"),
    ),
    "patchmixer_quantile": ModelSpec(
        key="patchmixer_quantile",
        family="patchmixer",
        builder_module="modeling_module.models.model_builder",
        builder_attr="build_patch_mixer_quantile",
        label="PatchMixer Quantile",
        aliases=("patchmixerquantile", "patchmixerq"),
        class_names=("PatchMixerQuantileModel",),
        checkpoint_aliases=("PatchMixerQuantile",),
    ),
    "titan_base": ModelSpec(
        key="titan_base",
        family="titan",
        builder_module="modeling_module.models.model_builder",
        builder_attr="build_titan_base",
        label="Titan Base",
        aliases=("titanbase",),
        class_names=("TitanBaseModel",),
        checkpoint_aliases=("TitanBase", "TitanBaseDist"),
    ),
    "titan_lmm": ModelSpec(
        key="titan_lmm",
        family="titan",
        builder_module="modeling_module.models.model_builder",
        builder_attr="build_titan_lmm",
        label="Titan LMM",
        aliases=("titanlmm",),
        class_names=("TitanLMMModel",),
        checkpoint_aliases=("TitanLMM", "TitanLMMDist"),
    ),
    "titan_seq2seq": ModelSpec(
        key="titan_seq2seq",
        family="titan",
        builder_module="modeling_module.models.model_builder",
        builder_attr="build_titan_seq2seq",
        label="Titan Seq2Seq",
        aliases=("titanseq2seq", "titanseq"),
        class_names=("TitanSeq2SeqModel",),
        checkpoint_aliases=("TitanSeq2Seq", "TitanSeq2SeqDist"),
    ),
    "exotst_base": ModelSpec(
        key="exotst_base",
        family="exotst",
        builder_module="modeling_module.models.model_builder",
        builder_attr="build_exotst",
        label="ExoTST Base",
        aliases=("exotst", "exotstbase"),
        class_names=("ExoTST",),
        checkpoint_aliases=("ExoTST", "ExoTSTBase"),
    ),
}


TRAINING_FAMILY_DEFAULTS: dict[str, tuple[str, ...]] = {
    "patchtst": ("patchtst_base", "patchtst_quantile"),
    "patchmixer": ("patchmixer_base", "patchmixer_quantile"),
    "titan": ("titan_base", "titan_lmm", "titan_seq2seq"),
    "exotst": ("exotst_base",),
}


TRAINING_FAMILY_ALIASES: dict[str, tuple[str, ...]] = {
    "patchtst": ("patchtst",),
    "patchmixer": ("patchmixer",),
    "titan": ("titan",),
    "exotst": ("exotst",),
}


_ARTIFACT_ALIAS_TO_KEY: dict[str, str] = {}
for _key, _spec in MODEL_SPECS.items():
    for _name in (_key, *_spec.aliases, *_spec.class_names, *_spec.checkpoint_aliases):
        _ARTIFACT_ALIAS_TO_KEY[_norm_name(_name)] = _key

_FAMILY_ALIAS_TO_KEY: dict[str, str] = {}
for _family, _aliases in TRAINING_FAMILY_ALIASES.items():
    for _name in (_family, *_aliases):
        _FAMILY_ALIAS_TO_KEY[_norm_name(_name)] = _family


def list_available_model_keys() -> list[str]:
    return sorted(MODEL_SPECS.keys())


def list_trainable_model_keys() -> list[str]:
    return sorted(key for key, spec in MODEL_SPECS.items() if spec.trainable)


def list_training_families() -> list[str]:
    return list(TRAINING_FAMILY_DEFAULTS.keys())


def get_model_spec(key: str) -> ModelSpec:
    canonical = resolve_artifact_model_key(key)
    return MODEL_SPECS[canonical]


def get_model_builder(key: str) -> Callable[..., Any]:
    return get_model_spec(key).load_builder()


def get_model_builders(keys: Optional[Iterable[str]] = None) -> dict[str, Callable[..., Any]]:
    selected = list(keys) if keys is not None else list(MODEL_SPECS.keys())
    return {resolve_artifact_model_key(key): get_model_builder(key) for key in selected}


def resolve_artifact_model_key(name: str) -> str:
    normalized = _norm_name(name)
    if normalized in _ARTIFACT_ALIAS_TO_KEY:
        return _ARTIFACT_ALIAS_TO_KEY[normalized]
    raise ValueError(f"Unknown artifact model name: {name!r}")


def resolve_training_request_key(name: str) -> str:
    normalized = _norm_name(name)
    if normalized in _FAMILY_ALIAS_TO_KEY:
        return _FAMILY_ALIAS_TO_KEY[normalized]

    artifact_key = resolve_artifact_model_key(name)
    spec = MODEL_SPECS[artifact_key]
    if not spec.trainable:
        raise ValueError(f"Model {artifact_key!r} is not trainable through the public training API.")
    return artifact_key


def expand_training_targets(targets: Optional[Iterable[str]]) -> list[str]:
    requested = list(targets) if targets is not None else []
    if not requested:
        requested = ["patchtst"]
    expanded: list[str] = []

    for raw in requested:
        key = resolve_training_request_key(raw)
        if key in TRAINING_FAMILY_DEFAULTS:
            expanded.extend(TRAINING_FAMILY_DEFAULTS[key])
        else:
            expanded.append(key)

    return _dedupe_preserve_order(expanded)


def family_for_artifact_key(key: str) -> str:
    return MODEL_SPECS[resolve_artifact_model_key(key)].family


def ordered_training_families_for_targets(targets: Iterable[str]) -> list[str]:
    families: list[str] = []
    seen: set[str] = set()
    for key in targets:
        family = family_for_artifact_key(key)
        if family not in seen:
            seen.add(family)
            families.append(family)
    return families


def filter_targets_for_family(targets: Iterable[str], family: str) -> list[str]:
    return [resolve_artifact_model_key(key) for key in targets if family_for_artifact_key(key) == family]


def infer_artifact_model_key_from_checkpoint(
    ckpt: Mapping[str, Any],
    *,
    ckpt_path: Optional[str] = None,
) -> str:
    candidates: list[str] = []

    meta = ckpt.get("meta")
    if isinstance(meta, Mapping):
        for key in ("model_key", "artifact_key"):
            value = meta.get(key)
            if value:
                candidates.append(str(value))

    for key in ("model_key", "artifact_key", "model_class"):
        value = ckpt.get(key)
        if value:
            candidates.append(str(value))

    if ckpt_path:
        candidates.append(Path(ckpt_path).stem)

    for candidate in candidates:
        try:
            return resolve_artifact_model_key(candidate)
        except ValueError:
            continue

    raise ValueError(f"Unable to infer artifact model key from checkpoint: {ckpt_path or '<memory>'}")


def build_model(name: str, cfg: Any) -> Any:
    builder = get_model_builder(name)
    return builder(cfg)


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out

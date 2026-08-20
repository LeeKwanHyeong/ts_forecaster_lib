from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from modeling_module._internal.optional_features import SELLM_AVAILABLE


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
    deprecated: bool = False
    deprecation_message: Optional[str] = None
    load_only: bool = False
    exogenous_policy: str = "none"
    exogenous_inputs: tuple[str, ...] = ()
    fusion_strategy: Optional[str] = None

    def load_builder(self) -> Callable[..., Any]:
        module = import_module(self.builder_module)
        return getattr(module, self.builder_attr)


_TITAN_DEPRECATION_MESSAGE = (
    "Titan public training is deprecated and excluded from DSIO defaults and promotion runs. "
    "Explicit Titan training remains available for compatibility, and existing supported titan_* "
    "checkpoints remain loadable. Migrate new training runs to another supported family."
)


MODEL_SPECS: dict[str, ModelSpec] = {
    "cgmm": ModelSpec(
        key="cgmm",
        family="cgmm",
        builder_module="modeling_module.models.model_builder",
        builder_attr="build_cgmm",
        label="Conditional Gaussian Mixture Model",
        aliases=("conditionalgmm", "lifecyclecgmm", "ltbcgmm"),
        class_names=(
            "CGMMForecaster",
            "ConditionalGaussianMixtureForecaster",
        ),
        checkpoint_aliases=("CGMM", "ConditionalGMM"),
        trainable=False,
        included_in_family=False,
        exogenous_policy="lifecycle_conditional",
        exogenous_inputs=(
            "static",
            "observed",
            "known_future",
        ),
        fusion_strategy="joint_condition_target_gaussian_mixture",
    ),
    "similar_lifecycle": ModelSpec(
        key="similar_lifecycle",
        family="similar_lifecycle",
        builder_module="modeling_module.models.model_builder",
        builder_attr="build_similar_lifecycle",
        label="Similar Lifecycle Retrieval",
        aliases=(
            "similarlifecycle",
            "lifecycleknn",
            "ltbsimilarlifecycle",
        ),
        class_names=("SimilarLifecycleForecaster",),
        checkpoint_aliases=("SimilarLifecycle", "LifecycleKNN"),
        trainable=False,
        included_in_family=False,
        exogenous_policy="lifecycle_conditional",
        exogenous_inputs=("static", "observed"),
        fusion_strategy="inverse_distance_weighted_lifecycle_retrieval",
    ),
    "patchtst_base": ModelSpec(
        key="patchtst_base",
        family="patchtst",
        builder_module="modeling_module.models.model_builder",
        builder_attr="build_patchTST",
        label="PatchTST Base",
        aliases=("patchtstbase", "patchtstpoint", "patchtstdist", "patchtst"),
        class_names=("PatchTSTModel", "PatchTSTEndogenousModel"),
        checkpoint_aliases=("PatchTST", "PatchTSTBase", "PatchTSTDist"),
        exogenous_policy="optional_legacy",
        exogenous_inputs=("past_cont", "past_cat", "future_cont"),
        fusion_strategy="legacy_config_routing",
    ),
    "patchtst_exogenous": ModelSpec(
        key="patchtst_exogenous",
        family="patchtst",
        builder_module="modeling_module.models.model_builder",
        builder_attr="build_patchTST_exogenous",
        label="PatchTST Exogenous",
        aliases=("patchtstexo", "patchtstexogenous"),
        class_names=("PatchTSTExogenousModel",),
        checkpoint_aliases=("PatchTSTExogenous", "PatchTSTExo"),
        included_in_family=False,
        exogenous_policy="required",
        exogenous_inputs=("past_cont", "past_cat", "future_cont"),
        fusion_strategy="patch_concat+future_cross_attention",
    ),
    "patchtst_quantile": ModelSpec(
        key="patchtst_quantile",
        family="patchtst",
        builder_module="modeling_module.models.model_builder",
        builder_attr="build_patchTST_quantile",
        label="PatchTST Quantile",
        aliases=("patchtstquantile", "patchtstq"),
        class_names=("PatchTSTQuantileModel", "PatchTSTQuantileEndogenousModel"),
        checkpoint_aliases=("PatchTSTQuantile",),
        exogenous_policy="optional_legacy",
        exogenous_inputs=("past_cont", "past_cat", "future_cont"),
        fusion_strategy="legacy_config_routing",
    ),
    "patchtst_quantile_exogenous": ModelSpec(
        key="patchtst_quantile_exogenous",
        family="patchtst",
        builder_module="modeling_module.models.model_builder",
        builder_attr="build_patchTST_quantile_exogenous",
        label="PatchTST Quantile Exogenous",
        aliases=("patchtstquantileexo", "patchtstquantileexogenous"),
        class_names=("PatchTSTQuantileExogenousModel",),
        checkpoint_aliases=("PatchTSTQuantileExogenous",),
        included_in_family=False,
        exogenous_policy="required",
        exogenous_inputs=("past_cont", "past_cat", "future_cont"),
        fusion_strategy="patch_concat+future_cross_attention",
    ),
    "patchmixer": ModelSpec(
        key="patchmixer",
        family="patchmixer",
        builder_module="modeling_module.models.model_builder",
        builder_attr="build_patch_mixer",
        label="PatchMixer",
        aliases=(
            "patchmixer_original",
            "patchmixeroriginal",
            "patchmixercanonical",
            "patchmixerupstream",
        ),
        class_names=("PatchMixerModel", "PatchMixerOriginalModel"),
        checkpoint_aliases=("PatchMixer", "PatchMixerOriginal", "PatchMixerCanonical"),
        exogenous_policy="none",
    ),
    "patchmixer_exo": ModelSpec(
        key="patchmixer_exo",
        family="patchmixer",
        builder_module="modeling_module.models.model_builder",
        builder_attr="build_patch_mixer_exogenous",
        label="PatchMixer Exogenous",
        aliases=("patchmixer_exogenous", "patchmixerexo", "patchmixerexogenous"),
        class_names=("PatchMixerExogenousModel",),
        checkpoint_aliases=("PatchMixerExogenous", "PatchMixerExo"),
        exogenous_policy="required",
        exogenous_inputs=("past_cont", "past_cat", "future_cont"),
        fusion_strategy="gated_residual+future_shift",
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
        deprecated=True,
        deprecation_message=_TITAN_DEPRECATION_MESSAGE,
        exogenous_policy="optional_legacy",
        exogenous_inputs=("past_cont", "future_cont"),
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
        deprecated=True,
        deprecation_message=_TITAN_DEPRECATION_MESSAGE,
        exogenous_policy="optional_legacy",
        exogenous_inputs=("past_cont", "future_cont"),
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
        deprecated=True,
        deprecation_message=_TITAN_DEPRECATION_MESSAGE,
        exogenous_policy="optional_legacy",
        exogenous_inputs=("past_cont", "future_cont"),
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
        exogenous_policy="required",
        exogenous_inputs=("past_cont", "future_cont"),
        fusion_strategy="dedicated_exogenous_encoder",
    ),
    "nhits_base": ModelSpec(
        key="nhits_base",
        family="nhits",
        builder_module="modeling_module.models.model_builder",
        builder_attr="build_nhits",
        label="N-HiTS Base",
        aliases=("nhits", "nhitsbase", "n-hits"),
        class_names=("NHITSModel",),
        checkpoint_aliases=("NHITSBase", "NHiTSBase"),
        exogenous_policy="none",
    ),
    "timemixer": ModelSpec(
        key="timemixer",
        family="timemixer",
        builder_module="modeling_module.models.model_builder",
        builder_attr="build_timemixer",
        label="TimeMixer",
        aliases=("timemixerbase", "timemixercanonical"),
        class_names=("TimeMixerModel",),
        checkpoint_aliases=("TimeMixer", "TimeMixerCanonical"),
        trainable=True,
        exogenous_policy="none",
    ),
    "timexer_base": ModelSpec(
        key="timexer_base",
        family="timexer",
        builder_module="modeling_module.models.model_builder",
        builder_attr="build_timexer",
        label="TimeXer Base",
        aliases=("timexer", "timexerbase"),
        class_names=("TimeXerModel",),
        checkpoint_aliases=("TimeXer", "TimeXerBase"),
        exogenous_policy="required",
        exogenous_inputs=("past_cont",),
        fusion_strategy="global_token_cross_attention",
    ),
}

if SELLM_AVAILABLE:
    MODEL_SPECS["sellm_base"] = ModelSpec(
        key="sellm_base",
        family="sellm",
        builder_module="modeling_module.models.model_builder",
        builder_attr="build_sellm",
        label="SELLM Base",
        aliases=("sellm", "sellmbase", "se_llm", "sellmforecast"),
        class_names=("SELLMModel",),
        checkpoint_aliases=("SELLM", "SELLMBase", "SE-LLM"),
        exogenous_policy="optional",
        exogenous_inputs=("future_cont",),
        fusion_strategy="semantic_future_conditioning",
    )


_PATCHMIXER_LEGACY_MESSAGE = (
    "This PatchMixer artifact is load-only. Enhanced endogenous, distribution, and "
    "quantile training were retired; use 'patchmixer' or 'patchmixer_exo' for new runs."
)


LEGACY_MODEL_SPECS: dict[str, ModelSpec] = {
    "patchmixer_base": ModelSpec(
        key="patchmixer_base",
        family="patchmixer",
        builder_module="modeling_module.models.model_builder",
        builder_attr="build_patch_mixer_legacy",
        label="PatchMixer Enhanced (load-only)",
        aliases=("patchmixerbase", "patchmixerdist"),
        class_names=(
            "PatchMixerEnhancedModel",
            "PatchMixerPointModel",
            "PatchMixerDistributionModel",
            "PatchMixerEndogenousModel",
            "BaseModel",
        ),
        checkpoint_aliases=("PatchMixerBase", "PatchMixerDist"),
        trainable=False,
        included_in_family=False,
        deprecated=True,
        deprecation_message=_PATCHMIXER_LEGACY_MESSAGE,
        load_only=True,
        exogenous_policy="optional_legacy",
        exogenous_inputs=("past_cont", "past_cat", "future_cont"),
        fusion_strategy="legacy_config_routing",
    ),
    "patchmixer_quantile": ModelSpec(
        key="patchmixer_quantile",
        family="patchmixer",
        builder_module="modeling_module.models.model_builder",
        builder_attr="build_patch_mixer_quantile_legacy",
        label="PatchMixer Quantile (load-only)",
        aliases=("patchmixerquantile", "patchmixerq"),
        class_names=("PatchMixerQuantileModel", "PatchMixerQuantileEndogenousModel", "QuantileModel"),
        checkpoint_aliases=("PatchMixerQuantile",),
        trainable=False,
        included_in_family=False,
        deprecated=True,
        deprecation_message=_PATCHMIXER_LEGACY_MESSAGE,
        load_only=True,
        exogenous_policy="optional_legacy",
        exogenous_inputs=("past_cont", "past_cat", "future_cont"),
        fusion_strategy="legacy_config_routing",
    ),
    "patchmixer_quantile_exogenous": ModelSpec(
        key="patchmixer_quantile_exogenous",
        family="patchmixer",
        builder_module="modeling_module.models.model_builder",
        builder_attr="build_patch_mixer_quantile_legacy",
        label="PatchMixer Quantile Exogenous (load-only)",
        aliases=("patchmixerquantileexo", "patchmixerquantileexogenous"),
        class_names=("PatchMixerQuantileExogenousModel",),
        checkpoint_aliases=("PatchMixerQuantileExogenous",),
        trainable=False,
        included_in_family=False,
        deprecated=True,
        deprecation_message=_PATCHMIXER_LEGACY_MESSAGE,
        load_only=True,
        exogenous_policy="required",
        exogenous_inputs=("past_cont", "past_cat", "future_cont"),
        fusion_strategy="gated_residual+future_shift",
    ),
}


_ALL_MODEL_SPECS = {**MODEL_SPECS, **LEGACY_MODEL_SPECS}


TRAINING_FAMILY_DEFAULTS: dict[str, tuple[str, ...]] = {
    "patchtst": ("patchtst_base", "patchtst_quantile"),
    "patchmixer": ("patchmixer",),
    "titan": ("titan_base", "titan_lmm", "titan_seq2seq"),
    "exotst": ("exotst_base",),
    "nhits": ("nhits_base",),
    "timemixer": ("timemixer",),
    "timexer": ("timexer_base",),
}

if SELLM_AVAILABLE:
    TRAINING_FAMILY_DEFAULTS["sellm"] = ("sellm_base",)


PRODUCTION_REFIT_ARTIFACT_KEYS: tuple[str, ...] = (
    "patchtst_base",
    "patchtst_quantile",
    "patchmixer",
    "nhits_base",
    "timemixer",
    "exotst_base",
    "patchtst_exogenous",
    "timexer_base",
)


PATCHTST_CAPABILITY_DEFAULTS: dict[str, str] = {
    "endogenous_point": "patchtst_base",
    "exogenous_point": "patchtst_exogenous",
    "endogenous_distribution": "patchtst_base",
    "exogenous_distribution": "patchtst_exogenous",
    "endogenous_quantile": "patchtst_quantile",
    "exogenous_quantile": "patchtst_quantile_exogenous",
}


_PATCHTST_CAPABILITY_ALIASES: dict[str, str] = {
    _norm_name("point"): "endogenous_point",
    _norm_name("endogenous_point"): "endogenous_point",
    _norm_name("exogenous_point"): "exogenous_point",
    _norm_name("distribution"): "endogenous_distribution",
    _norm_name("dist"): "endogenous_distribution",
    _norm_name("endogenous_distribution"): "endogenous_distribution",
    _norm_name("exogenous_distribution"): "exogenous_distribution",
    _norm_name("exogenous_dist"): "exogenous_distribution",
    _norm_name("quantile"): "endogenous_quantile",
    _norm_name("endogenous_quantile"): "endogenous_quantile",
    _norm_name("exogenous_quantile"): "exogenous_quantile",
}


PATCHMIXER_CAPABILITY_DEFAULTS: dict[str, str] = {
    "endogenous_point": "patchmixer",
    "exogenous_point": "patchmixer_exo",
}


_PATCHMIXER_CAPABILITY_ALIASES: dict[str, str] = {
    _norm_name("point"): "endogenous_point",
    _norm_name("endogenous_point"): "endogenous_point",
    _norm_name("exogenous_point"): "exogenous_point",
}


TRAINING_FAMILY_ALIASES: dict[str, tuple[str, ...]] = {
    "patchtst": ("patchtst",),
    "patchmixer": ("patchmixer",),
    "titan": ("titan",),
    "exotst": ("exotst",),
    "nhits": ("nhits", "n-hits"),
    "timemixer": ("timemixer",),
    "timexer": ("timexer",),
}

if SELLM_AVAILABLE:
    TRAINING_FAMILY_ALIASES["sellm"] = ("sellm", "se_llm")


_ARTIFACT_ALIAS_TO_KEY: dict[str, str] = {}
for _key, _spec in _ALL_MODEL_SPECS.items():
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


def get_patchtst_default_model_key(capability: str = "endogenous_point") -> str:
    """Return the PatchTST artifact responsible for a forecasting capability."""
    normalized = _norm_name(capability)
    canonical = _PATCHTST_CAPABILITY_ALIASES.get(normalized)
    if canonical is None:
        supported = ", ".join(PATCHTST_CAPABILITY_DEFAULTS)
        raise ValueError(
            f"Unknown PatchTST capability: {capability!r}. Supported: {supported}."
        )
    return PATCHTST_CAPABILITY_DEFAULTS[canonical]


def get_patchmixer_default_model_key(capability: str = "endogenous_point") -> str:
    """Return the promoted PatchMixer artifact for a forecasting capability.

    This selector does not alter family expansion or checkpoint aliases. It is
    intended for callers that have already resolved the requested capability.
    """
    normalized = _norm_name(capability)
    canonical = _PATCHMIXER_CAPABILITY_ALIASES.get(normalized)
    if canonical is None:
        supported = ", ".join(PATCHMIXER_CAPABILITY_DEFAULTS)
        raise ValueError(
            f"Unknown PatchMixer capability: {capability!r}. Supported: {supported}."
        )
    return PATCHMIXER_CAPABILITY_DEFAULTS[canonical]


def get_model_spec(key: str) -> ModelSpec:
    canonical = resolve_artifact_model_key(key)
    return _ALL_MODEL_SPECS[canonical]


def get_model_builder(key: str) -> Callable[..., Any]:
    return get_model_spec(key).load_builder()


def get_training_deprecation_messages(targets: Iterable[str]) -> list[str]:
    """Return unique deprecation messages for canonical public training targets."""
    messages: list[str] = []
    seen: set[str] = set()
    for key in targets:
        message = get_model_spec(key).deprecation_message
        if message and message not in seen:
            seen.add(message)
            messages.append(message)
    return messages


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
    spec = _ALL_MODEL_SPECS[artifact_key]
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
    return _ALL_MODEL_SPECS[resolve_artifact_model_key(key)].family


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
    # Before consolidation, the Enhanced implementation also serialized the
    # class name `PatchMixerModel`. Its state dict is structurally distinct from
    # the paper model (`backbone.*` versus `model.*`), so route it load-only.
    if _norm_name(str(ckpt.get("model_class", ""))) == "patchmixermodel":
        state = None
        for state_key in ("model_state", "state_dict", "model_state_dict", "model", "net", "weights"):
            candidate = ckpt.get(state_key)
            if isinstance(candidate, Mapping):
                state = candidate
                break
        if state is not None and any(
            str(name).startswith(("backbone.", "expander.", "head."))
            for name in state
        ):
            return "patchmixer_base"

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

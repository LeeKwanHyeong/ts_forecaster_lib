from __future__ import annotations

import inspect
import json
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from modeling_module.api.data import _materialize_payload as _materialize_data_payload
from modeling_module.api.data import build_datamodule
from modeling_module._internal.checkpoint_runtime import (
    save_training_manifest,
    summarize_training_results,
)
from modeling_module._internal.device_runtime import default_device, resolve_device
from modeling_module._internal.model_registry import (
    expand_training_targets,
    resolve_training_request_key,
)
from modeling_module._internal.training_runtime import (
    infer_future_exo_spec_from_loader,
    infer_past_exo_dim_from_loader_for_exotst,
    get_freq_spec,
    run_total_train,
)

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _is_default_value(value: Any, default_value: Any) -> bool:
    if default_value is MISSING:
        return False
    if default_value is None:
        return value is None
    try:
        return value == default_value
    except Exception:
        return False


@dataclass
class TrainerConfig:
    """
    Supervised training controls.

    Parameters
    - `epochs`: Single-stage shortcut. When set alone, it is forwarded to `warmup_epochs`.
    - `lr`: Single-stage shortcut. Forwarded to `base_lr` when `base_lr` is omitted.
    - `warmup_epochs`: Number of epochs for the first supervised stage.
    - `spike_epochs`: Optional second stage used by the internal staged trainer.
    - `base_lr`: Base learning rate used to build the internal stage configs.
    - `loss`: Backward-compatible alias for `loss_point`.
    - `loss_point`: Loss for point/distribution style models.
    - `loss_quantile`: Loss for quantile models.
    - `use_intermittent`: Enable intermittent-demand weighting logic.
    - `val_use_weights`: Apply intermittent weights during validation as well.
    """
    epochs: Optional[int] = None
    lr: Optional[float] = None
    warmup_epochs: Optional[int] = None
    spike_epochs: Optional[int] = None
    base_lr: Optional[float] = None
    loss: Any = None
    loss_point: Any = None
    loss_quantile: Any = None
    use_intermittent: Optional[bool] = None
    val_use_weights: Optional[bool] = None


@dataclass
class SSLConfig:
    """
    Self-supervised learning options.

    Parameters
    - `mode`: One of `sl_only`, `ssl_only`, `full`.
      `sl_only` runs only supervised training.
      `ssl_only` runs only SSL pretraining.
      `full` runs SSL pretraining and then supervised finetuning.
    - `pretrain_epochs`: Number of SSL pretraining epochs.
    - `mask_ratio`: Masking ratio used by PatchTST pretraining.
    - `loss_type`: Reconstruction loss type for SSL pretraining.
    - `freeze_encoder_before_ft`: Optionally freeze encoder blocks at the beginning of finetuning.
    - `pretrained_ckpt_path`: Reuse an existing SSL checkpoint instead of creating a new one.

    Notes
    - `full` / `ssl_only` are currently meaningful for PatchTST family training.
    """
    mode: Optional[str] = None
    pretrain_epochs: Optional[int] = None
    mask_ratio: Optional[float] = None
    loss_type: Optional[str] = None
    freeze_encoder_before_ft: Optional[bool] = None
    pretrained_ckpt_path: Optional[str] = None


@dataclass
class RuntimeConfig:
    """
    Runtime settings.

    Parameters
    - `device`: Target runtime device such as `cpu`, `cuda`, `cuda:0`, or `mps`.
      If omitted, the library picks the first usable accelerator and otherwise falls back to CPU.
    """
    device: Optional[str] = None


@dataclass
class ArtifactConfig:
    """
    Output artifact settings.

    Parameters
    - `save_dir`: Directory where checkpoints and manifest files are written.
    - `auto_save_dir`: When `True`, create a timestamped artifact directory if `save_dir` is omitted.
    """
    save_dir: Optional[str] = None
    auto_save_dir: bool = True


@dataclass
class PatchTSTArchitectureConfig:
    """
    PatchTST family architecture overrides.

    These values override the frequency-policy defaults for PatchTST artifacts.
    """
    patch_len: Optional[int] = None
    stride: Optional[int] = None
    d_model: Optional[int] = None
    n_layers: Optional[int] = None
    d_ff: Optional[int] = None
    dropout: Optional[float] = None
    norm: Optional[str] = None
    pre_norm: Optional[bool] = None
    act: Optional[str] = None
    use_revin: Optional[bool] = None
    pe: Optional[str] = None
    learn_pe: Optional[bool] = None
    padding_patch: Optional[str] = None
    future_exo_fusion_mode: Optional[str] = None
    future_exo_fusion_dropout: Optional[float] = None


@dataclass
class TitanArchitectureConfig:
    """
    Titan family architecture overrides.
    """
    d_model: Optional[int] = None
    n_layers: Optional[int] = None
    n_heads: Optional[int] = None
    d_ff: Optional[int] = None
    dropout: Optional[float] = None
    contextual_mem_size: Optional[int] = None
    persistent_mem_size: Optional[int] = None
    use_revin: Optional[bool] = None
    final_clamp_nonneg: Optional[bool] = None


@dataclass
class PatchMixerArchitectureConfig:
    """
    PatchMixer family architecture overrides.
    """
    patch_len: Optional[int] = None
    stride: Optional[int] = None
    d_model: Optional[int] = None
    e_layers: Optional[int] = None
    f_out: Optional[int] = None
    head_hidden: Optional[int] = None
    dropout: Optional[float] = None
    head_dropout: Optional[float] = None
    use_revin: Optional[bool] = None
    final_nonneg: Optional[bool] = None
    expander_n_harmonics: Optional[int] = None


@dataclass
class ExoTSTArchitectureConfig:
    """
    ExoTST family architecture overrides.
    """
    patch_len: Optional[int] = None
    stride: Optional[int] = None
    d_model: Optional[int] = None
    n_heads: Optional[int] = None
    d_ff: Optional[int] = None
    dropout: Optional[float] = None
    attn_dropout: Optional[float] = None
    exo_enc_layers: Optional[int] = None
    fusion_layers: Optional[int] = None
    endo_dec_layers: Optional[int] = None
    exo_memory_mode: Optional[str] = None
    exo_nan_policy: Optional[str] = None
    use_revin: Optional[bool] = None
    subtract_last: Optional[bool] = None


@dataclass
class TimexerArchitectureConfig:
    """
    TimeXer family architecture overrides.

    Notes
    - TimeXer v1 intentionally keeps the original paper contract:
      past continuous exogenous inputs only, point forecast only.
    """

    patch_len: Optional[int] = None
    d_model: Optional[int] = None
    n_heads: Optional[int] = None
    d_ff: Optional[int] = None
    e_layers: Optional[int] = None
    dropout: Optional[float] = None
    factor: Optional[int] = None
    activation: Optional[str] = None
    use_norm: Optional[bool] = None


@dataclass
class ArchitectureConfig:
    """
    Family-level model architecture overrides.

    Notes
    - Overrides are applied per family. For example, `patchtst` settings affect
      both `patchtst_base` and `patchtst_quantile`.
    - Mapping-style input is also supported. Keys may be family names such as
      `patchtst`, `patchmixer`, `titan`, `exotst`, or canonical artifact keys
      such as `titan_base`.
    """
    patchtst: Optional[PatchTSTArchitectureConfig | Mapping[str, Any]] = None
    titan: Optional[TitanArchitectureConfig | Mapping[str, Any]] = None
    patchmixer: Optional[PatchMixerArchitectureConfig | Mapping[str, Any]] = None
    exotst: Optional[ExoTSTArchitectureConfig | Mapping[str, Any]] = None
    timexer: Optional[TimexerArchitectureConfig | Mapping[str, Any]] = None


@dataclass
class TrainRequest:
    """
    Preferred public request object for `train(...)`.

    Recommended usage is to provide nested configs:
    - `data`: Data loading / schema / exogenous setup
    - `trainer`: Supervised training options
    - `ssl`: Optional SSL pretraining options
    - `runtime`: Device selection
    - `artifacts`: Checkpoint / manifest output settings
    - `architecture`: Family-level model architecture overrides

    Flat fields such as `lookback`, `horizon`, `device`, `save_dir`, `use_ssl_mode`,
    and `loss_point` are still accepted for backward compatibility, but new code should
    prefer the nested config objects above.

    Parameters
    - `config_path`: Optional JSON/YAML config file to load first.
    - `config`: Optional in-memory mapping merged before explicit request fields.
    - `train_loader`, `val_loader`: Prebuilt loaders. Must be provided together.
    - `data`: `DataRequest` or mapping used to build a datamodule when loaders are not supplied.
    - `trainer`, `ssl`, `runtime`, `artifacts`: Nested config groups for public API usage.
    - `models` / `model` / `models_to_run`: Requested training targets. Families such as
      `patchtst` expand to their default artifacts.

    Backward-compatible flat aliases
    - `freq`, `lookback`, `horizon`, `device`
    - `warmup_epochs`, `spike_epochs`, `base_lr`
    - `save_dir`, `auto_save_dir`
    - `use_exogenous_mode`, `use_past_exogenous`, `use_future_exogenous`
    - `loss`, `loss_point`, `loss_quantile`
    - `use_ssl_mode`, `ssl_pretrain_epochs`, `ssl_mask_ratio`, `ssl_loss_type`,
      `ssl_freeze_encoder_before_ft`, `ssl_pretrained_ckpt_path`
    - `use_intermittent`, `val_use_weights`
    """
    config_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

    train_loader: Any = None
    val_loader: Any = None
    data: Any = None

    # Preferred grouped configs for new code.
    trainer: Optional[TrainerConfig | Mapping[str, Any]] = None
    ssl: Optional[SSLConfig | Mapping[str, Any]] = None
    runtime: Optional[RuntimeConfig | Mapping[str, Any]] = None
    artifacts: Optional[ArtifactConfig | Mapping[str, Any]] = None
    architecture: Optional[ArchitectureConfig | Mapping[str, Any]] = None

    # Flat aliases kept for backward compatibility.
    freq: Optional[str] = None
    lookback: Optional[int] = None
    horizon: Optional[int] = None
    device: Optional[str] = None

    warmup_epochs: Optional[int] = None
    spike_epochs: Optional[int] = None
    base_lr: Optional[float] = None
    save_dir: Optional[str] = None
    auto_save_dir: bool = True

    use_exogenous_mode: Optional[bool] = None
    use_past_exogenous: Optional[bool] = None
    use_future_exogenous: Optional[bool] = None
    models_to_run: Optional[Sequence[str]] = None
    model: Any = None
    models: Any = None

    loss_point: Any = None
    loss_quantile: Any = None
    loss: Any = None

    use_ssl_mode: Optional[str] = None
    ssl_pretrain_epochs: Optional[int] = None
    ssl_mask_ratio: Optional[float] = None
    ssl_loss_type: Optional[str] = None
    ssl_freeze_encoder_before_ft: Optional[bool] = None
    ssl_pretrained_ckpt_path: Optional[str] = None

    use_intermittent: Optional[bool] = None
    val_use_weights: Optional[bool] = None


@dataclass
class TrainResult:
    """
    Normalized training output returned by the public API.

    Fields
    - `results`: Per-model summarized result objects.
    - `requested_models`: Canonical model keys requested by the caller.
    - `save_dir`: Artifact directory used for this run.
    - `ckpt_paths`: Final supervised checkpoints keyed by canonical model key.
    - `pretrain_ckpt_paths`: Optional SSL pretraining checkpoints keyed by canonical model key.
    - `manifest_path`: Path to the generated `training_manifest.json`.
    - `primary_result_name`: Canonical model key of the single produced checkpoint.
      This is populated only when exactly one final checkpoint is produced.
    - `primary_ckpt_path`: Convenience shortcut for single-model runs.
      This is populated only when exactly one final checkpoint is produced.
    - `best_ckpt_path`: Backward-compatible alias for `primary_ckpt_path`.
      For multi-model or family runs this stays `None`; use `ckpt_paths` instead.
    - `datamodule`: Datamodule built by the API when loaders were not provided directly.
    """
    results: Dict[str, Dict[str, Any]]
    requested_models: tuple[str, ...]
    save_dir: Optional[str] = None
    ckpt_paths: Dict[str, str] = field(default_factory=dict)
    pretrain_ckpt_paths: Dict[str, str] = field(default_factory=dict)
    manifest_path: Optional[str] = None
    primary_result_name: Optional[str] = None
    primary_ckpt_path: Optional[str] = None
    best_ckpt_path: Optional[str] = None
    datamodule: Any = None


def _request_to_dict(req: TrainRequest | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(req, TrainRequest):
        payload: dict[str, Any] = {}

        if req.config_path:
            payload.update(_load_config_file(req.config_path))
        if req.config:
            payload.update(dict(req.config))

        for field_ in fields(TrainRequest):
            if field_.name in {"config", "config_path"}:
                continue
            value = getattr(req, field_.name)
            has_default = field_.default is not MISSING
            default_value = field_.default if has_default else MISSING
            if value is not None and not _is_default_value(value, default_value):
                payload[field_.name] = value

        return payload

    if isinstance(req, Mapping):
        return dict(req)

    raise TypeError(f"Unsupported train request type: {type(req)}")


def _coerce_mapping(value: Any) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if is_dataclass(value):
        out: dict[str, Any] = {}
        for field_ in fields(value):
            item = getattr(value, field_.name)
            has_default = field_.default is not MISSING
            default_value = field_.default if has_default else MISSING
            if item is not None and not _is_default_value(item, default_value):
                out[field_.name] = item
        return out
    if isinstance(value, Mapping):
        return dict(value)
    return None


_ARCHITECTURE_ALLOWED_KEYS: dict[str, set[str]] = {
    "patchtst": {
        "patch_len",
        "stride",
        "d_model",
        "n_layers",
        "d_ff",
        "dropout",
        "norm",
        "pre_norm",
        "act",
        "use_revin",
        "pe",
        "learn_pe",
        "padding_patch",
        "future_exo_fusion_mode",
        "future_exo_fusion_dropout",
    },
    "titan": {
        "d_model",
        "n_layers",
        "n_heads",
        "d_ff",
        "dropout",
        "contextual_mem_size",
        "persistent_mem_size",
        "use_revin",
        "final_clamp_nonneg",
    },
    "patchmixer": {
        "patch_len",
        "stride",
        "d_model",
        "e_layers",
        "f_out",
        "head_hidden",
        "dropout",
        "head_dropout",
        "use_revin",
        "final_nonneg",
        "expander_n_harmonics",
    },
    "exotst": {
        "patch_len",
        "stride",
        "d_model",
        "n_heads",
        "d_ff",
        "dropout",
        "attn_dropout",
        "exo_enc_layers",
        "fusion_layers",
        "endo_dec_layers",
        "exo_memory_mode",
        "exo_nan_policy",
        "use_revin",
        "subtract_last",
    },
    "timexer": {
        "patch_len",
        "d_model",
        "n_heads",
        "d_ff",
        "e_layers",
        "dropout",
        "factor",
        "activation",
        "use_norm",
    },
}


def _family_from_training_target(name: str) -> str:
    canonical = resolve_training_request_key(name)
    if canonical == "patchtst" or canonical.startswith("patchtst_"):
        return "patchtst"
    if canonical == "patchmixer" or canonical.startswith("patchmixer_"):
        return "patchmixer"
    if canonical == "titan" or canonical.startswith("titan_"):
        return "titan"
    if canonical == "exotst" or canonical.startswith("exotst_"):
        return "exotst"
    if canonical == "timexer" or canonical.startswith("timexer_"):
        return "timexer"
    raise ValueError(f"Unsupported architecture override target: {name!r}")


def _normalize_model_architecture(value: Any) -> Optional[dict[str, dict[str, Any]]]:
    mapping = _coerce_mapping(value)
    if not mapping:
        return None

    normalized: dict[str, dict[str, Any]] = {}
    for raw_key, raw_section in mapping.items():
        section = _coerce_mapping(raw_section)
        if not section:
            continue

        family = _family_from_training_target(str(raw_key))
        allowed = _ARCHITECTURE_ALLOWED_KEYS[family]
        unknown = sorted(key for key in section if key not in allowed)
        if unknown:
            allowed_list = ", ".join(sorted(allowed))
            raise ValueError(
                f"Unsupported architecture override keys for {family}: {', '.join(unknown)}. "
                f"Supported keys: {allowed_list}."
            )

        family_section = normalized.setdefault(family, {})
        for key, item in section.items():
            if item is not None:
                family_section[key] = item

    return normalized or None


def _load_config_file(path: str) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    suffix = cfg_path.suffix.lower()
    text = cfg_path.read_text(encoding="utf-8")

    if suffix == ".json":
        return json.loads(text)

    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to load yaml configs.")
        data = yaml.safe_load(text)
        return dict(data or {})

    raise ValueError(f"Unsupported config format: {cfg_path}")


def _canonical_training_target(name: str) -> str:
    return resolve_training_request_key(name)


def _coerce_models(spec: Any) -> Optional[list[str]]:
    if spec is None:
        return None

    if isinstance(spec, str):
        return [_canonical_training_target(spec)]

    if isinstance(spec, Mapping):
        name = spec.get("name") or spec.get("model_name")
        if name is None:
            raise ValueError("Model mapping must include `name`.")
        return [_canonical_training_target(name)]

    if isinstance(spec, Sequence):
        out: list[str] = []
        for item in spec:
            if isinstance(item, Mapping):
                name = item.get("name") or item.get("model_name")
            else:
                name = item
            if name is None:
                raise ValueError("Each model entry must be a string or include `name`.")
            out.append(_canonical_training_target(name))
        return out

    raise TypeError(f"Unsupported model spec type: {type(spec)}")


def _default_save_dir() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path.cwd() / "artifacts" / "training" / stamp)


def _normalize_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(raw)

    section_mappings = {
        "trainer": {
            "epochs": "epochs",
            "lr": "lr",
            "warmup_epochs": "warmup_epochs",
            "spike_epochs": "spike_epochs",
            "base_lr": "base_lr",
            "loss": "loss",
            "loss_point": "loss_point",
            "loss_quantile": "loss_quantile",
            "use_intermittent": "use_intermittent",
            "val_use_weights": "val_use_weights",
        },
        "ssl": {
            "mode": "use_ssl_mode",
            "use_ssl_mode": "use_ssl_mode",
            "pretrain_epochs": "ssl_pretrain_epochs",
            "ssl_pretrain_epochs": "ssl_pretrain_epochs",
            "mask_ratio": "ssl_mask_ratio",
            "ssl_mask_ratio": "ssl_mask_ratio",
            "loss_type": "ssl_loss_type",
            "ssl_loss_type": "ssl_loss_type",
            "freeze_encoder_before_ft": "ssl_freeze_encoder_before_ft",
            "ssl_freeze_encoder_before_ft": "ssl_freeze_encoder_before_ft",
            "pretrained_ckpt_path": "ssl_pretrained_ckpt_path",
            "ssl_pretrained_ckpt_path": "ssl_pretrained_ckpt_path",
        },
        "runtime": {
            "device": "device",
        },
        "artifacts": {
            "save_dir": "save_dir",
            "auto_save_dir": "auto_save_dir",
            "save_root": "save_root",
        },
    }
    for section_name, key_mapping in section_mappings.items():
        section = _coerce_mapping(payload.get(section_name))
        if not section:
            continue
        for raw_key, value in section.items():
            payload.setdefault(key_mapping.get(raw_key, raw_key), value)

    architecture_cfg = payload.get("architecture")
    if architecture_cfg is None:
        architecture_cfg = payload.get("model_architecture")
    if architecture_cfg is None:
        architecture_cfg = payload.get("architecture_overrides")
    normalized_architecture = _normalize_model_architecture(architecture_cfg)
    if normalized_architecture is not None:
        payload["model_architecture"] = normalized_architecture

    if "frequency" in payload and "freq" not in payload:
        payload["freq"] = payload["frequency"]

    if "use_exogenous" in payload and "use_exogenous_mode" not in payload:
        payload["use_exogenous_mode"] = payload["use_exogenous"]

    if "lr" in payload and "base_lr" not in payload:
        payload["base_lr"] = payload["lr"]

    if (
        "epochs" in payload
        and "warmup_epochs" not in payload
        and "spike_epochs" not in payload
    ):
        payload["warmup_epochs"] = payload["epochs"]

    if "save_root" in payload and "save_dir" not in payload:
        payload["save_dir"] = payload["save_root"]

    if "models_to_run" not in payload:
        models_spec = payload.get("models")
        if models_spec is None:
            models_spec = payload.get("model")
        if models_spec is not None:
            payload["models_to_run"] = _coerce_models(models_spec)
    elif payload["models_to_run"] is not None:
        payload["models_to_run"] = _coerce_models(payload["models_to_run"])

    if payload.get("models_to_run") is not None:
        payload["models_to_run"] = expand_training_targets(payload["models_to_run"])

    if payload.get("device") is None:
        payload["device"] = default_device()
    else:
        payload["device"] = resolve_device(str(payload["device"]))

    if payload.get("save_dir") is None and bool(payload.get("auto_save_dir", True)):
        payload["save_dir"] = _default_save_dir()

    return payload


def _merged_data_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    data_cfg = payload.get("data")
    if isinstance(data_cfg, Mapping):
        merged = dict(data_cfg)
    elif is_dataclass(data_cfg):
        merged = _coerce_mapping(data_cfg) or {}
    elif data_cfg is None:
        merged = {}
    else:
        merged = {"data": data_cfg}

    for key in (
        "lookback",
        "horizon",
        "freq",
        "batch_size",
        "val_ratio",
        "shuffle",
        "seed",
        "use_exogenous_mode",
        "backend",
        "id_col",
        "date_col",
        "y_col",
        "past_exo_cont_cols",
        "past_exo_cat_cols",
        "future_exo_cont_cols",
        "fill_missing",
        "target_back_steps",
        "future_exo_cb",
        "part_future_exo_fn",
        "date_indexer",
        "build_cat_indexer_from",
        "cat_indexer_target_col",
        "split_mode",
        "path",
        "df",
    ):
        value = payload.get(key)
        if value is not None and key not in merged:
            merged[key] = value

    return _materialize_data_payload(merged)


def _build_loader_runtime_kwargs(
    data_cfg: Mapping[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}

    for key in ("batch_size", "num_workers", "pin_memory", "persistent_workers", "prefetch_factor"):
        value = data_cfg.get(key)
        if value is not None:
            kwargs[key] = value

    if stage == "train":
        shuffle = data_cfg.get("shuffle")
        if shuffle is not None:
            kwargs["shuffle"] = shuffle

    drop_last = data_cfg.get("drop_last")
    if drop_last is not None:
        kwargs["drop_last"] = drop_last

    return kwargs


def _call_loader_method(datamodule: Any, method_name: str, kwargs: Mapping[str, Any]) -> Any:
    method = getattr(datamodule, method_name)
    signature = inspect.signature(method)

    accepted: dict[str, Any] = {}
    for name, value in kwargs.items():
        if name in signature.parameters:
            accepted[name] = value

    return method(**accepted)


def _resolve_loaders(payload: dict[str, Any]) -> tuple[Any, Any, Any]:
    train_loader = payload.get("train_loader")
    val_loader = payload.get("val_loader")
    datamodule = None

    if (train_loader is None) != (val_loader is None):
        raise ValueError("`train_loader` and `val_loader` must be provided together.")

    if train_loader is None:
        data_cfg = _merged_data_config(payload)
        if not data_cfg:
            raise ValueError("Provide either (`train_loader`, `val_loader`) or a `data` config.")

        datamodule = build_datamodule(data_cfg)
        train_loader = _call_loader_method(
            datamodule,
            "get_train_loader",
            _build_loader_runtime_kwargs(data_cfg, stage="train"),
        )
        val_loader = _call_loader_method(
            datamodule,
            "get_val_loader",
            _build_loader_runtime_kwargs(data_cfg, stage="val"),
        )

        for key in (
            "lookback",
            "horizon",
            "freq",
            "use_exogenous_mode",
            "use_past_exogenous",
            "use_future_exogenous",
        ):
            if payload.get(key) is None and data_cfg.get(key) is not None:
                payload[key] = data_cfg[key]

    return train_loader, val_loader, datamodule


def _make_result(
    *,
    request_payload: Mapping[str, Any],
    results: Dict[str, Dict[str, Any]],
    requested_models: Sequence[str],
    save_dir: Optional[str],
    datamodule: Any,
) -> TrainResult:
    summarized_results = _canonicalize_result_keys(summarize_training_results(results))
    ckpt_paths = {
        name: str(info["ckpt_path"])
        for name, info in summarized_results.items()
        if isinstance(info, Mapping) and info.get("ckpt_path")
    }
    pretrain_ckpt_paths = {
        name: str(info["pretrain_ckpt_path"])
        for name, info in summarized_results.items()
        if isinstance(info, Mapping) and info.get("pretrain_ckpt_path")
    }

    if len(ckpt_paths) == 1:
        primary_result_name = next(iter(ckpt_paths))
        primary_ckpt_path = ckpt_paths[primary_result_name]
        best_ckpt_path = primary_ckpt_path
    else:
        primary_result_name = None
        primary_ckpt_path = None
        best_ckpt_path = None
    manifest_path = None

    if save_dir is not None:
        manifest_request = _summarize_request_for_manifest(request_payload)

        manifest_path = save_training_manifest(
            save_dir,
            request=manifest_request,
            results=summarized_results,
            extra_meta={
                "requested_models": list(requested_models),
                "primary_result_name": primary_result_name,
                "primary_ckpt_path": primary_ckpt_path,
                "best_ckpt_path": best_ckpt_path,
            },
        )

    return TrainResult(
        results=summarized_results,
        requested_models=tuple(requested_models),
        save_dir=save_dir,
        ckpt_paths=ckpt_paths,
        pretrain_ckpt_paths=pretrain_ckpt_paths,
        manifest_path=manifest_path,
        primary_result_name=primary_result_name,
        primary_ckpt_path=primary_ckpt_path,
        best_ckpt_path=best_ckpt_path,
        datamodule=datamodule,
    )


def _canonicalize_result_keys(results: Mapping[str, Any]) -> dict[str, Any]:
    canonicalized: dict[str, Any] = {}

    for result_name, info in results.items():
        if isinstance(info, Mapping):
            entry = dict(info)
            target_key = str(entry.get("model_key") or result_name)
            entry.setdefault("result_name", str(result_name))
        else:
            entry = info
            target_key = str(result_name)

        canonicalized[target_key] = entry

    return canonicalized


def _summarize_request_for_manifest(request_payload: Mapping[str, Any]) -> dict[str, Any]:
    manifest_request = dict(request_payload)
    manifest_request.pop("train_loader", None)
    manifest_request.pop("val_loader", None)
    manifest_request.pop("loss_point", None)
    manifest_request.pop("loss_quantile", None)
    manifest_request.pop("loss", None)
    manifest_request.pop("df", None)

    data_cfg = manifest_request.get("data")
    if isinstance(data_cfg, Mapping):
        data_summary = dict(data_cfg)
        if "df" in data_summary:
            df_obj = data_summary.pop("df")
            shape = getattr(df_obj, "shape", None)
            data_summary["df_summary"] = {
                "type": type(df_obj).__name__,
                "shape": list(shape) if shape is not None else None,
            }
        manifest_request["data"] = data_summary
    elif data_cfg is not None and not isinstance(data_cfg, (str, int, float, bool)):
        manifest_request["data"] = {"type": type(data_cfg).__name__}

    return manifest_request


def _validate_training_request(
    *,
    payload: Mapping[str, Any],
    train_loader: Any,
    requested_models: Sequence[str],
) -> None:
    freq = payload.get("freq")
    lookback = payload.get("lookback")
    horizon = payload.get("horizon")
    if freq is None or lookback is None or horizon is None:
        return

    freq_value = str(freq).strip().lower()
    if freq_value not in {"weekly", "monthly", "daily", "hourly"}:
        raise ValueError(
            f"Unsupported freq={freq!r}. Use one of: 'weekly', 'monthly', 'daily', 'hourly'."
        )

    lookback_i = int(lookback)
    horizon_i = int(horizon)
    if lookback_i <= 0 or horizon_i <= 0:
        raise ValueError("`lookback` and `horizon` must be positive integers.")

    freq_spec = get_freq_spec(freq_value)
    architecture = payload.get("model_architecture") or {}
    patch_requirements: dict[str, int] = {}
    for key in requested_models:
        family = _family_from_training_target(key)
        if family not in {"patchtst", "patchmixer", "exotst", "timexer"}:
            continue
        family_arch = architecture.get(family) or {}
        patch_requirements[key] = int(family_arch.get("patch_len", freq_spec.patch_len))

    if patch_requirements:
        required_patch_len = max(patch_requirements.values())
        if lookback_i < required_patch_len:
            model_list = ", ".join(
                f"{name}(patch_len={patch_requirements[name]})" for name in sorted(patch_requirements)
            )
            raise ValueError(
                "Invalid training request: "
                f"lookback={lookback_i} is too short for {model_list} at freq='{freq_value}'. "
                f"Set lookback >= {required_patch_len}."
            )

    if "timexer_base" in requested_models:
        timexer_patch_len = patch_requirements.get("timexer_base", freq_spec.patch_len)
        if lookback_i % timexer_patch_len != 0:
            raise ValueError(
                "Invalid training request for timexer_base: "
                f"TimeXer requires non-overlapping patches, so lookback={lookback_i} "
                f"must be divisible by patch_len={timexer_patch_len}."
            )

    has_future_exo, future_exo_dim = infer_future_exo_spec_from_loader(
        train_loader,
        lookback=lookback_i,
        horizon=horizon_i,
    )
    past_cont_dim, past_cat_dim = infer_past_exo_dim_from_loader_for_exotst(
        train_loader,
        lookback=lookback_i,
        horizon=horizon_i,
    )

    if "timexer_base" in requested_models:
        if not bool(payload.get("use_exogenous_mode", False)):
            raise ValueError(
                "Invalid training request for timexer_base: TimeXer requires use_exogenous_mode=True."
            )

        if int(future_exo_dim) > 0:
            raise ValueError(
                "Invalid training request for timexer_base: TimeXer v1 does not support future exogenous inputs. "
                "Remove `future_exo_cont_cols` or `future_exo_cb`."
            )

        if int(past_cont_dim) <= 0:
            raise ValueError(
                "Invalid training request for timexer_base: TimeXer requires past continuous exogenous inputs. "
                "Provide `past_exo_cont_cols`."
            )

        if int(past_cat_dim) > 0:
            raise ValueError(
                "Invalid training request for timexer_base: TimeXer v1 does not consume past categorical "
                "exogenous inputs. Encode them into continuous channels first."
            )

    if "exotst_base" not in requested_models:
        return

    if not bool(payload.get("use_exogenous_mode", False)):
        raise ValueError(
            "Invalid training request for exotst_base: ExoTST requires use_exogenous_mode=True."
        )

    if not has_future_exo or future_exo_dim <= 0:
        raise ValueError(
            "Invalid training request for exotst_base: ExoTST requires future exogenous inputs. "
            "Provide `future_exo_cont_cols`, a loader-supplied future exo tensor, or `future_exo_cb`."
        )

    if int(past_cont_dim) <= 0:
        raise ValueError(
            "Invalid training request for exotst_base: ExoTST requires past continuous exogenous inputs. "
            "Provide `past_exo_cont_cols` (or a loader that emits `pe_cont`)."
        )


def train(req: TrainRequest | Mapping[str, Any]) -> TrainResult:
    """
    Public training entrypoint.

    Supported styles
    - `train(TrainRequest(...))`
    - `train({...})`

    Preferred request shape
    - `data`: `DataRequest` or mapping with `window / columns / exogenous / loader`
    - `models`: model family or artifact keys such as `patchtst`, `patchtst_base`, `exotst_base`
    - `trainer`: supervised optimization config
    - `ssl`: optional PatchTST SSL config
    - `runtime`: runtime device config
    - `artifacts`: output checkpoint/manifest config

    Result contract
    - For single-model runs, `primary_result_name`, `primary_ckpt_path`, and `best_ckpt_path`
      are populated for convenience.
    - For multi-model or family runs, those convenience fields stay `None` and callers should
      read the produced checkpoints from `ckpt_paths`.

    Minimal dataclass example
    ```python
    train(
        TrainRequest(
            data=DataRequest(
                df=df,
                window=DataWindowConfig(lookback=52, horizon=12, freq="weekly"),
            ),
            models=["patchtst_base"],
            trainer=TrainerConfig(epochs=5, lr=1e-3),
        )
    )
    ```
    """
    payload = _normalize_payload(_request_to_dict(req))
    train_loader, val_loader, datamodule = _resolve_loaders(payload)

    freq = payload.get("freq")
    lookback = payload.get("lookback")
    horizon = payload.get("horizon")
    if freq is None or lookback is None or horizon is None:
        raise ValueError("`freq`, `lookback`, and `horizon` are required for training.")

    requested_models = payload.get("models_to_run") or expand_training_targets(None)
    _validate_training_request(
        payload=payload,
        train_loader=train_loader,
        requested_models=requested_models,
    )
    use_ssl_mode = payload.get("use_ssl_mode") or "sl_only"
    ssl_pretrain_epochs = payload.get("ssl_pretrain_epochs")
    ssl_mask_ratio = payload.get("ssl_mask_ratio")
    ssl_loss_type = payload.get("ssl_loss_type")
    ssl_freeze_encoder_before_ft = payload.get("ssl_freeze_encoder_before_ft")
    use_intermittent = payload.get("use_intermittent")
    val_use_weights = payload.get("val_use_weights")

    results = run_total_train(
        train_loader,
        val_loader,
        freq=str(freq),
        lookback=int(lookback),
        horizon=int(horizon),
        device=str(payload["device"]),
        warmup_epochs=payload.get("warmup_epochs"),
        spike_epochs=payload.get("spike_epochs"),
        base_lr=payload.get("base_lr"),
        save_dir=payload.get("save_dir"),
        use_exogenous_mode=bool(payload.get("use_exogenous_mode", False)),
        use_past_exogenous=bool(payload.get("use_past_exogenous", True)),
        use_future_exogenous=bool(payload.get("use_future_exogenous", True)),
        models_to_run=requested_models,
        model_architecture=payload.get("model_architecture"),
        loss_point=payload.get("loss_point"),
        loss_quantile=payload.get("loss_quantile"),
        loss=payload.get("loss"),
        use_ssl_mode=str(use_ssl_mode),
        ssl_pretrain_epochs=int(ssl_pretrain_epochs if ssl_pretrain_epochs is not None else 10),
        ssl_mask_ratio=float(ssl_mask_ratio if ssl_mask_ratio is not None else 0.3),
        ssl_loss_type=str(ssl_loss_type or "mse"),
        ssl_freeze_encoder_before_ft=bool(
            ssl_freeze_encoder_before_ft if ssl_freeze_encoder_before_ft is not None else False
        ),
        ssl_pretrained_ckpt_path=payload.get("ssl_pretrained_ckpt_path"),
        use_intermittent=bool(use_intermittent if use_intermittent is not None else True),
        val_use_weights=bool(val_use_weights if val_use_weights is not None else True),
    )

    return _make_result(
        request_payload=payload,
        results=results,
        requested_models=requested_models,
        save_dir=payload.get("save_dir"),
        datamodule=datamodule,
    )


__all__ = [
    "ArtifactConfig",
    "ArchitectureConfig",
    "ExoTSTArchitectureConfig",
    "PatchMixerArchitectureConfig",
    "PatchTSTArchitectureConfig",
    "RuntimeConfig",
    "SSLConfig",
    "TimexerArchitectureConfig",
    "TitanArchitectureConfig",
    "TrainRequest",
    "TrainResult",
    "TrainerConfig",
    "train",
]

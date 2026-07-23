from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch

from modeling_module._internal.checkpoint_runtime import (
    _drop_revin_buffers,
    _extract_cfg_obj,
    _extract_state_dict,
    _partial_load_with_shape_filter,
)
from modeling_module._internal.device_runtime import default_device, resolve_device
from modeling_module._internal.inference_runtime import DMSForecaster, _unpack_batch_for_export
from modeling_module._internal.model_registry import (
    family_for_artifact_key,
    get_model_builder,
    infer_artifact_model_key_from_checkpoint,
)


def _config_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _requires_exact_unversioned_patchmixer_restore(
    ckpt: Mapping[str, Any],
    model_key: str,
) -> bool:
    """Prevent silent partial restores for unversioned project checkpoints."""
    if ckpt.get("format_version") is not None or not model_key.startswith("patchmixer"):
        return False
    model_class = "".join(
        ch for ch in str(ckpt.get("model_class", "")).casefold() if ch.isalnum()
    )
    return model_class in {"basemodel", "quantilemodel"}


def _load_single_model(
    ckpt_path: str,
    *,
    device: Optional[str] = None,
    strict: bool = False,
) -> tuple[torch.nn.Module, Any, str]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_obj = _extract_cfg_obj(ckpt)
    if cfg_obj is None:
        raise ValueError(f"No config found in checkpoint: {ckpt_path}")

    model_key = infer_artifact_model_key_from_checkpoint(ckpt, ckpt_path=ckpt_path)
    builder = get_model_builder(model_key)
    model = builder(cfg_obj)

    state_dict = dict(_extract_state_dict(ckpt))
    if "patchtst" in model.__class__.__name__.lower():
        state_dict = _drop_revin_buffers(state_dict)

    exact_legacy_restore = _requires_exact_unversioned_patchmixer_restore(
        ckpt,
        model_key,
    )
    try:
        model.load_state_dict(state_dict, strict=strict or exact_legacy_restore)
    except RuntimeError as exc:
        if exact_legacy_restore:
            raise ValueError(
                "Unsupported pre-version PatchMixer checkpoint: its BaseModel/QuantileModel "
                "state schema cannot be restored exactly by the maintained load-only model. "
                "Recreate the historical source environment and resave it in checkpoint "
                "format v3 before loading it through the public API. "
                f"Checkpoint: {ckpt_path}"
            ) from exc
        if strict:
            raise
        _partial_load_with_shape_filter(model, state_dict)

    resolved_device = resolve_device(device)
    model.to(resolved_device).eval()
    return model, cfg_obj, model_key


def _normalize_predict_payload(batch: Any) -> dict[str, Any]:
    if torch.is_tensor(batch):
        return {"x": batch}

    if isinstance(batch, Mapping):
        payload = dict(batch)
        if "x" not in payload and "x_init" in payload:
            payload["x"] = payload.pop("x_init")
        if "future_exo_batch" not in payload and "future_exo" in payload:
            payload["future_exo_batch"] = payload.pop("future_exo")
        return payload

    if isinstance(batch, (tuple, list)):
        unpacked = _unpack_batch_for_export(batch)
        return {
            "x": unpacked["x"],
            "part_ids": unpacked["part_ids"],
            "future_exo_batch": unpacked["future_exo"],
            "past_exo_cont": unpacked["past_exo_cont"],
            "past_exo_cat": unpacked["past_exo_cat"],
        }

    raise TypeError(f"Unsupported prediction input type: {type(batch)}")


@dataclass
class LoadedPredictor:
    """
    Loaded checkpoint wrapper exposed by the public inference API.

    The object is callable and delegates to `predict(...)` with the checkpoint configuration
    already restored.
    """
    model: torch.nn.Module
    config: Any
    model_key: str
    ckpt_path: str
    device: str
    forecaster_kwargs: Optional[dict[str, Any]] = None

    @property
    def family_key(self) -> str:
        return family_for_artifact_key(self.model_key)

    @property
    def default_horizon(self) -> Optional[int]:
        horizon = _config_get(self.config, "horizon")
        if horizon is None:
            horizon = getattr(self.model, "horizon", None)
        return int(horizon) if horizon is not None else None

    def predict(self, batch: Any, **kwargs) -> Any:
        """
        Run inference for a normalized batch payload.

        Accepted input shapes
        - tensor: treated as `x`
        - mapping: expects `x` (or `x_init`) and optional exogenous fields
        - tuple/list: normalized via the library batch unpacker
        """
        payload = _normalize_predict_payload(batch)
        x = payload.pop("x", None)
        if x is None:
            raise ValueError("Prediction input must include `x` or `x_init`.")

        horizon = kwargs.pop("horizon", None)
        if horizon is None:
            horizon = payload.pop("horizon", None)
        if horizon is None:
            horizon = self.default_horizon
        if horizon is None:
            raise ValueError("`horizon` is required for prediction.")

        runtime_device = kwargs.pop("device", None) or self.device
        missing = object()

        def _resolve_forward_arg(name: str, default: Any = missing) -> Any:
            # kwargs에 명시적으로 들어온 값이 있으면 그 값을 우선 사용하고,
            # payload의 동일 키는 소비만 하고 무시한다.
            # 이렇게 해야 wrapper 기본값과 caller kwargs가 중복 전달되지 않는다.
            value = kwargs.pop(name, missing)
            if value is not missing:
                payload.pop(name, None)
                return value
            if default is missing:
                return payload.pop(name, None)
            return payload.pop(name, default)

        mode = _resolve_forward_arg("mode", "eval")
        part_ids = _resolve_forward_arg("part_ids", None)
        past_exo_cont = _resolve_forward_arg("past_exo_cont", None)
        past_exo_cat = _resolve_forward_arg("past_exo_cat", None)
        future_exo_batch = _resolve_forward_arg("future_exo_batch", None)
        future_exo_cb = _resolve_forward_arg("future_exo_cb", None)
        extension_policy = _resolve_forward_arg("extension_policy", None)
        tail_model = _resolve_forward_arg("tail_model", "exp")
        tail_fit_window = _resolve_forward_arg("tail_fit_window", 18)
        tail_anchor = _resolve_forward_arg("tail_anchor", "mean_last_3")
        state_prior = _resolve_forward_arg("state_prior", None)
        is_IMS = bool(_resolve_forward_arg("is_IMS", True))
        is_linear_decay = bool(_resolve_forward_arg("is_linear_decay", True))

        forecaster = DMSForecaster(self.model, **(self.forecaster_kwargs or {}))
        return forecaster.predict(
            x,
            horizon=int(horizon),
            device=runtime_device,
            mode=mode,
            part_ids=part_ids,
            past_exo_cont=past_exo_cont,
            past_exo_cat=past_exo_cat,
            future_exo_batch=future_exo_batch,
            future_exo_cb=future_exo_cb,
            extension_policy=extension_policy,
            tail_model=tail_model,
            tail_fit_window=tail_fit_window,
            tail_anchor=tail_anchor,
            state_prior=state_prior,
            is_IMS=is_IMS,
            is_linear_decay=is_linear_decay,
            **kwargs,
        )

    __call__ = predict


def load_predictor(
    ckpt_path: str,
    *,
    device: Optional[str] = None,
    strict: bool = False,
    forecaster_kwargs: Optional[dict[str, Any]] = None,
) -> LoadedPredictor:
    """
    Load a single checkpoint and expose it as a callable predictor.

    This is the preferred public inference entrypoint when a checkpoint will be reused
    for multiple prediction calls.
    """
    resolved_device = resolve_device(device)
    model, cfg, model_key = _load_single_model(ckpt_path, device=resolved_device, strict=strict)
    return LoadedPredictor(
        model=model,
        config=cfg,
        model_key=model_key,
        ckpt_path=ckpt_path,
        device=resolved_device,
        forecaster_kwargs=forecaster_kwargs,
    )


def predict(ckpt_path: str, batch: Any, *, device: Optional[str] = None, **kwargs) -> Any:
    """
    Convenience helper that combines `load_predictor(...)` and a single prediction call.
    """
    predictor = load_predictor(ckpt_path, device=device)
    return predictor(batch, device=device, **kwargs)


__all__ = [
    "LoadedPredictor",
    "load_predictor",
    "predict",
]

"""Public training-artifact inference API for ICL-enabled checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import polars as pl
import torch

from modeling_module._internal.device_runtime import resolve_device
from modeling_module._internal.icl_runtime import (
    AutoTimesICLAdapter,
    ICLEpisodeDataModule,
    ICLSplit,
    SELLMICLAdapter,
    read_icl_episode_artifact,
)
from modeling_module.api.infer import load_predictor


@dataclass(frozen=True)
class ICLForecastRuntimeConfig:
    batch_size: int = 32
    num_workers: int = 0
    device: str | None = None
    pin_memory: bool = False
    llm_local_path: str | Path | None = None

    def __post_init__(self) -> None:
        if int(self.batch_size) <= 0:
            raise ValueError("ICL forecast batch_size must be positive.")
        if int(self.num_workers) < 0:
            raise ValueError("ICL forecast num_workers must be non-negative.")


@dataclass(frozen=True)
class ICLForecastRequest:
    checkpoint_path: str | Path
    episode_artifact_dir: str | Path
    expected_model_key: str | None = None
    split: ICLSplit | str = ICLSplit.TEST
    runtime: ICLForecastRuntimeConfig = field(
        default_factory=ICLForecastRuntimeConfig
    )


@dataclass(frozen=True)
class ICLForecastResult:
    predictions: pl.DataFrame
    model_key: str
    manifest_hash: str
    episode_file_sha256: str
    split: str


def _forecast_batch(model_key: str, model: torch.nn.Module, batch) -> torch.Tensor:
    if model_key == "autotimes_base":
        inputs = AutoTimesICLAdapter().adapt(batch)
        return model.forward_icl(
            inputs.packed_context,
            prompt_mask=inputs.prompt_mask,
            packed_exogenous=inputs.packed_exogenous,
            query_target_exogenous=inputs.query_target_exogenous,
        )
    if model_key == "sellm_base":
        inputs = SELLMICLAdapter().adapt(batch)
        return model.forward_icl(
            demonstration_contexts=inputs.demonstration_contexts,
            demonstration_targets=inputs.demonstration_targets,
            query_context=inputs.query_context,
            prompt_mask=inputs.prompt_mask,
            demonstration_context_exogenous=inputs.demonstration_context_exogenous,
            demonstration_target_exogenous=inputs.demonstration_target_exogenous,
            query_context_exogenous=inputs.query_context_exogenous,
            query_target_exogenous=inputs.query_target_exogenous,
        )
    raise ValueError(f"Model {model_key!r} does not expose the ICL v1 execution contract.")


def _config_value(config, name: str):
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def forecast_icl(request: ICLForecastRequest) -> ICLForecastResult:
    """Forecast sealed ICL episodes with a strict ICL-enabled checkpoint."""

    split = ICLSplit(request.split)
    bundle, artifact_receipt = read_icl_episode_artifact(
        request.episode_artifact_dir
    )
    device = resolve_device(request.runtime.device)
    predictor = load_predictor(
        str(request.checkpoint_path),
        device=device,
        strict=True,
        config_overrides=(
            None
            if request.runtime.llm_local_path is None
            else {"llm_local_path": request.runtime.llm_local_path}
        ),
    )
    if request.expected_model_key is not None:
        expected = str(request.expected_model_key).strip()
        if predictor.model_key != expected:
            raise ValueError(
                "ICL checkpoint model key mismatch: "
                f"expected={expected!r}, actual={predictor.model_key!r}."
            )
    if not bool(_config_value(predictor.config, "icl_enabled")):
        raise ValueError("ICL forecast requires a checkpoint with icl_enabled=True.")
    configured_schema_hash = _config_value(
        predictor.config,
        "icl_exogenous_schema_hash",
    )
    artifact_schema = bundle.manifest.exogenous_schema
    artifact_schema_hash = None if artifact_schema is None else artifact_schema.fingerprint
    if configured_schema_hash != artifact_schema_hash:
        raise ValueError("ICL checkpoint and Episode exogenous schema hash differ.")

    module = ICLEpisodeDataModule(
        bundle,
        batch_size=int(request.runtime.batch_size),
        num_workers=int(request.runtime.num_workers),
        pin_memory=bool(request.runtime.pin_memory),
    )
    loader = module.loader(split, shuffle=False)
    split_episodes = bundle.for_split(split)
    if split is ICLSplit.INFERENCE and any(
        item.query_target_observed for item in split_episodes
    ):
        raise ValueError("ICL inference split contains an observed future target.")
    episode_by_id = {episode.episode_id: episode for episode in bundle.for_split(split)}
    rows: list[dict[str, object]] = []
    predictor.model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            prediction = _forecast_batch(
                predictor.model_key,
                predictor.model,
                batch,
            )
            if prediction.shape != batch.query_target.shape:
                raise ValueError(
                    "ICL prediction shape must match the sealed query target: "
                    f"{tuple(prediction.shape)} != {tuple(batch.query_target.shape)}."
                )
            if not torch.isfinite(prediction).all():
                raise RuntimeError("ICL checkpoint produced a non-finite prediction.")
            values = prediction.detach().cpu()
            for batch_index, episode_id in enumerate(batch.episode_ids):
                episode = episode_by_id[episode_id]
                for horizon_step, forecast_week in enumerate(
                    episode.query_target.weeks
                ):
                    rows.append(
                        {
                            "episode_id": episode_id,
                            "series_id": episode.series_id,
                            "model_key": predictor.model_key,
                            "split": split.value,
                            "forecast_origin": episode.origin_week,
                            "forecast_week": int(forecast_week),
                            "horizon_step": int(horizon_step),
                            "point": float(values[batch_index, horizon_step, 0]),
                            "manifest_hash": bundle.manifest.manifest_hash,
                        }
                    )

    return ICLForecastResult(
        predictions=pl.DataFrame(rows),
        model_key=predictor.model_key,
        manifest_hash=bundle.manifest.manifest_hash,
        episode_file_sha256=artifact_receipt.episode_file_sha256,
        split=split.value,
    )


__all__ = [
    "ICLForecastRequest",
    "ICLForecastResult",
    "ICLForecastRuntimeConfig",
    "forecast_icl",
]

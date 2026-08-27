"""Shared training loop for models that consume sealed ICL episode batches."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TYPE_CHECKING, Callable, Iterable, Literal, Mapping

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from modeling_module.data_loader.icl_episode_data_module import ICLBatch
    from modeling_module.icl.contracts import ICLEpisodeBundle


ICLForward = Callable[[torch.nn.Module, "ICLBatch"], torch.Tensor]


@dataclass(frozen=True)
class ICLTrainerConfig:
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    max_grad_norm: float | None = None
    training_mode: Literal["qualification", "production_refit"] = "qualification"

    def __post_init__(self) -> None:
        if int(self.epochs) <= 0:
            raise ValueError("ICL trainer epochs must be positive.")
        if not math.isfinite(float(self.lr)) or float(self.lr) <= 0.0:
            raise ValueError("ICL trainer learning rate must be finite and positive.")
        if not math.isfinite(float(self.weight_decay)) or float(self.weight_decay) < 0.0:
            raise ValueError("ICL trainer weight decay must be finite and non-negative.")
        if self.max_grad_norm is not None:
            value = float(self.max_grad_norm)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("max_grad_norm must be finite and positive when provided.")
        if self.training_mode not in {"qualification", "production_refit"}:
            raise ValueError(
                "ICL training_mode must be 'qualification' or 'production_refit'."
            )


@dataclass(frozen=True)
class ICLTrainingResult:
    model: torch.nn.Module
    best_validation_loss: float | None
    final_train_loss: float
    epochs_completed: int
    epoch_history: tuple[dict[str, float | int | None], ...] = ()
    training_mode: str = "qualification"
    validation_enabled: bool = True
    state_selection: str = "best_validation"


@dataclass(frozen=True)
class _ICLEpochStats:
    loss: float
    mae: float
    wape: float


def _run_epoch(
    model: torch.nn.Module,
    loader: Iterable["ICLBatch"],
    *,
    forward: ICLForward,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_grad_norm: float | None,
) -> _ICLEpochStats:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    total_absolute_error = 0.0
    total_absolute_target = 0.0
    total_points = 0

    grad_context = torch.enable_grad() if training else torch.inference_mode()
    with grad_context:
        for batch_index, batch in enumerate(loader, start=1):
            from modeling_module.data_loader.icl_episode_data_module import ICLBatch

            if not isinstance(batch, ICLBatch):
                raise TypeError("ICL trainer requires ICLEpisodeDataModule batches.")
            if not bool(batch.query_target_observed.all()):
                raise ValueError(
                    "ICL training cannot consume inference episodes with unobserved targets."
                )
            batch = batch.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = forward(model, batch)
            if prediction.shape != batch.query_target.shape:
                raise ValueError(
                    "ICL prediction shape must match query target: "
                    f"{tuple(prediction.shape)} != {tuple(batch.query_target.shape)}."
                )
            if not torch.isfinite(prediction).all():
                raise RuntimeError(
                    "ICL model produced a non-finite prediction at "
                    f"batch_index={batch_index}."
                )
            loss = F.mse_loss(prediction, batch.query_target)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"ICL training loss became non-finite at batch_index={batch_index}."
                )
            if training:
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in model.parameters() if parameter.requires_grad],
                        float(max_grad_norm),
                    )
                optimizer.step()
            batch_size = int(batch.query_target.shape[0])
            total_loss += float(loss.detach()) * batch_size
            total_examples += batch_size
            total_absolute_error += float(
                torch.abs(prediction.detach() - batch.query_target).sum()
            )
            total_absolute_target += float(torch.abs(batch.query_target).sum())
            total_points += int(batch.query_target.numel())

    if total_examples == 0:
        raise ValueError("ICL trainer received an empty loader.")
    return _ICLEpochStats(
        loss=total_loss / total_examples,
        mae=total_absolute_error / total_points,
        wape=(
            total_absolute_error / total_absolute_target
            if total_absolute_target > 0.0
            else 0.0
        ),
    )


def fit_icl_model(
    model: torch.nn.Module,
    train_loader: Iterable["ICLBatch"],
    val_loader: Iterable["ICLBatch"] | None,
    *,
    forward: ICLForward,
    config: ICLTrainerConfig | None = None,
) -> ICLTrainingResult:
    """Fit one ICL model under qualification or final-epoch refit semantics."""

    cfg = config or ICLTrainerConfig()
    production_refit = cfg.training_mode == "production_refit"
    if production_refit and val_loader is not None:
        raise ValueError("ICL production_refit requires val_loader=None.")
    device = torch.device(cfg.device)
    model = model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("ICL model has no trainable parameters.")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    final_train_loss = float("nan")
    epoch_history: list[dict[str, float | int | None]] = []
    for epoch in range(1, int(cfg.epochs) + 1):
        train_stats = _run_epoch(
            model,
            train_loader,
            forward=forward,
            device=device,
            optimizer=optimizer,
            max_grad_norm=cfg.max_grad_norm,
        )
        final_train_loss = train_stats.loss
        validation_stats: _ICLEpochStats | None = None
        selection_loss = final_train_loss
        if val_loader is not None:
            validation_stats = _run_epoch(
                model,
                val_loader,
                forward=forward,
                device=device,
                optimizer=None,
                max_grad_norm=None,
            )
            selection_loss = validation_stats.loss
        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_mae": train_stats.mae,
                "train_wape": train_stats.wape,
                "validation_loss": (
                    None if validation_stats is None else validation_stats.loss
                ),
                "validation_mae": (
                    None if validation_stats is None else validation_stats.mae
                ),
                "validation_wape": (
                    None if validation_stats is None else validation_stats.wape
                ),
            }
        )
        if not production_refit and selection_loss < best_loss:
            best_loss = selection_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if production_refit:
        state_selection = "final_epoch"
    else:
        if best_state is None:
            raise RuntimeError("ICL trainer did not produce a model state.")
        model.load_state_dict(best_state)
        state_selection = (
            "best_validation" if val_loader is not None else "best_train"
        )
    model.eval()
    return ICLTrainingResult(
        model=model,
        best_validation_loss=(best_loss if val_loader is not None else None),
        final_train_loss=final_train_loss,
        epochs_completed=int(cfg.epochs),
        epoch_history=tuple(epoch_history),
        training_mode=cfg.training_mode,
        validation_enabled=val_loader is not None,
        state_selection=state_selection,
    )


def save_icl_production_checkpoint(
    result: ICLTrainingResult,
    path: str | Path,
    *,
    model_key: str,
    bundle: "ICLEpisodeBundle",
    trainer_config: ICLTrainerConfig,
    random_seed: int,
    data_cutoff: int,
    eligible_series_count: int,
    backbone_contract: Mapping[str, Any],
) -> Path:
    """Save a final-epoch ICL refit with its data and Qwen identities."""

    from modeling_module.data_loader.temporal import normalize_period_key
    from modeling_module.icl.contracts import ICLSplit
    from modeling_module.utils.checkpoint import save_model

    if trainer_config.training_mode != "production_refit":
        raise ValueError("ICL production checkpoint requires production_refit mode.")
    if (
        result.training_mode != "production_refit"
        or result.validation_enabled
        or result.state_selection != "final_epoch"
    ):
        raise ValueError("ICL training result is not a final-epoch production refit.")
    if not bundle.episodes or any(
        item.split is not ICLSplit.TRAIN or not item.query_target_observed
        for item in bundle.episodes
    ):
        raise ValueError(
            "ICL production refit bundle must contain observed train episodes only."
        )
    if int(eligible_series_count) <= 0:
        raise ValueError("eligible_series_count must be positive.")
    if bundle.manifest.series_count != int(eligible_series_count):
        raise ValueError(
            "ICL production refit must contain the complete eligible series set: "
            f"expected={int(eligible_series_count)}, "
            f"actual={bundle.manifest.series_count}."
        )
    cutoff = normalize_period_key(data_cutoff, "weekly")
    latest_by_series: dict[str, int] = {}
    for item in bundle.episodes:
        latest_by_series[item.series_id] = max(
            latest_by_series.get(item.series_id, item.query_target.end_week),
            item.query_target.end_week,
        )
    mismatched_cutoffs = {
        series_id: value
        for series_id, value in latest_by_series.items()
        if value != cutoff
    }
    if mismatched_cutoffs:
        raise ValueError(
            "Every ICL production series must reach the sealed data cutoff: "
            f"{mismatched_cutoffs}."
        )
    schema = bundle.manifest.exogenous_schema
    schema_hash = None if schema is None else schema.fingerprint
    configured_schema_hash = getattr(
        result.model.cfg,
        "icl_exogenous_schema_hash",
        None,
    )
    if configured_schema_hash != schema_hash:
        raise ValueError(
            "ICL production checkpoint and Episode feature schema differ."
        )
    required_backbone = {
        "model_id",
        "revision",
        "manifest_sha256",
        "contract_sha256",
    }
    missing_backbone = required_backbone - set(backbone_contract)
    if missing_backbone:
        raise ValueError(
            "ICL production checkpoint backbone contract is incomplete: "
            f"{sorted(missing_backbone)}."
        )
    configured_revision = str(
        getattr(result.model.cfg, "llm_revision", None) or ""
    ).strip()
    backbone_revision = str(backbone_contract["revision"] or "").strip()
    if not configured_revision or configured_revision != backbone_revision:
        raise ValueError(
            "ICL production checkpoint config and Qwen revision differ."
        )

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_config = replace(
        result.model.cfg,
        epochs=int(trainer_config.epochs),
        lr=float(trainer_config.lr),
        weight_decay=float(trainer_config.weight_decay),
        max_grad_norm=trainer_config.max_grad_norm,
        training_mode="production_refit",
        random_seed=int(random_seed),
    )
    save_model(
        result.model,
        checkpoint_config,
        str(output),
        extra_meta={
            "model_key": str(model_key),
            "family_key": str(model_key).removesuffix("_base"),
            "training_mode": "production_refit",
            "validation_enabled": False,
            "state_selection": "final_epoch",
            "random_seed": int(random_seed),
            "epochs": int(trainer_config.epochs),
            "completed_epochs": int(result.epochs_completed),
            "final_train_loss": float(result.final_train_loss),
            "train_data_cutoff": cutoff,
            "episode_manifest_hash": bundle.manifest.manifest_hash,
            "episode_schema_hash": schema_hash,
            "episode_count": len(bundle.episodes),
            "series_count": bundle.manifest.series_count,
            "eligible_series_count": int(eligible_series_count),
            "backbone_contract": dict(backbone_contract),
            "operational_admission_status": "approved_by_exception",
        },
    )
    return output


__all__ = [
    "ICLForward",
    "ICLTrainerConfig",
    "ICLTrainingResult",
    "fit_icl_model",
    "save_icl_production_checkpoint",
]

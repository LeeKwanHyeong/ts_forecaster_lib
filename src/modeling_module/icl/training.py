"""Shared training loop for models that consume sealed ICL episode batches."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from modeling_module.data_loader.icl_episode_data_module import ICLBatch


ICLForward = Callable[[torch.nn.Module, "ICLBatch"], torch.Tensor]


@dataclass(frozen=True)
class ICLTrainerConfig:
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    max_grad_norm: float | None = None

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


@dataclass(frozen=True)
class ICLTrainingResult:
    model: torch.nn.Module
    best_validation_loss: float | None
    final_train_loss: float
    epochs_completed: int
    epoch_history: tuple[dict[str, float | int | None], ...] = ()


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
        for batch in loader:
            from modeling_module.data_loader.icl_episode_data_module import ICLBatch

            if not isinstance(batch, ICLBatch):
                raise TypeError("ICL trainer requires ICLEpisodeDataModule batches.")
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
                raise RuntimeError("ICL model produced a non-finite prediction.")
            loss = F.mse_loss(prediction, batch.query_target)
            if not torch.isfinite(loss):
                raise RuntimeError("ICL training loss became non-finite.")
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
    """Fit one ICL-enabled model and restore the best validation state."""

    cfg = config or ICLTrainerConfig()
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
        if selection_loss < best_loss:
            best_loss = selection_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("ICL trainer did not produce a model state.")
    model.load_state_dict(best_state)
    model.eval()
    return ICLTrainingResult(
        model=model,
        best_validation_loss=(best_loss if val_loader is not None else None),
        final_train_loss=final_train_loss,
        epochs_completed=int(cfg.epochs),
        epoch_history=tuple(epoch_history),
    )


__all__ = [
    "ICLForward",
    "ICLTrainerConfig",
    "ICLTrainingResult",
    "fit_icl_model",
]

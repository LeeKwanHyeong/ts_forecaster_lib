"""NHITS.py

Main N-HiTS model wrapper.

This file keeps the user-facing (wrapper) class `NHITS` that instantiates the
backbone model and exposes training/validation helpers.

Backbone components (basis, blocks, and the core `_NHITS` model) are located in
`backbone.py`.
"""

from __future__ import annotations

import random

import numpy as np
import torch as t
import torch.nn as nn
from torch import optim


# ---------------------------------------------------------------------------
# Backbone import
# ---------------------------------------------------------------------------
# Support both (a) package usage: `from .backbone import _NHITS`
# and (b) script/local usage: `from backbone import _NHITS`.
try:
    from .backbone import Backbone  # type: ignore
except Exception:
    from backbone import Backbone  # type: ignore


# ---------------------------------------------------------------------------
# Optional loss imports
# ---------------------------------------------------------------------------
# NOTE:
# The original NHITS wrapper references MAELoss and LossFunction, which may
# live elsewhere in your project. We intentionally keep these names untouched
# to avoid breaking existing training code.
try:
    from .losses import MAELoss, LossFunction  # type: ignore
except Exception:
    try:
        from losses import MAELoss, LossFunction  # type: ignore
    except Exception:
        MAELoss = None  # type: ignore
        LossFunction = None  # type: ignore


class NHITS(nn.Module):
    def __init__(
        self,
        n_time_in,
        n_time_out,
        n_x,
        n_x_hidden,
        n_s,
        n_s_hidden,
        shared_weights,
        activation,
        initialization,
        stack_types,
        n_blocks,
        n_layers,
        n_theta_hidden,
        n_pool_kernel_size,
        n_freq_downsample,
        pooling_mode,
        interpolation_mode,
        batch_normalization,
        dropout_prob_theta,
        learning_rate,
        lr_decay,
        lr_decay_step_size,
        weight_decay,
        loss_train,
        loss_hypar,
        loss_valid,
        frequency,
        random_seed,
        seasonality,
    ):
        super().__init__()
        """N-HiTS model wrapper.

        This wrapper instantiates the backbone `_NHITS` model and provides
        training/validation helpers compatible with the original code.
        """

        if activation == "SELU":
            initialization = "lecun_normal"

        # ------------------------ Model Attributes ------------------------ #
        # Architecture parameters
        self.n_time_in = n_time_in
        self.n_time_out = n_time_out
        self.n_x = n_x
        self.n_x_hidden = n_x_hidden
        self.n_s = n_s
        self.n_s_hidden = n_s_hidden
        self.shared_weights = shared_weights
        self.activation = activation
        self.initialization = initialization
        self.stack_types = stack_types
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.n_theta_hidden = n_theta_hidden
        self.n_pool_kernel_size = n_pool_kernel_size
        self.n_freq_downsample = n_freq_downsample
        self.pooling_mode = pooling_mode
        self.interpolation_mode = interpolation_mode

        # Loss functions
        self.loss_train = loss_train
        self.loss_hypar = loss_hypar
        self.loss_valid = loss_valid

        # Keep the original names/behavior as-is; only set if available.
        self.loss_fn_train = MAELoss() if MAELoss is not None else None
        self.loss_fn_valid = (
            LossFunction(loss_valid, seasonality=self.loss_hypar) if LossFunction is not None else None
        )

        # Regularization and optimization parameters
        self.batch_normalization = batch_normalization
        self.dropout_prob_theta = dropout_prob_theta
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.lr_decay_step_size = lr_decay_step_size
        self.random_seed = random_seed

        # Data parameters
        self.frequency = frequency
        self.seasonality = seasonality
        self.return_decomposition = False

        """
                N-HiTS model.

                Parameters
                ----------
                n_time_in: int
                    Multiplier to get in_sample size.
                    In_sample size(Lookback) = n_time_in * output_size
                n_time_out: int
                    Forecast horizon.
                shared_weights: bool
                    If True, repeats first block.
                activation: str
                    Activation function.
                    An item from ['relu', 'softplus', 'tanh', 'selu', 'lrelu', 'prelu', 'sigmoid'].
                initialization: str
                    Initialization function.
                    An item from ['orthogonal', 'he_uniform', 'glorot_uniform', 'glorot_normal', 'lecun_normal'].
                stack_types: List[str]
                    List of stack types.
                    Subset from ['identity'].
                n_blocks: List[int]
                    Number of blocks for each stack type.
                    Note that len(n_blocks) = len(stack_types).
                n_layers: List[int]
                    Number of layers for each stack type.
                    Note that len(n_layers) = len(stack_types).
                n_theta_hidden: List[List[int]]
                    Structure of hidden layers for each stack type.
                    Each internal list should contain the number of units of each hidden layer.
                    Note that len(n_theta_hidden) = len(stack_types).
                n_pool_kernel_size List[int]:
                    Pooling size for input for each stack.
                    Note that len(n_pool_kernel_size) = len(stack_types).
                n_freq_downsample List[int]:
                    Downsample multiplier of output for each stack.
                    Note that len(n_freq_downsample) = len(stack_types).
                batch_normalization: bool
                    Whether perform batch normalization.
                dropout_prob_theta: float
                    Float between (0, 1).
                    Dropout for Nbeats basis.
                learning_rate: float
                    Learning rate between (0, 1).
                lr_decay: float
                    Decreasing multiplier for the learning rate.
                lr_decay_step_size: int
                    Steps between each lerning rate decay.
                weight_decay: float
                    L2 penalty for optimizer.
                loss_train: str
                    Loss to optimize.
                    An item from ['MAPE', 'MASE', 'SMAPE', 'MSE', 'MAE', 'PINBALL', 'PINBALL2'].
                loss_hypar:
                    Hyperparameter for chosen loss.
                loss_valid:
                    Validation loss.
                    An item from ['MAPE', 'MASE', 'SMAPE', 'RMSE', 'MAE', 'PINBALL'].
                frequency: str
                    Time series frequency.
                random_seed: int
                    random_seed for pseudo random pytorch initializer and
                    numpy random generator.
                seasonality: int
                    Time series seasonality.
                    Usually 7 for daily data, 12 for monthly data and 4 for weekly data.
                """

        # Backbone
        self.model = Backbone(
            n_time_in=self.n_time_in,
            n_time_out=self.n_time_out,
            n_s=self.n_s,
            n_x=self.n_x,
            n_s_hidden=self.n_s_hidden,
            n_x_hidden=self.n_x_hidden,
            stack_types=self.stack_types,
            n_blocks=self.n_blocks,
            n_layers=self.n_layers,
            n_theta_hidden=self.n_theta_hidden,
            n_pool_kernel_size=self.n_pool_kernel_size,
            n_freq_downsample=self.n_freq_downsample,
            pooling_mode=self.pooling_mode,
            interpolation_mode=self.interpolation_mode,
            dropout_prob_theta=self.dropout_prob_theta,
            activation=self.activation,
            initialization=self.initialization,
            batch_normalization=self.batch_normalization,
            shared_weights=self.shared_weights,
        )

    # ---------------------------------------------------------------------
    # Training helpers (kept for compatibility)
    # ---------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        if self.loss_fn_train is None:
            raise RuntimeError("MAELoss is not available. Please import/provide MAELoss in your project.")

        S = batch["S"]
        Y = batch["Y"]
        X = batch["X"]
        sample_mask = batch["sample_mask"]
        available_mask = batch["available_mask"]

        outsample_y, forecast, outsample_mask = self.model(
            S=S,
            Y=Y,
            X=X,
            insample_mask=available_mask,
            outsample_mask=sample_mask,
            return_decomposition=False,
        )

        loss = self.loss_fn_train(y=outsample_y, y_hat=forecast, mask=outsample_mask, y_insample=Y)

        # If you are using Lightning, `self.log` will exist; otherwise ignore.
        if hasattr(self, "log"):
            self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, idx):
        if self.loss_fn_valid is None:
            raise RuntimeError(
                "LossFunction is not available. Please import/provide LossFunction in your project."
            )

        S = batch["S"]
        Y = batch["Y"]
        X = batch["X"]
        sample_mask = batch["sample_mask"]
        available_mask = batch["available_mask"]

        outsample_y, forecast, outsample_mask = self.model(
            S=S,
            Y=Y,
            X=X,
            insample_mask=available_mask,
            outsample_mask=sample_mask,
            return_decomposition=False,
        )

        loss = self.loss_fn_valid(y=outsample_y, y_hat=forecast, mask=outsample_mask, y_insample=Y)

        if hasattr(self, "log"):
            self.log("val_loss", loss, prog_bar=True)

        return loss

    def on_fit_start(self):
        t.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def forward(self, batch):
        S = batch["S"]
        Y = batch["Y"]
        X = batch["X"]
        sample_mask = batch["sample_mask"]
        available_mask = batch["available_mask"]

        if self.return_decomposition:
            outsample_y, forecast, block_forecast, outsample_mask = self.model(
                S=S,
                Y=Y,
                X=X,
                insample_mask=available_mask,
                outsample_mask=sample_mask,
                return_decomposition=True,
            )
            return outsample_y, forecast, block_forecast, outsample_mask

        outsample_y, forecast, outsample_mask = self.model(
            S=S,
            Y=Y,
            X=X,
            insample_mask=available_mask,
            outsample_mask=sample_mask,
            return_decomposition=False,
        )
        return outsample_y, forecast, outsample_mask

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_decay_step_size, gamma=self.lr_decay
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


__all__ = ["NHITS"]

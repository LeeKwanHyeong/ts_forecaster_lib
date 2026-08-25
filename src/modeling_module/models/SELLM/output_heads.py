from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroInflatedSoftplusHead(nn.Module):
    """Return the expected non-negative demand from occurrence and magnitude terms."""

    history_feature_count = 4

    def __init__(
        self,
        *,
        horizon: int,
        hidden_dim: int,
        softplus_beta: float,
        initial_nonzero_probability: float,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.softplus_beta = float(softplus_beta)
        self.occurrence_gate = nn.Sequential(
            nn.Linear(self.history_feature_count, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.horizon),
        )
        probability = float(initial_nonzero_probability)
        initial_logit = math.log(probability / (1.0 - probability))
        final_layer = self.occurrence_gate[-1]
        nn.init.normal_(final_layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(final_layer.bias, initial_logit)

    @staticmethod
    def history_features(history: torch.Tensor) -> torch.Tensor:
        """Build scale-stable occurrence features for each batch/channel pair."""

        if history.ndim != 3:
            raise ValueError(
                "SELLM positive output history must be [B,L,C], "
                f"got shape={tuple(history.shape)}."
            )
        if not torch.isfinite(history).all():
            raise ValueError("SELLM positive output history must contain finite values only.")

        demand = history.clamp_min(0.0)
        mean = demand.mean(dim=1)
        last = demand[:, -1, :]
        zero_ratio = (demand <= 0.0).to(dtype=demand.dtype).mean(dim=1)
        half_window = max((int(demand.size(1)) + 1) // 2, 1)
        early_mean = demand[:, :half_window, :].mean(dim=1)
        recent_mean = demand[:, -half_window:, :].mean(dim=1)
        scale = mean.clamp_min(1.0)
        relative_trend = torch.tanh((recent_mean - early_mean) / scale)
        return torch.stack(
            [
                torch.log1p(mean),
                torch.log1p(last),
                zero_ratio,
                relative_trend,
            ],
            dim=-1,
        )

    def forward(self, raw_forecast: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        if raw_forecast.ndim != 3:
            raise ValueError(
                "SELLM raw forecast must be [B,H,C], "
                f"got shape={tuple(raw_forecast.shape)}."
            )
        if int(raw_forecast.size(1)) != self.horizon:
            raise ValueError(
                f"SELLM raw forecast horizon must be {self.horizon}, "
                f"got {int(raw_forecast.size(1))}."
            )
        if int(history.size(0)) != int(raw_forecast.size(0)) or int(
            history.size(2)
        ) != int(raw_forecast.size(2)):
            raise ValueError("SELLM positive output history and forecast shapes do not align.")

        features = self.history_features(history)
        batch, channels, _ = features.shape
        logits = self.occurrence_gate(
            features.reshape(batch * channels, self.history_feature_count)
        )
        nonzero_probability = torch.sigmoid(
            logits.reshape(batch, channels, self.horizon).permute(0, 2, 1)
        )
        positive_magnitude = F.softplus(
            raw_forecast,
            beta=self.softplus_beta,
        )
        return nonzero_probability * positive_magnitude


__all__ = ["ZeroInflatedSoftplusHead"]

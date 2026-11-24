"""The common heads used by PHEN and the matched-head baselines."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .time_likelihood import positive_integer_log_mass
from .contracts import EventModelConfig


class NextEventModel(nn.Module):
    """Encode observed events and predict one next duration and quantity."""

    event_config: EventModelConfig
    event_model_key: str

    def __init__(self, hidden_dim, train_log_mean, *, time_scale=1.0,
                 time_initial_location=0.0, time_initial_scale=1.0,
                 time_sigma_floor=1e-3, time_observation_contract=None):
        super().__init__()
        if not math.isfinite(train_log_mean) or train_log_mean <= 0:
            raise ValueError("train_log_mean must be finite and positive")
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale must be finite and positive")
        if not math.isfinite(time_initial_location):
            raise ValueError("time_initial_location must be finite")
        if not 0 < time_sigma_floor < time_initial_scale < math.inf:
            raise ValueError("Time scale initialization must exceed its positive floor")
        observation = dict(time_observation_contract or {})
        if (set(observation) != {"mode", "unit", "top_code"}
                or observation["mode"] != "positive_integer_round_clamp_v1"
                or observation["unit"] not in {"hour", "day", "week", "month"}):
            raise ValueError("An explicit positive-integer observation contract is required")
        code = observation["top_code"]
        if code is not None and (type(code) is not int or code < 2):
            raise ValueError("top_code must be null or an integer >= 2")
        self.hidden_dim = hidden_dim
        self.time_scale = float(time_scale)
        self.time_sigma_floor = float(time_sigma_floor)
        self.time_observation_contract = observation
        # Keep the historical initialization order for seeded equivalence.
        self.v_t = nn.Linear(hidden_dim, 1, bias=False)
        self.b_t = nn.Parameter(torch.full((1,), float(time_initial_location)))
        sigma = time_initial_scale - time_sigma_floor
        self.w_raw = nn.Parameter(torch.full((1,), sigma + math.log(-math.expm1(-sigma))))
        with torch.random.fork_rng(devices=[]):
            self.time_scale_weight = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.time_scale_weight.weight)
        self.quantity_head = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.v_t.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.quantity_head.weight)
        nn.init.constant_(self.quantity_head.bias, float(np.log(np.expm1(train_log_mean))))

    @staticmethod
    def continuous_features(dts, history_quantities, mask):
        features = torch.stack((torch.log1p(dts.float().clamp_min(0)),
                                torch.log1p(history_quantities.float().clamp_min(0))), -1)
        return features * mask.unsqueeze(-1).to(features.dtype)

    def encode(self, dts, history_quantities, mask):
        raise NotImplementedError

    def encode_task_states(self, dts, history_quantities, mask, *, memory_write_mask=None):
        observed = mask if memory_write_mask is None else mask & memory_write_mask
        safe_dts = torch.where(observed, dts, 0.0)
        safe_qty = torch.where(observed, history_quantities, 0.0)
        encoded = self.encode(safe_dts, safe_qty, observed)
        return encoded, encoded

    def predict_quantity(self, hidden):
        log_quantity = F.softplus(self.quantity_head(hidden).squeeze(-1))
        return log_quantity, torch.expm1(log_quantity)

    def quantity_outputs(self, hidden, true_quantity):
        location, prediction = self.predict_quantity(hidden)
        loss = F.mse_loss(location, torch.log1p(true_quantity.clamp_min(0)), reduction="none")
        return {"train_loss": loss, "log_mse": loss, "point_prediction": prediction}

    def duration_parameters(self, hidden):
        hidden = hidden.to(torch.float64)
        location = F.linear(hidden, self.v_t.weight.to(torch.float64)).squeeze(-1)
        location = location + self.b_t.to(torch.float64)
        raw_scale = F.linear(hidden, self.time_scale_weight.weight.to(torch.float64)).squeeze(-1)
        scale = self.time_sigma_floor + F.softplus(raw_scale + self.w_raw.to(torch.float64))
        return location, scale

    def log_observation_dt(self, hidden, dt_next):
        location, scale = self.duration_parameters(hidden)
        return positive_integer_log_mass(dt_next, location, scale, time_scale=self.time_scale,
                                        top_code=self.time_observation_contract["top_code"])

    def predict_time_median(self, hidden):
        location = F.linear(hidden.to(torch.float64), self.v_t.weight.to(torch.float64)).squeeze(-1)
        location = location + self.b_t.to(torch.float64)
        return self.time_scale * torch.exp(location)

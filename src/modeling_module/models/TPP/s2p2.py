"""S2P2 matched-head variant; pinned kernels and licenses are in _vendor/s2p2."""

import math

import torch
from torch import nn

from .heads import NextEventModel
from ._vendor.s2p2.models import Int_Backward_LLH


def affine_scan(a, b, initial):
    """Inclusive x_i=a_i*x_{i-1}+b_i, avoiding complex log(0).

    Associative composition (a2,b2) o (a1,b1)=(a2*a1,a2*b1+b2).
    Equivalent to the authors' sequential recurrence; O(N log N) element work,
    O(log N) stages. Do not claim the original work-efficient scan's complexity.
    """
    step = 1
    while step < a.shape[-2]:
        b = torch.cat((b[..., :step, :], b[..., step:, :] + a[..., step:, :] * b[..., :-step, :]), -2)
        a = torch.cat((a[..., :step, :], a[..., step:, :] * a[..., :-step, :]), -2)
        step *= 2
    return b + a * initial.unsqueeze(-2)


class StableBackwardLLH(Int_Backward_LLH):
    """Original backward ZOH LLH; only parallel recurrence arithmetic changes."""
    def _ssm(self, left_u_NH, right_u_NH, impulse_NP, dt_N, initial_state_P):
        rates = self.get_lambda(right_u_NH=right_u_NH, shift_u=True)
        lam = rates.get("lambda_rescaled_NP", rates.get("lambda_rescaled_P"))
        decay = torch.exp(dt_N.unsqueeze(-1) * lam)
        if left_u_NH is None:
            integrated, left_skip, right_skip = 0.0, 0.0, 0.0
        else:
            integrated = torch.einsum("...np,ph,...nh->...np", decay - 1.0,
                                     self.B_tilde_PH, left_u_NH.to(torch.complex64))
            left_skip = left_u_NH * self.D_HH
            right_skip = right_u_NH * self.D_HH
        right = affine_scan(decay, integrated + impulse_NP, initial_state_P)
        previous = torch.cat((initial_state_P.unsqueeze(-2), right[..., :-1, :]), -2)
        left = decay * previous + integrated
        left_y = 2 * torch.einsum("hp,...np->...nh", self.C_tilde_HP, left).real + left_skip
        right_y = 2 * torch.einsum("hp,...np->...nh", self.C_tilde_HP, right).real + right_skip
        return right, left_y, right_y


class S2P2(NextEventModel):
    def __init__(self, hidden_dim, train_log_mean, **kwargs):
        super().__init__(hidden_dim, train_log_mean, **kwargs)
        self.input_projection = nn.Linear(2, hidden_dim)
        self.layers = nn.ModuleList([
            StableBackwardLLH(P=16, H=hidden_dim, dropout_rate=.1,
                              pre_norm=False, post_norm=True, relative_time=True,
                              complex_values=True, simple_mark=True, is_first_layer=i == 0)
            for i in range(2)
        ])

    def encode(self, dts, history_quantities, mask):
        # Production caller uses right-padding. Canonicalizing here also makes
        # left-padding and holes inert, then restores the original positions.
        n = mask.shape[1]
        positions = torch.arange(n, device=mask.device).expand_as(mask)
        order = torch.argsort((~mask).long() * n + positions, dim=1)
        inverse = torch.argsort(order, dim=1)
        valid = mask.gather(1, order)
        gaps = torch.where(valid, dts.gather(1, order).float().clamp_min(0), 0.)
        qty = torch.where(valid, history_quantities.gather(1, order), 0.)
        alpha = self.input_projection(self.continuous_features(gaps, qty, valid))
        alpha = alpha * valid.unsqueeze(-1)
        left = right = None
        for layer in self.layers:
            _, left, right = layer(left, right, alpha, gaps)
            left = left * valid.unsqueeze(-1)
            right = right * valid.unsqueeze(-1)
        return right.gather(1, inverse.unsqueeze(-1).expand(-1, -1, self.hidden_dim))

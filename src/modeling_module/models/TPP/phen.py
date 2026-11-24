"""PHEN: two causal blocks, gated history correction and static retrieval."""

import torch

from .phen_backbone import EventEncoder, HistoryCorrection, PrototypeRetrieval
from .heads import NextEventModel


class PHEN(NextEventModel):
    def __init__(self, hidden_dim, train_log_mean, max_seq_len, *,
                 history=True, prototypes=True, **head_kwargs):
        if hidden_dim != 64 or type(max_seq_len) is not int or max_seq_len < 2:
            raise ValueError("Recorded PHEN requires hidden_dim=64 and max_seq_len >= 2")
        super().__init__(hidden_dim, train_log_mean, **head_kwargs)
        self.max_seq_len = max_seq_len
        self.encoder = EventEncoder(hidden_dim, max_seq_len)
        # Build both before removing ablated components to retain seeded initialization.
        self.prototype = PrototypeRetrieval(hidden_dim)
        self.history_correction = HistoryCorrection(hidden_dim)
        if not history:
            self.history_correction = None
        if not prototypes:
            self.prototype = None

    def encode_task_states(self, dts, history_quantities, mask, *, memory_write_mask=None):
        if (dts.ndim != 2 or history_quantities.shape != dts.shape
                or mask.shape != dts.shape or mask.dtype != torch.bool
                or dts.size(1) > self.max_seq_len or dts.size(1) == 0):
            raise ValueError("Expected event tensors [batch,length] within max_seq_len and a boolean mask")
        if memory_write_mask is not None and (memory_write_mask.shape != mask.shape
                                               or memory_write_mask.dtype != torch.bool):
            raise ValueError("memory_write_mask must match the boolean event mask")
        observed = mask if memory_write_mask is None else mask & memory_write_mask
        safe_dts = torch.where(observed, dts, 0.0)
        safe_qty = torch.where(observed, history_quantities, 0.0)
        if (not torch.isfinite(safe_dts).all() or not torch.isfinite(safe_qty).all()
                or (safe_dts < 0).any() or (safe_qty < 0).any()):
            raise ValueError("Observed gaps and quantities must be finite and nonnegative")
        x = self.encoder.input_proj(self.continuous_features(safe_dts, safe_qty, observed))
        x = torch.where(observed.unsqueeze(-1), x + self.encoder.pos_emb[:, :x.size(1)], 0.0)
        x = self.encoder.layers[0](x, observed)
        if self.history_correction is not None:
            x = x + self.history_correction(x, mask, observed=observed)
        x = self.encoder.layers[1](x, observed)
        if self.prototype is not None:
            x = self.prototype(x)
        x = torch.where(observed.unsqueeze(-1), x, 0.0)
        return x, x

    def encode(self, dts, history_quantities, mask):
        return self.encode_task_states(dts, history_quantities, mask)[0]

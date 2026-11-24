"""The existing GRU-based RMTPP variant under the common event heads."""

from torch import nn

from .heads import NextEventModel


class RMTPP(NextEventModel):
    def __init__(self, hidden_dim, train_log_mean, **head_kwargs):
        super().__init__(hidden_dim, train_log_mean, **head_kwargs)
        self.input_projection = nn.Linear(2, hidden_dim)
        self.input_dropout = nn.Dropout(0.1)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def encode(self, dts, history_quantities, mask):
        x = self.input_dropout(self.input_projection(self.continuous_features(dts, history_quantities, mask)))
        encoded, _ = self.encoder(x)
        return encoded * mask.unsqueeze(-1).to(encoded.dtype)

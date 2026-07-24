from __future__ import annotations

from numbers import Integral
from typing import Sequence

import torch
from torch import nn


_INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)


class FutureCategoricalEmbedding(nn.Module):
    """Independent embedding tables for future categorical feature columns."""

    def __init__(
        self,
        *,
        cardinalities: Sequence[int],
        embedding_dim: int,
        horizon: int,
    ) -> None:
        super().__init__()
        normalized_cardinalities: list[int] = []
        for value in cardinalities:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(
                    "cardinalities must contain positive integers; "
                    f"got {value!r}."
                )
            cardinality = int(value)
            if cardinality <= 0:
                raise ValueError(
                    "cardinalities must contain positive integers; "
                    f"got {cardinality}."
                )
            normalized_cardinalities.append(cardinality)
        if not normalized_cardinalities:
            raise ValueError("cardinalities must contain at least one feature.")

        if isinstance(embedding_dim, bool) or not isinstance(
            embedding_dim,
            Integral,
        ):
            raise ValueError(
                f"embedding_dim must be a positive integer, got {embedding_dim!r}."
            )
        if isinstance(horizon, bool) or not isinstance(horizon, Integral):
            raise ValueError(f"horizon must be a positive integer, got {horizon!r}.")

        self.cardinalities = tuple(normalized_cardinalities)
        self.embedding_dim = int(embedding_dim)
        self.horizon = int(horizon)
        if self.embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, got {self.embedding_dim}."
            )
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}.")

        # ID 0 is a learnable UNK representation, not padding.
        self.tables = nn.ModuleList(
            nn.Embedding(cardinality, self.embedding_dim)
            for cardinality in self.cardinalities
        )

    @property
    def num_features(self) -> int:
        return len(self.cardinalities)

    @property
    def output_dim(self) -> int:
        return self.num_features * self.embedding_dim

    def forward(
        self,
        future_exo_cat: torch.Tensor,
        *,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(future_exo_cat):
            raise TypeError(
                "future_exo_cat must be a torch.Tensor with shape [B,H,K]."
            )
        if future_exo_cat.ndim != 3:
            raise ValueError(
                "future_exo_cat must have rank 3 [B,H,K], "
                f"got shape {tuple(future_exo_cat.shape)}."
            )
        if future_exo_cat.dtype not in _INTEGER_DTYPES:
            raise TypeError(
                "future_exo_cat must use an integer dtype, "
                f"got {future_exo_cat.dtype}."
            )

        actual_batch, actual_horizon, actual_features = future_exo_cat.shape
        if batch_size is not None and actual_batch != int(batch_size):
            raise ValueError(
                "future_exo_cat batch mismatch: "
                f"got {actual_batch}, expected {int(batch_size)}."
            )
        if actual_horizon != self.horizon:
            raise ValueError(
                "future_exo_cat horizon mismatch: "
                f"got {actual_horizon}, expected {self.horizon}."
            )
        if actual_features != self.num_features:
            raise ValueError(
                "future_exo_cat feature-width mismatch: "
                f"got {actual_features}, expected {self.num_features}."
            )

        embedded_features: list[torch.Tensor] = []
        for feature_index, (table, cardinality) in enumerate(
            zip(self.tables, self.cardinalities)
        ):
            feature_ids = future_exo_cat[..., feature_index]
            if feature_ids.numel() > 0:
                min_id = int(feature_ids.min().item())
                max_id = int(feature_ids.max().item())
                if min_id < 0 or max_id >= cardinality:
                    raise ValueError(
                        "future_exo_cat category IDs for feature index "
                        f"{feature_index} must be in [0, {cardinality - 1}], "
                        f"got range [{min_id}, {max_id}]."
                    )
            embedded_features.append(table(feature_ids.to(dtype=torch.long)))

        return torch.cat(embedded_features, dim=-1)


__all__ = ["FutureCategoricalEmbedding"]

from __future__ import annotations

import pytest
import torch

from modeling_module.models.PatchTST.common.configs import (
    AttentionConfig,
    PatchTSTConfig,
)
from modeling_module.models.model_builder import build_patchTST_exogenous
from modeling_module.training.adapters import DefaultAdapter, PatchTSTAdapter


class _FutureCategoricalProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.horizon = 2
        self.received_future_cat = None

    def forward(
        self,
        x: torch.Tensor,
        *,
        future_exo_cat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.received_future_cat = future_exo_cat
        return x[:, : self.horizon, 0]


class _LegacyPointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.horizon = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, : self.horizon, 0]


@pytest.mark.parametrize("adapter_type", [DefaultAdapter, PatchTSTAdapter])
def test_adapter_forwards_future_cat_without_copying(adapter_type):
    adapter = adapter_type()
    model = _FutureCategoricalProbe()
    future_cat = torch.tensor(
        [
            [[0], [1]],
            [[2], [0]],
        ],
        dtype=torch.long,
    )

    output = adapter.forward(
        model,
        torch.ones(2, 4, 1),
        future_exo_cat=future_cat,
        mode="train",
    )

    assert model.received_future_cat is future_cat
    assert output.shape == (2, 2)


@pytest.mark.parametrize("batch_kind", ["dict", "tuple"])
def test_default_adapter_preserves_future_cat_for_structured_inputs(batch_kind):
    adapter = DefaultAdapter()
    model = _FutureCategoricalProbe()
    x = torch.ones(2, 4, 1)
    future_cat = torch.zeros(2, 2, 1, dtype=torch.long)
    x_batch = {"x": x} if batch_kind == "dict" else (x,)

    output = adapter.forward(
        model,
        x_batch,
        future_exo_cat=future_cat,
    )

    assert model.received_future_cat is future_cat
    assert output.shape == (2, 2)


def test_default_adapter_preserves_legacy_model_call_without_future_cat():
    adapter = DefaultAdapter()
    model = _LegacyPointModel()

    output = adapter.forward(model, torch.ones(2, 4, 1), mode="eval")

    assert output.shape == (2, 2)


def test_default_adapter_does_not_silently_drop_future_cat():
    adapter = DefaultAdapter()
    model = _LegacyPointModel()

    with pytest.raises(
        NotImplementedError,
        match="does not declare `future_exo_cat`",
    ):
        adapter.forward(
            model,
            torch.ones(2, 4, 1),
            future_exo_cat=torch.zeros(2, 2, 1, dtype=torch.long),
        )


def _patchtst_config() -> PatchTSTConfig:
    return PatchTSTConfig(
        lookback=8,
        horizon=2,
        patch_len=4,
        stride=2,
        padding_patch="end",
        d_model=8,
        d_ff=16,
        n_layers=1,
        dropout=0.0,
        c_in=1,
        past_exo_cont_dim=1,
        future_exo_dim=1,
        future_exo_cat_cardinalities=(3,),
        future_exo_cat_embedding_dim=4,
        future_exo_fusion_dropout=0.0,
        use_revin=False,
        attn=AttentionConfig(
            n_heads=2,
            d_model=8,
            attn_dropout=0.0,
            proj_dropout=0.0,
        ),
    )


def test_patchtst_adapter_runs_categorical_fusion_end_to_end():
    model = build_patchTST_exogenous(_patchtst_config())
    adapter = PatchTSTAdapter()

    output = adapter.forward(
        model,
        torch.ones(2, 8, 1),
        future_exo=torch.ones(2, 2, 1),
        future_exo_cat=torch.zeros(2, 2, 1, dtype=torch.long),
        past_exo_cont=torch.ones(2, 8, 1),
        mode="train",
    )

    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert model.future_cat_embedding is not None
    assert model.future_cat_embedding.tables[0].weight.grad is not None

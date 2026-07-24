from __future__ import annotations

import pytest
import torch

from modeling_module.models.PatchTST.common.configs import (
    AttentionConfig,
    PatchTSTConfig,
)
from modeling_module.models.model_builder import (
    build_patchTST,
    build_patchTST_quantile,
)
from modeling_module.training.model_losses.loss_module import DistributionLoss


def _config(
    *,
    future_cont_dim: int = 0,
    future_cat_cardinalities: tuple[int, ...] = (),
    loss=None,
) -> PatchTSTConfig:
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
        future_exo_dim=future_cont_dim,
        future_exo_cat_cardinalities=future_cat_cardinalities,
        future_exo_cat_embedding_dim=4,
        future_exo_fusion_dropout=0.0,
        use_revin=False,
        loss=loss,
        attn=AttentionConfig(
            n_heads=2,
            d_model=8,
            attn_dropout=0.0,
            proj_dropout=0.0,
        ),
    )


def _future_inputs(future_cont_dim: int):
    future_cont = (
        torch.tensor(
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.5, 0.6], [0.7, 0.8]],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        if future_cont_dim > 0
        else None
    )
    category_a = torch.ones(2, 2, 1, dtype=torch.long)
    category_b = torch.full((2, 2, 1), 2, dtype=torch.long)
    return future_cont, category_a, category_b


def _make_category_embeddings_distinct(model) -> None:
    assert model.future_cat_embedding is not None
    with torch.no_grad():
        table = model.future_cat_embedding.tables[0]
        table.weight.zero_()
        table.weight[1].copy_(
            torch.tensor([1.0, -0.5, 0.25, 2.0])
        )
        table.weight[2].copy_(
            torch.tensor([-1.0, 0.75, 1.5, -0.25])
        )


@pytest.mark.parametrize(
    ("builder", "output_key"),
    (
        (build_patchTST, None),
        (build_patchTST_quantile, "q"),
    ),
)
@pytest.mark.parametrize("future_cont_dim", (0, 2))
def test_category_values_change_point_and_quantile_predictions(
    builder,
    output_key,
    future_cont_dim,
) -> None:
    torch.manual_seed(104)
    model = builder(
        _config(
            future_cont_dim=future_cont_dim,
            future_cat_cardinalities=(3,),
        )
    )
    model.eval()
    _make_category_embeddings_distinct(model)
    x = torch.linspace(-1.0, 1.0, 16).reshape(2, 8, 1)
    future_cont, category_a, category_b = _future_inputs(future_cont_dim)

    raw_a = model(
        x,
        future_exo=future_cont,
        future_exo_cat=category_a,
    )
    raw_b = model(
        x,
        future_exo=future_cont,
        future_exo_cat=category_b,
    )
    output_a = raw_a if output_key is None else raw_a[output_key]
    output_b = raw_b if output_key is None else raw_b[output_key]

    assert output_a.shape[:2] == (2, 2)
    assert torch.isfinite(output_a).all()
    assert not torch.allclose(output_a, output_b, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    ("builder", "output_key"),
    (
        (build_patchTST, None),
        (build_patchTST_quantile, "q"),
    ),
)
def test_continuous_only_predictions_match_legacy_fusion_path(
    builder,
    output_key,
) -> None:
    torch.manual_seed(105)
    model = builder(_config(future_cont_dim=2))
    model.eval()
    x = torch.linspace(-1.0, 1.0, 16).reshape(2, 8, 1)
    future_cont, _, _ = _future_inputs(2)

    with torch.no_grad():
        z = model.backbone(
            x,
            past_exo_cont=None,
            past_exo_cat=None,
        )
        assert model.future_fuser is not None
        z = model.future_fuser(z, future_cont)
        expected = model.head(z, future_exo=future_cont)
        raw_actual = model(x, future_exo=future_cont)
        actual = (
            raw_actual
            if output_key is None
            else raw_actual[output_key]
        )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("distribution", ("Normal", "StudentT"))
@pytest.mark.parametrize("future_cont_dim", (0, 2))
def test_distribution_categorical_fusion_changes_loc_and_preserves_domain(
    distribution,
    future_cont_dim,
) -> None:
    torch.manual_seed(106)
    loss = DistributionLoss(distribution)
    model = build_patchTST(
        _config(
            future_cont_dim=future_cont_dim,
            future_cat_cardinalities=(3,),
            loss=loss,
        )
    )
    model.eval()
    _make_category_embeddings_distinct(model)
    x = torch.linspace(-1.0, 1.0, 16).reshape(2, 8, 1)
    future_cont, category_a, category_b = _future_inputs(future_cont_dim)

    output_a = model(
        x,
        future_exo=future_cont,
        future_exo_cat=category_a,
    )
    output_b = model(
        x,
        future_exo=future_cont,
        future_exo_cat=category_b,
    )
    loc_index = model.param_names.index("-loc")
    loc_a = output_a[..., loc_index]
    loc_b = output_b[..., loc_index]

    assert output_a.shape == (
        2,
        2,
        loss.outputsize_multiplier,
    )
    assert torch.isfinite(output_a).all()
    assert not torch.allclose(loc_a, loc_b, rtol=1e-6, atol=1e-7)

    domain_params = loss.domain_map(output_a)
    valid_params = loss.scale_decouple(domain_params)
    if distribution == "Normal":
        _, scale = valid_params
    else:
        df, _, scale = valid_params
        assert torch.all(df > 3.0)
    assert torch.all(scale > 0.0)

    loc_a.square().mean().backward()
    assert model.future_cat_embedding is not None
    embedding_grad = model.future_cat_embedding.tables[0].weight.grad
    assert embedding_grad is not None
    assert float(embedding_grad.abs().sum()) > 0.0
    assert model.future_fuser is not None
    assert model.future_fuser.future_proj.weight.grad is not None

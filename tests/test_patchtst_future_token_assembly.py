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


def _config(
    *,
    future_cont_dim: int = 0,
    future_cat_cardinalities: tuple[int, ...] = (),
    future_cat_embedding_dim: int = 4,
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
        future_exo_cat_embedding_dim=future_cat_embedding_dim,
        future_exo_fusion_dropout=0.0,
        use_revin=False,
        attn=AttentionConfig(
            n_heads=2,
            d_model=8,
            attn_dropout=0.0,
            proj_dropout=0.0,
        ),
    )


@pytest.mark.parametrize(
    "builder",
    (build_patchTST, build_patchTST_quantile),
)
@pytest.mark.parametrize(
    (
        "future_cont_dim",
        "future_cat_cardinalities",
        "expected_width",
    ),
    (
        (0, (), 0),
        (2, (), 2),
        (0, (3, 5), 8),
        (2, (3, 5), 10),
    ),
)
def test_future_token_assembly_supports_all_input_combinations(
    builder,
    future_cont_dim,
    future_cat_cardinalities,
    expected_width,
) -> None:
    model = builder(
        _config(
            future_cont_dim=future_cont_dim,
            future_cat_cardinalities=future_cat_cardinalities,
        )
    )
    future_cont = (
        torch.randn(2, 2, future_cont_dim)
        if future_cont_dim > 0
        else None
    )
    future_cat = (
        torch.tensor(
            [
                [[0, 0], [1, 3]],
                [[2, 4], [0, 0]],
            ],
            dtype=torch.long,
        )
        if future_cat_cardinalities
        else None
    )

    tokens = model.build_future_exogenous_tokens(
        future_cont,
        future_cat,
        batch_size=2,
    )

    assert model.future_exo_token_dim == expected_width
    if expected_width == 0:
        assert tokens is None
        assert model.future_fuser is None
        return

    assert tokens is not None
    assert tokens.shape == (2, 2, expected_width)
    assert model.future_fuser is not None
    assert model.future_fuser.input_dim == expected_width
    assert model.future_fuser.d_future == expected_width
    assert model.future_fuser.future_proj.in_features == expected_width
    if future_cont_dim > 0 and not future_cat_cardinalities:
        assert tokens is future_cont
    if future_cat_cardinalities:
        expected_cat = model.encode_future_categorical(
            future_cat,
            batch_size=2,
        )
        assert expected_cat is not None
        if future_cont_dim > 0:
            torch.testing.assert_close(
                tokens[..., :future_cont_dim],
                future_cont,
            )
            torch.testing.assert_close(
                tokens[..., future_cont_dim:],
                expected_cat,
            )
        else:
            torch.testing.assert_close(tokens, expected_cat)


def test_combined_future_tokens_preserve_continuous_and_embedding_gradients() -> None:
    model = build_patchTST(
        _config(
            future_cont_dim=2,
            future_cat_cardinalities=(3, 5),
        )
    )
    future_cont = torch.randn(2, 2, 2, requires_grad=True)
    future_cat = torch.tensor(
        [
            [[0, 0], [1, 3]],
            [[2, 4], [0, 0]],
        ],
        dtype=torch.int32,
    )

    tokens = model.build_future_exogenous_tokens(
        future_cont,
        future_cat,
        batch_size=2,
    )

    assert tokens is not None
    assert tokens.shape == (2, 2, 10)
    tokens.square().mean().backward()
    assert future_cont.grad is not None
    assert float(future_cont.grad.abs().sum()) > 0.0
    assert model.future_cat_embedding is not None
    for table in model.future_cat_embedding.tables:
        assert table.weight.grad is not None
        assert float(table.weight.grad.abs().sum()) > 0.0


@pytest.mark.parametrize("future_cont_dim", (0, 2))
def test_categorical_tokens_are_accepted_by_cross_attention(
    future_cont_dim,
) -> None:
    model = build_patchTST(
        _config(
            future_cont_dim=future_cont_dim,
            future_cat_cardinalities=(3, 5),
        )
    )
    future_cont = (
        torch.randn(2, 2, future_cont_dim, requires_grad=True)
        if future_cont_dim > 0
        else None
    )
    future_cat = torch.tensor(
        [
            [[0, 0], [1, 3]],
            [[2, 4], [0, 0]],
        ],
        dtype=torch.long,
    )
    tokens = model.build_future_exogenous_tokens(
        future_cont,
        future_cat,
        batch_size=2,
    )
    z = torch.randn(2, 3, 8, requires_grad=True)

    assert tokens is not None
    assert model.future_fuser is not None
    output = model.future_fuser(z, tokens)

    assert output.shape == z.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert z.grad is not None
    assert float(z.grad.abs().sum()) > 0.0
    assert model.future_cat_embedding is not None
    for table in model.future_cat_embedding.tables:
        assert table.weight.grad is not None
        assert float(table.weight.grad.abs().sum()) > 0.0
    if future_cont is not None:
        assert future_cont.grad is not None
        assert float(future_cont.grad.abs().sum()) > 0.0


@pytest.mark.parametrize(
    ("builder", "output_key"),
    (
        (build_patchTST, None),
        (build_patchTST_quantile, "q"),
    ),
)
@pytest.mark.parametrize("future_cont_dim", (0, 2))
def test_point_and_quantile_run_combined_future_tokens_end_to_end(
    builder,
    output_key,
    future_cont_dim,
) -> None:
    model = builder(
        _config(
            future_cont_dim=future_cont_dim,
            future_cat_cardinalities=(3, 5),
        )
    )
    x = torch.randn(2, 8, 1)
    future_cont = (
        torch.randn(2, 2, future_cont_dim, requires_grad=True)
        if future_cont_dim > 0
        else None
    )
    future_cat = torch.tensor(
        [
            [[0, 0], [1, 3]],
            [[2, 4], [0, 0]],
        ],
        dtype=torch.long,
    )

    raw_output = model(
        x,
        future_exo=future_cont,
        future_exo_cat=future_cat,
    )
    output = raw_output if output_key is None else raw_output[output_key]

    assert output.shape[:2] == (2, 2)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert model.future_fuser is not None
    assert model.future_fuser.future_proj.weight.grad is not None
    assert model.future_cat_embedding is not None
    for table in model.future_cat_embedding.tables:
        assert table.weight.grad is not None
        assert float(table.weight.grad.abs().sum()) > 0.0
    if future_cont is not None:
        assert future_cont.grad is not None
        assert float(future_cont.grad.abs().sum()) > 0.0


def test_future_token_assembly_rejects_non_floating_continuous_input() -> None:
    model = build_patchTST(_config(future_cont_dim=1))

    with pytest.raises(TypeError, match="floating dtype"):
        model.build_future_exogenous_tokens(
            torch.ones(2, 2, 1, dtype=torch.long),
            None,
            batch_size=2,
        )

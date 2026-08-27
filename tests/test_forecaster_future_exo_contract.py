from __future__ import annotations

import pytest
import torch

from modeling_module.models.PatchTST.common.configs import (
    AttentionConfig,
    PatchTSTConfig,
)
from modeling_module.models.TimeXer.TimeXer import TimeXerModel
from modeling_module.models.TimeXer.configs import TimeXerConfig
from modeling_module.models.model_builder import build_patchTST_exogenous
from modeling_module.training.forecater import (
    DMSForecaster,
    _infer_d_future_expected,
    _safe_forward,
)


class _ShapeRejectingModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, future_exo: torch.Tensor | None = None) -> torch.Tensor:
        if future_exo is not None:
            raise RuntimeError("future-exogenous-shape-error")
        return x


def test_safe_forward_does_not_swallow_model_shape_errors_during_signature_fallback():
    model = _ShapeRejectingModel()

    with pytest.raises(RuntimeError, match="future-exogenous-shape-error"):
        _safe_forward(
            model,
            torch.zeros(1, 2, 1),
            future_exo=torch.zeros(1, 1, 2),
            unsupported_alias=torch.zeros(1, 1, 2),
        )


def _tiny_timexer() -> TimeXerModel:
    return TimeXerModel(
        TimeXerConfig(
            device="cpu",
            lookback=2,
            horizon=1,
            y_dim=1,
            past_exo_cont_dim=1,
            patch_len=1,
            d_model=4,
            n_heads=1,
            d_ff=8,
            e_layers=1,
            dropout=0.0,
            factor=1,
            use_norm=False,
        )
    )


def _tiny_categorical_patchtst() -> torch.nn.Module:
    return build_patchTST_exogenous(
        PatchTSTConfig(
            device="cpu",
            lookback=2,
            horizon=1,
            c_in=1,
            patch_len=1,
            stride=1,
            d_model=4,
            d_ff=8,
            n_layers=1,
            dropout=0.0,
            past_exo_cat_dim=1,
            cat_cardinalities=[3],
            d_cat_emb=2,
            use_revin=False,
            attn=AttentionConfig(
                n_heads=1,
                d_model=4,
                attn_dropout=0.0,
                proj_dropout=0.0,
            ),
        )
    )


def test_timexer_inherited_training_default_is_not_mistaken_for_future_exogenous_support():
    assert _infer_d_future_expected(_tiny_timexer()) is None


def test_timexer_future_exogenous_rejection_is_preserved_through_forecaster():
    forecaster = DMSForecaster(_tiny_timexer())

    with pytest.raises(ValueError, match="TimeXer v1 does not consume future exogenous inputs"):
        forecaster.predict(
            torch.zeros(1, 2, 1),
            horizon=1,
            device="cpu",
            past_exo_cont=torch.zeros(1, 2, 1),
            future_exo_batch=torch.zeros(1, 1, 1),
        )


@pytest.mark.parametrize(
    "model_factory",
    [_tiny_timexer, _tiny_categorical_patchtst],
    ids=["timexer", "patchtst-exogenous"],
)
def test_public_forecaster_rejects_nonempty_categorical_input_before_model_forward(
    model_factory,
):
    forecaster = DMSForecaster(model_factory())

    with pytest.raises(RuntimeError, match="does not accept categorical past exogenous inputs"):
        forecaster.predict(
            torch.zeros(1, 2, 1),
            horizon=1,
            device="cpu",
            past_exo_cat=torch.zeros(1, 2, 1, dtype=torch.long),
        )


@pytest.mark.parametrize("future_shape", [(1,), (1, 1, 1, 1)])
def test_timexer_invalid_future_exogenous_rank_is_not_silently_ignored(future_shape):
    forecaster = DMSForecaster(_tiny_timexer())

    with pytest.raises(RuntimeError, match="must have rank 2.*or rank 3"):
        forecaster.predict(
            torch.zeros(1, 2, 1),
            horizon=1,
            device="cpu",
            past_exo_cont=torch.zeros(1, 2, 1),
            future_exo_batch=torch.zeros(future_shape),
        )

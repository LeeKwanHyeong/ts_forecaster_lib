from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from modeling_module.models.ExoTST.ExoTST import ExoTST
from modeling_module.models.ExoTST.configs import ExoTSTConfig
from modeling_module.models.PatchMixer.PatchMixer import PatchMixerPointModel
from modeling_module.models.PatchMixer.common.configs import PatchMixerConfig
from modeling_module.models.PatchTST.common.configs import AttentionConfig, PatchTSTConfig
from modeling_module.models.PatchTST.supervised.PatchTST import PatchTSTModel
from modeling_module.models.Titan.Titans import TitanBaseModel
from modeling_module.models.Titan.common.configs import TitanConfig


ModelCase = tuple[torch.nn.Module, torch.Tensor, dict[str, torch.Tensor]]


def _patchtst_case(future_width: int) -> ModelCase:
    cfg = PatchTSTConfig(
        lookback=8,
        horizon=2,
        patch_len=4,
        stride=2,
        d_model=8,
        d_ff=16,
        n_layers=1,
        dropout=0.0,
        c_in=1,
        future_exo_dim=future_width,
        use_revin=False,
        attn=AttentionConfig(
            n_heads=2,
            d_model=8,
            attn_dropout=0.0,
            proj_dropout=0.0,
        ),
    )
    return PatchTSTModel(cfg).eval(), torch.randn(2, 8, 1), {}


def _patchmixer_case(future_width: int) -> ModelCase:
    cfg = PatchMixerConfig(
        lookback=8,
        horizon=2,
        patch_len=4,
        stride=2,
        d_model=8,
        e_layers=1,
        mixer_kernel_size=3,
        f_out=8,
        head_hidden=8,
        dropout=0.0,
        future_exo_dim=future_width,
        use_revin=False,
        final_nonneg=False,
    )
    return PatchMixerPointModel(cfg).eval(), torch.randn(2, 8, 1), {}


def _titan_case(future_width: int) -> ModelCase:
    cfg = TitanConfig(
        lookback=8,
        horizon=2,
        future_exo_dim=future_width,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        dropout=0.0,
        contextual_mem_size=2,
        persistent_mem_size=2,
        use_revin=False,
        max_len=16,
    )
    return TitanBaseModel(cfg).eval(), torch.randn(2, 8, 1), {}


def _exotst_case(future_width: int) -> ModelCase:
    cfg = ExoTSTConfig(
        lookback=8,
        horizon=2,
        y_dim=1,
        exo_dim_past=1,
        exo_dim_future=future_width,
        use_past_exo=True,
        use_future_exo=future_width > 0,
        exo_nan_policy="zero",
        patch_len=2,
        stride=1,
        d_model=8,
        n_heads=2,
        d_ff=16,
        dropout=0.0,
        attn_dropout=0.0,
        exo_enc_layers=1,
        fusion_layers=1,
        endo_dec_layers=1,
        use_revin=False,
    )
    return (
        ExoTST(cfg).eval(),
        torch.randn(2, 8, 1),
        {"past_exo_cont": torch.randn(2, 8, 1)},
    )


MODEL_CASES: tuple[tuple[str, Callable[[int], ModelCase]], ...] = (
    ("PatchTST", _patchtst_case),
    ("PatchMixer", _patchmixer_case),
    ("Titan", _titan_case),
    ("ExoTST", _exotst_case),
)


@pytest.mark.parametrize(("family", "make_case"), MODEL_CASES, ids=lambda value: str(value))
def test_model_requires_configured_future_exogenous_input(
    family: str,
    make_case: Callable[[int], ModelCase],
) -> None:
    model, x, kwargs = make_case(1)

    with pytest.raises(RuntimeError, match=rf"\[{family}\].*future_exo is required"):
        model(x, **kwargs)


@pytest.mark.parametrize(("family", "make_case"), MODEL_CASES, ids=lambda value: str(value))
@pytest.mark.parametrize(
    ("future_exo", "message"),
    (
        (torch.zeros(2, 1), "rank-3"),
        (torch.zeros(1, 2, 1), "batch mismatch"),
        (torch.zeros(2, 3, 1), "horizon mismatch"),
        (torch.zeros(2, 2, 2), "last dimension mismatch"),
    ),
    ids=("rank", "batch", "horizon", "width"),
)
def test_model_rejects_future_exogenous_shape_mismatch(
    family: str,
    make_case: Callable[[int], ModelCase],
    future_exo: torch.Tensor,
    message: str,
) -> None:
    model, x, kwargs = make_case(1)

    with pytest.raises(RuntimeError, match=rf"\[{family}\].*{message}"):
        model(x, future_exo=future_exo, **kwargs)


@pytest.mark.parametrize(("family", "make_case"), MODEL_CASES, ids=lambda value: str(value))
def test_model_rejects_nonempty_future_exogenous_input_when_disabled(
    family: str,
    make_case: Callable[[int], ModelCase],
) -> None:
    model, x, kwargs = make_case(0)

    with pytest.raises(RuntimeError, match=rf"\[{family}\].*configured future width=0"):
        model(x, future_exo=torch.zeros(2, 2, 1), **kwargs)


@pytest.mark.parametrize(("family", "make_case"), MODEL_CASES, ids=lambda value: str(value))
def test_model_prediction_is_sensitive_to_valid_future_exogenous_input(
    family: str,
    make_case: Callable[[int], ModelCase],
) -> None:
    torch.manual_seed(20260720)
    model, x, kwargs = make_case(1)

    with torch.no_grad():
        zeros = model(x, future_exo=torch.zeros(2, 2, 1), **kwargs)
        ones = model(x, future_exo=torch.ones(2, 2, 1), **kwargs)

    assert zeros.shape == ones.shape
    assert not torch.allclose(zeros, ones), f"{family} silently ignored valid future exogenous input"

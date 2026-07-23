from __future__ import annotations

from pathlib import Path

import pytest
import torch

from modeling_module import load_predictor
from modeling_module.models.PatchMixer import (
    PatchMixerConfig,
    PatchMixerExogenousConfig,
    PatchMixerExogenousModel,
    PatchMixerModel,
)
from modeling_module.utils.checkpoint import save_model


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


def _paper_config() -> PatchMixerConfig:
    return PatchMixerConfig(
        lookback=16,
        horizon=4,
        enc_in=1,
        patch_len=4,
        stride=2,
        mixer_kernel_size=3,
        d_model=8,
        e_layers=1,
        dropout=0.0,
        head_dropout=0.0,
    )


def _exogenous_config() -> PatchMixerExogenousConfig:
    return PatchMixerExogenousConfig(
        device="cuda",
        lookback=16,
        horizon=4,
        enc_in=1,
        patch_len=4,
        stride=2,
        mixer_kernel_size=3,
        d_model=8,
        e_layers=1,
        dropout=0.0,
        head_dropout=0.0,
        f_out=8,
        head_hidden=8,
        use_revin=True,
        final_nonneg=False,
        past_exo_mode="z_gate",
        past_exo_cont_dim=2,
        future_exo_dim=2,
    )


def _assert_finite_nonzero_gradients(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient: {name}"
        assert torch.count_nonzero(parameter.grad) > 0, f"zero gradient: {name}"


def test_patchmixer_cuda_backward_and_strict_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(20260723)
    config = _paper_config()
    model = PatchMixerModel(config).cuda().train()
    x = torch.randn(3, config.lookback, config.enc_in, device="cuda")
    model(x).square().mean().backward()
    _assert_finite_nonzero_gradients(model)

    model.eval()
    with torch.no_grad():
        expected = model(x)
    path = tmp_path / "patchmixer.pt"
    save_model(model, config, str(path), extra_meta={"model_key": "patchmixer"})
    restored = load_predictor(str(path), device="cuda", strict=True)
    with torch.no_grad():
        actual = restored.model(x)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_patchmixer_exo_cuda_backward_and_strict_checkpoint_roundtrip(
    tmp_path: Path,
) -> None:
    torch.manual_seed(20260724)
    config = _exogenous_config()
    model = PatchMixerExogenousModel(config).cuda().train()
    x = torch.randn(3, config.lookback, 1, device="cuda")
    past = torch.randn(3, config.lookback, config.past_exo_cont_dim, device="cuda")
    future = torch.randn(3, config.horizon, config.future_exo_dim, device="cuda")
    model(x, past_exo_cont=past, future_exo=future).square().mean().backward()
    _assert_finite_nonzero_gradients(model)

    model.eval()
    with torch.no_grad():
        expected = model(x, past_exo_cont=past, future_exo=future)
    path = tmp_path / "patchmixer_exo.pt"
    save_model(model, config, str(path), extra_meta={"model_key": "patchmixer_exo"})
    restored = load_predictor(str(path), device="cuda", strict=True)
    with torch.no_grad():
        actual = restored.model(x, past_exo_cont=past, future_exo=future)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

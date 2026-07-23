from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from modeling_module import load_predictor
from modeling_module.models.PatchMixer.common.configs import PatchMixerExogenousConfig
from modeling_module.models.PatchMixer.variants import PatchMixerExogenousModel
from modeling_module.utils.checkpoint import build_checkpoint_payload


def _normalized_config(
    *, residual_limit: float | None
) -> PatchMixerExogenousConfig:
    return PatchMixerExogenousConfig(
        device="cpu",
        lookback=8,
        horizon=2,
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
        past_exo_mode="none",
        future_exo_dim=2,
        future_exo_shift_space="normalized",
        future_exo_normalized_residual_limit=residual_limit,
    )


@pytest.mark.parametrize("residual_limit", (None, 0.15), ids=("unbounded", "bounded"))
def test_patchmixer_exogenous_normalized_checkpoint_strict_load_and_predict(
    tmp_path: Path,
    residual_limit: float | None,
) -> None:
    torch.manual_seed(20260812)
    config = _normalized_config(residual_limit=residual_limit)
    model = PatchMixerExogenousModel(config).eval()
    x = torch.linspace(-2.0, 3.0, steps=16).reshape(2, 8, 1)
    future = torch.linspace(-0.5, 0.75, steps=8).reshape(2, 2, 2)

    with torch.no_grad():
        expected = model(x, future_exo=future)

    payload = build_checkpoint_payload(
        model,
        config,
        extra_meta={"model_key": "patchmixer_exo", "family_key": "patchmixer"},
    )
    checkpoint_path = tmp_path / f"patchmixer_exo_normalized_{residual_limit}.pt"
    torch.save(payload, checkpoint_path)

    predictor = load_predictor(str(checkpoint_path), device="cpu", strict=True)
    assert predictor.model_key == "patchmixer_exo"
    assert predictor.config["future_exo_shift_space"] == "normalized"
    assert predictor.config["future_exo_normalized_residual_limit"] == residual_limit
    assert predictor.model.state_dict().keys() == model.state_dict().keys()

    with torch.no_grad():
        restored = predictor.model(x, future_exo=future)
    torch.testing.assert_close(restored, expected, rtol=0.0, atol=0.0)

    result = predictor.predict(
        {"x": x, "future_exo": future},
        horizon=config.horizon,
    )
    np.testing.assert_allclose(result["point"], expected.numpy().reshape(-1))

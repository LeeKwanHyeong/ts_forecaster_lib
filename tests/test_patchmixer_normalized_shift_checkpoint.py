from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from modeling_module import load_predictor
from modeling_module.models.PatchMixer.common.configs import PatchMixerConfig
from modeling_module.models.PatchMixer.variants import (
    PatchMixerExogenousModel,
    PatchMixerQuantileExogenousModel,
)
from modeling_module.training.model_losses.loss_module import DistributionLoss
from modeling_module.utils.checkpoint import build_checkpoint_payload


def _normalized_config(
    mode: str,
    *,
    residual_limit: float | None,
) -> PatchMixerConfig:
    kwargs = {}
    if mode == "distribution":
        loss = DistributionLoss(distribution="StudentT", validate_args=False)
        kwargs.update(
            loss=loss,
            out_mul=loss.outputsize_multiplier,
            param_names=list(loss.param_names),
        )

    return PatchMixerConfig(
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
        patch_cfgs=((4, 2, 3),),
        per_branch_dim=4,
        fused_dim=8,
        quantiles=(0.1, 0.5, 0.9),
        q_clip_norm=None,
        **kwargs,
    )


def _model_and_key(mode: str, config: PatchMixerConfig):
    if mode == "quantile":
        return PatchMixerQuantileExogenousModel(config), "patchmixer_quantile_exogenous"
    return PatchMixerExogenousModel(config), "patchmixer_exogenous"


def _output_tensor(output):
    return output["q"] if isinstance(output, dict) else output


@pytest.mark.parametrize("mode", ("point", "quantile", "distribution"))
@pytest.mark.parametrize("residual_limit", (None, 0.15), ids=("unbounded", "bounded"))
def test_patchmixer_normalized_checkpoint_strict_load_and_public_predict(
    tmp_path: Path,
    mode: str,
    residual_limit: float | None,
) -> None:
    torch.manual_seed(20260812)
    config = _normalized_config(mode, residual_limit=residual_limit)
    model, model_key = _model_and_key(mode, config)
    model.eval()
    x = torch.linspace(-2.0, 3.0, steps=16).reshape(2, 8, 1)
    future = torch.linspace(-0.5, 0.75, steps=8).reshape(2, 2, 2)

    with torch.no_grad():
        expected = _output_tensor(model(x, future_exo=future))

    payload = build_checkpoint_payload(
        model,
        config,
        extra_meta={"model_key": model_key, "family_key": "patchmixer"},
    )
    checkpoint_path = tmp_path / (
        f"patchmixer_{mode}_normalized_{residual_limit}.pt"
    )
    torch.save(payload, checkpoint_path)

    predictor = load_predictor(str(checkpoint_path), device="cpu", strict=True)
    assert predictor.model_key == model_key
    assert predictor.config["future_exo_shift_space"] == "normalized"
    assert (
        predictor.config["future_exo_normalized_residual_limit"]
        == residual_limit
    )
    assert predictor.model.future_exo_shift_space == "normalized"
    assert predictor.model.future_exo_normalized_residual_limit == residual_limit
    assert predictor.model.state_dict().keys() == model.state_dict().keys()

    with torch.no_grad():
        restored = _output_tensor(predictor.model(x, future_exo=future))
    torch.testing.assert_close(restored, expected, rtol=0.0, atol=0.0)

    result = predictor.predict(
        {"x": x, "future_exo": future},
        horizon=config.horizon,
    )
    expected_numpy = expected.detach().cpu().numpy()
    if mode == "quantile":
        np.testing.assert_allclose(result["q10"], expected_numpy[:, 0, :].reshape(-1))
        np.testing.assert_allclose(result["q50"], expected_numpy[:, 1, :].reshape(-1))
        np.testing.assert_allclose(result["q90"], expected_numpy[:, 2, :].reshape(-1))
        np.testing.assert_allclose(result["point"], result["q50"])
    else:
        if mode == "distribution":
            expected_numpy = expected_numpy[..., model.loc_idx]
        np.testing.assert_allclose(result["point"], expected_numpy.reshape(-1))

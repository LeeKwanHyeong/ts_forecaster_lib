from __future__ import annotations

import torch
import torch.nn.functional as F

from modeling_module.models.PatchMixer import (
    PatchMixerConfig,
    PatchMixerModel,
)
from modeling_module.models.PatchMixer.provenance import (
    PATCHMIXER_REFERENCE_CONFIG,
    PATCHMIXER_REFERENCE_PARAMETER_COUNTS,
    PATCHMIXER_UPSTREAM_COMMIT,
    PATCHMIXER_UPSTREAM_MODEL_BLOB,
)


UPSTREAM_COMMIT = "cfc6c1386e7fe1633f92ef4b258ff1a4649008b4"
UPSTREAM_MODEL_BLOB = "bf3867109192da6cd8816f4aec8ab0bf16ec80af"


def _small_config() -> PatchMixerConfig:
    return PatchMixerConfig(
        lookback=24,
        horizon=6,
        enc_in=3,
        patch_len=6,
        stride=3,
        mixer_kernel_size=3,
        d_model=8,
        e_layers=2,
        dropout=0.0,
        head_dropout=0.0,
        use_revin=True,
        revin_affine=True,
        revin_subtract_last=False,
    )


def _upstream_functional_forward(
    model: PatchMixerModel,
    x: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the pinned upstream equations without calling model.forward()."""
    config = model.configs
    state = model.state_dict()
    batch_size, _, nvars = x.shape

    mean = torch.mean(x, dim=1, keepdim=True).detach()
    stdev = torch.sqrt(
        torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
    ).detach()
    x = x - mean
    x = x / stdev
    x = x * state["model.revin_layer.affine_weight"]
    x = x + state["model.revin_layer.affine_bias"]

    x = x.permute(0, 2, 1)
    x = F.pad(x, (0, config.stride), mode="replicate")
    x = x.unfold(-1, config.patch_len, config.stride)
    x = F.linear(
        x,
        state["model.W_P.weight"],
        state["model.W_P.bias"],
    )
    patch_num = x.shape[2]
    x = x.reshape(batch_size * nvars, patch_num, config.d_model)

    linear_forecast = F.linear(
        x.flatten(start_dim=-2),
        state["model.head0.1.weight"],
        state["model.head0.1.bias"],
    )

    conv_padding = (config.mixer_kernel_size - 1) // 2
    for layer_index in range(config.e_layers):
        prefix = f"model.PatchMixer_blocks.{layer_index}"
        residual = F.conv1d(
            x,
            state[f"{prefix}.Resnet.0.weight"],
            state[f"{prefix}.Resnet.0.bias"],
            padding=conv_padding,
            groups=patch_num,
        )
        residual = F.gelu(residual)
        residual = F.batch_norm(
            residual,
            state[f"{prefix}.Resnet.2.running_mean"],
            state[f"{prefix}.Resnet.2.running_var"],
            state[f"{prefix}.Resnet.2.weight"],
            state[f"{prefix}.Resnet.2.bias"],
            training=False,
            eps=1e-5,
        )
        x = x + residual
        x = F.conv1d(
            x,
            state[f"{prefix}.Conv_1x1.0.weight"],
            state[f"{prefix}.Conv_1x1.0.bias"],
        )
        x = F.gelu(x)
        x = F.batch_norm(
            x,
            state[f"{prefix}.Conv_1x1.2.running_mean"],
            state[f"{prefix}.Conv_1x1.2.running_var"],
            state[f"{prefix}.Conv_1x1.2.weight"],
            state[f"{prefix}.Conv_1x1.2.bias"],
            training=False,
            eps=1e-5,
        )

    nonlinear_forecast = F.linear(
        x.flatten(start_dim=-2),
        state["model.head1.1.weight"],
        state["model.head1.1.bias"],
    )
    nonlinear_forecast = F.gelu(nonlinear_forecast)
    nonlinear_forecast = F.linear(
        nonlinear_forecast,
        state["model.head1.4.weight"],
        state["model.head1.4.bias"],
    )

    forecast = linear_forecast + nonlinear_forecast
    forecast = forecast.reshape(batch_size, nvars, config.horizon)
    forecast = forecast.permute(0, 2, 1)
    forecast = forecast - state["model.revin_layer.affine_bias"]
    forecast = forecast / (
        state["model.revin_layer.affine_weight"] + 1e-10
    )
    return forecast * stdev + mean


def _expected_state_dict_keys(config: PatchMixerConfig) -> set[str]:
    keys = {
        "model.W_P.weight",
        "model.W_P.bias",
        "model.head0.1.weight",
        "model.head0.1.bias",
        "model.head1.1.weight",
        "model.head1.1.bias",
        "model.head1.4.weight",
        "model.head1.4.bias",
        "model.revin_layer.affine_weight",
        "model.revin_layer.affine_bias",
    }
    for layer_index in range(config.e_layers):
        for block_name in ("Resnet", "Conv_1x1"):
            prefix = f"model.PatchMixer_blocks.{layer_index}.{block_name}"
            keys.update(
                {
                    f"{prefix}.0.weight",
                    f"{prefix}.0.bias",
                    f"{prefix}.2.weight",
                    f"{prefix}.2.bias",
                    f"{prefix}.2.running_mean",
                    f"{prefix}.2.running_var",
                    f"{prefix}.2.num_batches_tracked",
                }
            )
    return keys


def test_paper_output_matches_pinned_upstream_equations() -> None:
    torch.manual_seed(20260721)
    model = PatchMixerModel(_small_config()).eval()
    x = torch.randn(4, model.configs.lookback, model.configs.enc_in)

    with torch.no_grad():
        actual = model(x)
        expected = _upstream_functional_forward(model, x)

    assert actual.shape == (4, model.configs.horizon, model.configs.enc_in)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=5e-7)


def test_paper_upstream_state_dict_and_parameter_count_are_pinned() -> None:
    config = _small_config()
    model = PatchMixerModel(config)

    assert PATCHMIXER_UPSTREAM_COMMIT == UPSTREAM_COMMIT
    assert PATCHMIXER_UPSTREAM_MODEL_BLOB == UPSTREAM_MODEL_BLOB
    assert model.upstream_commit == UPSTREAM_COMMIT
    assert set(model.state_dict()) == _expected_state_dict_keys(config)

    reference_model = PatchMixerModel(
        dict(PATCHMIXER_REFERENCE_CONFIG)
    )
    expected_counts = dict(PATCHMIXER_REFERENCE_PARAMETER_COUNTS)
    parameter_count = sum(parameter.numel() for parameter in reference_model.parameters())
    assert parameter_count == expected_counts["original"]


def test_paper_backward_reaches_every_trainable_parameter() -> None:
    torch.manual_seed(20260722)
    model = PatchMixerModel(_small_config()).train()
    x = torch.randn(
        5,
        model.configs.lookback,
        model.configs.enc_in,
        requires_grad=True,
    )
    target = torch.randn(5, model.configs.horizon, model.configs.enc_in)

    loss = F.mse_loss(model(x), target)
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad) > 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient: {name}"
        assert torch.count_nonzero(parameter.grad) > 0, f"zero gradient: {name}"


def test_paper_forecast_is_channel_independent() -> None:
    torch.manual_seed(20260723)
    model = PatchMixerModel(_small_config()).eval()
    x = torch.randn(3, model.configs.lookback, model.configs.enc_in)
    perturbed = x.clone()
    waveform = torch.linspace(-3.0, 2.0, model.configs.lookback).pow(3)
    perturbed[:, :, 1] = perturbed[:, :, 1] + waveform

    with torch.no_grad():
        baseline = model(x)
        changed = model(perturbed)

    torch.testing.assert_close(
        changed[:, :, (0, 2)],
        baseline[:, :, (0, 2)],
        rtol=0.0,
        atol=0.0,
    )
    assert torch.max(torch.abs(changed[:, :, 1] - baseline[:, :, 1])) > 1e-5

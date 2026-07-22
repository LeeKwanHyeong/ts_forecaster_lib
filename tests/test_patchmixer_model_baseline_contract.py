from __future__ import annotations

import hashlib

import torch

import modeling_module.models.PatchMixer as patchmixer_public
from modeling_module.models.PatchMixer.PatchMixer import (
    PatchMixerEnhancedModel,
    PatchMixerOriginalBackbone as LegacyPatchMixerOriginalBackbone,
    PatchMixerOriginalLayer as LegacyPatchMixerOriginalLayer,
    PatchMixerOriginalModel,
    PatchMixerPointModel,
    PatchMixerQuantileModel,
    make_patch_cfgs as legacy_make_patch_cfgs,
)
from modeling_module.models.PatchMixer.backbone import (
    MultiScalePatchMixerBackbone,
    PatchMixerBackbone,
    PatchMixerOriginalBackbone,
    PatchMixerOriginalLayer,
    PatchMixerOriginalRevIN,
    make_patch_cfgs,
)
from modeling_module.models.PatchMixer.common.configs import (
    PatchMixerConfig,
    PatchMixerOriginalConfig,
)


def _input() -> torch.Tensor:
    return torch.linspace(-1.5, 2.0, steps=2 * 16 * 2).reshape(2, 16, 2)


def _enhanced_config() -> PatchMixerConfig:
    return PatchMixerConfig(
        device="cpu",
        lookback=16,
        horizon=4,
        enc_in=2,
        patch_len=4,
        stride=2,
        mixer_kernel_size=3,
        d_model=8,
        e_layers=1,
        dropout=0.0,
        head_dropout=0.0,
        f_out=8,
        head_hidden=8,
        use_revin=False,
        final_nonneg=False,
        past_exo_mode="none",
        patch_cfgs=((4, 2, 3), (8, 4, 5)),
        per_branch_dim=4,
        fused_dim=8,
        quantiles=(0.1, 0.5, 0.9),
    )


def _original_config() -> PatchMixerOriginalConfig:
    return PatchMixerOriginalConfig(
        lookback=16,
        horizon=4,
        enc_in=2,
        patch_len=4,
        stride=2,
        mixer_kernel_size=3,
        d_model=8,
        e_layers=1,
        dropout=0.0,
        head_dropout=0.0,
    )


def _state_schema_digest(model: torch.nn.Module) -> str:
    schema = "\n".join(
        f"{key}:{tuple(value.shape)}:{value.dtype}"
        for key, value in model.state_dict().items()
    )
    return hashlib.sha256(schema.encode()).hexdigest()


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_patchmixer_original_characterization_baseline() -> None:
    torch.manual_seed(20260724)
    model = PatchMixerOriginalModel(_original_config()).eval()

    with torch.no_grad():
        output = model(_input())

    expected = torch.tensor(
        [
            [
                [-0.72185004, -0.66629446],
                [-0.69748700, -0.64193141],
                [-0.92518568, -0.86963010],
                [-0.41591567, -0.36036003],
            ],
            [
                [1.05592775, 1.11148345],
                [1.08029079, 1.13584650],
                [0.85259211, 0.90814775],
                [1.36186206, 1.41741776],
            ],
        ]
    )

    assert output.shape == (2, 4, 2)
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
    assert len(model.state_dict()) == 24
    assert _state_schema_digest(model) == (
        "d5013ef1b2f334455e719f0c163d141bb8c4d7542d895b22b5363a98ed65cf19"
    )
    assert _parameter_count(model) == 996


def test_patchmixer_enhanced_characterization_baseline() -> None:
    torch.manual_seed(20260725)
    model = PatchMixerEnhancedModel(_enhanced_config()).eval()

    with torch.no_grad():
        output = model(_input())

    expected = torch.tensor(
        [
            [0.06410693, 0.06112923, 0.06971565, 0.06722046],
            [1.86621642, 1.86635876, 1.87538803, 1.87117016],
        ]
    )

    assert output.shape == (2, 4)
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
    assert len(model.state_dict()) == 45
    assert _state_schema_digest(model) == (
        "73817136afaab3760a58085738c3593a71d3050512d87440ee66fc5eb65d8b71"
    )
    assert _parameter_count(model) == 40_675


def test_patchmixer_quantile_characterization_baseline() -> None:
    torch.manual_seed(20260726)
    model = PatchMixerQuantileModel(_enhanced_config()).eval()

    with torch.no_grad():
        output = model(_input())["q"]

    expected = torch.tensor(
        [
            [
                [-0.91946810, -1.31314158, -1.51995206, -1.27745509],
                [0.19712697, -0.19654655, -0.40335709, -0.16086012],
                [1.31372190, 0.92004848, 0.71323794, 0.95573491],
            ],
            [
                [0.97733217, 0.63894439, 0.48349571, 0.72900224],
                [2.09392715, 1.75553942, 1.60009074, 1.84559727],
                [3.21052217, 2.87213445, 2.71668577, 2.96219230],
            ],
        ]
    )

    assert output.shape == (2, 3, 4)
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
    assert len(model.state_dict()) == 65
    assert _state_schema_digest(model) == (
        "b86ad895fa6e8e6026e1c16224fe441052fb428bb3c358adf4675cf46a77cdb3"
    )
    assert _parameter_count(model) == 2_904


def test_patchmixer_canonical_public_imports_are_identity_preserving() -> None:
    assert patchmixer_public.PatchMixerBackbone is PatchMixerBackbone
    assert (
        patchmixer_public.MultiScalePatchMixerBackbone
        is MultiScalePatchMixerBackbone
    )
    assert patchmixer_public.PatchMixerOriginalLayer is PatchMixerOriginalLayer
    assert patchmixer_public.PatchMixerOriginalBackbone is PatchMixerOriginalBackbone
    assert LegacyPatchMixerOriginalLayer is PatchMixerOriginalLayer
    assert LegacyPatchMixerOriginalBackbone is PatchMixerOriginalBackbone
    assert patchmixer_public.PatchMixerOriginalModel is PatchMixerOriginalModel
    assert patchmixer_public.PatchMixerEnhancedModel is PatchMixerEnhancedModel
    assert patchmixer_public.PatchMixerModel is not None
    assert issubclass(PatchMixerPointModel, PatchMixerEnhancedModel)
    assert PatchMixerOriginalRevIN.__module__.endswith("PatchMixer.backbone")
    assert PatchMixerOriginalLayer.__module__.endswith("PatchMixer.backbone")
    assert PatchMixerOriginalBackbone.__module__.endswith("PatchMixer.backbone")


def test_patchmixer_legacy_public_aliases_are_resolvable() -> None:
    assert patchmixer_public.BaseModel is PatchMixerPointModel
    assert patchmixer_public.QuantileModel is PatchMixerQuantileModel
    assert patchmixer_public.make_patch_cfgs is make_patch_cfgs
    assert legacy_make_patch_cfgs is make_patch_cfgs
    assert patchmixer_public.make_patch_cfgs(16, n_branches=2) == [
        (4, 2, 3),
        (8, 4, 5),
    ]

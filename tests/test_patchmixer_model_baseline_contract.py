from __future__ import annotations

import hashlib

import torch

import modeling_module.models.PatchMixer as patchmixer_public
from modeling_module.models.PatchMixer.PatchMixer import (
    PatchMixerModel,
    _PatchMixerLegacyModel,
)
from modeling_module.models.PatchMixer.backbone import (
    PatchMixerBackbone,
    PatchMixerLayer,
    PatchMixerRevIN,
)
from modeling_module.models.PatchMixer.common.configs import (
    PatchMixerConfig,
    PatchMixerExogenousConfig,
)


def _input() -> torch.Tensor:
    return torch.linspace(-1.5, 2.0, steps=2 * 16 * 2).reshape(2, 16, 2)


def _paper_config() -> PatchMixerConfig:
    return PatchMixerConfig(
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


def _legacy_config() -> PatchMixerExogenousConfig:
    return PatchMixerExogenousConfig(
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
    )


def _state_schema_digest(model: torch.nn.Module) -> str:
    schema = "\n".join(
        f"{key}:{tuple(value.shape)}:{value.dtype}"
        for key, value in model.state_dict().items()
    )
    return hashlib.sha256(schema.encode()).hexdigest()


def test_patchmixer_paper_characterization_baseline() -> None:
    torch.manual_seed(20260724)
    model = PatchMixerModel(_paper_config()).eval()

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

    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
    assert len(model.state_dict()) == 24
    assert _state_schema_digest(model) == (
        "d5013ef1b2f334455e719f0c163d141bb8c4d7542d895b22b5363a98ed65cf19"
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 996


def test_retired_enhanced_state_and_output_remain_load_compatible() -> None:
    torch.manual_seed(20260725)
    model = _PatchMixerLegacyModel(_legacy_config()).eval()

    with torch.no_grad():
        output = model(_input())

    expected = torch.tensor(
        [
            [0.06410693, 0.06112923, 0.06971565, 0.06722046],
            [1.86621642, 1.86635876, 1.87538803, 1.87117016],
        ]
    )
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
    assert len(model.state_dict()) == 45
    assert _state_schema_digest(model) == (
        "73817136afaab3760a58085738c3593a71d3050512d87440ee66fc5eb65d8b71"
    )


def test_patchmixer_public_surface_contains_only_active_responsibilities() -> None:
    assert patchmixer_public.PatchMixerModel is PatchMixerModel
    assert patchmixer_public.PatchMixerConfig is PatchMixerConfig
    assert PatchMixerRevIN.__module__.endswith("PatchMixer.backbone")
    assert PatchMixerLayer.__module__.endswith("PatchMixer.backbone")
    assert PatchMixerBackbone.__module__.endswith("PatchMixer.backbone")
    assert not hasattr(patchmixer_public, "PatchMixerEnhancedModel")
    assert not hasattr(patchmixer_public, "PatchMixerQuantileModel")
    assert not hasattr(patchmixer_public, "BaseModel")

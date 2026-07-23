from __future__ import annotations

import pytest
import torch

from modeling_module.models.TimeMixer import TimeMixerConfig
from modeling_module.models.TimeMixer.backbone import TimeMixerBackbone
from modeling_module.models.TimeMixer.provenance import (
    TIMEMIXER_UPSTREAM_COMMIT,
    TIMEMIXER_UPSTREAM_REPOSITORY,
)


def _config(**overrides) -> TimeMixerConfig:
    values = {
        "lookback": 16,
        "horizon": 4,
        "down_sampling_window": 2,
        "channel_independence": True,
        "e_layers": 1,
        "moving_avg": 3,
        "y_dim": 1,
        "use_future_temporal_feature": False,
        "d_model": 4,
        "d_ff": 8,
        "embed": "timeF",
        "freq": "h",
        "dropout": 0.0,
        "use_norm": True,
        "down_sampling_layers": 2,
        "down_sampling_method": "avg",
        "decomp_method": "moving_avg",
    }
    values.update(overrides)
    return TimeMixerConfig(**values)


def test_timemixer_forecasting_backbone_has_finite_upstream_shape() -> None:
    torch.manual_seed(20260723)
    model = TimeMixerBackbone(_config()).eval()
    x = torch.randn(2, 16, 1)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 4, 1)
    assert torch.isfinite(output).all()
    assert model.upstream_repository == TIMEMIXER_UPSTREAM_REPOSITORY
    assert model.upstream_commit == TIMEMIXER_UPSTREAM_COMMIT


def test_timemixer_backbone_supports_the_single_scale_boundary() -> None:
    torch.manual_seed(20260724)
    model = TimeMixerBackbone(_config(down_sampling_layers=0)).eval()
    x = torch.randn(2, 16, 1)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 4, 1)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"channel_independence": False}, "channel_independence=True"),
        ({"down_sampling_method": "conv"}, "down_sampling_method"),
        ({"decomp_method": "dft_decomp"}, "decomp_method"),
        ({"use_future_temporal_feature": True}, "future temporal features"),
    ),
)
def test_timemixer_backbone_rejects_out_of_scope_upstream_branches(
    override: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**override)

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from modeling_module.models.TimeMixer.backbone import TimeMixerBackbone
from modeling_module.models.TimeMixer.provenance import (
    TIMEMIXER_UPSTREAM_COMMIT,
    TIMEMIXER_UPSTREAM_REPOSITORY,
)


def _config(**overrides) -> SimpleNamespace:
    values = {
        "task_name": "long_term_forecast",
        "seq_len": 16,
        "label_len": 0,
        "pred_len": 4,
        "down_sampling_window": 2,
        "channel_independence": 1,
        "e_layers": 1,
        "moving_avg": 3,
        "enc_in": 1,
        "c_out": 1,
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
    return SimpleNamespace(**values)


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
        ({"task_name": "classification"}, "forecasting tasks only"),
        ({"channel_independence": 0}, "channel_independence=1"),
        ({"down_sampling_method": "conv"}, "average downsampling"),
        ({"decomp_method": "dft_decomp"}, "moving-average decomposition"),
        ({"use_future_temporal_feature": True}, "future temporal features"),
        ({"c_out": 2}, "enc_in and c_out"),
    ),
)
def test_timemixer_backbone_rejects_out_of_scope_upstream_branches(
    override: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TimeMixerBackbone(_config(**override))

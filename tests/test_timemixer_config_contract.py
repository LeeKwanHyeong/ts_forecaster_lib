from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from modeling_module.models.TimeMixer import TimeMixerConfig
from modeling_module.models.TimeMixer.backbone import TimeMixerBackbone


def test_timemixer_config_exposes_validated_upstream_aliases() -> None:
    config = TimeMixerConfig(
        lookback=54,
        horizon=27,
        down_sampling_layers=3,
        down_sampling_window=2,
    )

    assert config.scale_lengths == (54, 27, 13, 6)
    assert config.task_name == "long_term_forecast"
    assert config.seq_len == config.lookback
    assert config.pred_len == config.horizon
    assert config.label_len == 0
    assert config.enc_in == config.c_out == config.y_dim == 1
    assert config.use_exogenous_mode is False
    assert config.future_exo_dim == 0
    serialized = asdict(config)
    assert serialized["lookback"] == 54
    assert serialized["horizon"] == 27
    assert "scale_lengths" not in serialized
    assert "seq_len" not in serialized
    assert "pred_len" not in serialized
    assert "enc_in" not in serialized
    assert "c_out" not in serialized


def test_timemixer_config_single_scale_builds_a_finite_backbone() -> None:
    config = TimeMixerConfig(
        lookback=9,
        horizon=3,
        d_model=5,
        d_ff=7,
        e_layers=1,
        moving_avg=3,
        down_sampling_layers=0,
        down_sampling_window=1,
        dropout=0.0,
    )
    model = TimeMixerBackbone(config).eval()

    with torch.no_grad():
        output = model(torch.randn(2, 9, 1))

    assert config.scale_lengths == (9,)
    assert output.shape == (2, 3, 1)
    assert torch.isfinite(output).all()


def test_timemixer_average_pooling_matches_validated_scale_lengths() -> None:
    config = TimeMixerConfig(
        lookback=54,
        horizon=3,
        d_model=4,
        d_ff=8,
        e_layers=1,
        moving_avg=3,
        down_sampling_layers=3,
        down_sampling_window=2,
        dropout=0.0,
    )
    model = TimeMixerBackbone(config)

    scales = model._multi_scale_process_inputs(torch.randn(2, 54, 1))

    assert tuple(scale.shape[1] for scale in scales) == config.scale_lengths
    assert all(scale.shape[::2] == (2, 1) for scale in scales)


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"lookback": 0}, "lookback must be positive"),
        ({"horizon": 0}, "horizon must be positive"),
        ({"y_dim": 2}, "exactly one target channel"),
        ({"d_model": 0}, "d_model must be positive"),
        ({"d_ff": 0}, "d_ff must be positive"),
        ({"e_layers": 0}, "e_layers must be positive"),
        ({"moving_avg": 4}, "moving_avg must be odd"),
        ({"down_sampling_layers": -1}, "must be non-negative"),
        ({"down_sampling_window": 1}, "must be greater than 1"),
        (
            {"lookback": 7, "down_sampling_layers": 3},
            "collapses the coarsest sequence",
        ),
        ({"dropout": 1.0}, "dropout must be in"),
        ({"down_sampling_method": "max"}, "down_sampling_method"),
        ({"decomp_method": "dft_decomp"}, "decomp_method"),
        ({"channel_independence": False}, "channel_independence=True"),
        ({"use_norm": 1}, "use_norm must be a boolean"),
        ({"use_future_temporal_feature": True}, "future temporal features"),
        ({"use_exogenous_mode": True}, "endogenous-only"),
        ({"future_exo_dim": 1}, "endogenous-only"),
        ({"embed": "unknown"}, "embed must be one of"),
        ({"freq": "q"}, "freq must be one of"),
    ),
)
def test_timemixer_config_rejects_invalid_multiscale_contracts(
    override: dict,
    message: str,
) -> None:
    values = {
        "lookback": 16,
        "horizon": 4,
        "moving_avg": 3,
        "down_sampling_layers": 2,
        "down_sampling_window": 2,
    }
    values.update(override)

    with pytest.raises(ValueError, match=message):
        TimeMixerConfig(**values)

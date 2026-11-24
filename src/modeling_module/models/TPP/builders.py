"""Construct the research encoders without importing them on regular API use."""

from .contracts import EventModelConfig


def _shared_kwargs(config: EventModelConfig) -> dict[str, object]:
    return {
        "hidden_dim": 64,
        "train_log_mean": config.train_log_mean,
        "time_scale": config.time_scale,
        "time_initial_location": config.time_initial_location,
        "time_initial_scale": config.time_initial_scale,
        "time_sigma_floor": config.time_sigma_floor,
        "time_observation_contract": {
            "mode": "positive_integer_round_clamp_v1",
            "unit": config.time_unit,
            "top_code": config.top_code,
        },
    }


def build_rmtpp(config: EventModelConfig):
    from .rmtpp import RMTPP

    return RMTPP(**_shared_kwargs(config))


def build_phen(config: EventModelConfig):
    from .phen import PHEN

    return PHEN(max_seq_len=config.max_history_events, **_shared_kwargs(config))


def build_s2p2(config: EventModelConfig):
    from .s2p2 import S2P2

    return S2P2(**_shared_kwargs(config))

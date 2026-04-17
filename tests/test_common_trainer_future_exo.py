import torch

from modeling_module.training.adapters import DefaultAdapter
from modeling_module.training.config import TrainingConfig
from modeling_module.training.engine import CommonTrainer


def test_common_trainer_resolves_future_exo_without_nan_stat_attribute_error():
    cfg = TrainingConfig(
        device="cpu",
        amp_device="cpu",
        use_amp=False,
        lookback=4,
        horizon=2,
    )
    trainer = CommonTrainer(
        cfg=cfg,
        adapter=DefaultAdapter(),
        device="cpu",
    )

    x = torch.zeros(3, 4, 1)
    y = torch.zeros(3, 2)
    future_exo = torch.randn(3, 2, 5)

    resolved = trainer._resolve_future_exo(future_exo, x, y, device=torch.device("cpu"))

    assert resolved is not None
    assert resolved.shape == (3, 2, 5)
    assert resolved.device.type == "cpu"

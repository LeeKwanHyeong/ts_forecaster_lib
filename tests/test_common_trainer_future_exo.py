import types

import pytest
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


def test_production_refit_runs_exact_epochs_without_best_state_restore():
    cfg = TrainingConfig(
        device="cpu",
        amp_device="cpu",
        use_amp=False,
        lookback=4,
        horizon=2,
        epochs=3,
        training_mode="production_refit",
    )
    trainer = CommonTrainer(
        cfg=cfg,
        adapter=DefaultAdapter(),
        device="cpu",
        logger=lambda _: None,
    )
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()

    epoch_values: list[float] = []

    def fake_run_epoch(self, current_model, _loader, *, train):
        assert train is True
        self.opt.zero_grad()
        self.opt.step()
        value = float(len(epoch_values) + 1)
        with torch.no_grad():
            current_model.weight.fill_(value)
        epoch_values.append(value)
        return value

    trainer._run_epoch = types.MethodType(fake_run_epoch, trainer)
    trained = trainer.fit(model, [object()], None)

    assert epoch_values == [1.0, 2.0, 3.0]
    assert trained.weight.item() == pytest.approx(3.0)
    assert trainer.epochs_completed_ == 3
    assert trainer.final_train_loss_ == pytest.approx(3.0)
    assert trainer.best_loss_ is None
    assert trainer.validation_enabled_ is False


def test_qualification_still_requires_validation_loader():
    cfg = TrainingConfig(
        device="cpu",
        amp_device="cpu",
        use_amp=False,
        epochs=1,
    )
    trainer = CommonTrainer(
        cfg=cfg,
        adapter=DefaultAdapter(),
        device="cpu",
        logger=lambda _: None,
    )

    with pytest.raises(ValueError, match="requires a validation loader"):
        trainer.fit(torch.nn.Linear(1, 1), [object()], None)

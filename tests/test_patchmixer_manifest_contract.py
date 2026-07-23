from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import torch

from modeling_module.training.config import SpikeLossConfig, StageConfig, TrainingConfig
from modeling_module.utils.checkpoint import save_training_manifest


def test_patchmixer_records_global_best_val_loss_in_manifest(monkeypatch, tmp_path):
    patchmixer_train = importlib.import_module(
        "modeling_module.training.model_trainers.patchmixer_train"
    )
    stage_outcomes = iter(((0.25, 1.0), (0.75, 2.0)))

    class FakeCommonTrainer:
        def __init__(self, **kwargs):
            self.best_loss_, self.weight_value = next(stage_outcomes)

        def fit(self, model, train_loader, val_loader, *, tta_steps=0):
            with torch.no_grad():
                model.weight.fill_(self.weight_value)
            return model

    monkeypatch.setattr(patchmixer_train, "CommonTrainer", FakeCommonTrainer)
    monkeypatch.setattr(
        patchmixer_train,
        "amp_type_set",
        lambda cfg: ("cpu", False, torch.bfloat16),
    )

    model = torch.nn.Linear(1, 1, bias=False)
    train_cfg = TrainingConfig(
        device="cpu",
        epochs=1,
        use_amp=False,
        use_exogenous_mode=False,
        spike_loss=SpikeLossConfig(enabled=False),
    )

    result = patchmixer_train.train_patchmixer(
        model,
        train_loader=object(),
        val_loader=object(),
        device="cpu",
        train_cfg=train_cfg,
        stages=[
            StageConfig(epochs=1, spike_enabled=False),
            StageConfig(epochs=1, spike_enabled=False),
        ],
    )

    assert result["best_val_loss"] == pytest.approx(0.25)
    torch.testing.assert_close(result["model"].weight, torch.tensor([[1.0]]))

    manifest_path = save_training_manifest(
        tmp_path,
        results={"patchmixer": result},
    )
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    assert manifest["results"]["patchmixer"]["best_val_loss"] == pytest.approx(0.25)

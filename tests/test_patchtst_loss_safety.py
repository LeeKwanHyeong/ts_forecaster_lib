"""정답 스케일 축소와 실패한 학습을 0 loss로 보고하는 경로의 회귀 테스트."""

from __future__ import annotations

from dataclasses import fields
import inspect

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from modeling_module.models.PatchTST.common.configs import AttentionConfig, PatchTSTConfig
from modeling_module.models.PatchTST.self_supervised.PatchTST import (
    PatchTSTPretrainModel, _masked_recon_loss,
)
from modeling_module.models.PatchTST.supervised.PatchTST import PatchTSTModel
from modeling_module.training.adapters import DefaultAdapter
from modeling_module.training.config import StageConfig, TrainingConfig
from modeling_module.training.engine import CommonTrainer
from modeling_module.training.loss_safety import checked_loss_mean, require_finite_loss
from modeling_module.training.model_losses.loss_module import MAE
from modeling_module.training.model_trainers.patchtst_pretrain import (
    _eval_pretrain, train_patchtst_pretrain,
)
from modeling_module.training.model_trainers.patchtst_finetune import train_patchtst_finetune


def tiny_pretrain():
    cfg = PatchTSTConfig(
        lookback=16, horizon=4, patch_len=4, stride=4,
        d_model=8, d_ff=16, n_layers=1, dropout=0.0,
        norm="LayerNorm", future_exo_dim=0,
    )
    cfg.attn = AttentionConfig(n_heads=2, attn_dropout=0.0, proj_dropout=0.0)
    return PatchTSTPretrainModel(cfg)


def loader(size=4, *, batch_size=2, drop_last=False):
    return DataLoader(
        TensorDataset(torch.ones(size, 16, 1), torch.ones(size, 4, 1)),
        batch_size=batch_size, drop_last=drop_last,
    )


class PointModel(torch.nn.Module):
    def __init__(self, *, bad=None, bad_validation=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.bad = bad
        self.bad_validation = bad_validation

    def forward(self, x):
        result = x[:, :4, :] * self.weight
        if self.bad is not None and (not self.bad_validation or not self.training):
            result = result + self.bad
        return result


def trainer():
    return CommonTrainer(
        TrainingConfig(epochs=1, use_amp=False, loss=MAE(), use_exogenous_mode=False),
        DefaultAdapter(), device="cpu", logger=lambda *args: None,
    )


def eval_pretrain(model, batches):
    kwargs = dict(mask_ratio=0.5, loss_type="mse")
    # V2의 고정 검증 마스크 RNG 계약도 동일 테스트로 검증.
    if "eval_seed" in inspect.signature(_eval_pretrain).parameters:
        kwargs["eval_seed"] = 73
    return _eval_pretrain(model, batches, torch.device("cpu"), **kwargs)


def test_affine_disabled_and_fixed_target_scale():
    model = tiny_pretrain().eval()
    assert model.revin_layer.affine is False
    assert model.revin_layer.subtract_last is False
    assert model.revin_layer.use_std is True
    assert not any("affine_" in name for name, _ in model.named_parameters())
    assert not any(name.startswith("revin_layer.affine_") for name in model.state_dict())
    assert PatchTSTModel(model.cfg).revin_layer.affine is False
    x = torch.arange(32.0).reshape(2, 16, 1)
    mask = torch.tensor([[True, False, True, False]]).expand(2, -1)
    with torch.no_grad():
        model.backbone.pretrain_head.weight.zero_()
        model.backbone.pretrain_head.bias.zero_()
    out = model(x, patch_mask=mask)
    assert not out["patch_target"].requires_grad
    assert float(out["loss"].detach()) > 0.1
    target = out["patch_target"].detach().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(3):
        opt.zero_grad()
        out = model(x, patch_mask=mask)
        torch.testing.assert_close(out["patch_target"], target)
        out["loss"].backward()
        opt.step()


def test_new_pretrain_state_roundtrip_without_affine():
    first, second = tiny_pretrain(), tiny_pretrain()
    second.load_state_dict(first.state_dict(), strict=True)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_mask_ratio_is_rejected(ratio):
    with pytest.raises(ValueError, match="mask_ratio"):
        tiny_pretrain()._make_patch_mask(2, 1, ratio, torch.device("cpu"))


def test_random_mask_contains_at_least_one_patch_per_sample():
    mask = tiny_pretrain()._make_patch_mask(100, 1, 1e-12, torch.device("cpu"))
    assert mask.all()


def test_explicit_empty_mask_is_not_a_zero_loss():
    with pytest.raises(ValueError, match="at least one masked patch"):
        _masked_recon_loss(torch.zeros(2, 4, 4), torch.ones(2, 4, 4), torch.zeros(2, 4, dtype=torch.bool))


def test_genuine_zero_reconstruction_loss_is_valid():
    values = torch.ones(2, 4, 4)
    assert _masked_recon_loss(values, values, torch.ones(2, 4, dtype=torch.bool)).item() == 0.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_loss_is_rejected(bad):
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        require_finite_loss(torch.tensor(bad), context="unit test")


def test_empty_pretrain_validation_is_not_zero():
    with pytest.raises(ValueError, match="no batches"):
        eval_pretrain(tiny_pretrain(), loader(0))


@pytest.mark.parametrize("drop", [False, True])
def test_empty_pretrain_train_loader_rejected_before_update(drop):
    empty = loader(1 if drop else 0, batch_size=2, drop_last=drop)
    with pytest.raises(ValueError, match="no batches"):
        train_patchtst_pretrain(tiny_pretrain(), empty, loader(), device="cpu", train_cfg=TrainingConfig(device="cpu"))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_pretrain_nonfinite_loss_aborts_training_and_validation(bad):
    class BadReconstruction(PointModel):
        def forward(self, x, **kwargs):
            return {"loss": self.weight * 0 + bad}
    model = BadReconstruction()
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        train_patchtst_pretrain(model, loader(), device="cpu", stages=[StageConfig(epochs=1)], train_cfg=TrainingConfig(device="cpu", use_amp=False))
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        eval_pretrain(model, loader())


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_supervised_nonfinite_training_is_not_reported_as_zero(bad, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FloatingPointError, match="training: non-finite loss"):
        trainer().fit(PointModel(bad=bad), loader(), loader())


def test_supervised_nonfinite_validation_aborts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FloatingPointError, match="validation: non-finite loss"):
        trainer().fit(PointModel(bad=float("nan"), bad_validation=True), loader(), loader())


@pytest.mark.parametrize("which", ["training", "validation"])
def test_supervised_empty_loader_rejected_before_update(which):
    model = PointModel()
    train, val = (loader(0), loader()) if which == "training" else (loader(), loader(0))
    with pytest.raises(ValueError, match=which + ": loader contains no batches"):
        trainer().fit(model, train, val)
    assert model.weight.item() == 0.0


def test_no_observations_does_not_mean_zero_loss():
    with pytest.raises(ValueError, match="no valid batches"):
        checked_loss_mean(0.0, None, 0, context="validation")


def test_mean_uses_processed_batches_not_loader_length(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    class ShortLoader:
        def __len__(self):
            return 100
        def __iter__(self):
            yield torch.ones(2, 16, 1), torch.ones(2, 4, 1)
            yield torch.ones(2, 16, 1), torch.ones(2, 4, 1)
    instance = trainer()
    assert instance._run_epoch(PointModel(), ShortLoader(), train=False) == 1.0
    assert instance.last_epoch_batches_ == 2


def test_valid_supervised_training_still_updates_parameters(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model, instance = PointModel(), trainer()
    instance.fit(model, loader(), loader())
    assert model.weight.item() > 0.0
    assert instance.last_epoch_batches_ == instance.last_validation_batches_ == 2
    assert torch.isfinite(torch.tensor(instance.best_loss_))


def test_v2_production_refit_keeps_validation_disabled(tmp_path, monkeypatch):
    if "training_mode" not in {item.name for item in fields(TrainingConfig)}:
        pytest.skip("production_refit belongs to the V2 training contract")
    monkeypatch.chdir(tmp_path)
    instance, model = trainer(), PointModel()
    instance.cfg.training_mode = "production_refit"
    instance.fit(model, loader(), None)
    assert instance.validation_enabled_ is False
    assert model.weight.item() > 0.0


def test_pretrain_checkpoint_transfers_to_exogenous_finetuning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(143)
    pretrain = tiny_pretrain()
    x, y = torch.rand(4, 16, 1) + 1, torch.rand(4, 4, 1) + 1
    rows = [(x[i], y[i], str(i), torch.ones(4, 1),
             torch.empty(16, 0), torch.empty(16, 0, dtype=torch.long)) for i in range(4)]

    def collate(batch):
        columns = tuple(zip(*batch))
        return (torch.stack(columns[0]), torch.stack(columns[1]), list(columns[2]),
                torch.stack(columns[3]), torch.stack(columns[4]), torch.stack(columns[5]))

    batches = DataLoader(rows, batch_size=2, collate_fn=collate)
    cfg = TrainingConfig(device="cpu", epochs=1, use_amp=False, loss=MAE(), use_exogenous_mode=True)
    stage = [StageConfig(epochs=1, lr=1e-3, spike_enabled=False)]
    train_patchtst_pretrain(pretrain, batches, batches, device="cpu", train_cfg=cfg,
                           stages=stage, save_dir=str(tmp_path / "pretrain"))
    checkpoint = tmp_path / "pretrain" / "patchtst_pretrain_best.pt"
    assert checkpoint.is_file()
    assert not any("revin_layer.affine_" in key for key in torch.load(checkpoint, weights_only=False)["state_dict"])
    pretrain.cfg.future_exo_dim = 1
    pretrain.cfg.loss = MAE()
    model = PatchTSTModel(pretrain.cfg)
    result = train_patchtst_finetune(model, batches, batches, device="cpu", train_cfg=cfg,
                                    stages=stage, pretrain_ckpt_path=str(checkpoint))
    assert result["model"].revin_layer.affine is False
    assert torch.isfinite(torch.tensor(result["best_val_loss"]))
    assert torch.isfinite(model(x[:1], future_exo=torch.ones(1, 4, 1))).all()

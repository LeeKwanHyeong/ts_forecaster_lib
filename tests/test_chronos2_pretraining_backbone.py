"""공식 가중치 없이 native 모델 초기화·마스크·역전파를 검증하는 CPU 테스트."""

from dataclasses import replace
import importlib.util
import subprocess
import sys

import pytest
import torch

from modeling_module.models.Chronos2.training import Chronos2TrainableConfig, Chronos2TrainingBatch
from modeling_module.models.Chronos2.trainable_backbone import Chronos2TrainableBackbone
import modeling_module.models.Chronos2.trainable_backbone as module


HAS_NATIVE = all(importlib.util.find_spec(name) is not None for name in ("transformers", "einops"))
native_required = pytest.mark.skipif(not HAS_NATIVE, reason="선택 의존성 chronos2-native가 없어 내부 소스 테스트 생략")


def tiny_config(**changes):
    return replace(Chronos2TrainableConfig(d_model=16, d_ff=32, num_layers=1, num_heads=2,
        d_kv=8, dropout_rate=0, quantile_levels=(0.1, 0.5, 0.9)), **changes)


def batch():
    return Chronos2TrainingBatch(
        context=torch.linspace(3, 20, 52).repeat(2, 1),
        future_target=torch.linspace(22, 30, 27).repeat(2, 1))


@pytest.mark.parametrize("changes", [
    {"context_length": 0}, {"horizon": True}, {"patch_size": -1}, {"d_model": 15},
    {"num_layers": 0}, {"seed": -1}, {"seed": True}, {"dropout_rate": 1.0},
    {"initializer_factor": 0}, {"device": "auto"}, {"use_arcsinh": 1},
    {"quantile_levels": (0.1, 0.9)}, {"quantile_levels": (0.5, 0.1)},
    {"quantile_levels": (float("nan"), 0.5)},
])
def test_invalid_trainable_config_rejected(changes):
    with pytest.raises((TypeError, ValueError)):
        replace(tiny_config(), **changes)


def test_training_module_import_does_not_import_optional_runtime():
    code = '''
import importlib.abc, sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'chronos', 'transformers', 'huggingface_hub'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockOptional())
from modeling_module.models.Chronos2.trainable_backbone import Chronos2TrainableBackbone
from modeling_module.models.Chronos2.training import Chronos2TrainableConfig
assert Chronos2TrainableConfig().horizon == 27
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_missing_dependency_error_is_actionable(monkeypatch):
    def missing(name):
        raise module.importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(module.importlib.metadata, "version", missing)
    with pytest.raises(ImportError, match="chronos2"):
        Chronos2TrainableBackbone(tiny_config())


@native_required
def test_random_constructor_never_reads_pretrained_weights_or_downloads(monkeypatch):
    from modeling_module.models.Chronos2.native.model import Chronos2Model
    from modeling_module.models.Chronos2.native.config import Chronos2CoreConfig
    import huggingface_hub
    def forbidden(*args, **kwargs):
        raise AssertionError("pretrained checkpoint or network requested")
    for target, name in ((Chronos2Model, "from_pretrained"),
                         (Chronos2CoreConfig, "from_pretrained"), (huggingface_hub, "snapshot_download"),
                         (huggingface_hub, "hf_hub_download")):
        monkeypatch.setattr(target, name, forbidden)
    model = Chronos2TrainableBackbone(tiny_config())
    info = model.provenance()
    assert info["initialization_source"] == "random" and not info["pretrained_weights_loaded"]
    assert info["parameters"] == info["trainable_parameters"] > 0
    with pytest.raises(TypeError):
        Chronos2TrainableConfig(model_path="official")


@native_required
def test_seed_reproducibility_and_callers_cpu_rng_are_preserved():
    state = torch.get_rng_state().clone()
    first = Chronos2TrainableBackbone(tiny_config())
    torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
    second = Chronos2TrainableBackbone(tiny_config())
    third = Chronos2TrainableBackbone(tiny_config(seed=43))
    assert first.initial_parameters_sha256 == second.initial_parameters_sha256
    assert first.initial_parameters_sha256 != third.initial_parameters_sha256
    assert len(first.initial_parameters_sha256) == 64


@native_required
def test_forward_matches_native_loss_and_one_optimizer_step_updates_parameters():
    model = Chronos2TrainableBackbone(tiny_config())
    values = batch()
    result = model(values)
    direct = model.native_model(**values.native_inputs(model.cfg, "cpu"), num_output_patches=2)
    torch.testing.assert_close(result.loss, direct.loss)
    assert result.point.shape == (2, 27, 1) and result.quantiles.shape == (2, 27, 3)
    assert result.raw_prediction_length == 32 and result.observed_target_points == 54
    assert result.loss.requires_grad and torch.isfinite(result.loss)
    parameter = model.native_model.output_patch_embedding.output_layer.weight
    before = parameter.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    result.loss.backward()
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(parameter, before)


@native_required
def test_hidden_history_is_removed_before_scaling_and_targets_do_not_leak():
    model = Chronos2TrainableBackbone(tiny_config()).eval()
    values = batch()
    mask = torch.ones_like(values.context, dtype=torch.bool)
    mask[:, :5] = False
    first = model(replace(values, context_mask=mask))
    changed = values.context.clone()
    changed[:, :5] = 1e8
    second = model(replace(values, context=changed, context_mask=mask))
    third = model(replace(values, context_mask=mask, future_target=values.future_target * 100))
    torch.testing.assert_close(first.point, second.point, rtol=0, atol=0)
    torch.testing.assert_close(first.loss, second.loss, rtol=0, atol=0)
    torch.testing.assert_close(first.point, third.point, rtol=0, atol=0)
    assert first.loss != third.loss
    assert values.context[0, 0].item() == 3.0


@native_required
def test_h27_padding_and_missing_targets_are_masked_out_of_native_loss():
    model = Chronos2TrainableBackbone(tiny_config()).eval()
    values = batch()
    inputs = values.native_inputs(model.cfg, "cpu")
    normal = model(values)
    extended = {**inputs, "future_target": torch.cat([inputs["future_target"], torch.full((2, 5), 99999.)], 1),
                "future_target_mask": torch.cat([inputs["future_target_mask"], torch.zeros(2, 5, dtype=torch.bool)], 1)}
    padded = model.native_model(**extended, num_output_patches=2)
    torch.testing.assert_close(normal.loss, padded.loss, rtol=0, atol=0)
    mask = torch.ones_like(values.future_target, dtype=torch.bool)
    mask[:, 3] = False
    a = model(replace(values, future_target_mask=mask))
    target = values.future_target.clone()
    target[:, 3] = 999999.
    b = model(replace(values, future_target=target, future_target_mask=mask))
    torch.testing.assert_close(a.loss, b.loss, rtol=0, atol=0)
    assert a.observed_target_points == 52


@native_required
def test_independent_groups_cannot_share_other_series_values():
    model = Chronos2TrainableBackbone(tiny_config()).eval()
    values = batch()
    a = model(values)
    context = values.context.clone()
    context[1] *= 1000
    b = model(replace(values, context=context))
    torch.testing.assert_close(a.point[0], b.point[0], rtol=0, atol=0)
    with pytest.raises(ValueError, match="독립"):
        model(replace(values, group_ids=torch.tensor([1, 1])))
    model.train()
    assert model.training and model.native_model.training
    model.eval()
    assert not model.training and not model.native_model.training


def test_zero_demand_is_observed_not_missing_and_inputs_are_not_mutated():
    values = Chronos2TrainingBatch(torch.zeros(2, 52), torch.zeros(2, 27))
    prepared = values.native_inputs(tiny_config(), "cpu")
    assert prepared["context_mask"].sum() == 104 and prepared["future_target_mask"].sum() == 54
    assert prepared["context"].eq(0).all() and prepared["future_target"].eq(0).all()
    prepared["context"][0, 0] = 999
    assert values.context.eq(0).all()


@native_required
def test_default_quantiles_and_all_zero_demand_have_finite_native_outputs():
    model = Chronos2TrainableBackbone(replace(tiny_config(),
        quantile_levels=Chronos2TrainableConfig().quantile_levels)).eval()
    result = model(Chronos2TrainingBatch(torch.zeros(2, 52), torch.zeros(2, 27)))
    assert result.quantiles.shape == (2, 27, 21)
    assert result.observed_target_points == 54
    assert torch.isfinite(result.loss) and torch.isfinite(result.quantiles).all()


def test_explicitly_observed_nan_is_not_a_zero_demand_label():
    values = batch()
    target = values.future_target.clone()
    target[0, 0] = torch.nan
    with pytest.raises(ValueError, match="유한한"):
        replace(values, future_target=target,
                future_target_mask=torch.ones_like(target, dtype=torch.bool)).native_inputs(tiny_config(), "cpu")


@pytest.mark.parametrize("field,value", [
    ("context", torch.ones(2, 51)), ("future_target", torch.ones(2, 26)),
    ("context", torch.full((2, 52), float("inf"))), ("context", torch.full((2, 52), float("nan"))),
    ("context", -torch.ones(2, 52)), ("future_target", -torch.ones(2, 27)),
    ("future_target_mask", torch.zeros(2, 27, dtype=torch.bool)),
    ("context_mask", torch.ones(2, 52)), ("future_target", torch.ones(2, 27, dtype=torch.int64)),
    ("group_ids", torch.tensor([0., 1.])), ("group_ids", torch.tensor([0])),
])
def test_invalid_batch_cannot_succeed_with_false_zero_loss(field, value):
    with pytest.raises((TypeError, ValueError)):
        replace(batch(), **{field: value}).native_inputs(tiny_config(), "cpu")

"""다운로드 없이 입력 계약, 지연 로딩, 원 출력 변환을 검사하는 테스트."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from modeling_module.models.Chronos2 import Chronos2Backbone, Chronos2Config, Chronos2Model
from modeling_module.models.Chronos2.configs import CHRONOS_PACKAGE_VERSION
from modeling_module.models.Chronos2.contracts import prepare_inputs
import modeling_module.models.Chronos2.backbone as backbone_module


class FakeNativeModel(nn.Module):
    """실제 사전학습 모델과 구분한 shape 검사 전용 대역."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    @property
    def device(self):
        return self.weight.device

    def encode(self, context, num_output_patches):
        patches = (context.shape[-1] + 15) // 16
        hidden = torch.ones(context.shape[0], patches + 1 + num_output_patches, 8)
        return SimpleNamespace(last_hidden_state=hidden), (torch.zeros(len(context), 1), torch.ones(len(context), 1)), None, patches


class FakePipeline:
    model_context_length = 8192
    model_prediction_length = 1024
    model_output_patch_size = 16
    quantiles = [0.1, 0.5, 0.9]

    def __init__(self):
        self.model = FakeNativeModel()
        self.calls = []
        self.bad_output = False

    def predict_quantiles(self, **kwargs):
        self.calls.append({**kwargs, "grad_enabled": torch.is_grad_enabled()})
        length = kwargs["prediction_length"]
        values = torch.tensor([10.0, 20.0, 30.0]).expand(1, length, 3).clone()
        if self.bad_output:
            values[0, 0, 0] = float("nan")
        return [values.clone() for _ in kwargs["inputs"]], [torch.full((1, length), 999.0) for _ in kwargs["inputs"]]


@pytest.fixture
def fake_model():
    model = Chronos2Model()
    pipeline = FakePipeline()
    model.backbone._pipeline = pipeline
    model.backbone._model = pipeline.model.requires_grad_(False).eval()
    return model, pipeline


@pytest.mark.parametrize("changes", [
    {"horizon": 0}, {"context_length": True}, {"batch_size": -1}, {"revision": "main"},
    {"quantile_levels": [0.1, 0.9]}, {"quantile_levels": [0.5, 0.1]},
    {"quantile_levels": [0.5, 0.5]}, {"quantile_levels": [0.0, 0.5]},
    {"quantile_levels": [0.5, float("nan")]}, {"device": "auto"}, {"dtype": "int8"},
    {"model_path": ""}, {"expected_weights_sha256": "bad"}, {"local_files_only": 1},
])
def test_invalid_config_rejected(changes):
    with pytest.raises((ValueError, TypeError)):
        Chronos2Config(**changes)


def test_default_is_weekly_27_and_no_model_load():
    model = Chronos2Model()
    assert model.horizon == 27 and model.cfg.context_length == 52
    assert model.cfg.local_files_only and not model.backbone.is_loaded
    assert list(model.parameters()) == []


@pytest.mark.parametrize("x", [torch.ones(52), torch.ones(2, 52), torch.ones(2, 52, 2),
    torch.ones(0, 52, 1), torch.ones(2, 0, 1), torch.full((2, 52, 1), float("nan")),
    torch.full((2, 52, 1), float("inf")), torch.ones(2, 52, 1, dtype=torch.bool)])
def test_invalid_target_rejected_before_loading(x):
    model = Chronos2Model()
    with pytest.raises((ValueError, TypeError)):
        model.predict(x)
    assert not model.backbone.is_loaded


def test_missing_and_zero_demand_remain_distinct_and_input_unchanged(fake_model):
    model, pipeline = fake_model
    x = torch.ones(2, 60, 1)
    x[0, -1] = float("nan")
    x[1] = 0
    original = x.clone()
    result = model.predict(x)
    assert result.point.shape == (2, 27, 1) and result.quantiles.shape == (2, 27, 3)
    assert torch.isnan(pipeline.calls[0]["inputs"][0]["target"][-1])
    assert pipeline.calls[0]["inputs"][1]["target"].eq(0).all()
    torch.testing.assert_close(x, original, equal_nan=True)


def test_median_is_selected_and_cross_learning_disabled(fake_model):
    model, pipeline = fake_model
    result = model.predict(np.ones((2, 52, 1)))
    assert result.point.eq(20).all()  # 대역의 mean=999를 중앙값으로 오인하지 않음.
    assert result.metadata["point_method"] == "median"
    assert not pipeline.calls[0]["cross_learning"] and not pipeline.calls[0]["grad_enabled"]
    model.train()
    assert not model.backbone.training and not pipeline.model.training
    assert not any(p.requires_grad for p in model.parameters())


def test_covariates_are_trimmed_with_target(fake_model):
    model, pipeline = fake_model
    past = torch.arange(60).repeat(2, 1)
    future = torch.ones(2, 27)
    model.predict(torch.ones(2, 60, 1), past_covariates={"calendar": past}, future_covariates={"calendar": future})
    data = pipeline.calls[0]["inputs"][0]
    assert data["target"].shape == (52,)
    assert data["past_covariates"]["calendar"].tolist() == list(range(8, 60))
    assert data["future_covariates"]["calendar"].shape == (27,)


@pytest.mark.parametrize("past,future", [
    ({}, {"calendar": torch.ones(2, 27)}),
    ({"calendar": torch.ones(2, 51)}, {}),
    ({"calendar": torch.ones(2, 52)}, {"calendar": torch.ones(2, 26)}),
    ({"calendar": torch.ones(2, 52)}, {"calendar": torch.full((2, 27), float("nan"))}),
    ({"calendar": torch.full((2, 52), float("nan"))}, {}),
    ({"": torch.ones(2, 52)}, {}),
])
def test_invalid_covariates_rejected(past, future):
    with pytest.raises(ValueError):
        prepare_inputs(torch.ones(2, 52, 1), Chronos2Config(), past_covariates=past, future_covariates=future)


def test_empty_recent_context_and_small_covariate_batch_rejected():
    x = torch.full((1, 60, 1), float("nan"))
    x[:, 0] = 10
    with pytest.raises(ValueError):
        prepare_inputs(x, Chronos2Config())
    with pytest.raises(ValueError):
        prepare_inputs(torch.ones(1, 52, 1), Chronos2Config(batch_size=1), past_covariates={"feature": torch.ones(1, 52)})


def test_nonfinite_provider_result_rejected(fake_model):
    model, pipeline = fake_model
    pipeline.bad_output = True
    with pytest.raises(RuntimeError, match="출력"):
        model.predict(torch.ones(1, 52, 1))


def test_future_features_are_two_patches_for_27_weeks(fake_model):
    model, _ = fake_model
    features = model.encode_future_tokens(torch.ones(2, 52, 1))
    assert features.future_tokens.shape == (2, 2, 8)
    assert features.horizon == 27 and features.output_patch_size == 16


def test_import_does_not_require_chronos_or_download():
    code = '''
import importlib.abc, sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"chronos", "huggingface_hub", "transformers"}:
            raise AssertionError("optional runtime was imported: " + fullname)
sys.meta_path.insert(0, BlockOptional())
from modeling_module.models.Chronos2 import Chronos2Model
assert not Chronos2Model().backbone.is_loaded
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_local_load_checks_checksum_and_freezes_model(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"chronos_pipeline_class": "Chronos2Pipeline"}))
    (tmp_path / "model.safetensors").write_bytes(b"unit-test-only")
    digest = hashlib.sha256(b"unit-test-only").hexdigest()
    cfg = Chronos2Config(model_path=str(tmp_path), expected_weights_sha256=digest,
                        expected_config_sha256=hashlib.sha256((tmp_path / "config.json").read_bytes()).hexdigest())
    pipeline = FakePipeline()
    calls = []
    def from_pretrained(path, **kwargs):
        calls.append((path, kwargs))
        return pipeline
    module = ModuleType("chronos")
    module.Chronos2Pipeline = SimpleNamespace(from_pretrained=from_pretrained)
    monkeypatch.setitem(sys.modules, "chronos", module)
    monkeypatch.setattr(backbone_module.importlib.metadata, "version", lambda name: CHRONOS_PACKAGE_VERSION)
    backbone = Chronos2Backbone(cfg).load()
    backbone.load()
    assert len(calls) == 1 and calls[0][1]["local_files_only"]
    assert not calls[0][1]["trust_remote_code"]
    assert not any(p.requires_grad for p in backbone.parameters())
    assert backbone.provenance()["weights_verified"]
    assert backbone.provenance()["config_verified"]
    with pytest.raises(ValueError, match="SHA-256"):
        Chronos2Backbone(replace(cfg, expected_weights_sha256="a" * 64)).load()


def test_missing_optional_dependency_has_actionable_error(monkeypatch):
    def missing(name):
        raise backbone_module.importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(backbone_module.importlib.metadata, "version", missing)
    with pytest.raises(ImportError, match="chronos2"):
        Chronos2Model().load()

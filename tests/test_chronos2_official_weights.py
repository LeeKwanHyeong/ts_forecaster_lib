"""명시한 로컬 공식 가중치로만 실행하는 실제 모델 계약 테스트."""

from __future__ import annotations

import os

import pytest
import torch

from modeling_module.models.Chronos2 import Chronos2Config, Chronos2Model
from modeling_module.models.Chronos2.contracts import prepare_inputs


@pytest.fixture(scope="module")
def official_model():
    path = os.environ.get("TSF_CHRONOS2_MODEL_DIR")
    if not path:
        pytest.skip("공식 가중치 테스트는 TSF_CHRONOS2_MODEL_DIR 지정 시에만 실행")
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = Chronos2Model(Chronos2Config(model_path=path)).load()
        yield model
    finally:
        torch.set_num_threads(previous)


def example_context():
    t = torch.arange(52, dtype=torch.float32)
    return torch.stack([20 + 5 * torch.sin(t / 8), 5 + t * 0.3])[:, :, None]


def test_real_weights_zero_shot_27_and_native_pipeline_parity(official_model):
    model = official_model
    x = example_context()
    forecast = model.predict(x)
    assert forecast.point.shape == (2, 27, 1)
    assert forecast.quantiles.shape == (2, 27, 3)
    assert torch.isfinite(forecast.quantiles).all()
    assert forecast.metadata["weights_verified"]
    _, inputs = prepare_inputs(x, model.cfg)
    native, _ = model.backbone._pipeline.predict_quantiles(inputs=inputs, prediction_length=27,
        quantile_levels=[0.1, 0.5, 0.9], context_length=52, batch_size=32, cross_learning=False, limit_prediction_length=True)
    torch.testing.assert_close(forecast.quantiles, torch.stack(native)[:, 0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(forecast.point[:, :, 0], forecast.quantiles[:, :, 1])
    assert not any(p.requires_grad for p in model.parameters())


def test_real_predictions_do_not_mix_parts(official_model):
    x = example_context()
    combined = official_model(x)
    separate = torch.cat([official_model(x[index:index + 1]) for index in range(2)])
    torch.testing.assert_close(combined, separate, rtol=2e-5, atol=2e-5)


def test_real_future_tokens_are_head_inputs(official_model):
    model = official_model
    x = example_context()
    features = model.encode_future_tokens(x)
    assert features.future_tokens.shape == (2, 2, 768)
    assert torch.isfinite(features.future_tokens).all()
    # 공식 head와 역정규화만 적용하면 원래 예측을 복원할 수 있는지 확인.
    native = model.backbone._model
    with torch.inference_mode():
        values = native.output_patch_embedding(features.future_tokens)
        values = values.reshape(2, 2, native.num_quantiles, 16).permute(0, 2, 1, 3).reshape(2, native.num_quantiles, 32)
        values = native.instance_norm.inverse(values.reshape(2, -1), (features.location, features.scale))
        values = values.reshape(2, native.num_quantiles, 32)[:, :, :27]
        median = values[:, native.chronos_config.quantiles.index(0.5)]
    torch.testing.assert_close(median, model(x)[:, :, 0], rtol=1e-5, atol=1e-5)


def test_real_known_future_covariate_and_missing_context(official_model):
    x = example_context()
    x[:, 0] = float("nan")
    past = {"calendar": torch.sin(torch.arange(52, dtype=torch.float32))[None].repeat(2, 1)}
    future = {"calendar": torch.sin(torch.arange(52, 79, dtype=torch.float32))[None].repeat(2, 1)}
    forecast = official_model.predict(x, past_covariates=past, future_covariates=future)
    assert forecast.point.shape == (2, 27, 1) and torch.isfinite(forecast.point).all()


def test_real_zero_and_constant_series_remain_finite(official_model):
    x = torch.stack([torch.zeros(52), torch.ones(52) * 7])[:, :, None]
    assert torch.isfinite(official_model(x)).all()

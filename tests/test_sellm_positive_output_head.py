from __future__ import annotations

import hashlib
from types import MethodType

import pytest
import torch

from modeling_module.api import load_predictor
from modeling_module.models.SELLM.SELLM import SELLMModel
from modeling_module.models.SELLM.configs import SELLMConfig
from modeling_module.models.SELLM.output_heads import ZeroInflatedSoftplusHead
from modeling_module.utils.checkpoint import save_model


def _paper_config(**overrides) -> SELLMConfig:
    values = {
        "lookback": 8,
        "horizon": 5,
        "y_dim": 1,
        "future_exo_dim": 0,
        "architecture_variant": "paper_v1",
        "token_len": 2,
        "d_model": 8,
        "n_heads": 2,
        "dropout": 0.0,
        "mlp_hidden_dim": 8,
        "semantic_vocab_size": 6,
        "semantic_top_k": 2,
        "tscc_latent_dim": 2,
        "tscc_hidden_dim": 4,
        "tscc_kl_weight": 0.0,
        "use_pretrained_llm": False,
        "fallback_layers": 1,
        "d_ff": 16,
        "use_norm": False,
        "final_nonneg": False,
    }
    values.update(overrides)
    return SELLMConfig(**values)


def _baseline_input() -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, steps=16).reshape(2, 8, 1)


def test_identity_head_preserves_paper_v1_output_parameter_and_state_baseline():
    torch.manual_seed(31)
    model = SELLMModel(_paper_config()).eval()

    with torch.no_grad():
        output = model(_baseline_input())

    expected = torch.tensor(
        [
            0.7570339441,
            -0.0068285614,
            0.8053463697,
            0.1064155623,
            0.8349062800,
            0.8800234199,
            0.2088552713,
            0.8594478965,
            0.1524332166,
            0.8562584519,
        ]
    ).reshape(2, 5, 1)
    schema = hashlib.sha256("\n".join(model.state_dict()).encode()).hexdigest()

    assert model.output_head_mode == "identity"
    assert model.positive_output_head is None
    assert sum(parameter.numel() for parameter in model.parameters()) == 1626
    assert schema == "f36143f80135e2f06db64bc13e0435bf188b54ad4d65d3c76a02299654dadc5b"
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("output_head_mode", "relu", "output_head_mode"),
        ("output_head_hidden_dim", 0, "hidden_dim"),
        ("output_head_softplus_beta", 0.0, "softplus_beta"),
        (
            "output_head_initial_nonzero_probability",
            1.0,
            "initial_nonzero_probability",
        ),
    ],
)
def test_positive_output_config_rejects_invalid_values(field, value, message):
    with pytest.raises(ValueError, match=message):
        _paper_config(**{field: value})


def test_positive_output_config_rejects_double_constraints():
    with pytest.raises(ValueError, match="final_nonneg"):
        _paper_config(output_head_mode="softplus", final_nonneg=True)
    with pytest.raises(ValueError, match="negative_output_penalty_weight"):
        _paper_config(
            output_head_mode="zero_inflated_softplus",
            negative_output_penalty_weight=0.1,
        )


def test_softplus_head_is_parameter_free_nonnegative_and_differentiable():
    model = SELLMModel(
        _paper_config(output_head_mode="softplus", output_head_softplus_beta=2.0)
    )
    raw = torch.tensor([[[-3.0], [0.0], [2.0], [-1.0], [4.0]]], requires_grad=True)
    history = torch.zeros(1, 8, 1)

    output = model._apply_output_head(raw, history=history)
    output.sum().backward()

    assert model.positive_output_head is None
    assert torch.all(output >= 0.0)
    assert raw.grad is not None
    assert torch.all(raw.grad > 0.0)


def test_zero_inflated_head_uses_history_and_propagates_all_gradients():
    torch.manual_seed(41)
    head = ZeroInflatedSoftplusHead(
        horizon=3,
        hidden_dim=5,
        softplus_beta=1.0,
        initial_nonzero_probability=0.4,
    )
    raw = torch.tensor([[[-2.0], [0.0], [2.0]]], requires_grad=True)
    dense_history = torch.ones(1, 8, 1)
    sparse_history = torch.tensor(
        [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [2.0]]]
    )

    dense = head(raw, dense_history)
    sparse = head(raw, sparse_history)
    (dense.sum() + sparse.sum()).backward()

    assert torch.all(dense >= 0.0)
    assert torch.all(sparse >= 0.0)
    assert not torch.equal(dense, sparse)
    assert raw.grad is not None and torch.all(raw.grad > 0.0)
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0.0, name


def test_zero_inflated_head_applies_after_demand_space_denormalization():
    model = SELLMModel(
        _paper_config(
            use_norm=True,
            output_head_mode="zero_inflated_softplus",
        )
    )
    captured: dict[str, torch.Tensor] = {}

    def fixed_raw_forecast(self, normalized):
        captured["normalized"] = normalized.detach().clone()
        return torch.full(
            (normalized.size(0), self.horizon, normalized.size(2)),
            -2.0,
            dtype=normalized.dtype,
            device=normalized.device,
        )

    model._paper_forecast = MethodType(fixed_raw_forecast, model)
    history = torch.arange(1.0, 9.0).reshape(1, 8, 1)
    output = model(history)

    assert captured["normalized"].mean().abs() < 1e-6
    assert torch.all(output >= 0.0)
    assert output.shape == (1, 5, 1)


def test_zero_inflated_head_strict_save_load_restores_identical_prediction(tmp_path):
    torch.manual_seed(43)
    config = _paper_config(output_head_mode="zero_inflated_softplus")
    model = SELLMModel(config).eval()
    value = _baseline_input().clamp_min(0.0)
    with torch.no_grad():
        expected = model(value)
    path = tmp_path / "sellm-positive.pt"

    save_model(
        model,
        model.cfg,
        str(path),
        extra_meta={"model_key": "sellm_base", "family": "sellm"},
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    predictor = load_predictor(str(path), device="cpu", strict=True)
    with torch.no_grad():
        restored = predictor.model(value)

    assert checkpoint["config"]["output_head_mode"] == "zero_inflated_softplus"
    assert checkpoint["meta"]["output_head_mode"] == "zero_inflated_softplus"
    assert any(
        key.startswith("positive_output_head.occurrence_gate")
        for key in checkpoint["state_dict"]
    )
    assert predictor.model.output_head_mode == "zero_inflated_softplus"
    torch.testing.assert_close(restored, expected, rtol=0.0, atol=0.0)


def test_checkpoint_without_output_head_fields_restores_identity_strictly(tmp_path):
    torch.manual_seed(47)
    model = SELLMModel(_paper_config()).eval()
    value = _baseline_input()
    with torch.no_grad():
        expected = model(value)
    path = tmp_path / "sellm-legacy-paper.pt"
    save_model(
        model,
        model.cfg,
        str(path),
        extra_meta={"model_key": "sellm_base", "family": "sellm"},
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    fields = (
        "output_head_mode",
        "output_head_hidden_dim",
        "output_head_softplus_beta",
        "output_head_initial_nonzero_probability",
    )
    for field in fields:
        checkpoint["config"].pop(field, None)
        checkpoint["cfg_state"].pop(field, None)
    checkpoint["meta"].pop("output_head_mode", None)
    torch.save(checkpoint, path)

    predictor = load_predictor(str(path), device="cpu", strict=True)
    with torch.no_grad():
        restored = predictor.model(value)

    assert predictor.model.output_head_mode == "identity"
    assert predictor.model.positive_output_head is None
    torch.testing.assert_close(restored, expected, rtol=0.0, atol=0.0)


def test_icl_forward_uses_positive_output_head():
    torch.manual_seed(53)
    model = SELLMModel(
        _paper_config(
            horizon=4,
            icl_enabled=True,
            output_head_mode="zero_inflated_softplus",
        )
    )
    demonstrations = torch.rand(1, 2, 8, 1)
    targets = torch.rand(1, 2, 4, 1)
    query = torch.rand(1, 8, 1)

    output = model.forward_icl(
        demonstration_contexts=demonstrations,
        demonstration_targets=targets,
        query_context=query,
        prompt_mask=torch.ones(1, 2, dtype=torch.bool),
    )

    assert output.shape == (1, 4, 1)
    assert torch.isfinite(output).all()
    assert torch.all(output >= 0.0)

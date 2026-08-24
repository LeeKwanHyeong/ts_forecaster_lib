from __future__ import annotations

import hashlib

import torch

from modeling_module.api import load_predictor
from modeling_module.models.SELLM.SELLM import SELLMModel
from modeling_module.models.SELLM.configs import SELLMConfig
from modeling_module.utils.checkpoint import save_model


def _legacy_config() -> SELLMConfig:
    return SELLMConfig(
        lookback=8,
        horizon=4,
        y_dim=1,
        future_exo_dim=0,
        architecture_variant="legacy_v1",
        token_len=2,
        d_model=8,
        n_heads=2,
        dropout=0.0,
        mlp_hidden_dim=8,
        semantic_vocab_size=6,
        semantic_top_k=2,
        tscc_latent_dim=2,
        tscc_hidden_dim=4,
        tscc_kl_weight=0.0,
        use_pretrained_llm=False,
        fallback_layers=1,
        d_ff=16,
        head_hidden_dim=6,
        use_norm=False,
        final_nonneg=False,
    )


def _legacy_input() -> torch.Tensor:
    return torch.tensor(
        [[[0.5], [1.0], [-0.5], [2.0], [0.0], [1.5], [-1.0], [0.25]]],
        dtype=torch.float32,
    )


def test_sellm_legacy_output_parameter_and_state_schema_baseline():
    torch.manual_seed(1234)
    model = SELLMModel(_legacy_config()).eval()

    with torch.no_grad():
        output = model(_legacy_input())

    expected = torch.tensor(
        [[[-0.0241680890], [-0.0102943890], [-0.2261065245], [0.1700636148]]]
    )
    assert torch.allclose(output, expected, atol=1e-6, rtol=1e-6)
    assert sum(parameter.numel() for parameter in model.parameters()) == 1510
    assert sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    ) == 1510

    schema = "\n".join(model.state_dict()).encode("utf-8")
    assert hashlib.sha256(schema).hexdigest() == (
        "5d441c65b97108a0fe8d20100a1879fe09a076dedf9c049985ac2882315c86a2"
    )


def test_sellm_legacy_gradient_reaches_each_trainable_stage():
    torch.manual_seed(1234)
    model = SELLMModel(_legacy_config()).train()
    torch.manual_seed(5678)

    model(_legacy_input()).square().mean().backward()

    named_parameters = dict(model.named_parameters())
    expected_gradient_parameters = (
        "ts_encoder.net.0.weight",
        "tscc.cross_attn.in_proj_weight",
        "tscc.vae.fc_mu.weight",
        "tscc.fusion.gate.0.weight",
        "fallback_encoder.layers.0.self_attn.in_proj_weight",
        "pool_head.0.weight",
        "pool_head.3.weight",
    )
    for name in expected_gradient_parameters:
        gradient = named_parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert gradient.abs().sum() > 0, name


def test_sellm_legacy_fallback_strict_save_load_predict(tmp_path):
    torch.manual_seed(1234)
    model = SELLMModel(_legacy_config()).eval()
    checkpoint_path = tmp_path / "sellm_base.pt"
    save_model(
        model,
        model.cfg,
        str(checkpoint_path),
        extra_meta={"model_key": "sellm_base", "family": "sellm"},
    )

    predictor = load_predictor(str(checkpoint_path), device="cpu", strict=True)
    with torch.no_grad():
        expected = model(_legacy_input())
        restored = predictor.model(_legacy_input())

    assert predictor.model_key == "sellm_base"
    assert predictor.config["architecture_variant"] == "legacy_v1"
    assert tuple(model.state_dict()) == tuple(predictor.model.state_dict())
    assert torch.equal(expected, restored)


def test_sellm_checkpoint_without_variant_restores_as_legacy_v1(tmp_path):
    torch.manual_seed(1234)
    model = SELLMModel(_legacy_config()).eval()
    checkpoint_path = tmp_path / "sellm_base.pt"
    save_model(
        model,
        model.cfg,
        str(checkpoint_path),
        extra_meta={"model_key": "sellm_base", "family": "sellm"},
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["config"].pop("architecture_variant", None)
    checkpoint["cfg_state"].pop("architecture_variant", None)
    torch.save(checkpoint, checkpoint_path)

    predictor = load_predictor(str(checkpoint_path), device="cpu", strict=True)
    with torch.no_grad():
        expected = model(_legacy_input())
        restored = predictor.model(_legacy_input())

    assert predictor.model.architecture_variant == "legacy_v1"
    assert torch.equal(expected, restored)

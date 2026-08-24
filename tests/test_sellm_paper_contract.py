from __future__ import annotations

from types import MethodType
from types import SimpleNamespace

import torch
import torch.nn as nn

from modeling_module.api import load_predictor
from modeling_module.models.SELLM.SELLM import SELLMModel
from modeling_module.models.SELLM.backbone import (
    PaperTSCC,
    PaperTimeProjectionAdapter,
    VocabularySemanticProjection,
)
from modeling_module.models.SELLM.configs import SELLMConfig
from modeling_module.models.SELLM.provenance import (
    SELLM_PAPER_URL,
    SELLM_UPSTREAM_COMMIT,
    SELLM_UPSTREAM_LICENSE,
    SELLM_UPSTREAM_REPOSITORY,
    SELLM_UPSTREAM_REVIEW_FILES,
)
from modeling_module.utils.checkpoint import save_model


def _paper_config(*, horizon: int = 5) -> SELLMConfig:
    return SELLMConfig(
        lookback=8,
        horizon=horizon,
        y_dim=1,
        future_exo_dim=0,
        architecture_variant="paper_v1",
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
        use_norm=False,
        final_nonneg=False,
    )


def test_paper_time_adapter_has_two_temporal_stages_and_zero_init_parity():
    torch.manual_seed(7)
    original = nn.Linear(6, 4, bias=False)
    adapter = PaperTimeProjectionAdapter(original, rank=3)
    value = torch.randn(2, 5, 6)

    assert adapter.long_term.input_size == 3
    assert adapter.long_term.hidden_size == 6
    assert adapter.short_term.input_size == 6
    assert adapter.short_term.hidden_size == 3
    assert torch.equal(adapter(value), original(value))

    nn.init.constant_(adapter.up.weight, 0.1)
    output = adapter(value)
    assert not torch.equal(output, original(value))
    output.square().mean().backward()
    for parameter in (
        adapter.down.weight,
        adapter.long_term.weight_ih_l0,
        adapter.short_term.weight_ih_l0,
        adapter.up.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_paper_time_adapter_accepts_bfloat16_llm_projection():
    torch.manual_seed(7)
    original = nn.Linear(6, 4, bias=False).to(dtype=torch.bfloat16)
    adapter = PaperTimeProjectionAdapter(original, rank=3)
    value = torch.randn(2, 5, 6, dtype=torch.bfloat16)

    output = adapter(value)

    assert output.dtype == torch.bfloat16
    assert torch.equal(output, original(value))
    nn.init.constant_(adapter.up.weight, 0.1)
    output = adapter(value)
    output.float().square().mean().backward()
    assert adapter.down.weight.grad is not None
    assert adapter.up.weight.grad is not None
    assert adapter.down.weight.grad.abs().sum() > 0
    assert adapter.up.weight.grad.abs().sum() > 0


def test_vocabulary_projection_matches_paper_matrix_orientation():
    projection = VocabularySemanticProjection(vocabulary_size=3, prototype_count=2)
    with torch.no_grad():
        projection.projection.weight.copy_(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.25, 0.75]])
        )
    word_embeddings = torch.tensor(
        [[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]]
    )

    prototypes = projection(word_embeddings)

    assert torch.equal(
        prototypes,
        torch.tensor([[1.0, 10.0], [3.5, 35.0]]),
    )


class _FixedCrossAttention(nn.Module):
    def forward(self, time_tokens, prototypes):
        del prototypes
        return torch.zeros_like(time_tokens)


class _FixedAMVAE(nn.Module):
    last_kl_loss = torch.tensor(0.0)

    def forward(self, joint_space):
        return torch.full_like(joint_space, 2.0), torch.full_like(joint_space, 3.0)


class _IdentityFusion(nn.Module):
    def forward(self, time_tokens, semantic_component, prototypes):
        del time_tokens, prototypes
        return semantic_component


def test_paper_tscc_fuses_two_semantic_branches_without_legacy_residual():
    tscc = PaperTSCC(d_model=4, hidden_dim=3, latent_dim=2, top_k=1)
    tscc.cross_attention = _FixedCrossAttention()
    tscc.am_vae = _FixedAMVAE()
    tscc.anomaly_fusion = _IdentityFusion()
    tscc.deanomaly_fusion = _IdentityFusion()

    output = tscc(torch.full((1, 2, 4), 11.0), torch.ones(3, 4))

    assert torch.equal(output, torch.full((1, 2, 4), 5.0))


def test_paper_model_rolls_out_token_segments_to_exact_horizon():
    model = SELLMModel(_paper_config(horizon=5))
    calls: list[torch.Tensor] = []

    def _fixed_decode(self, context):
        calls.append(context.detach().clone())
        step = float(len(calls))
        decoded = torch.zeros_like(context)
        decoded[:, -self.token_len :, :] = step
        return decoded

    model._encode_paper_context = MethodType(_fixed_decode, model)
    output = model(torch.zeros(1, 8, 1))

    assert len(calls) == 3
    assert torch.equal(
        output,
        torch.tensor([[[1.0], [1.0], [2.0], [2.0], [3.0]]]),
    )
    assert torch.equal(calls[1][:, -2:, :], torch.ones(1, 2, 1))
    assert torch.equal(calls[2][:, -2:, :], torch.full((1, 2, 1), 2.0))


def test_paper_fallback_output_gradient_and_strict_checkpoint_restore(tmp_path):
    torch.manual_seed(31)
    model = SELLMModel(_paper_config()).eval()
    value = torch.linspace(-1.0, 1.0, steps=16).reshape(2, 8, 1)
    with torch.no_grad():
        expected = model(value)

    assert expected.shape == (2, 5, 1)
    assert torch.isfinite(expected).all()

    model.train()
    torch.manual_seed(37)
    model(value).square().mean().backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name

    model.eval()
    checkpoint_path = tmp_path / "sellm_base.pt"
    save_model(
        model,
        model.cfg,
        str(checkpoint_path),
        extra_meta={"model_key": "sellm_base", "family": "sellm"},
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["meta"]["architecture_variant"] == "paper_v1"
    assert checkpoint["meta"]["upstream_repository"] == SELLM_UPSTREAM_REPOSITORY
    assert checkpoint["meta"]["upstream_commit"] == SELLM_UPSTREAM_COMMIT
    predictor = load_predictor(str(checkpoint_path), device="cpu", strict=True)
    with torch.no_grad():
        restored = predictor.model(value)

    assert predictor.config["architecture_variant"] == "paper_v1"
    assert predictor.model.architecture_variant == "paper_v1"
    assert torch.equal(expected, restored)


def test_paper_baseline_rejects_future_exogenous_extension():
    config = _paper_config()
    config.future_exo_dim = 2

    try:
        SELLMModel(config)
    except ValueError as exc:
        assert "endogenous SELLM baseline" in str(exc)
    else:
        raise AssertionError("paper_v1 must reject the unversioned exogenous extension.")


def test_sellm_review_provenance_is_pinned_without_vendored_upstream_code():
    assert SELLM_PAPER_URL == "https://arxiv.org/abs/2508.07697"
    assert SELLM_UPSTREAM_REPOSITORY == "https://github.com/LH325/SE-LLM"
    assert SELLM_UPSTREAM_COMMIT == "9fab871b9c4774cd4b58d025de992d55a24c18e7"
    assert SELLM_UPSTREAM_LICENSE is None
    assert {path for path, _ in SELLM_UPSTREAM_REVIEW_FILES} == {
        "models/SELLM.py",
        "models/TimeAdapter.py",
        "models/TSCC.py",
    }
    assert all(len(digest) == 64 for _, digest in SELLM_UPSTREAM_REVIEW_FILES)


class _FakeQwenAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.k_proj = nn.Linear(8, 4, bias=False)
        self.v_proj = nn.Linear(8, 4, bias=False)


class _FakeQwenLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _FakeQwenAttention()


class _FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(11, 8)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_FakeQwenLayer(), _FakeQwenLayer()])
        self.last_input_dtype = None

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, inputs_embeds):
        self.last_input_dtype = inputs_embeds.dtype
        return SimpleNamespace(last_hidden_state=inputs_embeds)


def test_paper_qwen_contract_freezes_base_and_installs_trainable_adapters(monkeypatch):
    fake_qwen = _FakeQwen()
    monkeypatch.setattr(SELLMModel, "_load_llm", staticmethod(lambda cfg: fake_qwen))
    config = _paper_config(horizon=3)
    config.use_pretrained_llm = True
    config.use_time_adapter = True
    config.time_adapter_rank = 2
    config.time_adapter_layers = 1
    config.semantic_vocab_size = 5

    model = SELLMModel(config)

    first_attention = model.llm.model.layers[0].self_attn
    second_attention = model.llm.model.layers[1].self_attn
    assert isinstance(first_attention.k_proj, PaperTimeProjectionAdapter)
    assert isinstance(first_attention.v_proj, PaperTimeProjectionAdapter)
    assert isinstance(second_attention.k_proj, nn.Linear)
    assert model.semantic_vocabulary_projection.projection.weight.shape == (5, 11)
    assert not first_attention.k_proj.original_layer.weight.requires_grad
    assert first_attention.k_proj.down.weight.requires_grad
    assert first_attention.k_proj.long_term.weight_ih_l0.requires_grad
    assert first_attention.k_proj.short_term.weight_ih_l0.requires_grad
    assert first_attention.k_proj.up.weight.requires_grad
    assert not model.llm.embedding.weight.requires_grad
    assert model.semantic_vocabulary_projection.projection.weight.requires_grad

    output = model(torch.randn(2, 8, 1))
    assert output.shape == (2, 3, 1)
    assert torch.isfinite(output).all()


def test_paper_qwen_contract_bridges_bfloat16_llm_and_float32_decoder(monkeypatch):
    fake_qwen = _FakeQwen().to(dtype=torch.bfloat16)
    monkeypatch.setattr(SELLMModel, "_load_llm", staticmethod(lambda cfg: fake_qwen))
    config = _paper_config(horizon=3)
    config.use_pretrained_llm = True
    config.use_time_adapter = True
    config.time_adapter_rank = 2
    config.time_adapter_layers = 1
    config.semantic_vocab_size = 5

    model = SELLMModel(config)
    output = model(torch.randn(2, 8, 1))

    assert fake_qwen.last_input_dtype == torch.bfloat16
    assert output.dtype == torch.float32
    assert output.shape == (2, 3, 1)
    assert torch.isfinite(output).all()

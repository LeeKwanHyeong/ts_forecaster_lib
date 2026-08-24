from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(name: str) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name!r}")


class PaperSegmentMLP(nn.Module):
    """Two-layer numeric segment projection used by the paper encoder and decoder."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            _activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(out_dim)),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class VocabularySemanticProjection(nn.Module):
    """Project the frozen word-embedding vocabulary V into K semantic prototypes."""

    def __init__(self, vocabulary_size: int, prototype_count: int) -> None:
        super().__init__()
        if int(vocabulary_size) <= 0:
            raise ValueError("vocabulary_size must be positive.")
        if int(prototype_count) <= 0:
            raise ValueError("prototype_count must be positive.")
        self.vocabulary_size = int(vocabulary_size)
        self.prototype_count = int(prototype_count)
        self.projection = nn.Linear(
            self.vocabulary_size,
            self.prototype_count,
            bias=False,
        )

    def forward(self, word_embeddings: torch.Tensor) -> torch.Tensor:
        if word_embeddings.ndim != 2:
            raise ValueError(
                "word_embeddings must be 2D [V, C], got "
                f"{tuple(word_embeddings.shape)}."
            )
        if int(word_embeddings.size(0)) != self.vocabulary_size:
            raise ValueError(
                "word embedding vocabulary mismatch: expected "
                f"{self.vocabulary_size}, got {int(word_embeddings.size(0))}."
            )
        return self.projection(word_embeddings.transpose(0, 1)).transpose(0, 1)


class PaperAMVAE(nn.Module):
    """AM-VAE decomposition from the paper's joint semantic space equations."""

    def __init__(self, d_model: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU())
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.last_kl_loss: Optional[torch.Tensor] = None

    def forward(self, joint_space: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(joint_space)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)
        if self.training:
            latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            latent = mu
        anomaly = self.decoder(latent)
        deanomaly = joint_space - anomaly
        self.last_kl_loss = -0.5 * torch.mean(
            1.0 + logvar - mu.square() - logvar.exp()
        )
        return deanomaly, anomaly


class PaperCrossModalAttention(nn.Module):
    """Scaled dot-product CrossAttn(H, S) from the paper."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.scale = float(d_model) ** -0.5
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(
        self,
        time_tokens: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        semantic = prototypes.unsqueeze(0).expand(time_tokens.size(0), -1, -1)
        query = self.query(time_tokens)
        key = self.key(semantic)
        value = self.value(semantic)
        attention = torch.softmax(query @ key.transpose(1, 2) * self.scale, dim=-1)
        return attention @ value


class PaperSemanticGatedFusion(nn.Module):
    """Equations 4-7: top-k structural prior and channel-wise gated fusion."""

    def __init__(self, d_model: int, top_k: int) -> None:
        super().__init__()
        self.top_k = int(top_k)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.output = nn.Linear(d_model, d_model)

    def forward(
        self,
        time_tokens: torch.Tensor,
        semantic_component: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        time_summary = F.normalize(time_tokens.mean(dim=1), p=2, dim=-1)
        normalized_prototypes = F.normalize(prototypes, p=2, dim=-1)
        count = min(max(self.top_k, 1), int(prototypes.size(0)))
        indices = torch.topk(
            time_summary @ normalized_prototypes.transpose(0, 1),
            k=count,
            dim=-1,
        ).indices
        structural_prior = prototypes[indices].mean(dim=1).unsqueeze(1)
        enhanced_component = semantic_component * structural_prior
        gate = self.gate(torch.cat([time_tokens, enhanced_component], dim=-1))
        return self.output(
            gate * time_tokens + (1.0 - gate) * enhanced_component
        )


class PaperTSCC(nn.Module):
    """Paper-faithful TSCC without the legacy extra temporal residual."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        latent_dim: int,
        top_k: int,
    ) -> None:
        super().__init__()
        self.cross_attention = PaperCrossModalAttention(d_model)
        self.am_vae = PaperAMVAE(d_model, hidden_dim, latent_dim)
        self.anomaly_fusion = PaperSemanticGatedFusion(d_model, top_k)
        self.deanomaly_fusion = PaperSemanticGatedFusion(d_model, top_k)
        self.last_kl_loss: Optional[torch.Tensor] = None

    def forward(
        self,
        time_tokens: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        joint_space = self.cross_attention(time_tokens, prototypes)
        deanomaly, anomaly = self.am_vae(joint_space)
        anomaly_output = self.anomaly_fusion(time_tokens, anomaly, prototypes)
        deanomaly_output = self.deanomaly_fusion(
            time_tokens,
            deanomaly,
            prototypes,
        )
        self.last_kl_loss = self.am_vae.last_kl_loss
        return anomaly_output + deanomaly_output


class PaperTimeProjectionAdapter(nn.Module):
    """Two-LSTM temporal residual for an attention key or value projection."""

    def __init__(self, original_layer: nn.Module, rank: int) -> None:
        super().__init__()
        if not hasattr(original_layer, "in_features") or not hasattr(
            original_layer, "out_features"
        ):
            raise TypeError("PaperTimeProjectionAdapter expects an nn.Linear-like layer.")
        input_dim = int(original_layer.in_features)
        output_dim = int(original_layer.out_features)
        rank = max(int(rank), 1)
        self.original_layer = original_layer
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.long_term = nn.LSTM(rank, input_dim, batch_first=True)
        self.short_term = nn.LSTM(input_dim, rank, batch_first=True)
        self.up = nn.Linear(rank, output_dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.original_layer(value)
        temporal = self.down(value)
        temporal, _ = self.long_term(temporal)
        temporal, _ = self.short_term(temporal)
        return residual + self.up(temporal)


def _iter_llm_layers(model: nn.Module) -> list[nn.Module]:
    if hasattr(model, "layers"):
        return list(model.layers)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return list(inner.layers)
    return []


def add_paper_time_adapter(
    model: nn.Module,
    *,
    rank: int,
    num_layers: int,
) -> nn.Module:
    applied = 0
    for layer in _iter_llm_layers(model)[: max(int(num_layers), 0)]:
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            continue
        for name in ("k_proj", "v_proj"):
            projection = getattr(attention, name, None)
            if projection is None:
                continue
            setattr(
                attention,
                name,
                PaperTimeProjectionAdapter(projection, rank=rank),
            )
            applied += 1
    if applied == 0:
        raise RuntimeError(
            "Unable to install paper Time-Adapter: no k_proj/v_proj layers were found."
        )
    return model

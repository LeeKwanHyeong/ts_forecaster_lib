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


class ICLSemanticPromptEncoder(nn.Module):
    """Convert labeled numeric demonstrations into masked semantic prompt tokens."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(int(d_model) * 2, int(d_model)),
            nn.GELU(),
            nn.Linear(int(d_model), int(d_model)),
        )
        self.norm = nn.LayerNorm(int(d_model))

    def forward(
        self,
        context_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return one semantic token per demonstration and target channel."""

        if context_tokens.ndim != 5 or target_tokens.ndim != 5:
            raise ValueError("SELLM ICL token inputs must be [B,C,K,N,D].")
        if context_tokens.shape[:3] != target_tokens.shape[:3]:
            raise ValueError("SELLM ICL context and target prompt dimensions must match.")
        if prompt_mask.shape != (
            int(context_tokens.shape[0]),
            int(context_tokens.shape[2]),
        ):
            raise ValueError("SELLM ICL prompt mask must be [B,K].")
        context_summary = context_tokens.mean(dim=3)
        target_summary = target_tokens.mean(dim=3)
        prompt = self.norm(
            self.projection(torch.cat([context_summary, target_summary], dim=-1))
        )
        mask = prompt_mask[:, None, :, None].to(
            device=prompt.device,
            dtype=prompt.dtype,
        )
        return prompt * mask


class ICLExogenousPromptEncoder(nn.Module):
    """Encode observed-past and known-future features as semantic prompt tokens."""

    def __init__(self, past_dim: int, future_dim: int, d_model: int) -> None:
        super().__init__()
        self.past_dim = int(past_dim)
        self.future_dim = int(future_dim)
        self.past_projection = nn.Linear(self.past_dim, int(d_model))
        self.future_projection = nn.Linear(self.future_dim, int(d_model))
        self.demo_norm = nn.LayerNorm(int(d_model))
        self.query_norm = nn.LayerNorm(int(d_model))

    def demonstrations(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        if past.ndim != 4 or future.ndim != 4:
            raise ValueError("SELLM demonstration exogenous tensors must be [B,K,T,E].")
        if past.shape[:2] != future.shape[:2] or past.shape[:2] != prompt_mask.shape:
            raise ValueError("SELLM demonstration exogenous prompt dimensions differ.")
        if int(past.shape[-1]) != self.past_dim or int(future.shape[-1]) != self.future_dim:
            raise ValueError("SELLM demonstration exogenous width does not match config.")
        token = self.demo_norm(
            self.past_projection(past).mean(dim=2)
            + self.future_projection(future).mean(dim=2)
        )
        return token * prompt_mask[..., None].to(dtype=token.dtype)

    def query(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        if past.ndim != 3 or future.ndim != 3:
            raise ValueError("SELLM query exogenous tensors must be [B,T,E].")
        if int(past.shape[0]) != int(future.shape[0]):
            raise ValueError("SELLM query exogenous batch dimensions differ.")
        if int(past.shape[-1]) != self.past_dim or int(future.shape[-1]) != self.future_dim:
            raise ValueError("SELLM query exogenous width does not match config.")
        tokens = torch.cat(
            [self.past_projection(past), self.future_projection(future)],
            dim=1,
        )
        return self.query_norm(tokens)


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
        adapter_dtype = self.down.weight.dtype
        temporal = self.down(value.to(dtype=adapter_dtype))
        temporal, _ = self.long_term(temporal)
        temporal, _ = self.short_term(temporal)
        return residual + self.up(temporal).to(dtype=residual.dtype)


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

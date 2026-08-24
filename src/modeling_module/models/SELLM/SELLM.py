from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling_module.models.SELLM.configs import SELLMConfig
from modeling_module.models.SELLM.backbone import (
    PaperSegmentMLP,
    PaperTSCC,
    VocabularySemanticProjection,
    add_paper_time_adapter,
)
from modeling_module.models.SELLM.provenance import (
    SELLM_UPSTREAM_COMMIT,
    SELLM_UPSTREAM_REPOSITORY,
)


def _activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name!r}")


class SegmentMLP(nn.Module):
    """Encode or decode one numeric segment."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float, activation: str):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            _activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AMVAE(nn.Module):
    """Lightweight anomaly/de-anomaly semantic decomposition block."""

    def __init__(self, d_model: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.last_kl_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        anomaly = self.decoder(z)
        deanomaly = x - anomaly
        self.last_kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        return deanomaly, anomaly


class SemanticGatedFusion(nn.Module):
    """Fuse temporal embeddings with one semantic component conditioned by top-k prototypes."""

    def __init__(self, d_model: int, top_k: int):
        super().__init__()
        self.top_k = int(top_k)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.out_proj = nn.Linear(d_model, d_model)

    @staticmethod
    def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)

    def forward(
        self,
        time_feat: torch.Tensor,
        semantic_component: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        time_mean = self._l2_normalize(time_feat.mean(dim=1))
        proto_norm = self._l2_normalize(prototypes)
        k = min(max(self.top_k, 1), int(prototypes.size(0)))
        topk_idx = torch.topk(time_mean @ proto_norm.transpose(0, 1), k=k, dim=1).indices
        prior = prototypes[topk_idx].mean(dim=1).unsqueeze(1)
        enhanced = semantic_component * torch.sigmoid(prior)
        gate = self.gate(torch.cat([time_feat, enhanced], dim=-1))
        fused = gate * time_feat + (1.0 - gate) * enhanced
        return self.out_proj(fused)


class TemporalSemanticCrossCorrelation(nn.Module):
    """
    TSCC-style block:
    1. Cross-attend time tokens to semantic prototypes.
    2. Decompose joint semantic space into anomaly and de-anomaly components.
    3. Fuse both branches back into temporal tokens.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        latent_dim: int,
        hidden_dim: int,
        top_k: int,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=int(d_model),
            num_heads=int(n_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.vae = AMVAE(int(d_model), int(hidden_dim), int(latent_dim))
        self.fusion = SemanticGatedFusion(int(d_model), int(top_k))
        self.norm = nn.LayerNorm(int(d_model))
        self.last_kl_loss: Optional[torch.Tensor] = None

    def forward(self, time_tokens: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        semantic = prototypes.unsqueeze(0).expand(time_tokens.size(0), -1, -1)
        joint, _ = self.cross_attn(time_tokens, semantic, semantic, need_weights=False)
        deanomaly, anomaly = self.vae(joint)
        anomaly_branch = self.fusion(time_tokens, anomaly, prototypes)
        deanomaly_branch = self.fusion(time_tokens, deanomaly, prototypes)
        self.last_kl_loss = self.vae.last_kl_loss
        return self.norm(time_tokens + anomaly_branch + deanomaly_branch)


class TimeProjectionAdapter(nn.Module):
    """LoRA-style temporal branch for attention projections."""

    def __init__(self, original_layer: nn.Module, rank: int):
        super().__init__()
        if not hasattr(original_layer, "in_features") or not hasattr(original_layer, "out_features"):
            raise TypeError("TimeProjectionAdapter expects an nn.Linear-like layer.")
        self.original_layer = original_layer
        self.down = nn.Linear(int(original_layer.in_features), int(rank), bias=False)
        self.temporal = nn.LSTM(int(rank), int(rank), batch_first=True)
        self.up = nn.Linear(int(rank), int(original_layer.out_features), bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.original_layer(x)
        z = self.down(x)
        z, _ = self.temporal(z)
        return residual + self.up(z)


def _iter_llm_layers(model: nn.Module):
    if hasattr(model, "layers"):
        return list(model.layers)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return list(inner.layers)
    transformer = getattr(model, "transformer", None)
    if transformer is not None and hasattr(transformer, "h"):
        return list(transformer.h)
    return []


def add_time_adapter(model: nn.Module, *, rank: int, num_layers: int) -> nn.Module:
    applied = 0
    for layer in _iter_llm_layers(model)[: max(int(num_layers), 0)]:
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn is None:
            continue
        for name in ("k_proj", "v_proj"):
            proj = getattr(attn, name, None)
            if proj is not None and hasattr(proj, "in_features") and hasattr(proj, "out_features"):
                setattr(attn, name, TimeProjectionAdapter(proj, rank=max(int(rank), 1)))
                applied += 1
    if applied == 0:
        raise RuntimeError("Unable to install TimeAdapter: no k_proj/v_proj attention layers were found.")
    return model


class SELLMModel(nn.Module):
    """Semantic-enhanced LLM forecaster aligned with the library's [B, L, C] contract."""

    def __init__(self, cfg: SELLMConfig):
        super().__init__()
        self.cfg = cfg
        self.lookback = int(cfg.lookback)
        self.horizon = int(cfg.horizon)
        self.y_dim = int(cfg.y_dim)
        self.future_exo_dim = int(cfg.future_exo_dim)
        self.architecture_variant = str(cfg.architecture_variant).strip().lower()
        if self.architecture_variant == "paper_v1":
            self.upstream_repository = SELLM_UPSTREAM_REPOSITORY
            self.upstream_commit = SELLM_UPSTREAM_COMMIT
        self.token_len = int(cfg.token_len)
        self.use_norm = bool(cfg.use_norm)
        self.final_nonneg = bool(cfg.final_nonneg)
        self.use_pretrained_llm = bool(cfg.use_pretrained_llm)

        if self.lookback <= 0:
            raise ValueError(f"lookback must be positive, got {self.lookback}")
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if self.y_dim <= 0:
            raise ValueError(f"y_dim must be positive, got {self.y_dim}")
        if self.token_len <= 0:
            raise ValueError(f"token_len must be positive, got {self.token_len}")
        if self.token_len > self.lookback:
            raise ValueError(
                f"token_len={self.token_len} cannot exceed lookback={self.lookback}."
            )
        if self.architecture_variant not in {"legacy_v1", "paper_v1"}:
            raise ValueError(
                "architecture_variant must be 'legacy_v1' or 'paper_v1', got "
                f"{cfg.architecture_variant!r}."
            )
        if self.architecture_variant == "paper_v1" and self.future_exo_dim > 0:
            raise ValueError(
                "paper_v1 is the endogenous SELLM baseline and does not accept "
                "future exogenous inputs. Use legacy_v1 until a separate sellm_exo "
                "contract is introduced."
            )

        self.llm: Optional[nn.Module] = None
        self.fallback_encoder: Optional[nn.Module] = None
        self.semantic_bank: Optional[nn.Parameter] = None
        self.semantic_source_bank: Optional[nn.Parameter] = None
        self.semantic_proj: Optional[nn.Module] = None
        self.semantic_vocabulary_projection: Optional[nn.Module] = None

        d_model = int(cfg.d_model)
        if self.use_pretrained_llm:
            self.llm = self._load_llm(cfg)
            d_model = self._infer_hidden_size(self.llm)
            if bool(cfg.freeze_llm):
                for parameter in self.llm.parameters():
                    parameter.requires_grad = False
            if bool(cfg.use_time_adapter):
                if self.architecture_variant == "paper_v1":
                    self.llm = add_paper_time_adapter(
                        self.llm,
                        rank=int(cfg.time_adapter_rank),
                        num_layers=int(cfg.time_adapter_layers),
                    )
                else:
                    self.llm = add_time_adapter(
                        self.llm,
                        rank=int(cfg.time_adapter_rank),
                        num_layers=int(cfg.time_adapter_layers),
                    )
            embedding = self.llm.get_input_embeddings().weight
            if self.architecture_variant == "paper_v1":
                self.semantic_vocabulary_projection = VocabularySemanticProjection(
                    vocabulary_size=int(embedding.size(0)),
                    prototype_count=int(cfg.semantic_vocab_size),
                )
            else:
                self.semantic_proj = nn.Linear(d_model, d_model)
        else:
            if d_model % int(cfg.n_heads) != 0:
                raise ValueError(f"d_model={d_model} must be divisible by n_heads={cfg.n_heads}.")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=int(cfg.n_heads),
                dim_feedforward=int(cfg.d_ff),
                dropout=float(cfg.dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.fallback_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=int(cfg.fallback_layers),
                norm=nn.LayerNorm(d_model),
            )
            if self.architecture_variant == "paper_v1":
                self.semantic_source_bank = nn.Parameter(
                    torch.randn(int(cfg.semantic_vocab_size), d_model) * 0.02
                )
                self.semantic_vocabulary_projection = VocabularySemanticProjection(
                    vocabulary_size=int(cfg.semantic_vocab_size),
                    prototype_count=int(cfg.semantic_vocab_size),
                )
            else:
                self.semantic_bank = nn.Parameter(
                    torch.randn(int(cfg.semantic_vocab_size), d_model) * 0.02
                )

        self.d_model = int(d_model)
        self.cfg = replace(cfg, d_model=self.d_model)

        n_heads = int(cfg.n_heads)
        if self.d_model % n_heads != 0:
            fallback_head = self._largest_divisor_at_most(self.d_model, n_heads)
            n_heads = fallback_head

        segment_mlp = (
            PaperSegmentMLP
            if self.architecture_variant == "paper_v1"
            else SegmentMLP
        )
        self.ts_encoder = segment_mlp(
            self.token_len,
            self.d_model,
            int(cfg.mlp_hidden_dim),
            float(cfg.dropout),
            str(cfg.mlp_activation),
        )
        if self.architecture_variant == "paper_v1":
            self.tscc = PaperTSCC(
                d_model=self.d_model,
                latent_dim=int(cfg.tscc_latent_dim),
                hidden_dim=int(cfg.tscc_hidden_dim),
                top_k=int(cfg.semantic_top_k),
            )
            self.ts_decoder: Optional[nn.Module] = PaperSegmentMLP(
                self.d_model,
                self.token_len,
                int(cfg.mlp_hidden_dim),
                float(cfg.dropout),
                str(cfg.mlp_activation),
            )
            self.pool_head: Optional[nn.Module] = None
        else:
            self.tscc = TemporalSemanticCrossCorrelation(
                d_model=self.d_model,
                n_heads=n_heads,
                dropout=float(cfg.dropout),
                latent_dim=int(cfg.tscc_latent_dim),
                hidden_dim=int(cfg.tscc_hidden_dim),
                top_k=int(cfg.semantic_top_k),
            )
            self.ts_decoder = None
            self.pool_head = nn.Sequential(
                nn.Linear(self.d_model * 2, int(cfg.head_hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(cfg.dropout)),
                nn.Linear(int(cfg.head_hidden_dim), self.horizon),
            )
        if self.future_exo_dim > 0:
            self.future_exo_head = nn.Sequential(
                nn.Linear(self.future_exo_dim, int(cfg.head_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(cfg.head_hidden_dim), self.y_dim),
            )
        else:
            self.future_exo_head = None

    @classmethod
    def from_config(cls, config: SELLMConfig) -> "SELLMModel":
        return cls(cfg=config)

    @staticmethod
    def _largest_divisor_at_most(value: int, limit: int) -> int:
        for candidate in range(max(int(limit), 1), 0, -1):
            if int(value) % candidate == 0:
                return candidate
        return 1

    @staticmethod
    def _load_llm(cfg: SELLMConfig) -> nn.Module:
        source = str(cfg.llm_source).strip().lower()
        load_kwargs: dict[str, object] = {}

        if source == "huggingface":
            model_name = str(cfg.llm_model_name).strip()
            if not model_name:
                raise ValueError("llm_model_name is required when llm_source='huggingface'.")
            load_target = model_name
            revision = str(cfg.llm_revision).strip() if cfg.llm_revision is not None else ""
            if revision:
                load_kwargs["revision"] = revision
        elif source == "local":
            local_path_value = str(cfg.llm_local_path).strip() if cfg.llm_local_path is not None else ""
            if not local_path_value:
                raise ValueError("llm_local_path is required when llm_source='local'.")
            local_path = Path(local_path_value).expanduser()
            if not local_path.is_dir():
                raise FileNotFoundError(f"Local LLM directory does not exist: {local_path}")
            if not (local_path / "config.json").is_file():
                raise FileNotFoundError(f"Local LLM config.json does not exist: {local_path}")
            load_target = str(local_path)
            load_kwargs["local_files_only"] = True
        else:
            raise ValueError(
                f"Unsupported llm_source={cfg.llm_source!r}; expected 'huggingface' or 'local'."
            )

        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "SELLM with use_pretrained_llm=True requires the optional LLM dependencies. "
                "Install with `pip install modeling-module[sellm]` or use the 5090 conda env."
            ) from exc

        return AutoModel.from_pretrained(load_target, **load_kwargs)

    @staticmethod
    def _infer_hidden_size(model: nn.Module) -> int:
        config = getattr(model, "config", None)
        for attr in ("hidden_size", "n_embd", "d_model"):
            value = getattr(config, attr, None)
            if value is not None:
                return int(value)
        embedding = model.get_input_embeddings()
        return int(embedding.weight.shape[-1])

    def _make_semantic_prototypes(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.architecture_variant == "paper_v1":
            if self.semantic_vocabulary_projection is None:
                raise RuntimeError("paper_v1 semantic vocabulary projection is missing.")
            if self.llm is not None:
                source = self.llm.get_input_embeddings().weight
            elif self.semantic_source_bank is not None:
                source = self.semantic_source_bank
            else:
                raise RuntimeError("paper_v1 semantic source embeddings are missing.")
            source = source.to(device=device, dtype=dtype)
            return self.semantic_vocabulary_projection(source)

        if self.llm is not None:
            weight = self.llm.get_input_embeddings().weight
            vocab_size = int(weight.size(0))
            k = min(max(int(self.cfg.semantic_vocab_size), 1), vocab_size)
            idx = torch.linspace(0, vocab_size - 1, steps=k, device=weight.device).long()
            prototypes = weight.index_select(0, idx).to(device=device, dtype=dtype)
            if self.semantic_proj is not None:
                prototypes = self.semantic_proj(prototypes)
            return prototypes

        if self.semantic_bank is None:
            raise RuntimeError("semantic_bank is missing for fallback SELLM mode.")
        return self.semantic_bank.to(device=device, dtype=dtype)

    def _segment(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        remainder = int(x.size(-1)) % self.token_len
        if remainder:
            pad = self.token_len - remainder
            x = F.pad(x, (0, pad), mode="replicate")
        tokens = x.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        return tokens.contiguous(), int(tokens.size(1))

    def reg_loss(self) -> Optional[torch.Tensor]:
        kl = self.tscc.last_kl_loss
        if kl is None:
            return None
        weight = float(getattr(self.cfg, "tscc_kl_weight", 0.0) or 0.0)
        if weight <= 0.0:
            return None
        return kl * weight

    def _encode_paper_context(self, context: torch.Tensor) -> torch.Tensor:
        batch_size, _, n_vars = context.shape
        flat = context.permute(0, 2, 1).reshape(batch_size * n_vars, -1)
        segments, _ = self._segment(flat)
        time_tokens = self.ts_encoder(segments)
        prototypes = self._make_semantic_prototypes(
            time_tokens.device,
            time_tokens.dtype,
        )
        fused = self.tscc(time_tokens, prototypes)
        if self.llm is not None:
            encoded = self.llm(inputs_embeds=fused).last_hidden_state
        elif self.fallback_encoder is not None:
            encoded = self.fallback_encoder(fused)
        else:
            raise RuntimeError("paper_v1 encoder backend is missing.")
        if self.ts_decoder is None:
            raise RuntimeError("paper_v1 numeric token decoder is missing.")
        decoded = self.ts_decoder(encoded)
        return decoded.reshape(batch_size, n_vars, -1).permute(0, 2, 1).contiguous()

    def _paper_forecast(self, context: torch.Tensor) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        remaining = self.horizon
        rolling_context = context
        while remaining > 0:
            reconstructed = self._encode_paper_context(rolling_context)
            next_segment = reconstructed[:, -self.token_len :, :]
            take = min(remaining, self.token_len)
            chunks.append(next_segment[:, :take, :])
            remaining -= take
            if remaining > 0:
                rolling_context = torch.cat(
                    [rolling_context[:, self.token_len :, :], next_segment],
                    dim=1,
                )
        return torch.cat(chunks, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        future_exo: Optional[torch.Tensor] = None,
        past_exo_cont: Optional[torch.Tensor] = None,
        past_exo_cat: Optional[torch.Tensor] = None,
        part_ids: Optional[torch.Tensor] = None,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        del past_exo_cont, past_exo_cat, part_ids, mode

        if x.dim() != 3:
            raise ValueError(f"x must be 3D [B, L, C], got {tuple(x.shape)}")
        if int(x.size(1)) != self.lookback:
            raise ValueError(f"x lookback mismatch: expected {self.lookback}, got {tuple(x.shape)}")
        if int(x.size(2)) != self.y_dim:
            raise ValueError(f"x channel mismatch: expected {self.y_dim}, got {tuple(x.shape)}")

        if self.future_exo_dim > 0:
            if future_exo is None:
                raise ValueError(f"SELLM expects future_exo with last dim={self.future_exo_dim}.")
            if future_exo.dim() == 2:
                future_exo = future_exo.unsqueeze(0).expand(x.size(0), -1, -1)
            if future_exo.dim() != 3:
                raise ValueError(f"future_exo must be 3D [B, H, E], got {tuple(future_exo.shape)}")
            if int(future_exo.size(0)) != int(x.size(0)):
                raise ValueError("future_exo batch mismatch.")
            if int(future_exo.size(1)) != self.horizon:
                raise ValueError(f"future_exo horizon mismatch: expected {self.horizon}, got {tuple(future_exo.shape)}")
            if int(future_exo.size(2)) != self.future_exo_dim:
                raise ValueError(
                    f"future_exo feature mismatch: expected {self.future_exo_dim}, got {tuple(future_exo.shape)}"
                )
        elif future_exo is not None and int(future_exo.size(-1)) > 0:
            raise ValueError("SELLM was configured without future exogenous inputs.")

        if self.use_norm:
            means = x.mean(dim=1, keepdim=True).detach()
            centered = x - means
            stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_norm = centered / stdev
        else:
            x_norm = x
            means = None
            stdev = None

        if self.architecture_variant == "paper_v1":
            forecast = self._paper_forecast(x_norm)
        else:
            batch_size, _, n_vars = x_norm.shape
            flat = x_norm.permute(0, 2, 1).reshape(batch_size * n_vars, -1)
            segments, _ = self._segment(flat)
            time_tokens = self.ts_encoder(segments)
            prototypes = self._make_semantic_prototypes(time_tokens.device, time_tokens.dtype)
            fused = self.tscc(time_tokens, prototypes)

            if self.llm is not None:
                encoded = self.llm(inputs_embeds=fused).last_hidden_state
            elif self.fallback_encoder is not None:
                encoded = self.fallback_encoder(fused)
            else:
                encoded = fused

            pooled = torch.cat([encoded.mean(dim=1), encoded[:, -1, :]], dim=-1)
            if self.pool_head is None:
                raise RuntimeError("legacy_v1 pooling head is missing.")
            forecast = self.pool_head(pooled).reshape(
                batch_size,
                n_vars,
                self.horizon,
            )
            forecast = forecast.permute(0, 2, 1).contiguous()

        if self.future_exo_head is not None and future_exo is not None:
            forecast = forecast + self.future_exo_head(future_exo.to(dtype=forecast.dtype))

        if self.use_norm and means is not None and stdev is not None:
            forecast = forecast * stdev[:, 0, :].unsqueeze(1).repeat(1, self.horizon, 1)
            forecast = forecast + means[:, 0, :].unsqueeze(1).repeat(1, self.horizon, 1)

        if self.final_nonneg:
            forecast = F.softplus(forecast)

        return forecast

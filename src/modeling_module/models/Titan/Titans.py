from __future__ import annotations

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling_module.models.Titan import TitanConfig, MemoryEncoder, LMM, TitanDecoder
from modeling_module.models.Titan.common.decoder import TitanCrossAttnDecoder
from modeling_module.models.common_layers.RevIN import RevIN


def _validate_future_exo_contract(
    future_exo: Optional[torch.Tensor],
    *,
    batch_size: int,
    horizon: int,
    expected_dim: int,
) -> None:
    expected_dim = int(expected_dim)
    if future_exo is None:
        if expected_dim > 0:
            raise RuntimeError(
                f"[Titan] future_exo is required when configured future width={expected_dim}; "
                f"expected shape ({batch_size}, {horizon}, {expected_dim})."
            )
        return

    if not torch.is_tensor(future_exo):
        raise RuntimeError(
            f"[Titan] future_exo must be a tensor with rank-3 [B,H,E], got {type(future_exo).__name__}."
        )
    if expected_dim <= 0:
        if future_exo.numel() > 0:
            raise RuntimeError(
                "[Titan] future_exo is not accepted when configured future width=0; "
                f"got non-empty shape {tuple(future_exo.shape)}."
            )
        return
    if future_exo.dim() != 3:
        raise RuntimeError(
            f"[Titan] future_exo must be rank-3 [B,H,E], got shape {tuple(future_exo.shape)}."
        )

    actual_batch, actual_horizon, actual_dim = future_exo.shape
    if actual_batch != batch_size:
        raise RuntimeError(
            f"[Titan] future_exo batch mismatch: got {actual_batch}, expected {batch_size}."
        )
    if actual_horizon != horizon:
        raise RuntimeError(
            f"[Titan] future_exo horizon mismatch: got {actual_horizon}, expected {horizon}."
        )
    if actual_dim != expected_dim:
        raise RuntimeError(
            f"[Titan] future_exo last dimension mismatch: got {actual_dim}, expected {expected_dim}."
        )


class _PastExoEmbed(nn.Module):
    """
    Past exogenous embedding:
      - continuous: identity (concatenate)
      - categorical: (단일) embedding then concatenate

    규약 (단일 categorical):
      - cfg.past_exo_cat_dim: int (vocab size / cardinality). 0이면 categorical 미사용.
      - cfg.past_exo_cat_embed_dim: Optional[int] (embedding dim). None이면 자동 결정.
      - past_exo_cat 입력 텐서: (B, L) 또는 (B, L, 1), dtype=torch.long
    """

    def __init__(self, cont_dim: int, cat_dim: int, cat_embed_dim: Optional[int]):
        super().__init__()
        self.cont_dim = int(cont_dim)

        self.cat_dim = int(cat_dim)
        self.use_cat = self.cat_dim > 0

        if self.use_cat:
            if cat_embed_dim is None:
                # 간단 휴리스틱 (필요 시 정책화 가능)
                cat_embed_dim = min(16, max(2, self.cat_dim // 4))
            self.cat_embed_dim = int(cat_embed_dim)
            self.cat_emb = nn.Embedding(num_embeddings=self.cat_dim, embedding_dim=self.cat_embed_dim)
        else:
            self.cat_embed_dim = 0
            self.cat_emb = None

        self.out_dim = self.cont_dim + self.cat_embed_dim

    def forward(
        self,
        past_exo_cont: Optional[torch.Tensor],
        past_exo_cat: Optional[torch.Tensor],
        B: int,
        L: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        feats: List[torch.Tensor] = []

        # continuous
        if self.cont_dim > 0:
            if past_exo_cont is None:
                feats.append(torch.zeros(B, L, self.cont_dim, device=device, dtype=dtype))
            else:
                feats.append(past_exo_cont.to(device=device, dtype=dtype))

        # categorical (single)
        if self.use_cat:
            if past_exo_cat is None:
                idx = torch.zeros(B, L, device=device, dtype=torch.long)
            else:
                idx = past_exo_cat.to(device=device, dtype=torch.long)
                # (B, L, 1) -> (B, L)
                if idx.ndim == 3:
                    if idx.size(-1) != 1:
                        raise ValueError(
                            f"past_exo_cat last dim must be 1 in single-cat mode. got {tuple(idx.shape)}"
                        )
                    idx = idx.squeeze(-1)
                elif idx.ndim != 2:
                    raise ValueError(f"past_exo_cat must be (B,L) or (B,L,1). got {tuple(idx.shape)}")

            emb = self.cat_emb(idx)  # [B, L, cat_embed_dim]
            feats.append(emb.to(dtype=dtype))

        if len(feats) == 0:
            return torch.zeros(B, L, 0, device=device, dtype=dtype)

        return torch.cat(feats, dim=-1)


class _TitanBase(nn.Module):
    def __init__(
        self,
        cfg: TitanConfig,
        *,
        has_memory: bool,
        has_decoder: bool,
        out_mult: Optional[int] = None,
        param_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.lookback = int(cfg.lookback)
        self.horizon = int(cfg.horizon)
        self.future_exo_dim = int(getattr(cfg, "future_exo_dim", getattr(cfg, "exo_dim", 0)))

        self.d_model = int(cfg.d_model)
        if out_mult is None:
            out_mult = int(getattr(cfg, "out_mul", 1))
        if param_names is None:
            param_names = getattr(cfg, "param_names", None)
        self.out_mult = int(out_mult)
        self.param_names = list(param_names) if param_names is not None else None

        # RevIN (x: [B, L, 1] 이므로 num_features=1로 고정해도 무방)
        self.use_revin = bool(getattr(cfg, "use_revin", True))
        self.revin = RevIN(int(getattr(cfg, "enc_in", 1)), affine=False, subtract_last=True) if self.use_revin else None

        # Past exo embed (단일 cat 규약)
        self.past_exo_embed = _PastExoEmbed(
            cont_dim=int(getattr(cfg, "past_exo_cont_dim", 0)),
            cat_dim=int(getattr(cfg, "past_exo_cat_dim", 0)),
            cat_embed_dim=getattr(cfg, "past_exo_cat_embed_dim", None),
        )

        # Encoder input dim = y(1) + past_exo_embed
        encoder_input_dim = 1 + self.past_exo_embed.out_dim

        self.encoder = MemoryEncoder(
            input_dim=encoder_input_dim,
            d_model=self.d_model,
            n_layers=int(cfg.n_layers),
            n_heads=int(cfg.n_heads),
            d_ff=int(cfg.d_ff),
            contextual_mem_size=int(cfg.contextual_mem_size),
            persistent_mem_size=int(cfg.persistent_mem_size),
            dropout=float(cfg.dropout),
            use_context_update=bool(getattr(cfg, "use_context_update", False)),
            use_pos_emb=bool(getattr(cfg, "use_pos_emb", True)),
            max_len=int(getattr(cfg, "max_len", 512)),
        )

        # Optional LMM
        self.has_memory = bool(has_memory)
        if self.has_memory:
            self.lmm = LMM(
                d_model=self.d_model,
                mem_size=int(getattr(cfg, "mem_size", 128)),
                topk=int(getattr(cfg, "mem_topk", 8)),
            )
        else:
            self.lmm = None

        # Optional decoder
        self.has_decoder = bool(has_decoder)
        if self.has_decoder:
            self.decoder = TitanCrossAttnDecoder(
                d_model=self.d_model,
                horizon=self.horizon,
                exo_dim=self.future_exo_dim,
                n_heads=int(getattr(cfg, "dec_n_heads", getattr(cfg, "n_heads", 8))),
                dropout=float(cfg.dropout),
                use_step_emb=True,
                pre_norm=True,
            )
        else:
            self.decoder = None

        # Output head
        self.head = nn.Linear(self.d_model, self.out_mult)
        self.proj = nn.Linear(self.d_model, self.out_mult)

        # Clamp
        self.clamp_min = getattr(cfg, "clamp_min", 0.0)
        self.clamp_max = getattr(cfg, "clamp_max", None)

    def _inv_softplus(self, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Stable inverse of softplus.
        For large y, expm1(-y) -> -1 (stable), so no overflow.
        """
        y = torch.clamp(y, min=eps)
        return y + torch.log(-torch.expm1(-y))

    def _maybe_revin_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin is None:
            return x
        return self.revin(x, mode="norm")

    def _maybe_revin_denorm(self, y: torch.Tensor) -> torch.Tensor:
        if self.revin is None:
            return y
        return self.revin(y, mode="denorm")

    def _make_encoder_input(
        self,
        x: torch.Tensor,
        past_exo_cont: Optional[torch.Tensor],
        past_exo_cat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # x: [B, L, 1]
        B, L, _ = x.shape
        exo = self.past_exo_embed(
            past_exo_cont=past_exo_cont,
            past_exo_cat=past_exo_cat,
            B=B,
            L=L,
            device=x.device,
            dtype=x.dtype,
        )  # [B, L, E_p] or [B, L, 0]
        return torch.cat([x, exo], dim=-1)  # [B, L, 1+E_p]

    def _clamp(self, y: torch.Tensor) -> torch.Tensor:
        if not getattr(self.cfg, "final_clamp_nonneg", False):
            return y
        return F.softplus(y)

    def forward(
        self,
        x: torch.Tensor,  # [B, L, 1]
        future_exo: Optional[torch.Tensor] = None,  # [B, H, E] or None
        past_exo_cont: Optional[torch.Tensor] = None,
        past_exo_cat: Optional[torch.Tensor] = None,
        part_ids=None,
        mode: Optional[str] = None,
        **_,
    ) -> torch.Tensor:
        _validate_future_exo_contract(
            future_exo,
            batch_size=x.size(0),
            horizon=self.horizon,
            expected_dim=self.future_exo_dim,
        )

        # 1) norm
        x_n = self._maybe_revin_norm(x)

        # 2) encoder
        enc_in = self._make_encoder_input(x_n, past_exo_cont=past_exo_cont, past_exo_cat=past_exo_cat)
        h = self.encoder(enc_in)  # [B, L, D]

        # 3) optional LMM
        if self.lmm is not None:
            h = self.lmm(h)

        # 4) decoder or repeat last
        if self.decoder is not None:
            h_dec = self.decoder(h, future_exo=future_exo)  # [B, H, D]
        else:
            h_last = h[:, -1:, :]  # [B, 1, D]
            h_dec = h_last.expand(h_last.size(0), self.horizon, self.d_model)  # [B, H, D]

        out = self.head(h_dec)  # [B, H, out_mult] (or [B, H, 1])

        # RevIN std (없으면 1로 처리)
        if self.revin is None:
            stdev = torch.ones(out.size(0), 1, 1, device=out.device, dtype=out.dtype)
        else:
            stdev = self.revin.std.clamp_min(1e-6)  # [B,1,1]

        # 6) denorm / clamp
        if self.out_mult == 1:
            out = self._maybe_revin_denorm(out)  # [B, H, 1]
            out = self._clamp(out)
            return out.squeeze(-1)  # [B, H]

        elif self.out_mult == 2:
            loc = out[..., 0:1]
            scale_raw = out[..., 1:2]

            loc = self._maybe_revin_denorm(loc)

            scale = F.softplus(scale_raw)
            raw_scale = (scale * stdev).clamp(min=1e-6, max=1e6)
            scale_raw = self._inv_softplus(raw_scale)

            return torch.cat([loc, scale_raw], dim=-1)  # [B, H, 2]

        elif self.out_mult == 3:
            df_raw = out[..., 0:1]
            loc = out[..., 1:2]
            scale_raw = out[..., 2:3]

            loc = self._maybe_revin_denorm(loc)

            scale = F.softplus(scale_raw)
            raw_scale = (scale * stdev).clamp(min=1e-6, max=1e6)
            scale_raw = self._inv_softplus(raw_scale)

            return torch.cat([df_raw, loc, scale_raw], dim=-1)  # [B, H, 3]

        else:
            # distribution/packed outputs: do not denorm here
            return out


class TitanBaseModel(_TitanBase):
    def __init__(self, cfg: TitanConfig, out_mult: Optional[int] = None, param_names: Optional[List[str]] = None):
        super().__init__(cfg, has_memory=False, has_decoder=True, out_mult=out_mult, param_names=param_names)

    @classmethod
    def from_config(cls, config: "TitanConfig"):
        return cls(cfg=config)


class TitanLMMModel(_TitanBase):
    def __init__(self, cfg: TitanConfig, out_mult: Optional[int] = None, param_names: Optional[List[str]] = None):
        super().__init__(cfg, has_memory=True, has_decoder=True, out_mult=out_mult, param_names=param_names)

    @classmethod
    def from_config(cls, config: "TitanConfig"):
        return cls(cfg=config)


class TitanSeq2SeqModel(_TitanBase):
    def __init__(self, cfg: TitanConfig, out_mult: Optional[int] = None, param_names: Optional[List[str]] = None):
        super().__init__(cfg, has_memory=True, has_decoder=True, out_mult=out_mult, param_names=param_names)

    @classmethod
    def from_config(cls, config: "TitanConfig"):
        return cls(cfg=config)

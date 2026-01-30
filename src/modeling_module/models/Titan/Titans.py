from __future__ import annotations

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling_module.models.Titan import TitanConfig, MemoryEncoder, LMM, TitanDecoder
from modeling_module.models.Titan.common.decoder import TitanCrossAttnDecoder
from modeling_module.models.common_layers.RevIN import RevIN


class _PastExoEmbed(nn.Module):
    """
    Past exogenous embedding:
      - continuous: identity (concatenate)
      - categorical: embedding tables then concatenate
    """
    def __init__(self, cont_dim: int, cat_dims: Optional[List[int]], cat_embed_dims: Optional[List[int]]):
        super().__init__()
        self.cont_dim = int(cont_dim)
        self.cat_dims = list(cat_dims) if cat_dims else []
        if cat_embed_dims is None and self.cat_dims:
            cat_embed_dims = [min(16, max(2, d // 4)) for d in self.cat_dims]
        self.cat_embed_dims = list(cat_embed_dims) if cat_embed_dims else []

        assert len(self.cat_dims) == len(self.cat_embed_dims), "cat_dims and cat_embed_dims length mismatch"

        self.cat_embs = nn.ModuleList(
            [nn.Embedding(num_embeddings=int(cd), embedding_dim=int(ed)) for cd, ed in zip(self.cat_dims, self.cat_embed_dims)]
        )

        self.out_dim = self.cont_dim + sum(self.cat_embed_dims)

    def forward(
        self,
        past_exo_cont: Optional[torch.Tensor],
        past_exo_cat: Optional[torch.Tensor],
        B: int,
        L: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        feats = []

        # cont
        if self.cont_dim > 0:
            if past_exo_cont is None:
                feats.append(torch.zeros(B, L, self.cont_dim, device=device, dtype=dtype))
            else:
                feats.append(past_exo_cont.to(device=device, dtype=dtype))

        # cat
        if len(self.cat_embs) > 0:
            if past_exo_cat is None:
                past_exo_cat = torch.zeros(B, L, len(self.cat_embs), device=device, dtype=torch.long)
            else:
                past_exo_cat = past_exo_cat.to(device=device, dtype=torch.long)

            for j, emb in enumerate(self.cat_embs):
                ej = emb(past_exo_cat[..., j])  # [B, L, ed]
                feats.append(ej.to(dtype=dtype))

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
        out_mult: int = 1,
        param_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.lookback = int(cfg.lookback)
        self.horizon = int(cfg.horizon)

        self.d_model = int(cfg.d_model)
        self.out_mult = int(out_mult)
        self.param_names = param_names

        # RevIN
        self.use_revin = bool(getattr(cfg, "use_revin", True))
        self.revin = RevIN(int(getattr(cfg, "enc_in", 1)), affine = False, subtract_last=True) if self.use_revin else None

        # Past exo embed
        self.past_exo_embed = _PastExoEmbed(
            cont_dim=int(getattr(cfg, "past_exo_cont_dim", 0)),
            cat_dims=getattr(cfg, "past_exo_cat_dims", None),
            cat_embed_dims=getattr(cfg, "past_exo_cat_embed_dims", None),
        )

        # Encoder input dim
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
                exo_dim=int(getattr(cfg, "exo_dim", 0)),
                n_heads=int(getattr(cfg, "dec_n_heads", getattr(cfg, "n_heads", 8))),
                dropout=float(cfg.dropout),
                use_step_emb=True,
                pre_norm=True,
            )
        else:
            self.decoder = None

        # h_dec is [B, H, D], so projecting per step is correct.
        self.head = nn.Linear(self.d_model, self.out_mult)

        # proj out_features를 out_mult로
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
        return y + torch.log(-torch.expm1(-y))  # <-- 핵심: expm1(-y)

    def _maybe_revin_norm(self, x: torch.Tensor):
        if self.revin is None:
            return x, None
        x_n = self.revin(x, mode="norm")
        return x_n

    def _maybe_revin_denorm(self, y: torch.Tensor):
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
        )  # [B, L, E_p] or [B,L,0]
        return torch.cat([x, exo], dim=-1)  # [B, L, 1+E_p]

    def _clamp(self, y: torch.Tensor) -> torch.Tensor:
        if not getattr(self.cfg, 'final_clamp_nonneg', False):
            return y
        return F.softplus(y)  # 항상 양수 + 음수 구간도 gradient 존재

    def forward(
        self,
        x: torch.Tensor,                          # [B, L, 1]
        future_exo: Optional[torch.Tensor] = None, # [B, H, E] or None
        past_exo_cont: Optional[torch.Tensor] = None,
        past_exo_cat: Optional[torch.Tensor] = None,
        part_ids=None,
        mode: Optional[str] = None,
        **_,
    ) -> torch.Tensor:
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
            h_last = h[:, -1:, :]  # [B,1,D]
            h_dec = h_last.expand(h_last.size(0), self.horizon, self.d_model)  # [B,H,D]

        out = self.head(h_dec)

        # RevIN 통계: mean/stdev는 norm 호출 때 내부 저장됨
        # shape: [B,1,C], 여기서 C=1
        stdev = self.revin.std.clamp_min(1e-6)  # 0 방지 [B,1,1]

        # 6) denorm / clamp
        if self.out_mult == 1:
            out = self._maybe_revin_denorm(out)  # out: [B,H,1]
            out = self._clamp(out)
            return out.squeeze(-1)  # [B, H]

        elif self.out_mult == 2:
            loc = out[..., 0:1]
            scale_raw = out[..., 1:2]

            # loc denorm
            loc = self._maybe_revin_denorm(loc)
            # scale_raw를 "loss의 softplus 이후 scale"이 raw 스케일이 되게 변환:
            # scale = softplus(scale_raw_norm)  (norm 스케일)
            # raw_scale = scale * stdev
            # -> scale_raw_out = inv_softplus(raw_scale)
            scale = F.softplus(scale_raw)
            raw_scale = (scale * stdev).clamp(min=1e-6, max=1e6)  # broadcast to [B,H,1]
            scale_raw = self._inv_softplus(raw_scale)
            return torch.cat([loc, scale_raw], dim = -1) # [B, H, 2]

        elif self.out_mult == 3:
            df_raw = out[..., 0:1]
            loc = out[..., 1:2]
            scale_raw = out[..., 2:3]

            loc = self._maybe_revin_denorm(loc)
            scale = F.softplus(scale_raw)
            raw_scale = (scale * stdev).clamp(min=1e-6, max=1e6)
            scale_raw = self._inv_softplus(raw_scale)

            return torch.cat([df_raw, loc, scale_raw], dim = -1) # [B, H, 3]

        else:
            # distribution/packed outputs: do not denorm here
            return out


class TitanBaseModel(_TitanBase):
    def __init__(self, cfg: TitanConfig, out_mult: int = 1, param_names: Optional[List[str]] = None):
        super().__init__(cfg, has_memory=False, has_decoder=True, out_mult=out_mult, param_names=param_names)

    @classmethod
    def from_config(cls, config: "TitanConfig"):
        return cls(cfg=config)


class TitanLMMModel(_TitanBase):
    def __init__(self, cfg: TitanConfig, out_mult: int = 1, param_names: Optional[List[str]] = None):
        super().__init__(cfg, has_memory=True, has_decoder=True, out_mult=out_mult, param_names=param_names)

    @classmethod
    def from_config(cls, config: "TitanConfig"):
        return cls(cfg=config)


class TitanSeq2SeqModel(_TitanBase):
    def __init__(self, cfg: TitanConfig, out_mult: int = 1, param_names: Optional[List[str]] = None):
        super().__init__(cfg, has_memory=True, has_decoder=True, out_mult=out_mult, param_names=param_names)

    @classmethod
    def from_config(cls, config: "TitanConfig"):
        return cls(cfg=config)
"""PatchMixer_clean.py

A simplified, *optimizer-safe* PatchMixer implementation.

Design goals
- No parameter creation/resizing inside forward() (prevents optimizer missing params).
- Accepts:
  - future exogenous: future_exo (B, H, E)
  - past exogenous continuous: past_exo_cont (B, L, E_p)
  - past exogenous categorical: past_exo_cat (B, L, K) with fixed vocab sizes
  - part_ids (B,)
- Robust to dim mismatches by **pad/slice (no new params)**:
  - future_exo last dim != cfg.exo_dim
  - past_exo_cont last dim != cfg.past_exo_cont_dim
  - past_exo_cat last dim != cfg.past_exo_cat_dim

Past exogenous injection
- mode: cfg.past_exo_mode in {'none','z_gate'}
- z_gate: pool past exo over time -> vector -> project to z-dim -> gated add

Notes
- For categorical past exo, you MUST provide fixed vocab sizes/embed dims in config
  (no dynamic embedding table resize).
- If you do not want categorical, set past_exo_cat_dim=0.

This file is intended to replace modeling_module/models/PatchMixer/PatchMixer.py
(or be imported by your model_builder).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from modeling_module.models.PatchMixer.backbone import PatchMixerBackbone, MultiScalePatchMixerBackbone
from modeling_module.models.PatchMixer.common.configs import PatchMixerConfig
from modeling_module.models.common_layers.RevIN import RevIN
from modeling_module.models.common_layers.heads.quantile_heads.decomposition_quantile_head import DecompositionQuantileHead
from modeling_module.utils.exogenous_utils import apply_exo_shift_linear
from modeling_module.utils.temporal_expander import TemporalExpander


# -------------------------
# helpers
# -------------------------

def _pad_or_slice_last_dim(x: torch.Tensor, target_dim: int, *, pad_value: float = 0.0) -> torch.Tensor:
    """Adjust last-dim to target_dim by slicing or right-padding with constants.

    This is intentionally parameter-free to remain optimizer-safe.
    """
    if x is None:
        return x
    if target_dim <= 0:
        # treat as disabled
        return None
    if x.size(-1) == target_dim:
        return x
    if x.size(-1) > target_dim:
        return x[..., :target_dim]
    # pad
    pad = target_dim - x.size(-1)
    pad_shape = list(x.shape[:-1]) + [pad]
    pad_t = x.new_full(pad_shape, pad_value)
    return torch.cat([x, pad_t], dim=-1)


def _infer_patch_cfgs(lookback: int, n_branches: int = 3) -> List[Tuple[int, int, int]]:
    # minimal deterministic presets
    assert lookback >= 8
    fracs = [1 / 4, 1 / 2, 3 / 4][:n_branches]
    raw = [max(4, min(lookback, int(round(lookback * f)))) for f in fracs]
    P = sorted(list(dict.fromkeys(raw)))
    cfgs: List[Tuple[int, int, int]] = []
    for i, p in enumerate(P):
        s = max(1, p // 2)
        k = [3, 5, 7][min(i, 2)]
        if k % 2 == 0:
            k += 1
        cfgs.append((p, s, k))
    return cfgs


# =====================================================================
# Core mixin: past/future exo handling (parameter-safe)
# =====================================================================

class _ExoMixin(nn.Module):
    def _init_exo(self, cfg: PatchMixerConfig, *, z_dim: int):
        # future exo
        self.exo_dim = int(getattr(cfg, "exo_dim", 0) or 0)
        self.exo_is_normalized_default = bool(getattr(cfg, "exo_is_normalized_default", False))
        self.exo_head: Optional[nn.Module] = None
        if self.exo_dim > 0:
            self.exo_head = nn.Sequential(
                nn.Linear(self.exo_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

        # past exo
        self.past_exo_mode = str(getattr(cfg, "past_exo_mode", "none") or "none").lower()
        if self.past_exo_mode not in ("none", "z_gate"):
            # keep it simple and stable
            raise ValueError(f"Unsupported past_exo_mode={self.past_exo_mode}. Use 'none' or 'z_gate'.")

        self.past_exo_cont_dim = int(getattr(cfg, "past_exo_cont_dim", 0) or 0)
        self.past_exo_cat_dim = int(getattr(cfg, "past_exo_cat_dim", 0) or 0)

        # categorical: fixed vocab sizes/embed dims
        vocab_sizes = tuple(getattr(cfg, "past_exo_cat_vocab_sizes", ()) or ())
        embed_dims = tuple(getattr(cfg, "past_exo_cat_embed_dims", ()) or ())

        if self.past_exo_cat_dim > 0:
            if len(vocab_sizes) != self.past_exo_cat_dim or len(embed_dims) != self.past_exo_cat_dim:
                raise ValueError(
                    "past_exo_cat_dim>0 requires past_exo_cat_vocab_sizes and past_exo_cat_embed_dims with same length"
                )

        self._cat_embs: Optional[nn.ModuleList] = None
        self._cat_embed_total = 0
        if self.past_exo_cat_dim > 0:
            embs = []
            total = 0
            for vs, ed in zip(vocab_sizes, embed_dims):
                embs.append(nn.Embedding(int(vs), int(ed)))
                total += int(ed)
            self._cat_embs = nn.ModuleList(embs)
            self._cat_embed_total = total

        # z-gate projection (built once)
        self._z_exo_proj: Optional[nn.Linear] = None
        self._z_gate: Optional[nn.Linear] = None
        if self.past_exo_mode == "z_gate":
            in_dim = self.past_exo_cont_dim + self._cat_embed_total
            # allow "no past exo" even if mode=z_gate (then it is a no-op)
            if in_dim > 0:
                self._z_exo_proj = nn.Linear(in_dim, z_dim, bias=True)
                self._z_gate = nn.Linear(z_dim, z_dim, bias=True)

    def _pool_past_exo(self, past_exo_cont: Optional[torch.Tensor], past_exo_cat: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Return pooled past-exo vector (B, E_total) in float."""
        feats: List[torch.Tensor] = []

        if past_exo_cont is not None and past_exo_cont.numel() > 0 and self.past_exo_cont_dim > 0:
            pe = _pad_or_slice_last_dim(past_exo_cont.float(), self.past_exo_cont_dim, pad_value=0.0)
            feats.append(pe.mean(dim=1))  # (B, E_c)

        if past_exo_cat is not None and past_exo_cat.numel() > 0 and self.past_exo_cat_dim > 0:
            # (B, L, K) -> ensure K
            pc = _pad_or_slice_last_dim(past_exo_cat.long(), self.past_exo_cat_dim, pad_value=0)
            assert self._cat_embs is not None
            emb_list: List[torch.Tensor] = []
            for j, emb in enumerate(self._cat_embs):
                ids = pc[..., j]
                # safe clamp to vocab
                ids = ids.clamp_min(0).clamp_max(emb.num_embeddings - 1)
                e = emb(ids)  # (B, L, d)
                emb_list.append(e.mean(dim=1))
            feats.append(torch.cat(emb_list, dim=-1))

        if not feats:
            return None
        return torch.cat(feats, dim=-1)

    def _inject_past_exo_z_gate(self, z: torch.Tensor, past_exo_cont: Optional[torch.Tensor], past_exo_cat: Optional[torch.Tensor]) -> torch.Tensor:
        if self.past_exo_mode != "z_gate":
            return z
        if self._z_exo_proj is None or self._z_gate is None:
            return z  # nothing to inject

        v = self._pool_past_exo(past_exo_cont, past_exo_cat)
        if v is None:
            return z

        v = _pad_or_slice_last_dim(v, self._z_exo_proj.in_features, pad_value=0.0)
        exo_z = self._z_exo_proj(v)
        gate = torch.sigmoid(self._z_gate(z))
        return z + gate * exo_z

    def _apply_future_exo_shift(self, y: torch.Tensor, future_exo: Optional[torch.Tensor], *, exo_is_normalized: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (y_after, ex_shift_raw). ex_shift_raw is (B, H) in y-space."""
        if future_exo is None or self.exo_head is None or self.exo_dim <= 0:
            return y, None

        fe = _pad_or_slice_last_dim(future_exo.float(), self.exo_dim, pad_value=0.0)
        ex = apply_exo_shift_linear(
            self.exo_head,
            fe,
            horizon=int(getattr(self, "horizon")),
            out_dtype=y.dtype,
            out_device=y.device,
        )
        if exo_is_normalized:
            y = y + ex
        return y, ex


# =====================================================================
# Point model
# =====================================================================

class PatchMixerPointModel(_ExoMixin):
    def __init__(self, cfg: PatchMixerConfig):
        super().__init__()
        self.configs = cfg
        self.horizon = int(cfg.horizon)
        self.f_out = int(getattr(cfg, "f_out", 128))
        self.final_nonneg = bool(getattr(cfg, "final_nonneg", True))

        # backbone
        self.backbone = PatchMixerBackbone(configs=cfg)
        z_dim = int(getattr(self.backbone, "out_dim", getattr(self.backbone, "patch_repr_dim", 0)))
        if z_dim <= 0:
            raise RuntimeError("Backbone must expose out_dim or patch_repr_dim")
        self.z_dim = z_dim

        # RevIN
        self.use_revin = bool(getattr(cfg, "use_revin", True))
        self.revin = RevIN(int(getattr(cfg, "enc_in", 1)))

        # optional part embedding
        self.use_part_embedding = bool(getattr(cfg, "use_part_embedding", False))
        self.part_emb: Optional[nn.Embedding] = None
        self.z_fuser: Optional[nn.Linear] = None
        if self.use_part_embedding and int(getattr(cfg, "part_vocab_size", 0)) > 0:
            pdim = int(getattr(cfg, "part_embed_dim", 16))
            self.part_emb = nn.Embedding(int(cfg.part_vocab_size), pdim)
            self.z_fuser = nn.Linear(z_dim + pdim, z_dim)

        # expander
        self.expander = TemporalExpander(
            d_in=z_dim,
            horizon=self.horizon,
            f_out=self.f_out,
            dropout=float(getattr(cfg, "dropout", 0.1)),
            use_sinus=True,
            season_period=int(getattr(cfg, "expander_season_period", 52)),
            max_harmonics=int(getattr(cfg, "expander_max_harmonics", getattr(cfg, "max_harmonics", 16))),
            use_conv=True,
        )

        # head (simple)
        head_hidden = int(getattr(cfg, "head_hidden", self.f_out))
        self.pre_ln = nn.LayerNorm(self.f_out)
        self.head = nn.Sequential(
            nn.Linear(self.f_out, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )

        # scale stabilizers (optional)
        self.learn_output_scale = bool(getattr(cfg, "learn_output_scale", True))
        if self.learn_output_scale:
            self.out_scale = nn.Parameter(torch.tensor(1.0))
            self.out_bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("out_scale", torch.tensor(1.0))
            self.register_buffer("out_bias", torch.tensor(0.0))

        self.learn_dw_gain = bool(getattr(cfg, "learn_dw_gain", True))
        self.dw_head = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        if self.learn_dw_gain:
            self.dw_gain = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("dw_gain", torch.tensor(1.0))

        # init exo handling (must be after z_dim known)
        self._init_exo(cfg, z_dim=z_dim)

    def forward(
        self,
        x: torch.Tensor,  # (B, L, C)
        future_exo: Optional[torch.Tensor] = None,  # (B, H, E)
        *,
        past_exo_cont: Optional[torch.Tensor] = None,  # (B, L, E_p)
        past_exo_cat: Optional[torch.Tensor] = None,  # (B, L, K)
        part_ids: Optional[torch.Tensor] = None,  # (B,)
        exo_is_normalized: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        if exo_is_normalized is None:
            exo_is_normalized = self.exo_is_normalized_default

        # 1) RevIN + backbone
        x_in = self.revin(x, "norm") if self.use_revin else x
        z = self.backbone(x_in)
        if z.dim() != 2 or z.size(-1) != self.z_dim:
            raise RuntimeError(f"Unexpected backbone output shape: {tuple(z.shape)} expected (*, {self.z_dim})")

        # 2) past exo injection (z_gate)
        z = self._inject_past_exo_z_gate(z, past_exo_cont, past_exo_cat)

        # 3) part embedding
        if self.part_emb is not None and part_ids is not None:
            pe = self.part_emb(part_ids.long())
            z = self.z_fuser(torch.cat([z, pe], dim=-1))

        # 4) expand + head
        f = self.pre_ln(self.expander(z))          # (B, H, F)
        y = self.head(f).squeeze(-1)               # (B, H)

        # 5) future exo shift
        y, ex = self._apply_future_exo_shift(y, future_exo, exo_is_normalized=exo_is_normalized)

        # 6) scale + dw residual
        y = y * self.out_scale + self.out_bias
        y = y + self.dw_gain * self.dw_head(y.unsqueeze(1)).squeeze(1)

        # 7) denorm + (if needed) add exo in raw space
        if self.use_revin:
            y = self.revin(y.unsqueeze(-1), "denorm").squeeze(-1)
        if (ex is not None) and (not exo_is_normalized):
            y = y + ex

        if self.final_nonneg and (not self.training):
            y = torch.clamp_min(y, 0.0)
        return y


# =====================================================================
# Quantile model
# =====================================================================

class PatchMixerQuantileModel(_ExoMixin):
    def __init__(self, cfg: PatchMixerConfig):
        super().__init__()
        self.is_quantile = True
        self.configs = cfg
        self.horizon = int(cfg.horizon)
        self.f_out = int(getattr(cfg, "f_out", 128))
        self.final_nonneg = bool(getattr(cfg, "final_nonneg", True))

        # multiscale backbone
        patch_cfgs = tuple(getattr(cfg, "patch_cfgs", ()) or ())
        if not patch_cfgs:
            patch_cfgs = tuple(_infer_patch_cfgs(int(cfg.lookback), n_branches=3))
        self.backbone = MultiScalePatchMixerBackbone(
            base_configs=cfg,
            patch_cfgs=patch_cfgs,
            per_branch_dim=int(getattr(cfg, "per_branch_dim", 64)),
            fused_dim=int(getattr(cfg, "fused_dim", 128)),
            fusion=str(getattr(cfg, "fusion", "concat")),
        )
        self.z_dim = int(self.backbone.out_dim)

        # RevIN
        self.use_revin = bool(getattr(cfg, "use_revin", True))
        self.revin = RevIN(int(getattr(cfg, "enc_in", 1)))

        # part embedding
        self.use_part_embedding = bool(getattr(cfg, "use_part_embedding", False))
        self.part_emb: Optional[nn.Embedding] = None
        self.z_fuser: Optional[nn.Linear] = None
        if self.use_part_embedding and int(getattr(cfg, "part_vocab_size", 0)) > 0:
            pdim = int(getattr(cfg, "part_embed_dim", 16))
            self.part_emb = nn.Embedding(int(cfg.part_vocab_size), pdim)
            self.z_fuser = nn.Linear(self.z_dim + pdim, self.z_dim)

        # expander
        self.expander = TemporalExpander(
            d_in=self.z_dim,
            horizon=self.horizon,
            f_out=self.f_out,
            dropout=float(getattr(cfg, "dropout", 0.1)),
            use_sinus=True,
            season_period=int(getattr(cfg, "expander_season_period", 52)),
            max_harmonics=int(getattr(cfg, "expander_max_harmonics", getattr(cfg, "max_harmonics", 16))),
            use_conv=True,
        )

        # quantile head
        head_hidden = int(getattr(cfg, "head_hidden", 128))
        self.head = DecompositionQuantileHead(
            in_features=self.f_out,
            quantiles=list(getattr(cfg, "quantiles", (0.1, 0.5, 0.9))),
            hidden=head_hidden,
            dropout=float(getattr(cfg, "head_dropout", 0.0) or 0.0),
            mid=0.5,
            use_trend=True,
            fourier_k=int(getattr(cfg, "expander_n_harmonics", getattr(cfg, "expander_n_harmonics", 8))),
            agg="mean",
        )

        # init exo handling
        self._init_exo(cfg, z_dim=self.z_dim)

    def forward(
        self,
        x: torch.Tensor,
        future_exo: Optional[torch.Tensor] = None,
        *,
        past_exo_cont: Optional[torch.Tensor] = None,
        past_exo_cat: Optional[torch.Tensor] = None,
        part_ids: Optional[torch.Tensor] = None,
        exo_is_normalized: Optional[bool] = None,
        **kwargs,
    ):
        if exo_is_normalized is None:
            exo_is_normalized = self.exo_is_normalized_default

        x_in = self.revin(x, "norm") if self.use_revin else x
        z = self.backbone(x_in)
        if z.dim() != 2 or z.size(-1) != self.z_dim:
            raise RuntimeError(f"Unexpected backbone output shape: {tuple(z.shape)} expected (*, {self.z_dim})")

        z = self._inject_past_exo_z_gate(z, past_exo_cont, past_exo_cat)

        if self.part_emb is not None and part_ids is not None:
            pe = self.part_emb(part_ids.long())
            z = self.z_fuser(torch.cat([z, pe], dim=-1))

        f = self.expander(z)  # (B, H, F)
        q = self.head(f)      # (B, Q, H) or (B, H, Q)

        # future exo shift (in normalized space)
        ex = None
        if future_exo is not None and self.exo_head is not None and self.exo_dim > 0:
            fe = _pad_or_slice_last_dim(future_exo.float(), self.exo_dim, pad_value=0.0)
            ex = apply_exo_shift_linear(
                self.exo_head,
                fe,
                horizon=self.horizon,
                out_dtype=q.dtype,
                out_device=q.device,
            )
            if exo_is_normalized:
                if q.dim() == 3 and q.shape[1] != self.horizon:
                    # (B, Q, H)
                    q = q + ex.unsqueeze(1)
                else:
                    # (B, H, Q)
                    q = q + ex.unsqueeze(-1)

        # denorm quantiles in raw space
        if q.dim() != 3:
            raise RuntimeError(f"Unexpected quantile tensor rank: {q.dim()}")

        if q.shape[1] == self.horizon:
            q = q.transpose(1, 2)  # -> (B, Q, H)

        # q: (B, Q, H)
        qs: List[torch.Tensor] = []
        for i in range(q.size(1)):
            qi = q[:, i, :]
            qi = self.revin(qi.unsqueeze(-1), "denorm").squeeze(-1) if self.use_revin else qi
            qs.append(qi.unsqueeze(1))
        q_raw = torch.cat(qs, dim=1)  # (B, Q, H)

        if (ex is not None) and (not exo_is_normalized):
            q_raw = q_raw + ex.unsqueeze(1)

        if self.final_nonneg and (not self.training):
            q_raw = torch.clamp_min(q_raw, 0.0)
        return {"q": q_raw}


# ---------------------------------------------------------------------
# Backward-compatible aliases (if your builders import BaseModel/QuantileModel)
# ---------------------------------------------------------------------

BaseModel = PatchMixerPointModel
QuantileModel = PatchMixerQuantileModel

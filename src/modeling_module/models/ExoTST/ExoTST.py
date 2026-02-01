from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling_module.models.ExoTST.backbone import num_patches, PatchEmbedding1D, ExoEncoder, CrossTemporalFusionLayer, \
    EndoDecoder, HorizonMLPHead, HorizonDistMLPHead
from modeling_module.models.ExoTST.configs import ExoTSTConfig
from modeling_module.models.common_layers.RevIN import RevIN


def _nan_policy_apply(x: torch.Tensor, policy: str) -> torch.Tensor:
    """
    x: (B, T, E)
    policy:
        - "zero": NaN -> 0
        - "zero+indicator": NaN -> 0 and concat indicator (same dim) -> (B, T, 2E)
    """
    if x is None:
        return None
    if policy == 'zero':
        return torch.nan_to_num(x, nan = 0.0, posinf = 0.0, neginf = 0.0)
    if policy == 'zero+indicator':
        ind = torch.isnan(x).to(x.dtype)
        x2 = torch.nan_to_num(x, nan = 0.0, posinf = 0.0, neginf = 0.0)
        return torch.cat([x2, ind], dim=-1)
    raise ValueError(f"Unknown exo_nan_policy: {policy}")


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    # stable inverse softplus: inv_sp(x) = x + log( -expm1(-x) )
    x = torch.clamp(x, min=1e-8)
    return x + torch.log(-torch.expm1(-x))

def _denorm_scale_with_revin(revin: RevIN, scale: torch.Tensor) -> torch.Tensor:
    """
    scale: (B, H) or (B, H, 1) assumed "std-like positive scale"
    PatchTSTModel._denorm_scale 로직과 동일한 역할
    """
    if scale.dim() == 2:
        s = scale.unsqueeze(-1)  # (B,H,1)
    else:
        s = scale

    # affine 역변환: std류는 |w|로 나누는 편이 안전
    if getattr(revin, "affine", False) and hasattr(revin, "affine_weight"):
        w = revin.affine_weight.view(1, 1, -1)  # (1,1,C)
        s = s / (w.abs() + 1e-8)

    # std 역변환: RevIN 구현체에 std가 저장되어 있다는 가정(당신 프로젝트 PatchTST와 동일)
    if hasattr(revin, "std") and revin.std is not None:
        s = s * revin.std  # (B,1,C) broadcast

    return s.squeeze(-1)  # (B,H)

class ExoTST(nn.Module):
    """
    ExoTST main model (paper-aligned, project-friendly).

    Forward signature aligned to your adapter:
        forward(x, future_exo = None, past_exo_cont=None, part_ids=None, mode=None)

    Shapes:
        x:              (B, L, Cy)
        past_exo:       (B, L, E_p) or None
        future_exo:     (B, H, E_f) or None
        yhat:           (B, H, Cy)
    """

    def __init__(self, cfg: ExoTSTConfig):
        super().__init__()
        self.cfg = cfg

        self.lookback = int(cfg.lookback)
        self.horizon = int(cfg.horizon)
        self.y_dim = int(cfg.y_dim)



        # compute token counts (fixed by config)
        self.ny = num_patches(self.lookback, cfg.patch_len, cfg.stride)
        self.np = num_patches(self.lookback, cfg.patch_len, cfg.stride)
        self.nf = num_patches(self.horizon, cfg.patch_len, cfg.stride)

        # We need max_tokens for learnable pos-enc
        # - endogenous: Ny
        # - exogenous: Np (+agg) or Nf (+agg)
        max_endo_tokens = self.ny
        max_exo_tokens = max(self.np, self.nf) + 1 # +agg token

        # -------------------------
        # RevIN for endogenous
        # -------------------------
        if self.use_revin:
            self.revin = RevIN(
                num_features = self.y_dim,
                eps = cfg.revin_eps,
                affine = cfg.revin_affine,
                subtract_last = cfg.revin_subtract_last,
            )
        else:
            self.revin = None


        # -------------------------
        # loss / head type
        # -------------------------
        self.loss = cfg.loss
        self.loss_type = "point" if not hasattr(self.loss, "distribution") else "distribution"

        if self.loss_type == "point":
            self.param_names = None
            self.out_mult = 1
            self.head = HorizonMLPHead(
                ny=self.ny,
                d_model=cfg.d_model,
                horizon=self.horizon,
                y_dim=self.y_dim,
                dropout=cfg.dropout,
            )

        elif self.loss_type == "distribution":
            # PatchTST와 동일 컨벤션
            self.param_names = list(self.loss.param_names)  # 예: ["-df","-loc","-scale"]
            self.out_mult = int(self.loss.outputsize_multiplier)  # 예: StudentT=3, Normal=2
            self.head = HorizonDistMLPHead(
                ny=self.ny,
                d_model=cfg.d_model,
                horizon=self.horizon,
                y_dim=self.y_dim,
                out_mult=self.out_mult,
                dropout=cfg.dropout,
            )

            self.dist_min_scale = float(getattr(cfg, "dist_min_scale", 1e-3))

        # -------------------------
        # (2) Patch embeddings
        # -------------------------
        self.endo_patch = PatchEmbedding1D(
            patch_len = cfg.patch_len,
            stride = cfg.stride,
            d_model = cfg.d_model,
            add_agg_token = False,
            max_tokens = max_endo_tokens,
            dropout = cfg.dropout,
        )

        # Exogenous patch embeddings (past / future separate)
        self.past_exo_patch = PatchEmbedding1D(
            patch_len=cfg.patch_len,
            stride=cfg.stride,
            d_model=cfg.d_model,
            add_agg_token=True,
            max_tokens=max_exo_tokens,
            dropout=cfg.dropout,
        )
        self.future_exo_patch = PatchEmbedding1D(
            patch_len=cfg.patch_len,
            stride=cfg.stride,
            d_model=cfg.d_model,
            add_agg_token=True,
            max_tokens=max_exo_tokens,
            dropout=cfg.dropout,
        )

        # -------------------------
        # (3) Two exo encoders
        # -------------------------
        self.past_exo_enc = ExoEncoder(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            layers=cfg.exo_enc_layers,
            dropout=cfg.dropout,
            attn_dropout=cfg.attn_dropout,
        )
        self.future_exo_enc = ExoEncoder(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            layers=cfg.exo_enc_layers,
            dropout=cfg.dropout,
            attn_dropout=cfg.attn_dropout,
        )

        # -------------------------
        # (4) Fusion stack
        # -------------------------
        self.fusion = nn.ModuleList(
            [CrossTemporalFusionLayer(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(cfg.fusion_layers)]
        )

        # -------------------------
        # (5) Endogenous decoder
        # -------------------------
        self.endo_dec = EndoDecoder(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            layers=cfg.endo_dec_layers,
            dropout=cfg.dropout,
        )

        # -------------------------
        # (6) Head
        # -------------------------
        self.head = HorizonMLPHead(
            ny=self.ny,
            d_model=cfg.d_model,
            horizon=self.horizon,
            y_dim=self.y_dim,
            dropout=cfg.dropout,
        )
    def _build_exo_memory(self, hp: torch.Tensor, hf: torch.Tensor) -> torch.Tensor:
        """
        hp: (B, Cx, Np+1, D)
        hf: (B, Cx, Nf+1, D)
        return exo_mem: (B, M, D)
        """
        if self.cfg.exo_memory_mode == "agg":
            # use only agg tokens from past/future: (B,C,1,D) + (B,C,1,D) -> (B,2C,D)
            ap = hp[:, :, :1, :]
            af = hf[:, :, :1, :]
            mem = torch.cat([ap, af], dim=2)  # (B, C, 2, D)
            b, c, n, d = mem.shape
            return mem.reshape(b, c * n, d)

        # "all": concat full token sets per channel
        mem = torch.cat([hp, hf], dim=2)  # (B, C, (Np+1)+(Nf+1), D)
        b, c, n, d = mem.shape
        return mem.reshape(b, c * n, d)

    def forward(
            self,
            x: torch.Tensor,
            future_exo: Optional[torch.Tensor] = None,
            past_exo_cont: Optional[torch.Tensor] = None,
            past_exo_cat: Optional[torch.Tensor] = None,  # reserved
            part_ids: Optional[torch.Tensor] = None,  # reserved
            mode: Optional[str] = None,
    ) -> torch.Tensor:
        """
        x: (B, L, Cy)
        future_exo: (B, H, E_f) or None
        past_exo_cont: (B, L, E_p) or None
        """
        if x.dim() != 3:
            raise ValueError("ExoTST expects x shape (B, L, Cy)")

        b, L, cy = x.shape
        if self.cfg.strict_shape:
            if L != self.lookback:
                raise ValueError(f"lookback mismatch: got L={L}, expected {self.lookback}")
            if cy != self.y_dim:
                raise ValueError(f"y_dim mismatch: got Cy={cy}, expected {self.y_dim}")

        # -------------------------
        # 0) RevIN normalize endogenous
        # -------------------------
        if self.revin is not None:
            x_norm, stats = self.revin(x, 'norm')
        else:
            x_norm, stats = x, None

        # -------------------------
        # 1) Endogenous patch embedding
        # -------------------------
        y_tok = self.endo_patch(x_norm)  # (B, Cy, Ny, D)

        # -------------------------
        # 2) Prepare exogenous
        # -------------------------
        use_past = bool(self.cfg.use_past_exo) and (self.cfg.exo_dim_past > 0)
        use_future = bool(self.cfg.use_future_exo) and (self.cfg.exo_dim_future > 0)

        if use_past:
            if past_exo_cont is None or past_exo_cont.shape[-1] == 0:
                raise RuntimeError("[ExoTST] use_past_exo=True but past_exo_cont is None or dim==0")
            xp = _nan_policy_apply(past_exo_cont, self.cfg.exo_nan_policy)  # (B,L,E')  E' = E or 2E
        else:
            xp = None

        if use_future:
            if future_exo is None or future_exo.shape[-1] == 0:
                raise RuntimeError("[ExoTST] use_future_exo=True but future_exo is None or dim==0")
            xf = _nan_policy_apply(future_exo, self.cfg.exo_nan_policy)  # (B,H,E')
        else:
            xf = None

        # -------------------------
        # 3) Exogenous enc + fusion
        # -------------------------
        # If one side is disabled, we still can proceed by mirroring tokens,
        # but the "paper-aligned" ExoTST assumes both past & future exo exist.
        if xp is None or xf is None:
            raise RuntimeError("[ExoTST] ExoTST requires BOTH past and future exogenous to run (paper-aligned).")

        hp = self.past_exo_patch(xp)  # (B, Cx, Np+1, D)
        hf = self.future_exo_patch(xf)  # (B, Cx, Nf+1, D)

        hp = self.past_exo_enc(hp)
        hf = self.future_exo_enc(hf)

        for layer in self.fusion:
            hp, hf = layer(hp, hf)

        exo_mem = self._build_exo_memory(hp, hf)  # (B, M, D)

        # -------------------------
        # 4) Endogenous decoder (cross-attn to exo_mem)
        # -------------------------
        z = self.endo_dec(y_tok, exo_mem)  # (B, Cy, Ny, D)

        # -------------------------
        # 5) Head -> horizon
        # -------------------------
        head_out = self.head(z)

        # -------------------------
        # 6) Output by loss_type
        # -------------------------
        if self.loss_type == "point":
            # HorizonMLPHead는 (B,H,Cy) 반환
            yhat = head_out
            if self.revin is not None:
                yhat = self.revin(yhat, "denorm")
            # PatchTST처럼 Cy==1이면 (B,H)로 반환하고 싶다면:
            if self.y_dim == 1:
                return yhat.squeeze(-1)  # (B,H)
            return yhat  # (B,H,Cy)

        # ---- distribution branch ----
        # head_out expected (B,H,out_mult) (y_dim==1) or (B,H,Cy,out_mult)
        if not torch.is_tensor(head_out):
            raise TypeError(f"[ExoTST] head_out must be Tensor, got {type(head_out)}")

        if self.y_dim != 1:
            # 우선 PatchTST와 동일하게 univariate 기준으로 안전하게 제한
            raise RuntimeError("[ExoTST] distribution mode currently supports y_dim==1 only.")

        if head_out.dim() != 3 or head_out.size(-1) != self.out_mult:
            raise TypeError(
                f"[ExoTST] head_out must be (B,H,{self.out_mult}), got {type(head_out)} {getattr(head_out, 'shape', None)}"
            )

        params = {name: head_out[..., i] for i, name in enumerate(self.param_names)}

        # 1) loc denorm
        loc_n = params.get("-loc")
        if loc_n is None:
            raise RuntimeError(f"[ExoTST] '-loc' not found in param_names={self.param_names}")

        loc = self.revin(loc_n.unsqueeze(-1), "denorm").squeeze(-1) if self.revin is not None else loc_n

        # 2) scale: raw -> pos -> denorm_scale -> inverse-softplus(raw_for_loss)
        scale_raw_n = params.get("-scale")
        if scale_raw_n is None:
            raise RuntimeError(f"[ExoTST] '-scale' not found in param_names={self.param_names}")

        scale_pos = F.softplus(scale_raw_n) + getattr(self, "dist_min_scale", 1e-3)

        if self.revin is not None:
            scale_pos = _denorm_scale_with_revin(self.revin, scale_pos)  # (B,H)
        else:
            scale_pos = torch.clamp(scale_pos, min=getattr(self, "dist_min_scale", 1e-3))

        # loss가 다시 softplus를 타므로 raw로 되돌려 반환
        x = torch.clamp(scale_pos - getattr(self, "dist_min_scale", 1e-3), min=1e-8)
        scale_raw_for_loss = _inverse_softplus(x)

        # 3) param_names 순서대로 pack
        outs = []
        for name in self.param_names:
            if name == "-loc":
                outs.append(loc)
            elif name == "-scale":
                outs.append(scale_raw_for_loss)
            elif name == "-df":
                # StudentT 스타일(필요 시 PatchTST와 동일)
                df_raw = params.get("-df")
                if df_raw is None:
                    raise RuntimeError(f"[ExoTST] '-df' not found in param_names={self.param_names}")
                df_val = F.softplus(df_raw + 2.0)
                outs.append(df_val)
            else:
                # 필요 시 확장: -logits, -total_count, -log_mu 등
                raise RuntimeError(f"[ExoTST] Unsupported param name: {name}")

        return torch.stack(outs, dim=-1)  # (B,H,out_mult)
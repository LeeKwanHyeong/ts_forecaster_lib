import torch
import torch.nn as nn
import torch.nn.functional as F
# decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class TitanCrossAttnDecoder(nn.Module):
    """
    Cross-Attention horizon decoder.

    Query  : per-step tokens (step embedding + optional future_exo projection)
    Key/Val: encoder sequence h_enc

    Output : [B, H, D]
    """
    def __init__(
        self,
        d_model: int,
        horizon: int,
        exo_dim: int = 0,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_step_emb: bool = True,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.horizon = int(horizon)
        self.exo_dim = int(exo_dim)
        self.n_heads = int(n_heads)
        self.use_step_emb = bool(use_step_emb)
        self.pre_norm = bool(pre_norm)

        # step tokens
        self.step_emb = nn.Embedding(self.horizon, self.d_model) if self.use_step_emb else None
        self.exo_proj = nn.Linear(self.exo_dim, self.d_model) if self.exo_dim > 0 else None

        # cross-attn (batch_first=True => [B, T, D])
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=float(dropout),
            batch_first=True,
        )

        # FFN block
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, 4 * self.d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(4 * self.d_model, self.d_model),
        )

        self.drop = nn.Dropout(float(dropout))
        self.ln_q = nn.LayerNorm(self.d_model)
        self.ln_kv = nn.LayerNorm(self.d_model)
        self.ln_out = nn.LayerNorm(self.d_model)

        # optional small scale to help avoid "tiny variation" collapse in early training
        self.out_scale = nn.Parameter(torch.tensor(20.0))
        self.exo_gain = nn.Parameter(torch.tensor(10.0))

    def forward(self, h_enc: torch.Tensor, future_exo: torch.Tensor | None = None) -> torch.Tensor:
        """
        h_enc     : [B, L, D]
        future_exo: [B, H, E] or None
        return    : [B, H, D]
        """
        B, L, D = h_enc.shape
        H = self.horizon
        assert D == self.d_model, f"h_enc last dim mismatch: {D} vs {self.d_model}"

        # --- Build per-step query tokens ---
        if self.step_emb is not None:
            step_idx = torch.arange(H, device=h_enc.device)
            q = self.step_emb(step_idx).unsqueeze(0).expand(B, H, D)  # [B,H,D]
        else:
            q = torch.zeros(B, H, D, device=h_enc.device, dtype=h_enc.dtype)

        if self.exo_proj is not None:
            if future_exo is None:
                exo = torch.zeros(B, H, self.exo_dim, device=h_enc.device, dtype=h_enc.dtype)
            else:
                assert future_exo.dim() == 3 and future_exo.size(1) == H, \
                    f"future_exo must be [B,H,E], got {tuple(future_exo.shape)}"
                exo = future_exo.to(device=h_enc.device, dtype=h_enc.dtype)
            q = q + self.exo_proj(exo) * self.exo_gain  # [B,H,D]

        # --- Cross attention ---
        if self.pre_norm:
            qn = self.ln_q(q)
            kvn = self.ln_kv(h_enc)
            attn_out, _ = self.cross_attn(query=qn, key=kvn, value=kvn, need_weights=False)
            y = q + self.drop(attn_out)  # residual
            y = y + self.drop(self.ffn(self.ln_out(y)))  # FFN residual
        else:
            attn_out, _ = self.cross_attn(query=q, key=h_enc, value=h_enc, need_weights=False)
            y = self.ln_out(q + self.drop(attn_out))
            y = y + self.drop(self.ffn(y))
            y = self.ln_out(y)

        return y * self.out_scale


class TitanDecoder(nn.Module):
    """
    Horizon-aware decoder.

    Key properties:
      - ALWAYS breaks symmetry across horizon via step embeddings.
      - Injects future_exo per-step via linear projection (no mean pooling).
    """
    def __init__(
        self,
        d_model: int,
        horizon: int,
        exo_dim: int = 0,
        dropout: float = 0.0,
        use_step_emb: bool = True,
        ctx_pool: str = "last",  # "last" | "mean"
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.horizon = int(horizon)
        self.exo_dim = int(exo_dim)
        self.use_step_emb = bool(use_step_emb)
        self.ctx_pool = str(ctx_pool)

        self.ctx_proj = nn.Linear(self.d_model, self.d_model)

        if self.use_step_emb:
            self.step_emb = nn.Embedding(self.horizon, self.d_model)
        else:
            self.step_emb = None

        self.exo_proj = nn.Linear(self.exo_dim, self.d_model) if self.exo_dim > 0 else None

        self.drop = nn.Dropout(float(dropout))

        # 작은 FFN으로 decoding capacity 부여 (필수는 아니지만 추천)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, 4 * self.d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(4 * self.d_model, self.d_model),
        )
        self.ln = nn.LayerNorm(self.d_model)

    def forward(self, h_enc: torch.Tensor, future_exo: torch.Tensor | None = None) -> torch.Tensor:
        """
        h_enc: [B, L, D]
        future_exo: [B, H, E] or None
        return: [B, H, D]
        """
        B, L, D = h_enc.shape
        H = self.horizon

        # 1) context pooling
        if self.ctx_pool == "mean":
            ctx = h_enc.mean(dim=1)          # [B, D]
        else:
            ctx = h_enc[:, -1, :]            # [B, D]

        base = self.ctx_proj(ctx).unsqueeze(1).expand(B, H, D)  # [B,H,D]

        # 2) horizon step embedding (symmetry breaker)
        if self.step_emb is not None:
            step_idx = torch.arange(H, device=h_enc.device)
            step = self.step_emb(step_idx).unsqueeze(0).expand(B, H, D)  # [B,H,D]
            y = base + step
        else:
            y = base

        # 3) per-step future exo injection
        if self.exo_proj is not None:
            if future_exo is None:
                # future_exo가 없으면 0으로 처리 (그래도 step_emb 때문에 H는 달라짐)
                exo = torch.zeros(B, H, self.exo_dim, device=h_enc.device, dtype=h_enc.dtype)
            else:
                # shape 체크
                assert future_exo.dim() == 3 and future_exo.size(1) == H, \
                    f"future_exo must be [B,H,E], got {tuple(future_exo.shape)}"
                exo = future_exo.to(device=h_enc.device, dtype=h_enc.dtype)

            y = y + self.exo_proj(exo)  # [B,H,D]

        # 4) lightweight decoder block
        y = self.drop(y)
        y = self.ln(y + self.ffn(y))
        return y

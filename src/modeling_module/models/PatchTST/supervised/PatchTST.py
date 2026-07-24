import torch
from torch import nn
import torch.nn.functional as F

from modeling_module.models.PatchTST.common import compute_patch_num
from modeling_module.models.PatchTST.common.configs import (
    PatchTSTConfig,
    validate_patchtst_future_categorical_config,
)
from modeling_module.models.PatchTST.heads.distribution_head import DistHeadWithExo
from modeling_module.models.PatchTST.heads.point_head import PointHeadWithExo
from modeling_module.models.PatchTST.heads.quantile_head import QuantileHeadWithExo
from modeling_module.models.PatchTST.supervised.backbone import SupervisedBackbone
from modeling_module.models.common_layers.RevIN import RevIN

from .future_categorical import FutureCategoricalEmbedding


def _validate_future_exo_contract(
    future_exo: torch.Tensor | None,
    *,
    batch_size: int,
    horizon: int,
    expected_dim: int,
) -> None:
    """Validate PatchTST's raw future-continuous input before any model work."""
    expected_dim = int(expected_dim)
    if future_exo is None:
        if expected_dim > 0:
            raise RuntimeError(
                f"[PatchTST] future_exo is required when configured future width={expected_dim}; "
                f"expected shape ({batch_size}, {horizon}, {expected_dim})."
            )
        return

    if not torch.is_tensor(future_exo):
        raise RuntimeError(
            f"[PatchTST] future_exo must be a tensor with rank-3 [B,H,E], got {type(future_exo).__name__}."
        )
    if expected_dim <= 0:
        if future_exo.numel() > 0:
            raise RuntimeError(
                "[PatchTST] future_exo is not accepted when configured future width=0; "
                f"got non-empty shape {tuple(future_exo.shape)}."
            )
        return
    if future_exo.dim() != 3:
        raise RuntimeError(
            f"[PatchTST] future_exo must be rank-3 [B,H,E], got shape {tuple(future_exo.shape)}."
        )

    actual_batch, actual_horizon, actual_dim = future_exo.shape
    if actual_batch != batch_size:
        raise RuntimeError(
            f"[PatchTST] future_exo batch mismatch: got {actual_batch}, expected {batch_size}."
        )
    if actual_horizon != horizon:
        raise RuntimeError(
            f"[PatchTST] future_exo horizon mismatch: got {actual_horizon}, expected {horizon}."
        )
    if actual_dim != expected_dim:
        raise RuntimeError(
            f"[PatchTST] future_exo last dimension mismatch: got {actual_dim}, expected {expected_dim}."
        )


def _bind_future_categorical_config(model: nn.Module, cfg: PatchTSTConfig) -> None:
    cardinalities, embedding_dim = validate_patchtst_future_categorical_config(
        cfg
    )
    model.future_exo_cat_cardinalities = cardinalities
    model.future_exo_cat_dim = len(cardinalities)
    model.future_exo_cat_embedding_dim = embedding_dim
    model.future_exo_token_dim = int(
        getattr(cfg, "future_exo_dim", getattr(cfg, "d_future", 0))
    ) + len(cardinalities) * embedding_dim
    model.future_cat_embedding = (
        FutureCategoricalEmbedding(
            cardinalities=cardinalities,
            embedding_dim=embedding_dim,
            horizon=int(cfg.horizon),
        )
        if cardinalities
        else None
    )


def _encode_future_categorical(
    model: nn.Module,
    future_exo_cat: torch.Tensor | None,
    *,
    batch_size: int | None = None,
) -> torch.Tensor | None:
    embedding = model.future_cat_embedding
    if embedding is None:
        if future_exo_cat is not None:
            raise RuntimeError(
                "[PatchTST] future_exo_cat is not accepted when no future "
                "categorical cardinalities are configured."
            )
        return None
    if future_exo_cat is None:
        raise RuntimeError(
            "[PatchTST] future_exo_cat is required when future categorical "
            f"width={model.future_exo_cat_dim}; expected shape "
            f"({batch_size or 'B'}, {embedding.horizon}, "
            f"{model.future_exo_cat_dim})."
        )
    return embedding(future_exo_cat, batch_size=batch_size)


def _build_future_exogenous_tokens(
    model: nn.Module,
    future_exo: torch.Tensor | None,
    future_exo_cat: torch.Tensor | None,
    *,
    batch_size: int,
) -> torch.Tensor | None:
    _validate_future_exo_contract(
        future_exo,
        batch_size=batch_size,
        horizon=model.horizon,
        expected_dim=model.d_future,
    )
    if (
        future_exo is not None
        and model.d_future > 0
        and not torch.is_floating_point(future_exo)
    ):
        raise TypeError(
            "[PatchTST] future_exo must use a floating dtype, "
            f"got {future_exo.dtype}."
        )

    future_cat_embedding = _encode_future_categorical(
        model,
        future_exo_cat,
        batch_size=batch_size,
    )
    token_parts: list[torch.Tensor] = []
    if model.d_future > 0:
        if future_exo is None:  # guarded by _validate_future_exo_contract
            raise RuntimeError("[PatchTST] future_exo validation did not resolve an input.")
        token_parts.append(future_exo)
    if future_cat_embedding is not None:
        token_parts.append(future_cat_embedding)

    if not token_parts:
        return None
    if len(token_parts) == 1:
        tokens = token_parts[0]
    else:
        devices = {part.device for part in token_parts}
        if len(devices) != 1:
            raise ValueError(
                "future_exo and future_exo_cat embeddings must share one "
                f"device, got {sorted(map(str, devices))}."
            )
        tokens = torch.cat(token_parts, dim=-1)

    if int(tokens.shape[-1]) != int(model.future_exo_token_dim):
        raise RuntimeError(
            "[PatchTST] combined future token width mismatch: "
            f"got {int(tokens.shape[-1])}, "
            f"expected {int(model.future_exo_token_dim)}."
        )
    return tokens


class FutureExoTokenFusion(nn.Module):
    """
    Token-wise future exogenous fusion for PatchTST.

    Instead of flattening the whole future horizon into a single vector,
    project each horizon step into a token and let backbone patch tokens
    attend to that future token sequence.
    """

    def __init__(
        self,
        *,
        d_model: int,
        d_future: int,
        horizon: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(d_future)
        self.d_future = self.input_dim
        self.horizon = int(horizon)
        self.future_proj = nn.Linear(self.input_dim, int(d_model))
        self.future_pos = nn.Parameter(torch.zeros(1, self.horizon, int(d_model)))
        nn.init.normal_(self.future_pos, mean=0.0, std=0.02)

        self.query_norm = nn.LayerNorm(int(d_model))
        self.memory_norm = nn.LayerNorm(int(d_model))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=int(d_model),
            num_heads=int(n_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.resid_norm = nn.LayerNorm(int(d_model))
        self.ff_norm = nn.LayerNorm(int(d_model))
        self.ff = nn.Sequential(
            nn.Linear(int(d_model), int(d_model) * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_model) * 4, int(d_model)),
        )

        # Start conservatively so exo does not immediately overwhelm the backbone.
        self.cross_gate = nn.Parameter(torch.tensor(-2.0))
        self.ff_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, z: torch.Tensor, future_exo: torch.Tensor | None) -> torch.Tensor:
        if self.input_dim <= 0 or future_exo is None:
            return z

        if future_exo.dim() == 2:
            future_exo = future_exo.unsqueeze(0).expand(z.size(0), -1, -1)

        if future_exo.dim() != 3:
            raise RuntimeError(
                f"[PatchTST-FutureFusion] future_exo must be (B,H,E) or (H,E), got {tuple(future_exo.shape)}"
            )

        b2, H, D = future_exo.shape
        if b2 != z.size(0):
            raise RuntimeError(f"[PatchTST-FutureFusion] batch mismatch: {b2} != {z.size(0)}")
        if H != self.horizon:
            raise RuntimeError(f"[PatchTST-FutureFusion] horizon mismatch: {H} != {self.horizon}")
        if D != self.input_dim:
            raise RuntimeError(
                "[PatchTST-FutureFusion] future token dim mismatch: "
                f"{D} != {self.input_dim}"
            )

        exo_tokens = self.future_proj(future_exo) + self.future_pos[:, :H, :]
        exo_tokens = self.memory_norm(self.dropout(exo_tokens))

        attn_out, _ = self.cross_attn(
            self.query_norm(z),
            exo_tokens,
            exo_tokens,
            need_weights=False,
        )
        z = self.resid_norm(z + torch.sigmoid(self.cross_gate) * self.dropout(attn_out))

        ff_out = self.ff(self.ff_norm(z))
        z = self.resid_norm(z + torch.sigmoid(self.ff_gate) * self.dropout(ff_out))
        return z


def _resolve_future_exo_fusion_dropout(cfg) -> float:
    raw = getattr(cfg, 'future_exo_fusion_dropout', None)
    if raw is None:
        raw = getattr(cfg, 'dropout', 0.1)
    return float(raw)


class PatchTSTModel(nn.Module):

    def _denorm_scale(self, scale: torch.Tensor) -> torch.Tensor:
        """RevIN denorm for scale (std-like). scale must be positive."""
        if not self.use_revin:
            return scale

        s = scale.unsqueeze(-1)  # (B, H, 1)

        # affine 역변환: std는 |w|로 나누는 편이 안전
        if getattr(self.revin_layer, "affine", False):
            w = self.revin_layer.affine_weight.view(1, 1, -1)
            s = s / (w.abs() + 1e-8)

        # std 역변환
        if getattr(self.revin_layer, "use_std", True):
            std = self.revin_layer.std  # (B, 1, C)
            s = s * std

        return s.squeeze(-1)  # (B, H)

    def __init__(self, cfg: PatchTSTConfig):
        super().__init__()
        self.model_name = 'PatchTSTModel'
        self.cfg = cfg
        _bind_future_categorical_config(self, cfg)

        self.attn_type = getattr(cfg.attn.attn_core, "type", "full").lower()
        self.lookback = int(getattr(cfg, 'lookback', 52))
        self.horizon = int(getattr(cfg, 'horizon', 27))
        self.d_model = int(getattr(cfg, 'd_model', 128))
        self.d_future = int(getattr(cfg, 'future_exo_dim', getattr(cfg, 'd_future', 0)))
        self.act = getattr(cfg, 'act', 'gelu')
        self.patch_len = int(getattr(cfg, 'patch_len', 8))
        self.stride = int(getattr(cfg, 'stride', self.patch_len // 2))
        self.padding_patch = getattr(cfg, 'padding_patch', None)
        self.is_quantile = False

        self.backbone: SupervisedBackbone = SupervisedBackbone(cfg, attn_type=self.attn_type)

        self.use_revin = bool(getattr(cfg, 'use_revin', True))
        self.revin_layer = RevIN(num_features=cfg.c_in, affine = False, subtract_last = True, use_std = True)

        self.future_fuser: FutureExoTokenFusion | None = None
        self._rebuild_future_exo_path(self.d_future)

        self.loss = cfg.loss
        self.loss_type = 'point' if not hasattr(self.loss, 'distribution') else 'distribution'

        if self.loss_type == 'point':
            self.param_names = None
            self.out_mult = 1
            self.head = PointHeadWithExo(
                d_model=self.d_model,
                horizon=self.horizon,
                d_future=self._head_future_dim(),
                patch_num=compute_patch_num(
                    lookback=self.lookback,
                    patch_len=self.patch_len,
                    stride=self.stride,
                    padding_patch=self.padding_patch
                )
            )
        elif self.loss_type == 'distribution':
            self.param_names = list(self.loss.param_names)  # 예: StudentT -> ["-df","-loc","-scale"]
            self.out_mult = int(self.loss.outputsize_multiplier)  # 예: StudentT -> 3, Normal -> 2
            self.head = DistHeadWithExo(
                d_model=self.d_model,
                horizon=self.horizon,
                d_future=self._head_future_dim(),
                act=self.act,
                out_mult=self.out_mult,
            )

        self.dist_min_scale = float(getattr(cfg, 'dist_min_scale', 1e-3))
        print(f'[PatchTST] dist_min_scale: {self.dist_min_scale}')
        self.out_mul = int(getattr(cfg, 'out_mul', 1))

    @classmethod
    def from_config(cls, config: "PatchTSTConfig"):
        return cls(cfg=config)

    def _head_future_dim(self) -> int:
        return 0

    def _rebuild_future_exo_path(self, d_future: int) -> None:
        self.d_future = int(d_future)
        fusion_input_dim = int(self.future_exo_token_dim)
        if fusion_input_dim > 0:
            self.future_fuser = FutureExoTokenFusion(
                d_model=self.d_model,
                d_future=fusion_input_dim,
                horizon=self.horizon,
                n_heads=int(getattr(self.cfg.attn, 'n_heads', 8)),
                dropout=_resolve_future_exo_fusion_dropout(self.cfg),
            )
        else:
            self.future_fuser = None

    def encode_future_categorical(
        self,
        future_exo_cat: torch.Tensor | None,
        *,
        batch_size: int | None = None,
    ) -> torch.Tensor | None:
        return _encode_future_categorical(
            self,
            future_exo_cat,
            batch_size=batch_size,
        )

    def build_future_exogenous_tokens(
        self,
        future_exo: torch.Tensor | None,
        future_exo_cat: torch.Tensor | None,
        *,
        batch_size: int,
    ) -> torch.Tensor | None:
        return _build_future_exogenous_tokens(
            self,
            future_exo,
            future_exo_cat,
            batch_size=batch_size,
        )

    def forward(
            self,
            x: torch.Tensor,
            future_exo: torch.Tensor | None = None,
            future_exo_cat: torch.Tensor | None = None,
            past_exo_cont: torch.Tensor | None = None,
            past_exo_cat: torch.Tensor | None = None,
            # part_ids = None,
            # mode: str | None = None,
            **kwargs
    ):
        future_tokens = self.build_future_exogenous_tokens(
            future_exo,
            future_exo_cat,
            batch_size=x.size(0),
        )

        # 1) 입력 정규화
        x_n = self.revin_layer(x, 'norm') if self.use_revin else x

        # 2) Backbone Encoding (Inject Past Exogenous)
        z = self.backbone(x_n, past_exo_cont=past_exo_cont, past_exo_cat=past_exo_cat)  # [B, N, d_model]

        # 2.5) Token-wise future exo fusion before the head
        if self.future_fuser is not None:
            z = self.future_fuser(z, future_tokens)

        # 3) Head Forecasting (Inject Future Exogenous)
        head_future_exo = future_exo if self._head_future_dim() > 0 else None
        head_out = self.head(z, future_exo=head_future_exo)  # [B, H]

        if self.loss_type == 'point':
            if self.use_revin:
                y = self.revin_layer(head_out.unsqueeze(-1), 'denorm').squeeze(-1)  # [B, H]
                return y
            return head_out

        if not torch.is_tensor(head_out) or head_out.dim() != 3 or head_out.size(-1) != self.out_mult:
            raise TypeError(
                f"[PatchTSTModel] head_out must be (B,H,{self.out_mult}), got {type(head_out)} {getattr(head_out, 'shape', None)}")

        params = {name: head_out[..., i] for i, name in enumerate(self.param_names)}

        loc_n = params.get('-loc')
        if loc_n is None:
            raise RuntimeError(f"[PatchTSTModel] '-loc' not found in param_names={self.param_names}")

        loc = self.revin_layer(loc_n.unsqueeze(-1), 'denorm').squeeze(-1) if self.use_revin else loc_n

        # ---- scale 처리 (기존 로직 유지: raw -> pos -> denorm -> raw) ----
        scale_raw_n = params.get('-scale')
        if scale_raw_n is None:
            raise RuntimeError(f"[PatchTSTModel] '-scale' not found in param_names={self.param_names}")

        scale_pos = F.softplus(scale_raw_n) + self.dist_min_scale
        scale_pos = self._denorm_scale(scale_pos) if self.use_revin else torch.clamp(scale_pos, min=self.dist_min_scale)

        # inverse-softplus (DistributionLoss가 다시 softplus를 타므로 raw로 되돌려서 반환)
        x = torch.clamp(scale_pos - self.dist_min_scale, min=1e-8)
        scale_raw_for_loss = x + torch.log(-torch.expm1(-x))
        outs = []
        for name in self.param_names:
            if name == '-loc':  # Normal, Poisson, StudentT
                outs.append(loc)
            elif name == '-scale':  # Normal, StudentT
                outs.append(scale_raw_for_loss)
            elif name == '-df':  # StudentT
                df_val = F.softplus(params.get('-df', None) + 2.0)
                outs.append(df_val)
            elif name == '-logits':  # Bernoulli, NegativeBinomial
                print(f"[PatchTSTModel] '-logits' not found in param_names={params}"
                      f"It seems 'Bernoulli' or 'NegativeBinomial' but not yet developed")
                pass
            elif name == '-total_count':  # NegativeBinomial
                print(f"[PatchTSTModel] '-logits' not found in param_names={params}"
                      f"It seems 'NegativeBinomial' not yet developed")
            elif name == '-log_mu':  # Tweedie
                pass
            else:
                pass
        return torch.stack(outs, dim=-1)


class PatchTSTQuantileModel(nn.Module):
    """
    PatchTST 기반 분위수 예측(Quantile Forecasting) 모델.

    구조:
    - RevIN: 정규화.
    - SupervisedBackbone: 특징 추출.
    - QuantileHeadWithExo: 분위수 회귀를 위한 다중 출력 헤드.
    """

    def __init__(self, cfg, attn_core=None):
        super().__init__()
        self.cfg = cfg
        _bind_future_categorical_config(self, cfg)
        # 백본 초기화
        self.attn_type = getattr(cfg.attn.attn_core, "type", "full").lower()
        self.backbone = SupervisedBackbone(cfg, self.attn_type)
        self.d_future = int(getattr(cfg, 'future_exo_dim', getattr(cfg, 'd_future', 0)))

        # 분위수 헤드 초기화
        self.head = QuantileHeadWithExo(
            d_model=cfg.d_model,
            horizon=cfg.horizon,
            d_future=self._head_future_dim(),
            quantiles=getattr(cfg, "quantiles", (0.1, 0.5, 0.9)),
            hidden=getattr(cfg, "q_hidden", 128),
            monotonic=getattr(cfg, "monotonic_quantiles", True),
        )

        self.is_quantile = True
        self.horizon = cfg.horizon
        self.model_name = "PatchTST QuantileModel"

        self.revin_layer = RevIN(num_features=cfg.c_in, affine = False, subtract_last = True, use_std = True)
        self.future_fuser: FutureExoTokenFusion | None = None
        self._rebuild_future_exo_path(self.d_future)

    @classmethod
    def from_config(cls, config: "PatchTSTConfig"):
        return cls(cfg=config)

    def _head_future_dim(self) -> int:
        return 0

    def _rebuild_future_exo_path(self, d_future: int) -> None:
        self.d_future = int(d_future)
        fusion_input_dim = int(self.future_exo_token_dim)
        if fusion_input_dim > 0:
            self.future_fuser = FutureExoTokenFusion(
                d_model=int(self.cfg.d_model),
                d_future=fusion_input_dim,
                horizon=int(self.cfg.horizon),
                n_heads=int(getattr(self.cfg.attn, 'n_heads', 8)),
                dropout=_resolve_future_exo_fusion_dropout(self.cfg),
            )
        else:
            self.future_fuser = None

    def encode_future_categorical(
        self,
        future_exo_cat: torch.Tensor | None,
        *,
        batch_size: int | None = None,
    ) -> torch.Tensor | None:
        return _encode_future_categorical(
            self,
            future_exo_cat,
            batch_size=batch_size,
        )

    def build_future_exogenous_tokens(
        self,
        future_exo: torch.Tensor | None,
        future_exo_cat: torch.Tensor | None,
        *,
        batch_size: int,
    ) -> torch.Tensor | None:
        return _build_future_exogenous_tokens(
            self,
            future_exo,
            future_exo_cat,
            batch_size=batch_size,
        )

    def forward(
            self,
            x: torch.Tensor,
            future_exo: torch.Tensor | None = None,
            future_exo_cat: torch.Tensor | None = None,
            past_exo_cont: torch.Tensor | None = None,
            past_exo_cat: torch.Tensor | None = None,
            part_ids=None,
            mode: str | None = None,
            **kwargs,
    ):
        """
        순전파 수행.
        Returns:
            {"q": [B, H, Quantiles]} 딕셔너리 반환.
        """
        future_tokens = self.build_future_exogenous_tokens(
            future_exo,
            future_exo_cat,
            batch_size=x.size(0),
        )
        use_revin = getattr(self.cfg, "use_revin", True)

        # 1) 입력 정규화
        x_n = self.revin_layer(x, "norm") if use_revin else x  # [B, L, C]

        # 2) 백본 인코딩
        z = self.backbone(x_n, past_exo_cont=past_exo_cont, past_exo_cat=past_exo_cat)  # [B, N, d_model]

        if self.future_fuser is not None:
            z = self.future_fuser(z, future_tokens)

        # 3) 헤드 예측 -> [B, H, Q]
        head_future_exo = future_exo if self._head_future_dim() > 0 else None
        q_n = self.head(z, future_exo=head_future_exo)

        # 4) 역정규화 (Denormalization)
        if use_revin:
            if q_n.dim() == 2:
                # [B,H] -> [B,H,1] 확장 후 역정규화
                q_den = self.revin_layer(q_n.unsqueeze(-1), "denorm").squeeze(-1)  # [B,H]
                return {"q": q_den}

            elif q_n.dim() == 3:
                # [B,H,Q] -> 평탄화 -> 역정규화 -> 구조 복원
                # (채널별 정규화 특성상 Q차원을 독립 채널로 보지 않고, 단일 변수 예측값의 분포로 간주)
                B, H, Q = q_n.shape
                q_flat = q_n.reshape(B, H * Q, 1)  # [B, H*Q, 1]
                q_den = self.revin_layer(q_flat, "denorm").reshape(B, H, Q)
                return {"q": q_den}

            else:
                raise RuntimeError(f"[PatchTSTQuantile] unexpected q_n.dim={q_n.dim()} shape={tuple(q_n.shape)}")

        return {"q": q_n}

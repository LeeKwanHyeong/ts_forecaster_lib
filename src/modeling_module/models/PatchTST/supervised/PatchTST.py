import torch
from torch import nn

from modeling_module.models.PatchTST.common import get_activation_fn
from modeling_module.models.PatchTST.common.configs import PatchTSTConfig
from modeling_module.models.PatchTST.supervised.backbone import SupervisedBackbone
from modeling_module.models.common_layers.RevIN import RevIN
from typing import Optional, Tuple
import torch.nn.functional as F

class PointHeadWithExo(nn.Module):
    """
    백본 출력과 미래 외생 변수(Future Exo)를 결합하여 단일 값 예측(Horizon)을 수행하는 헤드.

    기능:
    - 백본의 패치 임베딩을 집약(Mean or Last).
    - 미래 외생 변수를 투영(Projection) 후 잠재 벡터와 결합(Concat).
    - 최종 선형 레이어를 통해 시계열 예측값 산출.
    """

    def __init__(self, d_model: int, horizon: int, d_future: int, patch_num: int, agg: str = "mean"):
        super().__init__()
        self.agg = agg
        self.horizon = horizon
        self.d_future = d_future

        # 미래 외생 변수 투영 레이어: [B, H * d_future] -> [B, d_model]
        # 시계열 전체 문맥(Context)에 맞게 미래 정보를 압축
        self.future_proj = nn.Linear(horizon * d_future, d_model) if d_future > 0 else None

        # 최종 예측 레이어
        # 입력 차원: 백본 특징(d_model) + 미래 외생 특징(d_model, 존재 시)
        in_dim = d_model * 2 if d_future > 0 else d_model

        self.proj = nn.Linear(in_dim, horizon)

    def forward(self, z_bld: torch.Tensor, future_exo: torch.Tensor = None) -> torch.Tensor:
        """
        순전파 수행.
        Args:
            z_bld: 백본 출력 [B, Num_Patches, d_model]
            future_exo: 미래 외생 변수 [B, Horizon, d_future]
        Returns:
            예측값 [B, Horizon]
        """
        # 백본 출력 집약 (평균 또는 마지막 패치)
        if self.agg == "mean":
            feat = z_bld.mean(dim=1)
        else:
            feat = z_bld[:, -1, :]

        # 미래 외생 변수 결합 로직
        if self.d_future > 0:
            if future_exo is None:
                raise RuntimeError(
                    f"[PatchTST] d_future={self.d_future}인데 future_exo가 None입니다. "
                    f"Adapter/forward 시그니처 호환을 확인하세요."
                )
            B, H, D = future_exo.shape
            if D != self.d_future:
                raise RuntimeError(
                    f"[PatchTST] future_exo last-dim(D)={D} != d_future={self.d_future}"
                )

            # 미래 변수 평탄화 및 투영 후 결합
            f_flat = future_exo.reshape(B, -1)
            f_feat = self.future_proj(f_flat)
            feat = torch.cat([feat, f_feat], dim=-1)

        return self.proj(feat)


class QuantileHeadWithExo(nn.Module):
    """
    백본 출력과 미래 외생 변수를 결합하여 분위수(Quantile) 예측을 수행하는 헤드.

    기능:
    - 백본 특징과 투영된 미래 외생 특징 결합.
    - MLP를 통해 다중 분위수(예: 0.1, 0.5, 0.9) 동시 예측.
    - 분위수 교차(Quantile Crossing) 방지를 위한 정렬(Sort) 옵션 지원.

    출력: [B, H, Q] (Q는 분위수 개수)
    """

    def __init__(
            self,
            d_model: int,
            horizon: int,
            d_future: int,
            quantiles=(0.1, 0.5, 0.9),
            hidden: int = 128,
            monotonic: bool = True,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.quantiles = tuple(quantiles)
        self.Q = len(self.quantiles)
        self.monotonic = bool(monotonic)
        self.d_future = int(d_future)

        # 미래 외생 변수 투영
        self.future_proj = nn.Linear(self.horizon * self.d_future, d_model) if self.d_future > 0 else None
        in_dim = d_model * 2 if self.d_future > 0 else d_model

        # 분위수 예측 MLP
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.horizon * self.Q),
        )

    def forward(self, z_bld: torch.Tensor, future_exo: torch.Tensor = None) -> torch.Tensor:
        """
        순전파 수행.
        Returns:
            분위수 예측값 [B, Horizon, Quantiles]
        """
        B = z_bld.size(0)
        feat = z_bld.mean(dim=1)  # [B, d_model] - 백본 출력 평균 집약

        # 미래 외생 변수 처리
        if self.d_future > 0:
            if future_exo is None:
                raise RuntimeError(
                    f"[PatchTST-Quantile] d_future={self.d_future}인데 future_exo가 None입니다."
                )
            if future_exo.dim() == 2:  # (H,E) -> (B,H,E) 브로드캐스팅 지원
                future_exo = future_exo.unsqueeze(0).expand(B, -1, -1)

            if future_exo.dim() != 3:
                raise RuntimeError(f"[PatchTST-Quantile] future_exo must be 3D, got {tuple(future_exo.shape)}")

            b2, H, D = future_exo.shape
            if b2 != B:
                raise RuntimeError(f"[PatchTST-Quantile] future_exo batch mismatch: {b2} != {B}")
            if H != self.horizon:
                raise RuntimeError(f"[PatchTST-Quantile] future_exo horizon mismatch: {H} != {self.horizon}")
            if D != self.d_future:
                raise RuntimeError(
                    f"[PatchTST-Quantile] future_exo last-dim(D)={D} != d_future={self.d_future}"
                )

            # 미래 변수 결합
            f_flat = future_exo.reshape(B, -1)  # [B, H*D]
            f_feat = self.future_proj(f_flat)  # [B, d_model]
            feat = torch.cat([feat, f_feat], dim=-1)  # [B, 2*d_model]

        # 예측 수행 및 차원 변환
        out = self.net(feat).view(B, self.horizon, self.Q)  # [B, H, Q]

        # 분위수 단조성(Monotonicity) 보장
        if self.monotonic:
            out, _ = torch.sort(out, dim=-1)
        return out


class PatchTSTPointModel(nn.Module):
    """
    PatchTST 기반 점 예측(Point Forecasting) 모델.

    구조:
    - RevIN: 입력 데이터 정규화.
    - SupervisedBackbone: 패치 임베딩 및 트랜스포머 인코딩 (과거 외생 변수 포함).
    - PointHeadWithExo: 미래 외생 변수를 반영한 최종 예측.
    """

    def __init__(self, cfg, attn_core=None):
        super().__init__()
        self.cfg = cfg

        # 1. 백본 초기화 (과거 외생 변수 처리 포함)
        self.backbone = SupervisedBackbone(cfg, attn_core)

        # 2. 패치 개수 계산 (Head 초기화용)
        from modeling_module.models.PatchTST.common.patching import compute_patch_num
        patch_num = compute_patch_num(cfg.lookback, cfg.patch_len, cfg.stride, cfg.padding_patch)

        # 3. 예측 헤드 초기화 (미래 외생 변수 처리 포함)
        self.head = PointHeadWithExo(
            d_model=cfg.d_model,
            horizon=cfg.horizon,
            d_future=getattr(cfg, 'd_future', 0),
            patch_num=patch_num,
            agg="mean"  # 기본적으로 평균 풀링 사용
        )

        self.is_quantile = False
        self.horizon = cfg.horizon
        self.model_name = "PatchTST ExoModel"

        # 4. 정규화 모듈 (타겟 변수 전용)
        self.revin_layer = RevIN(num_features=cfg.c_in)

    @classmethod
    def from_config(cls, config: "PatchTSTConfig"):
        return cls(cfg=config)

    def forward(
            self,
            x: torch.Tensor,
            # 신규 인터페이스 (Trainer/Adapter 호환용)
            future_exo: torch.Tensor | None = None,
            past_exo_cont: torch.Tensor | None = None,
            past_exo_cat: torch.Tensor | None = None,
            part_ids=None,
            mode: str | None = None,
            # 레거시 인터페이스 (하위 호환성)
            fe_cont: torch.Tensor | None = None,
            pe_cont: torch.Tensor | None = None,
            pe_cat: torch.Tensor | None = None,
            **kwargs,
    ):
        """
        순전파 수행.
        신규/구형 인자명을 모두 수용하여 모델의 유연성 확보.
        """

        # 0) 인자명 통일 (Alias Resolution)
        fe = future_exo if future_exo is not None else fe_cont
        p_cont = past_exo_cont if past_exo_cont is not None else pe_cont
        p_cat = past_exo_cat if past_exo_cat is not None else pe_cat

        # 1) 입력 정규화 (RevIN)
        use_revin = getattr(self.cfg, "use_revin", True)
        x_n = self.revin_layer(x, "norm") if use_revin else x

        # 2) 백본 인코딩 (과거 외생 변수 주입)
        z = self.backbone(x_n, p_cont=p_cont, p_cat=p_cat)  # [B, N, d_model]

        # 3) 헤드 예측 (미래 외생 변수 주입)
        y_n = self.head(z, future_exo=fe)  # [B, H]

        # 4) 역정규화 (Denormalization)
        if use_revin:
            y = self.revin_layer(y_n.unsqueeze(-1), "denorm").squeeze(-1)  # [B,H]
            return y

        return y_n


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
        # 백본 초기화
        self.backbone = SupervisedBackbone(cfg, attn_core)

        # 분위수 헤드 초기화
        self.head = QuantileHeadWithExo(
            d_model=cfg.d_model,
            horizon=cfg.horizon,
            d_future=getattr(cfg, "d_future", 0),
            quantiles=getattr(cfg, "quantiles", (0.1, 0.5, 0.9)),
            hidden=getattr(cfg, "q_hidden", 128),
            monotonic=getattr(cfg, "monotonic_quantiles", True),
        )

        self.is_quantile = True
        self.horizon = cfg.horizon
        self.model_name = "PatchTST QuantileModel"

        self.revin_layer = RevIN(num_features=cfg.c_in)

    @classmethod
    def from_config(cls, config: "PatchTSTConfig"):
        return cls(cfg=config)

    def forward(
            self,
            x: torch.Tensor,
            future_exo: torch.Tensor | None = None,
            past_exo_cont: torch.Tensor | None = None,
            past_exo_cat: torch.Tensor | None = None,
            part_ids=None,
            mode: str | None = None,
            fe_cont: torch.Tensor | None = None,
            pe_cont: torch.Tensor | None = None,
            pe_cat: torch.Tensor | None = None,
            **kwargs,
    ):
        """
        순전파 수행.
        Returns:
            {"q": [B, H, Quantiles]} 딕셔너리 반환.
        """
        use_revin = getattr(self.cfg, "use_revin", True)

        # 인자명 통일
        fe = future_exo if future_exo is not None else fe_cont
        p_cont = past_exo_cont if past_exo_cont is not None else pe_cont
        p_cat = past_exo_cat if past_exo_cat is not None else pe_cat

        # 1) 입력 정규화
        x_n = self.revin_layer(x, "norm") if use_revin else x  # [B, L, C]

        # 2) 백본 인코딩
        z = self.backbone(x_n, p_cont=p_cont, p_cat=p_cat)  # [B, N, d_model]

        # 3) 헤드 예측 -> [B, H, Q]
        q_n = self.head(z, future_exo=fe)

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

# =========================================================
# Distribution Head / Model (Gaussian: loc + scale)
# =========================================================

# class DistHeadWithExo(nn.Module):
#     """(y, exo) -> (loc, scale_raw)
#
#     - backbone 출력 h: (B, N, D) 또는 (B, D) 형태를 모두 지원 (현재 구현은 (B, N, D) 가정).
#     - exo_future: (B, H, E) (옵션)
#     - 반환:
#         loc: (B, H)
#         scale_raw: (B, H)  # 양수 제약은 LossComputer에서 처리(softplus/exp + min_scale)
#     """
#
#     def __init__(self, d_model: int, horizon: int, *, d_future: int = 0, act: str = "gelu"):
#         super().__init__()
#         self.horizon = int(horizon)
#         self.d_future = int(d_future)
#
#         # (B, N, D) -> (B, D)
#         self.pool = nn.AdaptiveAvgPool1d(1)
#
#         if self.d_future > 0:
#             self.exo_proj = nn.Linear(self.d_future, d_model)
#             self.fuse = nn.Sequential(
#                 nn.Linear(d_model, d_model),
#                 get_activation_fn(act),
#             )
#         else:
#             self.exo_proj = None
#             self.fuse = None
#
#         self.loc_head = nn.Linear(d_model, self.horizon)
#         self.scale_head = nn.Linear(d_model, self.horizon)
#
#     def forward(self,
#                 h: torch.Tensor,
#                 future_exo: Optional[torch.Tensor] = None,
#                 ) -> Tuple[torch.Tensor, torch.Tensor]:
#
#         # h: (B, N, D) -> (B, D)
#         if h.dim() == 3:
#             B, N, D = h.shape
#             h_pool = self.pool(h.transpose(1, 2)).squeeze(-1)  # (B, D)
#         elif h.dim() == 2:
#             h_pool = h
#         else:
#             raise ValueError(f"DistHeadWithExo expects 2D/3D tensor, got {tuple(h.shape)}")
#
#         if (self.d_future > 0) and (future_exo is not None):
#             # future_exo: (B, H, E) -> (B, H, D) -> 평균 풀링 -> (B, D)
#             exo_h = self.exo_proj(future_exo).mean(dim=1)
#             h_pool = self.fuse(h_pool + exo_h)
#
#         loc = self.loc_head(h_pool)         # (B, H)
#         scale_raw = self.scale_head(h_pool) # (B, H)
#         return loc, scale_raw
class DistHeadWithExo(nn.Module):
    """
    backbone 출력 (B, N_patch, d_model)을 pool해서 (B, d_model)로 만든 뒤,
    (B, horizon, out_mult)를 출력.
    - out_mult: Normal=2 (loc, scale_raw), StudentT=3 (df, loc, scale_raw)
    """
    def __init__(
        self,
        d_model: int,
        horizon: int,
        d_future: int = 0,
        act: str = "gelu",
        out_mult: int = 2,
        hidden: int = 128,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.out_mult = int(out_mult)
        self.d_future = int(d_future)

        # 미래 외생: (B, H, E) -> (B, H*E) -> (B, d_model)
        self.future_proj = nn.Linear(self.horizon * self.d_future, d_model) if self.d_future > 0 else None
        in_dim = d_model * 2 if self.d_future > 0 else d_model

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            get_activation_fn(act),
            nn.Linear(hidden, self.horizon * self.out_mult),
        )

    def forward(self, h: torch.Tensor, *, future_exo: Optional[torch.Tensor] = None) -> torch.Tensor:
        # h: (B, N_patch, d_model)  -> pool -> (B, d_model)
        if h.dim() != 3:
            raise ValueError(f"[DistHeadWithExo] expected (B,N,D), got {tuple(h.shape)}")
        feat = h.mean(dim=1)  # (B, d_model)

        # future_exo 결합
        if self.d_future > 0:
            if future_exo is None:
                raise ValueError("[DistHeadWithExo] future_exo is required when d_future>0")
            if future_exo.dim() == 2:  # (H,E) -> (B,H,E)
                future_exo = future_exo.unsqueeze(0).expand(h.size(0), -1, -1)
            B, H, E = future_exo.shape
            if H != self.horizon or E != self.d_future:
                raise ValueError(f"[DistHeadWithExo] future_exo shape mismatch: {tuple(future_exo.shape)}")
            f_flat = future_exo.reshape(B, -1)
            f_feat = self.future_proj(f_flat)
            feat = torch.cat([feat, f_feat], dim=-1)  # (B, 2*d_model)

        out = self.net(feat).view(h.size(0), self.horizon, self.out_mult)  # (B, H, out_mult)
        return out


class PatchTSTDistModel(nn.Module):
    """PatchTST Gaussian distribution head (loc + scale).

    - forward 반환 pred는 dict:
        {"loc": loc, "scale": scale_raw}
      (scale 양수화/최소값 적용은 LossComputer(dist)에서 수행)
    - RevIN 사용 시:
        loc는 기존 point 모델과 동일하게 denorm.
        scale은 mean/last는 무시하고 std 및 affine만 역변환하여 denorm.
    """

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

    def __init__(self, cfg: PatchTSTConfig, *, min_scale: float = 1e-3):
        super().__init__()
        self.cfg = cfg
        self.horizon = int(cfg.horizon)
        self.min_scale = float(min_scale)

        # ----------------------------
        # 1) loss 스펙을 "그대로" 사용 (핵심)
        # ----------------------------
        loss = cfg.loss
        if not hasattr(loss, "param_names") or not hasattr(loss, "outputsize_multiplier"):
            raise TypeError(f"[PatchTSTDistModel] cfg.loss must be DistributionLoss-like, got={type(loss)}")

        self.param_names = list(loss.param_names)                  # 예: StudentT -> ["-df","-loc","-scale"]
        self.out_mult = int(loss.outputsize_multiplier)            # 예: StudentT -> 3, Normal -> 2

        # (옵션) 방어: StudentT인데 3이 아니면 즉시 실패
        if getattr(loss, "distribution", None) == "StudentT" and self.out_mult != 3:
            raise RuntimeError(f"[PatchTSTDistModel] StudentT requires out_mult=3, got {self.out_mult}")

        # Backbone / RevIN
        self.backbone = SupervisedBackbone(cfg)
        self.use_revin = bool(cfg.use_revin)
        self.revin_layer = RevIN(num_features=cfg.c_in)

        # ----------------------------
        # 2) Head: 반드시 out_mult를 주입
        # ----------------------------
        self.head = DistHeadWithExo(
            d_model=cfg.d_model,
            horizon=self.horizon,
            d_future=getattr(cfg, "d_future", 0),
            act=getattr(cfg, "act", "gelu"),
            out_mult=self.out_mult,
        )

        # ----------------------------
        # 3) 즉시 검증 (지금 상태를 잡아내는 가장 빠른 방법)
        # ----------------------------
        if getattr(self.head, "out_mult", None) != self.out_mult:
            raise RuntimeError(
                f"[PatchTSTDistModel] head.out_mult({getattr(self.head,'out_mult',None)}) != model.out_mult({self.out_mult}). "
                f"Head is not configured for the requested distribution."
            )

        # Linear(out_features)까지 확인 가능한 구조면 더 강하게 체크
        if hasattr(self.head, "proj") and hasattr(self.head.proj, "out_features"):
            if self.head.proj.out_features != self.out_mult:
                raise RuntimeError(
                    f"[PatchTSTDistModel] head.proj.out_features({self.head.proj.out_features}) != out_mult({self.out_mult})"
                )


    def forward(
        self,
        x: torch.Tensor,
        # 신규 인터페이스 (Trainer/Adapter 호환)
        future_exo: torch.Tensor | None = None,
        past_exo_cont: torch.Tensor | None = None,
        past_exo_cat: torch.Tensor | None = None,
        part_ids=None,
        mode: str | None = None,
        # 레거시 인터페이스 (하위 호환)
        fe_cont: torch.Tensor | None = None,
        pe_cont: torch.Tensor | None = None,
        pe_cat: torch.Tensor | None = None,
        **kwargs,
    ):
        # 0) alias 통일
        fe = future_exo if future_exo is not None else fe_cont
        p_cont = past_exo_cont if past_exo_cont is not None else pe_cont
        p_cat  = past_exo_cat  if past_exo_cat  is not None else pe_cat

        # 1) RevIN norm
        x_n = self.revin_layer(x, "norm") if self.use_revin else x

        # 2) Backbone에 과거 외생 주입 (중요)
        h = self.backbone(x_n, p_cont=p_cont, p_cat=p_cat)

        # 3) Head에 미래 외생 주입
        head_out = self.head(h, future_exo=fe)

        if not torch.is_tensor(head_out) or head_out.dim() != 3 or head_out.size(-1) != self.out_mult:
            raise TypeError(
                f"[PatchTSTDistModel] head_out must be (B,H,{self.out_mult}), got {type(head_out)} {getattr(head_out, 'shape', None)}")

            # param_names 순서 그대로 분해
            # e.g. Normal: ["-loc","-scale"]
            #      StudentT: ["-df","-loc","-scale"]
        params = {name: head_out[..., i] for i, name in enumerate(self.param_names)}

        # ---- loc 처리 (기존과 동일) ----
        loc_n = params.get("-loc")
        if loc_n is None:
            raise RuntimeError(f"[PatchTSTDistModel] '-loc' not found in param_names={self.param_names}")

        if self.use_revin:
            loc = self.revin_layer(loc_n.unsqueeze(-1), "denorm").squeeze(-1)
        else:
            loc = loc_n

        # ---- scale 처리 (기존 로직 유지: raw -> pos -> denorm -> raw) ----
        scale_raw_n = params.get("-scale")
        if scale_raw_n is None:
            raise RuntimeError(f"[PatchTSTDistModel] '-scale' not found in param_names={self.param_names}")

        scale_pos = F.softplus(scale_raw_n) + self.min_scale
        if self.use_revin:
            scale_pos = self._denorm_scale(scale_pos)
        scale_pos = torch.clamp(scale_pos, min=self.min_scale)

        # inverse-softplus (DistributionLoss가 다시 softplus 하므로 raw로 되돌려서 반환)
        x = torch.clamp(scale_pos - self.min_scale, min=1e-8)
        scale_raw_for_loss = x + torch.log(-torch.expm1(-x))

        # ---- df (StudentT 전용) ----
        df_raw_n = params.get("-df", None)  # Normal이면 None

        # 최종 반환: param_names 순서 유지가 핵심
        outs = []
        for name in self.param_names:
            if name == "-loc":
                outs.append(loc)
            elif name == "-scale":
                outs.append(scale_raw_for_loss)
            elif name == "-df":
                # [수정된 부분]
                # df는 RevIN denorm 대상은 아니지만, 양수 제약조건이 필수입니다.
                # Linear 출력은 음수가 나올 수 있으므로 Softplus로 감싸야 합니다.
                # + 2.0을 해주는 이유는 df <= 2 일 때 분산이 무한대가 되는 것을 막아 학습 안정성을 높이기 위함입니다.
                df_val = F.softplus(df_raw_n) + 2.0
                outs.append(df_val)
            else:
                raise RuntimeError(f"[PatchTSTDistModel] unsupported param name: {name}")

        return torch.stack(outs, dim=-1)


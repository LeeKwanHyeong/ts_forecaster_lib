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
    """
    텐서의 마지막 차원을 목표 차원(target_dim)에 맞춰 조정(Slice 또는 Padding).

    특징:
    - Optimizer 안전성 유지를 위해 학습 파라미터 없는(Parameter-free) 연산 수행.
    - 입력이 목표보다 크면 자르고(Slice), 작으면 상수(pad_value)로 우측 패딩.
    """
    if x is None:
        return x
    if target_dim <= 0:
        # 차원이 0 이하인 경우 비활성화로 간주하여 None 반환
        return None
    if x.size(-1) == target_dim:
        return x
    if x.size(-1) > target_dim:
        # 목표 차원보다 큰 경우 슬라이싱
        return x[..., :target_dim]

    # 목표 차원보다 작은 경우 패딩 생성 및 결합
    pad = target_dim - x.size(-1)
    pad_shape = list(x.shape[:-1]) + [pad]
    pad_t = x.new_full(pad_shape, pad_value)
    return torch.cat([x, pad_t], dim=-1)


def _infer_patch_cfgs(lookback: int, n_branches: int = 3) -> List[Tuple[int, int, int]]:
    """
    Lookback 길이에 비례하여 결정론적(Deterministic)인 멀티스케일 패치 설정 생성.

    기능:
    - 입력 길이의 1/4, 1/2, 3/4 비율을 기반으로 패치 길이(Patch Len) 계산.
    - 각 패치 길이에 적합한 Stride(P//2)와 Kernel Size(3, 5, 7) 자동 매핑.
    - 반환 형식: List[(Patch_Len, Stride, Kernel_Size)]
    """
    # 최소 Lookback 길이 검증
    assert lookback >= 8

    # 비율에 따른 패치 길이 후보군 생성
    fracs = [1 / 4, 1 / 2, 3 / 4][:n_branches]
    raw = [max(4, min(lookback, int(round(lookback * f)))) for f in fracs]

    # 중복 제거 및 정렬
    P = sorted(list(dict.fromkeys(raw)))

    cfgs: List[Tuple[int, int, int]] = []
    for i, p in enumerate(P):
        s = max(1, p // 2)  # Stride는 패치 길이의 절반
        k = [3, 5, 7][min(i, 2)]  # 스케일별 커널 크기 차등 할당
        if k % 2 == 0:
            k += 1  # 홀수 커널 보장
        cfgs.append((p, s, k))
    return cfgs


# =====================================================================
# Core mixin: past/future exo handling (parameter-safe)
# =====================================================================
class _ExoMixin(nn.Module):
    """
    모델에 과거(Past) 및 미래(Future) 외생 변수 처리 기능을 부여하는 믹스인(Mixin) 클래스.

    기능:
    - Future Exo: 선형 변환을 통해 예측값(Forecast)에 가산적 편향(Shift) 적용.
    - Past Exo: 연속형/범주형 변수를 임베딩 및 풀링(Pooling)하여 잠재 벡터(Latent z)에 주입(Z-Gate).
    """

    def _init_exo(self, cfg: PatchMixerConfig, *, z_dim: int):
        """
        외생 변수 처리를 위한 모듈 및 차원 초기화.
        """
        # 1. 미래 외생 변수 (Future Exo) 설정
        self.exo_dim = int(getattr(cfg, "exo_dim", 0) or 0)
        self.exo_is_normalized_default = bool(getattr(cfg, "exo_is_normalized_default", False))
        self.exo_head: Optional[nn.Module] = None

        # 미래 외생 변수가 존재할 경우, 이를 예측값 보정에 사용할 헤드 생성 (MLP)
        if self.exo_dim > 0:
            self.exo_head = nn.Sequential(
                nn.Linear(self.exo_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

        # 2. 과거 외생 변수 (Past Exo) 설정
        self.past_exo_mode = str(getattr(cfg, "past_exo_mode", "none") or "none").lower()
        if self.past_exo_mode not in ("none", "z_gate"):
            raise ValueError(f"Unsupported past_exo_mode={self.past_exo_mode}. Use 'none' or 'z_gate'.")

        self.past_exo_cont_dim = int(getattr(cfg, "past_exo_cont_dim", 0) or 0)
        self.past_exo_cat_dim = int(getattr(cfg, "past_exo_cat_dim", 0) or 0)

        # 범주형 변수 설정 (Vocab Size, Embed Dim) 검증
        vocab_sizes = tuple(getattr(cfg, "past_exo_cat_vocab_sizes", ()) or ())
        embed_dims = tuple(getattr(cfg, "past_exo_cat_embed_dims", ()) or ())

        if self.past_exo_cat_dim > 0:
            if len(vocab_sizes) != self.past_exo_cat_dim or len(embed_dims) != self.past_exo_cat_dim:
                raise ValueError(
                    "past_exo_cat_dim>0 requires past_exo_cat_vocab_sizes and past_exo_cat_embed_dims with same length"
                )

        # 범주형 임베딩 레이어 생성
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

        # 3. Z-Gate 프로젝션 설정 (Past Exo 주입용)
        self._z_exo_proj: Optional[nn.Linear] = None
        self._z_gate: Optional[nn.Linear] = None

        if self.past_exo_mode == "z_gate":
            in_dim = self.past_exo_cont_dim + self._cat_embed_total
            # 입력 차원이 있을 경우에만 게이트 모듈 생성
            if in_dim > 0:
                self._z_exo_proj = nn.Linear(in_dim, z_dim, bias=True)  # 외생 정보를 z차원으로 투영
                self._z_gate = nn.Linear(z_dim, z_dim, bias=True)  # z벡터로부터 게이트 가중치 산출

    def _pool_past_exo(self, past_exo_cont: Optional[torch.Tensor], past_exo_cat: Optional[torch.Tensor]) -> Optional[
        torch.Tensor]:
        """
        과거 외생 변수들을 시간 축 기준으로 평균(Pooling)내어 하나의 벡터로 병합.

        반환:
            Pooled Vector (Batch, Total_Exo_Dim)
        """
        feats: List[torch.Tensor] = []

        # 연속형 변수 처리: 차원 조정 및 평균
        if past_exo_cont is not None and past_exo_cont.numel() > 0 and self.past_exo_cont_dim > 0:
            pe = _pad_or_slice_last_dim(past_exo_cont.float(), self.past_exo_cont_dim, pad_value=0.0)
            feats.append(pe.mean(dim=1))  # (B, E_c)

        # 범주형 변수 처리: 임베딩 조회 후 평균
        if past_exo_cat is not None and past_exo_cat.numel() > 0 and self.past_exo_cat_dim > 0:
            # 정수형 변환 및 차원 조정
            pc = _pad_or_slice_last_dim(past_exo_cat.long(), self.past_exo_cat_dim, pad_value=0)
            assert self._cat_embs is not None

            emb_list: List[torch.Tensor] = []
            for j, emb in enumerate(self._cat_embs):
                ids = pc[..., j]
                # Vocab 범위 내로 안전하게 클램핑
                ids = ids.clamp_min(0).clamp_max(emb.num_embeddings - 1)
                e = emb(ids)  # (B, L, Embed_Dim)
                emb_list.append(e.mean(dim=1))  # 시간 축 평균
            feats.append(torch.cat(emb_list, dim=-1))

        if not feats:
            return None
        return torch.cat(feats, dim=-1)

    def _inject_past_exo_z_gate(self, z: torch.Tensor, past_exo_cont: Optional[torch.Tensor],
                                past_exo_cat: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Z-Gate 메커니즘을 사용하여 잠재 벡터 z에 과거 외생 변수 정보를 주입.

        Logic:
            Out = z + Sigmoid(Gate(z)) * Projection(Exo)
        """
        if self.past_exo_mode != "z_gate":
            return z
        if self._z_exo_proj is None or self._z_gate is None:
            return z  # 주입할 모듈이 없으면 통과

        # 외생 변수 풀링
        v = self._pool_past_exo(past_exo_cont, past_exo_cat)
        if v is None:
            return z

        # 차원 안전장치 적용
        v = _pad_or_slice_last_dim(v, self._z_exo_proj.in_features, pad_value=0.0)

        # 정보 주입
        exo_z = self._z_exo_proj(v)  # 외생 정보를 잠재 공간으로 변환
        gate = torch.sigmoid(self._z_gate(z))  # z 상태에 따른 게이트 값(0~1) 계산
        return z + gate * exo_z  # 잔차 연결 방식으로 정보 합산

    def _apply_future_exo_shift(self, y: torch.Tensor, future_exo: Optional[torch.Tensor], *,
                                exo_is_normalized: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        미래 외생 변수(Future Exo)를 사용하여 예측값(y)을 보정(Shift).

        반환:
            (보정된 y, 계산된 Shift값)
        """
        if future_exo is None or self.exo_head is None or self.exo_dim <= 0:
            return y, None

        # 차원 조정
        fe = _pad_or_slice_last_dim(future_exo.float(), self.exo_dim, pad_value=0.0)

        # 보정값(Shift) 계산
        # apply_exo_shift_linear 함수가 외부에 정의되어 있다고 가정
        ex = apply_exo_shift_linear(
            self.exo_head,
            fe,
            horizon=int(getattr(self, "horizon")),
            out_dtype=y.dtype,
            out_device=y.device,
        )

        # 정규화된 공간에서의 연산이 허용된 경우 보정 적용
        if exo_is_normalized:
            y = y + ex

        return y, ex


# =====================================================================
# Point model
# =====================================================================
class PatchMixerPointModel(_ExoMixin):
    """
    PatchMixer 기반의 점 추정(Point Forecasting) 모델.

    구조:
    1. RevIN: 입력 데이터 정규화 (Distribution Shift 완화).
    2. Backbone: 패치 단위 믹싱을 통해 시계열의 잠재 특징(Latent z) 추출.
    3. Exogenous/Embedding Injection: 과거 외생 변수 및 ID 임베딩 정보 주입.
    4. Expander: 정적 잠재 벡터를 미래 예측 기간(Horizon)으로 시간적 확장.
    5. Head: 최종 예측값 산출.
    6. Refinement: 스케일 보정 및 잔차 학습을 통한 출력 안정화.
    """

    def __init__(self, cfg: PatchMixerConfig):
        super().__init__()
        self.configs = cfg
        self.horizon = int(cfg.horizon)
        self.f_out = int(getattr(cfg, "f_out", 128))
        self.final_nonneg = bool(getattr(cfg, "final_nonneg", True))

        # 1. 백본 네트워크 초기화 및 잠재 벡터(z) 차원 설정
        self.backbone = PatchMixerBackbone(configs=cfg)
        # 백본의 출력 차원 감지 (설정값 혹은 계산된 차원)
        z_dim = int(getattr(self.backbone, "out_dim", getattr(self.backbone, "patch_repr_dim", 0)))
        if z_dim <= 0:
            raise RuntimeError("Backbone must expose out_dim or patch_repr_dim")
        self.z_dim = z_dim

        # 2. RevIN (Reversible Instance Normalization) 모듈 설정
        self.use_revin = bool(getattr(cfg, "use_revin", True))
        self.revin = RevIN(int(getattr(cfg, "enc_in", 1)))

        # 3. 파트(ID) 임베딩 설정 (선택 사항)
        self.use_part_embedding = bool(getattr(cfg, "use_part_embedding", False))
        self.part_emb: Optional[nn.Embedding] = None
        self.z_fuser: Optional[nn.Linear] = None

        if self.use_part_embedding and int(getattr(cfg, "part_vocab_size", 0)) > 0:
            pdim = int(getattr(cfg, "part_embed_dim", 16))
            self.part_emb = nn.Embedding(int(cfg.part_vocab_size), pdim)
            # 잠재 벡터(z)와 임베딩을 결합(Concat) 후 원래 차원으로 압축하는 레이어
            self.z_fuser = nn.Linear(z_dim + pdim, z_dim)

        # 4. Temporal Expander (시간적 확장 모듈)
        # 백본의 정적 출력(Vector)을 시계열(Sequence) 형태로 변환
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

        # 5. 예측 헤드 (Prediction Head)
        # 확장된 특징을 최종 예측값으로 변환 (MLP 구조)
        head_hidden = int(getattr(cfg, "head_hidden", self.f_out))
        self.pre_ln = nn.LayerNorm(self.f_out)
        self.head = nn.Sequential(
            nn.Linear(self.f_out, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )

        # 6. 출력 스케일 안정화 모듈 (Scale Stabilizers)
        # 학습 초반 불안정성을 완화하고 스케일을 보정하는 파라미터
        self.learn_output_scale = bool(getattr(cfg, "learn_output_scale", True))
        if self.learn_output_scale:
            self.out_scale = nn.Parameter(torch.tensor(1.0))
            self.out_bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("out_scale", torch.tensor(1.0))
            self.register_buffer("out_bias", torch.tensor(0.0))

        # Depthwise Conv를 이용한 지역적 평활화(Smoothing) 및 잔차 보정
        self.learn_dw_gain = bool(getattr(cfg, "learn_dw_gain", True))
        self.dw_head = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        if self.learn_dw_gain:
            self.dw_gain = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("dw_gain", torch.tensor(1.0))

        # 7. 외생 변수 처리 믹스인 초기화
        # (반드시 z_dim 확정 후 호출 필요)
        self._init_exo(cfg, z_dim=z_dim)

    def forward(
            self,
            x: torch.Tensor,  # (Batch, Lookback, Channels)
            future_exo: Optional[torch.Tensor] = None,  # (Batch, Horizon, Exo_Dim)
            *,
            past_exo_cont: Optional[torch.Tensor] = None,  # (Batch, Lookback, Past_Exo_Dim)
            past_exo_cat: Optional[torch.Tensor] = None,  # (Batch, Lookback, Past_Cat_Dim)
            part_ids: Optional[torch.Tensor] = None,  # (Batch,) - ID 정보
            exo_is_normalized: Optional[bool] = None,
            **kwargs,
    ) -> torch.Tensor:
        """
        순전파 과정 수행.
        """
        # 외생 변수 정규화 여부 설정 (기본값 또는 입력값 사용)
        if exo_is_normalized is None:
            exo_is_normalized = self.exo_is_normalized_default

        # 1) 전처리 및 특징 추출
        # RevIN 정규화 (입력 분포 안정화)
        x_in = self.revin(x, "norm") if self.use_revin else x
        # 백본 통과 -> 잠재 벡터 z 생성
        z = self.backbone(x_in)

        # 백본 출력 형태 검증
        if z.dim() != 2 or z.size(-1) != self.z_dim:
            raise RuntimeError(f"Unexpected backbone output shape: {tuple(z.shape)} expected (*, {self.z_dim})")

        # 2) 과거 외생 변수 주입
        # Z-Gate 방식을 통해 z 벡터에 과거 정보 융합
        z = self._inject_past_exo_z_gate(z, past_exo_cont, past_exo_cat)

        # 3) 파트(ID) 정보 주입
        # ID 임베딩 조회 및 z 벡터와 결합(Fusion)
        if self.part_emb is not None and part_ids is not None:
            pe = self.part_emb(part_ids.long())
            z = self.z_fuser(torch.cat([z, pe], dim=-1))

        # 4) 시간적 확장 및 예측 생성
        # z (B, dim) -> Expander -> (B, Horizon, f_out)
        # LayerNorm -> MLP Head -> (B, Horizon)
        f = self.pre_ln(self.expander(z))
        y = self.head(f).squeeze(-1)

        # 5) 미래 외생 변수 보정
        # Future Exo를 기반으로 예측값에 선형 편향(Shift) 적용
        y, ex = self._apply_future_exo_shift(y, future_exo, exo_is_normalized=exo_is_normalized)

        # 6) 후처리 및 미세 조정 (Refinement)
        # 글로벌 스케일/편향 적용
        y = y * self.out_scale + self.out_bias
        # Conv1d를 이용한 지역적 패턴 보정 (Residual)
        y = y + self.dw_gain * self.dw_head(y.unsqueeze(1)).squeeze(1)

        # 7) 역정규화 (Denormalization)
        if self.use_revin:
            y = self.revin(y.unsqueeze(-1), "denorm").squeeze(-1)

        # 정규화되지 않은 외생 변수 보정이 필요한 경우, 역정규화 후 적용
        if (ex is not None) and (not exo_is_normalized):
            y = y + ex

        # 8) 출력 제약 조건 적용
        # 추론 시 음수 예측 방지 (Clamp)
        if self.final_nonneg:
            y = torch.clamp_min(y, 0.0)

        return y


# =====================================================================
# Quantile model
# =====================================================================
class PatchMixerQuantileModel(_ExoMixin):
    """
    PatchMixer 기반의 분위수 예측(Probabilistic Forecasting) 모델.

    특징:
    - Multi-scale Backbone: 다양한 패치 크기를 결합하여 강건한 특징 추출.
    - Quantile Head: 여러 분위수(Quantiles)를 동시에 예측하여 불확실성 구간 제공.
    - Decomposition: 추세(Trend)와 계절성(Seasonality)을 분해하여 예측 성능 향상.
    """

    def __init__(self, cfg: PatchMixerConfig):
        super().__init__()
        self.is_quantile = True
        self.configs = cfg
        self.horizon = int(cfg.horizon)
        self.f_out = int(getattr(cfg, "f_out", 128))
        self.final_nonneg = bool(getattr(cfg, "final_nonneg", True))

        # 1. 멀티스케일 백본(Multi-scale Backbone) 설정
        # 패치 설정(patch_cfgs)이 없으면 Lookback 길이에 맞춰 자동 추론
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

        # 2. RevIN (Reversible Instance Normalization) 설정
        self.use_revin = bool(getattr(cfg, "use_revin", True))
        self.revin = RevIN(int(getattr(cfg, "enc_in", 1)))

        # 3. 파트(ID) 임베딩 설정
        self.use_part_embedding = bool(getattr(cfg, "use_part_embedding", False))
        self.part_emb: Optional[nn.Embedding] = None
        self.z_fuser: Optional[nn.Linear] = None
        if self.use_part_embedding and int(getattr(cfg, "part_vocab_size", 0)) > 0:
            pdim = int(getattr(cfg, "part_embed_dim", 16))
            self.part_emb = nn.Embedding(int(cfg.part_vocab_size), pdim)
            self.z_fuser = nn.Linear(self.z_dim + pdim, self.z_dim)

        # 4. Temporal Expander (시간적 확장)
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

        # 5. 분위수 헤드 (Quantile Head)
        # DecompositionQuantileHead를 사용하여 시계열 분해 기반 분위수 예측 수행
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

        # 6. 외생 변수 처리 초기화
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
        """
        순전파 수행. 다중 분위수(Quantiles)에 대한 예측값 반환.
        """
        if exo_is_normalized is None:
            exo_is_normalized = self.exo_is_normalized_default

        # 1) 전처리 및 백본 통과
        x_in = self.revin(x, "norm") if self.use_revin else x
        z = self.backbone(x_in)

        # 차원 검증
        if z.dim() != 2 or z.size(-1) != self.z_dim:
            raise RuntimeError(f"Unexpected backbone output shape: {tuple(z.shape)} expected (*, {self.z_dim})")

        # 2) 과거 정보 주입 (Exo & Part ID)
        z = self._inject_past_exo_z_gate(z, past_exo_cont, past_exo_cat)

        if self.part_emb is not None and part_ids is not None:
            pe = self.part_emb(part_ids.long())
            z = self.z_fuser(torch.cat([z, pe], dim=-1))

        # 3) 확장 및 분위수 예측
        f = self.expander(z)  # (B, H, F)
        q = self.head(f)  # (B, Q, H) 또는 (B, H, Q) - 헤드 설정에 따라 상이

        if self.use_revin:
            clip = float(getattr(self.configs, "q_clip_norm", 15.0))  # 10~20 권장
            # hard clamp보다 tanh clip이 학습 안정적
            q = clip * torch.tanh(q / clip)

        # 4) 미래 외생 변수 보정 (Shift)
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
            # 정규화된 공간에서 보정 적용 시 브로드캐스팅 처리
            if exo_is_normalized:
                if q.dim() == 3 and q.shape[1] != self.horizon:
                    # Shape: (B, Q, H) 인 경우, H축(2번)에 더하기 위해 Unsqueeze(1)
                    q = q + ex.unsqueeze(1)
                else:
                    # Shape: (B, H, Q) 인 경우, Q축(2번)에 더하기 위해 Unsqueeze(-1)
                    q = q + ex.unsqueeze(-1)

        # 5) 역정규화 (Denormalization)
        if q.dim() != 3:
            raise RuntimeError(f"Unexpected quantile tensor rank: {q.dim()}")

        Q = len(getattr(self.configs, "quantiles", (0.1, 0.5, 0.9)))

        # (B,Q,H)로 통일
        if q.shape[1] == self.horizon and q.shape[2] == Q:  # (B,H,Q)
            q = q.transpose(1, 2).contiguous()  # -> (B,Q,H)
        elif q.shape[1] == Q and q.shape[2] == self.horizon:  # (B,Q,H)
            q = q.contiguous()
        else:
            raise RuntimeError(f"Unexpected q shape: {tuple(q.shape)}")

        # 각 분위수 채널별로 RevIN 역변환 적용
        qs: List[torch.Tensor] = []
        for i in range(q.size(1)):
            qi = q[:, i, :]
            qi = self.revin(qi.unsqueeze(-1), "denorm").squeeze(-1) if self.use_revin else qi
            qs.append(qi.unsqueeze(1))
        q_raw = torch.cat(qs, dim=1)  # (B, Q, H)

        # 6) 정규화되지 않은 외생 변수 보정
        if (ex is not None) and (not exo_is_normalized):
            q_raw = q_raw + ex.unsqueeze(1)

        # 7) 출력 제약 (Non-negative)
        # if self.final_nonneg:
        #     q_raw = torch.clamp_min(q_raw, 0.0)

        return {"q": q_raw}
# ---------------------------------------------------------------------
# Backward-compatible aliases (if your builders import BaseModel/QuantileModel)
# ---------------------------------------------------------------------

BaseModel = PatchMixerPointModel
QuantileModel = PatchMixerQuantileModel



# ============================================================
# Distribution Model (Normal/StudentT/etc.)
# ============================================================
class PatchMixerDistributionModel(_ExoMixin):
    """Distribution forecasting for PatchMixer.

    Outputs a packed tensor of shape [B, H, out_mult].
    Example:
      - Normal:   out_mult=2  -> [loc, scale_raw]
      - StudentT: out_mult=3  -> [df_raw, loc, scale_raw] (or permuted via param_names)

    NOTE:
      - Positivity transforms (scale/df) are applied in LossComputer (DistributionLoss branch),
        so the model can output raw unconstrained values for non-loc parameters.
    """

    def __init__(self, cfg: PatchMixerConfig):
        super().__init__()
        self.config = cfg
        self.horizon = int(cfg.horizon)
        self.f_out = int(getattr(cfg, "f_out", 128))
        self.final_nonneg = bool(getattr(cfg, "final_nonneg", True))

        self.loss = getattr(cfg, "loss", None)

        print(f"[PatchMixer] loss = {self.loss}")
        print(f"[PatchMiixer] loss.distribution = {self.loss.distribution}")
        print(f"[PatchMixer] param_names = {getattr(self.loss, 'param_names', 'no')}")
        self.param_names = list(getattr(self.loss, "param_names", [])) if self.loss is not None else []
        self.out_mult = int(getattr(self.loss, "outputsize_multiplier", 2)) if self.loss is not None else 2

        # locate loc index (fallback 0)
        self.loc_idx = 0
        for i, n in enumerate(self.param_names):
            if str(n).lstrip("-") == "loc":
                self.loc_idx = i
                break

        print(f"[PatchMixerDist] loss={type(self.loss).__name__} param_names={self.param_names} out_mult={self.out_mult}")

        self.backbone = MultiScalePatchMixerBackbone(configs=cfg) if getattr(cfg, "use_multiscale", False) else PatchMixerBackbone(configs=cfg)
        z_dim = int(getattr(self.backbone, "out_dim", getattr(self.backbone, "patch_repr_dim", 0)))
        if z_dim <= 0:
            raise RuntimeError("Backbone must expose out_dim or patch_repr_dim")
        self.z_dim = z_dim

        self.use_revin = bool(getattr(cfg, "use_revin", True))
        self.revin = RevIN(int(getattr(cfg, "enc_in", 1)))

        self.use_part_embedding = bool(getattr(cfg, "use_part_embedding", False))
        self.part_emb = None
        self.z_fuser = None
        if self.use_part_embedding and int(getattr(cfg, "part_vocab_size", 0)) > 0:
            p_dim = int(getattr(cfg, "part_embed_dim", 16))
            self.part_emb = nn.Embedding(int(cfg.part_vocab_size), p_dim)
            self.z_fuser = nn.Linear(z_dim + p_dim, z_dim)

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

        head_hidden = int(getattr(cfg, "head_hidden", self.f_out))
        self.pre_ln = nn.LayerNorm(self.f_out)
        self.head = nn.Sequential(
            nn.Linear(self.f_out, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, self.out_mult),
        )

        # loc stabilization
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

        self._init_exo(cfg, z_dim=z_dim)

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
        z = self._inject_past_exo_z_gate(z, past_exo_cont, past_exo_cat)

        if self.part_emb is not None and part_ids is not None:
            pe = self.part_emb(part_ids.long())
            z = self.z_fuser(torch.cat([z, pe], dim=-1))

        f = self.pre_ln(self.expander(z))      # (B,H,F)
        out = self.head(f)                     # (B,H,out_mult)

        loc = out[..., self.loc_idx:self.loc_idx + 1]  # (B,H,1)

        # future exogenous shift -> loc only
        ex = None
        if future_exo is not None and self.exo_head is not None and self.exo_dim > 0:
            fe = _pad_or_slice_last_dim(future_exo.float(), self.exo_dim, pad_value=0.0)
            ex = apply_exo_shift_linear(
                self.exo_head, fe, horizon=self.horizon,
                out_dtype=loc.dtype, out_device=loc.device,
            )
            if exo_is_normalized:
                loc = loc + ex.unsqueeze(-1)

        loc = loc * self.out_scale + self.out_bias
        loc = loc + self.dw_gain * self.dw_head(loc.transpose(1, 2)).transpose(1, 2)

        if self.use_revin:
            loc = self.revin(loc, "denorm")

        if (ex is not None) and (not exo_is_normalized):
            loc = loc + ex.unsqueeze(-1)

        if self.final_nonneg:
            loc = torch.clamp_min(loc, 0.0)

        out = out.clone()
        out[..., self.loc_idx:self.loc_idx + 1] = loc
        return out

# alias for builders
DistModel = PatchMixerDistributionModel

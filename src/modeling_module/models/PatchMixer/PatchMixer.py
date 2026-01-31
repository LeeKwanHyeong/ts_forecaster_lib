from __future__ import annotations

import math
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
import torch.nn.functional as F
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
# Quantile model
# =====================================================================
def _to_BQH(q: torch.Tensor, *, horizon: int, Q: int) -> torch.Tensor:
    """
    head 출력이 (B,Q,H) 또는 (B,H,Q)로 올 수 있으므로 (B,Q,H)로 통일.
    """
    if q.dim() != 3:
        raise RuntimeError(f"Unexpected q rank: {q.dim()}")

    if q.shape[1] == Q and q.shape[2] == horizon:          # (B,Q,H)
        return q.contiguous()
    if q.shape[1] == horizon and q.shape[2] == Q:          # (B,H,Q)
        return q.transpose(1, 2).contiguous()              # -> (B,Q,H)

    raise RuntimeError(f"Unexpected q shape: {tuple(q.shape)} (expect (B,Q,H) or (B,H,Q))")


def _pad_or_trim_H(ex: torch.Tensor, *, horizon: int) -> torch.Tensor:
    """
    ex: (B,H) 형태, H를 horizon에 맞춤.
    """
    if ex.dim() != 2:
        raise RuntimeError(f"ex must be (B,H). got {tuple(ex.shape)}")

    B, Hx = ex.shape
    if Hx == horizon:
        return ex
    if Hx < horizon:
        pad = ex.new_zeros((B, horizon - Hx))
        return torch.cat([ex, pad], dim=1)
    return ex[:, :horizon]


def _init_softplus_inv(x: float) -> float:
    """
    softplus(a)=x 가 되도록 하는 a의 초기값 (x>0)
    """
    # softplus(a)=log(1+exp(a))=x => exp(a)=exp(x)-1 => a=log(exp(x)-1)
    return float(math.log(math.exp(float(x)) - 1.0))


def _zero_init_linear(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _try_zero_init_decomp_head(head: nn.Module) -> None:
    """
    DecompositionQuantileHead 내부 구조를 모르므로,
    실무에서 흔히 쓰는 이름들을 '있으면' 0-init 하는 방어적 초기화.
    (없으면 조용히 스킵)
    """
    # 1) head.core.* 계열
    core = getattr(head, "core", None)
    if core is not None:
        for name in ["trend_head", "irreg_head", "delta_head"]:
            m = getattr(core, name, None)
            if isinstance(m, nn.Linear):
                _zero_init_linear(m)

        # season_time_head가 Sequential인 케이스: 마지막 Linear만 0-init
        sth = getattr(core, "season_time_head", None)
        if isinstance(sth, nn.Sequential) and len(sth) > 0:
            for layer in reversed(sth):
                if isinstance(layer, nn.Linear):
                    _zero_init_linear(layer)
                    break

    # 2) 혹시 head에 직접 붙어있는 Linear가 있으면 전부 0-init(과격할 수 있어 옵션으로만 사용 권장)
    # 필요 시 주석 해제
    # for m in head.modules():
    #     if isinstance(m, nn.Linear):
    #         _zero_init_linear(m)


def apply_exo_shift_linear_trainable(
    head: nn.Module,
    future_exo: torch.Tensor,   # (B,H,E) or (H,E)
    *,
    horizon: int,
    out_dtype=None,
    out_device=None
) -> torch.Tensor:
    """
    Returns:
        ex: (B, H)
    """
    ex = future_exo
    if ex.dim() == 2:  # (H,E) -> (1,H,E)
        ex = ex.unsqueeze(0)

    if out_device is None:
        out_device = ex.device
    if out_dtype is None:
        out_dtype = ex.dtype

    ex = ex.to(device=out_device, dtype=out_dtype, non_blocking=True)

    # IMPORTANT:
    # - do NOT call head.to(device) here (forward 내에서 .to는 지양)
    # - model/head는 외부에서 이미 device로 올려져 있어야 합니다.
    ex = head(ex).squeeze(-1)  # (B,H)

    # pad/trim to horizon
    B, Hx = ex.shape
    if Hx < horizon:
        pad = torch.zeros((B, horizon - Hx), device=ex.device, dtype=ex.dtype)
        ex = torch.cat([ex, pad], dim=1)
    elif Hx > horizon:
        ex = ex[:, :horizon]

    return ex


class PatchMixerQuantileModel(_ExoMixin):
    """
    PatchMixer 기반의 분위수 예측 모델 (Quantile).
    - backbone -> z
    - (past exo gate) -> z
    - expander -> f (B,H,F)
    - head -> q_pre (B,Q,H) or (B,H,Q)
    - (optional clip policy)
    - RevIN denorm (per-quantile)
    - (future exo shift) -> out-space add (B,Q,H)
    """

    def __init__(self, cfg: PatchMixerConfig):
        super().__init__()
        self.is_quantile = True
        self.configs = cfg
        self.horizon = int(cfg.horizon)
        self.f_out = int(getattr(cfg, "f_out", 128))

        # backbone
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
        self.revin_layer = RevIN(num_features=int(getattr(cfg, "enc_in", 1)), affine = False, subtract_last = True, use_std = True)

        # (optional) part embedding
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

        # head
        head_hidden = int(getattr(cfg, "head_hidden", 128))
        self.quantiles = list(getattr(cfg, "quantiles", (0.1, 0.5, 0.9)))
        self.Q = len(self.quantiles)

        self.head = DecompositionQuantileHead(
            in_features=self.f_out,
            quantiles=self.quantiles,
            hidden=head_hidden,
            dropout=float(getattr(cfg, "head_dropout", 0.0) or 0.0),
            mid=0.5,
            use_trend=False,
            fourier_k=int(getattr(cfg, "expander_n_harmonics", getattr(cfg, "expander_n_harmonics", 8))),
            agg="mean",
        )

        # ---- 안정화(권장): z/f LayerNorm ----
        self.use_z_ln = bool(getattr(cfg, "use_z_ln", True))
        self.use_f_ln = bool(getattr(cfg, "use_f_ln", True))
        self.z_ln = nn.LayerNorm(self.z_dim) if self.use_z_ln else nn.Identity()
        self.f_ln = nn.LayerNorm(self.f_out) if self.use_f_ln else nn.Identity()

        # ---- clip policy ----
        # eval 시에만 tanh clip (학습 중 포화로 gradient 0 방지)
        self.q_clip_eval = float(getattr(cfg, "q_clip_norm", 10.0))
        # 학습 중에는 필요하면 "hard clamp"로만 안전장치(포화 tanh보다 덜 치명적)
        # None이면 clamp 없음
        self.q_clip_train = getattr(cfg, "q_clip_train", None)
        if self.q_clip_train is not None:
            self.q_clip_train = float(self.q_clip_train)

        # ---- future exo shift scaling (exo가 backbone을 압도하는 것 방지용) ----
        self.exo_scale = float(getattr(cfg, "exo_scale", 1.0))

        # exo init
        self._init_exo(cfg, z_dim=self.z_dim)

    def _to_bqh(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: (B,Q,H) or (B,H,Q) -> (B,Q,H)
        """
        if q.dim() != 3:
            raise RuntimeError(f"Unexpected q rank: {q.dim()}")

        if q.shape[1] == self.Q and q.shape[2] == self.horizon:
            return q.contiguous()  # (B,Q,H)
        if q.shape[1] == self.horizon and q.shape[2] == self.Q:
            return q.transpose(1, 2).contiguous()  # (B,Q,H)

        raise RuntimeError(f"Unexpected q shape: {tuple(q.shape)} (expect (B,Q,H) or (B,H,Q))")

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
        # NOTE:
        # 본 구현은 future exo shift를 "denorm 이후(out-space)"에 더하는 방식으로 통일합니다.
        # 따라서 exo_is_normalized는 사실상 사용하지 않습니다(호환용으로만 받음).

        # 1) norm + backbone
        x_in = self.revin_layer(x, "norm") if self.use_revin else x
        z = self.backbone(x_in)
        z = self._inject_past_exo_z_gate(z, past_exo_cont, past_exo_cat)
        if self.part_emb is not None and part_ids is not None:
            pe = self.part_emb(part_ids.long())
            z = self.z_fuser(torch.cat([z, pe], dim=-1))
        z = self.z_ln(z)

        # 2) expander + head
        f = self.expander(z)  # (B,H,F)
        f = self.f_ln(f)
        q_pre = self.head(f)  # (B,Q,H) or (B,H,Q)

        q = self._to_bqh(q_pre)  # 먼저 (B,Q,H)로 통일

        # 3) last-anchor (normalized-space)
        base_last = x_in[:, -1, 0]  # (B,)
        q = q + base_last[:, None, None]

        # 4) clip policy
        if self.training:
            # 추천: 학습 중엔 clip 하지 않기
            pass
        else:
            c = self.q_clip_eval
            if (c is not None) and (c > 0):
                q = c * torch.tanh(q / c)

        # 5) denorm (per-quantile)
        if self.use_revin:
            qs = []
            for i in range(q.size(1)):
                qi = q[:, i, :]  # (B,H)
                qi = self.revin_layer(qi.unsqueeze(-1), "denorm").squeeze(-1)
                qs.append(qi.unsqueeze(1))
            q_raw = torch.cat(qs, dim=1)  # (B,Q,H)
        else:
            q_raw = q

        # 6) future exo shift (out-space add)
        #   - apply_exo_shift_linear_trainable를 사용 (grad flow OK)
        if (future_exo is not None) and (self.exo_head is not None) and (self.exo_dim > 0):
            fe = _pad_or_slice_last_dim(future_exo.float(), self.exo_dim, pad_value=0.0)
            ex = apply_exo_shift_linear_trainable(
                self.exo_head,
                fe,
                horizon=self.horizon,
                out_dtype=q_raw.dtype,
                out_device=q_raw.device,
            )  # (B,H)
            q_raw = q_raw + (self.exo_scale * ex).unsqueeze(1)  # (B,Q,H)

        return {"q": q_raw}

class PatchMixerModel(_ExoMixin):
    """
    Unified PatchMixer model for both point and distribution forecasting.

    Output:
      - out_mult == 1 : (B, H) point forecast
      - out_mult  > 1 : (B, H, out_mult) packed distribution params

    Convention (default):
      - Normal:   out_mult=2  -> [loc, scale_raw]
      - StudentT: out_mult=3  -> [df_raw, loc, scale_raw]
    If `param_names` is given, 'loc' index is detected from it.
    """

    def __init__(
        self,
        cfg: PatchMixerConfig,
    ):
        super().__init__()
        self.configs = cfg
        self.horizon = int(cfg.horizon)
        self.f_out = int(getattr(cfg, "f_out", 128))
        self.final_nonneg = bool(getattr(cfg, "final_nonneg", True))
        self.out_mul = int(getattr(cfg, "out_mul", 1))
        print(f'[PatchMixerModel] out_mul:: {self.out_mul}')
        if self.out_mul > 1:
            self.param_names = list(getattr(cfg, "param_names", None))
            print(f'[PatchMixerModel] param_names:: {self.param_names}')
            # ---- dist param safety clip (NaN 방지; 특히 AMP에서 expm1 터지는 케이스 방지) ----
            # LossComputer에서 softplus/transform을 적용할 때 raw가 너무 크면 overflow로 NaN이 납니다.
            self.dist_param_clip = float(getattr(cfg, "dist_param_clip", 15.0))  # 권장: 10~20

            self.loc_idx = 0
            self.scale_idx = None
            self.df_idx = None
            for i, n in enumerate(self.param_names):
                nnm = str(n).lstrip("-")
                if nnm == "loc":
                    self.loc_idx = i
                elif nnm == "scale":
                    self.scale_idx = i
                elif nnm == "df":
                    self.df_idx = i

            # DistributionLoss transform settings (must match LossComputer)
            self.dist_scale_transform = str(getattr(cfg, "dist_scale_transform", "softplus"))
            self.dist_eps = float(getattr(cfg, "dist_eps", 1e-8))
            self.dist_min_scale = float(getattr(cfg, "dist_min_scale", 0.0))
        # self.out_mult = int(out_mult)
        # self.param_names = list(param_names) if param_names is not None else None

        # 1) backbone
        self.backbone = PatchMixerBackbone(configs=cfg)
        z_dim = int(getattr(self.backbone, "out_dim", getattr(self.backbone, "patch_repr_dim", 0)))
        if z_dim <= 0:
            raise RuntimeError("Backbone must expose out_dim or patch_repr_dim")
        self.z_dim = z_dim

        # 2) RevIN
        self.use_revin = bool(getattr(cfg, "use_revin", True))
        self.revin_layer = RevIN(num_features=int(getattr(cfg, "enc_in", 1)), affine = False, subtract_last = True, use_std = True)

        # 3) part embedding (optional)
        self.use_part_embedding = bool(getattr(cfg, "use_part_embedding", False))
        self.part_emb: Optional[nn.Embedding] = None
        self.z_fuser: Optional[nn.Linear] = None

        if self.use_part_embedding and int(getattr(cfg, "part_vocab_size", 0)) > 0:
            pdim = int(getattr(cfg, "part_embed_dim", 16))
            self.part_emb = nn.Embedding(int(cfg.part_vocab_size), pdim)
            self.z_fuser = nn.Linear(z_dim + pdim, z_dim)

        # 4) Temporal Expander
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

        # 5) head (unified)
        head_hidden = int(getattr(cfg, "head_hidden", self.f_out))
        self.pre_ln = nn.LayerNorm(self.f_out)
        self.head = nn.Sequential(
            nn.Linear(self.f_out, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, self.out_mul),
        )

        # 6) point-mode stabilizers (kept for compatibility; only used when out_mult==1)
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

        # 7) exogenous mixin init
        self._init_exo(cfg, z_dim=z_dim)

        # ---- 안정화(권장): z/f LayerNorm ----
        self.use_z_ln = bool(getattr(cfg, "use_z_ln", True))
        self.use_f_ln = bool(getattr(cfg, "use_f_ln", True))
        self.z_ln = nn.LayerNorm(self.z_dim) if self.use_z_ln else nn.Identity()
        self.f_ln = nn.LayerNorm(self.f_out) if self.use_f_ln else nn.Identity()

        self.exo_scale = float(getattr(cfg, "exo_scale", 1.0))



    def _loc_index(self) -> int:
        if self.param_names is not None and "loc" in self.param_names:
            return int(self.param_names.index("loc"))
        # default convention
        if self.out_mul == 3:
            return 1  # [df_raw, loc, scale_raw]
        return 0      # [loc, scale_raw] or others

    def _clip_nonloc_params_(self, out: torch.Tensor) -> torch.Tensor:
        """
        out: (B,H,out_mult)
        loc 제외 나머지 raw 파라미터를 clip 하여 overflow/NaN 위험을 줄입니다.
        """
        if self.out_mul <= 1:
            return out
        loc_i = self._loc_index()
        if self.dist_param_clip is None or self.dist_param_clip <= 0:
            return out

        # clone to avoid in-place on autograd graph unexpectedly
        o = out
        # non-loc indices
        for j in range(self.out_mul):
            if j == loc_i:
                continue
            o[..., j] = o[..., j].clamp(min=-self.dist_param_clip, max=self.dist_param_clip)
        return o

    # -----------------------------
    # stable inverse transforms
    # -----------------------------
    @staticmethod
    def _inv_softplus_stable(y: torch.Tensor) -> torch.Tensor:
        """Stable inverse of softplus for y>0."""
        y = y.clamp_min(1e-12)
        thr = 20.0
        small = torch.log(torch.expm1(y))
        large = y + torch.log1p(-torch.exp(-y))
        return torch.where(y > thr, large, small)

    def _scale_pos_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        t = self.dist_scale_transform.lower()
        eps = self.dist_eps
        if t == "softplus":
            return F.softplus(raw) + eps
        if t == "exp":
            return torch.exp(raw) + eps
        if t == "relu":
            return F.relu(raw) + eps
        if t == "abs":
            return raw.abs() + eps
        if t == "square":
            return raw.square() + eps
        return F.softplus(raw) + eps

    def _inv_scale_transform(self, scale_pos_minus_eps: torch.Tensor) -> torch.Tensor:
        t = self.dist_scale_transform.lower()
        x = scale_pos_minus_eps.clamp_min(1e-12)
        if t == "softplus":
            return self._inv_softplus_stable(x)
        if t == "exp":
            return torch.log(x)
        if t in ("relu", "abs"):
            return x
        if t == "square":
            return torch.sqrt(x)
        return self._inv_softplus_stable(x)
    def forward(
        self,
        x: torch.Tensor,
        future_exo: Optional[torch.Tensor] = None,
        *,
        past_exo_cont: Optional[torch.Tensor] = None,
        past_exo_cat: Optional[torch.Tensor] = None,
        part_ids: Optional[torch.Tensor] = None,
        exo_is_normalized: Optional[bool] = None,  # 호환용
        **kwargs,
    ):
        # 1) norm + backbone
        x_in = self.revin_layer(x, "norm") if (self.use_revin and self.revin_layer is not None) else x
        z = self.backbone(x_in)
        if z.dim() != 2 or z.size(-1) != self.z_dim:
            raise RuntimeError(f"Unexpected backbone output shape: {tuple(z.shape)} expected (*, {self.z_dim})")

        # 2) past exo gate (+ part embedding)
        z = self._inject_past_exo_z_gate(z, past_exo_cont, past_exo_cat)

        if self.part_emb is not None and part_ids is not None:
            pe = self.part_emb(part_ids.long())
            z = self.z_fuser(torch.cat([z, pe], dim=-1))

        z = self.z_ln(z)

        # 3) expander + head
        f = self.expander(z)               # (B,H,F)
        f = self.f_ln(f)
        f = self.pre_ln(f)
        out = self.head(f)                 # (B,H,out_mult)
        # ---- Point mode ----
        if self.out_mul == 1:
            y_pre = out.squeeze(-1)  # (B,H)

            # 레벨 앵커링 (normalized-space)
            base_last = x_in[:, -1, 0]          # (B,)
            y = y_pre + base_last[:, None]      # (B,H)

            # eval clip (optional)
            if not self.training:
                c = getattr(self, "y_clip_eval", None)
                if (c is not None) and (c > 0):
                    y = c * torch.tanh(y / c)

            # denorm
            if self.use_revin:
                y_raw = self.revin_layer(y.unsqueeze(-1), "denorm").squeeze(-1)  # (B,H)
            else:
                y_raw = y

            # future exo shift (out-space add)
            if (future_exo is not None) and (self.exo_head is not None) and (self.exo_dim > 0):
                fe = _pad_or_slice_last_dim(future_exo.float(), self.exo_dim, pad_value=0.0)
                ex = apply_exo_shift_linear_trainable(
                    self.exo_head,
                    fe,
                    horizon=self.horizon,
                    out_dtype=y_raw.dtype,
                    out_device=y_raw.device,
                )  # (B,H)
                y_raw = y_raw + (self.exo_scale * ex)

            # nonneg policy (optional)
            if self.final_nonneg:
                y_raw = F.softplus(y_raw)

            return y_raw  # (B,H)

        loc = out[..., self.loc_idx:self.loc_idx + 1].clone()  # (B,H,1)

        base_last = x_in[:, -1:, 0:1]     # (B,1,1)
        loc = loc + base_last

        loc = loc * self.out_scale + self.out_bias
        loc = loc + self.dw_gain * self.dw_head(loc.transpose(1, 2)).transpose(1, 2)

        if self.use_revin:
            loc = self.revin_layer(loc, "denorm")  # (B,H,1)

        if (future_exo is not None) and (self.exo_head is not None) and (self.exo_dim > 0):
            fe = _pad_or_slice_last_dim(future_exo.float(), self.exo_dim, pad_value=0.0)
            ex = apply_exo_shift_linear(
                self.exo_head, fe, horizon=self.horizon,
                out_dtype=loc.dtype, out_device=loc.device,
            )  # (B,H)
            loc = loc + ex.unsqueeze(-1)

        if self.final_nonneg:
            loc = torch.clamp_min(loc, 0.0)

        # -----------------------------
        # scale path (make it consistent with out-space)
        # -----------------------------
        scale_raw_out = None
        if self.use_revin and (self.scale_idx is not None):
            scale_raw_norm = out[..., self.scale_idx:self.scale_idx + 1].clone()  # (B,H,1)
            scale_pos_norm = self._scale_pos_from_raw(scale_raw_norm)             # (B,H,1)

            # RevIN has denorm_scale() in your codebase (important!)
            scale_pos_out = self.revin_layer.denorm_scale(scale_pos_norm)

            if self.dist_min_scale and self.dist_min_scale > 0:
                scale_pos_out = torch.clamp_min(scale_pos_out, self.dist_min_scale)

            pre = (scale_pos_out - self.dist_eps).clamp_min(1e-12)
            scale_raw_out = self._inv_scale_transform(pre)

        # -----------------------------
        # re-pack output (replace loc/scale only)
        # -----------------------------
        parts = []
        for i in range(self.out_mul):
            if i == self.loc_idx:
                parts.append(loc)
            elif (scale_raw_out is not None) and (i == self.scale_idx):
                parts.append(scale_raw_out)
            else:
                parts.append(out[..., i:i + 1])

        out2 = torch.cat(parts, dim=-1)
        return out2

class PatchMixerPointModel(PatchMixerModel):
    def __init__(self, cfg: PatchMixerConfig):
        super().__init__(cfg)


class PatchMixerDistributionModel(PatchMixerModel):
    def __init__(self, cfg: PatchMixerConfig):
        super().__init__(cfg)
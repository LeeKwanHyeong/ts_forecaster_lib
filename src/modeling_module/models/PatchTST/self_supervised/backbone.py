# modeling_module/models/PatchTST/self_supervised/backbone.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


def _cfg_get(cfg, name: str, default):
    return getattr(cfg, name, default)


def _sinusoidal_pos_emb(n: int, d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    사인/코사인 기반의 고정 위치 임베딩(Sinusoidal Positional Embedding) 생성.

    기능:
    - 시퀀스 길이(n)와 차원(d)에 따른 위치 정보 행렬 계산.
    - 짝수 인덱스는 sin, 홀수 인덱스는 cos 함수 적용.
    """
    if d <= 0:
        raise ValueError(f"d_model must be > 0. got d={d}")
    pos = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)  # [n,1]
    i = torch.arange(d, device=device, dtype=dtype).unsqueeze(0)  # [1,d]
    angle_rates = 1.0 / torch.pow(10000.0, (2 * (i // 2)) / d)
    angles = pos * angle_rates
    pe = torch.zeros((n, d), device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe.unsqueeze(0)  # [1,n,d]


class Patchify(nn.Module):
    """
    시계열 데이터의 패치화(Patching) 및 차원 변환 모듈.

    기능:
    - 입력 시계열 [B, C, L]을 패치 시퀀스 [B, N, C*P]로 변환.
    - 입력 길이가 패치 길이보다 짧을 경우 우측 패딩 적용.
    - Unfold 연산을 통한 슬라이딩 윈도우 패치 생성.
    """

    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        if self.patch_len <= 0:
            raise ValueError("patch_len must be > 0")
        if self.stride <= 0:
            raise ValueError("stride must be > 0")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        패치화 연산 수행.

        Args:
            x: 원본 시계열 [B,C,L]
        Returns:
            patches: 변환된 패치 텐서 [B,N,C*patch_len]
            N: 생성된 패치 개수
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x as [B,C,L]. got {tuple(x.shape)}")

        B, C, L = x.shape
        # 입력 길이가 패치보다 짧은 경우 패딩 처리
        if L < self.patch_len:
            pad = self.patch_len - L
            x = torch.nn.functional.pad(x, (0, pad))  # 시간 축 우측 패딩
            L = self.patch_len

        # Unfold를 이용한 패치 생성: [B,C,N,patch_len]
        x_unf = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        N = x_unf.size(-2)

        # 차원 재배열 및 병합: [B,N,C,patch_len] -> [B,N,C*patch_len]
        patches = x_unf.permute(0, 2, 1, 3).contiguous().view(B, N, C * self.patch_len)
        return patches, N


class PatchTSTSelfSupBackbone(nn.Module):
    """
    자기지도 학습(Self-Supervised Learning)을 위한 PatchTST 백본.
    마스킹된 패치 재구성(Masked Patch Reconstruction) 태스크 수행.

    구조:
    - Patchify: 시계열 패치화.
    - Masking: 일부 패치를 학습 가능한 마스크 토큰으로 대체.
    - Encoder: 트랜스포머 인코더를 통한 잠재 표현 학습.
    - Head: 패치 원본 값 재구성을 위한 선형 투영.
    """

    def __init__(self, cfg, attn_core=None):
        super().__init__()

        # 설정 참조 저장
        self.cfg = cfg

        # 하이퍼파라미터 로드
        self.patch_len = int(_cfg_get(cfg, "patch_len", 16))
        self.stride = int(_cfg_get(cfg, "stride", self.patch_len))
        self.d_model = int(_cfg_get(cfg, "d_model", 128))
        self.n_heads = int(_cfg_get(cfg, "n_heads", 8))
        self.e_layers = int(_cfg_get(cfg, "e_layers", 3))
        self.d_ff = int(_cfg_get(cfg, "d_ff", self.d_model * 4))
        self.dropout = float(_cfg_get(cfg, "dropout", 0.1))
        self.activation = str(_cfg_get(cfg, "activation", "gelu"))

        # 변수/채널 수 설정
        self.n_vars = int(_cfg_get(cfg, "n_vars", _cfg_get(cfg, "enc_in", 1)))

        # 패치화 모듈 초기화
        self.patchify = Patchify(patch_len=self.patch_len, stride=self.stride)

        # 학습 가능한 마스크 토큰 초기화 (패치 차원과 동일)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.n_vars * self.patch_len))

        # 패치 임베딩 레이어: (C*patch_len) -> d_model
        self.patch_embed = nn.Linear(self.n_vars * self.patch_len, self.d_model)

        # 인코더 생성 (Supervised 모델의 Encoder 재사용)
        self.encoder = self._build_encoder(attn_core=attn_core)

        # 출력 정규화
        self.norm_out = nn.LayerNorm(self.d_model)

        # 사전학습 헤드(Decoder): d_model -> (C*patch_len) 복원
        self.pretrain_head = nn.Linear(self.d_model, self.n_vars * self.patch_len)

    def _build_encoder(self, attn_core=None) -> nn.Module:
        """
        Supervised 학습용 TSTEncoder와의 호환성을 위한 설정 매핑 및 인코더 생성.

        기능:
        - Self-supervised 설정 키를 Supervised 인코더가 기대하는 키로 매핑.
        - modeling_module 내의 TSTEncoder 클래스를 사용하여 인스턴스 생성.
        """

        # ---- 설정 동기화 (Contract 유지) ----
        cfg = self.cfg

        # 레이어 수 매핑 (e_layers -> n_layers)
        if not hasattr(cfg, "n_layers"):
            setattr(cfg, "n_layers", int(getattr(cfg, "e_layers", self.e_layers)))

        # 기타 인코더 필수 설정 보장
        if not hasattr(cfg, "d_ff"):
            setattr(cfg, "d_ff", int(getattr(cfg, "d_ff", self.d_ff)))
        if not hasattr(cfg, "act"):
            setattr(cfg, "act", str(getattr(cfg, "activation", self.activation)))
        if not hasattr(cfg, "norm"):
            setattr(cfg, "norm", str(getattr(cfg, "norm", "BatchNorm")))
        if not hasattr(cfg, "dropout"):
            setattr(cfg, "dropout", float(getattr(cfg, "dropout", self.dropout)))
        if not hasattr(cfg, "pre_norm"):
            setattr(cfg, "pre_norm", bool(getattr(cfg, "pre_norm", False)))
        if not hasattr(cfg, "store_attn"):
            setattr(cfg, "store_attn", bool(getattr(cfg, "store_attn", False)))

        # 모델 차원 및 헤드 수 보장
        if not hasattr(cfg, "d_model"):
            setattr(cfg, "d_model", int(self.d_model))
        if not hasattr(cfg, "n_heads"):
            setattr(cfg, "n_heads", int(self.n_heads))

        # ---- 인코더 인스턴스 생성 ----
        from modeling_module.models.PatchTST.common.encoder import \
            TSTEncoder  # 내부 Encoder 클래스 참조

        # 시퀀스 길이 추정 (안전 장치)
        seq_len = getattr(cfg, "seq_len", getattr(cfg, "lookback", None))
        if seq_len is None:
            q_len = 1
        else:
            q_len = self.infer_patch_num(int(seq_len))

        return TSTEncoder(q_len=q_len, cfg=cfg)

    @torch.no_grad()
    def infer_patch_num(self, L: int) -> int:
        """입력 길이에 따른 패치 개수 계산."""
        L = int(L)
        if L < self.patch_len:
            return 1
        return ((L - self.patch_len) // self.stride) + 1

    def forward_from_patches(
            self,
            patches: torch.Tensor,
            patch_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        패치 입력 기반 마스킹, 인코딩, 복원 수행.

        Args:
            patches: 입력 패치 [B,N,C*patch_len]
            patch_mask: 마스킹 여부 [B,N] (True=Masked)
        Returns:
            z: 인코더 출력 잠재 벡터 [B,N,d_model]
            patch_pred: 재구성된 패치 [B,N,C*patch_len]
        """
        if patches.dim() != 3:
            raise ValueError(f"Expected patches as [B,N,D]. got {tuple(patches.shape)}")

        # 마스킹 적용: 마스크된 위치를 학습 가능한 토큰으로 대체
        if patch_mask is not None:
            if patch_mask.dtype != torch.bool:
                patch_mask = patch_mask.bool()
            if patch_mask.dim() != 2 or patch_mask.shape[:2] != patches.shape[:2]:
                raise ValueError(f"patch_mask must be [B,N] matching patches. got {tuple(patch_mask.shape)}")

            mask_token = self.mask_token.expand(patches.size(0), patches.size(1), -1)  # [B,N,D]
            patches = torch.where(patch_mask.unsqueeze(-1), mask_token, patches)

        # 임베딩 및 위치 인코딩 추가
        z = self.patch_embed(patches)  # [B,N,d_model]
        z = z + _sinusoidal_pos_emb(z.size(1), z.size(2), device=z.device, dtype=z.dtype)
        z = torch.nn.functional.dropout(z, p=self.dropout, training=self.training)

        # 인코더 통과 및 정규화
        z = self.encoder(z)
        z = self.norm_out(z)

        # 패치 공간으로 복원 (Decoding)
        patch_pred = self.pretrain_head(z)  # [B,N,C*patch_len]
        return z, patch_pred

    def forward(
            self,
            x: torch.Tensor,
            patch_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        입력 시계열의 패치화 및 자기지도 학습 순전파.

        Args:
            x: 원본 시계열 [B,C,L]
            patch_mask: 패치 마스크 [B,N]
        Returns:
            patches: 원본 패치 (Target)
            z: 잠재 표현
            patch_pred: 재구성된 패치 (Prediction)
        """
        # 시계열 -> 패치 변환
        patches, _ = self.patchify(x)

        # 마스킹 및 복원 과정 수행
        z, patch_pred = self.forward_from_patches(patches=patches, patch_mask=patch_mask)
        return patches, z, patch_pred
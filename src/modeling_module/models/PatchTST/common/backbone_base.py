# src/modeling_module/models/PatchTST/common/backbone_base.py

import torch
from torch import nn
from modeling_module.models.PatchTST.common.patching import compute_patch_num
from modeling_module.models.PatchTST.common.pos_encoding import positional_encoding, PositionalEncoding

import torch
import torch.nn as nn


class PatchBackboneBase(nn.Module):
    """
    PatchTST의 기본 백본 구조를 확장하여 다양한 입력(Target, Exo)을 처리하는 기저 클래스.

    기능:
    - 입력 시계열의 패치화(Patching) 및 임베딩(Embedding).
    - 타겟 변수 외 연속형/범주형 외생 변수의 통합 처리.
    - 입력 차원을 모델의 잠재 차원(d_model)으로 투영.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.c_in = cfg.c_in
        self.d_model = cfg.d_model
        self.patch_len = cfg.patch_len
        self.stride = cfg.stride

        # 1) 패치 개수 산출
        # Lookback 윈도우와 패치 설정(길이, 스트라이드, 패딩)에 따른 총 패치 수 계산
        self.patch_num = compute_patch_num(cfg.lookback, cfg.patch_len, cfg.stride, cfg.padding_patch)

        # -----------------------------------------------------------
        # [Refactoring] Input Projection Layer (Mixer)
        # 구조: Target Patch + Past Cont Patch + Past Cat Patch -> d_model
        # -----------------------------------------------------------

        # A. Target Dimension 설정
        # 단일 패치 내 타겟 데이터 차원: patch_len * c_in
        self.target_input_dim = self.patch_len * self.c_in

        # B. Past Continuous Exo Dimension 설정
        # 과거 연속형 외생 변수 차원 계산
        self.d_past_cont = getattr(cfg, 'd_past_cont', 0)
        self.cont_input_dim = self.patch_len * self.d_past_cont

        # C. Past Categorical Exo Dimension 설정
        # 과거 범주형 외생 변수 처리 및 임베딩 차원 계산
        self.d_past_cat = getattr(cfg, 'd_past_cat', 0)
        self.cat_cardinalities = getattr(cfg, 'cat_cardinalities', [])
        self.d_cat_emb = getattr(cfg, 'd_cat_emb', 8)

        # 범주형 임베딩 레이어 생성
        if self.d_past_cat > 0:
            assert len(self.cat_cardinalities) == self.d_past_cat, \
                f"cat_cardinalities length({len(self.cat_cardinalities)}) must match d_past_cat({self.d_past_cat})"

            # 각 범주형 변수별 독립적 임베딩 레이어 구성
            self.cat_embs = nn.ModuleList([
                nn.Embedding(c_num, self.d_cat_emb) for c_num in self.cat_cardinalities
            ])
            # 범주형 입력 총 차원: patch_len * (변수개수 * 임베딩차원)
            self.cat_input_dim = self.patch_len * (self.d_past_cat * self.d_cat_emb)
        else:
            self.cat_embs = nn.ModuleList()
            self.cat_input_dim = 0

        # D. Total Input Dimension per Patch 설정
        # 모든 입력 소스(Target, Cont, Cat)의 차원 합산
        self.total_input_dim = self.target_input_dim + self.cont_input_dim + self.cat_input_dim

        print(f"[DBG-backbone-init] d_past_cont={self.d_past_cont} cont_input_dim={self.cont_input_dim} "
              f"target_input_dim={self.target_input_dim} total_input_dim={self.total_input_dim}")

        # E. Projection Layer (All -> d_model)
        # 결합된 패치 벡터를 모델의 잠재 차원(d_model)으로 선형 투영
        # Unfold -> Concat -> Linear 흐름을 통한 다중 소스 통합
        self.input_proj = nn.Linear(self.total_input_dim, self.d_model)

        # Positional Encoding 초기화
        # 패치 순서 정보 주입을 위한 위치 인코딩 설정
        self.pos_enc = positional_encoding(
            pe=getattr(cfg, "pe", "sincos"),
            learn_pe=getattr(cfg, "learn_pe", True),
            q_len=self.patch_num,
            d_model=self.d_model,
        )

    def _process_categorical(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        범주형 변수 처리 및 임베딩 변환.

        Args:
            x_cat: [B, L, n_cat] - 정수 인덱스 형태의 범주형 입력
        Return:
            [B, L, n_cat * d_emb] - 임베딩 후 병합된 연속형 벡터
        """
        if self.d_past_cat == 0:
            return None

        embeddings = []
        for i, emb_layer in enumerate(self.cat_embs):
            # 각 변수별 임베딩 조회: [B, L] -> [B, L, d_emb]
            embeddings.append(emb_layer(x_cat[..., i]))

        # 임베딩 차원 기준 병합 -> [B, L, n_cat * d_emb]
        return torch.cat(embeddings, dim=-1)

    def _patchify_and_embed(self, x: torch.Tensor, p_cont: torch.Tensor, p_cat: torch.Tensor) -> torch.Tensor:
        """
        입력 데이터 패치화, 특징 결합, 투영 및 위치 인코딩 적용.

        Args:
            x: [B, L, C] (Target)
            p_cont: [B, L, d_past_cont] (Continuous Exo)
            p_cat: [B, L, d_past_cat] (Categorical Exo)
        Return:
            z: [B, Patch_Num, d_model] - 최종 임베딩된 패치 시퀀스
        """
        # 1. 입력 데이터 전처리 (Padding)
        # -----------------------------------
        # 'end' 패딩 전략 사용 시 마지막 패치 구성을 위한 복제 패딩 적용
        if self.cfg.padding_patch == 'end':
            # Permute를 통해 [B, C, L] 형태로 변환 후 패딩 적용
            pad_func = nn.ReplicationPad1d((0, self.stride))

            x = pad_func(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L+S, C]
            if p_cont is not None and p_cont.shape[-1] > 0:
                p_cont = pad_func(p_cont.permute(0, 2, 1)).permute(0, 2, 1)
            if p_cat is not None and p_cat.shape[-1] > 0:
                p_cat = pad_func(p_cat.permute(0, 2, 1)).permute(0, 2, 1)

        # 2. 범주형 변수 임베딩 (Before Patching)
        # -----------------------------------
        # 패치화 이전에 임베딩 수행: [B, L, n_cat] -> [B, L, n_cat * d_emb]
        p_cat_emb = self._process_categorical(p_cat) if (p_cat is not None and self.d_past_cat > 0) else None

        # 3. 패치 생성 (Unfold)
        # -----------------------------------
        # Sliding Window를 통한 패치 분할 및 평탄화

        def make_patches(tensor):
            if tensor is None: return None
            # unfold(dimension, size, step) 적용
            # [B, L, C] -> [B, N, C, P]
            patches = tensor.unfold(1, self.patch_len, self.stride)
            B, N, C, P = patches.shape
            # [B, N, C*P] 형태로 차원 병합
            return patches.reshape(B, N, C * P)

        # 타겟 데이터 패치화
        x_patches = make_patches(x)  # [B, N, c_in * P]

        features = [x_patches]

        # 연속형 외생 변수 패치화
        if self.d_past_cont > 0:
            cont_patches = make_patches(p_cont)  # [B, N, d_cont * P]
            features.append(cont_patches)

        # 임베딩된 범주형 외생 변수 패치화
        if self.d_past_cat > 0 and p_cat_emb is not None:
            cat_patches = make_patches(p_cat_emb)  # [B, N, d_cat_total * P]
            features.append(cat_patches)

        # 4. 특징 결합 및 투영 (Concatenate & Project)
        # -----------------------------------
        # 모든 특징 벡터를 마지막 차원 기준으로 결합 -> [B, N, Total_Dim]
        input_vectors = torch.cat(features, dim=-1)

        # 선형 투영을 통해 d_model 차원으로 변환 -> [B, N, d_model]
        z = self.input_proj(input_vectors)

        # 5. 위치 인코딩 추가 (Add Positional Encoding)
        # -----------------------------------
        # 패치 길이에 맞는 위치 정보 주입
        if self.pos_enc.size(0) < z.size(1):
            new_pe = PositionalEncoding(z.size(1), self.d_model).to(z.device)
            z = z + new_pe.unsqueeze(0)
        else:
            z = z + self.pos_enc[:z.size(1), :].unsqueeze(0)

        return z
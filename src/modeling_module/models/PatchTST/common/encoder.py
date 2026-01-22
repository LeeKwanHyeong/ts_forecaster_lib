from typing import Optional

import torch
from torch import nn

from modeling_module.models.common_layers.Attention import build_attention
from modeling_module.models.PatchTST.common.basics import Transpose, get_activation_fn
from modeling_module.models.PatchTST.common.pos_encoding import positional_encoding


class TSTEncoderLayer(nn.Module):
    """
    트랜스포머 인코더의 단일 레이어(Block) 정의.

    기능:
    - Multi-Head Attention (MHA) 및 Feed-Forward Network (FFN) 수행.
    - Residual Connection 및 Normalization(Batch/Layer) 적용.
    - Pre-Norm 구조 지원 (Norm -> Attention -> Add).
    """

    def __init__(self,
                 d_model,
                 d_ff=256,
                 norm='BatchNorm',
                 dropout=0.0,
                 activation='gelu',
                 mha: nn.Module = None,
                 pre_norm=False,
                 store_attn=False,
                 use_residual_logits=True):
        super().__init__()
        assert mha is not None, "mha 모듈 필요. (build_attention으로 생성하여 주입)."
        self.mha = mha
        self.pre_norm = pre_norm
        self.store_attn = store_attn
        self.use_residual_logits = use_residual_logits

        # Attention 블록용 Dropout 및 Normalization
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = (nn.Sequential(
            Transpose(1, 2),
            nn.BatchNorm1d(d_model),
            Transpose(1, 2)
        ) if 'batch' in norm.lower() else nn.LayerNorm(d_model))

        # Feed-Forward Network (FFN) 구성
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # FFN 블록용 Dropout 및 Normalization
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = (nn.Sequential(
            Transpose(1, 2),
            nn.BatchNorm1d(d_model),
            Transpose(1, 2)
        ) if 'batch' in norm.lower() else nn.LayerNorm(d_model))

    def forward(self, src, prev_logits=None, attn_mask=None):
        """
        인코더 레이어 순전파 수행.
        입력 -> (Norm) -> Attention -> Residual -> (Norm) -> FFN -> Residual.
        """
        # src: [Batch, Length, d_model]

        # 1. Attention Block
        # Pre-Norm 적용 여부에 따른 분기
        x = self.norm_attn(src) if self.pre_norm else src

        # Multi-Head Attention 수행
        out, attn, logits = self.mha(x, attn_mask=attn_mask, prev_logits=prev_logits)
        if self.store_attn: self.attn = attn

        # Residual Connection (입력 + Attention 출력)
        src = src + self.dropout_attn(out)

        # 2. Feed-Forward Block
        # Pre-Norm 적용
        y = self.norm_ffn(src) if self.pre_norm else src
        y = self.ff(y)

        # Residual Connection (Attention 결과 + FFN 출력)
        out = src + y
        return out, logits


class TSTEncoder(nn.Module):
    """
    다수의 TSTEncoderLayer를 적층한 전체 인코더 모듈.

    기능:
    - 설정된 레이어 수(n_layers)만큼 인코더 블록 반복 수행.
    - 계층 간 Logit 전달(Residual Logits) 지원.
    """

    def __init__(self, q_len, cfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList()

        # 인코더 레이어 스택 생성
        for _ in range(cfg.n_layers):
            cfg.attn.d_model = cfg.d_model
            mha = build_attention(cfg.attn)
            self.layers.append(
                TSTEncoderLayer(
                    d_model=cfg.d_model,
                    d_ff=cfg.d_ff,
                    norm=cfg.norm,
                    dropout=cfg.dropout,
                    activation=cfg.act,
                    mha=mha,
                    pre_norm=cfg.pre_norm,
                    store_attn=cfg.store_attn,
                    use_residual_logits=cfg.attn.residual_logits
                )
            )

    def forward(self, src: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """
        순차적으로 모든 인코더 레이어 통과.
        """
        logits = None
        x = src
        for layer in self.layers:
            x, logits = layer(x, prev_logits=logits, attn_mask=attn_mask)
        return x


class TSTiEncoder(nn.Module):  # channel-independent
    """
    채널 독립적(Channel-independent) PatchTST 인코더.

    기능:
    - 패치 투영(Projection) 및 위치 인코딩(Positional Encoding) 적용.
    - 배치(Batch)와 변수(Variable) 차원을 병합하여 모든 시계열을 독립적으로 처리.
    - 트랜스포머 인코딩 후 원래 차원으로 복원.
    """

    def __init__(self, c_in, patch_num, patch_len, cfg):
        super().__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.cfg = cfg

        q_len = patch_num
        # 패치 투영 레이어 (Patch_Len -> d_model)
        self.W_P = nn.Linear(patch_len, cfg.d_model)
        self.seq_len = q_len

        # 위치 인코딩 초기화
        self.W_pos = positional_encoding(cfg.pe, cfg.learn_pe, q_len, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.encoder = TSTEncoder(q_len, cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        패치 임베딩 및 인코딩 수행.
        Input: [B, nvars, patch_len, patch_num]
        Output: [B, nvars, d_model, patch_num]
        """
        # x: [B, nvars, patch_len, patch_num]
        B, n_vars, _, _ = x.shape

        # 차원 변경: [B, n_vars, Num_Patches, Patch_Len]
        x = x.permute(0, 1, 3, 2)

        # 선형 투영: [B, n_vars, Num_Patches, d_model]
        x = self.W_P(x)

        # 채널 독립성 확보를 위한 차원 병합: [B * n_vars, Num_Patches, d_model]
        u = x.reshape(B * n_vars, x.size(2), x.size(3))

        # 위치 인코딩 가산 (Broadcasting)
        u = self.dropout(u + self.W_pos)

        # 트랜스포머 인코더 통과
        z = self.encoder(u)  # [B * n_vars, N, d_model]

        # 차원 복원: [B, n_vars, Num_Patches, d_model]
        z = z.reshape(B, n_vars, x.size(2), z.size(2))

        # 출력 차원 조정: [B, n_vars, d_model, Num_Patches]
        z = z.permute(0, 1, 3, 2)
        return z
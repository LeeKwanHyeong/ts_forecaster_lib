import torch
from torch import nn

from modeling_module.models.PatchTST.common.backbone_base import PatchBackboneBase
from modeling_module.models.PatchTST.common.encoder import TSTEncoder, TSTEncoderLayer
from modeling_module.models.common_layers.Attention import MultiHeadAttention, FullAttention, ProbAttention, \
    FullAttentionWithLogits


class SupervisedBackbone(PatchBackboneBase):
    """
    PatchTST 지도학습(Supervised Learning) 전용 백본 모델.

    기능:
    - 입력 시계열(Target + Exo)을 패치화 및 임베딩.
    - 트랜스포머 인코더를 통한 잠재 표현(Latent Representation) 학습.
    - 설정(Config)에 따른 어텐션 메커니즘(Full/ProbSparse) 동적 선택.

    입력: [Batch, Length, Channel] (Exo 포함 가능)
    출력: [Batch, Num_Patches, d_model]
    """

    def __init__(self, cfg, attn_core=None):
        super().__init__(cfg)
        self.cfg = cfg

        # -----------------------------------
        # 1. Attention Core 자동 선택
        # -----------------------------------
        # 설정된 어텐션 타입에 따라 적절한 모듈 초기화
        if attn_core is None:
            attn_type = getattr(cfg.attn, "type", "full").lower()

            if attn_type == "full":
                # 표준 O(L^2) 어텐션
                attn_core = FullAttention(
                    mask_flag=cfg.attn.causal,
                    attention_dropout=cfg.attn.attn_dropout,
                    output_attention=cfg.attn.output_attention,
                )
            elif attn_type == "probsparse":
                # Informer 스타일의 O(L log L) 희소 어텐션
                attn_core = ProbAttention(
                    mask_flag=cfg.attn.causal,
                    factor=cfg.attn.factor,
                    attention_dropout=cfg.attn.attn_dropout,
                    output_attention=cfg.attn.output_attention,
                    # residual=cfg.attn.residual_logits,
                )
            elif attn_type == 'full_logits':
                # Logit 반환을 지원하는 확장형 어텐션
                attn_core = FullAttentionWithLogits(
                    mask_flag=True,
                    attention_dropout=0.1,
                    output_attention=False,
                    return_logits=True,
                    use_realformer_residual=True
                )
            else:
                raise ValueError(f"Invalid attention type: {attn_type}")
        self.attn_core = attn_core

        # -----------------------------------
        # 2. Encoder Layer Stack 구성
        # -----------------------------------
        # TSTEncoder 내부에서 레이어 스택을 빌드하도록 설정 전달
        # (상단의 layers 리스트 생성 루프는 TSTEncoder 내부 로직과 중복될 수 있으나,
        #  일반적으로 MHA 구성을 명시하기 위한 코드 블록임)
        layers = []
        for i in range(cfg.n_layers):
            mha = MultiHeadAttention(
                d_model=self.d_model,
                n_heads=cfg.attn.n_heads,
                d_k=cfg.attn.d_k,
                d_v=cfg.attn.d_v,
                proj_dropout=cfg.attn.proj_dropout,
                attn_core=self.attn_core,
            )
            layers.append(
                TSTEncoderLayer(
                    mha=mha,
                    d_model=self.d_model,
                    d_ff=cfg.d_ff,
                    dropout=cfg.dropout,
                    pre_norm=cfg.pre_norm,
                    store_attn=getattr(cfg, "store_attn", False),
                )
            )

        # 전체 인코더 모듈 초기화
        self.encoder = TSTEncoder(q_len=self.patch_num, cfg=cfg)

        # 최종 출력 정규화 레이어
        self.norm_out = nn.LayerNorm(cfg.d_model)

    # -----------------------------------
    # 3. Forward (Patchify → Encoder)
    # -----------------------------------
    def forward(self, x: torch.Tensor, p_cont: torch.Tensor = None, p_cat: torch.Tensor = None) -> torch.Tensor:
        """
        순전파 수행.

        Args:
            x: 타겟 시계열 [B, L, C] (RevIN 적용됨)
            p_cont: 과거 연속형 외생 변수 [B, L, d_past_cont]
            p_cat: 과거 범주형 외생 변수 [B, L, d_past_cat]
        Returns:
            z: 인코딩된 잠재 벡터 [B, N, d_model]
        """
        # 1) 패치화 및 임베딩 (부모 클래스 메서드 활용)
        # 입력 데이터들을 결합하여 패치 단위 임베딩 생성
        z = self._patchify_and_embed(x, p_cont, p_cat)  # [B, N, d_model]

        # 2) 트랜스포머 인코더 통과
        # 패치 간의 시간적 관계 및 문맥 학습
        z = self.encoder(z)  # [B, N, d_model]

        # 3) 출력 정규화
        z = self.norm_out(z)

        return z
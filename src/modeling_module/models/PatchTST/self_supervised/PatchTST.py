# modeling_module/models/PatchTST/self_supervised/PatchTST.py

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple

import torch
from torch import nn

from modeling_module.models.PatchTST.common.configs import PatchTSTConfig
from modeling_module.models.common_layers.RevIN import RevIN

from modeling_module.models.PatchTST.self_supervised.backbone import PatchTSTSelfSupBackbone


def _masked_recon_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        loss_type: str = "mse",
        eps: float = 1e-12,
) -> torch.Tensor:
    """
    마스킹된 위치에 대해서만 재구성 손실(Reconstruction Loss) 계산.

    Args:
        pred:   예측된 패치 값 [B,N,D]
        target: 원본 패치 값 [B,N,D]
        mask:   마스크 여부 [B,N] (True=손실 계산 포함)
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()

    # 마스크 차원 확장 [B,N] -> [B,N,1] (채널 D에 브로드캐스팅)
    m = mask.unsqueeze(-1)

    diff = pred - target
    if loss_type.lower() == "mae":
        per_elem = diff.abs()
    else:
        per_elem = diff * diff  # mse

    # 마스킹된 부분만 추출
    masked = per_elem * m  # [B,N,D]

    # 마스킹된 요소의 총 개수로 나누어 평균 손실 산출
    denom = m.sum() * pred.size(-1)
    denom = denom.clamp_min(eps)  # 0으로 나누기 방지
    return masked.sum() / denom


class PatchTSTPretrainModel(nn.Module):
    """
    PatchTST 자기지도 사전학습(Self-Supervised Pretraining) 모델.

    기능:
    - 마스킹된 패치 재구성(Masked Patch Reconstruction) 학습 수행.
    - 지도학습(Forecasting) 모델과 호환되는 입력 인터페이스 제공 (외생 변수는 무시).
    - RevIN을 통한 입력 정규화 및 정규화된 공간에서의 재구성 학습.

    출력 (Dict):
        loss: 재구성 손실 (Scalar)
        patch_mask: 생성된 마스크 [B,N]
        patch_pred: 예측된 패치 [B,C,N,patch_len]
        patch_target: 타겟 패치 [B,C,N,patch_len]
        z: 인코더 잠재 표현 [B,N,d_model]
    """

    def __init__(self, cfg: PatchTSTConfig, attn_core=None):
        super().__init__()
        self.cfg = cfg
        self.n_vars = int(getattr(cfg, "n_vars", getattr(cfg, "enc_in", 1)))
        self.patch_len = int(getattr(cfg, "patch_len", 16))
        self.stride = int(getattr(cfg, "stride", self.patch_len))

        # 정규화 모듈 (RevIN) 설정
        self.use_revin = bool(getattr(cfg, "revin", True))
        self.revin_layer = RevIN(num_features=self.n_vars) if self.use_revin else None

        # 사전학습용 백본 초기화
        self.backbone = PatchTSTSelfSupBackbone(cfg, attn_core=attn_core)

        # 기본 사전학습 설정 (마스크 비율, 손실 함수)
        self.default_mask_ratio = float(getattr(cfg, "mask_ratio", 0.3))
        self.default_loss_type = str(getattr(cfg, "pretrain_loss", "mse"))

    @classmethod
    def from_config(cls, config: PatchTSTConfig):
        """설정 객체로부터 모델 인스턴스 생성 팩토리 메서드."""
        return cls(cfg=config)

    def _make_patch_mask(self, B: int, N: int, mask_ratio: float, device) -> torch.Tensor:
        """
        랜덤 패치 마스크 생성.
        Returns:
            mask [B,N] bool (True=Masked)
        """
        if mask_ratio <= 0.0:
            return torch.zeros((B, N), device=device, dtype=torch.bool)
        if mask_ratio >= 1.0:
            return torch.ones((B, N), device=device, dtype=torch.bool)

        # 샘플별 무작위 마스킹
        rand = torch.rand((B, N), device=device)
        return rand < mask_ratio

    def forward(
            self,
            x: torch.Tensor,
            # Trainer/Adapter 호환성 유지를 위한 더미 인자 (Self-Supervised에서는 무시됨)
            future_exo: Optional[torch.Tensor] = None,
            past_exo_cont: Optional[torch.Tensor] = None,
            past_exo_cat: Optional[torch.Tensor] = None,
            part_ids=None,
            mode: Optional[str] = None,
            # 사전학습 제어 인자
            mask_ratio: Optional[float] = None,
            patch_mask: Optional[torch.Tensor] = None,
            return_loss: bool = True,
            loss_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        순전파 및 재구성 손실 계산.

        Args:
            x: 입력 시계열 [B,C,L] 또는 [B,L,C]
            mask_ratio: 마스킹 비율 (patch_mask가 없을 때 사용)
            patch_mask: 외부에서 지정한 마스크 [B,N]
        """

        if x.dim() != 3:
            raise ValueError(f"Expected x as 3D tensor. got {tuple(x.shape)}")

        B, d1, d2 = x.shape

        # 입력 차원 자동 감지 및 변환: [B, C, L] 포맷으로 통일
        if d1 == self.n_vars:
            x_bcl = x
        elif d2 == self.n_vars:
            x_bcl = x.permute(0, 2, 1).contiguous()
        else:
            raise ValueError(
                f"Input shape mismatch. cfg.n_vars={self.n_vars}, "
                f"but got x.shape={tuple(x.shape)}. "
                f"Expected dim1==n_vars ([B,C,L]) or dim2==n_vars ([B,L,C])."
            )

        B, C, L = x_bcl.shape
        if C != self.n_vars:
            raise ValueError(f"cfg.n_vars={self.n_vars} but got x.shape[1]={C}")

        # 1. 입력 정규화 (RevIN)
        # PatchTST는 일반적으로 시간 도메인에서 정규화를 수행
        if self.use_revin:
            # RevIN은 [B,L,C] 입력을 기대하므로 변환 후 적용
            x_t = x_bcl.permute(0, 2, 1).contiguous()  # [B,L,C]
            x_n = self.revin_layer(x_t, "norm").permute(0, 2, 1).contiguous()  # [B,C,L]
        else:
            x_n = x

        # 2. 타겟 패치 생성 (정규화된 공간)
        patches_target, N = self.backbone.patchify(x_n)  # [B,N,C*patch_len]

        # 3. 마스크 생성
        if patch_mask is None:
            mr = self.default_mask_ratio if mask_ratio is None else float(mask_ratio)
            patch_mask = self._make_patch_mask(B=B, N=patches_target.size(1), mask_ratio=mr, device=x.device)
        else:
            if patch_mask.dtype != torch.bool:
                patch_mask = patch_mask.bool()

        # 4. 마스킹된 입력에 대한 순전파 (재구성)
        z, patches_pred = self.backbone.forward_from_patches(patches=patches_target, patch_mask=patch_mask)

        # 결과 시각화를 위한 뷰 변환 [B,C,N,patch_len]
        patches_pred_view = patches_pred.view(B, N, C, self.patch_len).permute(0, 2, 1, 3).contiguous()
        patches_tgt_view = patches_target.view(B, N, C, self.patch_len).permute(0, 2, 1, 3).contiguous()

        out: Dict[str, Any] = {
            "patch_mask": patch_mask,  # [B,N]
            "patch_pred": patches_pred_view,  # [B,C,N,patch_len]
            "patch_target": patches_tgt_view,  # [B,C,N,patch_len]
            "z": z,  # [B,N,d_model]
        }

        # 5. 손실 계산 (선택 사항)
        if return_loss:
            lt = self.default_loss_type if loss_type is None else str(loss_type)
            loss = _masked_recon_loss(
                pred=patches_pred,  # [B,N,C*patch_len]
                target=patches_target,  # [B,N,C*patch_len]
                mask=patch_mask,  # [B,N]
                loss_type=lt,
            )
            out["loss"] = loss

        return out

    def export_encoder_state(self) -> Dict[str, torch.Tensor]:
        """
        Downstream(지도학습 Forecasting) 모델 이식을 위한 인코더 상태 추출.

        기능:
        - 전체 state_dict 중 백본의 핵심 블록(Encoder, Embedding 등)만 필터링하여 반환.
        - Supervised 모델 로드 시 키 매칭을 용이하게 함.
        """
        sd = {}
        for k, v in self.state_dict().items():
            # 필요한 접두사를 가진 키만 선택
            if k.startswith("backbone.patch_embed.") or k.startswith("backbone.encoder.") or k.startswith(
                    "backbone.norm_out."):
                sd[k] = v
        return sd
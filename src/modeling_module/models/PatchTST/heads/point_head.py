import torch.nn as nn
import torch
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
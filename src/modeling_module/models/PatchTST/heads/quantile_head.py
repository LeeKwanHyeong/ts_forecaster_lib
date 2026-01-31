import torch
import torch.nn as nn

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

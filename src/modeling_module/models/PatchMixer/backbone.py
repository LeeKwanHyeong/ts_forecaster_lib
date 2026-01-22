# -------------------------
# PatchMixer Backbone
# -------------------------
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAlign(nn.Module):
    """
    입력 텐서의 차원을 동적으로 감지하여 목표 차원(out_dim)으로 투영하는 유틸리티 모듈.

    기능:
    - 선형 레이어(Linear)를 이용한 차원 맞춤.
    - 입력 차원이 변경될 경우 자동으로 내부 가중치를 재생성(Lazy Initialization).
    """

    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.fc: nn.Linear | None = None
        self.in_dim: int | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.size(-1)
        # 초기화 상태이거나 입력 차원이 달라진 경우 레이어 재생성
        if (self.fc is None) or (self.in_dim != d):
            self.fc = nn.Linear(d, self.out_dim, bias=True).to(x.device)
            self.in_dim = d
        return self.fc(x)


class PatchMixerLayer(nn.Module):
    """
    PatchMixer의 핵심 연산 블록. Depthwise Conv와 Pointwise Conv를 사용해 시간/채널 정보를 혼합.

    구조:
    1. Token Mixer: 시간 축(Patch Axis) 정보 교환 (Depthwise Conv).
    2. Channel Mixer: 특징 축(Channel Axis) 정보 교환 (Pointwise Conv).
    3. Residual Connection: 원본 정보 보존.

    입력: (Batch, Channel=d_model, Length=patch_num)
    출력: (Batch, Channel, Length) - 입력과 동일한 크기 유지.
    """

    def __init__(self, d_model: int, kernel_size: int = 5, dropout: float = 0.0, dilation: int = 1):
        super().__init__()
        self.d_model = d_model
        self.ks = int(kernel_size)
        self.dilation = int(dilation)

        # 1. Token Mixer (Depthwise): 패치 간의 관계 학습
        # groups=d_model로 설정하여 채널별 독립 연산 수행
        self.token_mixer = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=self.ks, stride=1, padding=0,
                      dilation=self.dilation, groups=d_model),
            nn.GELU(),
            nn.GroupNorm(num_groups=min(32, d_model), num_channels=d_model),
        )

        # 2. Channel Mixer (Pointwise): 채널 간의 관계 학습
        # kernel_size=1로 설정하여 시간 축은 유지하고 채널 정보만 혼합
        self.channel_mixer = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.GELU(),
            nn.GroupNorm(num_groups=min(32, d_model), num_channels=d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def _same_pad_1d(self, L: int) -> tuple[int, int]:
        """
        Conv1d 출력 길이를 입력과 동일하게 유지하기 위한 SAME 패딩 크기 계산.
        """
        total = self.dilation * (self.ks - 1)
        left = total // 2
        right = total - left
        return left, right

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*, D, A)
        res = x
        L = x.size(-1)

        # 패딩 적용 (Left, Right)
        pl, pr = self._same_pad_1d(L)
        if pl or pr:
            x = F.pad(x, (pl, pr))

        # 믹싱 연산 수행
        x = self.token_mixer(x)  # (B*, D, A) 길이 보존
        x = self.channel_mixer(x)
        x = self.dropout(x)

        # 잔차 연결 (Residual Connection)
        return x + res


class PatchMixerBackbone(nn.Module):
    """
    단일 스케일 PatchMixer 백본 네트워크.

    기능:
    - 시계열 패치화(Patching): 입력 시계열을 겹치는 윈도우로 분할.
    - 투영(Projection): 각 패치를 임베딩 벡터로 변환.
    - 믹싱(Mixing): PatchMixerLayer를 통한 특징 추출.
    - 집약(Aggregation): 변수(N) 차원에 대한 평균을 통해 최종 표현 벡터 생성.

    Input:  (B, L=lookback, N=n_vars)
    Output: (B, Out_Dim)
    """

    def __init__(self, configs, revin: bool = True, affine: bool = True, subtract_last: bool = False):
        super().__init__()
        self.configs = configs

        # 하이퍼파라미터 설정
        self.n_vals: int = int(configs.enc_in)
        self.lookback: int = int(configs.lookback)
        self.patch_size: int = int(configs.patch_len)
        self.stride: int = int(configs.stride)
        self.d_model: int = int(configs.d_model)
        self.depth: int = int(configs.e_layers)
        self.dropout_rate: float = float(getattr(configs, "head_dropout", 0.0))

        # 패치 개수 계산 (Unfold 연산 기준)
        base = (self.lookback - self.patch_size) / float(self.stride) + 1.0
        base = int(base)
        self.patch_num: int = base + 1  # 패딩 포함
        if self.patch_num < 1:
            self.patch_num = 1

        # 출력 차원 정의 (Patch_Num * D_Model)
        self.patch_repr_dim: int = self.patch_num * self.d_model
        self.out_dim: int = self.patch_repr_dim

        # Unfold를 위한 복제 패딩 레이어
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))

        # 믹서 블록 스택 생성
        self.blocks = nn.ModuleList([
            PatchMixerLayer(d_model=self.d_model, kernel_size=int(configs.mixer_kernel_size), dropout=self.dropout_rate)
            for _ in range(self.depth)
        ])

        # 패치 투영 레이어 (Patch Size -> D_Model)
        self.W_P = nn.Linear(self.patch_size, self.d_model)

        # 평탄화 레이어 (Channel, Length) -> (Channel * Length)
        self.flatten = nn.Flatten(start_dim=-2)

    @torch.no_grad()
    def _assert_3d(self, x: torch.Tensor) -> None:
        if x.dim() != 3:
            raise ValueError(f"Expected input 3D tensor (B, L, N). Got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파 흐름:
        1. 전처리: 입력 길이 검증 및 패딩/자르기.
        2. 패치화: Unfold를 통해 (Batch, N, Patch_Num, Patch_Size) 생성.
        3. 투영: 선형 변환으로 임베딩 차원(D_Model)으로 매핑.
        4. 믹싱: PatchMixerLayer 반복 통과.
        5. 집약: 변수(N) 차원 평균 후 평탄화하여 최종 벡터 생성.
        """
        self._assert_3d(x)
        B, L, N = x.shape

        # 입력 길이(L)와 설정된 Lookback 일치 여부 확인 및 보정
        if L != self.lookback:
            if L > self.lookback:
                # 긴 경우 최근 데이터 사용
                x = x[:, -self.lookback:, :]
            else:
                # 짧은 경우 앞부분 복제 패딩
                pad = self.lookback - L
                x = x.transpose(1, 2)  # (B, N, L)
                x = F.pad(x, (pad, 0), mode="replicate")
                x = x.transpose(1, 2)  # (B, L, N)

        # (B, N, L) 형태로 변환
        x = x.permute(0, 2, 1)

        # 패딩 및 Unfold 적용 (패치 생성)
        x = self.padding_patch_layer(x)  # (B, N, L + stride)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # (B, N, A_eff, P)

        # 실제 생성된 패치 수
        A_eff = x.size(2)

        # 패치 투영 (P -> D)
        x = self.W_P(x)  # (B, N, A_eff, D)

        # 배치와 변수 차원 병합 및 Mixer 입력 형태 변환
        # (B*N, D, A_eff) - Mixer는 (Channel, Length) 입력을 기대하므로 Permute 적용
        BNA = x.reshape(B * N, A_eff, self.d_model)
        BDA = BNA.permute(0, 2, 1)

        # Mixer 블록 통과
        for blk in self.blocks:
            BDA = blk(BDA)

        # 형태 복원 및 집약
        # (B*N, D*A_eff) -> (B, N, D*A_eff)
        z = self.flatten(BDA).view(B, N, -1)

        # 다변수 시계열(N)을 평균내어 하나의 컨텍스트 벡터로 요약
        z = z.mean(dim=1)  # (B, D*A_eff)

        # 학습 모드 시 차원 불일치 경고 (디버깅용)
        if z.size(-1) != self.out_dim:
            if self.training:
                print(f"[PatchMixerBackbone][warn] forward out_dim={z.size(-1)} "
                      f"!= declared out_dim={self.out_dim} "
                      f"(A_eff={A_eff}, A_cfg={self.patch_num})")

        return z


class MultiScalePatchMixerBackbone(nn.Module):
    """
    멀티 스케일(Multi-scale) PatchMixer 백본.

    기능:
    - 서로 다른 패치 크기/스트라이드를 가진 여러 PatchMixerBackbone을 병렬로 실행.
    - 각 분기(Branch)의 결과를 투영 후 융합(Fusion)하여 다양한 시간적 패턴 포착.

    Fusion Mode:
    - 'concat': 벡터 연결 후 선형 변환.
    - 'gated': 게이트 메커니즘을 통한 가중 합.
    """

    def __init__(
            self,
            base_configs,
            patch_cfgs: tuple = ((4, 2, 5), (8, 4, 7), (12, 6, 9)),  # (Patch_Len, Stride, Kernel)
            per_branch_dim: int = 128,
            fused_dim: int = 256,
            fusion: str = "concat",  # ['concat', 'gated']
            affine: bool = True,
            subtract_last: bool = False,
    ):
        super().__init__()
        self.fusion = fusion
        self.branches = nn.ModuleList()
        self.projs = nn.ModuleList()

        # 각 설정별 백본 생성
        for (pl, st, ks) in patch_cfgs:
            cfg = copy.deepcopy(base_configs)
            cfg.patch_len = int(pl)
            cfg.stride = int(st)
            cfg.mixer_kernel_size = int(ks)

            # 개별 스케일 백본 (RevIN은 외부에서 처리 가정하에 False)
            branch = PatchMixerBackbone(cfg, revin=False)
            self.branches.append(branch)
            # 차원 통일용 투영 레이어
            self.projs.append(nn.Linear(branch.out_dim, per_branch_dim))

        # 융합 레이어 정의
        if fusion == "concat":
            self.fuse = nn.Linear(per_branch_dim * len(self.branches), fused_dim)
        elif fusion == "gated":
            self.fuse = nn.Linear(per_branch_dim, fused_dim)
            self.gate = nn.Linear(per_branch_dim, 1)  # 중요도 산출용 게이트
        else:
            raise ValueError("fusion must be 'concat' or 'gated'")

        self.out_dim = fused_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reps, gates = [], []
        # 모든 분기 병렬 실행
        for branch, proj in zip(self.branches, self.projs):
            b = branch(x)  # (B, A_i*D_i)
            b = proj(b)  # (B, per_branch_dim)
            reps.append(b)
            if self.fusion == "gated":
                gates.append(self.gate(b))  # (B, 1)

        # 결과 융합
        if self.fusion == "concat":
            z = torch.cat(reps, dim=1)  # 모든 분기 연결
            z = self.fuse(z)  # (B, fused_dim)
        else:
            # Gated Fusion: 각 분기에 학습된 중요도(Gate) 반영
            G = torch.softmax(torch.cat(gates, dim=1), dim=1)  # (B, n_branch)
            S = torch.stack(reps, dim=1)  # (B, n_branch, per_branch_dim)
            z = (G.unsqueeze(-1) * S).sum(dim=1)  # 가중 합
            z = self.fuse(z)  # (B, fused_dim)
        return z


class PatchMixerBackboneWithPatcher(nn.Module):
    """
    외부 패처(External Patcher) 모듈을 사용하는 확장형 백본.

    특징:
    - 기본 Unfold 방식 대신 DynamicPatcher 등 복잡한 패치 생성기 사용 가능.
    - 패처가 생성한 임베딩을 그대로 Mixer 블록에 전달.

    입력: (B, L, N)
    출력: (B, Out_Dim)
    """

    def __init__(self, configs, patcher: nn.Module, e_layers: int | None = None, dropout_rate: float | None = None):
        super().__init__()
        self.cfg = configs
        self.patcher = patcher

        # 패처에서 차원 정보 추출
        self.a: int = int(getattr(patcher, "patch_num"))
        self.d_model: int = int(getattr(patcher, "d_model"))
        self.patch_repr_dim: int = self.a * self.d_model
        self.out_dim: int = self.patch_repr_dim

        self.depth = int(e_layers if e_layers is not None else configs.e_layers)
        self.dropout_rate = float(dropout_rate if dropout_rate is not None else configs.head_dropout)

        # Mixer 블록 스택
        self.blocks = nn.ModuleList([
            PatchMixerLayer(d_model=self.d_model, kernel_size=int(configs.mixer_kernel_size), dropout=self.dropout_rate)
            for _ in range(self.depth)
        ])
        self.flatten = nn.Flatten(start_dim=-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, N = x.shape
        # 1. 외부 패처 호출 (패치 생성 및 투영)
        z = self.patcher(x)  # (B*N, A, D)

        # 2. Mixer 블록 통과
        # 주의: PatchMixerLayer는 (D, A) 순서를 기대하므로
        # 구현에 따라 여기서 permute(0, 2, 1)이 필요할 수 있음.
        # 현재 코드에서는 patcher 출력이 (B*N, A, D)이고
        # PatchMixerLayer 내부 Conv1d가 (D, D, k)이므로 차원 확인 필요.
        # 일반적인 Conv1d 입력은 (Batch, Channel, Length) -> (B*N, D, A)

        # 만약 patcher가 (B*N, A, D)를 뱉는다면 아래 루프 전에 permute 필요 가능성 높음
        z = z.permute(0, 2, 1)  # (B*N, D, A)로 변환 가정

        for blk in self.blocks:
            z = blk(z)  # (B*N, D, A)

        # 3. 결과 평탄화 및 집약
        z = self.flatten(z)  # (B*N, D*A)
        z = z.view(B, N, -1).mean(1)  # (B, D*A) - 변수 축 평균
        return z
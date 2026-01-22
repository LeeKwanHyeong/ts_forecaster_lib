from typing import Optional

import torch.nn as nn


def compute_patch_num(
        lookback: int,
        patch_len: int,
        stride: int,
        padding_patch: Optional[str] = None  # 'end' or None
) -> int:
    """
    입력 시퀀스 길이와 패치 설정에 따른 총 패치 개수 산출.

    기능:
    - 기본 공식: floor((lookback - patch_len) / stride) + 1
    - 패딩 옵션('end') 활성화 시, 마지막 구간 커버를 위해 패치 개수 +1 보정.
    - 입력 파라미터 유효성 검증(Validation) 수행.
    """
    if lookback < patch_len:
        raise ValueError(f"lookback({lookback}) must be >= patch_len({patch_len})")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, gor {stride}")
    if padding_patch not in (None, 'end'):
        raise ValueError("padding_patch must be one of {None, 'end'}")

    n = (lookback - patch_len) // stride + 1
    if padding_patch == 'end':
        n += 1
    return n


def do_patch(
        x,
        patch_len: int,
        stride: int,
        padding_patch: Optional[str] = None
):
    """
    시계열 데이터를 패치 단위로 분할 및 차원 변환.

    기능:
    - 패딩 처리: 옵션에 따라 시계열 끝단에 복제 패딩(Replication Pad) 적용.
    - 언폴드(Unfold): 슬라이딩 윈도우 방식을 통해 시계열을 패치로 분할.
    - 차원 재배열: [Batch, Vars, Lookback] -> [Batch, Vars, Patch_Len, Patch_Num].
    """
    if padding_patch == 'end':
        # 마지막 패치를 위한 우측 패딩 적용
        pad = nn.ReplicationPad1d((0, stride))
        x = pad(x)

    # 슬라이딩 윈도우 적용: [B, n_vars, patch_num, patch_len]
    x = x.unfold(dimension=-1, size=patch_len, step=stride)

    # 차원 치환: [B, n_vars, patch_len, patch_num]
    return x.permute(0, 1, 3, 2)


def unpatch(x_patched):
    """
    패치화된 데이터를 원본 시계열 형태로 복원 (현재 미구현).
    - x_patched: [B, n_vars, patch_len, patch_num]
    """
    # 중첩된 패치를 다시 시계열로 합치는 로직 (Fold 등) 필요
    raise NotImplementedError
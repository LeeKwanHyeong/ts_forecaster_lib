import torch
import torch.nn as nn
from typing import Literal, Optional, Callable, List
from typing import Callable, Union, Sequence
import numpy as np
from datetime import date, datetime

from modeling_module.utils.date_util import DateUtil

def calendar_sin_cos(t: torch.Tensor, period: float, device = 'cpu') -> torch.Tensor:
    """
    단일 주기에 대한 sin/cos 쌍 반환 (..., 2)
    """

    t = torch.as_tensor(t, device=device, dtype=torch.float32)


    return torch.stack([
        torch.sin(2 * torch.pi * t / period),
        torch.cos(2 * torch.pi * t / period)
    ], dim=-1)


def yyyyww_to_week_ordinal(yyyyww: int, *, anchor=(1970, 1, 1)) -> int:
    """
    yyyyww(예: 202511)를 ISO week 기준으로 '연속 주차 index'로 변환.
    anchor=(1970,1,1) 기준으로 몇 주 차이인지 반환.
    """
    y = int(yyyyww) // 100
    w = int(yyyyww) % 100

    # ISO week의 월요일 날짜
    d = date.fromisocalendar(y, w, 1)

    # anchor 날짜(월요일로 맞추는 게 더 깔끔하지만, 일단 주 단위 차이만 쓰면 충분)
    a = date(*anchor)

    # 주 단위 ordinal
    return (d - a).days // 7


def yyyymm_to_month_ordinal(yyyymm: int, *, anchor=(1970, 1)) -> int:
    year = int(yyyymm) // 100
    month = int(yyyymm) % 100
    return (year - int(anchor[0])) * 12 + (month - int(anchor[1]))


def yyyymmdd_to_day_ordinal(yyyymmdd: int, *, anchor=date(1970, 1, 1)) -> int:
    d = DateUtil.yyyymmdd_to_date(int(yyyymmdd))
    return (d - anchor).days


def yyyymmddhh_to_hour_ordinal(yyyymmddhh: int, *, anchor=datetime(1970, 1, 1, 0, 0, 0)) -> int:
    dt = datetime.strptime(str(int(yyyymmddhh)), "%Y%m%d%H")
    return int((dt - anchor).total_seconds() // 3600)


def compose_exo_calendar_cb(date_type: str = "W", *, sincos: bool = True):
    dt = str(date_type).strip().lower()
    dt = {
        "w": "weekly",
        "weekly": "weekly",
        "m": "monthly",
        "monthly": "monthly",
        "d": "daily",
        "daily": "daily",
        "h": "hourly",
        "hourly": "hourly",
    }.get(dt, "weekly")

    if dt == "monthly":
        periods = [12.0]
        ordinal_fn = yyyymm_to_month_ordinal
    elif dt == "weekly":
        periods = [52.0]
        ordinal_fn = yyyyww_to_week_ordinal
    elif dt == "daily":
        periods = [7.0, 365.25]
        ordinal_fn = yyyymmdd_to_day_ordinal
    else:
        periods = [24.0, 168.0]
        ordinal_fn = yyyymmddhh_to_hour_ordinal

    def _calendar_sin_cos(t: torch.Tensor, p: float) -> torch.Tensor:
        ang = 2.0 * np.pi * (t / float(p))
        ang = torch.as_tensor(ang, device=t.device, dtype=t.dtype)
        return torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)

    def cb(start_idx, H: int, device: str = "cpu"):
        H = int(H)
        dev = torch.device(device)

        is_scalar = isinstance(start_idx, (int, np.integer))
        if is_scalar:
            s_raw = np.array([int(start_idx)], dtype=np.int64)
        else:
            s_raw = np.asarray(start_idx, dtype=np.int64).reshape(-1)

        s_ord = np.array([ordinal_fn(v) for v in s_raw], dtype=np.int64)
        s = torch.as_tensor(s_ord, device=dev, dtype=torch.float32)
        offs = torch.arange(0, H, device=dev, dtype=torch.float32)
        t = s[:, None] + offs[None, :]

        feats = []
        if sincos:
            for p in periods:
                feats.append(_calendar_sin_cos(t, p))
            exo = torch.cat(feats, dim=-1)
        else:
            for p in periods:
                feats.append(((t % float(p)) / float(p)).unsqueeze(-1))
            exo = torch.cat(feats, dim=-1)

        return exo[0] if is_scalar else exo

    return cb

def compose_exo_calendar_cb_yyyyww(date_type: str = "weekly", *, sincos: bool = True):
    dt = date_type.lower()
    if dt == "monthly":
        periods = [12.0]
    elif dt == "weekly":
        periods = [52.0]
    elif dt == "daily":
        periods = [7.0, 365.25]
    elif dt == "hourly":
        periods = [24.0, 168.0]
    else:
        periods = [52.0]

    def calendar_sin_cos(t, p: float):
        # t: (B,H) float
        ang = 2.0 * np.pi * (t / float(p))
        ang = torch.as_tensor(ang, device=t.device, dtype=t.dtype)
        return torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)  # (B,H,2)

    def cb(start_idx, H: int, device: str = "cpu"):
        H = int(H)
        dev = torch.device(device)

        is_scalar = isinstance(start_idx, (int, np.integer))
        if is_scalar:
            s_raw = np.array([int(start_idx)], dtype=np.int64)
        else:
            s_raw = np.asarray(start_idx, dtype=np.int64).reshape(-1)

        s_ord = np.array([yyyyww_to_week_ordinal(v) for v in s_raw], dtype=np.int64)

        s = torch.as_tensor(s_ord, device=dev, dtype=torch.float32)  # (B,)
        offs = torch.arange(0, H, device=dev, dtype=torch.float32)   # (H,)
        t = s[:, None] + offs[None, :]                                # (B,H)

        feats = []
        if sincos:
            for p in periods:
                feats.append(calendar_sin_cos(t, p))                  # (B,H,2)
            exo = torch.cat(feats, dim=-1)                            # (B,H,2*len(periods))
        else:
            for p in periods:
                feats.append(((t % float(p)) / float(p)).unsqueeze(-1))
            exo = torch.cat(feats, dim=-1)

        return exo[0] if is_scalar else exo

    return cb

def apply_exo_shift_linear(
    head: nn.Module,
    future_exo: torch.Tensor,  # (B,H,E) or (H,E)
    *,
    horizon: int,
    out_dtype=None,
    out_device=None,
) -> torch.Tensor:  # (B,H)

    # head 기준 device/dtype (학습 안정성 때문에 head dtype으로 계산)
    p = next(head.parameters(), None)
    head_device = p.device if p is not None else future_exo.device
    head_dtype  = p.dtype  if p is not None else future_exo.dtype

    ex = future_exo
    if ex.dim() == 2:  # (H,E) -> (1,H,E)
        ex = ex.unsqueeze(0)

    # (중요) head.to(...)를 여기서 하지 마세요. model.to(device)로 한번에 맞추는 게 정석입니다.
    ex = ex.to(device=head_device, dtype=head_dtype, non_blocking=True)

    ex = head(ex).squeeze(-1)  # (B,H)

    # pad/trim
    B, Hx = ex.shape
    if Hx < horizon:
        ex = torch.cat([ex, ex.new_zeros((B, horizon - Hx))], dim=1)
    elif Hx > horizon:
        ex = ex[:, :horizon]

    # 출력 dtype/device로 정렬 (필요할 때만)
    if (out_device is not None) or (out_dtype is not None):
        ex = ex.to(device=(out_device or ex.device), dtype=(out_dtype or ex.dtype), non_blocking=True)

    return ex


def compose_exo_calendar_age_warranty_cb(
        *,
        date_type: Literal['W', 'M'] = 'W',
        use_sincos: bool = True,
        use_age: bool = True,
        use_warranty: bool = True,
        wty_month: Optional[float] = None,
        age_origin_idx: Optional[int] = None,
        age_norm_mode: Literal['H', 'const', 'none'] = 'H',
        age_norm_div: Optional[float] = None,
        include_in_warranty_flag: bool = True,
        include_time_to_warranty_end: bool = True,
) -> callable:
    """
    (기존 유지) Warranty 관련 로직은 주간/월간 위주로 설계됨.
    Daily/Hourly 지원이 필요하다면 별도 확장이 필요하지만,
    현재 요청 범위(캘린더 주기성)에는 영향을 주지 않으므로 기존 로직을 유지합니다.
    """
    if date_type == 'W':
        period = 52

        def _wty_units(months: float) -> float:
            return float(months) * 4.345
    elif date_type == 'M':
        period = 12

        def _wty_units(months: float) -> float:
            return float(months)
    else:
        # D/H 등 미지원 타입이 들어오면 에러 방지를 위해 기본 W 처리하거나 에러 발생
        # 여기서는 안전하게 W로 폴백하지 않고 에러 유지
        raise ValueError("compose_exo_calendar_age_warranty_cb currently supports only 'W' or 'M'.")

    def _normalize_age(age: torch.Tensor, H: int) -> torch.Tensor:
        if age_norm_mode == 'H':
            denom = float(max(1, H))
            return age / denom
        elif age_norm_mode == 'const':
            denom = float(age_norm_div) if (age_norm_div is not None) else 100.0
            return age / max(1.0, denom)
        else:
            return age

    def cb(start_idx: int, H: int, device='cuda' if torch.cuda.is_available() else 'mps') -> torch.Tensor:
        t = torch.arange(start_idx, start_idx + H, device=device, dtype=torch.float32)
        feats = []

        # 1) sin/cos
        if use_sincos:
            feats.append(torch.sin(2 * torch.pi * t / period).unsqueeze(-1))
            feats.append(torch.cos(2 * torch.pi * t / period).unsqueeze(-1))

        # 2) age (sequence)
        if use_age:
            if age_origin_idx is None:
                age = t
            else:
                age = t - float(age_origin_idx)
                age = torch.clamp(age, min=0.0)
            age = _normalize_age(age, H).unsqueeze(-1)
            feats.append(age)

        # 3) warranty
        if use_warranty and (wty_month is not None):
            w_units = _wty_units(wty_month)
            if age_origin_idx is None:
                age_raw = t
            else:
                age_raw = torch.clamp(t - float(age_origin_idx), min=0.0)

            if include_in_warranty_flag:
                in_wty = (age_raw < w_units).to(torch.float32).unsqueeze(-1)
                feats.append(in_wty)

            if include_time_to_warranty_end:
                rem = torch.clamp(w_units - age_raw, min=0.0)
                rem_norm = (rem / max(1.0, float(w_units))).unsqueeze(-1)
                feats.append(rem_norm)

        if not feats:
            return torch.zeros(H, 0, device=device, dtype=torch.float32)
        return torch.cat(feats, dim=-1)

    return cb

from typing import Optional, List, Dict, Tuple, Sequence, Callable, Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta

from modeling_module.utils.date_util import DateUtil


# -----------------------------
# Utility
# -----------------------------
def _to_numpy(x):
    """
    입력을 NumPy 배열로 변환합니다.
    Polars Series가 입력될 경우 호환성을 위해 .to_numpy()를 호출합니다.
    """
    if isinstance(x, pl.Series):
        return x.to_numpy()
    return np.asarray(x)


def identity_date_indexer(x: int) -> int:
    """입력된 값을 변환 없이 그대로 반환하는 식별 함수(Identity Function)."""
    return int(x)


def _add_time(dt_int: int, amount: int, freq: str) -> int:
    """
    정수형 날짜 포맷(YYYYMM, YYYYMMDD, YYYYWW 등)에 시간을 더하거나 뺍니다.
    """
    s = str(dt_int)

    if freq == "hourly":
        fmt = "%Y%m%d%H"
        dt_obj = datetime.strptime(s, fmt)
        new_dt = dt_obj + timedelta(hours=amount)
        return int(new_dt.strftime(fmt))

    if freq == "daily":
        fmt = "%Y%m%d"
        dt_obj = datetime.strptime(s, fmt)
        new_dt = dt_obj + timedelta(days=amount)
        return int(new_dt.strftime(fmt))

    if freq == "weekly":
        if DateUtil:
            return DateUtil.add_weeks_yyyyww(dt_int, amount)
        raise ImportError("Weekly logic requires DateUtil module.")

    if freq == "monthly":
        if DateUtil:
            return DateUtil.add_months_yyyymm(dt_int, amount)

        y = dt_int // 100
        m = dt_int % 100
        m += amount
        while m < 1:
            m += 12
            y -= 1
        while m > 12:
            m -= 12
            y += 1
        return y * 100 + m

    return dt_int


def _generate_time_seq(plan_dt: int, length: int, freq: str) -> np.ndarray:
    """
    기준 날짜(plan_dt) '직전'부터 과거로 length 만큼의 날짜 시퀀스를 생성합니다.
    반환은 과거 -> 최근 오름차순.
    """
    seq = []
    current = _add_time(plan_dt, -1, freq)
    for _ in range(length):
        seq.append(current)
        current = _add_time(current, -1, freq)
    return np.asarray(seq[::-1], dtype=np.int64)


# ============================================================
# 1) Training Dataset (index_map 기반)
# ============================================================
class MultiPartExoTrainingDataset(Dataset):
    """
    슬라이딩 윈도우(Sliding Window) 학습용 Dataset.

    성능 최적화 포인트:
      - __init__에서 dtype을 확정(float32/int64) → __getitem__에서 .to(dtype) 호출 제거
      - __getitem__에서는 torch.from_numpy(...)만 사용(복사 최소화)
      - 불필요한 np.zeros 생성 제거(빈 feature는 (T,0) 텐서 슬라이스로 처리)
    """

    def __init__(
        self,
        df: pl.DataFrame,
        lookback: int,
        horizon: int,
        freq: str = "weekly",
        *,
        id_col: str = "unique_id",
        date_col: str = "date",
        qty_col: str = "y",
        past_exo_cont_cols: Optional[Sequence[str]] = None,
        past_exo_cat_cols: Optional[Sequence[str]] = None,
        future_exo_cb: Optional[Callable] = None,  # kept for signature compatibility
        date_indexer: Optional[Callable[[int], int]] = None,
        cat_indexers: Optional[Dict[str, Any]] = None,
    ):
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.freq = str(freq).lower()

        self.id_col = id_col
        self.date_col = date_col
        self.qty_col = qty_col

        self.past_exo_cont_cols = list(past_exo_cont_cols) if past_exo_cont_cols else []
        self.past_exo_cat_cols = list(past_exo_cat_cols) if past_exo_cat_cols else []

        self.future_exo_cb = future_exo_cb
        self.date_indexer = date_indexer or identity_date_indexer
        self.cat_indexers = cat_indexers or {}

        self.series: Dict[str, Dict[str, np.ndarray]] = {}
        self.index_map: List[Tuple[str, int]] = []
        self.id_to_indices: Dict[str, List[int]] = {}

        if self.id_col not in df.columns:
            raise KeyError(f"id_col='{self.id_col}' not found in df.columns")
        if self.date_col not in df.columns:
            raise KeyError(f"date_col='{self.date_col}' not found in df.columns")
        if self.qty_col not in df.columns:
            raise KeyError(f"qty_col='{self.qty_col}' not found in df.columns")

        for g in df.partition_by(self.id_col):
            g = g.sort(self.date_col)
            uid = str(g[self.id_col][0])

            # dtype 확정 (핵심)
            y_all = _to_numpy(g[self.qty_col]).astype(np.float32, copy=False)   # [T] float32
            d_all = _to_numpy(g[self.date_col]).astype(np.int64, copy=False)   # [T] int64

            T = len(y_all)
            if T < self.lookback + self.horizon:
                continue

            # Past continuous exo: [T, E_cont] float32
            cont_list: List[np.ndarray] = []
            for col in self.past_exo_cont_cols:
                if col in g.columns:
                    cont_list.append(_to_numpy(g[col]).astype(np.float32, copy=False))
            exo_cont = (
                np.stack(cont_list, axis=-1).astype(np.float32, copy=False)
                if cont_list else np.zeros((T, 0), dtype=np.float32)
            )

            # Past categorical exo: [T, E_cat] int64
            cat_list: List[np.ndarray] = []
            for col in self.past_exo_cat_cols:
                if col not in g.columns:
                    continue
                s = g[col]
                if s.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                    cat_list.append(_to_numpy(s).astype(np.int64, copy=False))
                else:
                    if col not in self.cat_indexers:
                        raise TypeError(f"Categorical '{col}' needs a CategoryIndexer or integer IDs.")
                    mapped = self.cat_indexers[col].map_series(s)
                    cat_list.append(_to_numpy(mapped).astype(np.int64, copy=False))

            exo_cat = (
                np.stack(cat_list, axis=-1).astype(np.int64, copy=False)
                if cat_list else np.zeros((T, 0), dtype=np.int64)
            )

            self.series[uid] = {"y": y_all, "d": d_all, "exo_cont": exo_cont, "exo_cat": exo_cat}

            n_windows = T - self.lookback - self.horizon + 1
            if n_windows <= 0:
                continue

            self.id_to_indices[uid] = []
            for i in range(n_windows):
                gidx = len(self.index_map)
                self.index_map.append((uid, i))
                self.id_to_indices[uid].append(gidx)

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        uid, i = self.index_map[idx]
        pack = self.series[uid]

        y_all: np.ndarray = pack["y"]           # float32
        d_all: np.ndarray = pack["d"]           # int64
        exo_cont: np.ndarray = pack["exo_cont"] # float32, (T, E_cont)
        exo_cat: np.ndarray = pack["exo_cat"]   # int64,   (T, E_cat)

        L = self.lookback
        H = self.horizon

        x_win = y_all[i:i + L]               # (L,) float32
        y_win = y_all[i + L:i + L + H]       # (H,) float32
        pe_cont = exo_cont[i:i + L]          # (L, E_cont) float32
        pe_cat = exo_cat[i:i + L]            # (L, E_cat)  int64

        last_dt = int(d_all[i + L - 1])
        next_dt = _add_time(last_dt, 1, self.freq)
        start_idx = int(self.date_indexer(next_dt))

        x = torch.from_numpy(x_win).unsqueeze(-1)      # float32
        y = torch.from_numpy(y_win)                    # float32
        pe_cont_t = torch.from_numpy(pe_cont)          # float32
        pe_cat_t = torch.from_numpy(pe_cat)            # int64 == torch.long

        return x, y, uid, start_idx, pe_cont_t, pe_cat_t


# ============================================================
# 2) Inference Dataset (Unified for Monthly/Weekly/Daily/Hourly)
# ============================================================
class MultiPartExoAnchoredInferenceDataset(Dataset):
    """
    특정 시점(plan_dt)을 기준으로 과거 데이터를 조회하여 추론 입력을 생성하는 Dataset.

    성능 최적화 포인트:
      - __init__에서 numpy dtype을 확정(float32/int64) 후 저장
      - __getitem__에서 torch.tensor(...) 금지 → torch.from_numpy(...)로 교체
      - y_dummy는 __init__에서 1회 생성 후 재사용
    """

    def __init__(
        self,
        df: pl.DataFrame,
        lookback: int,
        horizon: int,
        plan_dt: int,
        freq: str,
        *,
        id_col: str = "unique_id",
        date_col: str = "date",
        qty_col: str = "y",
        past_exo_cont_cols: Optional[Sequence[str]] = None,
        past_exo_cat_cols: Optional[Sequence[str]] = None,
        fill_missing: str = "ffill",
        target_back_steps: int = 100,
        future_exo_cb: Optional[Callable] = None,
        date_indexer: Optional[Callable[[int], int]] = None,
        cat_indexers: Optional[Dict[str, Any]] = None,
    ):
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.plan_dt = int(plan_dt)
        self.freq = freq.lower()

        self.id_col = id_col
        self.date_col = date_col
        self.qty_col = qty_col

        self.past_exo_cont_cols = list(past_exo_cont_cols) if past_exo_cont_cols else []
        self.past_exo_cat_cols = list(past_exo_cat_cols) if past_exo_cat_cols else []

        self.fill_missing = fill_missing
        self.target_back_steps = int(target_back_steps)
        self.future_exo_cb = future_exo_cb
        self.date_indexer = date_indexer or identity_date_indexer
        self.cat_indexers = cat_indexers or {}

        self.y_dummy = torch.zeros((self.horizon,), dtype=torch.float32)

        self.start_idxs: List[int] = []
        self.inputs: List[np.ndarray] = []
        self.ids: List[str] = []
        self.past_exo_conts: List[np.ndarray] = []
        self.past_exo_cats: List[np.ndarray] = []
        self.future_exo_conts: List[np.ndarray] = []

        win_dates = _generate_time_seq(self.plan_dt, self.lookback, self.freq)

        for g in df.partition_by(self.id_col):
            uid = str(g[self.id_col][0])

            dts = _to_numpy(g[self.date_col]).astype(np.int64, copy=False)
            vals = _to_numpy(g[self.qty_col]).astype(np.float32, copy=False)
            if len(dts) == 0:
                continue

            qty_map = {int(d): float(v) for d, v in zip(dts, vals)}
            earliest = int(dts.min())

            # 1) target x
            x = np.empty(self.lookback, dtype=np.float32)
            for j, curr_dt in enumerate(win_dates):
                if curr_dt in qty_map:
                    x[j] = np.float32(qty_map[curr_dt])
                    continue

                if self.fill_missing == "zero":
                    x[j] = np.float32(0.0)
                    continue
                if self.fill_missing == "nan":
                    x[j] = np.float32(np.nan)
                    continue

                prev, found = curr_dt, False
                for _ in range(self.target_back_steps):
                    prev = _add_time(prev, -1, self.freq)
                    if prev < earliest:
                        break
                    if prev in qty_map:
                        x[j] = np.float32(qty_map[prev])
                        found = True
                        break
                if not found:
                    x[j] = np.float32(0.0)

            if self.fill_missing == "nan" and not np.any(np.isfinite(x)):
                continue

            # 2) past cont
            pe_cont_list: List[np.ndarray] = []
            for col in self.past_exo_cont_cols:
                if col not in g.columns:
                    continue
                col_vals = _to_numpy(g[col]).astype(np.float32, copy=False)
                val_map = {int(d): float(v) for d, v in zip(dts, col_vals)}

                e = np.empty(self.lookback, dtype=np.float32)
                for j, curr_dt in enumerate(win_dates):
                    if curr_dt in val_map:
                        e[j] = np.float32(val_map[curr_dt])
                        continue

                    if self.fill_missing == "zero":
                        e[j] = np.float32(0.0)
                        continue
                    if self.fill_missing == "nan":
                        e[j] = np.float32(np.nan)
                        continue

                    prev, found = curr_dt, False
                    for _ in range(self.target_back_steps):
                        prev = _add_time(prev, -1, self.freq)
                        if prev < earliest:
                            break
                        if prev in val_map:
                            e[j] = np.float32(val_map[prev])
                            found = True
                            break
                    if not found:
                        e[j] = np.float32(0.0)

                pe_cont_list.append(e)

            pe_cont_mat = (
                np.stack(pe_cont_list, axis=-1).astype(np.float32, copy=False)
                if pe_cont_list else np.zeros((self.lookback, 0), dtype=np.float32)
            )

            # 3) past cat
            pe_cat_list: List[np.ndarray] = []
            for col in self.past_exo_cat_cols:
                if col not in g.columns:
                    continue
                s = g[col]

                if s.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                    vals_int = _to_numpy(s).astype(np.int64, copy=False)
                    unk = 0
                else:
                    idxr = self.cat_indexers.get(col, None)
                    if idxr is None:
                        vals_int = np.zeros(len(s), dtype=np.int64)
                        unk = 0
                    else:
                        vals_int = np.array([idxr.id_of(v) for v in s.to_list()], dtype=np.int64)
                        unk = int(getattr(idxr, "unk_id", 0))

                val_map = {int(d): int(v) for d, v in zip(dts, vals_int)}

                e = np.empty(self.lookback, dtype=np.int64)
                for j, curr_dt in enumerate(win_dates):
                    if curr_dt in val_map:
                        e[j] = np.int64(val_map[curr_dt])
                        continue

                    if self.fill_missing in ("zero", "nan"):
                        e[j] = np.int64(unk)
                        continue

                    prev, found = curr_dt, False
                    for _ in range(self.target_back_steps):
                        prev = _add_time(prev, -1, self.freq)
                        if prev < earliest:
                            break
                        if prev in val_map:
                            e[j] = np.int64(val_map[prev])
                            found = True
                            break
                    if not found:
                        e[j] = np.int64(unk)

                pe_cat_list.append(e)

            pe_cat_mat = (
                np.stack(pe_cat_list, axis=-1).astype(np.int64, copy=False)
                if pe_cat_list else np.zeros((self.lookback, 0), dtype=np.int64)
            )

            # 4) future exo
            last_hist = int(win_dates[-1])
            next_dt = _add_time(last_hist, 1, self.freq)
            start_idx = int(self.date_indexer(next_dt))
            self.start_idxs.append(start_idx)

            fe = np.zeros((self.horizon, 0), dtype=np.float32)
            if self.future_exo_cb is not None:
                res = self.future_exo_cb(start_idx, self.horizon, device="cpu")
                if isinstance(res, torch.Tensor):
                    fe = res.detach().cpu().to(torch.float32).numpy()
                else:
                    fe = np.asarray(res, dtype=np.float32)
                if fe.ndim != 2 or fe.shape[0] != self.horizon:
                    raise ValueError(f"future_exo_cb must return (H, E). got {fe.shape}")

            self.inputs.append(x.astype(np.float32, copy=False))
            self.past_exo_conts.append(pe_cont_mat)
            self.past_exo_cats.append(pe_cat_mat)
            self.future_exo_conts.append(fe.astype(np.float32, copy=False))
            self.ids.append(uid)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.inputs[idx]).unsqueeze(-1)  # (L,1) float32
        peC = torch.from_numpy(self.past_exo_conts[idx])      # (L,E_cont) float32
        peK = torch.from_numpy(self.past_exo_cats[idx])       # (L,E_cat) int64

        y_dummy = self.y_dummy
        start_idx = int(self.start_idxs[idx])
        uid = self.ids[idx]

        return x, y_dummy, uid, start_idx, peC, peK

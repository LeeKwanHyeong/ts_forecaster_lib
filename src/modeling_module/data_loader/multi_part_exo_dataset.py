from datetime import date, datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from modeling_module.data_loader.temporal import (
    add_period,
    lookback_periods,
    normalize_period_key,
    normalize_temporal_frame,
)


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
    """Shift a canonical integer time key by ``amount`` periods."""
    return add_period(dt_int, amount, freq)


def _generate_time_seq(plan_dt: int, length: int, freq: str) -> np.ndarray:
    """
    기준 날짜(plan_dt) '직전'부터 과거로 length 만큼의 날짜 시퀀스를 생성합니다.
    반환은 과거 -> 최근 오름차순.
    """
    return lookback_periods(plan_dt, length, freq)


def _select_series_frame(
    df: pl.DataFrame,
    id_col: str,
    series_ids: Optional[Sequence[str]],
    *,
    unknown_series_policy: Literal["error", "ignore"],
) -> pl.DataFrame:
    """Select series deterministically for anchored inference.

    ``None`` selects all series in canonical string order. Explicit IDs retain
    their first-occurrence request order and unknown IDs fail by default.
    """
    if id_col not in df.columns:
        raise KeyError(f"id_col='{id_col}' not found in df.columns")
    if unknown_series_policy not in {"error", "ignore"}:
        raise ValueError("unknown_series_policy must be 'error' or 'ignore'")

    normalized = df.with_columns(pl.col(id_col).cast(pl.String))
    available = set(normalized[id_col].unique().to_list())
    if series_ids is None:
        requested = sorted(available)
    else:
        requested = list(dict.fromkeys(str(value) for value in series_ids))
        if not requested:
            raise ValueError("series_ids must not be empty; use None to select all series")

    unknown = [series_id for series_id in requested if series_id not in available]
    if unknown and unknown_series_policy == "error":
        raise ValueError(f"Unknown series_ids: {unknown}")
    selected = [series_id for series_id in requested if series_id in available]
    if not selected:
        raise ValueError("series_ids selection contains no known series")
    return pl.concat(
        [normalized.filter(pl.col(id_col) == series_id) for series_id in selected],
        how="vertical",
    )


def _lookup_float_sequence(
    query_dates: Sequence[int],
    value_map: Dict[int, float],
    *,
    fill_missing: str,
    target_back_steps: int,
    freq: str,
    earliest: int,
) -> np.ndarray:
    """Resolve a float sequence using the configured missing-value policy."""
    output = np.empty(len(query_dates), dtype=np.float32)
    for index, current_date in enumerate(query_dates):
        value = value_map.get(int(current_date))
        if value is not None and np.isfinite(value):
            output[index] = value
            continue
        if fill_missing == "zero":
            output[index] = 0.0
            continue
        if fill_missing == "nan":
            output[index] = np.nan
            continue

        previous = int(current_date)
        found = False
        for _ in range(target_back_steps):
            previous = _add_time(previous, -1, freq)
            if previous < earliest:
                break
            previous_value = value_map.get(previous)
            if previous_value is not None and np.isfinite(previous_value):
                output[index] = previous_value
                found = True
                break
        if not found:
            output[index] = 0.0
    return output


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
        future_exo_cont_cols: Optional[Sequence[str]] = None,
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
        self.future_exo_cont_cols = list(future_exo_cont_cols) if future_exo_cont_cols else []

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

        normalized_df = normalize_temporal_frame(df, self.date_col, self.freq)
        for g in normalized_df.partition_by(self.id_col, maintain_order=True):
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

            future_cont_list: List[np.ndarray] = []
            for col in self.future_exo_cont_cols:
                if col in g.columns:
                    future_cont_list.append(_to_numpy(g[col]).astype(np.float32, copy=False))
            future_exo_cont = (
                np.stack(future_cont_list, axis=-1).astype(np.float32, copy=False)
                if future_cont_list else np.zeros((T, 0), dtype=np.float32)
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

            self.series[uid] = {
                "y": y_all,
                "d": d_all,
                "exo_cont": exo_cont,
                "exo_cat": exo_cat,
                "future_exo_cont": future_exo_cont,
            }

            n_windows = T - self.lookback - self.horizon + 1
            if n_windows <= 0:
                continue

            self.id_to_indices[uid] = []
            for i in range(n_windows):
                if not np.isfinite(y_all[i:i + self.lookback + self.horizon]).all():
                    continue
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
        future_exo_cont: np.ndarray = pack["future_exo_cont"]

        L = self.lookback
        H = self.horizon

        x_win = y_all[i:i + L]               # (L,) float32
        y_win = y_all[i + L:i + L + H]       # (H,) float32
        pe_cont = exo_cont[i:i + L]          # (L, E_cont) float32
        pe_cat = exo_cat[i:i + L]            # (L, E_cat)  int64
        fe_cont = future_exo_cont[i + L:i + L + H]

        last_dt = int(d_all[i + L - 1])
        next_dt = _add_time(last_dt, 1, self.freq)
        start_idx = int(self.date_indexer(next_dt))

        x = torch.from_numpy(x_win).unsqueeze(-1)      # float32
        y = torch.from_numpy(y_win)                    # float32
        pe_cont_t = torch.from_numpy(pe_cont)          # float32
        pe_cat_t = torch.from_numpy(pe_cat)            # int64 == torch.long

        future_payload: int | torch.Tensor = start_idx
        if fe_cont.shape[-1] > 0:
            future_payload = torch.from_numpy(fe_cont)
        return x, y, uid, future_payload, pe_cont_t, pe_cat_t


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
        plan_dt: date | datetime | int,
        freq: str,
        *,
        id_col: str = "unique_id",
        date_col: str = "date",
        qty_col: str = "y",
        past_exo_cont_cols: Optional[Sequence[str]] = None,
        past_exo_cat_cols: Optional[Sequence[str]] = None,
        future_exo_cont_cols: Optional[Sequence[str]] = None,
        series_ids: Optional[Sequence[str]] = None,
        unknown_series_policy: Literal["error", "ignore"] = "error",
        fill_missing: str = "ffill",
        target_back_steps: int = 100,
        future_exo_cb: Optional[Callable] = None,
        date_indexer: Optional[Callable[[int], int]] = None,
        cat_indexers: Optional[Dict[str, Any]] = None,
    ):
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.freq = freq.lower()
        self.plan_dt = normalize_period_key(plan_dt, self.freq)

        self.id_col = id_col
        self.date_col = date_col
        self.qty_col = qty_col

        self.past_exo_cont_cols = list(past_exo_cont_cols) if past_exo_cont_cols else []
        self.past_exo_cat_cols = list(past_exo_cat_cols) if past_exo_cat_cols else []
        self.future_exo_cont_cols = list(future_exo_cont_cols) if future_exo_cont_cols else []

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

        normalized_df = normalize_temporal_frame(df, self.date_col, self.freq)
        selected_df = _select_series_frame(
            normalized_df,
            self.id_col,
            series_ids,
            unknown_series_policy=unknown_series_policy,
        )
        win_dates = _generate_time_seq(self.plan_dt, self.lookback, self.freq)

        for g in selected_df.partition_by(self.id_col, maintain_order=True):
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

            future_dates = np.asarray(
                [_add_time(self.plan_dt, step, self.freq) for step in range(self.horizon)],
                dtype=np.int64,
            )
            future_cont_list: List[np.ndarray] = []
            for col in self.future_exo_cont_cols:
                if col not in g.columns:
                    continue
                values = _to_numpy(g[col]).astype(np.float32, copy=False)
                value_map = {int(dt): float(value) for dt, value in zip(dts, values)}
                future_cont_list.append(
                    _lookup_float_sequence(
                        future_dates,
                        value_map,
                        fill_missing=self.fill_missing,
                        target_back_steps=self.target_back_steps,
                        freq=self.freq,
                        earliest=earliest,
                    )
                )
            fe = (
                np.stack(future_cont_list, axis=-1).astype(np.float32, copy=False)
                if future_cont_list else np.zeros((self.horizon, 0), dtype=np.float32)
            )

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

        fe = torch.from_numpy(self.future_exo_conts[idx])
        future_payload: int | torch.Tensor = fe if fe.shape[-1] > 0 else start_idx
        return x, y_dummy, uid, future_payload, peC, peK

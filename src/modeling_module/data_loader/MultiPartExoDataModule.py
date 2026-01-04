import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler, Subset
import numpy as np
from typing import Callable, Optional, Sequence, Dict, Any, Iterable
from collections import defaultdict
from datetime import datetime, timedelta

try:
    from modeling_module.utils.date_util import DateUtil
except ImportError:
    DateUtil = None


# -----------------------------
# 유틸
# -----------------------------
def _to_numpy(x):
    if isinstance(x, pl.Series):
        return x.to_numpy()
    return np.asarray(x)


def _add_time(dt_int: int, amount: int, freq: str) -> int:
    s = str(dt_int)

    if freq == 'hourly':
        fmt = "%Y%m%d%H"
        dt_obj = datetime.strptime(s, fmt)
        new_dt = dt_obj + timedelta(hours=amount)
        return int(new_dt.strftime(fmt))

    elif freq == 'daily':
        fmt = "%Y%m%d"
        dt_obj = datetime.strptime(s, fmt)
        new_dt = dt_obj + timedelta(days=amount)
        return int(new_dt.strftime(fmt))

    elif freq == 'weekly':
        if DateUtil:
            return DateUtil.add_weeks_yyyyww(dt_int, amount)
        raise ImportError("Weekly logic requires DateUtil module.")

    elif freq == 'monthly':
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
    seq = []
    current = _add_time(plan_dt, -1, freq)
    for _ in range(length):
        seq.append(current)
        current = _add_time(current, -1, freq)
    return np.array(seq[::-1], dtype=np.int64)


class CategoryIndexer:
    def __init__(self, mapping: Optional[Dict[Any, int]] = None):
        self.unk_id = 0
        self.mapping: Dict[Any, int] = mapping or {}

    @staticmethod
    def build_from_series(series: pl.Series, sort: bool = True) -> "CategoryIndexer":
        vals = series.drop_nulls().unique().to_list()
        if sort:
            try:
                vals = sorted(vals)
            except Exception:
                pass
        mapping = {}
        next_id = 1
        for v in vals:
            if v not in mapping:
                mapping[v] = next_id
                next_id += 1
        return CategoryIndexer(mapping)

    def id_of(self, value: Any) -> int:
        return self.mapping.get(value, self.unk_id)

    def map_series(self, s: pl.Series) -> np.ndarray:
        return np.asarray([self.id_of(v) for v in s.to_list()], dtype=np.int64)


# ============================================================
# 1) Training Dataset (Lifecycle meta 포함)
# ============================================================
class MultiPartExoTrainingDataset(Dataset):
    """
    슬라이딩 윈도우 학습 Dataset.
    - 샘플별 메타: part_id, last_dt, (옵션) lifecycle 값을 함께 저장
    """

    def __init__(
        self,
        df: pl.DataFrame,
        lookback: int,
        horizon: int,
        *,
        part_col: str = "part_no",
        date_col: str = "demand_dt",
        qty_col: str = "demand_qty",
        past_exo_cont_cols: Optional[Sequence[str]] = None,
        past_exo_cat_cols: Optional[Sequence[str]] = None,
        future_exo_cb: Optional[Callable[[int, int, str], np.ndarray | torch.Tensor]] = None,
        date_indexer: Optional[Callable[[int], int]] = None,
        cat_indexers: Optional[Dict[str, CategoryIndexer]] = None,
        lifecycle_col: Optional[str] = None,  # NEW
    ):
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.part_col = part_col
        self.date_col = date_col
        self.qty_col = qty_col

        self.past_exo_cont_cols = list(past_exo_cont_cols) if past_exo_cont_cols else []
        self.past_exo_cat_cols = list(past_exo_cat_cols) if past_exo_cat_cols else []

        self.future_exo_cb = future_exo_cb
        self.date_indexer = date_indexer or (lambda x: x)
        self.cat_indexers = cat_indexers or {}
        self.lifecycle_col = lifecycle_col

        self.samples: list[dict[str, Any]] = []

        grouped = df.partition_by(part_col)
        for g in grouped:
            g = g.sort(date_col)
            part = g[part_col][0]

            y_all = _to_numpy(g[qty_col]).astype(float)
            d_all = _to_numpy(g[date_col]).astype(np.int64)
            T = len(y_all)
            if T < self.lookback + self.horizon:
                continue

            # lifecycle 시계열(옵션)
            life_all = None
            if self.lifecycle_col and (self.lifecycle_col in g.columns):
                life_all = g[self.lifecycle_col].to_list()

            # ----- 연속형 past exo -----
            if self.past_exo_cont_cols:
                cont_list = []
                for col in self.past_exo_cont_cols:
                    if col not in g.columns:
                        continue
                    cont_list.append(_to_numpy(g[col]).astype(float))
                exo_cont_mat = np.stack(cont_list, axis=-1) if cont_list else np.zeros((T, 0), dtype=float)
            else:
                exo_cont_mat = np.zeros((T, 0), dtype=float)

            # ----- 범주형 past exo -----
            if self.past_exo_cat_cols:
                cat_list = []
                for col in self.past_exo_cat_cols:
                    if col not in g.columns:
                        continue
                    s = g[col]
                    if s.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                        cat_list.append(_to_numpy(s).astype(np.int64))
                    else:
                        if col not in self.cat_indexers:
                            raise TypeError(f"Categorical '{col}' needs a CategoryIndexer or integer IDs.")
                        cat_list.append(self.cat_indexers[col].map_series(s))
                exo_cat_mat = np.stack(cat_list, axis=-1) if cat_list else np.zeros((T, 0), dtype=np.int64)
            else:
                exo_cat_mat = np.zeros((T, 0), dtype=np.int64)

            # ----- 윈도우 생성 -----
            for i in range(T - self.lookback - self.horizon + 1):
                x_win = y_all[i:i + self.lookback]
                y_win = y_all[i + self.lookback:i + self.lookback + self.horizon]

                p_cont = exo_cont_mat[i:i + self.lookback, :] if exo_cont_mat.size else np.zeros((self.lookback, 0), dtype=float)
                p_cat = exo_cat_mat[i:i + self.lookback, :] if exo_cat_mat.size else np.zeros((self.lookback, 0), dtype=np.int64)

                last_dt = int(d_all[i + self.lookback - 1])
                start_idx = int(self.date_indexer(last_dt)) + 1

                fe = np.zeros((self.horizon, 0), dtype=float)
                if self.future_exo_cb is not None:
                    res = self.future_exo_cb(start_idx, self.horizon, device="cpu")
                    fe = res.detach().cpu().numpy() if isinstance(res, torch.Tensor) else np.asarray(res, dtype=float)

                lifecycle_val = None
                if life_all is not None:
                    lifecycle_val = life_all[i + self.lookback - 1]

                self.samples.append(dict(
                    x=x_win, y=y_win,
                    past_exo_cont=p_cont, past_exo_cat=p_cat,
                    future_exo_cont=fe,
                    part_id=part,
                    last_dt=last_dt,                 # NEW
                    lifecycle=lifecycle_val,         # NEW
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = torch.tensor(s["x"], dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(s["y"], dtype=torch.float32)
        pe_cont = torch.tensor(s["past_exo_cont"], dtype=torch.float32)
        pe_cat = torch.tensor(s["past_exo_cat"], dtype=torch.long)
        fe_cont = torch.tensor(s["future_exo_cont"], dtype=torch.float32)
        return x, y, s["part_id"], fe_cont, pe_cont, pe_cat


# ============================================================
# 2) Inference Dataset (그대로)
# ============================================================
class MultiPartExoAnchoredInferenceDataset(Dataset):
    def __init__(
        self,
        df: pl.DataFrame,
        lookback: int,
        horizon: int,
        plan_dt: int,
        freq: str,
        *,
        part_col: str = "part_no",
        date_col: str = "demand_dt",
        qty_col: str = "demand_qty",
        past_exo_cont_cols: Optional[Sequence[str]] = None,
        past_exo_cat_cols: Optional[Sequence[str]] = None,
        fill_missing: str = "ffill",
        target_back_steps: int = 100,
        future_exo_cb: Optional[Callable[[int, int, str], np.ndarray | torch.Tensor]] = None,
        date_indexer: Optional[Callable[[int], int]] = None,
        cat_indexers: Optional[Dict[str, CategoryIndexer]] = None,
    ):
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.plan_dt = int(plan_dt)
        self.freq = freq.lower()

        self.part_col = part_col
        self.date_col = date_col
        self.qty_col = qty_col

        self.past_exo_cont_cols = list(past_exo_cont_cols) if past_exo_cont_cols else []
        self.past_exo_cat_cols = list(past_exo_cat_cols) if past_exo_cat_cols else []

        self.fill_missing = fill_missing
        self.target_back_steps = int(target_back_steps)
        self.future_exo_cb = future_exo_cb
        self.date_indexer = date_indexer or (lambda x: x)
        self.cat_indexers = cat_indexers or {}

        self.inputs, self.part_ids = [], []
        self.past_exo_conts, self.past_exo_cats = [], []
        self.future_exo_conts = []

        win_dates = _generate_time_seq(self.plan_dt, self.lookback, self.freq)

        grouped = df.partition_by(part_col)
        for g in grouped:
            part = g[part_col][0]

            dts = _to_numpy(g[date_col]).astype(np.int64)
            vals = _to_numpy(g[qty_col]).astype(float)
            if len(dts) == 0:
                continue

            qty_map = {int(d): float(v) for d, v in zip(dts, vals)}
            earliest = int(dts.min())

            x = np.empty(self.lookback, dtype=float)
            for i, curr_dt in enumerate(win_dates):
                if curr_dt in qty_map:
                    x[i] = qty_map[curr_dt]
                else:
                    if self.fill_missing == "zero":
                        x[i] = 0.0
                    elif self.fill_missing == "nan":
                        x[i] = np.nan
                    else:
                        prev, found = curr_dt, False
                        for _ in range(self.target_back_steps):
                            prev = _add_time(prev, -1, self.freq)
                            if prev < earliest:
                                break
                            if prev in qty_map:
                                x[i] = qty_map[prev]
                                found = True
                                break
                        if not found:
                            x[i] = 0.0

            if self.fill_missing == "nan" and not np.any(np.isfinite(x)):
                continue

            # cont past exo
            pe_cont_list = []
            for col in self.past_exo_cont_cols:
                if col not in g.columns:
                    continue
                val_map = {int(d): float(v) for d, v in zip(dts, _to_numpy(g[col]).astype(float))}
                e = np.empty(self.lookback, dtype=float)
                for i, curr_dt in enumerate(win_dates):
                    if curr_dt in val_map:
                        e[i] = val_map[curr_dt]
                    else:
                        if self.fill_missing == "zero":
                            e[i] = 0.0
                        elif self.fill_missing == "nan":
                            e[i] = np.nan
                        else:
                            prev, found = curr_dt, False
                            for _ in range(self.target_back_steps):
                                prev = _add_time(prev, -1, self.freq)
                                if prev < earliest:
                                    break
                                if prev in val_map:
                                    e[i] = val_map[prev]
                                    found = True
                                    break
                            if not found:
                                e[i] = 0.0
                pe_cont_list.append(e)
            pe_cont_mat = np.stack(pe_cont_list, axis=-1) if pe_cont_list else np.zeros((self.lookback, 0), dtype=float)

            # cat past exo
            pe_cat_list = []
            for col in self.past_exo_cat_cols:
                if col not in g.columns:
                    continue
                s = g[col]
                if s.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                    vals_int = _to_numpy(s).astype(np.int64)
                    unk = 0
                else:
                    if col not in self.cat_indexers:
                        unk = 0
                        vals_int = np.zeros(len(s), dtype=np.int64)
                    else:
                        idxr = self.cat_indexers[col]
                        vals_int = np.array([idxr.id_of(v) for v in s.to_list()], dtype=np.int64)
                        unk = idxr.unk_id

                val_map = {int(d): int(v) for d, v in zip(dts, vals_int)}
                e = np.empty(self.lookback, dtype=np.int64)
                for i, curr_dt in enumerate(win_dates):
                    if curr_dt in val_map:
                        e[i] = val_map[curr_dt]
                    else:
                        if self.fill_missing in ("zero", "nan"):
                            e[i] = unk
                        else:
                            prev, found = curr_dt, False
                            for _ in range(self.target_back_steps):
                                prev = _add_time(prev, -1, self.freq)
                                if prev < earliest:
                                    break
                                if prev in val_map:
                                    e[i] = val_map[prev]
                                    found = True
                                    break
                            if not found:
                                e[i] = unk
                pe_cat_list.append(e)
            pe_cat_mat = np.stack(pe_cat_list, axis=-1) if pe_cat_list else np.zeros((self.lookback, 0), dtype=np.int64)

            # future exo
            last_hist = int(win_dates[-1])
            start_idx = int(self.date_indexer(last_hist)) + 1
            fe = np.zeros((self.horizon, 0), dtype=float)
            if self.future_exo_cb is not None:
                res = self.future_exo_cb(start_idx, self.horizon, device="cpu")
                fe = res.detach().cpu().numpy() if isinstance(res, torch.Tensor) else np.asarray(res, dtype=float)

            self.inputs.append(x)
            self.past_exo_conts.append(pe_cont_mat)
            self.past_exo_cats.append(pe_cat_mat)
            self.future_exo_conts.append(fe)
            self.part_ids.append(part)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32).unsqueeze(-1)
        peC = torch.tensor(self.past_exo_conts[idx], dtype=torch.float32)
        peK = torch.tensor(self.past_exo_cats[idx], dtype=torch.long)
        feC = torch.tensor(self.future_exo_conts[idx], dtype=torch.float32)
        return x, self.part_ids[idx], feC, peC, peK


# ============================================================
# 3) DataModule (Weighted Sampling + part별 loader 추가)
# ============================================================
class MultiPartExoDataModule:
    """
    추가된 기능:
    - lifecycle 기반 WeightedRandomSampler 지원
    - part_id 별 DataLoader 생성 (train/val/inference)
    """

    def __init__(
        self,
        df: pl.DataFrame,
        lookback: int,
        horizon: int,
        *,
        freq: str = 'weekly',
        batch_size: int = 32,
        val_ratio: float = 0.2,
        shuffle: bool = False,
        seed: int = 42,
        part_col: str = "unique_id",
        date_col: str = "date",
        qty_col: str = "HUFL",
        past_exo_cont_cols: Optional[Sequence[str]] = None,
        past_exo_cat_cols: Optional[Sequence[str]] = None,
        fill_missing: str = "ffill",
        future_exo_cb: Optional[Callable[[int, int, str], np.ndarray | torch.Tensor]] = None,
        date_indexer: Optional[Callable[[int], int]] = None,
        build_cat_indexer_from: Optional[Sequence[str]] = None,
        cat_indexer_target_col: Optional[str] = None,

        # ---- NEW: Weighted sampling ----
        use_weighted_sampling: bool = False,
        lifecycle_col: Optional[str] = None,                         # df에 존재하는 lifecycle 컬럼명
        lifecycle_weight_map: Optional[Dict[Any, float]] = None,     # lifecycle 값 -> weight
        part_balance_alpha: float = 0.0,                             # 0이면 비활성, >0이면 희소 part 가중
        custom_sample_weight_fn: Optional[Callable[[dict], float]] = None,  # 완전 커스텀 가중치 함수
        min_weight: float = 1e-6,
    ):
        self.df = df
        self.lookback = int(lookback)
        self.horizon = int(horizon)

        valid_freqs = ('monthly', 'weekly', 'daily', 'hourly')
        if freq not in valid_freqs:
            raise ValueError(f"freq must be one of {valid_freqs}, got '{freq}'")
        self.freq = freq

        self.batch_size = int(batch_size)
        self.val_ratio = float(val_ratio)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

        self.part_col = part_col
        self.date_col = date_col
        self.qty_col = qty_col

        self.past_exo_cont_cols = list(past_exo_cont_cols) if past_exo_cont_cols else []
        self.past_exo_cat_cols = list(past_exo_cat_cols) if past_exo_cat_cols else []

        self.fill_missing = fill_missing
        self.future_exo_cb = future_exo_cb
        self.date_indexer = date_indexer or (lambda x: x)

        self.cat_indexers: Dict[str, CategoryIndexer] = {}

        # NEW: sampling 설정
        self.use_weighted_sampling = bool(use_weighted_sampling)
        self.lifecycle_col = lifecycle_col
        self.lifecycle_weight_map = lifecycle_weight_map or {}
        self.part_balance_alpha = float(part_balance_alpha)
        self.custom_sample_weight_fn = custom_sample_weight_fn
        self.min_weight = float(min_weight)

        # 문자열 카테고리 -> 정수 ID 매핑
        if build_cat_indexer_from:
            for raw_col in build_cat_indexer_from:
                if raw_col in self.df.columns:
                    idxr = CategoryIndexer.build_from_series(self.df[raw_col])
                    self.cat_indexers[raw_col] = idxr

                    target_col = cat_indexer_target_col if cat_indexer_target_col else f"{raw_col}_id"
                    self.df = self.df.with_columns(
                        pl.Series(name=target_col, values=idxr.map_series(self.df[raw_col])).cast(pl.Int32)
                    )
                    if target_col not in self.past_exo_cat_cols:
                        self.past_exo_cat_cols.append(target_col)

        self.full_dataset: Optional[MultiPartExoTrainingDataset] = None
        self.train_dataset: Optional[Subset] = None
        self.val_dataset: Optional[Subset] = None

        # 캐시: part별 인덱스 맵
        self._train_part_to_indices: Optional[Dict[Any, list[int]]] = None
        self._val_part_to_indices: Optional[Dict[Any, list[int]]] = None

    def setup(self):
        self.full_dataset = MultiPartExoTrainingDataset(
            self.df, self.lookback, self.horizon,
            part_col=self.part_col,
            date_col=self.date_col,
            qty_col=self.qty_col,
            past_exo_cont_cols=self.past_exo_cont_cols,
            past_exo_cat_cols=self.past_exo_cat_cols,
            future_exo_cb=self.future_exo_cb,
            date_indexer=self.date_indexer,
            cat_indexers=self.cat_indexers,
            lifecycle_col=self.lifecycle_col,   # NEW
        )

        total_len = len(self.full_dataset)
        val_len = int(total_len * self.val_ratio)
        train_len = max(0, total_len - val_len)
        gen = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(self.full_dataset, [train_len, val_len], generator=gen)

        # part별 인덱스 맵 캐시 생성
        self._train_part_to_indices = self._build_part_index_map(self.train_dataset)
        self._val_part_to_indices = self._build_part_index_map(self.val_dataset)

    def _build_part_index_map(self, subset: Subset) -> Dict[Any, list[int]]:
        """
        subset.indices: full_dataset 기준 index 목록
        반환: part_id -> [full_dataset_index, ...]
        """
        assert self.full_dataset is not None
        part_map: Dict[Any, list[int]] = defaultdict(list)
        for full_idx in subset.indices:
            part_id = self.full_dataset.samples[full_idx]["part_id"]
            part_map[part_id].append(full_idx)
        return dict(part_map)

    def _compute_weights_for_subset(self, subset: Subset) -> torch.DoubleTensor:
        """
        subset(indices)에 대응하는 샘플별 weight 벡터 생성
        - lifecycle_weight_map
        - part_balance_alpha
        - custom_sample_weight_fn (있으면 곱)
        """
        assert self.full_dataset is not None
        samples = self.full_dataset.samples

        # part별 샘플 수(균형 가중)
        part_counts: Dict[Any, int] = defaultdict(int)
        if self.part_balance_alpha > 0:
            for full_idx in subset.indices:
                part_counts[samples[full_idx]["part_id"]] += 1

        weights = []
        for full_idx in subset.indices:
            s = samples[full_idx]

            w = 1.0

            # (1) lifecycle weight
            if self.lifecycle_weight_map:
                life = s.get("lifecycle", None)
                if life in self.lifecycle_weight_map:
                    w *= float(self.lifecycle_weight_map[life])

            # (2) part balance weight (희소 part 업샘플)
            if self.part_balance_alpha > 0:
                c = max(1, int(part_counts.get(s["part_id"], 1)))
                # count가 작을수록 커짐: (1/c)^alpha
                w *= float((1.0 / c) ** self.part_balance_alpha)

            # (3) custom hook
            if self.custom_sample_weight_fn is not None:
                w *= float(self.custom_sample_weight_fn(s))

            if not np.isfinite(w) or w <= 0:
                w = self.min_weight
            weights.append(max(self.min_weight, w))

        return torch.tensor(weights, dtype=torch.double)

    def get_train_loader(self):
        if self.train_dataset is None:
            self.setup()

        # Weighted Sampling 사용 시: sampler가 shuffle을 대체
        if self.use_weighted_sampling:
            weights = self._compute_weights_for_subset(self.train_dataset)
            gen = torch.Generator().manual_seed(self.seed)
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
                generator=gen,
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                shuffle=False,
                drop_last=True,
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True
        )

    def get_val_loader(self):
        if self.val_dataset is None:
            self.setup()
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

    # -----------------------------
    # NEW: part_id 별 DataLoader 생성
    # -----------------------------
    def get_train_loaders_by_part(self, *, batch_size: Optional[int] = None, drop_last: bool = True) -> Dict[Any, DataLoader]:
        if self.train_dataset is None:
            self.setup()
        assert self._train_part_to_indices is not None

        bs = int(batch_size) if batch_size is not None else self.batch_size
        loaders: Dict[Any, DataLoader] = {}
        for part_id, full_indices in self._train_part_to_indices.items():
            subset = Subset(self.full_dataset, full_indices)  # full_dataset 기준으로 part subset
            loaders[part_id] = DataLoader(subset, batch_size=bs, shuffle=False, drop_last=drop_last)
        return loaders

    def get_val_loaders_by_part(self, *, batch_size: Optional[int] = None, drop_last: bool = False) -> Dict[Any, DataLoader]:
        if self.val_dataset is None:
            self.setup()
        assert self._val_part_to_indices is not None

        bs = int(batch_size) if batch_size is not None else self.batch_size
        loaders: Dict[Any, DataLoader] = {}
        for part_id, full_indices in self._val_part_to_indices.items():
            subset = Subset(self.full_dataset, full_indices)
            loaders[part_id] = DataLoader(subset, batch_size=bs, shuffle=False, drop_last=drop_last)
        return loaders

    def get_inference_loader_at_plan(self, plan_dt: int):
        ds = MultiPartExoAnchoredInferenceDataset(
            df=self.df,
            lookback=self.lookback,
            horizon=self.horizon,
            plan_dt=int(plan_dt),
            freq=self.freq,
            part_col=self.part_col,
            date_col=self.date_col,
            qty_col=self.qty_col,
            past_exo_cont_cols=self.past_exo_cont_cols,
            past_exo_cat_cols=self.past_exo_cat_cols,
            fill_missing=self.fill_missing,
            future_exo_cb=self.future_exo_cb,
            date_indexer=self.date_indexer,
            cat_indexers=self.cat_indexers,
        )
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False)

    def get_inference_loaders_at_plan_by_part(self, plan_dt: int, *, batch_size: Optional[int] = None) -> Dict[Any, DataLoader]:
        """
        plan_dt 기준 inference dataset을 만든 뒤, part_id 별로 쪼개서 loader 반환
        """
        ds = MultiPartExoAnchoredInferenceDataset(
            df=self.df,
            lookback=self.lookback,
            horizon=self.horizon,
            plan_dt=int(plan_dt),
            freq=self.freq,
            part_col=self.part_col,
            date_col=self.date_col,
            qty_col=self.qty_col,
            past_exo_cont_cols=self.past_exo_cont_cols,
            past_exo_cat_cols=self.past_exo_cat_cols,
            fill_missing=self.fill_missing,
            future_exo_cb=self.future_exo_cb,
            date_indexer=self.date_indexer,
            cat_indexers=self.cat_indexers,
        )

        part_map: Dict[Any, list[int]] = defaultdict(list)
        for i, pid in enumerate(ds.part_ids):
            part_map[pid].append(i)

        bs = int(batch_size) if batch_size is not None else self.batch_size
        loaders: Dict[Any, DataLoader] = {}
        for pid, idxs in part_map.items():
            subset = Subset(ds, idxs)
            loaders[pid] = DataLoader(subset, batch_size=bs, shuffle=False)
        return dict(loaders)
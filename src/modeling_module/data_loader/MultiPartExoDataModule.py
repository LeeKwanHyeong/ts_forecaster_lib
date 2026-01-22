import time

import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import numpy as np
from typing import Callable, Optional, Sequence, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field

# 기존 DateUtil이 있다면 사용하고, 없으면 내부 로직 사용을 위해 import는 유지
try:
    from modeling_module.utils.date_util import DateUtil
except ImportError:
    DateUtil = None


# -----------------------------
# Utility
# -----------------------------
def _to_numpy(x):
    """
        입력을 NumPy 배열로 변환합니다.
        Polars Series가 입력될 경우 호환성을 위해 .to_numpy()를 호출합니다.

        Args:
            x: 변환할 데이터 (Polars Series, List, 또는 Array-like)

        Returns:
            np.ndarray: 변환된 NumPy 배열
        """
    if isinstance(x, pl.Series):
        return x.to_numpy()
    return np.asarray(x)
def identity_date_indexer(x: int) -> int:
    """
        입력된 값을 변환 없이 그대로 반환하는 식별 함수(Identity Function)입니다.
        주로 인덱서 파이프라인에서 변환이 필요 없을 때 Placeholder로 사용됩니다.
    """
    return int(x)

def _add_time(dt_int: int, amount: int, freq: str) -> int:
    """
    정수형 날짜 포맷(YYYYMM, YYYYMMDD 등)에 시간을 더하거나 뺍니다.
    문자열 파싱 -> 날짜 연산 -> 다시 정수형 변환의 과정을 거칩니다.

    Args:
        dt_int (int): 기준 날짜 (예: 20230101, 202312)
        amount (int): 더하거나 뺄 시간의 양 (음수일 경우 과거로 이동)
        freq (str): 시간 단위 ('hourly', 'daily', 'weekly', 'monthly')

    Returns:
        int: 연산이 적용된 정수형 날짜

    Raises:
        ImportError: 'weekly' 로직 사용 시 DateUtil 모듈이 없을 경우 발생
    """
    s = str(dt_int)

    # 1. 시간 단위 연산 (YYYYMMDDHH)
    if freq == 'hourly':
        fmt = "%Y%m%d%H"
        dt_obj = datetime.strptime(s, fmt)
        new_dt = dt_obj + timedelta(hours=amount)
        return int(new_dt.strftime(fmt))

    # 2. 일 단위 연산 (YYYYMMDD)
    elif freq == 'daily':
        fmt = "%Y%m%d"
        dt_obj = datetime.strptime(s, fmt)
        new_dt = dt_obj + timedelta(days=amount)
        return int(new_dt.strftime(fmt))

    # 3. 주 단위 연산 (YYYYWW) - 별도 유틸 필요
    elif freq == 'weekly':
        if DateUtil:
            return DateUtil.add_weeks_yyyyww(dt_int, amount)
        raise ImportError("Weekly logic requires DateUtil module.")

    # 4. 월 단위 연산 (YYYYMM)
    elif freq == 'monthly':
        # 외부 유틸이 있다면 우선 사용
        if DateUtil:
            return DateUtil.add_months_yyyymm(dt_int, amount)

        # DateUtil이 없을 경우의 기본적인 월 연산 구현
        y = dt_int // 100
        m = dt_int % 100
        m += amount

        # 월(Month)이 1~12 범위를 벗어날 경우 연도(Year) 보정
        while m < 1:
            m += 12
            y -= 1
        while m > 12:
            m -= 12
            y += 1
        return y * 100 + m

    # 해당되는 freq가 없을 경우 원본 반환
    return dt_int


def _generate_time_seq(plan_dt: int, length: int, freq: str) -> np.ndarray:
    """
    기준 날짜(plan_dt) '직전'부터 과거로 length 만큼의 날짜 시퀀스를 생성합니다.
    (Look-back Window 생성 용도)

    예: plan_dt=20230105, length=3, freq='daily'
    -> 결과: [20230102, 20230103, 20230104] (오름차순 정렬됨)

    Args:
        plan_dt (int): 예측/계획 기준 시점
        length (int): 생성할 과거 시점의 길이 (Sequence Length)
        freq (str): 시간 단위

    Returns:
        np.ndarray: 과거 날짜들이 담긴 NumPy 배열 (int64)
    """
    seq = []
    # 기준 시점 바로 전 단계부터 시작 (Lag 1)
    current = _add_time(plan_dt, -1, freq)

    for _ in range(length):
        seq.append(current)
        # 계속해서 과거로 이동
        current = _add_time(current, -1, freq)

    # seq는 [어제, 그제, 3일전...] 순서이므로
    # [::-1]을 사용하여 시간 순서(과거 -> 최근)로 정렬하여 반환
    return np.array(seq[::-1], dtype=np.int64)


def _build_train_collate_fn(
    *,
    horizon: int,
    future_exo_cb: Optional[Callable] = None,
    cache_size: int = 4096,
):
    # 주의: future_exo_cb도 반드시 pickle 가능한 “top-level 함수/클래스”여야 합니다.
    return TrainCollateWithFutureExo(horizon=int(horizon), future_exo_cb=future_exo_cb, cache_size=int(cache_size))


class CategoryIndexer:
    """
    문자열/임의 카테고리를 일관된 정수 ID로 변환하는 헬퍼 클래스.
    - UNK(미등록) 토큰: 0으로 예약
    - 등록된 값(Known values): 1..K 순차 부여
    """

    def __init__(self, mapping: Optional[Dict[Any, int]] = None):
        # UNK 토큰 ID (기본값 0)
        self.unk_id = 0
        # 값 -> ID 매핑 딕셔너리 초기화 (None일 경우 빈 딕셔너리)
        self.mapping: Dict[Any, int] = mapping or {}

    @staticmethod
    def build_from_series(series: pl.Series, sort: bool = True) -> "CategoryIndexer":
        """
        Polars Series의 유니크 값들을 기반으로 인덱서 생성 및 반환.
        """
        # 결측치(Null) 제거 및 유니크 값 리스트 추출
        vals = series.drop_nulls().unique().to_list()

        # 정렬 옵션 처리 (가능한 경우 오름차순 정렬)
        if sort:
            try:
                vals = sorted(vals)
            except Exception:
                # 정렬 불가능한 타입(혼합 타입 등)일 경우 에러 무시
                pass

        mapping = {}
        next_id = 1  # ID는 1부터 시작 (0은 UNK용으로 예약됨)

        # 유니크 값들에 순차적으로 ID 부여
        for v in vals:
            if v not in mapping:
                mapping[v] = next_id
                next_id += 1

        # 생성된 매핑으로 클래스 인스턴스 반환
        return CategoryIndexer(mapping)

    def id_of(self, value: Any) -> int:
        """
        단일 값에 대한 ID 조회.
        매핑에 없는 값일 경우 UNK ID(0) 반환.
        """
        return self.mapping.get(value, self.unk_id)

    def map_series(self, s: pl.Series) -> np.ndarray:
        """
        Polars Series 전체를 정수 ID 배열(NumPy int64)로 변환.
        """
        # 리스트 변환 후 각 원소에 id_of 적용, 최종적으로 NumPy 배열 생성
        return np.asarray([self.id_of(v) for v in s.to_list()], dtype=np.int64)


# ============================================================
# 1) Training Dataset (index_map 기반)
# ============================================================
import torch
from torch.utils.data import Dataset
import polars as pl
import numpy as np
from typing import Optional, List, Dict, Tuple, Sequence, Callable, Any


# ============================================================
# 1) Training Dataset (index_map 기반)
# ============================================================
class MultiPartExoTrainingDataset(Dataset):
    """
    슬라이딩 윈도우(Sliding Window) 학습을 위한 Dataset 클래스.

    특징:
      - 메모리 효율성: 샘플을 미리 복제하지 않고, ID별 원본 배열과 인덱스 맵(index_map)만 유지.
      - 다중 분할 지원: ID 단위 Split을 위한 id -> indices 매핑 제공.
      - 시계열 처리: 과거/미래 데이터 및 범주형/연속형 외생 변수(Exogenous variables) 처리.

    반환값 (Tuple):
      - x: [L, 1] (float32) - Lookback 구간 타겟 시퀀스
      - y: [H] (float32) - Horizon 구간 정답 시퀀스
      - id: (str) - 시계열 식별자
      - start_idx: (int) - 미래 외생 변수 조회를 위한 시작 시점 인덱스
      - pe_cont_t: [L, E_cont] (float32) - 과거 연속형 외생 변수
      - pe_cat_t: [L, E_cat] (long) - 과거 범주형 외생 변수
    """

    def __init__(
            self,
            df: pl.DataFrame,
            lookback: int,
            horizon: int,
            *,
            id_col: str = "unique_id",
            date_col: str = "date",
            qty_col: str = "y",
            past_exo_cont_cols: Optional[Sequence[str]] = None,
            past_exo_cat_cols: Optional[Sequence[str]] = None,
            future_exo_cb: Optional[Callable[[int, int, str], np.ndarray | torch.Tensor]] = None,
            date_indexer: Optional[Callable[[int], int]] = None,
            cat_indexers: Optional[Dict[str, Any]] = None,  # Type hint adjusted
    ):
        # 윈도우 설정
        self.lookback = int(lookback)
        self.horizon = int(horizon)

        # 컬럼명 설정
        self.id_col = id_col
        self.date_col = date_col
        self.qty_col = qty_col

        # 외생 변수 컬럼 리스트 초기화
        self.past_exo_cont_cols = list(past_exo_cont_cols) if past_exo_cont_cols else []
        self.past_exo_cat_cols = list(past_exo_cat_cols) if past_exo_cat_cols else []

        # 헬퍼 함수 및 인덱서 설정
        self.future_exo_cb = future_exo_cb
        self.date_indexer = date_indexer or identity_date_indexer
        self.cat_indexers = cat_indexers or {}

        # 데이터 저장소 초기화
        # self.series[id] = {key: np.array} 구조로 원본 데이터 보관
        self.series: Dict[str, Dict[str, np.ndarray]] = {}

        # 전역 인덱스 맵: 전체 데이터셋의 i번째 샘플이 (어떤 series의, 몇 번째 시점인지) 매핑
        self.index_map: List[Tuple[str, int]] = []

        # ID별 인덱스 매핑: 특정 ID가 포함된 전역 인덱스 리스트 (Split 용도)
        self.id_to_indices: Dict[str, List[int]] = {}

        # 필수 컬럼 존재 여부 검증
        if self.id_col not in df.columns:
            raise KeyError(f"id_col='{self.id_col}' not found in df.columns")
        if self.date_col not in df.columns:
            raise KeyError(f"date_col='{self.date_col}' not found in df.columns")
        if self.qty_col not in df.columns:
            raise KeyError(f"qty_col='{self.qty_col}' not found in df.columns")

        # ID별 데이터 파티셔닝 및 전처리
        for g in df.partition_by(self.id_col):
            # 날짜순 정렬 보장
            g = g.sort(self.date_col)
            uid = str(g[self.id_col][0])

            # 타겟 및 날짜 데이터를 NumPy 배열로 변환
            y_all = _to_numpy(g[self.qty_col]).astype(np.float32)  # [T]
            d_all = _to_numpy(g[self.date_col]).astype(np.int64)  # [T]

            T = len(y_all)
            # 데이터 길이가 학습에 필요한 최소 길이(Lookback + Horizon)보다 짧으면 스킵
            if T < self.lookback + self.horizon:
                continue

            # ----- 연속형 과거 외생 변수 처리 (Past Continuous Exo) -----
            cont_list = []
            for col in self.past_exo_cont_cols:
                if col in g.columns:
                    cont_list.append(_to_numpy(g[col]).astype(np.float32))
            # [T, Feature] 형태로 스택
            exo_cont = np.stack(cont_list, axis=-1) if cont_list else np.zeros((T, 0), dtype=np.float32)

            # ----- 범주형 과거 외생 변수 처리 (Past Categorical Exo) -----
            cat_list = []
            for col in self.past_exo_cat_cols:
                if col not in g.columns:
                    continue
                s = g[col]
                # 이미 정수형 ID인 경우 그대로 사용
                if s.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                    cat_list.append(_to_numpy(s).astype(np.int64))
                else:
                    # 문자열 등인 경우 Indexer를 통해 정수 ID로 변환
                    if col not in self.cat_indexers:
                        raise TypeError(f"Categorical '{col}' needs a CategoryIndexer or integer IDs.")
                    cat_list.append(self.cat_indexers[col].map_series(s))
            # [T, Feature] 형태로 스택
            exo_cat = np.stack(cat_list, axis=-1) if cat_list else np.zeros((T, 0), dtype=np.int64)

            # 처리된 데이터를 메모리에 저장
            self.series[uid] = {"y": y_all, "d": d_all, "exo_cont": exo_cont, "exo_cat": exo_cat}

            # ----- 슬라이딩 윈도우 인덱스 생성 -----
            n_windows = T - self.lookback - self.horizon + 1
            if n_windows <= 0:
                continue

            self.id_to_indices[uid] = []
            for i in range(n_windows):
                gidx = len(self.index_map)
                # (ID, 시작 위치) 정보를 전역 맵에 등록
                self.index_map.append((uid, i))
                # ID별 인덱스 목록 업데이트
                self.id_to_indices[uid].append(gidx)

    def __len__(self):
        """전체 샘플(윈도우) 개수 반환."""
        return len(self.index_map)

    def __getitem__(self, idx: int):
        """
        인덱스에 해당하는 학습 샘플 추출 및 텐서 변환.
        """
        # 인덱스 맵에서 ID와 시작 위치 조회
        uid, i = self.index_map[idx]
        pack = self.series[uid]

        # 데이터 참조 로드
        y_all = pack["y"]
        d_all = pack["d"]
        exo_cont = pack["exo_cont"]
        exo_cat = pack["exo_cat"]

        L = self.lookback
        H = self.horizon

        # 슬라이싱: Lookback 구간(Input)과 Horizon 구간(Target) 추출
        x_win = y_all[i:i + L]  # [L]
        y_win = y_all[i + L:i + L + H]  # [H]

        # 외생 변수 슬라이싱 (과거 구간만 필요)
        pe_cont = exo_cont[i:i + L, :] if exo_cont.shape[1] > 0 else np.zeros((L, 0), dtype=np.float32)
        pe_cat = exo_cat[i:i + L, :] if exo_cat.shape[1] > 0 else np.zeros((L, 0), dtype=np.int64)

        # 미래 외생 변수 시작 시점 계산
        # Lookback 마지막 시점의 날짜 조회
        last_dt = int(d_all[i + L - 1])
        # 인덱서를 통해 '예측 시작 시점(Horizon 첫 번째)'의 정수형 날짜/인덱스 계산
        start_idx = int(self.date_indexer(last_dt)) + 1

        # Tensor 변환 (최소한의 연산으로 수행)
        x = torch.from_numpy(x_win).to(torch.float32).unsqueeze(-1)  # [L, 1]
        y = torch.from_numpy(y_win).to(torch.float32)  # [H]
        pe_cont_t = torch.from_numpy(pe_cont).to(torch.float32)  # [L, E_cont]
        pe_cat_t = torch.from_numpy(pe_cat).to(torch.long)  # [L, E_cat]

        # Future Exo Tensor는 여기서 생성하지 않고 start_idx만 반환 (DataCollator 등에서 처리 유도)
        return x, y, uid, start_idx, pe_cont_t, pe_cat_t

# ============================================================
# 2) Inference Dataset (Unified for Monthly/Weekly/Daily/Hourly)
# ============================================================
class MultiPartExoAnchoredInferenceDataset(Dataset):
    """
    특정 시점(plan_dt)을 기준으로 과거 데이터를 조회하여 추론 입력을 생성하는 Dataset.

    특징:
      - 앵커링(Anchoring): 학습용 슬라이딩 윈도우와 달리, '특정 예측 시점' 하나에 고정된 과거 데이터 생성.
      - 결측치 처리: 시계열 끊김 발생 시 ffill(이전 값 참조), zero, nan 등 유연한 채움 로직 지원.
      - 시간 단위(Freq) 대응: Monthly/Weekly/Daily 등 다양한 주기에 따른 날짜 계산 분기.
    """

    def __init__(
            self,
            df: pl.DataFrame,
            lookback: int,
            horizon: int,
            plan_dt: int,
            freq: str,  # 'monthly', 'weekly', 'daily', 'hourly'
            *,
            id_col: str = "unique_id",
            date_col: str = "date",
            qty_col: str = "y",
            past_exo_cont_cols: Optional[Sequence[str]] = None,
            past_exo_cat_cols: Optional[Sequence[str]] = None,
            fill_missing: str = "ffill",
            target_back_steps: int = 100,  # 결측치 채울 때 얼마나 뒤를 볼지
            future_exo_cb: Optional[Callable[[int, int, str], np.ndarray | torch.Tensor]] = None,
            date_indexer: Optional[Callable[[int], int]] = None,
            cat_indexers: Optional[Dict[str, Any]] = None,  # Type hint adjusted
    ):
        # 윈도우 및 기준 시점 설정
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.plan_dt = int(plan_dt)
        self.freq = freq.lower()

        # 컬럼 매핑 설정
        self.id_col = id_col
        self.date_col = date_col
        self.qty_col = qty_col

        # 외생 변수 컬럼 리스트
        self.past_exo_cont_cols = list(past_exo_cont_cols) if past_exo_cont_cols else []
        self.past_exo_cat_cols = list(past_exo_cat_cols) if past_exo_cat_cols else []

        # 결측치 처리 및 헬퍼 설정
        self.fill_missing = fill_missing
        self.target_back_steps = int(target_back_steps)
        self.future_exo_cb = future_exo_cb
        self.date_indexer = date_indexer or identity_date_indexer
        self.cat_indexers = cat_indexers or {}

        # 최종 데이터 저장 리스트 초기화
        self.inputs, self.ids = [], []
        self.past_exo_conts, self.past_exo_cats = [], []
        self.future_exo_conts = []

        # freq에 따른 과거 시점 시퀀스 생성 (Lookback Window 구성)
        # 예: plan_dt=20240101 -> [20231229, 20231230, 20231231]
        win_dates = _generate_time_seq(self.plan_dt, self.lookback, self.freq)

        # ID별 데이터 파티셔닝 및 처리
        grouped = df.partition_by(self.id_col)
        for g in grouped:
            uid = str(g[self.id_col][0])

            # 날짜 및 타겟 값 추출 (NumPy 변환)
            dts = _to_numpy(g[self.date_col]).astype(np.int64)
            vals = _to_numpy(g[self.qty_col]).astype(float)
            if len(dts) == 0:
                continue

            # 빠른 조회를 위한 {날짜: 값} 매핑 생성
            qty_map = {int(d): float(v) for d, v in zip(dts, vals)}
            earliest = int(dts.min())

            # 1) 타겟 데이터(x) 생성 및 결측치 처리
            x = np.empty(self.lookback, dtype=float)
            for i, curr_dt in enumerate(win_dates):
                if curr_dt in qty_map:
                    # 데이터 존재 시 할당
                    x[i] = qty_map[curr_dt]
                else:
                    # 데이터 부재 시 전략에 따른 채움
                    if self.fill_missing == "zero":
                        x[i] = 0.0
                    elif self.fill_missing == "nan":
                        x[i] = np.nan
                    else:
                        # ffill 로직: 과거 시점으로 거슬러 올라가며 값 탐색
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

            # 모든 값이 NaN인 경우(유효 데이터 없음) 해당 샘플 스킵
            if self.fill_missing == "nan" and not np.any(np.isfinite(x)):
                continue

            # 2) 연속형 과거 외생 변수(Continuous Past Exo) 처리
            pe_cont_list = []
            for col in self.past_exo_cont_cols:
                if col not in g.columns:
                    continue
                val_map = {int(d): float(v) for d, v in zip(dts, _to_numpy(g[col]).astype(float))}

                e = np.empty(self.lookback, dtype=float)
                for i, curr_dt in enumerate(win_dates):
                    # 타겟 변수와 동일한 결측치 채움 로직 적용
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

            # [L, Features] 형태로 스택
            pe_cont_mat = np.stack(pe_cont_list, axis=-1) if pe_cont_list else np.zeros((self.lookback, 0), dtype=float)

            # 3) 범주형 과거 외생 변수(Categorical Past Exo) 처리
            pe_cat_list = []
            for col in self.past_exo_cat_cols:
                if col not in g.columns:
                    continue
                s = g[col]

                # 값 -> 정수 ID 변환
                if s.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                    vals_int = _to_numpy(s).astype(np.int64)
                    unk = 0
                else:
                    if col not in self.cat_indexers:
                        # Indexer 부재 시 전체 0(UNK) 처리
                        vals_int = np.zeros(len(s), dtype=np.int64)
                        unk = 0
                    else:
                        # Indexer 사용 변환
                        idxr = self.cat_indexers[col]
                        vals_int = np.array([idxr.id_of(v) for v in s.to_list()], dtype=np.int64)
                        unk = idxr.unk_id

                val_map = {int(d): int(v) for d, v in zip(dts, vals_int)}

                e = np.empty(self.lookback, dtype=np.int64)
                for i, curr_dt in enumerate(win_dates):
                    # 범주형 결측 처리는 UNK(0) 또는 최근 값(ffill) 사용
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

            # [L, Features] 형태로 스택
            pe_cat_mat = np.stack(pe_cat_list, axis=-1) if pe_cat_list else np.zeros((self.lookback, 0), dtype=np.int64)

            # 4) 미래 외생 변수(Future Exo) 처리
            # 마지막 과거 시점을 기준으로 미래 시작 인덱스 계산
            last_hist = int(win_dates[-1])
            start_idx = int(self.date_indexer(last_hist)) + 1

            fe = np.zeros((self.horizon, 0), dtype=float)
            if self.future_exo_cb is not None:
                # Callback을 통해 미래 시점의 외생 변수 조회
                res = self.future_exo_cb(start_idx, self.horizon, device="cpu")
                fe = res.detach().cpu().numpy() if isinstance(res, torch.Tensor) else np.asarray(res, dtype=float)

            # 처리된 샘플 저장
            self.inputs.append(x)
            self.past_exo_conts.append(pe_cont_mat)
            self.past_exo_cats.append(pe_cat_mat)
            self.future_exo_conts.append(fe)
            self.ids.append(uid)

    def __len__(self):
        """생성된 추론용 샘플 수 반환."""
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        인덱스에 해당하는 추론 샘플 반환 및 텐서 변환.
        Returns:
            x: [Lookback, 1]
            id: 식별자 문자열
            feC: [Horizon, Future_Exo_Dim]
            peC: [Lookback, Past_Exo_Cont_Dim]
            peK: [Lookback, Past_Exo_Cat_Dim]
        """
        x = torch.tensor(self.inputs[idx], dtype=torch.float32).unsqueeze(-1)
        peC = torch.tensor(self.past_exo_conts[idx], dtype=torch.float32)
        peK = torch.tensor(self.past_exo_cats[idx], dtype=torch.long)
        feC = torch.tensor(self.future_exo_conts[idx], dtype=torch.float32)
        return x, self.ids[idx], feC, peC, peK


# ============================================================
# 3) Main DataModule (split_mode: window | multi)
# ============================================================
from typing import Optional, Sequence, Callable, Dict, List, Union
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Subset, random_split


class MultiPartExoDataModule:
    """
    시계열 데이터 로딩 및 관리를 위한 DataModule 클래스.

    기능:
    - 데이터 주기(Freq) 및 포맷 검증.
    - 학습/검증 데이터 분할 전략 (Window 단위 vs ID 단위) 구현.
    - 범주형 변수 인덱싱 자동화.
    - 학습 및 추론용 DataLoader 생성.

    Split Mode:
      - 'window': 전체 윈도우를 무작위로 분할 (ID 섞임 허용).
      - 'multi' : ID 단위로 그룹화하여 분할 (ID 간 데이터 누수 방지).
    """

    def __init__(
            self,
            df: pl.DataFrame,
            lookback: int,
            horizon: int,
            *,
            freq: str = 'weekly',
            batch_size: int = 512,
            val_ratio: float = 0.2,
            shuffle: bool = True,  # 학습 시 셔플 여부 (기본 True 권장)
            seed: int = 42,
            id_col: str = "unique_id",
            date_col: str = "date",
            y_col: str = "HUFL",
            past_exo_cont_cols: Optional[Sequence[str]] = None,
            past_exo_cat_cols: Optional[Sequence[str]] = None,
            fill_missing: str = "ffill",
            target_back_steps: int = 100,
            future_exo_cb: Optional[Callable[[int, int, str], np.ndarray | torch.Tensor]] = None,
            date_indexer: Optional[Callable[[int], int]] = None,
            build_cat_indexer_from: Optional[Sequence[str]] = None,
            cat_indexer_target_col: Optional[str] = None,
            split_mode: str = "window",  # 'window' | 'multi'
    ):
        self.df = df
        self.lookback = int(lookback)
        self.horizon = int(horizon)

        # 주기(Frequency) 유효성 검증
        valid_freqs = ('monthly', 'weekly', 'daily', 'hourly')
        if freq not in valid_freqs:
            raise ValueError(f"freq must be one of {valid_freqs}, got '{freq}'")
        self.freq = freq

        # 분할 모드 검증
        if split_mode not in ("window", "multi"):
            raise ValueError("split_mode must be 'window' or 'multi'")
        self.split_mode = split_mode

        # 하이퍼파라미터 및 설정 저장
        self.batch_size = int(batch_size)
        self.val_ratio = float(val_ratio)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

        # 컬럼 매핑
        self.id_col = id_col
        self.date_col = date_col
        self.qty_col = y_col

        self.past_exo_cont_cols = list(past_exo_cont_cols) if past_exo_cont_cols else []
        self.past_exo_cat_cols = list(past_exo_cat_cols) if past_exo_cat_cols else []

        # 전처리 및 콜백 설정
        self.fill_missing = fill_missing
        self.target_back_steps = int(target_back_steps)
        self.future_exo_cb = future_exo_cb
        self.date_indexer = date_indexer or identity_date_indexer

        self.cat_indexers: Dict[str, CategoryIndexer] = {}

        # 범주형 인덱서(Category Indexer) 자동 생성 및 매핑 적용
        if build_cat_indexer_from:
            for raw_col in build_cat_indexer_from:
                if raw_col in self.df.columns:
                    # 인덱서 빌드
                    idxr = CategoryIndexer.build_from_series(self.df[raw_col])
                    self.cat_indexers[raw_col] = idxr

                    # 변환된 ID 컬럼 생성
                    target_col = cat_indexer_target_col if cat_indexer_target_col else f"{raw_col}_id"

                    self.df = self.df.with_columns(
                        pl.Series(name=target_col, values=idxr.map_series(self.df[raw_col])).cast(pl.Int32)
                    )

                    # 학습 피처 목록에 추가
                    if target_col not in self.past_exo_cat_cols:
                        self.past_exo_cat_cols.append(target_col)

        self.train_dataset = None
        self.val_dataset = None

    def setup(self):
        """
        전체 데이터셋 생성 및 학습/검증 세트 분할 수행.
        """
        # 전체 데이터셋 초기화
        full_dataset = MultiPartExoTrainingDataset(
            self.df,
            self.lookback,
            self.horizon,
            id_col=self.id_col,
            date_col=self.date_col,
            qty_col=self.qty_col,
            past_exo_cont_cols=self.past_exo_cont_cols,
            past_exo_cat_cols=self.past_exo_cat_cols,
            future_exo_cb=self.future_exo_cb,
            date_indexer=self.date_indexer,
            cat_indexers=self.cat_indexers,
        )

        total_len = len(full_dataset)
        # 데이터가 없는 경우 처리
        if total_len == 0:
            self.train_dataset = full_dataset
            self.val_dataset = full_dataset
            return

        gen = torch.Generator().manual_seed(self.seed)

        # -------------------------
        # 1) Window Split 모드
        # -------------------------
        # 전체 윈도우를 무작위로 섞어서 분할 (단순 비율 분할)
        if self.split_mode == "window":
            val_len = int(total_len * self.val_ratio)
            train_len = max(0, total_len - val_len)
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_len, val_len], generator=gen)
            return

        # -------------------------
        # 2) Multi Split 모드 (ID 단위)
        # -------------------------
        # ID별로 그룹화하여 Train/Val ID를 분리 (Data Leakage 방지)
        ids = list(full_dataset.id_to_indices.keys())

        # ID가 1개 이하인 경우 ID 분할 불가능 -> Window Split으로 대체
        if len(ids) <= 1:
            val_len = int(total_len * self.val_ratio)
            train_len = max(0, total_len - val_len)
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_len, val_len], generator=gen)
            return

        # ID별 샘플(윈도우) 수 계산
        id_counts = {uid: len(full_dataset.id_to_indices[uid]) for uid in ids}
        total_windows = sum(id_counts.values())
        target_val_windows = int(total_windows * self.val_ratio)

        # ID 리스트 셔플
        rng = np.random.default_rng(self.seed)
        rng.shuffle(ids)

        # 검증(Validation)용 ID 선정 (Greedy 방식: 목표 비율 채울 때까지 추가)
        val_ids: List[str] = []
        cur = 0
        for uid in ids:
            if cur >= target_val_windows and len(val_ids) > 0:
                break
            val_ids.append(uid)
            cur += id_counts[uid]

        # Train/Val ID 집합 분리
        val_id_set = set(val_ids)
        train_ids = [uid for uid in ids if uid not in val_id_set]

        # 각 집합에 속하는 실제 윈도우 인덱스 수집
        train_indices: List[int] = []
        for uid in train_ids:
            train_indices.extend(full_dataset.id_to_indices[uid])

        val_indices: List[int] = []
        for uid in val_ids:
            val_indices.extend(full_dataset.id_to_indices[uid])

        # 예외 처리: 한쪽 셋이 비어버린 경우 안전하게 Window Split으로 회귀
        if len(train_indices) == 0 or len(val_indices) == 0:
            val_len = int(total_len * self.val_ratio)
            train_len = max(0, total_len - val_len)
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_len, val_len], generator=gen)
            return

        # Subset을 사용하여 최종 데이터셋 구성
        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)

    def get_train_loader(
            self,
            batch_size: Optional[int] = None,
            shuffle: Optional[bool] = None,
            drop_last: bool = True,
            num_workers: int = 0,
            pin_memory: bool = True,
            persistent_workers: bool = True,
            prefetch_factor: int = 2,
    ):
        """
        학습용 DataLoader 생성 및 반환.
        """
        if batch_size is None:
            batch_size = self.batch_size
        if shuffle is None:
            shuffle = self.shuffle

        # 1) setup 호출 보장 (Lazy Setup)
        if getattr(self, "train_dataset", None) is None:
            self.setup()

        # 2) setup 실패 여부 확인
        if getattr(self, "train_dataset", None) is None:
            raise RuntimeError(
                "[get_train_loader] train_dataset is None even after setup(). "
                "Check setup(): full_dataset 생성 및 train/val split 경로 확인 필요."
            )

        # 3) 배치 단위 Future Exo 생성을 위한 Collate Function 빌드
        collate_fn = _build_train_collate_fn(
            horizon=self.horizon,
            future_exo_cb=self.future_exo_cb,
            cache_size=15000,
        )

        # 4) DataLoader 생성 및 워커(Worker) 옵션 처리
        # Windows 환경 등에서 num_workers=0일 때 prefetch 옵션 사용 시 에러 방지
        loader_kwargs = dict(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        if num_workers > 0:
            loader_kwargs["persistent_workers"] = persistent_workers
            loader_kwargs["prefetch_factor"] = prefetch_factor
        else:
            loader_kwargs["persistent_workers"] = False  # 워커 없으므로 비활성화

        return DataLoader(**loader_kwargs)

    def get_val_loader(
            self,
            batch_size: Optional[int] = None,
            drop_last: bool = False,
            num_workers: int = 0,
            pin_memory: bool = True,
            persistent_workers: bool = True,
            prefetch_factor: int = 2,
    ):
        """
        검증용 DataLoader 생성 및 반환.
        """
        if self.val_dataset is None:
            self.setup()

        # 학습과 동일한 Collate 로직 적용 (캐시 공유 가능 시 이점)
        collate_fn = _build_train_collate_fn(
            horizon=self.horizon,
            future_exo_cb=self.future_exo_cb,
            cache_size=15000,
        )

        loader_kwargs = dict(
            dataset=self.val_dataset,
            batch_size=(batch_size or self.batch_size),
            shuffle=False,  # 검증 시 셔플 불필요
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        if num_workers > 0:
            loader_kwargs["persistent_workers"] = persistent_workers
            loader_kwargs["prefetch_factor"] = prefetch_factor
        else:
            loader_kwargs["persistent_workers"] = False

        return DataLoader(**loader_kwargs)

    def get_inference_loader_at_plan(self, plan_dt: int):
        """
        특정 계획 시점(plan_dt)을 기준으로 한 추론용 DataLoader 생성.
        Args:
            plan_dt: 추론 기준 시점 (포맷: YYYYMM, YYYYWW, YYYYMMDD 등)
        """
        # 앵커링된 추론 전용 데이터셋 생성
        ds = MultiPartExoAnchoredInferenceDataset(
            df=self.df,
            lookback=self.lookback,
            horizon=self.horizon,
            plan_dt=int(plan_dt),
            freq=self.freq,
            id_col=self.id_col,
            date_col=self.date_col,
            qty_col=self.qty_col,
            past_exo_cont_cols=self.past_exo_cont_cols,
            past_exo_cat_cols=self.past_exo_cat_cols,
            fill_missing=self.fill_missing,
            target_back_steps=self.target_back_steps,
            future_exo_cb=self.future_exo_cb,
            date_indexer=self.date_indexer,
            cat_indexers=self.cat_indexers,
        )
        # 추론용이므로 셔플 없음
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False)


from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Tuple
import numpy as np
import torch


@dataclass
class TrainCollateWithFutureExo:
    """
    학습 배치 생성 시 Future Exogenous(미래 외생 변수) 데이터를 동적으로 생성 및 병합하는 Collate 클래스.

    특징:
      - 캐싱(Caching): 빈번히 조회되는 시점의 외생 변수 데이터를 메모리에 저장하여 연산 부하 감소.
      - 배치 처리 지원: 콜백 함수가 배치 입력을 지원할 경우 한 번에 생성, 실패 시 개별 루프로 Fallback.
    """
    horizon: int
    future_exo_cb: Optional[Callable] = None
    cache_size: int = 15000

    # 캐시 저장소 (Key: start_idx, Value: Exo Array)
    cache: Dict[int, np.ndarray] = field(default_factory=dict)
    # 캐시 Eviction 관리를 위한 키 리스트 (FIFO 방식)
    cache_keys: List[int] = field(default_factory=list)

    def _cache_get(self, k: int) -> Optional[np.ndarray]:
        """캐시된 데이터 조회."""
        return self.cache.get(k, None)

    def _cache_put(self, k: int, v: np.ndarray):
        """
        데이터 캐싱 및 사이즈 관리.
        캐시 크기 초과 시 가장 오래된 항목(FIFO) 제거.
        """
        if self.cache_size <= 0:
            return
        if k in self.cache:
            return

        self.cache[k] = v
        self.cache_keys.append(k)

        # 용량 초과 시 오래된 항목 제거
        if len(self.cache_keys) > self.cache_size:
            old = self.cache_keys.pop(0)
            self.cache.pop(old, None)

    def __call__(self, batch):
        """
        DataLoader로부터 받은 샘플 리스트를 배치 텐서로 변환.

        Args:
            batch: (x, y, uid, start_idx, pe_cont, pe_cat) 튜플 리스트

        Returns:
            x, y, uid_list, fe, pe_cont, pe_cat 텐서 조합
        """
        # 배치 데이터 언패킹 (Unzipping)
        xs, ys, uids, start_idxs, pe_conts, pe_cats = zip(*batch)

        # 기본 텐서 스택 (Stacking)
        x = torch.stack(xs, dim=0)  # [B, L, 1]
        y = torch.stack(ys, dim=0)  # [B, H]
        pe_cont = torch.stack(pe_conts, 0)  # [B, L, E_cont]
        pe_cat = torch.stack(pe_cats, 0)  # [B, L, E_cat]
        uid_list = list(uids)

        B = len(start_idxs)
        H = int(self.horizon)

        # Future Exo 콜백이 없는 경우 빈 텐서 반환
        if self.future_exo_cb is None:
            fe = torch.zeros((B, H, 0), dtype=torch.float32)
            return x, y, uid_list, fe, pe_cont, pe_cat

        # 1) 캐시 조회 및 미적중(Miss) 데이터 식별
        fe_list: List[Optional[np.ndarray]] = []
        miss: List[int] = []  # 캐시에 없는 start_idx 리스트
        miss_pos: List[int] = []  # 해당 샘플의 배치 내 인덱스

        for bi, s in enumerate(start_idxs):
            s_int = int(s)
            cached = self._cache_get(s_int)
            if cached is None:
                miss.append(s_int)
                miss_pos.append(bi)
                fe_list.append(None)
            else:
                fe_list.append(cached)

        # 2) 미적중(Miss) 데이터 생성
        if miss:
            try:
                # 우선 배치 단위 생성 시도 (속도 최적화)
                res = self.future_exo_cb(miss, H, device="cpu")

                if isinstance(res, torch.Tensor):
                    res = res.detach().cpu().numpy()
                res = np.asarray(res, dtype=np.float32)

                # 형태(Shape) 검증
                if res.ndim != 3 or res.shape[0] != len(miss) or res.shape[1] != H:
                    raise ValueError(f"Batch shape mismatch. got={res.shape}, expected=({len(miss)}, {H}, E)")

                # 결과 매핑 및 캐시 업데이트
                for k, bi in enumerate(miss_pos):
                    fe_arr = res[k]
                    fe_list[bi] = fe_arr
                    self._cache_put(miss[k], fe_arr)

            except Exception:
                # 배치 생성 실패 시 개별(Loop) 생성으로 Fallback
                for s_val, bi in zip(miss, miss_pos):
                    res = self.future_exo_cb(s_val, H, device="cpu")
                    if isinstance(res, torch.Tensor):
                        res = res.detach().cpu().numpy()
                    fe_arr = np.asarray(res, dtype=np.float32)

                    fe_list[bi] = fe_arr
                    self._cache_put(s_val, fe_arr)

        # 3) 최종 결과 스택 및 텐서 변환
        fe_np = np.stack(fe_list, axis=0).astype(np.float32)  # [B, H, E]
        fe = torch.from_numpy(fe_np).to(torch.float32)

        return x, y, uid_list, fe, pe_cont, pe_cat


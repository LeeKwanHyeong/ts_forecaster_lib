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
            freq: str = 'weekly',
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
        self.freq = str(freq).lower()

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
        next_dt = _add_time(last_dt, 1, self.freq)
        # 인덱서를 통해 '예측 시작 시점(Horizon 첫 번째)'의 정수형 날짜/인덱스 계산
        start_idx = int(self.date_indexer(next_dt))

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

        self.start_idxs = []

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
            next_dt = _add_time(last_hist, 1, self.freq)
            # 인덱서를 통해 '예측 시작 시점(Horizon 첫 번째)'의 정수형 날짜/인덱스 계산
            start_idx = int(self.date_indexer(next_dt))
            self.start_idxs.append(start_idx)

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
        x = torch.tensor(self.inputs[idx], dtype=torch.float32).unsqueeze(-1)  # [L,1]
        peC = torch.tensor(self.past_exo_conts[idx], dtype=torch.float32)  # [L,E_cont]
        peK = torch.tensor(self.past_exo_cats[idx], dtype=torch.long)  # [L,E_cat]

        # y dummy: inference에서는 의미 없음 (shape만 horizon 맞추기)
        y_dummy = torch.zeros((self.horizon,), dtype=torch.float32)  # [H]

        # start_idx를 dataset에서 이미 계산해뒀으니 같이 저장해두는 게 정석
        # 현재 __init__에서 start_idx를 계산만 하고 리스트에 저장하지 않음 → 아래처럼 self.start_idxs를 만들어 저장 필요
        start_idx = int(self.start_idxs[idx])

        uid = self.ids[idx]
        return x, y_dummy, uid, start_idx, peC, peK

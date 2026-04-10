from typing import Optional, List, Dict, Sequence, Callable, Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Subset, random_split

from modeling_module.data_loader.future_scenario_store import FutureScenarioStore, TrainCollateWithFutureExo
from modeling_module.data_loader.multi_part_exo_dataset import identity_date_indexer, MultiPartExoTrainingDataset, \
    MultiPartExoAnchoredInferenceDataset

# 기존 DateUtil이 있다면 사용하고, 없으면 내부 로직 사용을 위해 import는 유지
try:
    from modeling_module.utils.date_util import DateUtil
except ImportError:
    DateUtil = None





def _build_train_collate_fn(*, horizon: int, future_exo_cb=None, cache_size: int = 4096,
                            scenario_store: Optional[FutureScenarioStore] = None,
                            scenario_mode: str = "append",
                            scenario_missing_policy: str = "error"):
    return TrainCollateWithFutureExo(
        horizon=int(horizon),
        future_exo_cb=future_exo_cb,
        cache_size=int(cache_size),
        scenario_store=scenario_store,
        scenario_mode=scenario_mode,
        scenario_missing_policy=scenario_missing_policy,
    )


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
# 3) Main DataModule (split_mode: window | multi)
# ============================================================

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
            self.freq,
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

        collate_fn = _build_train_collate_fn(
            horizon=self.horizon,
            future_exo_cb=self.future_exo_cb,
            cache_size=15000,
        )
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)


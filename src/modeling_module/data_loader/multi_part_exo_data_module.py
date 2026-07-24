from datetime import date, datetime
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Subset, random_split

from modeling_module.data_loader.categorical_vocabulary import (
    CategoricalVocabularyArtifact,
)
from modeling_module.data_loader.exogenous_contracts import ExogenousFeatureSchema
from modeling_module.data_loader.future_scenario_store import FutureScenarioStore, TrainCollateWithFutureExo
from modeling_module.data_loader.multi_part_exo_dataset import identity_date_indexer, MultiPartExoTrainingDataset, \
    MultiPartExoAnchoredInferenceDataset
from modeling_module.data_loader.temporal import normalize_temporal_frame

# 기존 DateUtil이 있다면 사용하고, 없으면 내부 로직 사용을 위해 import는 유지
try:
    from modeling_module.utils.date_util import DateUtil
except ImportError:
    DateUtil = None





def _build_train_collate_fn(*, horizon: int, future_exo_cb=None, cache_size: int = 4096,
                            part_future_exo_fn=None,
                            scenario_store: Optional[FutureScenarioStore] = None,
                            scenario_mode: str = "append",
                            scenario_missing_policy: str = "error"):
    return TrainCollateWithFutureExo(
        horizon=int(horizon),
        future_exo_cb=future_exo_cb,
        part_future_exo_fn=part_future_exo_fn,
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
            future_exo_cont_cols: Optional[Sequence[str]] = None,
            future_exo_cat_cols: Optional[Sequence[str]] = None,
            fill_missing: str = "ffill",
            target_back_steps: int = 100,
            future_exo_cb: Optional[Callable[[int, int, str], np.ndarray | torch.Tensor]] = None,
            part_future_exo_fn: Optional[Callable] = None,
            date_indexer: Optional[Callable[[int], int]] = None,
            build_cat_indexer_from: Optional[Sequence[str]] = None,
            cat_indexer_target_col: Optional[str] = None,
            categorical_vocabulary_artifact: Optional[
                CategoricalVocabularyArtifact | Mapping[str, Any]
            ] = None,
            split_mode: str = "window",  # 'window' | 'multi'
    ):
        self._source_df = df
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
        self.future_exo_cont_cols = list(future_exo_cont_cols) if future_exo_cont_cols else []
        self.future_exo_cat_cols = list(future_exo_cat_cols) if future_exo_cat_cols else []

        # 전처리 및 콜백 설정
        self.fill_missing = fill_missing
        self.target_back_steps = int(target_back_steps)
        self.future_exo_cb = future_exo_cb
        self.part_future_exo_fn = part_future_exo_fn
        self.date_indexer = date_indexer or identity_date_indexer

        self.cat_indexers: Dict[str, Any] = {}
        self._categorical_source_by_feature: Dict[str, str] = {
            feature_name: feature_name
            for feature_name in (
                *self.past_exo_cat_cols,
                *self.future_exo_cat_cols,
            )
        }

        legacy_sources = tuple(
            str(column).strip()
            for column in (build_cat_indexer_from or ())
        )
        if any(not column for column in legacy_sources):
            raise ValueError("build_cat_indexer_from cannot contain empty column names.")
        if len(set(legacy_sources)) != len(legacy_sources):
            raise ValueError(
                f"build_cat_indexer_from contains duplicate columns: {legacy_sources}."
            )
        if cat_indexer_target_col and len(legacy_sources) > 1:
            raise ValueError(
                "cat_indexer_target_col can be used with only one "
                "build_cat_indexer_from column."
            )
        missing_legacy_sources = tuple(
            column
            for column in legacy_sources
            if column not in self._source_df.columns
        )
        if missing_legacy_sources:
            raise ValueError(
                "build_cat_indexer_from references missing dataframe columns: "
                + ", ".join(missing_legacy_sources)
            )
        for source_column in legacy_sources:
            target_column = (
                str(cat_indexer_target_col).strip()
                if cat_indexer_target_col
                else f"{source_column}_id"
            )
            if not target_column:
                raise ValueError("cat_indexer_target_col cannot be empty.")
            self._categorical_source_by_feature[target_column] = source_column
            if target_column not in self.past_exo_cat_cols:
                self.past_exo_cat_cols.append(target_column)

        if isinstance(
            categorical_vocabulary_artifact,
            CategoricalVocabularyArtifact,
        ):
            provided_vocabulary = categorical_vocabulary_artifact
        elif isinstance(categorical_vocabulary_artifact, Mapping):
            provided_vocabulary = CategoricalVocabularyArtifact.from_dict(
                categorical_vocabulary_artifact
            )
        elif categorical_vocabulary_artifact is None:
            provided_vocabulary = None
        else:
            raise TypeError(
                "categorical_vocabulary_artifact must be a "
                "CategoricalVocabularyArtifact, mapping, or None."
            )

        expected_categorical_names = tuple(
            self._categorical_source_by_feature
        )
        if (
            provided_vocabulary is not None
            and provided_vocabulary.feature_names
            != expected_categorical_names
        ):
            raise ValueError(
                "Provided categorical vocabulary feature order does not "
                "match configured categorical columns: "
                f"{provided_vocabulary.feature_names} != "
                f"{expected_categorical_names}."
            )
        if provided_vocabulary is not None:
            for feature_name, source_column in (
                self._categorical_source_by_feature.items()
            ):
                vocabulary = provided_vocabulary.vocabulary_for(feature_name)
                self.cat_indexers[feature_name] = vocabulary
                self.cat_indexers[source_column] = vocabulary

        self.train_dataset = None
        self.val_dataset = None
        self._full_dataset: Optional[MultiPartExoTrainingDataset] = None
        self.resolved_split_mode: Optional[Literal["window", "multi"]] = None
        self._provided_categorical_vocabulary_artifact = provided_vocabulary
        self.categorical_vocabulary_artifact: Optional[
            CategoricalVocabularyArtifact
        ] = provided_vocabulary
        self.categorical_vocabulary_fingerprint: Optional[str] = (
            provided_vocabulary.fingerprint
            if provided_vocabulary is not None
            else None
        )

    def _build_training_dataset(
        self,
        frame: pl.DataFrame,
        *,
        include_categorical: bool,
    ) -> MultiPartExoTrainingDataset:
        return MultiPartExoTrainingDataset(
            frame,
            self.lookback,
            self.horizon,
            self.freq,
            id_col=self.id_col,
            date_col=self.date_col,
            qty_col=self.qty_col,
            past_exo_cont_cols=self.past_exo_cont_cols,
            past_exo_cat_cols=(
                self.past_exo_cat_cols
                if include_categorical
                else ()
            ),
            future_exo_cont_cols=self.future_exo_cont_cols,
            future_exo_cat_cols=(
                self.future_exo_cat_cols
                if include_categorical
                else ()
            ),
            future_exo_cb=self.future_exo_cb,
            date_indexer=self.date_indexer,
            cat_indexers=self.cat_indexers,
        )

    def _split_training_dataset(
        self,
        full_dataset: MultiPartExoTrainingDataset,
    ) -> tuple[Any, Any, Literal["window", "multi"]]:
        total_len = len(full_dataset)
        if total_len == 0:
            return full_dataset, full_dataset, "window"

        gen = torch.Generator().manual_seed(self.seed)

        if self.split_mode == "window":
            val_len = int(total_len * self.val_ratio)
            train_len = max(0, total_len - val_len)
            train_dataset, val_dataset = random_split(
                full_dataset,
                [train_len, val_len],
                generator=gen,
            )
            return train_dataset, val_dataset, "window"

        ids = list(full_dataset.id_to_indices.keys())
        if len(ids) <= 1:
            val_len = int(total_len * self.val_ratio)
            train_len = max(0, total_len - val_len)
            train_dataset, val_dataset = random_split(
                full_dataset,
                [train_len, val_len],
                generator=gen,
            )
            return train_dataset, val_dataset, "window"

        id_counts = {uid: len(full_dataset.id_to_indices[uid]) for uid in ids}
        total_windows = sum(id_counts.values())
        target_val_windows = int(total_windows * self.val_ratio)

        rng = np.random.default_rng(self.seed)
        rng.shuffle(ids)

        val_ids: List[str] = []
        cur = 0
        for uid in ids:
            if cur >= target_val_windows and len(val_ids) > 0:
                break
            val_ids.append(uid)
            cur += id_counts[uid]

        val_id_set = set(val_ids)
        train_ids = [uid for uid in ids if uid not in val_id_set]

        train_indices: List[int] = []
        for uid in train_ids:
            train_indices.extend(full_dataset.id_to_indices[uid])

        val_indices: List[int] = []
        for uid in val_ids:
            val_indices.extend(full_dataset.id_to_indices[uid])

        if len(train_indices) == 0 or len(val_indices) == 0:
            val_len = int(total_len * self.val_ratio)
            train_len = max(0, total_len - val_len)
            train_dataset, val_dataset = random_split(
                full_dataset,
                [train_len, val_len],
                generator=gen,
            )
            return train_dataset, val_dataset, "window"

        return (
            Subset(full_dataset, train_indices),
            Subset(full_dataset, val_indices),
            "multi",
        )

    @staticmethod
    def _rebind_split_dataset(
        split_dataset: Any,
        *,
        previous_full_dataset: MultiPartExoTrainingDataset,
        final_full_dataset: MultiPartExoTrainingDataset,
    ) -> Any:
        if split_dataset is previous_full_dataset:
            return final_full_dataset
        if (
            isinstance(split_dataset, Subset)
            and split_dataset.dataset is previous_full_dataset
        ):
            return Subset(
                final_full_dataset,
                tuple(int(index) for index in split_dataset.indices),
            )
        raise RuntimeError(
            "Cannot bind the resolved split to the final categorical dataset."
        )

    def _encode_categorical_frame(
        self,
        artifact: CategoricalVocabularyArtifact,
    ) -> pl.DataFrame:
        encoded = self._source_df
        for feature_name in artifact.feature_names:
            source_column = self._categorical_source_by_feature[feature_name]
            vocabulary = artifact.vocabulary_for(feature_name)
            encoded = encoded.with_columns(
                pl.Series(
                    name=feature_name,
                    values=vocabulary.map_series(self._source_df[source_column]),
                    dtype=pl.Int64,
                )
            )
        return encoded

    def setup(self):
        """Split first, then fit and apply categorical vocabularies."""
        provided_vocabulary = (
            self._provided_categorical_vocabulary_artifact
        )
        self.categorical_vocabulary_artifact = provided_vocabulary
        self.categorical_vocabulary_fingerprint = (
            provided_vocabulary.fingerprint
            if provided_vocabulary is not None
            else None
        )
        self.cat_indexers = {}

        split_full_dataset = self._build_training_dataset(
            self._source_df,
            include_categorical=False,
        )
        train_dataset, val_dataset, resolved_split_mode = (
            self._split_training_dataset(split_full_dataset)
        )
        self._full_dataset = split_full_dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.resolved_split_mode = resolved_split_mode

        if not self.past_exo_cat_cols and not self.future_exo_cat_cols:
            self.df = self._source_df
            return

        schema = ExogenousFeatureSchema.from_columns(
            past_cat=self.past_exo_cat_cols,
            future_cat=self.future_exo_cat_cols,
        )
        artifact = provided_vocabulary or self.fit_categorical_vocabulary(
            schema,
            feature_sources=self._categorical_source_by_feature,
        )
        artifact.bind_schema(schema)
        for feature_name, source_column in self._categorical_source_by_feature.items():
            vocabulary = artifact.vocabulary_for(feature_name)
            self.cat_indexers[feature_name] = vocabulary
            self.cat_indexers[source_column] = vocabulary

        encoded_frame = self._encode_categorical_frame(artifact)
        final_full_dataset = self._build_training_dataset(
            encoded_frame,
            include_categorical=True,
        )
        if final_full_dataset.index_map != split_full_dataset.index_map:
            raise RuntimeError(
                "Categorical encoding changed the resolved training window index."
            )

        self.df = encoded_frame
        self._full_dataset = final_full_dataset
        self.train_dataset = self._rebind_split_dataset(
            train_dataset,
            previous_full_dataset=split_full_dataset,
            final_full_dataset=final_full_dataset,
        )
        self.val_dataset = self._rebind_split_dataset(
            val_dataset,
            previous_full_dataset=split_full_dataset,
            final_full_dataset=final_full_dataset,
        )
        self.categorical_vocabulary_artifact = artifact
        self.categorical_vocabulary_fingerprint = artifact.fingerprint
        self.exogenous_schema = artifact.bind_schema(
            ExogenousFeatureSchema.from_columns(
                past_cont=self.past_exo_cont_cols,
                past_cat=self.past_exo_cat_cols,
                future_cont=self.future_exo_cont_cols,
                future_cat=self.future_exo_cat_cols,
            )
        )

    def _training_window_indices(self) -> tuple[int, ...]:
        if self.train_dataset is None or self._full_dataset is None:
            self.setup()
        if self.train_dataset is self._full_dataset:
            return tuple(range(len(self._full_dataset)))
        if (
            isinstance(self.train_dataset, Subset)
            and self.train_dataset.dataset is self._full_dataset
        ):
            return tuple(int(index) for index in self.train_dataset.indices)
        raise RuntimeError(
            "Training dataset is not backed by the expected full exogenous dataset."
        )

    def categorical_training_frame(
        self,
        columns: Sequence[str],
    ) -> pl.DataFrame:
        """Return raw categorical rows allowed to participate in vocabulary fitting.

        ``window`` mode uses the randomly selected training windows.
        ``multi`` mode uses every valid window from the selected training
        series IDs. Both modes exclude rows not referenced by a valid training
        window, including inference-only trailing rows.
        """
        requested = tuple(str(column).strip() for column in columns)
        if any(not column for column in requested):
            raise ValueError("Categorical training columns cannot contain empty names.")
        if len(set(requested)) != len(requested):
            raise ValueError(
                f"Categorical training columns contain duplicates: {requested}."
            )
        missing = tuple(
            column
            for column in requested
            if column not in self._source_df.columns
        )
        if missing:
            raise ValueError(
                "Categorical training columns are missing from the dataframe: "
                + ", ".join(missing)
            )

        if self.train_dataset is None or self._full_dataset is None:
            self.setup()
        if self._full_dataset is None or self.resolved_split_mode is None:
            raise RuntimeError("Data module setup did not resolve a training split.")

        normalized = normalize_temporal_frame(
            self._source_df,
            self.date_col,
            self.freq,
        ).with_columns(pl.col(self.id_col).cast(pl.String))
        selected_columns = list(
            dict.fromkeys((self.id_col, self.date_col, *requested))
        )
        train_indices = self._training_window_indices()
        if not train_indices:
            return normalized.select(selected_columns).head(0)

        positions_by_uid = self._full_dataset.source_row_positions_for_windows(
            train_indices
        )
        scoped_frames: List[pl.DataFrame] = []
        for group in normalized.partition_by(self.id_col, maintain_order=True):
            group = group.sort(self.date_col)
            uid = str(group[self.id_col][0])
            positions = positions_by_uid.get(uid)
            if not positions:
                continue
            scoped_frames.append(
                group
                .with_row_index("__source_position")
                .filter(pl.col("__source_position").is_in(positions))
                .select(selected_columns)
            )

        if not scoped_frames:
            return normalized.select(selected_columns).head(0)
        return pl.concat(scoped_frames, how="vertical").sort(
            [self.id_col, self.date_col]
        )

    def fit_categorical_vocabulary(
        self,
        schema: ExogenousFeatureSchema,
        *,
        feature_sources: Optional[Mapping[str, str]] = None,
    ) -> CategoricalVocabularyArtifact:
        """Fit categorical vocabularies from the resolved training scope only."""
        if not isinstance(schema, ExogenousFeatureSchema):
            raise TypeError("schema must be an ExogenousFeatureSchema.")
        feature_names = schema.categorical_feature_names
        provided_sources = dict(feature_sources or {})
        unexpected_features = tuple(
            feature_name
            for feature_name in provided_sources
            if feature_name not in feature_names
        )
        if unexpected_features:
            raise ValueError(
                "feature_sources contains names outside the categorical schema: "
                + ", ".join(unexpected_features)
            )

        source_by_feature: Dict[str, str] = {}
        for feature_name in feature_names:
            source_column = str(
                provided_sources.get(feature_name, feature_name)
            ).strip()
            if not source_column:
                raise ValueError(
                    f"Categorical source column for {feature_name!r} cannot be empty."
                )
            source_by_feature[feature_name] = source_column

        source_columns = tuple(dict.fromkeys(source_by_feature.values()))
        scope = self.categorical_training_frame(source_columns)
        if feature_names and scope.height == 0:
            raise ValueError(
                "Cannot fit categorical vocabularies because the training scope "
                "contains no source rows."
            )
        return CategoricalVocabularyArtifact.fit(
            {
                feature_name: scope[source_by_feature[feature_name]]
                for feature_name in feature_names
            },
            feature_names=feature_names,
        )

    def _attach_exogenous_contract(self, loader: DataLoader) -> DataLoader:
        schema = getattr(self, "exogenous_schema", None)
        if schema is not None:
            loader.exogenous_schema = schema
        if self.categorical_vocabulary_fingerprint is not None:
            loader.categorical_vocabulary_fingerprint = (
                self.categorical_vocabulary_fingerprint
            )
        if self.categorical_vocabulary_artifact is not None:
            loader.categorical_vocabulary_artifact = (
                self.categorical_vocabulary_artifact
            )
        return loader

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
            part_future_exo_fn=self.part_future_exo_fn,
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

        return self._attach_exogenous_contract(DataLoader(**loader_kwargs))

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
            part_future_exo_fn=self.part_future_exo_fn,
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

        return self._attach_exogenous_contract(DataLoader(**loader_kwargs))

    def get_inference_dataset_at_plan(
        self,
        plan_dt: date | datetime | int,
        *,
        series_ids: Optional[Sequence[str]] = None,
        unknown_series_policy: Literal["error", "ignore"] = "error",
    ) -> MultiPartExoAnchoredInferenceDataset:
        """Build the deterministic anchored-inference Dataset."""
        if (
            (self.past_exo_cat_cols or self.future_exo_cat_cols)
            and self.categorical_vocabulary_artifact is None
        ):
            self.setup()

        return MultiPartExoAnchoredInferenceDataset(
            df=self.df,
            lookback=self.lookback,
            horizon=self.horizon,
            plan_dt=plan_dt,
            freq=self.freq,
            id_col=self.id_col,
            date_col=self.date_col,
            qty_col=self.qty_col,
            past_exo_cont_cols=self.past_exo_cont_cols,
            past_exo_cat_cols=self.past_exo_cat_cols,
            future_exo_cont_cols=self.future_exo_cont_cols,
            future_exo_cat_cols=self.future_exo_cat_cols,
            series_ids=series_ids,
            unknown_series_policy=unknown_series_policy,
            fill_missing=self.fill_missing,
            target_back_steps=self.target_back_steps,
            future_exo_cb=self.future_exo_cb,
            date_indexer=self.date_indexer,
            cat_indexers=self.cat_indexers,
        )

    def get_inference_loader_at_plan(
        self,
        plan_dt: date | datetime | int,
        *,
        series_ids: Optional[Sequence[str]] = None,
        unknown_series_policy: Literal["error", "ignore"] = "error",
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        drop_last: bool = False,
    ) -> DataLoader:
        """Build a deterministic anchored-inference loader."""
        ds = self.get_inference_dataset_at_plan(
            plan_dt,
            series_ids=series_ids,
            unknown_series_policy=unknown_series_policy,
        )

        collate_fn = _build_train_collate_fn(
            horizon=self.horizon,
            future_exo_cb=self.future_exo_cb,
            part_future_exo_fn=self.part_future_exo_fn,
            cache_size=15000,
        )
        loader_kwargs: Dict[str, Any] = {
            "dataset": ds,
            "batch_size": self.batch_size if batch_size is None else int(batch_size),
            "shuffle": False,
            "drop_last": bool(drop_last),
            "num_workers": int(num_workers),
            "pin_memory": bool(pin_memory),
            "collate_fn": collate_fn,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(persistent_workers)
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        else:
            loader_kwargs["persistent_workers"] = False
        return self._attach_exogenous_contract(DataLoader(**loader_kwargs))

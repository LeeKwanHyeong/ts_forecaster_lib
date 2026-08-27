"""Nearest-neighbor retrieval over completed lifecycle curves."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

from modeling_module.data_loader.lifecycle_contracts import LifecycleSample
from modeling_module.models.CGMM.configs import (
    CGMMPreprocessingConfig,
)
from modeling_module.models.CGMM.contracts import (
    CGMMCorrectionState,
    CGMMPreprocessingState,
)
from modeling_module.models.CGMM.preprocessing import CGMMPreprocessor
from modeling_module.models.SimilarLifecycle.configs import (
    SimilarLifecycleConfig,
    SimilarLifecycleDistanceProfile,
    default_similar_lifecycle_preprocessing,
)
from modeling_module.models.SimilarLifecycle.contracts import (
    SIMILAR_LIFECYCLE_MODEL_ID,
    SIMILAR_LIFECYCLE_MODEL_KEY,
    SimilarLifecycleContractError,
    SimilarLifecyclePrediction,
    SimilarLifecycleRepositoryState,
)


class SimilarLifecycleModelError(SimilarLifecycleContractError):
    """Raised when retrieval fitting, restoration, or prediction fails."""


class SimilarLifecycleForecaster:
    """Retrieve future shapes from lifecycles with similar known conditions."""

    model_key = SIMILAR_LIFECYCLE_MODEL_KEY
    model_id = SIMILAR_LIFECYCLE_MODEL_ID

    def __init__(
        self,
        config: SimilarLifecycleConfig | Mapping[str, object] | None = None,
        *,
        preprocessing_config: CGMMPreprocessingConfig | None = None,
    ) -> None:
        self.config = (
            SimilarLifecycleConfig()
            if config is None
            else SimilarLifecycleConfig.from_config(config)
        )
        resolved_preprocessing = (
            preprocessing_config
            if preprocessing_config is not None
            else default_similar_lifecycle_preprocessing()
        )
        if not isinstance(resolved_preprocessing, CGMMPreprocessingConfig):
            raise TypeError(
                "preprocessing_config must be CGMMPreprocessingConfig"
            )
        if resolved_preprocessing.feature_profile not in {
            "static_observed_v1",
            "static_observed_m0_v1",
        }:
            raise SimilarLifecycleModelError(
                "Similar Lifecycle requires a static observed preprocessing "
                "profile"
            )
        self.preprocessor = CGMMPreprocessor(resolved_preprocessing)
        self._repository_state: SimilarLifecycleRepositoryState | None = None
        self._model_fingerprint: str | None = None
        self._correction_state: CGMMCorrectionState | None = None

    @property
    def is_fitted(self) -> bool:
        return self._repository_state is not None

    @property
    def preprocessing_state(self) -> CGMMPreprocessingState:
        return self.preprocessor.state

    @property
    def repository_state(self) -> SimilarLifecycleRepositoryState:
        if self._repository_state is None:
            raise SimilarLifecycleModelError("model has not been fitted")
        return self._repository_state

    @property
    def model_fingerprint(self) -> str:
        if self._model_fingerprint is None:
            raise SimilarLifecycleModelError("model has not been fitted")
        return self._model_fingerprint

    @property
    def correction_state(self) -> CGMMCorrectionState | None:
        return self._correction_state

    @property
    def distance_feature_names(self) -> tuple[str, ...]:
        return self.repository_state.distance_feature_names

    @property
    def training_sample_count(self) -> int:
        return len(self.repository_state.sample_ids)

    def fit(
        self,
        samples: Iterable[LifecycleSample],
        *,
        dataset_fingerprint: str,
    ) -> "SimilarLifecycleForecaster":
        if self.is_fitted or self.preprocessor.is_fitted:
            raise SimilarLifecycleModelError(
                "model is already fitted; create a new instance to refit"
            )
        materialized = tuple(samples)
        prepared = self.preprocessor.fit_transform(
            materialized,
            dataset_fingerprint=dataset_fingerprint,
        )
        if prepared.normalized_future is None:
            raise SimilarLifecycleModelError(
                "fitting requires completed future targets"
            )
        if len(materialized) <= self.config.neighbor_count:
            raise SimilarLifecycleModelError(
                "training sample count must exceed neighbor_count"
            )
        feature_indices = self._distance_feature_indices(
            self.preprocessing_state.condition_feature_names
        )
        self._repository_state = SimilarLifecycleRepositoryState(
            sample_ids=prepared.sample_ids,
            lifecycle_start_months=prepared.lifecycle_start_months,
            distance_feature_names=tuple(
                self.preprocessing_state.condition_feature_names[index]
                for index in feature_indices
            ),
            train_condition=prepared.condition_matrix[:, feature_indices],
            train_future_ratio=np.expm1(prepared.normalized_future),
        )
        self._model_fingerprint = self._build_model_fingerprint()
        return self

    def attach_correction(
        self,
        state: CGMMCorrectionState | None,
    ) -> "SimilarLifecycleForecaster":
        if not self.is_fitted:
            raise SimilarLifecycleModelError(
                "model must be fitted before correction is attached"
            )
        if state is not None and not isinstance(state, CGMMCorrectionState):
            raise TypeError("state must be a lifecycle correction state or None")
        self._correction_state = state
        return self

    def predict(
        self,
        samples: Iterable[LifecycleSample],
        *,
        apply_correction: bool = True,
    ) -> SimilarLifecyclePrediction:
        materialized = tuple(samples)
        prediction = self.predict_many(
            materialized,
            neighbor_counts=(self.config.neighbor_count,),
        )[self.config.neighbor_count]
        if not apply_correction or self._correction_state is None:
            return prediction
        from modeling_module.models.SimilarLifecycle.correction import (
            apply_similar_lifecycle_correction,
        )

        return apply_similar_lifecycle_correction(
            prediction,
            materialized,
            self._correction_state,
        )

    def predict_with_neighbors(
        self,
        samples: Iterable[LifecycleSample],
        *,
        apply_correction: bool = True,
    ) -> SimilarLifecyclePrediction:
        """Compatibility name for retrieval-evidence prediction."""

        return self.predict(samples, apply_correction=apply_correction)

    def predict_many(
        self,
        samples: Iterable[LifecycleSample],
        *,
        neighbor_counts: tuple[int, ...],
    ) -> dict[int, SimilarLifecyclePrediction]:
        if not neighbor_counts or len(set(neighbor_counts)) != len(
            neighbor_counts
        ):
            raise SimilarLifecycleModelError(
                "neighbor_counts must contain unique positive integers"
            )
        if any(
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or count > self.config.neighbor_count
            for count in neighbor_counts
        ):
            raise SimilarLifecycleModelError(
                "neighbor_counts must be positive and cannot exceed fitted capacity"
            )
        materialized = tuple(samples)
        prepared = self.preprocessor.transform(materialized)
        feature_index_by_name = {
            name: index
            for index, name in enumerate(
                self.preprocessing_state.condition_feature_names
            )
        }
        try:
            feature_indices = tuple(
                feature_index_by_name[name]
                for name in self.distance_feature_names
            )
        except KeyError as exc:
            raise SimilarLifecycleModelError(
                "distance feature schema differs from fitted preprocessing"
            ) from exc
        query_condition = prepared.condition_matrix[:, feature_indices]
        nearest_indices, nearest_distances = self._nearest_neighbor_lookup(
            query_condition,
            query_sample_ids=prepared.sample_ids,
            maximum_neighbor_count=max(neighbor_counts),
        )
        repository = self.repository_state
        predictions: dict[int, SimilarLifecyclePrediction] = {}
        for neighbor_count in neighbor_counts:
            nearest = nearest_indices[:, :neighbor_count]
            distances = nearest_distances[:, :neighbor_count]
            weights = self._distance_weights(distances)
            candidates = repository.train_future_ratio[nearest]
            mean_ratio = np.einsum("nk,nkh->nh", weights, candidates)
            variance_ratio = np.maximum(
                np.einsum("nk,nkh->nh", weights, np.square(candidates))
                - np.square(mean_ratio),
                0.0,
            )
            scale = prepared.quantity_scale[:, None]
            mean = mean_ratio * scale
            standard_deviation = np.sqrt(variance_ratio) * scale
            predictions[neighbor_count] = SimilarLifecyclePrediction(
                sample_ids=prepared.sample_ids,
                mean_forecast=mean,
                forecast_std=standard_deviation,
                lower_bound=np.maximum(
                    mean - self.config.interval_z * standard_deviation,
                    0.0,
                ),
                upper_bound=mean + self.config.interval_z * standard_deviation,
                neighbor_sample_ids=tuple(
                    tuple(repository.sample_ids[index] for index in row)
                    for row in nearest
                ),
                neighbor_weights=weights,
                neighbor_distances=distances,
                model_key=self.model_key,
                model_id=self.model_id,
                model_fingerprint=self.model_fingerprint,
                preprocessing_fingerprint=self.preprocessing_state.fingerprint,
            )
        return predictions

    def calibrated_distance_threshold(self, *, quantile: float) -> float:
        if isinstance(quantile, bool):
            raise TypeError("quantile must be a real number")
        try:
            resolved_quantile = float(quantile)
        except (TypeError, ValueError) as exc:
            raise TypeError("quantile must be a real number") from exc
        if not np.isfinite(resolved_quantile) or not 0.0 < resolved_quantile < 1.0:
            raise ValueError("quantile must be between zero and one")
        repository = self.repository_state
        _, distances = self._nearest_neighbor_lookup(
            repository.train_condition,
            query_sample_ids=repository.sample_ids,
            maximum_neighbor_count=1,
        )
        return float(np.quantile(distances[:, 0], resolved_quantile))

    @classmethod
    def restore(
        cls,
        *,
        config: SimilarLifecycleConfig,
        preprocessing_state: CGMMPreprocessingState,
        repository_state: SimilarLifecycleRepositoryState,
        expected_model_fingerprint: str,
        correction_state: CGMMCorrectionState | None = None,
    ) -> "SimilarLifecycleForecaster":
        if not isinstance(preprocessing_state, CGMMPreprocessingState):
            raise TypeError("preprocessing_state must be CGMMPreprocessingState")
        if preprocessing_state.feature_profile not in {
            "static_observed_v1",
            "static_observed_m0_v1",
        }:
            raise SimilarLifecycleModelError(
                "restored preprocessing profile is not static observed"
            )
        if not isinstance(repository_state, SimilarLifecycleRepositoryState):
            raise TypeError(
                "repository_state must be SimilarLifecycleRepositoryState"
            )
        instance = cls(
            config,
            preprocessing_config=CGMMPreprocessingConfig(
                quantity_scale_floor=preprocessing_state.quantity_scale_floor,
                standard_deviation_floor=(
                    preprocessing_state.standard_deviation_floor
                ),
                feature_profile=preprocessing_state.feature_profile,
            ),
        )
        instance.preprocessor = CGMMPreprocessor.from_state(preprocessing_state)
        expected_indices = instance._distance_feature_indices(
            preprocessing_state.condition_feature_names
        )
        expected_names = tuple(
            preprocessing_state.condition_feature_names[index]
            for index in expected_indices
        )
        if repository_state.distance_feature_names != expected_names:
            raise SimilarLifecycleModelError(
                "repository distance features disagree with model configuration"
            )
        if len(repository_state.sample_ids) <= config.neighbor_count:
            raise SimilarLifecycleModelError(
                "repository sample count must exceed neighbor_count"
            )
        instance._repository_state = repository_state
        actual_fingerprint = instance._build_model_fingerprint()
        if actual_fingerprint != expected_model_fingerprint:
            raise SimilarLifecycleModelError(
                "restored model fingerprint differs from the artifact"
            )
        instance._model_fingerprint = actual_fingerprint
        instance.attach_correction(correction_state)
        return instance

    def _nearest_neighbor_lookup(
        self,
        query_condition: np.ndarray,
        *,
        query_sample_ids: tuple[str, ...],
        maximum_neighbor_count: int,
    ) -> tuple[NDArray[np.int64], np.ndarray]:
        repository = self.repository_state
        train_condition = repository.train_condition
        query = np.asarray(query_condition, dtype=np.float64)
        if query.shape != (
            len(query_sample_ids),
            train_condition.shape[1],
        ) or not np.isfinite(query).all():
            raise SimilarLifecycleModelError(
                "query condition has an incompatible distance schema"
            )
        if (
            isinstance(maximum_neighbor_count, bool)
            or not isinstance(maximum_neighbor_count, int)
            or not 1 <= maximum_neighbor_count < len(repository.sample_ids)
        ):
            raise SimilarLifecycleModelError(
                "neighbor lookup count must be smaller than the repository"
            )
        train_id_to_index = {
            sample_id: index
            for index, sample_id in enumerate(repository.sample_ids)
        }
        nearest_indices = np.empty(
            (query.shape[0], maximum_neighbor_count),
            dtype=np.int64,
        )
        nearest_distances = np.empty(
            (query.shape[0], maximum_neighbor_count),
            dtype=np.float64,
        )
        train_squared_norm = np.square(train_condition).sum(axis=1)
        for start in range(0, query.shape[0], self.config.query_batch_size):
            stop = min(start + self.config.query_batch_size, query.shape[0])
            query_block = query[start:stop]
            squared_distance = (
                np.square(query_block).sum(axis=1, keepdims=True)
                + train_squared_norm[None, :]
                - 2.0 * query_block @ train_condition.T
            )
            np.maximum(squared_distance, 0.0, out=squared_distance)
            for local_index, sample_id in enumerate(
                query_sample_ids[start:stop]
            ):
                own_index = train_id_to_index.get(sample_id)
                if own_index is not None:
                    squared_distance[local_index, own_index] = np.inf
            nearest = np.argpartition(
                squared_distance,
                maximum_neighbor_count - 1,
                axis=1,
            )[:, :maximum_neighbor_count]
            nearest_distance = np.take_along_axis(
                squared_distance,
                nearest,
                axis=1,
            )
            order = np.argsort(nearest_distance, axis=1, kind="stable")
            nearest = np.take_along_axis(nearest, order, axis=1)
            nearest_distance = np.sqrt(
                np.take_along_axis(squared_distance, nearest, axis=1)
            )
            nearest_indices[start:stop] = nearest
            nearest_distances[start:stop] = nearest_distance
        return nearest_indices, nearest_distances

    def _distance_feature_indices(
        self,
        feature_names: tuple[str, ...],
    ) -> tuple[int, ...]:
        profile = self.config.distance_profile

        def include(name: str) -> bool:
            is_shape = name.startswith("observed_target_log_ratio_")
            is_scale = name == "log_quantity_scale"
            is_static = name.startswith(("static_log1p:", "static_missing:"))
            is_sales = name.startswith("observed_log1p:")
            is_category = name.startswith("category:")
            is_m0 = name == "lifecycle_start_month_ordinal"
            if profile is SimilarLifecycleDistanceProfile.DEMAND_SHAPE:
                return is_shape
            if profile is SimilarLifecycleDistanceProfile.DEMAND_SHAPE_SCALE:
                return is_shape or is_scale
            if profile is SimilarLifecycleDistanceProfile.DEMAND_SHAPE_CATEGORIES:
                return is_shape or is_category
            if profile is SimilarLifecycleDistanceProfile.DEMAND_SHAPE_STATIC:
                return (
                    is_shape
                    or is_scale
                    or is_static
                    or is_category
                    or is_m0
                )
            if profile is SimilarLifecycleDistanceProfile.DEMAND_SHAPE_SALES:
                return is_shape or is_scale or is_sales
            return (
                is_shape
                or is_scale
                or is_static
                or is_sales
                or is_category
                or is_m0
            )

        indices = tuple(
            index for index, name in enumerate(feature_names) if include(name)
        )
        if not indices:
            raise SimilarLifecycleModelError(
                "distance profile selected no condition features"
            )
        return indices

    def _distance_weights(self, distances: np.ndarray) -> np.ndarray:
        exact = distances <= self.config.distance_floor
        has_exact = exact.any(axis=1)
        weights = np.empty_like(distances)
        if has_exact.any():
            exact_rows = exact[has_exact].astype(np.float64)
            weights[has_exact] = exact_rows / exact_rows.sum(
                axis=1,
                keepdims=True,
            )
        if (~has_exact).any():
            inverse = 1.0 / distances[~has_exact]
            weights[~has_exact] = inverse / inverse.sum(axis=1, keepdims=True)
        return weights

    def _build_model_fingerprint(self) -> str:
        repository = self.repository_state
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "model_id": self.model_id,
                    "config": self.config.to_dict(),
                    "preprocessing_fingerprint": (
                        self.preprocessing_state.fingerprint
                    ),
                    "distance_feature_names": list(
                        repository.distance_feature_names
                    ),
                    "train_sample_ids": list(repository.sample_ids),
                    "train_lifecycle_start_months": [
                        value.isoformat()
                        for value in repository.lifecycle_start_months
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        )
        for array in (
            repository.train_condition,
            repository.train_future_ratio,
        ):
            contiguous = np.ascontiguousarray(array, dtype="<f8")
            digest.update(str(contiguous.shape).encode("ascii"))
            digest.update(contiguous.tobytes())
        return digest.hexdigest()


__all__ = [
    "SimilarLifecycleForecaster",
    "SimilarLifecycleModelError",
]

"""Leakage-safe preprocessing for lifecycle CGMM inputs."""

from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np

from modeling_module.data_loader.lifecycle_contracts import (
    LTB_OBSERVED_MONTHS,
    LifecycleFeatureSchema,
    LifecycleSample,
    LifecycleSamplePurpose,
)
from modeling_module.models.CGMM.configs import CGMMPreprocessingConfig
from modeling_module.models.CGMM.contracts import (
    CGMM_PREPROCESSING_ID,
    CGMMContractError,
    CGMMPreparedBatch,
    CGMMPreprocessingState,
    fingerprint_payload,
    require_sha256,
)


class CGMMPreprocessingError(CGMMContractError):
    """Raised when train-only preprocessing cannot be fitted or applied."""


def _materialize_samples(
    samples: Iterable[LifecycleSample],
    *,
    require_training: bool,
) -> tuple[LifecycleSample, ...]:
    if isinstance(samples, LifecycleSample):
        raise TypeError("samples must be an iterable of LifecycleSample values")
    materialized = tuple(samples)
    if not materialized:
        raise CGMMPreprocessingError("samples cannot be empty")
    if any(not isinstance(sample, LifecycleSample) for sample in materialized):
        raise TypeError("samples must contain only LifecycleSample values")
    sample_ids = tuple(sample.sample_id for sample in materialized)
    if len(set(sample_ids)) != len(sample_ids):
        raise CGMMPreprocessingError("sample_ids must be unique")
    schema = materialized[0].feature_schema
    if any(sample.feature_schema != schema for sample in materialized[1:]):
        raise CGMMPreprocessingError(
            "all samples must use the same ordered feature schema"
        )
    if require_training and any(
        sample.purpose is not LifecycleSamplePurpose.TRAINING
        or sample.future_target is None
        for sample in materialized
    ):
        raise CGMMPreprocessingError(
            "preprocessing fit requires completed training samples"
        )
    future_presence = {sample.future_target is not None for sample in materialized}
    if len(future_presence) > 1:
        raise CGMMPreprocessingError(
            "a CGMM batch cannot mix labeled and unlabeled samples"
        )
    return materialized


def _category_token(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return f"bool:{str(value).lower()}"
    if isinstance(value, int):
        return f"int:{value}"
    if isinstance(value, float):
        if not np.isfinite(value):
            raise CGMMPreprocessingError(
                "categorical values cannot contain NaN or infinity"
            )
        return f"float:{format(value, '.17g')}"
    if isinstance(value, str):
        return f"str:{value}"
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CGMMPreprocessingError(
            "categorical values must be deterministic JSON values"
        ) from exc
    return f"json:{encoded}"


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def _numeric_slots(
    sample: LifecycleSample,
) -> tuple[tuple[str, ...], tuple[float | None, ...]]:
    schema = sample.feature_schema
    names: list[str] = []
    values: list[float | None] = []
    for name, value in zip(
        schema.static_cont_names,
        sample.static_cont,
        strict=True,
    ):
        names.append(f"static:{name}")
        values.append(value)
    for month, row in enumerate(sample.observed_cont):
        for name, value in zip(schema.observed_cont_names, row, strict=True):
            names.append(f"observed:{name}:m{month:02d}")
            values.append(value)
    for month, row in enumerate(sample.known_future_cont):
        for name, value in zip(
            schema.known_future_cont_names,
            row,
            strict=True,
        ):
            names.append(f"known_future:{name}:m{month:02d}")
            values.append(value)
    return tuple(names), tuple(values)


def _categorical_slots(
    sample: LifecycleSample,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    schema = sample.feature_schema
    names: list[str] = []
    values: list[str] = []
    for name, value in zip(
        schema.static_cat_names,
        sample.static_cat,
        strict=True,
    ):
        names.append(f"static:{name}")
        values.append(_category_token(value))
    for month, row in enumerate(sample.observed_cat):
        for name, value in zip(schema.observed_cat_names, row, strict=True):
            names.append(f"observed:{name}:m{month:02d}")
            values.append(_category_token(value))
    for month, row in enumerate(sample.known_future_cat):
        for name, value in zip(
            schema.known_future_cat_names,
            row,
            strict=True,
        ):
            names.append(f"known_future:{name}:m{month:02d}")
            values.append(_category_token(value))
    return tuple(names), tuple(values)


def _static_observed_categorical_slots(
    sample: LifecycleSample,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    schema = sample.feature_schema
    if (
        schema.observed_cat_names
        or schema.known_future_cont_names
        or schema.known_future_cat_names
    ):
        raise CGMMPreprocessingError(
            "static_observed_v1 supports static categorical, static "
            "continuous, and observed continuous features only"
        )
    names: list[str] = []
    values: list[str] = []
    for name, value in zip(
        schema.static_cat_names,
        sample.static_cat,
        strict=True,
    ):
        if not isinstance(value, str) or not value:
            raise CGMMPreprocessingError(
                "static_observed_v1 categorical values must be non-empty text"
            )
        names.append(f"static:{name}")
        values.append(value)
    return tuple(names), tuple(values)


def _validate_static_observed_numeric_values(
    samples: tuple[LifecycleSample, ...],
) -> None:
    for sample in samples:
        for value in sample.static_cont:
            if value is not None and value < 0.0:
                raise CGMMPreprocessingError(
                    "static_observed_v1 continuous values must be non-negative"
                )
        for row in sample.observed_cont:
            if any(value is None or value < 0.0 for value in row):
                raise CGMMPreprocessingError(
                    "static_observed_v1 observed continuous values must be "
                    "present and non-negative"
                )


class CGMMPreprocessor:
    """Fit feature statistics once and reuse them without mutation."""

    def __init__(
        self,
        config: CGMMPreprocessingConfig | None = None,
        *,
        state: CGMMPreprocessingState | None = None,
    ) -> None:
        self.config = config or CGMMPreprocessingConfig()
        if not isinstance(self.config, CGMMPreprocessingConfig):
            raise TypeError("config must be CGMMPreprocessingConfig")
        if state is not None and not isinstance(state, CGMMPreprocessingState):
            raise TypeError("state must be CGMMPreprocessingState")
        self._state = state
        if state is not None and (
            state.quantity_scale_floor != self.config.quantity_scale_floor
            or state.standard_deviation_floor
            != self.config.standard_deviation_floor
            or state.feature_profile != self.config.feature_profile
        ):
            raise CGMMPreprocessingError(
                "preprocessing config does not match restored state"
            )

    @classmethod
    def from_state(cls, state: CGMMPreprocessingState) -> "CGMMPreprocessor":
        return cls(
            CGMMPreprocessingConfig(
                quantity_scale_floor=state.quantity_scale_floor,
                standard_deviation_floor=state.standard_deviation_floor,
                feature_profile=state.feature_profile,
            ),
            state=state,
        )

    @property
    def is_fitted(self) -> bool:
        return self._state is not None

    @property
    def state(self) -> CGMMPreprocessingState:
        if self._state is None:
            raise CGMMPreprocessingError("preprocessor has not been fitted")
        return self._state

    def fit(
        self,
        samples: Iterable[LifecycleSample],
        *,
        dataset_fingerprint: str,
    ) -> CGMMPreprocessingState:
        if self._state is not None:
            raise CGMMPreprocessingError(
                "preprocessor is already fitted; create a new instance to refit"
            )
        materialized = _materialize_samples(samples, require_training=True)
        dataset_fingerprint = require_sha256(
            dataset_fingerprint,
            field_name="dataset_fingerprint",
        )
        schema = materialized[0].feature_schema
        numeric_names, _ = _numeric_slots(materialized[0])
        categorical_names, _ = _categorical_slots(materialized[0])

        numeric_rows = np.asarray(
            [
                [np.nan if value is None else float(value) for value in _numeric_slots(sample)[1]]
                for sample in materialized
            ],
            dtype=np.float64,
        )
        if numeric_rows.shape != (len(materialized), len(numeric_names)):
            raise CGMMPreprocessingError("numeric feature schema is inconsistent")
        if self.config.feature_profile == "static_observed_v1":
            _validate_static_observed_numeric_values(materialized)
            transformed_numeric = numeric_rows
        else:
            transformed_numeric = _signed_log1p(numeric_rows)
        fill_values: list[float] = []
        for column in transformed_numeric.T:
            observed = column[np.isfinite(column)]
            fill_values.append(
                float(np.median(observed)) if observed.size else 0.0
            )

        categorical_slot_reader = (
            _static_observed_categorical_slots
            if self.config.feature_profile == "static_observed_v1"
            else _categorical_slots
        )
        categorical_names, _ = categorical_slot_reader(materialized[0])
        categorical_rows = tuple(
            categorical_slot_reader(sample)[1] for sample in materialized
        )
        vocabularies = tuple(
            tuple(sorted({row[index] for row in categorical_rows}))
            for index in range(len(categorical_names))
        )
        raw_condition, feature_names, _ = self._raw_condition(
            materialized,
            numeric_slot_names=numeric_names,
            numeric_fill_values=tuple(fill_values),
            categorical_slot_names=categorical_names,
            categorical_vocabularies=vocabularies,
        )
        means = raw_condition.mean(axis=0)
        raw_scales = raw_condition.std(axis=0)
        scales = np.where(
            raw_scales >= self.config.standard_deviation_floor,
            raw_scales,
            1.0,
        )
        payload: dict[str, Any] = {
            "contract_id": CGMM_PREPROCESSING_ID,
            "dataset_fingerprint": dataset_fingerprint,
            "feature_schema": schema.to_dict(),
            "feature_schema_fingerprint": schema.fingerprint,
            "feature_profile": self.config.feature_profile,
            "quantity_scale_floor": self.config.quantity_scale_floor,
            "standard_deviation_floor": self.config.standard_deviation_floor,
            "numeric_slot_names": list(numeric_names),
            "numeric_fill_values": fill_values,
            "categorical_slot_names": list(categorical_names),
            "categorical_vocabularies": [
                list(vocabulary) for vocabulary in vocabularies
            ],
            "condition_feature_names": list(feature_names),
            "condition_means": means.tolist(),
            "condition_scales": scales.tolist(),
        }
        self._state = CGMMPreprocessingState(
            contract_id=CGMM_PREPROCESSING_ID,
            dataset_fingerprint=dataset_fingerprint,
            feature_schema=schema,
            feature_profile=self.config.feature_profile,
            quantity_scale_floor=self.config.quantity_scale_floor,
            standard_deviation_floor=self.config.standard_deviation_floor,
            numeric_slot_names=numeric_names,
            numeric_fill_values=tuple(fill_values),
            categorical_slot_names=categorical_names,
            categorical_vocabularies=vocabularies,
            condition_feature_names=feature_names,
            condition_means=tuple(means.tolist()),
            condition_scales=tuple(scales.tolist()),
            fingerprint=fingerprint_payload(payload),
        )
        return self._state

    def transform(
        self,
        samples: Iterable[LifecycleSample],
    ) -> CGMMPreparedBatch:
        materialized = _materialize_samples(samples, require_training=False)
        state = self.state
        if materialized[0].feature_schema != state.feature_schema:
            raise CGMMPreprocessingError(
                "input feature schema does not match the fitted artifact"
            )
        raw_condition, feature_names, quantity_scale = self._raw_condition(
            materialized,
            numeric_slot_names=state.numeric_slot_names,
            numeric_fill_values=state.numeric_fill_values,
            categorical_slot_names=state.categorical_slot_names,
            categorical_vocabularies=state.categorical_vocabularies,
        )
        if feature_names != state.condition_feature_names:
            raise CGMMPreprocessingError(
                "transformed condition schema differs from fitted state"
            )
        condition = (
            raw_condition - np.asarray(state.condition_means)[None, :]
        ) / np.asarray(state.condition_scales)[None, :]
        future = None
        if materialized[0].future_target is not None:
            future_values = np.asarray(
                [sample.future_target for sample in materialized],
                dtype=np.float64,
            )
            future = np.log1p(future_values / quantity_scale[:, None])
        return CGMMPreparedBatch(
            sample_ids=tuple(sample.sample_id for sample in materialized),
            lifecycle_start_months=tuple(
                sample.lifecycle_start_month for sample in materialized
            ),
            quantity_scale=quantity_scale,
            condition_matrix=condition,
            normalized_future=future,
            preprocessing_fingerprint=state.fingerprint,
        )

    def fit_transform(
        self,
        samples: Iterable[LifecycleSample],
        *,
        dataset_fingerprint: str,
    ) -> CGMMPreparedBatch:
        materialized = tuple(samples)
        self.fit(materialized, dataset_fingerprint=dataset_fingerprint)
        return self.transform(materialized)

    def _raw_condition(
        self,
        samples: tuple[LifecycleSample, ...],
        *,
        numeric_slot_names: tuple[str, ...],
        numeric_fill_values: tuple[float, ...],
        categorical_slot_names: tuple[str, ...],
        categorical_vocabularies: tuple[tuple[str, ...], ...],
    ) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
        if self.config.feature_profile == "static_observed_v1":
            return self._raw_static_observed_condition(
                samples,
                numeric_slot_names=numeric_slot_names,
                numeric_fill_values=numeric_fill_values,
                categorical_slot_names=categorical_slot_names,
                categorical_vocabularies=categorical_vocabularies,
            )
        sample_count = len(samples)
        observed_target = np.asarray(
            [sample.observed_target for sample in samples],
            dtype=np.float64,
        )
        quantity_scale = np.maximum(
            observed_target.mean(axis=1),
            self.config.quantity_scale_floor,
        )
        blocks: list[np.ndarray] = [
            np.log1p(observed_target / quantity_scale[:, None]),
            np.log1p(quantity_scale)[:, None],
        ]
        feature_names: list[str] = [
            *(f"observed_target_log_ratio_m{month:02d}" for month in range(LTB_OBSERVED_MONTHS)),
            "log_quantity_scale",
        ]

        actual_numeric_names, _ = _numeric_slots(samples[0])
        if actual_numeric_names != numeric_slot_names:
            raise CGMMPreprocessingError("numeric feature order mismatch")
        if numeric_slot_names:
            numeric = np.asarray(
                [
                    [np.nan if value is None else float(value) for value in _numeric_slots(sample)[1]]
                    for sample in samples
                ],
                dtype=np.float64,
            )
            transformed = _signed_log1p(numeric)
            missing = ~np.isfinite(transformed)
            filled = np.where(
                missing,
                np.asarray(numeric_fill_values, dtype=np.float64)[None, :],
                transformed,
            )
            blocks.extend((filled, missing.astype(np.float64)))
            feature_names.extend(
                f"continuous_signed_log1p:{name}" for name in numeric_slot_names
            )
            feature_names.extend(
                f"continuous_missing:{name}" for name in numeric_slot_names
            )

        actual_categorical_names, _ = _categorical_slots(samples[0])
        if actual_categorical_names != categorical_slot_names:
            raise CGMMPreprocessingError("categorical feature order mismatch")
        if categorical_slot_names:
            categorical_rows = tuple(
                _categorical_slots(sample)[1] for sample in samples
            )
            category_blocks: list[np.ndarray] = []
            for index, (slot_name, vocabulary) in enumerate(
                zip(
                    categorical_slot_names,
                    categorical_vocabularies,
                    strict=True,
                )
            ):
                values = np.asarray(
                    [row[index] for row in categorical_rows],
                    dtype=np.str_,
                )
                for category in vocabulary:
                    category_blocks.append(
                        (values == category).astype(np.float64)[:, None]
                    )
                    feature_names.append(f"category:{slot_name}={category}")
                unknown = ~np.isin(values, np.asarray(vocabulary, dtype=np.str_))
                category_blocks.append(unknown.astype(np.float64)[:, None])
                feature_names.append(f"category:{slot_name}=<UNK>")
            blocks.extend(category_blocks)

        condition = np.concatenate(blocks, axis=1)
        if condition.shape[0] != sample_count or not np.isfinite(condition).all():
            raise CGMMPreprocessingError(
                "preprocessed condition features must be finite"
            )
        return condition, tuple(feature_names), quantity_scale

    def _raw_static_observed_condition(
        self,
        samples: tuple[LifecycleSample, ...],
        *,
        numeric_slot_names: tuple[str, ...],
        numeric_fill_values: tuple[float, ...],
        categorical_slot_names: tuple[str, ...],
        categorical_vocabularies: tuple[tuple[str, ...], ...],
    ) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
        _validate_static_observed_numeric_values(samples)
        schema = samples[0].feature_schema
        sample_count = len(samples)
        observed_target = np.asarray(
            [sample.observed_target for sample in samples],
            dtype=np.float64,
        )
        quantity_scale = np.maximum(
            observed_target.mean(axis=1),
            self.config.quantity_scale_floor,
        )
        static = np.asarray(
            [
                [np.nan if value is None else float(value) for value in sample.static_cont]
                for sample in samples
            ],
            dtype=np.float64,
        )
        observed = np.asarray(
            [sample.observed_cont for sample in samples],
            dtype=np.float64,
        )
        expected_numeric_names, _ = _numeric_slots(samples[0])
        if expected_numeric_names != numeric_slot_names:
            raise CGMMPreprocessingError("numeric feature order mismatch")
        static_width = len(schema.static_cont_names)
        static_missing = np.isnan(static)
        static_filled = np.where(
            static_missing,
            np.asarray(
                numeric_fill_values[:static_width],
                dtype=np.float64,
            )[None, :],
            static,
        )
        blocks: list[np.ndarray] = [
            np.log1p(observed_target / quantity_scale[:, None]),
            np.log1p(quantity_scale)[:, None],
        ]
        feature_names: list[str] = [
            *(
                f"observed_target_log_ratio_m{month:02d}"
                for month in range(LTB_OBSERVED_MONTHS)
            ),
            "log_quantity_scale",
        ]
        if static_width:
            blocks.extend(
                (
                    np.log1p(static_filled),
                    static_missing.astype(np.float64),
                )
            )
            feature_names.extend(
                f"static_log1p:{name}" for name in schema.static_cont_names
            )
            feature_names.extend(
                f"static_missing:{name}" for name in schema.static_cont_names
            )
        if schema.observed_cont_names:
            blocks.append(np.log1p(observed).reshape(sample_count, -1))
            feature_names.extend(
                f"observed_log1p:{name}:m{month:02d}"
                for month in range(LTB_OBSERVED_MONTHS)
                for name in schema.observed_cont_names
            )

        actual_categorical_names, _ = _static_observed_categorical_slots(
            samples[0]
        )
        if actual_categorical_names != categorical_slot_names:
            raise CGMMPreprocessingError("categorical feature order mismatch")
        categorical_rows = tuple(
            _static_observed_categorical_slots(sample)[1]
            for sample in samples
        )
        for index, (slot_name, vocabulary) in enumerate(
            zip(
                categorical_slot_names,
                categorical_vocabularies,
                strict=True,
            )
        ):
            values = np.asarray(
                [row[index] for row in categorical_rows],
                dtype=np.str_,
            )
            display_name = slot_name.removeprefix("static:")
            for category in vocabulary[1:]:
                blocks.append((values == category).astype(np.float64)[:, None])
                feature_names.append(
                    f"category:{display_name}={category}"
                )
            unknown = ~np.isin(values, np.asarray(vocabulary, dtype=np.str_))
            blocks.append(unknown.astype(np.float64)[:, None])
            feature_names.append(f"category:{display_name}=<UNK>")

        condition = np.concatenate(blocks, axis=1)
        if condition.shape[0] != sample_count or not np.isfinite(condition).all():
            raise CGMMPreprocessingError(
                "preprocessed condition features must be finite"
            )
        return condition, tuple(feature_names), quantity_scale


__all__ = [
    "CGMMPreprocessingError",
    "CGMMPreprocessor",
]

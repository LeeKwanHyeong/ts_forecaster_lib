"""학습 내부 purged OOF와 검증 calibration. 기존 backbone을 학습하거나 호출하지 않음."""

from dataclasses import dataclass
from copy import deepcopy

import numpy as np

from .native.basis import coarse_operator, tapered_basis, coarse_values, coefficients, reconstruct
from .native.coarse_future import CoarseEstimator, RidgeView, extrapolated
from .native.stagewise import fit_mapping, apply_mapping, reliability
from .native.calibration import calibrate


def purged_folds(query, folds, minimum):
    """held 블록의 raw [origin-L, origin+H-1]과 겹치는 학습 창을 같은 시계열에서 제거."""
    times = np.unique(query.origins)
    if len(times) < folds:
        raise ValueError("OOF 블록을 나눌 서로 다른 예측 시작 위치가 부족합니다.")
    start = query.origins - query.history.shape[1]
    end = query.origins + query.backbone_prediction.shape[1] - 1
    ids = np.asarray(query.series_ids)
    result = []
    for time_block in np.array_split(times, folds):
        held = np.flatnonzero(np.isin(query.origins, time_block))
        allowed = np.ones(len(start), dtype=bool)
        for name in np.unique(ids[held]):
            held_series = held[ids[held] == name]
            overlap = (ids == name) & (start <= end[held_series].max()) & (end >= start[held_series].min())
            allowed[overlap] = False
        train = np.flatnonzero(allowed)
        if len(train) < minimum:
            raise ValueError("raw 구간 중첩을 제거한 OOF 학습 창이 부족합니다. 이력 또는 query 간격을 늘려야 합니다.")
        result.append((train, held))
    return result


def first_crossfit(data, config, operator, basis):
    query = data.query
    coarse = coarse_values(query.backbone_prediction, operator)
    target = coefficients(data.target - query.backbone_prediction, basis)
    discrepancy = extrapolated(query.history, operator, config.period) - np.tile(coarse, (1, 1, 2))
    raw = np.zeros_like(target)
    learned = np.zeros_like(coarse)
    geometry = np.tile(operator @ basis, (2, 1))
    folds = purged_folds(query, config.folds, config.min_fold_train)
    for train, held in folds:
        mapping = fit_mapping(discrepancy[train], target[train], geometry, config)
        estimator = CoarseEstimator.fit(query.history[train], query.origins[train], data.target[train],
                                        operator, config.period, config.ridge_strength)
        raw[held] = apply_mapping(discrepancy[held], mapping)
        learned[held] = estimator.predict(query.history[held], query.origins[held])
    return raw, learned, folds


def second_crossfit(data, config, operator, basis, folds):
    """각 outer held 블록은 내부 coarse·gain·두 매핑의 적합 어디에도 사용하지 않는 nested OOF."""
    query = data.query
    target = coefficients(data.target - query.backbone_prediction, basis)
    raw, residual_target = np.zeros_like(target), np.zeros_like(target)
    geometry = operator @ basis
    for train, held in folds:
        training = data.take(train)
        inner_raw, inner_learned, inner_folds = first_crossfit(training, config, operator, basis)
        inner_target = target[train]
        gain = reliability(inner_raw, inner_target, [block for _, block in inner_folds], config.gain_max)
        inner_first = inner_raw * gain
        inner_base = coarse_values(training.query.backbone_prediction, operator)
        inner_discrepancy = inner_learned - inner_base - np.einsum("qk,nck->ncq", geometry, inner_first)
        second_mapping = fit_mapping(inner_discrepancy, inner_target - inner_first, geometry, config)
        first_discrepancy = extrapolated(training.query.history, operator, config.period) - np.tile(inner_base, (1, 1, 2))
        first_mapping = fit_mapping(first_discrepancy, inner_target, np.tile(geometry, (2, 1)), config)
        estimator = CoarseEstimator.fit(training.query.history, training.query.origins, training.target,
                                        operator, config.period, config.ridge_strength)
        held_base = coarse_values(query.backbone_prediction[held], operator)
        held_discrepancy = extrapolated(query.history[held], operator, config.period) - np.tile(held_base, (1, 1, 2))
        held_first = apply_mapping(held_discrepancy, first_mapping) * gain
        held_learned = estimator.predict(query.history[held], query.origins[held])
        held_second_d = held_learned - held_base - np.einsum("qk,nck->ncq", geometry, held_first)
        raw[held] = apply_mapping(held_second_d, second_mapping)
        residual_target[held] = target[held] - held_first
    return raw, residual_target


@dataclass(frozen=True)
class MORAState:
    operator: np.ndarray
    basis: np.ndarray
    estimator: CoarseEstimator
    first_mapping: np.ndarray
    second_mapping: np.ndarray
    first_gain: np.ndarray
    second_gain: np.ndarray
    calibration: np.ndarray
    metadata: dict


def canonical_state(state):
    """메모리 배치 차이로 재로딩 전후 BLAS 반올림이 달라지지 않도록 C-order로 고정."""
    def freeze(value):
        result = np.array(value, dtype=np.float64, order="C", copy=True)
        result.setflags(write=False)
        return result
    views = []
    for view in (state.estimator.absolute, state.estimator.relative):
        views.append(RidgeView(**{name: freeze(getattr(view, name)) for name in
                                  ("feature_mean", "feature_scale", "target_mean", "target_scale", "weight")}))
    return MORAState(**{name: freeze(getattr(state, name)) for name in
                        ("operator", "basis", "first_mapping", "second_mapping", "first_gain", "second_gain", "calibration")},
                      estimator=CoarseEstimator(*views, state.estimator.period), metadata=deepcopy(state.metadata))


def predict_coefficients(state, history, base, origins, config):
    coarse = coarse_values(base, state.operator)
    geometry = state.operator @ state.basis
    first_d = extrapolated(history, state.operator, config.period) - np.tile(coarse, (1, 1, 2))
    first = apply_mapping(first_d, state.first_mapping) * state.first_gain
    learned = state.estimator.predict(history, origins)
    second_d = learned - coarse - np.einsum("qk,nck->ncq", geometry, first)
    second = apply_mapping(second_d, state.second_mapping) * state.second_gain
    return first, second


def fit_state(train, validation, config):
    operator = coarse_operator(config.horizon, config.coarse_count)
    basis = tapered_basis(config.horizon, config.correction_rank, config.taper_steps)
    raw_first, learned_oof, folds = first_crossfit(train, config, operator, basis)
    target = coefficients(train.target - train.query.backbone_prediction, basis)
    blocks = [held for _, held in folds]
    first_gain = reliability(raw_first, target, blocks, config.gain_max)
    first_oof = raw_first * first_gain
    coarse = coarse_values(train.query.backbone_prediction, operator)
    geometry = operator @ basis
    first_d = extrapolated(train.query.history, operator, config.period) - np.tile(coarse, (1, 1, 2))
    first_mapping = fit_mapping(first_d, target, np.tile(geometry, (2, 1)), config)
    second_d = learned_oof - coarse - np.einsum("qk,nck->ncq", geometry, first_oof)
    second_mapping = fit_mapping(second_d, target - first_oof, geometry, config)
    raw_second, second_target = second_crossfit(train, config, operator, basis, folds)
    second_gain = reliability(raw_second, second_target, blocks, config.gain_max, conservative=True)
    estimator = CoarseEstimator.fit(train.query.history, train.query.origins, train.target,
                                    operator, config.period, config.ridge_strength)
    metadata = {"implementation": "paper_independent_v1", "source_code_copied": False,
                "signature": train.query.signature, "scope": config.scope,
                "fit_series_ids": sorted(set(train.query.series_ids)),
                "train_provenance": train.backbone_provenance, "oof_manifest_sha256": train.oof_manifest_sha256,
                "train_queries": len(train.target), "calibration_queries": len(validation.target),
                "train_target_end": int(train.query.origins.max() + config.horizon - 1),
                "calibration_target_end": int(validation.query.origins.max() + config.horizon - 1),
                "folds": [{"fit_count": len(fit), "held_count": len(held), "held_indices": held.tolist()}
                          for fit, held in folds], "purged_raw_overlap": True,
                "second_gain_nested_crossfit": True, "hyperparameter_search_executed": False}
    preliminary = MORAState(operator, basis, estimator, first_mapping, second_mapping, first_gain, second_gain,
                            np.ones((train.target.shape[2], config.correction_rank)), metadata)
    first, second = predict_coefficients(preliminary, validation.query.history, validation.query.backbone_prediction,
                                        validation.query.origins, config)
    scales, calibration_details = calibrate(validation.query.backbone_prediction, validation.target,
                                           first + second, basis, config, validation.query.origins)
    metadata["calibration_details"] = calibration_details
    return MORAState(operator, basis, estimator, first_mapping, second_mapping, first_gain, second_gain, scales, metadata)

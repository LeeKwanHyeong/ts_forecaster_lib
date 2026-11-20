"""논문 수식의 기하·특징·모멘트 계산을 독립 계산과 대조. 저자 코드와의 대조가 아님."""

import numpy as np
import pytest

from modeling_module.forecast_adapters.MORA import MORAConfig
from modeling_module.forecast_adapters.MORA.native.basis import coarse_operator, tapered_basis, coefficients, reconstruct
from modeling_module.forecast_adapters.MORA.native.features import compact_features
from modeling_module.forecast_adapters.MORA.native.coarse_future import CoarseEstimator
from modeling_module.forecast_adapters.MORA.native.stagewise import fit_mapping, reliability


@pytest.mark.parametrize("horizon,rank,count", [(27, 12, 6), (8, 5, 2), (52, 20, 13)])
def test_coarse_geometry_and_boundary_are_exact(horizon, rank, count):
    operator = coarse_operator(horizon, count)
    basis = tapered_basis(horizon, rank, min(4, horizon))
    np.testing.assert_allclose(operator.sum(1), 1)
    assert np.all((operator > 0).sum(0) == 1)
    np.testing.assert_array_equal(basis[0], 0)
    np.testing.assert_allclose(basis.T @ basis, np.eye(rank), atol=1e-12)
    value = np.random.default_rng(1).normal(size=(5, horizon, 2))
    projection = reconstruct(coefficients(value, basis), basis)
    np.testing.assert_array_equal(projection[:, 0], 0)
    assert np.linalg.norm(projection) <= np.linalg.norm(value) + 1e-12


@pytest.mark.parametrize("length", [24, 52, 96])
def test_features_use_only_history_and_have_compact_78_dimensions(length):
    history = np.random.default_rng(4).normal(size=(8, length, 2))
    origins = np.arange(8) * 100
    original = compact_features(history, origins, 13)
    assert original.shape == (8, 2, 78)
    altered = compact_features(history, origins + 1, 13)
    np.testing.assert_array_equal(original[..., :-6], altered[..., :-6])
    assert not np.array_equal(original[..., -6:], altered[..., -6:])
    centered = history - history[:, -1:]
    shifted = history + 20
    np.testing.assert_allclose(compact_features(centered, origins, 13),
                               compact_features(shifted - shifted[:, -1:], origins, 13), atol=1e-12)


def test_ridge_estimates_the_future_quantity_not_a_backbone_residual():
    history = np.ones((24, 52, 1)) * 5
    target = np.ones((24, 27, 1)) * 9
    estimator = CoarseEstimator.fit(history, np.arange(24), target, coarse_operator(27, 6), 13, 10)
    prediction = estimator.predict(history, np.arange(24))
    np.testing.assert_allclose(prediction, 9)


def test_second_order_mapping_matches_equation_36_by_independent_algebra():
    rng = np.random.default_rng(42)
    d, z, geometry = rng.normal(size=(40, 1, 6)), rng.normal(size=(40, 1, 4)), rng.normal(size=(6, 4))
    cfg = MORAConfig(shrinkage=1, structured_mix=0.25, noise_scale=2, weight_power=0)
    actual = fit_mapping(d, z, geometry, cfg)[0]
    x, y = d[:, 0], z[:, 0]
    zz = np.diag(np.diag(y.T @ y / len(y)))
    eta = x - y @ geometry.T
    noise = np.diag(np.diag(eta.T @ eta / len(y)))
    covariance = 0.75 * (geometry @ zz @ geometry.T + 2 * noise) + 0.25 * x.T @ x / len(y)
    numerator = 0.75 * zz @ geometry.T + 0.25 * y.T @ x / len(y)
    epsilon = 1e-6 * max(np.trace(covariance) / 6, 1e-12)
    expected = np.linalg.solve(covariance + epsilon * np.eye(6), numerator.T).T
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_zero_moments_and_negative_reliability_do_not_produce_false_signal():
    config = MORAConfig()
    zeros = fit_mapping(np.zeros((8, 1, 6)), np.zeros((8, 1, 4)), np.ones((6, 4)), config)
    np.testing.assert_array_equal(zeros, 0)
    prediction = np.ones((8, 1, 4))
    target = prediction.copy()
    target[:4] *= -1
    conservative = reliability(prediction, target, [np.arange(4), np.arange(4, 8)], 1.5, True)
    np.testing.assert_array_equal(conservative, 0)

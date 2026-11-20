"""합성 종단 검증, nested OOF, 보정기 저장 무결성 및 기존 모델 경계."""

from dataclasses import replace
import json

import numpy as np
import pytest

from modeling_module.forecast_adapters.MORA import MORAAdapter, MORAConfig
from modeling_module.forecast_adapters.MORA.checkpoints import state_digest
from modeling_module.forecast_adapters.MORA.fitting import first_crossfit, second_crossfit
from modeling_module.forecast_adapters.MORA.native.basis import coarse_operator, tapered_basis
from model_test.mora.synthetic import make_synthetic, point_metrics
from model_test.mora.run_synthetic import run


@pytest.fixture(scope="module")
def fitted():
    config = MORAConfig(period=13, correction_rank=12)
    train, validation, test = make_synthetic(config)
    return MORAAdapter(config).fit(train, validation), train, validation, test


def test_synthetic_end_to_end_preserves_base_and_first_prediction(fitted):
    model, train, validation, test = fitted
    before = test.query.backbone_prediction.copy()
    result = model.predict(test.query)
    assert result.point.shape == before.shape
    np.testing.assert_array_equal(before, test.query.backbone_prediction)
    np.testing.assert_array_equal(result.point[:, 0], before[:, 0])
    np.testing.assert_array_equal(result.correction[:, 0], 0)
    np.testing.assert_allclose(result.point, result.backbone_point + result.correction)
    assert point_metrics(test.target, result.point)["mse"] < point_metrics(test.target, before)["mse"] * 0.5
    assert result.metadata["quantiles_calibrated"] is False
    assert result.metadata["implementation"] == "paper_independent_v1"


def test_predict_does_not_take_targets_or_mutate_fitted_state(fitted):
    model, _, _, test = fitted
    state_before = model.diagnostics
    first = model.predict(test.query)
    second = model.predict(test.query)
    np.testing.assert_array_equal(first.point, second.point)
    assert state_before == model.diagnostics
    with pytest.raises(TypeError):
        model.predict(test.query, target=test.target)
    diagnostics = model.diagnostics
    diagnostics["signature"]["backbone_id"] = "changed"
    assert model.diagnostics == state_before
    with pytest.raises(AttributeError):
        model.config = MORAConfig(period=99)
    assert not model._state.first_mapping.flags.writeable


@pytest.mark.parametrize("field,value", [("backbone_id", "other_model"), ("scale_id", "other_units"),
                                          ("channel_names", ("signal_1", "signal_0")),
                                          ("series_ids", ("other_part",) * 24)])
def test_other_model_scale_channel_or_part_cannot_reuse_artifact(fitted, field, value):
    model, _, _, test = fitted
    with pytest.raises(ValueError):
        model.predict(replace(test.query, **{field: value}))


def test_calibration_targets_never_fit_the_coarse_or_residual_models(fitted):
    original, train, validation, _ = fitted
    changed = replace(validation, target=validation.target + 10)
    other = MORAAdapter(original.config).fit(train, changed)
    for name in ("first_mapping", "second_mapping", "first_gain", "second_gain"):
        np.testing.assert_array_equal(getattr(original._state, name), getattr(other._state, name))
    for view in ("absolute", "relative"):
        np.testing.assert_array_equal(getattr(original._state.estimator, view).weight,
                                      getattr(other._state.estimator, view).weight)
    assert not np.array_equal(original._state.calibration, other._state.calibration)


def test_nested_held_targets_cannot_change_their_own_second_stage_oof_prediction():
    config = MORAConfig(period=13, correction_rank=8, folds=3)
    train, _, _ = make_synthetic(config, train_count=48, channels=1)
    operator = coarse_operator(config.horizon, config.coarse_count)
    basis = tapered_basis(config.horizon, config.correction_rank, config.taper_steps)
    _, _, folds = first_crossfit(train, config, operator, basis)
    baseline, _ = second_crossfit(train, config, operator, basis, folds)
    held = folds[0][1]
    target = train.target.copy()
    target[held] += 1000
    changed = replace(train, target=target)
    candidate, _ = second_crossfit(changed, config, operator, basis, folds)
    np.testing.assert_array_equal(baseline[held], candidate[held])


def test_split_overlap_and_failed_refit_do_not_replace_a_good_state(fitted):
    model, train, validation, test = fitted
    before = model.predict(test.query).point
    overlapping = replace(validation, query=replace(validation.query, origins=validation.query.origins - 10000))
    with pytest.raises(ValueError, match="기간"):
        model.fit(train, overlapping)
    with pytest.raises(ValueError, match="이후"):
        model.predict(validation.query)
    np.testing.assert_array_equal(before, model.predict(test.query).point)


def test_zero_observed_demand_is_valid_and_finite():
    config = MORAConfig(period=13, correction_rank=8, folds=3)
    train, validation, test = make_synthetic(config, train_count=36, validation_count=8, test_count=8, channels=1)
    def zero_data(data):
        return replace(data, query=replace(data.query, history=np.zeros_like(data.query.history),
                                          backbone_prediction=np.zeros_like(data.query.backbone_prediction)),
                       target=np.zeros_like(data.target))
    model = MORAAdapter(config).fit(zero_data(train), zero_data(validation))
    result = model.predict(zero_data(test).query)
    np.testing.assert_array_equal(result.point, 0)


def test_pooled_scope_is_an_explicit_extension_not_automatic_part_personalization():
    config = MORAConfig(period=13, correction_rank=8, folds=3, scope="pooled_series")
    train, validation, test = make_synthetic(config, train_count=36, validation_count=8, test_count=8, channels=1)
    train = replace(train, query=replace(train.query, series_ids=tuple(f"train_{i % 3}" for i in range(36))))
    validation = replace(validation, query=replace(validation.query, series_ids=("new_calibration_part",) * 8))
    test = replace(test, query=replace(test.query, series_ids=("new_test_part",) * 8))
    result = MORAAdapter(config).fit(train, validation).predict(test.query)
    assert result.metadata["scope"] == "pooled_series" and np.isfinite(result.point).all()
    with pytest.raises(ValueError, match="하나"):
        MORAAdapter(replace(config, scope="single_series")).fit(train, validation)


def test_checkpoint_roundtrip_no_overwrite_and_tamper_detection(fitted, tmp_path):
    model, _, _, test = fitted
    path = model.save(tmp_path / "mora.npz")
    restored = MORAAdapter.load(path)
    np.testing.assert_array_equal(model.predict(test.query).point, restored.predict(test.query).point)
    with pytest.raises(FileExistsError):
        model.save(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["first_mapping"][0, 0, 0] += 1
    np.savez_compressed(tmp_path / "tampered.npz", **arrays)
    with pytest.raises(ValueError, match="SHA"):
        MORAAdapter.load(tmp_path / "tampered.npz")


def test_checkpoint_validates_shape_even_if_checksum_was_recomputed(fitted, tmp_path):
    path = fitted[0].save(tmp_path / "valid.npz")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    metadata = json.loads(arrays.pop("metadata_json").tobytes().decode())
    metadata.pop("state_sha256")
    arrays["first_gain"] = arrays["first_gain"][:-1]
    metadata["state_sha256"] = state_digest(metadata, arrays)
    arrays["metadata_json"] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
    np.savez_compressed(tmp_path / "wrong_shape.npz", **arrays)
    with pytest.raises(ValueError, match="shape"):
        MORAAdapter.load(tmp_path / "wrong_shape.npz")


def test_synthetic_runner_saves_replayable_evidence(tmp_path):
    output = tmp_path / "experiment"
    summary = run(output)
    assert summary["save_load_predictions_identical"] and not summary["real_data_used"]
    assert (output / "forecast.csv").is_file()
    assert (output / "evaluation_arrays.npz").is_file()
    with pytest.raises(FileExistsError):
        run(output)

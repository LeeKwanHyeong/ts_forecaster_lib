from __future__ import annotations

import importlib
import warnings

import pytest

from modeling_module import train


@pytest.mark.parametrize(
    "models",
    [
        pytest.param(["titan"], id="family"),
        pytest.param(["titan_lmm"], id="canonical"),
        pytest.param(["patchtst_base", "titan_base"], id="mixed"),
    ],
)
def test_public_titan_training_warns_once_before_data_resolution(monkeypatch, models):
    train_module = importlib.import_module("modeling_module.api.train")
    marker = RuntimeError("data-resolution-reached")

    def stop_at_data_resolution(payload):
        raise marker

    monkeypatch.setattr(train_module, "_resolve_loaders", stop_at_data_resolution)

    with pytest.warns(FutureWarning, match="Titan public training is deprecated") as caught:
        with pytest.raises(RuntimeError, match="data-resolution-reached") as exc_info:
            train(
                {
                    "models": models,
                    "device": "cpu",
                    "auto_save_dir": False,
                }
            )

    assert exc_info.value is marker
    assert len(caught) == 1


def test_non_titan_training_does_not_emit_the_deprecation_warning(monkeypatch):
    train_module = importlib.import_module("modeling_module.api.train")

    def stop_at_data_resolution(payload):
        raise RuntimeError("data-resolution-reached")

    monkeypatch.setattr(train_module, "_resolve_loaders", stop_at_data_resolution)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="data-resolution-reached"):
            train(
                {
                    "models": ["patchtst_base"],
                    "device": "cpu",
                    "auto_save_dir": False,
                }
            )

    assert not [item for item in caught if "Titan public training is deprecated" in str(item.message)]

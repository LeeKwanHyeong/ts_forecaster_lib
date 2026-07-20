from __future__ import annotations

import importlib

import pytest

from modeling_module import DistributionLoss, TrainRequest, TrainerConfig, train


@pytest.mark.parametrize(
    ("distribution", "distribution_kwargs"),
    [
        pytest.param("Poisson", {}, id="poisson"),
        pytest.param("Bernoulli", {}, id="bernoulli"),
        pytest.param("NegativeBinomial", {}, id="negative-binomial"),
        pytest.param("Tweedie", {"rho": 1.5}, id="tweedie"),
    ],
)
@pytest.mark.parametrize("loss_field", ["loss", "loss_point"])
def test_unsupported_distribution_fails_before_data_or_training(
    monkeypatch: pytest.MonkeyPatch,
    distribution: str,
    distribution_kwargs: dict[str, float],
    loss_field: str,
):
    train_module = importlib.import_module("modeling_module.api.train")
    reached: list[str] = []

    def unexpected_resolve_loaders(payload):
        reached.append("data")
        raise AssertionError("data materialization must not run")

    def unexpected_run_total_train(*args, **kwargs):
        reached.append("training")
        raise AssertionError("training must not run")

    monkeypatch.setattr(train_module, "_resolve_loaders", unexpected_resolve_loaders)
    monkeypatch.setattr(train_module, "run_total_train", unexpected_run_total_train)

    loss = DistributionLoss(distribution=distribution, **distribution_kwargs)
    request = TrainRequest(
        models=["patchtst_base"],
        trainer=TrainerConfig(**{loss_field: loss}),
    )

    with pytest.raises(ValueError) as exc_info:
        train(request)

    assert str(exc_info.value) == (
        "Unsupported distribution for public training checkpoints: "
        f"{distribution!r} from `{loss_field}`. "
        "Supported distributions: Normal, StudentT."
    )
    assert reached == []


@pytest.mark.parametrize("distribution", ["Normal", "StudentT"])
def test_supported_distribution_reaches_data_resolution(
    monkeypatch: pytest.MonkeyPatch,
    distribution: str,
):
    train_module = importlib.import_module("modeling_module.api.train")
    marker = RuntimeError("data-resolution-reached")

    def stop_at_data_resolution(payload):
        raise marker

    monkeypatch.setattr(train_module, "_resolve_loaders", stop_at_data_resolution)

    request = TrainRequest(
        models=["patchtst_base"],
        trainer=TrainerConfig(loss_point=DistributionLoss(distribution=distribution)),
    )

    with pytest.raises(RuntimeError, match="data-resolution-reached") as exc_info:
        train(request)

    assert exc_info.value is marker

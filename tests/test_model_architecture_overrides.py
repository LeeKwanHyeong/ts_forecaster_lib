from __future__ import annotations

import importlib
from types import SimpleNamespace


def test_run_total_train_forwards_family_architecture_overrides(monkeypatch):
    total_train = importlib.import_module("modeling_module.training.model_trainers.total_train")
    captured: dict[str, dict] = {}

    def fake_runner(**kwargs):
        captured[kwargs["requested_artifact_keys"][0]] = dict(kwargs)

    monkeypatch.setitem(total_train.MODEL_REGISTRY, "patchtst", fake_runner)
    monkeypatch.setitem(total_train.MODEL_REGISTRY, "titan", fake_runner)
    monkeypatch.setattr(
        total_train,
        "resolve_exogenous",
        lambda *args, **kwargs: SimpleNamespace(
            use_exogenous_mode=False,
            source="none",
            exo_dim=0,
            future_exo_cb=None,
            past_cont_dim=0,
            past_cat_dim=0,
        ),
    )

    total_train.run_total_train(
        train_loader=None,
        val_loader=None,
        freq="weekly",
        lookback=52,
        horizon=27,
        device="cpu",
        warmup_epochs=1,
        spike_epochs=None,
        base_lr=1e-3,
        save_dir=None,
        use_exogenous_mode=False,
        models_to_run=["patchtst_base", "titan_base"],
        model_architecture={
            "patchtst": {
                "patch_len": 13,
                "stride": 6,
                "d_model": 384,
            },
            "titan": {
                "d_model": 512,
                "n_layers": 5,
            },
        },
    )

    assert captured["patchtst_base"]["architecture_override"] == {
        "patch_len": 13,
        "stride": 6,
        "d_model": 384,
    }
    assert captured["titan_base"]["architecture_override"] == {
        "d_model": 512,
        "n_layers": 5,
    }

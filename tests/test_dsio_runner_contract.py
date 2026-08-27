from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "src" / "model_test" / "total_train" / "dsio_total_running.py"
RUNNER_MODULE = "_ts_forecaster_dsio_total_runner"


def _load_runner():
    spec = importlib.util.spec_from_file_location(RUNNER_MODULE, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def test_dsio_runner_defaults_match_executable_model_contracts():
    args = runner.build_parser().parse_args([])
    endo_models, exo_models = runner._resolve_model_groups(args)
    future_models, past_only_models = runner._split_exo_training_targets(exo_models)

    assert args.mode == "both"
    assert args.training_mode == "qualification"
    assert args.ssl_mode == "sl_only"
    assert args.ssl_pretrain_stride is None
    assert args.ssl_mask_ratio == 0.3
    assert args.lookback == 52
    assert args.horizon == 27
    assert args.train_end_week == 202544
    assert args.forecast_origin == 202545
    assert args.validation_origin == 202518
    assert args.window_stride == 4
    assert args.endo_loader_backend == "indexed_temporal"
    assert args.endo_batch_size == 1024
    assert args.exo_batch_size == 512
    assert args.warmup_epochs == 30
    assert args.spike_epochs == 0
    assert args.patchtst_d_model == 128
    assert args.patchtst_layers == 2
    assert args.patchtst_d_ff == 512
    architecture = runner.build_model_architecture(args)
    assert architecture.patchtst.d_model == 128
    assert architecture.patchtst.n_layers == 2
    assert architecture.patchtst.d_ff == 512
    assert endo_models == ["patchtst", "patchmixer", "nhits", "timemixer"]
    assert runner.expand_training_targets(endo_models) == [
        "patchtst_base",
        "patchtst_quantile",
        "patchmixer",
        "nhits_base",
        "timemixer",
    ]
    assert exo_models == ["exotst", "timexer"]
    assert future_models == ["exotst_base"]
    assert past_only_models == ["timexer_base"]


def test_linux_wrapper_uses_the_same_non_deprecated_endo_defaults():
    wrapper = (
        ROOT / "src" / "model_test" / "total_train" / "run_dsio_total_running_linux.sh"
    ).read_text(encoding="utf-8")

    assert 'MODE="${MODE:-endo}"' in wrapper
    assert 'TRAINING_MODE="${TRAINING_MODE:-qualification}"' in wrapper
    assert '--training-mode "$TRAINING_MODE"' in wrapper
    assert (
        'ENDO_MODELS="${ENDO_MODELS:-patchtst patchmixer nhits timemixer}"'
        in wrapper
    )
    assert 'ENDO_MODELS="${ENDO_MODELS:-patchtst patchmixer titan}"' not in wrapper
    assert 'TRAIN_END_WEEK="${TRAIN_END_WEEK:-202544}"' in wrapper
    assert 'FORECAST_ORIGIN="${FORECAST_ORIGIN:-202545}"' in wrapper
    assert 'SEED="${SEED:-42}"' in wrapper
    assert 'SSL_PRETRAIN_STRIDE="${SSL_PRETRAIN_STRIDE:-}"' in wrapper
    assert '--ssl-pretrain-stride "$SSL_PRETRAIN_STRIDE"' in wrapper


def test_dsio_runner_can_reproduce_the_previous_patchtst_capacity():
    args = runner.build_parser().parse_args(
        [
            "--patchtst-d-model",
            "384",
            "--patchtst-layers",
            "5",
            "--patchtst-d-ff",
            "1536",
        ]
    )

    architecture = runner.build_model_architecture(args)

    assert architecture.patchtst.d_model == 384
    assert architecture.patchtst.n_layers == 5
    assert architecture.patchtst.d_ff == 1536


@pytest.mark.parametrize(
    ("model_key", "epochs"),
    [
        ("patchtst_base", 8),
        ("patchtst_quantile", 3),
        ("patchmixer", 3),
        ("nhits_base", 31),
        ("timemixer", 33),
    ],
)
def test_dsio_runner_production_refit_contract_is_explicit(model_key, epochs):
    args = runner.build_parser().parse_args(
        [
            "--mode",
            "endo",
            "--training-mode",
            "production_refit",
            "--endo-models",
            model_key,
            "--warmup-epochs",
            str(epochs),
            "--seed",
            "42",
        ]
    )
    endo_models, _ = runner._resolve_model_groups(args)

    assert args.training_mode == "production_refit"
    assert args.warmup_epochs == epochs
    assert args.spike_epochs == 0
    assert args.seed == 42
    assert endo_models == [model_key]
    assert runner.expand_training_targets(endo_models) == [model_key]

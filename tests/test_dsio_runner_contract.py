from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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
    assert args.ssl_mode == "sl_only"
    assert args.endo_batch_size == 1024
    assert args.exo_batch_size == 512
    assert endo_models == ["patchtst", "patchmixer"]
    assert exo_models == ["exotst", "timexer"]
    assert future_models == ["exotst_base"]
    assert past_only_models == ["timexer_base"]


def test_linux_wrapper_uses_the_same_non_deprecated_endo_defaults():
    wrapper = (
        ROOT / "src" / "model_test" / "total_train" / "run_dsio_total_running_linux.sh"
    ).read_text(encoding="utf-8")

    assert 'ENDO_MODELS="${ENDO_MODELS:-patchtst patchmixer}"' in wrapper
    assert 'ENDO_MODELS="${ENDO_MODELS:-patchtst patchmixer titan}"' not in wrapper

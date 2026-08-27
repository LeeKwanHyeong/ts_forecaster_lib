from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/benchmark_patchtst_ssl_5090.py"
SPEC = importlib.util.spec_from_file_location(
    "_patchtst_ssl_5090_benchmark",
    TOOL,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _record(
    *,
    mode: str,
    seed: int,
    mae: float,
    training_seconds: float,
    peak_mib: float,
) -> dict:
    return {
        "mode": mode,
        "seed": seed,
        "pretrain": (
            {"best_epoch": 9}
            if mode == "full"
            else None
        ),
        "supervised_selection": {
            "best_epoch": seed,
            "best_validation_loss": mae,
        },
        "metrics": {
            "mae": mae,
            "wape_pct": mae * 2.0,
            "smape_pct": mae * 3.0,
            "inference_seconds": 0.5,
        },
        "training": {
            "elapsed_seconds": training_seconds,
            "gpu_memory": {"peak_delta_mib": peak_mib},
        },
        "evaluation": {
            "gpu_memory": {"peak_delta_mib": peak_mib / 2.0},
        },
    }


def test_pretrain_analysis_reports_best_and_stable_ranges() -> None:
    analysis = MODULE.analyze_pretrain_history(
        [
            {
                "global_epoch": 1,
                "train_loss": 4.0,
                "validation_loss": 3.0,
            },
            {
                "global_epoch": 2,
                "train_loss": 3.0,
                "validation_loss": 2.01,
            },
            {
                "global_epoch": 3,
                "train_loss": 2.0,
                "validation_loss": 2.0,
            },
            {
                "global_epoch": 4,
                "train_loss": 1.5,
                "validation_loss": 2.015,
            },
            {
                "global_epoch": 5,
                "train_loss": 1.0,
                "validation_loss": 2.2,
            },
        ],
        tolerance_fraction=0.01,
        rolling_window=3,
    )

    assert analysis["best_epoch"] == 3
    assert analysis["near_best_epoch_ranges"] == [[2, 4]]
    assert analysis["best_rolling_window"] == {
        "start_epoch": 2,
        "end_epoch": 4,
        "mean_validation_loss": pytest.approx(
            (2.01 + 2.0 + 2.015) / 3.0
        ),
    }


def test_overlap_exposure_diagnostic_detects_stride_shortcut() -> None:
    overlapping = MODULE.calculate_overlap_exposure(
        patch_len=13,
        stride=6,
        patch_count=7,
        mask_ratio=0.3,
    )
    non_overlapping = MODULE.calculate_overlap_exposure(
        patch_len=13,
        stride=13,
        patch_count=4,
        mask_ratio=0.3,
    )

    assert overlapping[
        "expected_exposed_value_fraction_per_masked_patch"
    ] == pytest.approx(0.6423076923)
    assert overlapping[
        "fully_exposed_masked_patch_fraction"
    ] == pytest.approx(0.35)
    assert non_overlapping[
        "expected_exposed_value_fraction_per_masked_patch"
    ] == 0.0
    assert non_overlapping[
        "fully_exposed_masked_patch_fraction"
    ] == 0.0


def test_comparison_summary_is_paired_by_seed() -> None:
    summary = MODULE.build_comparison_summary(
        [
            _record(
                mode="sl_only",
                seed=11,
                mae=10.0,
                training_seconds=100.0,
                peak_mib=1000.0,
            ),
            _record(
                mode="full",
                seed=11,
                mae=9.0,
                training_seconds=130.0,
                peak_mib=1200.0,
            ),
            _record(
                mode="sl_only",
                seed=22,
                mae=8.0,
                training_seconds=102.0,
                peak_mib=1000.0,
            ),
            _record(
                mode="full",
                seed=22,
                mae=8.5,
                training_seconds=132.0,
                peak_mib=1200.0,
            ),
        ]
    )

    assert summary["paired_seed_count"] == 2
    assert summary["full_accuracy_wins"]["mae"] == 1
    assert summary["modes"]["sl_only"]["metrics"]["mae"]["mean"] == 9.0
    assert summary["modes"]["full"]["pretrain_best_epochs"] == [9, 9]
    assert summary["modes"]["sl_only"]["supervised_best_epochs"] == [
        11,
        22,
    ]
    assert (
        summary["paired_by_seed"][0]["deltas"]["training_seconds"][
            "full_minus_sl_only"
        ]
        == 30.0
    )


def test_runner_command_fixes_data_split_and_capacity(tmp_path: Path) -> None:
    command = MODULE._runner_command(
        python=Path("/venv/bin/python"),
        target_source=tmp_path / "target.parquet",
        case_root=tmp_path / "case",
        mode="full",
        seed=33,
        pretrain_epochs=12,
        pretrain_stride=13,
        mask_ratio=0.4,
        supervised_epochs=40,
        batch_size=1024,
        num_workers=8,
        prefetch_factor=4,
    )
    joined = " ".join(command)

    assert "--train-end-week 202544" in joined
    assert "--validation-origin 202518" in joined
    assert "--forecast-origin 202545" in joined
    assert "--patchtst-d-model 128" in joined
    assert "--patchtst-layers 2" in joined
    assert "--patchtst-d-ff 512" in joined
    assert "--ssl-pretrain-epochs 12" in joined
    assert "--ssl-pretrain-stride 13" in joined
    assert "--ssl-mask-ratio 0.4" in joined
    assert "--patch-len 13" in joined
    assert "--stride 6" in joined
    assert "--warmup-epochs 40" in joined


def test_common_case_kwargs_preserve_virtualenv_python_symlink(
    tmp_path: Path,
) -> None:
    real_python = tmp_path / "python-real"
    real_python.write_text("", encoding="utf-8")
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(real_python)
    target = tmp_path / "target.parquet"
    target.write_text("", encoding="utf-8")

    kwargs = MODULE._common_case_kwargs(
        SimpleNamespace(
            python=venv_python,
            target_source=target,
            pretrain_epochs=12,
            pretrain_stride=13,
            mask_ratio=0.4,
            supervised_epochs=40,
            batch_size=1024,
            num_workers=8,
            prefetch_factor=4,
            poll_seconds=0.25,
            resume=True,
        )
    )

    assert kwargs["python"] == venv_python.absolute()
    assert kwargs["python"] != real_python.resolve()
    assert kwargs["pretrain_stride"] == 13
    assert kwargs["mask_ratio"] == 0.4


def test_resume_rejects_completed_case_with_old_overlap_contract(
    tmp_path: Path,
) -> None:
    phase_root = tmp_path / "comparison"
    case_root = phase_root / "full" / "seed_11"
    case_root.mkdir(parents=True)
    conditions = MODULE._case_conditions(
        mode="full",
        pretrain_epochs=12,
        pretrain_stride=13,
        mask_ratio=0.4,
        supervised_epochs=40,
        batch_size=1024,
        num_workers=8,
    )
    conditions["pretrain_stride"] = 6
    conditions["mask_ratio"] = 0.3
    (case_root / "benchmark_runtime.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "conditions": conditions,
            }
        ),
        encoding="utf-8",
    )
    target = tmp_path / "target.parquet"
    target.write_text("", encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="does not match the requested SSL patching contract",
    ):
        MODULE.run_case(
            phase_root=phase_root,
            python=Path("/venv/bin/python"),
            target_source=target,
            mode="full",
            seed=11,
            pretrain_epochs=12,
            pretrain_stride=13,
            mask_ratio=0.4,
            supervised_epochs=40,
            batch_size=1024,
            num_workers=8,
            prefetch_factor=4,
            poll_seconds=0.25,
            resume=True,
        )

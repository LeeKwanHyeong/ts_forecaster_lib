from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


TOOL = Path(__file__).resolve().parents[1] / "tools" / "compare_patchmixer_accuracy_5090.py"
SPEC = importlib.util.spec_from_file_location("_patchmixer_accuracy_tool", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

_metrics = MODULE._metrics
_paired_comparison = MODULE._paired_comparison
_split_ids = MODULE._split_ids


def test_accuracy_comparison_uses_deterministic_disjoint_series_splits():
    ids = [str(value) for value in range(1, 46)]

    first = _split_ids(ids, seed=11, val_ratio=0.15, test_ratio=0.15)
    second = _split_ids(ids, seed=11, val_ratio=0.15, test_ratio=0.15)

    assert first == second
    assert {name: len(values) for name, values in first.items()} == {
        "train": 31,
        "validation": 7,
        "test": 7,
    }
    split_sets = [set(values) for values in first.values()]
    assert set.union(*split_sets) == set(ids)
    assert all(
        split_sets[left].isdisjoint(split_sets[right])
        for left in range(len(split_sets))
        for right in range(left + 1, len(split_sets))
    )


def test_accuracy_comparison_metrics_and_improvement_direction():
    targets = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    original_predictions = np.asarray([[1.0, 2.0], [3.0, 5.0]])
    enhanced_predictions = np.asarray([[2.0, 3.0], [4.0, 6.0]])
    uids = np.asarray(["a", "b"], dtype=object)

    original_metrics = _metrics(targets, original_predictions)
    comparison = _paired_comparison(
        {
            "targets": targets,
            "predictions": original_predictions,
            "uids": uids,
        },
        {
            "targets": targets,
            "predictions": enhanced_predictions,
            "uids": uids,
        },
    )

    assert original_metrics["mae"] == 0.25
    assert comparison["overall_mae_winner"] == "original"
    assert comparison["original_relative_improvement_pct"]["mae"] > 0.0
    assert comparison["pointwise_absolute_error_win_rate"]["original"] == 1.0

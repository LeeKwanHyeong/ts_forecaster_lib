from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from tools.dsio_v100_h26_contract import V100H26ContractError
from tools.run_sellm_production_refit_5090 import HORIZON
from tools.analyze_sellm_production_negatives_5090 import (
    HISTORY_MEAN_LABELS,
    ZERO_RATIO_LABELS,
    _group_summary,
    _filter_histories,
    _load_included_part_ids,
    _history_mean_bins,
    _horizon_summary,
    _negative_magnitude_summary,
    _qualification_summary,
    _zero_ratio_bins,
)


def test_included_part_filter_preserves_target_order_and_source_identity(tmp_path):
    source = tmp_path / "active-parts.parquet"
    pl.DataFrame({"oper_part_no": ["P3", "P1", "P3"]}).write_parquet(source)
    included, identity = _load_included_part_ids(source)
    histories = np.arange(24).reshape(3, 4, 2)

    filtered, ordered = _filter_histories(
        histories,
        ["P1", "P2", "P3"],
        included,
    )

    assert ordered == ["P1", "P3"]
    assert np.array_equal(filtered, histories[[0, 2]])
    assert identity["part_count"] == 2
    assert len(identity["sha256"]) == 64


def test_included_part_filter_rejects_unknown_ids():
    with pytest.raises(V100H26ContractError, match="absent from target source"):
        _filter_histories(
            np.zeros((1, 2, 1)),
            ["P1"],
            ["P2"],
        )


def test_zero_ratio_and_history_scale_bins_are_stable():
    zero_ratio = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    history_mean = np.asarray([0.0, 0.5, 1.0, 3.0, 10.0, 11.0])

    assert _zero_ratio_bins(zero_ratio).tolist() == list(ZERO_RATIO_LABELS)
    assert _history_mean_bins(history_mean).tolist() == list(HISTORY_MEAN_LABELS)


def test_group_summary_preserves_clip_uplift_identity():
    raw = np.asarray(
        [[-2.0, 1.0], [-1.0, 3.0], [2.0, 4.0]],
        dtype=np.float32,
    )
    labels = np.asarray(["low", "low", "high"])

    rows = _group_summary(labels, ("low", "high"), raw)
    low = rows[0]

    assert low["raw_negative_count"] == 2
    assert low["raw_negative_rate"] == pytest.approx(0.5)
    assert low["clip_added_total"] == pytest.approx(3.0)
    assert low["clipped_mean"] - low["raw_mean"] == pytest.approx(0.75)


def test_horizon_summary_marks_token13_boundary():
    raw = np.zeros((2, HORIZON), dtype=np.float32)
    raw[:, 13] = -1.0

    rows = _horizon_summary(raw)

    assert rows[12]["token_segment"] == 1
    assert rows[13]["token_segment"] == 2
    assert rows[13]["raw_negative_rate"] == 1.0


def test_negative_magnitude_summary_accounts_for_count_and_volume():
    raw = np.asarray([[-0.0005, -0.05, -0.5, -3.0, -7.0, 1.0]])

    rows = _negative_magnitude_summary(raw)

    assert sum(row["count"] for row in rows) == 5
    assert sum(row["count_share"] for row in rows) == pytest.approx(1.0)
    assert sum(row["negative_volume_share"] for row in rows) == pytest.approx(1.0)


def test_qualification_summary_reads_fixed_epoch(tmp_path):
    paths = []
    for seed, negative_rate in ((11, 0.1), (22, 0.2), (33, 0.3)):
        path = tmp_path / f"seed{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "training": {"seed": seed},
                    "epochs": [
                        {
                            "epoch": 6,
                            "mae": 1.0,
                            "wape": 0.2,
                            "smape": 0.3,
                            "bias": 0.1,
                            "raw_negative_rate": negative_rate,
                            "raw_min": -2.0,
                        }
                    ],
                }
            ),
            encoding="ascii",
        )
        paths.append(path)

    summary = _qualification_summary(paths, fixed_epoch=6)

    assert summary is not None
    assert summary["means"]["raw_negative_rate"] == pytest.approx(0.2)
    assert [row["seed"] for row in summary["rows"]] == [11, 22, 33]

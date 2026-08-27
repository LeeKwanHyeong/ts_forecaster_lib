from __future__ import annotations

import polars as pl

from modeling_module.data_loader.temporal import add_period
from tools.run_sellm_qualification_parity_5090 import (
    BASELINE_MAE,
    BASELINE_RAW_NEGATIVE_RATE,
    MAX_MAE_RELATIVE_DRIFT,
    RAW_NEGATIVE_RATE_RANGE,
    _build_datamodule,
)


def test_sellm_seed42_parity_acceptance_boundary_is_frozen():
    assert BASELINE_MAE == 1.3977457284927368
    assert BASELINE_RAW_NEGATIVE_RATE == 0.1408725767902983
    assert MAX_MAE_RELATIVE_DRIFT == 0.03
    assert RAW_NEGATIVE_RATE_RANGE == (0.14, 0.17)


def test_qualification_seed_is_explicit_in_temporal_split():
    weeks = [
        add_period(202509, -offset, "weekly")
        for offset in reversed(range(120))
    ]
    frame = pl.DataFrame(
        {
            "oper_part_no": [part for part in ("A", "B") for _ in weeks],
            "demand_dt": weeks * 2,
            "demand_qty": [float(index % 5) for index in range(len(weeks) * 2)],
        }
    )

    module = _build_datamodule(frame, seed=11)

    assert module.seed == 11

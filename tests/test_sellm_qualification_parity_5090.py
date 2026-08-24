from __future__ import annotations

from tools.run_sellm_qualification_parity_5090 import (
    BASELINE_MAE,
    BASELINE_RAW_NEGATIVE_RATE,
    MAX_MAE_RELATIVE_DRIFT,
    RAW_NEGATIVE_RATE_RANGE,
)


def test_sellm_seed42_parity_acceptance_boundary_is_frozen():
    assert BASELINE_MAE == 1.3977457284927368
    assert BASELINE_RAW_NEGATIVE_RATE == 0.1408725767902983
    assert MAX_MAE_RELATIVE_DRIFT == 0.03
    assert RAW_NEGATIVE_RATE_RANGE == (0.14, 0.17)

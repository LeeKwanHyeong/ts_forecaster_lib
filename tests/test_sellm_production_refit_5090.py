from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from modeling_module.data_loader.temporal import add_period
from tools.dsio_v100_h26_contract import V100H26ContractError
from tools.run_sellm_production_refit_5090 import (
    BATCH_SIZE,
    EPOCHS,
    FORECAST_ORIGIN,
    HORIZON,
    LOOKBACK,
    MODEL_KEY,
    SEED,
    SEMANTIC_VOCAB_SIZE,
    TOKEN_LEN,
    TRAIN_END_WEEK,
    _build_datamodule,
    _expected_metadata,
    _latest_histories,
    _validate_checkpoint_payload,
)


def _frame() -> pl.DataFrame:
    weeks = [
        add_period(TRAIN_END_WEEK, -offset, "weekly")
        for offset in reversed(range(120))
    ]
    return pl.DataFrame(
        {
            "oper_part_no": [part for part in ("A", "B") for _ in weeks],
            "demand_dt": weeks * 2,
            "demand_qty": [float(index % 7) for index in range(len(weeks) * 2)],
        }
    )


def _checkpoint_payload() -> dict:
    return {
        "config": {
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "architecture_variant": "paper_v1",
            "token_len": TOKEN_LEN,
            "semantic_vocab_size": SEMANTIC_VOCAB_SIZE,
            "semantic_top_k": 32,
            "llm_source": "local",
            "freeze_llm": True,
            "use_time_adapter": True,
            "time_adapter_layers": 2,
            "random_seed": SEED,
            "negative_output_penalty_weight": 0.0,
        },
        "meta": {
            **_expected_metadata(),
            "final_train_loss": 1.25,
        },
    }


def test_sellm_production_policy_is_frozen():
    assert MODEL_KEY == "sellm_base"
    assert (LOOKBACK, HORIZON, FORECAST_ORIGIN) == (52, 26, 202510)
    assert (SEED, BATCH_SIZE, EPOCHS) == (42, 256, 6)
    assert (TOKEN_LEN, SEMANTIC_VOCAB_SIZE) == (13, 256)


def test_production_datamodule_uses_all_rows_through_202509():
    datamodule = _build_datamodule(_frame())

    assert datamodule.summary["series_count"] == 2
    assert datamodule.summary["train_target_max_week"] == TRAIN_END_WEEK
    assert datamodule.summary["validation_windows"] == 0
    assert _latest_histories(datamodule).shape == (2, LOOKBACK, 1)


def test_checkpoint_contract_accepts_final_epoch_metadata():
    contract = _validate_checkpoint_payload(_checkpoint_payload())

    assert contract["meta"]["state_selection"] == "final_epoch"
    assert contract["meta"]["final_train_loss"] == pytest.approx(1.25)


def test_checkpoint_contract_rejects_best_validation_state():
    payload = _checkpoint_payload()
    payload["meta"]["state_selection"] = "best_validation"

    with pytest.raises(V100H26ContractError, match="metadata drifted"):
        _validate_checkpoint_payload(payload)


def test_checkpoint_contract_rejects_nonfinite_final_loss():
    payload = _checkpoint_payload()
    payload["meta"]["final_train_loss"] = float("nan")

    with pytest.raises(V100H26ContractError, match="final_train_loss"):
        _validate_checkpoint_payload(payload)


def test_qwen_path_is_an_explicit_runtime_input():
    parser_source = Path(
        "tools/run_sellm_production_refit_5090.py"
    ).read_text(encoding="utf-8")

    assert 'parser.add_argument("--llm-local-path"' in parser_source

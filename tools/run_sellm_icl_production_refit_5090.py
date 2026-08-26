#!/usr/bin/env python3
"""Run and seal the governed SELLM ICL L52/H26 production refit."""

from __future__ import annotations

import json
from typing import Final

from tools import run_autotimes_icl_production_refit_5090 as _shared


LOOKBACK: Final = _shared.LOOKBACK
HORIZON: Final = _shared.HORIZON
TRAIN_END_WEEK: Final = _shared.TRAIN_END_WEEK
STRIDE: Final = _shared.STRIDE
SEED: Final = _shared.SEED
BATCH_SIZE: Final = 4
EPOCHS: Final = 5
LEARNING_RATE: Final = 1e-4
SEMANTIC_VOCAB_SIZE: Final = 256
TOKEN_LEN: Final = 13
MODEL_KEY: Final = "sellm_base"
CHECKPOINT_FILENAME: Final = "weekly_SELLMICLBase_L52_H26.pt"
RECEIPT_CONTRACT: Final = "modeling_module.sellm_icl_production_refit.v1"

SELLM_ICL_POLICY: Final = _shared.ICLProductionRefitPolicy(
    model_key=MODEL_KEY,
    checkpoint_filename=CHECKPOINT_FILENAME,
    receipt_contract=RECEIPT_CONTRACT,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    semantic_vocab_size=SEMANTIC_VOCAB_SIZE,
)


def run_refit(args):
    return _shared._run_refit(args, SELLM_ICL_POLICY)


def main() -> None:
    receipt = run_refit(_shared._parser().parse_args())
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "eligible_series_count": receipt["selection"][
                    "eligible_series_count"
                ],
                "receipt_sha256": receipt.get("receipt_sha256")
                or receipt.get("preflight_sha256"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

from tools.run_sellm_icl_production_refit_5090 import (
    BATCH_SIZE,
    CHECKPOINT_FILENAME,
    EPOCHS,
    HORIZON,
    LEARNING_RATE,
    LOOKBACK,
    MODEL_KEY,
    RECEIPT_CONTRACT,
    SEED,
    SEMANTIC_VOCAB_SIZE,
    STRIDE,
    TOKEN_LEN,
    TRAIN_END_WEEK,
)


def test_sellm_icl_production_policy_is_frozen() -> None:
    assert MODEL_KEY == "sellm_base"
    assert CHECKPOINT_FILENAME == "weekly_SELLMICLBase_L52_H26.pt"
    assert RECEIPT_CONTRACT == "modeling_module.sellm_icl_production_refit.v1"
    assert (LOOKBACK, HORIZON, TRAIN_END_WEEK) == (52, 26, 202509)
    assert (SEED, BATCH_SIZE, EPOCHS) == (42, 4, 5)
    assert (TOKEN_LEN, SEMANTIC_VOCAB_SIZE) == (13, 256)
    assert STRIDE == 26
    assert LEARNING_RATE == 1e-4

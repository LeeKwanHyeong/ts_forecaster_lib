# ExoTST implementation notes

## Public identity

- Canonical artifact: `exotst_base`
- Family request: `exotst`
- Model class: `ExoTST`
- Config class: `ExoTSTConfig`
- Public architecture override: `ExoTSTArchitectureConfig`

## Required data contract

ExoTST requires all three continuous tensors:

- target history `x: [B, lookback, 1]`
- past exogenous `past_exo_cont: [B, lookback, E_p]`
- known future exogenous `future_exo: [B, horizon, E_f]`

Public training therefore requires `use_exogenous_mode=True`, at least one past continuous column, and
at least one future continuous column or callback. Categorical past features are rejected.

## Model path

1. The target history is optionally normalized with RevIN and patch-embedded.
2. Past and future exogenous tensors are cleaned according to `exo_nan_policy` and embedded separately.
3. Dedicated exogenous encoders and cross-temporal fusion build exogenous memory.
4. The endogenous decoder cross-attends to that memory.
5. A point or distribution head produces the configured horizon.

`exo_memory_mode="agg"` uses aggregation tokens. `"all"` uses all encoded exogenous tokens.

## Output modes

- Point: direct horizon forecast
- Normal: `loc` and `scale` parameter contract
- StudentT: `df`, `loc`, and `scale` parameter contract

The public predictor exposes the location forecast as `point`. Quantile output is not implemented.

## Checkpoint behavior

Current training writes `modeling_module.ckpt.v3` with `model_key=exotst_base`, output mode,
distribution identity, parameter order, serialized loss specification, and complete model state.
Supported current artifacts load with `strict=True`.

Some legacy fixture combinations are intentionally rejected when model identity and distribution head
cannot be inferred without ambiguity. This fail-closed behavior prevents a distribution checkpoint from
being partially restored as a point model.

See [ExoTSTBaseline.md](ExoTSTBaseline.md) for the frozen verification evidence and source boundary.

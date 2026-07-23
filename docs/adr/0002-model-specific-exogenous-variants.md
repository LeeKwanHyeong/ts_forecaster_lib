# ADR 0002: Model-specific exogenous variants

## Status

Accepted. The PatchMixer artifact rows and compatibility policy are superseded
by ADR 0004; the model-owned fusion decision remains active.

## Decision

Exogenous neural fusion is owned by each forecasting model. The library does not
provide a universal neural `ExogenousAdapter`, fusion factory, or shared FiLM /
cross-attention switch.

Only the data boundary is shared:

- `ExogenousFeatureSchema` preserves ordered feature identity and validates the
  dataframe contract.
- `ExogenousBatch` normalizes legacy tensor names and validates batch, time,
  dtype, device, and schema dimensions.

## Artifact keys

| Artifact key | Input policy | Model-owned fusion |
| --- | --- | --- |
| `patchtst_base` | Endogenous by default; legacy exogenous config remains loadable | Compatibility routing |
| `patchtst_exogenous` | At least one exogenous input is required | Past patch concatenation and future cross-attention |
| `patchtst_quantile_exogenous` | At least one exogenous input is required | Past patch concatenation and future cross-attention |
| `patchmixer` | Endogenous only | Paper-faithful PatchMixer calculation |
| `patchmixer_exo` | At least one exogenous input is required | Pooled gated latent residual and future target shift |
| `exotst_base` | Past and future continuous inputs are required | Dedicated exogenous encoder |
| `timexer_base` | Past continuous inputs are required | Global-token cross-attention |
| `sellm_base` | Future continuous input is optional | Semantic future conditioning |

The explicit exogenous variants are not added to family expansion. Requesting
`patchtst` or `patchmixer` therefore keeps the existing artifact count and default
behavior.

## Compatibility

The legacy `patchtst_base` and `patchtst_quantile` builders inspect configured
exogenous widths. Retired PatchMixer Enhanced and quantile keys are load-only;
new PatchMixer training uses only `patchmixer` or `patchmixer_exo`.

Explicit keys save their own artifact identity and fusion metadata. A caller must
set `use_exogenous_mode=True` and provide at least one configured past or future
feature.

## Consequences

Model code contains only fusion logic meaningful to that architecture. Adding a
future model such as TimeMixer requires a dedicated endogenous/exogenous decision
and registry entry instead of another branch in a shared neural adapter.

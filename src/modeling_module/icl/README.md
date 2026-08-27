# ICL Dataset Contract

`modeling_module.icl` owns immutable dataset artifacts and the shared training
contract for in-context time-series forecasting. Checkpoint loading remains in the
public API layer.

## Boundaries

- `ICLEpisode` contains one query and same-series prompt demonstrations.
- `ICLSplit` is assigned by time within each series. No random row split is used.
- `ICLManifest` seals the normalized source, builder configuration, and episode hashes.
- `EndogenousICLDatasetBuilder` aggregates duplicate item-week demand rows by sum.
- Missing weekly rows are rejected. A source contract must explicitly materialize
  zero-demand weeks rather than letting the builder infer them.
- Exogenous feature matrices are optional fields on `ICLWindow`. The endogenous
  builder leaves them empty; a future exogenous builder must use the same contract.
- `ICLEpisodeDataModule` only consumes sealed episodes and creates Torch batches.
  It never rebuilds prompts or changes split membership.
- `write_icl_episode_artifact` writes `episodes.parquet` and `manifest.json` as one
  immutable directory. `read_icl_episode_artifact` verifies the file hash, manifest
  hash, row identities, and every episode hash before returning a bundle.

## Prompt Selection

Each endogenous episode uses two non-overlapping demonstrations from the same item:

1. A historical block immediately before the query context.
2. The nearest older block whose target starts on the same seasonal offset and does
   not overlap the historical block.

Both demonstrations end before the query context starts. This keeps prompt labels
and query labels temporally separated.

## Model Adapters

- `AutoTimesICLAdapter` concatenates demonstration context/target pairs and then
  appends the query context, matching AutoTimes numeric prompt semantics.
- `SELLMICLAdapter` preserves demonstration boundaries for SELLM's semantic prompt
  encoder.

## Training and Forecast

- `train_autotimes_icl` trains an ICL-enabled AutoTimes checkpoint from
  `ICLEpisodeDataModule` loaders. The frozen backbone remains frozen.
- `train_sellm_icl` trains SELLM with separate demonstration segments and its
  semantic prompt encoder.
- `forecast_icl` loads a strict checkpoint, verifies the episode artifact, and
  returns one point forecast per episode and horizon week.
- All three paths require a checkpoint configuration with `icl_enabled=True`.
- ICL v1 is endogenous. Episodes containing exogenous matrices are retained by the
  shared artifact contract but are rejected by these two model execution paths
  until their exogenous model contracts are approved.

The regular AutoTimes and SELLM `forward` methods are unchanged. ICL execution uses
the explicit `forward_icl` methods and therefore cannot silently alter a non-ICL
checkpoint's inference behavior.

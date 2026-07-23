# DSIO 202545 Endogenous Qualification Baseline

## Scope

- Qualification training commit: `0d278ec0158c3955223acc1ccae9a26cb9fec110`
- Evaluation commit: `6b0776253e0fb6fb111e66b8df01e00a4cb845f2`
- Evaluation date: `2026-07-23`
- Target source SHA-256:
  `328f547d5eb0a50c80dc60dc7bb89c09799599f8f6b8677406a0e3cc4a3ef547`
- Series: `7,000`
- Lookback / horizon: `52 / 27`
- Qualification origin: `202518`
- Qualification target: `202518..202544`
- Production forecast origin: `202545`
- Observations per model: `189,000` (`7,000 x 27`)
- Inference contract: public `load_predictor(..., strict=True)`
- Point contract: point model output; PatchTST Quantile uses `q50`

The results below compare the saved best qualification checkpoints. They are not
production-refit metrics.

## Point Forecast Metrics

All metrics are micro aggregates over every series-horizon observation. Lower is
better.

| Rank by MAE/WAPE | Model | MAE | WAPE | sMAPE | sMAPE rank |
|---:|---|---:|---:|---:|---:|
| 1 | `nhits_base` | 3.370834 | 19.8562% | 138.3597% | 1 |
| 2 | `patchmixer` | 8.167200 | 48.1095% | 138.9764% | 2 |
| 3 | `patchtst_base` | 10.289115 | 60.6088% | 139.1336% | 4 |
| 4 | `patchtst_quantile` (`q50`) | 10.932775 | 64.4004% | 139.1525% | 5 |
| 5 | `timemixer` | 11.184789 | 65.8849% | 139.0401% | 3 |

MAE and micro WAPE have the same ordering because every model is evaluated
against the same target denominator and observation count. N-HiTS is the clear
qualification leader on all three metrics; PatchMixer is second.

### Sparse-demand interpretation

`68.5561%` of qualification targets are zero. None of the five checkpoints emits
an exact zero through the public point path, so every zero-target/nonzero-forecast
cell contributes almost `200%` to sMAPE. Positive-target-only sMAPE ratios are
between `3.97%` and `6.49%`. The approximately `138%` aggregate sMAPE values are
therefore expected under the declared formula and are dominated by zero-demand
timing, not a percentage conversion error.

## Production Refit Epochs

The fixed epoch is the minimum model-specific validation-loss epoch from the
30-epoch qualification run. Log values were cross-checked against each
checkpoint's `training_manifest.json` `best_val_loss`.

| Model | Best validation loss | Fixed production-refit epochs |
|---|---:|---:|
| `patchtst_base` | 10.2655966622 | 18 |
| `patchtst_quantile` | 4.4498838697 | 5 |
| `patchmixer` | 8.1465595790 | 8 |
| `nhits_base` | 3.3642809732 | 12 |
| `timemixer` | 11.2211085728 | 3 |

PatchTST Quantile uses its quantile validation objective, so its validation-loss
value must not be compared numerically with point-model losses.

Production refit must:

1. initialize each model from scratch with the qualification seed and architecture;
2. include all target observations through `202544`;
3. retain the common cosine scheduler horizon (`t_max=40`);
4. run exactly the fixed epoch count above;
5. save the final epoch state, without reusing the consumed qualification holdout
   for early stopping or best-state restoration.

The existing qualification runner always constructs a validation loader and
`CommonTrainer` restores its best validation state. A dedicated production-refit
data/trainer mode is therefore required before these epoch counts are executed.

## Checkpoint Identity

| Model | Checkpoint SHA-256 |
|---|---|
| `patchtst_base` | `26ebd0f9e15441671bb10376cbd097db2a37a2c3ca06b455cca64975ebaa8c0f` |
| `patchtst_quantile` | `44544022d6ac22174b304f619bf4c964b3d99b283ba0c4444df688d6156b52e2` |
| `patchmixer` | `6ed261e95c290444349e055d576c54ce3ab6724e17243cfea3f95b64c132b9e7` |
| `nhits_base` | `0f4a30c2a72a0c539365c7a25798f561fdae370a49a0adb51ed208b820409146` |
| `timemixer` | `ef05a8659b57f68ae8361e414c0e2ae13ada99d4b7d62c66ceeb0e77f0aa64e0` |

The generated evaluation directory contains the full prediction, per-series,
per-horizon, metric, epoch-policy, and summary artifacts.

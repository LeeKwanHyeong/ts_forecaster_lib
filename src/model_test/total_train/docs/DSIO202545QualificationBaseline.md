# DSIO 202545 Endogenous Qualification Baseline

## Authoritative Protocol

- Training and evaluation code commit:
  `f6d2e84aa316f351c683da878bd81a01b279bda5`
- Evaluation date: `2026-07-23`
- Canonical target SHA-256:
  `328f547d5eb0a50c80dc60dc7bb89c09799599f8f6b8677406a0e3cc4a3ef547`
- Artifact isolation: one canonical artifact per new Python process
- Seed: `42` for every artifact process
- Maximum epochs / cosine `t_max`: `40 / 40`
- Supervised stages: warm-up `40`, spike `0`
- Series: `7,000`
- Lookback / horizon: `52 / 27`
- Qualification origin and target: `202518`, `202518..202544`
- Production forecast origin: `202545`
- Observations per model: `189,000` (`7,000 x 27`)
- Inference contract: public `load_predictor(..., strict=True)`
- Point contract: point model output; PatchTST Quantile uses `q50`

Artifact root:
`artifacts/dsio_202545_qualification40_seed42_isolated_f6d2e84/<model>/endo_only`.

Each artifact is trained in a separate process because a multi-model process
shares global and DataLoader RNG streams. Extending an earlier model from 30 to
40 epochs otherwise changes the random trajectory of every later model despite
the same top-level seed. A combined 40-epoch attempt exposed this coupling and
was stopped and preserved with the suffix `invalid_rng_coupled`.

The results below compare the isolated saved-best qualification checkpoints.
They are not production-refit metrics.

## Point Forecast Metrics

All metrics are micro aggregates over every series-horizon observation. Lower is
better.

| Rank by MAE/WAPE | Model | MAE | WAPE | sMAPE | sMAPE rank |
|---:|---|---:|---:|---:|---:|
| 1 | `nhits_base` | 4.135897 | 24.3628% | 138.1247% | 1 |
| 2 | `timemixer` | 9.292911 | 54.7406% | 139.8867% | 5 |
| 3 | `patchmixer` | 9.410501 | 55.4333% | 139.4685% | 4 |
| 4 | `patchtst_base` | 10.289115 | 60.6088% | 139.1336% | 2 |
| 5 | `patchtst_quantile` (`q50`) | 11.273874 | 66.4096% | 139.4403% | 3 |

MAE and micro WAPE have the same ordering because every model is evaluated
against the same target denominator and observation count. N-HiTS remains the
clear qualification leader on all three metrics. TimeMixer and PatchMixer are
close on MAE/WAPE, while their sparse-demand sMAPE ordering differs.

### Sparse-demand interpretation

`68.5561%` of qualification targets are zero. None of the five checkpoints emits
an exact zero through the public point path, so every zero-target/nonzero-forecast
cell contributes almost `200%` to sMAPE. The approximately `138%` to `140%`
aggregate sMAPE values are therefore expected under the declared formula and are
dominated by zero-demand timing, not a percentage conversion error.

## Epoch 31-40 Extension

The comparison uses each isolated run's own best epoch in `1..30` as its control.
This avoids comparing different RNG trajectories.

| Model | Best 1-30 | Best 31-40 | Late - early loss | Improved | Overall best |
|---|---:|---:|---:|---|---:|
| `patchtst_base` | e18 / 10.265597 | e34 / 10.713947 | +0.448350 | no | 18 |
| `patchtst_quantile` | e3 / 4.582612 | e33 / 5.517626 | +0.935014 | no | 3 |
| `patchmixer` | e3 / 9.352482 | e33 / 12.591407 | +3.238925 | no | 3 |
| `nhits_base` | e24 / 4.233539 | e31 / 4.119316 | -0.114223 | **yes** | **31** |
| `timemixer` | e30 / 9.343078 | e33 / 9.276788 | -0.066290 | **yes** | **33** |

The 40-epoch extension is material for N-HiTS and TimeMixer. PatchTST Base,
PatchTST Quantile, and PatchMixer do not improve after epoch 30.

## Production Refit Epochs

The fixed epoch is the minimum model-specific validation-loss epoch from the
seed-isolated 40-epoch qualification run. Every log contains exactly 40 epochs,
and each minimum was cross-checked against the checkpoint
`training_manifest.json` `best_val_loss`.

| Model | Best validation loss | Fixed production-refit epochs |
|---|---:|---:|
| `patchtst_base` | 10.2655966622 | 18 |
| `patchtst_quantile` | 4.5826121739 | 3 |
| `patchmixer` | 9.3524823870 | 3 |
| `nhits_base` | 4.1193161011 | 31 |
| `timemixer` | 9.2767878941 | 33 |

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
| `patchtst_base` | `f9949e740c491bf87556b647f7b21907f6cf8322e39a571ce43523d45d4f8bc8` |
| `patchtst_quantile` | `4f019b098f046e2e04d9921c1ee7314d165354f80e1ffba0afeb93ad83335079` |
| `patchmixer` | `3e36e60bc7fef846c653cf7e6623b76fb3fc5b4813a5de2153c7134d238f999a` |
| `nhits_base` | `66fa6520104980afd04d1a719f191dbe5fc03949056a7072a15cc999b2a369d5` |
| `timemixer` | `7512f198a6be5f22735d0c214903624aec6b28b5a7fb217e05e6cd7e56fadb23` |

Each model's evaluation directory contains the full prediction, per-series,
per-horizon, metric, epoch-policy, extension-analysis, and summary artifacts.
The earlier 30-epoch combined-process values remain historical evidence but are
superseded for production-refit epoch selection by this isolated baseline.

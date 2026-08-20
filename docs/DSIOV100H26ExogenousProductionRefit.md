# DSIO V100 H26 Exogenous Production Refit

## Scope

- Site: `V100`
- Frequency: weekly
- Input window: L52
- Forecast horizon: H26
- Training data end: `202509`
- Forecast origin: `202510`
- Training mode: `production_refit`
- Validation: disabled
- State selection: final fixed epoch
- Seed: 42

The production policy covers three independent full-H26 checkpoints.
PatchTSTExogenous, ExoTST, and TimeXer each predict W0-W25 with their own model
output; their horizons are not split or combined. The recorded RTX 5090 run in
this document created the first two checkpoints. TimeXer's fixed epoch is now
selected, but its production refit has not yet run. This work does not deploy
artifacts or change the Demand Engine registry.

## Epoch And Seed Policy

The fixed epoch is selected by the lowest mean validation loss at each epoch
across qualification seeds 11, 22, 33, and 42. Per-seed best epochs are kept as
diagnostic evidence but are not averaged directly into the production epoch.

| Model | Seed best epochs (11/22/33/42) | Fixed refit epoch | Mean validation loss at fixed epoch |
|---|---|---:|---:|
| ExoTST | 32 / 36 / 27 / 30 | 40 | 1.471662 |
| PatchTSTExogenous | 25 / 35 / 30 / 39 | 35 | 1.357135 |
| TimeXer | 29 / 25 / 10 / 26 | 20 | 1.42310275 |

At TimeXer epoch 20, seed 11/22/33/42 validation losses are `1.413029`,
`1.403064`, `1.428966`, and `1.447352`. Their sample standard deviation is
`0.01936867`; the worst-seed loss is `1.447352`. The complete 40-epoch curve
is stored in `DSIOV100H26TimeXerEpochPolicy.csv`.

Seed 42 is the project canonical seed. It was fixed before the production run
and was not chosen by validation rank.

## Data Contract

| Item | Value |
|---|---:|
| Source rows | 2,210,508 |
| Source series | 7,000 |
| Eligible production series | 7,000 |
| Excluded series | 0 |
| Training windows | 420,484 |
| Source range | `201801` to `202509` |
| Training target maximum | `202509` |
| Validation windows | 0 |
| Past continuous width | 12 |
| Future continuous width | 12 for ExoTST/PatchTSTExogenous; 0 for TimeXer |

The input Parquet SHA-256 is
`f5abf27149a8408f5011b0735fb622aec430ccebcf89f7f4ce797a668aafb416`.
The input manifest SHA-256 is
`fce2c853acb3fd3f295f6700b2259ca9246ef9d8caf7e5c0e896102b2f7d2a01`.

## RTX 5090 Results

Artifact root:

`/home/leekwanhyeong/artifacts/exogenous-h26-production-refit-cd4ccaf-seed42`

| Model | Epochs | Final train loss | Checkpoint SHA-256 |
|---|---:|---:|---|
| ExoTST | 40 | 1.011124 | `ff5e6e27c86d4d4589099af7dd5b572dcba1b14848b83f09f429d0e45410a172` |
| PatchTSTExogenous | 35 | 0.934760 | `084ea010a26cbae69f8fdb02a1f5af9bfb54fe50105ac6872e24b7fc52c5b5eb` |

Both checkpoints record seed 42, validation disabled, `final_epoch`, and an
exact completed epoch count. Both strictly restore on CUDA and return 52 finite
point values for a two-series `202510` through `202535` canary. Raw outputs can
be negative, so the existing Demand Engine `max(0, raw_point)` postprocessing
contract remains required.

The aggregate receipt SHA-256 seal is
`0b9da9f8f56c33171bf3f8a6b4f138416e4ab5bdb43347dd09594dcd9a404fb3`.
The aggregate receipt and both model receipts were independently recomputed and
matched their embedded seals.

## Provenance

- Production refit implementation: `cd4ccafe2c6df03c2ec558304494c94114cbab40`
- Canary shape and resume validation fix:
  `1492b9422ada170fe6036d2f76ff076ea3c85c94`
- ExoTST training source: `cd4ccaf`
- ExoTST strict validation source: `1492b94`
- PatchTSTExogenous training and validation source: `1492b94`
- Runtime: Python 3.12.13, PyTorch 2.11.0+cu130, NVIDIA GeForce RTX 5090

The first ExoTST post-training canary incorrectly required a `(2, 26)` array,
while the public predictor correctly returned its established flat 52-value
representation. The 40-epoch checkpoint and training manifest were preserved.
After fixing the canary to validate `B * H` finite values, the runner resumed
from strict validation without retraining ExoTST, then trained
PatchTSTExogenous and produced the final PASS aggregate receipt.

## Current Boundary

These files are independent production-refit candidates, not deployed runtime
artifacts. Any Demand Engine integration must register and call each model
separately for its complete H26 output. Checkpoint copying, wheel release, and
runtime deployment require a separate approved integration step.

TimeXer has a fixed production policy of seed 42 and 20 epochs, but no TimeXer
production checkpoint is claimed by the recorded two-model RTX 5090 result.

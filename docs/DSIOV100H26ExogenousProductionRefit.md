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
output; their horizons are not split or combined. The recorded RTX 5090 runs in
this document created and strictly validated all three checkpoints. This work
does not deploy artifacts or change the Demand Engine registry.

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

ExoTST and PatchTSTExogenous artifact root:

`/home/leekwanhyeong/artifacts/exogenous-h26-production-refit-cd4ccaf-seed42`

TimeXer artifact root:

`/home/leekwanhyeong/artifacts/exogenous-h26-timexer-production-refit-dd49e94-seed42-e20`

| Model | Epochs | Final train loss | Checkpoint SHA-256 |
|---|---:|---:|---|
| ExoTST | 40 | 1.011124 | `ff5e6e27c86d4d4589099af7dd5b572dcba1b14848b83f09f429d0e45410a172` |
| PatchTSTExogenous | 35 | 0.934760 | `084ea010a26cbae69f8fdb02a1f5af9bfb54fe50105ac6872e24b7fc52c5b5eb` |
| TimeXer | 20 | 1.528176 | `ed8f84f6dcb031ada1f0033e0c4dc808b53ce3221bb7b0ae3f2ed2d8d8237e35` |

All checkpoints record seed 42, validation disabled, `final_epoch`, and an exact
completed epoch count. They strictly restore on CUDA and return 52 finite point
values for a two-series `202510` through `202535` canary. Raw outputs can be
negative, so the existing Demand Engine `max(0, raw_point)` postprocessing
contract remains required.

The aggregate receipt SHA-256 seal is
`0b9da9f8f56c33171bf3f8a6b4f138416e4ab5bdb43347dd09594dcd9a404fb3`.
The aggregate receipt and both model receipts were independently recomputed and
matched their embedded seals.

The TimeXer checkpoint was produced separately at:

`/home/leekwanhyeong/artifacts/exogenous-h26-timexer-production-refit-dd49e94-seed42-e20/timexer_base/weekly_TimeXerBase_L52_H26.pt`

| TimeXer runtime evidence | Value |
|---|---:|
| End-to-end wall time | 170.017 seconds |
| Training time | 167.689 seconds |
| Training peak allocated VRAM | 144.191 MiB |
| Training peak reserved VRAM | 158.000 MiB |
| Strict load and H26 canary time | 0.752 seconds |
| Canary peak allocated VRAM | 96.114 MiB |
| Canary peak reserved VRAM | 118.000 MiB |
| Checkpoint size | 5,086,213 bytes |

The TimeXer receipt SHA-256 seal is
`b57593aacb1052804ff43e5649a230fa45d0b301302bc17f92a0aebeaac33526`.
An independent RTX 5090 verification recomputed the same seal and checkpoint
SHA-256, strictly restored the model, and produced 26 finite W0-W25 values for
each of the two canary series. The checkpoint records `completed_epochs=20`,
`configured_epochs=20`, `final_train_loss=1.5281756420809227`,
`training_mode=production_refit`, `validation_enabled=false`,
`state_selection=final_epoch`, and `random_seed=42`.

## Provenance

- Production refit implementation: `cd4ccafe2c6df03c2ec558304494c94114cbab40`
- Canary shape and resume validation fix:
  `1492b9422ada170fe6036d2f76ff076ea3c85c94`
- ExoTST training source: `cd4ccaf`
- ExoTST strict validation source: `1492b94`
- PatchTSTExogenous training and validation source: `1492b94`
- TimeXer epoch policy: `7deee31`
- TimeXer production refit support: `1d7801e`
- TimeXer training, runtime evidence, and strict validation source: `dd49e94`
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

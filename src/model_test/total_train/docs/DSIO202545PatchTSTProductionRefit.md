# DSIO 202545 PatchTST Production Refit

## Status

- Date: 2026-07-23
- Branch: `exogenous-models`
- Code commit: `5894dfe`
- Model key: `patchtst_base`
- Decision: Small production checkpoint fixed for the `202545` forecast origin

## Data Contract

| Item | Value |
|---|---:|
| Source rows | 2,455,508 |
| Series | 7,000 |
| Source range | `201801..202544` |
| Source SHA-256 | `328f547d5eb0a50c80dc60dc7bb89c09799599f8f6b8677406a0e3cc4a3ef547` |
| Lookback / horizon | `52 / 27` |
| Window stride | `4` |
| Training windows | 480,072 |
| Last training target | `202544` |
| Validation windows | 0 |

The `production_refit` path did not create or pass a validation loader. Validation,
early stopping, and best-state restoration were disabled. The checkpoint contains the
state after the eighth and final epoch.

## Training Contract

| Item | Value |
|---|---:|
| Capacity | `d_model=128`, `n_layers=2`, `d_ff=512` |
| Parameters | 403,099 |
| Seed | 42 |
| Epochs | 8 |
| Learning rate | `1e-3` |
| Scheduler | Cosine annealing, `t_max=40` |
| Final train loss | 3.2237984421 |
| Device | RTX 5090 / CUDA |
| Torch | `2.11.0+cu130` |
| State selection | `final_epoch` |

The epoch train losses were:

`6.860532, 4.859568, 4.248387, 3.851106, 3.632143, 3.487830, 3.348984, 3.223798`.

## Artifact Identity

Remote artifact root:

`/home/leekwanhyeong/workspace/ts_forecaster_lib_endo_0d278ec/artifacts/dsio_202545_patchtst_small_production_refit_5894dfe`

| Artifact | SHA-256 |
|---|---|
| `endo_only/weekly_PatchTST_L52_H27.pt` | `2674a5b01a882a7d3bf36af598d787136d2c15181879307989a8206a43fa2d78` |
| `verification/production_forecast_202545.parquet` | `84110bfa70b2bf8a6d56d687d49642d75587ba92ea79442cd886c9a3ab16f2a1` |

The checkpoint metadata fixes:

- `training_mode=production_refit`
- `validation_enabled=false`
- `state_selection=final_epoch`
- `configured_epochs=8`
- `completed_epochs=8`
- `random_seed=42`

## Restore And Forecast Verification

The checkpoint was loaded in a fresh process through
`load_predictor(..., strict=True)`. It produced finite point outputs for all 7,000
series and all 27 horizons:

| Item | Value |
|---|---:|
| Forecast origin / end | `202545 / 202619` |
| Forecast rows | 189,000 |
| Strict load | Passed |
| Checkpoint load | 0.181 seconds |
| CUDA inference | 0.197 seconds |
| Throughput | 35,504 series/second |

## Nonnegative Demand Boundary

The paper-lineage endogenous PatchTST point head does not enforce a nonnegative output.
The raw production verification contained 38,966 negative values across 4,079 series
(`20.62%` of all forecast rows; minimum `-165.782`). This does not invalidate the
checkpoint or strict restore, and the model definition is not changed at this baseline.

Demand Engine must apply a demand-domain postprocessing policy before writing or
exposing forecasts. The recommended initial compatibility policy is
`prediction=max(0, raw_point)`; its exact location and golden tests belong to the
Demand Engine integration task.

## Reproduction

```bash
MODE=endo \
TRAINING_MODE=production_refit \
ENDO_MODELS=patchtst_base \
WARMUP_EPOCHS=8 \
SPIKE_EPOCHS=0 \
SEED=42 \
SSL_MODE=sl_only \
ARTIFACT_ROOT="$PWD/artifacts/dsio_202545_patchtst_small_production_refit_5894dfe" \
src/model_test/total_train/run_dsio_total_running_linux.sh --device cuda

python tools/verify_dsio_production_refit.py \
  --checkpoint "$PWD/artifacts/dsio_202545_patchtst_small_production_refit_5894dfe/endo_only/weekly_PatchTST_L52_H27.pt" \
  --target-source "$PWD/raw_data/master/tb_master_target.parquet" \
  --output-dir "$PWD/artifacts/dsio_202545_patchtst_small_production_refit_5894dfe/verification" \
  --device cuda
```

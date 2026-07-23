# DSIO 202545 PatchTST Capacity Sweep

## Status

This seed-42 baseline remains immutable experiment evidence. Its promotion gate
has been completed by the seed `11 / 22 / 33 / 42` decision in
`DSIO202545PatchTSTMultiSeedDecision.md`.

## Decision Boundary

This experiment tests PatchTST encoder capacity only. It does not replace or
modify the authoritative seed-isolated 40-epoch five-model baseline in
`DSIO202545QualificationBaseline.md`.

The existing Current checkpoint remains the control:

- artifact:
  `artifacts/dsio_202545_qualification40_seed42_isolated_f6d2e84/patchtst_base/endo_only`
- checkpoint SHA-256:
  `f9949e740c491bf87556b647f7b21907f6cf8322e39a571ce43523d45d4f8bc8`
- best epoch / validation loss: `18 / 10.2655966622`

Small and Medium were trained in separate Python processes on the RTX 5090.
They use checkout `a8bf56940b665f2b92b07aa5a0e7d8528978b51f`.
There are no training-runtime changes between the Current training commit
`f6d2e84` and `a8bf569`; the intervening changes only affect evaluation,
tests, and documentation.

## Controlled Protocol

The following inputs and training settings are identical for all capacities:

- source SHA-256:
  `328f547d5eb0a50c80dc60dc7bb89c09799599f8f6b8677406a0e3cc4a3ef547`
- rows / series: `2,455,508 / 7,000`
- lookback / horizon: `52 / 27`
- qualification target: `202518..202544`
- train / validation windows: `432,676 / 7,000`
- patch length / stride: `13 / 6`
- attention heads: `16`
- dropout: `0.1`
- normalization: pre-LayerNorm with RevIN
- supervised point loss: default MAE with the same intermittent-demand
  weighting policy
- seed: `42`
- batch size / workers: `1,024 / 8`
- learning rate / weight decay: `1e-3 / 1e-3`
- warm-up / spike epochs: `40 / 0`
- cosine `t_max`: `40`
- AMP device: CUDA

Only `d_model`, `n_layers`, and `d_ff` change.

| Capacity | `d_model` | Layers | `d_ff` | Parameters |
|---|---:|---:|---:|---:|
| Small | 128 | 2 | 512 | 403,099 |
| Medium | 192 | 3 | 768 | 1,344,411 |
| Current | 384 | 5 | 1,536 | 8,891,931 |

## Qualification Results

All checkpoints are the saved-best state and were reloaded through
`load_predictor(..., strict=True)`. Metrics are micro aggregates over the same
189,000 series-horizon observations.

| Rank | Capacity | MAE | WAPE | sMAPE | Best epoch |
|---:|---|---:|---:|---:|---:|
| 1 | **Small** | **9.143390** | **53.8598%** | **138.8097%** | 12 |
| 2 | Medium | 10.222533 | 60.2166% | 139.2335% | 5 |
| 3 | Current | 10.289115 | 60.6088% | 139.1336% | 18 |

Small improves MAE and WAPE by `11.1353%` relative to Current. Medium improves
them by only `0.6471%`. Small also beats Current on paired MAE for
`4,782 / 7,000` series (`68.31%`) and `24 / 27` horizons, so the aggregate gain
is not explained by only a few large-demand observations.

| Small paired against | Observation wins | Series wins | Horizon wins |
|---|---:|---:|---:|
| Medium | 59.39% | 4,816 / 7,000 | 20 / 27 |
| Current | 67.41% | 4,782 / 7,000 | 24 / 27 |

## Capacity And Convergence

| Capacity | Best 1-30 | Best 31-40 | Late improvement | Epoch 40 train / val |
|---|---:|---:|---|---:|
| Small | e12 / 9.149590 | e37 / 10.421429 | no | 2.175864 / 10.516506 |
| Medium | e5 / 10.200519 | e32 / 10.945489 | no | 1.921812 / 11.195557 |
| Current | e18 / 10.265597 | e34 / 10.713947 | no | 2.230283 / 10.917744 |

None of the capacities improves in epochs 31-40. Medium continues reducing
training loss below Small while its validation loss remains worse. Together
with Small's broad paired improvement, this supports the hypothesis that the
Current encoder is oversized for this fixed weekly data and tokenization
contract. It does not establish that a smaller PatchTST is universally better
for larger or denser datasets.

## Runtime And Artifact Identity

Training elapsed is a wall-clock proxy from log creation to checkpoint write,
not a separately instrumented throughput benchmark.

| Capacity | Train elapsed | Inference, 7,000 series | Checkpoint size | Checkpoint SHA-256 |
|---|---:|---:|---:|---|
| Small | 93 s | 0.3121 s | 1.557 MiB | `0ac84618edaa46831a060da428b909b4f2eb09a71ab0f6773b60279a7b012cfa` |
| Medium | 123 s | 0.3262 s | 5.154 MiB | `2790633346a158395ae599ac14c814825e560d81f77bfdb02d96dd67ddac0ed4` |
| Current | 168 s | 0.3336 s | 33.958 MiB | `f9949e740c491bf87556b647f7b21907f6cf8322e39a571ce43523d45d4f8bc8` |

Capacity artifacts:

- Small:
  `artifacts/dsio_202545_patchtst_capacity_seed42_a8bf569/small/endo_only`
- Medium:
  `artifacts/dsio_202545_patchtst_capacity_seed42_a8bf569/medium/endo_only`

Each evaluation directory contains predictions, series and horizon metrics,
checkpoint identity, the complete training history, and epoch-extension
analysis.

## Current Decision

- Keep the existing seed-isolated 40-epoch result as the immutable Current
  control.
- Treat Small (`128 / 2 / 512`) as the promotion candidate for this dataset.
- Do not change the public PatchTST default or the Current production-refit
  epoch from this single-seed sweep.
- Promotion requires a seed-isolated Small-versus-Current confirmation across
  seeds `11 / 22 / 33`, followed by cohort checks for intermittent and dense
  demand series.

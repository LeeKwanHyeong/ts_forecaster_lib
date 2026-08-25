# SELLM Production Negative-Output Analysis

## Decision

Both SELLM production-refit checkpoints are retained as sealed forensic
artifacts but are rejected for Demand Engine registration and runtime
deployment. Their checkpoint and public inference contracts pass, but their
output behavior does not pass the operating-model acceptance boundary.

The runtime `clip_zero` policy remains unchanged. It is a required final
demand-domain guard, but it cannot be treated as a sufficient repair. The
shared-trainer rerun proves that trainer mismatch was only part of the problem;
a model-level negative-output remedy may now be evaluated under controlled
qualification before another production refit.

## Evidence boundary

- Checkpoint:
  `/home/leekwanhyeong/artifacts/sellm/production-refit/cfd5879-seed42-e6/weekly_SELLMBase_L52_H26.pt`
- Checkpoint SHA-256:
  `d77eaed462f0b8cbb0d93c9b493735bd29c4f54f25abc313eb9a9b03df89c1ab`
- Analysis source commit:
  `fb86eef7ed9c09b074bc5e8dbc9d37b3f29fd258`
- Analysis receipt:
  `/home/leekwanhyeong/artifacts/sellm/production-negative-analysis/fb86eef/analysis-receipt.json`
- Analysis receipt seal:
  `6608d10e56e3f68abef383189e6f7fe850ec1a6dd8aa716840f83bac92add910`
- Input: 7,000 series ending at 202509, L52, forecast origin 202510, H26
- Output: 182,000 W0-W25 raw points

The shared-trainer rerun evidence is:

- Checkpoint:
  `/home/leekwanhyeong/artifacts/sellm/production-refit/64a2cfe-shared-seed42-e6/weekly_SELLMBase_L52_H26.pt`
- Checkpoint SHA-256:
  `f63bd600f16ffb1251dd619097504d21875d83bf509d4a5cbcf1ef34d69dc196`
- Source commit:
  `64a2cfe2a5d05931f3b5d04d6307b512ff9d523e`
- Production receipt seal:
  `5f63abb0726655f95d0040f48b3209ece497315340318534bd28820754620e2c`
- Analysis receipt:
  `/home/leekwanhyeong/artifacts/sellm/production-negative-analysis/64a2cfe-shared/analysis-receipt.json`
- Analysis receipt seal:
  `2b77886c459d1a894649dc8df8d7916c41ff0b725e2f8aaabfb7cb6fdf5467fb`

The public predictor and direct model forward output matched exactly for all
182,000 points. The maximum absolute error and exact mismatch count were both
zero. The negative outputs therefore do not come from `DMSForecaster`, output
flattening, reshaping, or the public API.

Future actuals for origin 202510 are not available. Actual production MAE,
WAPE, sMAPE, and bias cannot be calculated. For nonnegative actual demand,
`clip_zero` is pointwise non-worsening for MAE and the WAPE numerator. Its bias
effect is different: clipping increases the forecast mean by exactly the
removed negative volume.

## Aggregate result

| Metric | Result |
| --- | ---: |
| Raw negative points | 111,966 / 182,000 |
| Raw negative rate | 61.52% |
| Series with at least one negative | 5,137 / 7,000 (73.39%) |
| Series with more than half negative | 4,461 / 7,000 (63.73%) |
| Series with all 26 points negative | 2,537 / 7,000 (36.24%) |
| Raw forecast total | 106,374.24 |
| Clipped forecast total | 398,793.63 |
| Quantity added by clipping | 292,419.38 |
| Added quantity as share of clipped total | 73.33% |
| Raw minimum / maximum | -106.49 / 107.74 |

The negative rate is not dominated by harmless floating-point noise. Values
below -5 represent 12.01% of negative points but 62.00% of removed negative
volume. Values below -1 represent 47.95% of negative points and 91.94% of the
removed volume.

## Horizon behavior

| Horizon | Raw negative rate | Raw mean | Clipped mean | Mean clip uplift |
| ---: | ---: | ---: | ---: | ---: |
| W0 | 41.83% | 2.688 | 2.872 | 0.184 |
| W6 | 61.29% | 1.655 | 2.441 | 0.786 |
| W12 | 62.44% | 0.537 | 2.199 | 1.662 |
| W13 | 61.76% | 0.401 | 2.006 | 1.605 |
| W16 | 64.74% | -0.009 | 1.987 | 1.997 |
| W20 | 64.64% | -0.504 | 1.971 | 2.474 |
| W25 | 65.50% | -1.180 | 1.849 | 3.029 |

Token length 13 generates H26 as two recursive segments. W13 starts the second
segment, but there is no discontinuous failure at that boundary: raw mean moves
from 0.537 at W12 to 0.401 at W13. The larger pattern is cumulative downward
rollout drift. Raw mean falls from 2.688 at W0, crosses below zero at W16, and
reaches -1.180 at W25. The second segment continues an error already present in
the first segment.

## Demand-pattern breakdown

Recent-history sparsity is associated with negative outputs, but it is not the
only cause.

| L52 zero ratio | Series | Raw negative rate | Raw mean | Clip share of clipped total |
| --- | ---: | ---: | ---: | ---: |
| 0-25% | 4,391 | 54.86% | 0.269 | 89.31% |
| 25-50% | 556 | 95.88% | -0.486 | 6,061.97% |
| 50-75% | 758 | 67.86% | 0.907 | 42.82% |
| 75-<100% | 1,295 | 65.64% | 1.926 | 19.58% |

The 25-50% group has the highest count rate and almost no positive forecast
volume. The dense 0-25% group is more important for total quantity: it contains
4,391 series and contributes 256,092.92 of the 292,419.38 units removed by
clipping. The problem therefore cannot be explained as an intermittent-only
edge case.

| L52 mean demand | Series | Raw negative rate | Raw mean | Clip share of clipped total |
| --- | ---: | ---: | ---: | ---: |
| (0, 0.5] | 389 | 83.37% | -0.136 | 727.61% |
| (0.5, 1] | 378 | 87.51% | -0.332 | 699.94% |
| (1, 3] | 1,861 | 70.78% | -0.217 | 143.65% |
| (3, 10] | 3,373 | 59.91% | 0.001 | 99.97% |
| >10 | 999 | 31.35% | 4.677 | 42.16% |

Low-scale series are most likely to go negative, which is consistent with an
unconstrained decoder around the zero-demand boundary. The issue still reaches
large series: the mean-demand-above-10 group contains the -106.49 minimum and
requires 88,536.94 units of clipping.

## Qualification comparison

The token13 fixed-epoch-6 qualification average across seeds 11, 22, and 33
had 17.18% raw negatives, a raw minimum of -7.48, clipped MAE 1.2914, clipped
WAPE 0.2744, and clipped bias +0.2001. The seed-42 batch-256 qualification had
14.51% raw negatives at six epochs. These accuracy values use known validation
actuals and `clip_zero`; they are not production-origin accuracy estimates.

The production result is not a controlled extension of that qualification.
The training contracts differ:

| Setting | Qualification benchmark | Production refit |
| --- | --- | --- |
| Optimizer | AdamW | AdamW |
| Weight decay | PyTorch default 0.01 | common trainer 0.001 |
| Learning rate | fixed `1e-4` | cosine schedule from `1e-4`, `t_max=40` |
| Numerical mode | FP32 | CUDA AMP enabled |
| State | fixed epoch or best validation | final epoch |
| Training windows | 375,072 | 420,484 |

Data exposure and trainer behavior changed together, so the 61.52% production
negative rate cannot be attributed only to the larger full-data refit. The
qualification-production trainer mismatch must be removed before evaluating
model-level remedies.

## Shared-trainer rerun

The controlled rerun used the same optimizer and numerical contract as the
successful seed-42 qualification. Artifact integrity, strict load, metadata,
public/direct parity, W0-W25 shape, and finite-output checks all passed.

| Metric | First refit | Shared-trainer refit | Change |
| --- | ---: | ---: | ---: |
| Raw negative points | 111,966 | 102,706 | -9,260 |
| Raw negative rate | 61.52% | 56.43% | -5.09 pp |
| Series with majority negative | 63.73% | 58.07% | -5.66 pp |
| Series with all 26 negative | 36.24% | 31.31% | -4.93 pp |
| Raw forecast total | 106,374.24 | 180,317.03 | +73,942.79 |
| Clipped forecast total | 398,793.63 | 407,654.06 | +8,860.44 |
| Quantity added by clipping | 292,419.38 | 227,337.03 | -65,082.34 |
| Clip-added share of clipped total | 73.33% | 55.77% | -17.56 pp |
| Raw minimum | -106.49 | -90.46 | +16.03 |

The improvement is real but insufficient. Even W0 remains negative for 37.43%
of series, and the worst horizons are W14, W20, W24, W25, and W15 at
60.21-62.04% raw negatives. Low-scale series remain especially unstable: the
raw negative rates are 78.10% for history mean `(0,0.5]`, 84.18% for
`(0.5,1]`, and 67.25% for `(1,3]`. The issue also affects larger series and
cannot be classified as harmless near-zero noise.

True production-origin accuracy remains unknowable because future actuals are
not available. The shared seed-42 qualification MAE of 1.3810 is the controlled
accuracy proxy. Production final train loss 1.3972 and raw/clipped volume ratios
of 15.58%/35.22% versus recent-history-scaled H26 are diagnostics, not accuracy
claims. The 55.77% clip-created volume is sufficient to reject the artifact.

## Operating decision

1. Keep both checkpoints and receipts unchanged for audit and debugging.
2. Do not register either artifact in Demand Engine, publish it as a production
   model, or replace any active runtime artifact.
3. Keep `clip_zero` as the final safety boundary. It remains mathematically
   safe for MAE on nonnegative demand, but it creates 73.33% and 55.77% of the
   two artifacts' final quantities and is therefore masking failed raw states.
4. The shared trainer production refit is complete and also rejected. Do not
   register, publish, or deploy either production checkpoint.
5. Move the next experiment back to qualification. Compare a zero-default
   demand-space negative penalty against an output-head alternative without
   changing Token13, K256, batch 256, seed, split, or trainer contract.
6. Promote a remedy to multi-seed qualification only if it improves clipped
   accuracy while substantially reducing raw negatives and positive bias does
   not regress.
7. Do not select the existing `final_nonneg=True` softplus head as the first
   remedy. It changes every output, maps a raw zero to positive demand, and can
   increase the already positive qualification bias for intermittent series.

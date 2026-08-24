# SELLM Production Negative-Output Analysis

## Decision

The checkpoint produced by the first SELLM production refit is retained as a
sealed forensic artifact but is rejected for Demand Engine registration and
runtime deployment. Its checkpoint and public inference contracts pass, but
its output behavior does not pass the operating-model acceptance boundary.

The runtime `clip_zero` policy remains unchanged. It is a required final
demand-domain guard, but it cannot be treated as a sufficient repair for this
checkpoint. No negative penalty or nonnegative output head should be selected
until qualification and production use the same optimizer and numerical
training contract.

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

## Operating decision

1. Keep the current checkpoint and receipt unchanged for audit and debugging.
2. Do not register it in Demand Engine, publish it as a production model, or
   replace any active runtime artifact.
3. Keep `clip_zero` as the final safety boundary. It remains mathematically
   safe for MAE on nonnegative demand, but in this run it creates 73.33% of the
   final quantity and is therefore masking a failed raw model state.
4. Unify qualification and production around one SELLM training path with
   explicit weight decay, LR scheduler policy, AMP policy, loss, seed, and
   optimizer-update metadata.
5. Re-run seed-42 qualification through that path before another full-data
   production refit. Output, gradient, checkpoint, and optimizer metadata must
   remain reproducible.
6. Consider a demand-space negative penalty only if the parity-controlled
   production refit still has material negatives. The penalty must default to
   zero and preserve the current checkpoint contract at zero.
7. Do not select the existing `final_nonneg=True` softplus head as the first
   remedy. It changes every output, maps a raw zero to positive demand, and can
   increase the already positive qualification bias for intermittent series.

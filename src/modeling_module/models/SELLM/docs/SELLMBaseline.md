# SELLM Baseline

## Scope

`SELLMModel` supports two explicitly versioned architecture contracts.

- `legacy_v1` preserves the first forecaster-lib implementation and its checkpoint schema.
- `paper_v1` implements the endogenous architecture described in *Semantic-Enhanced
  Time-Series Forecasting Via Large Language Models*.

The default remains `legacy_v1`. This is intentional: a checkpoint created before the
architecture field existed must rebuild with the same modules and state-dict keys.

## Legacy characterization

The frozen fallback fixture uses `lookback=8`, `horizon=4`, `token_len=2`, and
`d_model=8` with seed 1234. Its contract fixes:

- output values within floating-point tolerance;
- 1,510 total and trainable parameters;
- state-dict key schema SHA-256
  `5d441c65b97108a0fe8d20100a1879fe09a076dedf9c049985ac2882315c86a2`;
- non-zero finite gradients through numeric encoding, TSCC, AM-VAE, fallback LLM,
  and the direct horizon head;
- strict fallback checkpoint restoration and prediction equality.

Checkpoints without `architecture_variant` restore as `legacy_v1`.

## Paper architecture

`paper_v1` follows the equations in the paper instead of copying the upstream source.

1. Non-overlapping numeric segments pass through a two-layer MLP time encoder.
2. The frozen LLM word-embedding matrix `[V, C]` is projected across the vocabulary
   dimension into `K` semantic prototypes `[K, C]`.
3. TSCC computes `CrossAttn(H, S)`, decomposes the joint space with AM-VAE, applies
   top-k structural priors, and adds the anomaly and de-anomaly gated branches.
4. The LLM key and value projections receive a zero-initialized temporal residual:
   low-rank linear, long-term LSTM, short-term LSTM, and output linear.
5. A two-layer numeric decoder reconstructs segments from LLM tokens.
6. Forecasts longer than one segment are generated autoregressively and trimmed to the
   exact requested horizon.

Unlike `legacy_v1`, `paper_v1` does not add the original temporal token as a third TSCC
residual and does not use a pooled direct-horizon head.

## Exogenous boundary

`paper_v1` is an endogenous baseline. It rejects `future_exo_dim > 0`. The existing
future-continuous additive head remains available only through `legacy_v1` until a
separately named and tested `sellm_exo` model contract is introduced.

## Provenance and limitation

The comparison reference is `LH325/SE-LLM` commit
`9fab871b9c4774cd4b58d025de992d55a24c18e7`. File hashes are pinned in
`provenance.py`, but upstream source is not vendored. No license file was present in the
reviewed upstream snapshot, so the implementation is based on the published equations.

## RTX 5090 Qwen baseline

Commit `a85150df8ffbc6b6da138a09c2670f66439c7a72` was tested with the Python
runtime used by the service on port 8011. The runtime used PyTorch
`2.11.0+cu130`, Transformers `5.5.0`, and the local
`/home/leekwanhyeong/models/Qwen2-0.5B` artifact. Source was loaded through
`PYTHONPATH`; the installed wheel and running service were not changed.

The L52/H26 paper configuration with two adapted Qwen layers passed load,
forward, backward, and finite-gradient checks. At `semantic_vocab_size=1024`,
batch 64 used 10.43 GiB peak training memory and 2.27 GiB peak inference memory.
Repeated batch-64 inference took 62.6 ms, or about 1,023 series per second.

The K1024 v3 checkpoint smoke artifact was 1.578 GiB. The selected K256 default
uses the same strict artifact contract. A K512 intermediate artifact was 1.288
GiB. Strict restoration reproduced the `[2, 26, 1]` output exactly with zero
maximum absolute error.

## Semantic vocabulary capacity

The default semantic vocabulary is 256. Capacity was evaluated with a controlled
5090 pilot over 256, 512, and 1024 prototypes using:

- target SHA-256 `f5abf27149a8408f5011b0735fb622aec430ccebcf89f7f4ce797a668aafb416`;
- train targets ending at 202435 and validation targets from 202436 through 202509;
- 256 seed-42 sampled series, 13,271 training windows, and 256 validation windows;
- two epochs, batch 128, learning rate `1e-4`, and MAE plus TSCC KL loss;
- non-negative clipping only for validation metrics.

The initial seed-42 run favored K512:

| Prototypes | MAE | WAPE | sMAPE | Bias | Peak train VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 2.4759 | 0.4756 | 0.6508 | 0.9211 | 9.36 GiB |
| 512 | **2.3562** | **0.4526** | 0.7199 | **0.1915** | 11.55 GiB |
| 1024 | 2.9856 | 0.5735 | 0.7894 | 1.8519 | 15.93 GiB |

The same sampled series and training contract were then repeated with model and
loader seeds 11, 22, and 33. K256 won MAE and WAPE for every seed:

| Seed | K256 MAE | K512 MAE | K256 WAPE | K512 WAPE |
| ---: | ---: | ---: | ---: | ---: |
| 11 | 6.0677 | 6.1183 | 1.1656 | 1.1753 |
| 22 | 3.7976 | 6.4304 | 0.7295 | 1.2353 |
| 33 | 2.4398 | 5.8260 | 0.4687 | 1.1192 |

Across those three seeds, K256 had mean MAE 4.1017 and K512 had mean MAE
6.1249. K256 also used less peak training memory, 9.69 GiB versus 11.55 GiB,
and produced higher validation throughput. The seed-42 K512 selection was
therefore rejected. K256 is the maintained default, while K512 remains an
explicit experiment profile. K1024 remains rejected because its additional
parameters and memory did not improve the initial pilot accuracy.

The two-epoch K256 result was not a convergence result. Extending the same
seed-isolated runs to ten epochs reduced mean MAE from 4.1017 at epoch 2 to
2.1515 at epoch 9. The cross-seed standard deviation fell from 1.4966 to
0.2635. The best epochs for seeds 11, 22, and 33 were 10, 9, and 9, with MAE
2.3976, 2.0260, and 1.9103. K256 therefore converges in the 9-10 epoch range
under this sampled qualification contract; its earlier large seed spread was
primarily an under-training effect.

## Negative output policy

K512 produced 1,629 negative raw points out of 19,968 validation points across
seeds 11, 22, and 33, a rate of 8.16%. Applying `clip_zero` reduced MAE from
7.0915 to 6.1249 and WAPE from 1.3623 to 1.1766. Every clipped row improved
absolute error because observed demand is non-negative. The tradeoff is that
bias moved from 3.4444 to 4.4109, so clipping increased the existing positive
forecast bias.

Negative rates rose to 26.7% for histories with 50-75% zeros and 38.0% for
histories with 75-100% zeros. W7, W15, and W23 were the three highest-rate
horizons, aligning with the ends of the model's eight-step autoregressive token
segments.

## Token boundary investigation

The ten-epoch K256 runs traced the numeric decoder at every autoregressive call.
At local token position 7, which produces W7, W15, and W23, the mean decoder
input norm increased from 164.65 to 170.47 and 176.91 across the first three
rollouts. The reconstructed value mean drifted from -1.3546 to -1.6933 and
-2.0175, while its negative rate increased from 92.45% to 93.36% and 94.14%.
The absolute gradient mean remained non-zero for every sample but decreased from
0.003522 to 0.002719 and 0.002076.

This rules out a detached-gradient failure. The observed boundary error is a
decoder-position bias that is amplified by recursive rollout. With token length
8, L52 is also replicate-padded to 56 positions, and H26 requires four decoder
calls even though the fourth call contributes only two positions. Both effects
disappear with token length 13: L52 becomes four exact input tokens and H26
becomes two exact forecast tokens.

Three stabilization candidates were compared on the same 256 series, seeds 11,
22, and 33, ten epochs, batch 128, and learning rate `1e-4`:

| Candidate | Mean best MAE | Seed std | Mean raw negative rate | Mean epoch time |
| --- | ---: | ---: | ---: | ---: |
| token length 8 baseline | 2.1113 | 0.2079 | 13.89% | 13.50 s |
| token length 13 | **1.8632** | **0.0999** | 15.02% | **5.99 s** |
| token length 8, overlap 4 | 3.7059 | 0.8936 | 11.23% | 23.34 s |
| token-boundary delta loss, weight 0.1 | 3.6963 | 1.1807 | **8.45%** | 13.87 s |

Token length 13 reduced mean best MAE by 11.75%, halved seed variation, and cut
epoch time by 55.6%. Its actual token-end horizons W12 and W25 also improved
from baseline MAE 2.5124 and 2.0922 to 1.9386 and 1.6937. Simple overlap decoding
was rejected because additional rollout paths increased runtime and seed
instability. The unnormalized boundary loss was rejected because it reduced raw
negative output at the cost of large accuracy and seed-stability regressions.

The L52/H26 `paper_v1` experiment profile must therefore set `token_len=13`
explicitly. The global `SELLMConfig.token_len=8` default is retained because the
global architecture default is `legacy_v1`; changing it would alter the legacy
construction contract. No overlap or boundary-loss option is added to the public
configuration.

## Full-data token qualification

Commit `6e30114b553ce2323e85ac680f120377e1227c2b` repeated the token-length
comparison on all 6,952 eligible series. The fixed contract used seeds 11, 22,
and 33, five epochs, batch 256, learning rate `1e-4`, and best-validation state
selection. All six case receipts and the aggregate receipt passed their
canonical JSON seals. The aggregate seal is
`605c0cf4a1caefa2c4020c2c7e6c0243a4c2c6e78aff66e95c3c7d29fd816e19`.

| Metric | Token length 8 | Token length 13 | Token 13 change |
| --- | ---: | ---: | ---: |
| MAE | 1.4605 +/- 0.2717 | **1.3513 +/- 0.0630** | **-7.48%** |
| WAPE | 0.3103 +/- 0.0577 | **0.2871 +/- 0.0134** | **-7.48%** |
| sMAPE | **0.5975 +/- 0.0481** | 0.6269 +/- 0.0261 | +4.91% |
| Bias | **0.1236 +/- 0.1522** | 0.2942 +/- 0.0239 | +0.1706 |
| Raw negative rate | 21.91% | **15.38%** | **-29.83%** |
| Training seconds per epoch | 315.35 | **128.67** | **-59.20%** |
| Inference series per second | 2,881 | **7,268** | **+152.26%** |
| Peak training allocation | 14.95 GiB | **6.73 GiB** | **-55.00%** |
| Peak inference allocation | 2.11 GiB | **2.11 GiB** | -0.26% |

Token length 13 improved MAE at 23 of 26 horizons and reduced the raw negative
rate at every horizon. It also avoided the seed-33 token-length-8 regression:
the seed-level MAEs were 1.3335, 1.2100, and 1.8381 for token length 8 versus
1.3967, 1.2622, and 1.3950 for token length 13. The mean improvement therefore
comes from materially lower seed variance, not from winning every individual
seed. W22, W23, and W25 remained the three horizons where token length 13 had
higher MAE.

Token length 13 remains the L52/H26 `paper_v1` development default because the
primary MAE and WAPE metrics, seed stability, raw output stability, training
cost, and inference throughput improve together. The higher sMAPE and positive
bias remain explicit guardrails. Seeds 22 and 33 selected epoch 5 for token
length 13, so this five-epoch qualification does not establish the final
production-refit epoch.

The maintained policy is to preserve raw model outputs for diagnostics and
apply `clip_zero` at the public forecast or Demand Engine processing boundary.
A hard clamp is not added inside training because it would hide the rollout
boundary behavior and the bias tradeoff. Although token length 13 reduced the
full-data raw negative rate, 15.38% of raw points remained negative, so it does
not replace this post-processing safety boundary.

These runs establish the development default only. Production-refit epoch
selection and a production artifact remain separate approval steps.

## Token length 13 convergence qualification

Commit `ad8ee520ce847e90194e218df5bf81bb2948df4f` extended the full-data
token-length-13 qualification to ten epochs without changing the 6,952-series
dataset, chronological split, seeds 11/22/33, batch 256, learning rate `1e-4`,
or best-validation state contract. All three seed receipts passed, and the v2
aggregate receipt passed with seal
`f4652cf30feedd7af580f4b4b671dcd1c0c2de2d3e83dd8f5af2f94d3e2d0c48`.

The seed-specific best epochs were 10, 6, and 9. Their best-state MAEs were
1.2486, 1.2357, and 1.2979, producing mean MAE 1.2608 with standard deviation
0.0268. The later epochs therefore improved every seed beyond its five-epoch
state, but their independently selected epochs cannot be used as one fixed
production-refit schedule.

| Fixed epoch | MAE | MAE seed std | WAPE | sMAPE | Bias | Raw negative rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 1.3523 | 0.0637 | 0.2873 | 0.6038 | 0.2528 | 17.28% |
| 6 | **1.2914** | **0.0416** | **0.2744** | **0.5930** | 0.2001 | 17.18% |
| 7 | 1.5742 | 0.1320 | 0.3345 | 0.6546 | **0.0551** | 19.11% |
| 8 | 1.4136 | 0.1524 | 0.3003 | 0.6344 | 0.2224 | **16.08%** |
| 9 | 1.3567 | 0.0731 | 0.2882 | 0.6496 | 0.1430 | 16.20% |
| 10 | 1.4077 | 0.1261 | 0.2991 | 0.6145 | 0.1675 | 17.83% |

Epoch 6 is the recommended fixed production-refit epoch. Relative to epoch 5,
it reduced mean MAE and WAPE by 4.50%, reduced sMAPE by 1.79%, reduced positive
bias by 0.0527, and reduced MAE seed variation by 34.75%. Its raw negative rate
also fell slightly, from 17.28% to 17.18%. Epochs 7 through 10 did not preserve
the same cross-seed accuracy improvement: epoch 9 was close to epoch 5, while
epochs 7, 8, and 10 regressed more clearly.

This recommendation selects only the training duration. It does not approve a
production refit, build a production checkpoint, replace a wheel, or alter the
port-8011 runtime. The maintained `clip_zero` processing boundary remains
necessary because the selected fixed epoch still produces negative raw points.

## Batch-size qualification

Commit `5b06581d723265567d88823fba81afe4969fe684` compared training batches
256, 512, and 1024 in the RTX 5090 `ai_env`. Token length 13, seed 42, learning
rate `1e-4`, the full 6,952-series dataset, and all data boundaries remained
fixed. No run produced an OOM, non-finite output, or invalid receipt.

The one-epoch hardware probes produced:

| Batch | Epoch seconds | Train windows/s | Inference series/s | Peak train VRAM |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 127.72 | 2,937 | 7,275 | 6.73 GiB |
| 512 | 115.01 | 3,261 | 8,287 | 10.85 GiB |
| 1024 | 109.43 | 3,428 | 8,850 | 18.79 GiB |

Batch 512 improved training throughput by 11.0% and batch 1024 by 16.7% over
batch 256. These gains were much smaller than their 61% and 179% peak-memory
increases. The RTX 5090 was not memory-bound at batch 256, so filling VRAM did
not translate into proportional throughput.

At the same six data passes, the accuracy comparison was:

| Batch | Updates | MAE | WAPE | sMAPE | Bias | Raw negative rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 8,796 | 1.4271 | 0.3032 | **0.6109** | 0.3276 | 14.51% |
| 512 | 4,398 | **1.3919** | **0.2957** | 0.6287 | 0.4064 | **14.34%** |
| 1024 | 2,202 | 1.4703 | 0.3124 | 0.6493 | **0.2788** | 15.73% |

Batch 512 improved MAE and WAPE by 2.47% relative to batch 256, but sMAPE and
positive bias regressed. It also had an isolated validation MAE spike to 2.4159
at epoch 3. Batch 1024 was worse on MAE, WAPE, sMAPE, and raw negative rate at
the same data exposure.

A secondary comparison stopped every run at exactly 8,796 optimizer updates:

| Batch | Data passes | Total train time | Final MAE | WAPE | sMAPE | Bias | Raw negative rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 6 | 12.83 min | 1.4271 | 0.3032 | **0.6109** | 0.3276 | 14.51% |
| 512 | 12 | 23.11 min | 1.2912 | 0.2743 | 0.6173 | **0.0728** | 17.82% |
| 1024 | 24 | 43.99 min | **1.2360** | **0.2626** | 0.6224 | 0.2856 | **13.53%** |

The lower MAE at larger batches is not a pure batch-size gain: batch 512 saw
the dataset twice as many times and batch 1024 four times as many times. Total
training time increased by 80% and 243%, respectively. This result is retained
as an update-budget characterization, not as evidence for changing the fixed
six-epoch production policy.

Batch 256 remains the maintained L52/H26 `paper_v1` default because it already
has a multi-seed convergence baseline, uses substantially less VRAM, and has no
material throughput bottleneck. Batch 512 remains the only follow-up candidate
for a future multi-seed throughput profile. Batch 1024 is rejected as a default.
The three qualification aggregate seals are:

- batch 256: `e63e7ec694ebd9ce625b9269c0318d31be98180d370c334971497c01fb63fdf8`;
- batch 512: `6243928ef7972dba110c9c079f5977e550ccbd368e17e646faf48edf22a20996`;
- batch 1024: `2a3e04f099faf006d85daf6ea6c1244c1fe30854ac63b4b137e4971cc88b49a8`.

## Shared-trainer seed-42 parity

Commit `7587d2ef996f7fdaa010725f8b4fef39dd06e418` removed the
qualification-production optimization mismatch. SELLM qualification and all
future production refits now consume one explicit trainer contract:

| Setting | Shared value |
| --- | --- |
| Optimizer | AdamW |
| Learning rate | fixed `1e-4` |
| Weight decay | `0.01` |
| Scheduler | constant |
| Numerical mode | FP32, AMP disabled |
| Point loss | MAE |
| Gradient clipping | global norm `30.0` |

The public `TrainerConfig`, internal `TrainingConfig`, optimizer builder, AMP
policy, checkpoint metadata, production-refit runner, and qualification parity
runner now carry these fields explicitly. Existing non-SELLM runs preserve
their prior default cosine and AMP policies unless callers override them.

The RTX 5090 `ai_env` parity run used the unchanged L52/H26 chronological
qualification data, 375,072 training windows, 6,952 validation series, token
length 13, semantic vocabulary 256, batch 256, seed 42, and six epochs. The
shared trainer selected epoch 4 by validation loss and produced:

| Metric | Historical seed-42 baseline | Shared trainer | Change |
| --- | ---: | ---: | ---: |
| MAE | 1.3977 | **1.3810** | **-1.20%** |
| WAPE | 0.2970 | **0.2934** | **-1.20%** |
| sMAPE | 0.6175 | **0.5903** | **-4.40%** |
| Bias | +0.4548 | **+0.3864** | -0.0684 |
| Raw negative rate | 14.09% | 15.19% | +1.10 pp |

All 180,752 validation points were finite. The raw minimum was `-8.9638`.
MAE remained within the predefined 3% parity boundary and the raw negative
rate remained inside the predefined 14-17% range. Training took 773.88 seconds,
strict-load inference over all validation series took 0.95 seconds, and peak
CUDA allocation was 9,702,106,112 bytes.

The qualification checkpoint SHA-256 is
`e3100816eeded0683b10011b0c71db4221eafe2b9887c6ed30344862f20eb478`.
The sealed receipt is stored at
`/home/leekwanhyeong/artifacts/sellm/shared-trainer-parity/7587d2e-seed42/qualification-parity-receipt.json`.
Its independently verified canonical seal is
`b676952023155fb60e6655fdc6785b2232f8f5d438301b2862022ebea1d704f5`.

This result validates trainer parity only. The checkpoint is a qualification
artifact with `state_selection=best_validation`; it is not a production-refit
checkpoint and is not approved for Demand Engine registration. The prior
production checkpoint remains rejected because it was trained under the old
cosine/AMP/weight-decay contract. A new parity-controlled production refit is
the next approval boundary.

## Production refit artifact

Commit `cfd58795daa77d046b790967ef5b942c17f926bc` ran the approved SELLM
production refit in the RTX 5090 `ai_env`. The run used all 7,000 eligible
series and 420,484 supervised windows through week 202509. It preserved the
selected L52/H26 `paper_v1`, token length 13, semantic vocabulary 256, batch
256, seed 42, learning rate `1e-4`, and six fixed epochs. Validation was
disabled and the final epoch state was saved.

The production checkpoint is stored at
`/home/leekwanhyeong/artifacts/sellm/production-refit/cfd5879-seed42-e6/weekly_SELLMBase_L52_H26.pt`.
Its SHA-256 is
`d77eaed462f0b8cbb0d93c9b493735bd29c4f54f25abc313eb9a9b03df89c1ab`,
its size is 1,227,602,498 bytes, and its final training loss is 1.374697.
The checkpoint metadata records `training_mode=production_refit`,
`validation_enabled=false`, `state_selection=final_epoch`, six completed
epochs, and seed 42. A separate `ai_env` process reproduced the SHA, completed
a strict load, and returned exactly W0-W25 for each canary series.

Training took 630.37 seconds at 4,002 windows/s. Peak CUDA allocation was
4,838.72 MiB and peak reserved memory was 5,370 MiB. Strict load took 1.02
seconds. Full 7,000-series inference took 1.03 seconds at 6,823 series/s with
1,877.71 MiB peak allocation.

The raw production-origin canary returned 182,000 finite points with no
non-finite values, but 111,966 points were negative. The raw negative rate was
therefore 61.52%, with a minimum of -106.49 and a maximum of 107.74. The
checkpoint passes artifact integrity, metadata, strict-load, shape, and runtime
contracts, but the subsequent full-series analysis rejects it for Demand
Engine registration and runtime deployment. The existing `clip_zero` boundary
remains mandatory, but clipping adds 292,419.38 units and creates 73.33% of the
final clipped quantity. This is a failed raw-model state rather than a routine
post-processing correction.

The sealed production receipt is
`/home/leekwanhyeong/artifacts/sellm/production-refit/cfd5879-seed42-e6/production-refit-receipt.json`.
Its canonical receipt seal is
`a8643761d93792c12a81d505e838dee4bad88d2c47baf43cdada12ab918cda32`.

The detailed horizon, series, sparsity, scale, magnitude, and trainer-contract
analysis is recorded in `SELLMProductionNegativeAnalysis.md`. The analysis
receipt seal is
`6608d10e56e3f68abef383189e6f7fe850ec1a6dd8aa716840f83bac92add910`.
Public predictor and direct model outputs matched exactly for all 182,000
points.

## Shared-trainer production refit

Commit `64a2cfe2a5d05931f3b5d04d6307b512ff9d523e` repeated the approved
production refit after qualification and production were unified around the
same AdamW, weight decay `0.01`, fixed learning rate `1e-4`, FP32, MAE, and
gradient-clip-30 contract. The data, L52/H26 architecture, token length 13,
semantic vocabulary 256, batch 256, seed 42, and six-epoch final-state policy
were unchanged.

The checkpoint passed metadata validation, strict load, W0-W25 shape, finite
output, SHA-256, and receipt-seal checks. Its final training loss was 1.397190.
Training took 859.54 seconds at 2,935 windows/s with 6,888.84 MiB peak CUDA
allocation. Strict load took 1.14 seconds. Full 7,000-series inference took
1.00 second at 7,003 series/s.

| Production-origin diagnostic | Old trainer | Shared trainer | Change |
| --- | ---: | ---: | ---: |
| Raw negative rate | 61.52% | **56.43%** | -5.09 pp |
| Series with any negative | 73.39% | **70.81%** | -2.57 pp |
| Series with majority negative | 63.73% | **58.07%** | -5.66 pp |
| Series with all 26 negative | 36.24% | **31.31%** | -4.93 pp |
| Raw minimum | -106.49 | **-90.46** | +16.03 |
| Quantity added by clipping | 292,419.38 | **227,337.03** | -65,082.34 |
| Clip-added share of clipped total | 73.33% | **55.77%** | -17.56 pp |

The shared trainer materially improves raw-output behavior, proving that the
old optimizer and numerical mismatch contributed to the failure. It does not
resolve it. Clipping still creates more than half of the final forecast volume,
102,706 of 182,000 raw points are negative, and 2,192 series are negative at
every horizon. The new artifact is therefore rejected for Demand Engine
registration and runtime deployment.

Production-origin actuals are unavailable, so true production MAE, WAPE,
sMAPE, and bias cannot be calculated. The controlled seed-42 qualification MAE
of 1.3810 is the accuracy proxy with known actuals. The production final train
loss of 1.3972 is an optimization diagnostic only. As a scale sanity check,
the raw and clipped forecast totals are 15.58% and 35.22% of the recent-history
mean scaled to H26; clipping is too dominant for the clipped result to qualify
as trustworthy model output.

The shared-trainer checkpoint is stored at
`/home/leekwanhyeong/artifacts/sellm/production-refit/64a2cfe-shared-seed42-e6/weekly_SELLMBase_L52_H26.pt`.
Its SHA-256 is
`f63bd600f16ffb1251dd619097504d21875d83bf509d4a5cbcf1ef34d69dc196`.
The production receipt seal is
`5f63abb0726655f95d0040f48b3209ece497315340318534bd28820754620e2c`.
The detailed analysis is stored under
`/home/leekwanhyeong/artifacts/sellm/production-negative-analysis/64a2cfe-shared`,
with independently verified analysis seal
`2b77886c459d1a894649dc8df8d7916c41ff0b725e2f8aaabfb7cb6fdf5467fb`.

## Active-operation-part production decision

Commit `b01a0f450b0bd924408ddf449a0cd5e13d412de4` repeated the production-origin
analysis after Demand Engine removed lifecycle-complete operation parts. The
read-only input contains 4,270 active parts at W0 `202510`, giving 111,020
L52/H26 forecast points. The existing shared-trainer checkpoint remained
strict-load compatible, but its raw-output behavior still failed the operating
gate:

| Diagnostic | Active-part result |
| --- | ---: |
| Raw negative points | 39,407 / 111,020 (35.50%) |
| Series with any negative | 2,242 / 4,270 (52.51%) |
| Series with all 26 horizons negative | 422 / 4,270 (9.88%) |
| Raw output total | 328,877.59 |
| Zero-clipped output total | 399,904.56 |
| Quantity introduced by clipping | 71,026.97 (17.76%) |
| Raw range | -84.82 to 106.28 |

The analysis receipt is stored at
`/home/leekwanhyeong/artifacts/sellm/active-lifecycle-reevaluation/b01a0f4-existing-checkpoint/analysis-receipt.json`.
Lifecycle filtering materially improves the result compared with the former
7,000-series canary, but clipping still changes too much of the forecast for
the checkpoint to be registered as an approved operating model.

Three output-penalty candidates were then trained under the same seed-42,
L52/H26, token-13, vocabulary-256, batch-256, six-epoch qualification contract:

| Penalty weight | MAE | WAPE | sMAPE | Bias | Raw negative rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 1.3004 | 0.2763 | 0.5793 | +0.0434 | 19.49% |
| 0.10 | 1.4019 | 0.2978 | 0.6446 | +0.4115 | 12.70% |
| 1.00 | **1.2545** | **0.2665** | **0.5736** | **-0.0346** | 21.33% |

None reduced the raw negative rate by the required 50% while preserving the
accuracy and bias guards. The penalty remains an opt-in research setting with
default `0.0`; it is not part of the maintained SELLM checkpoint.

A Softplus nonnegative output head removed raw negatives completely. It was
therefore checked over seeds 11, 22, and 33 rather than accepted from one seed:

| Seed | MAE | WAPE | sMAPE | Bias | Raw negative rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 1.3572 | 0.2883 | 0.7936 | +0.1971 | 0.00% |
| 22 | **1.2130** | **0.2577** | **0.7767** | **+0.1654** | 0.00% |
| 33 | 1.4369 | 0.3053 | 0.8255 | +0.2420 | 0.00% |
| Mean | 1.3357 | 0.2838 | 0.7986 | +0.2015 | 0.00% |

Compared with the historical unconstrained token-13 epoch-6 multi-seed
baseline, the nonnegative head worsened mean MAE and WAPE by about 3.4% and
mean sMAPE by about 34.7%. It is rejected as the production architecture.

The final decision is **no approved SELLM production checkpoint**. The
existing checkpoint SHA-256
`f63bd600f16ffb1251dd619097504d21875d83bf509d4a5cbcf1ef34d69dc196`
is retained only as a rejected compatibility artifact. A SELLM-inclusive Wheel
may be released to distribute and test the implementation, but that Wheel does
not authorize Demand Engine registration or production inference for SELLM.

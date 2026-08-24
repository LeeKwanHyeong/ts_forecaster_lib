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

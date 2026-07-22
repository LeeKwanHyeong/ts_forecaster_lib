# PatchMixer lineage baseline

This file pins the two implementations used for PatchMixer lineage work. The
machine-readable values live in `provenance.py`.

## Original lineage

- Repository: <https://github.com/Zeying-Gong/PatchMixer>
- Commit: `cfc6c1386e7fe1633f92ef4b258ff1a4649008b4`
- `models/PatchMixer.py` Git blob: `bf3867109192da6cd8816f4aec8ab0bf16ec80af`
- License: MIT

The canonical point model must preserve the upstream tensor layout, separable
convolution, dual forecasting heads, channel-independent output, and RevIN
behavior. Exogenous, distribution, quantile, and nonnegative-output extensions
are outside the parity boundary.

## Enhanced baseline

- Repository commit: `e53269e8e038a2664a43020587f79303aa2b4ff8`
- `PatchMixer.py` Git blob: `97846c17f5101e97308761c9b44e8df03928b374`
- `backbone.py` Git blob: `f225ad28dbadfe5fbc2e18917b58b31b63fe5bc4`
- `common/configs.py` Git blob: `5004e814bb1fc0a751073c4e5e31502cfaaed68f`

The Enhanced implementation remains the compatibility baseline for existing
`patchmixer_base` and `patchmixer_quantile` checkpoints.

## Fixed comparison configuration

| Field | Value |
|---|---:|
| lookback / horizon | 54 / 27 |
| enc_in | 1 |
| patch_len / stride | 12 / 8 |
| mixer_kernel_size | 5 |
| d_model / e_layers | 128 / 6 |
| dropout / head_dropout | 0.1 / 0.02 |
| Enhanced f_out / head_hidden | 256 / 256 |

With external features disabled, this configuration has 76,564 parameters in
the Original model and 7,077,643 parameters in the Enhanced model. Runtime
numbers are intentionally not pinned because they depend on hardware and the
installed PyTorch build.

## RTX 5090 three-seed decision evidence

The controlled accuracy comparison uses the Walmart weekly dataset at SHA-256
`950a9a9ccc9424d09bb652d908a224d8e225b95f6b48d0d05f79e16c2bb4685f`.
Each seed creates an ID-disjoint 31/7/7 train/validation/test series split. Both
models use the same split, initialization seed, data order, FP32 MSE objective,
AdamW optimizer, and model-selection rule within a run.

Positive values below mean that Original has lower error than Enhanced.

| Seed | All rolling MAE | Last-origin MAE | All rolling winner | Last-origin winner |
|---:|---:|---:|---|---|
| 11 | +8.38% | +4.55% | Original | Original |
| 22 | -0.26% | -4.09% | Enhanced | Enhanced |
| 33 | +10.54% | +3.21% | Original | Original |
| seed-wise mean | +6.22% | +1.22% | Original 2/3 | Original 2/3 |

Across the three seeds, Original wins 14/21 series on all rolling windows. Its
mean relative improvements are +7.69% RMSE and +3.71% sMAPE there. On the much
smaller last-origin slice it improves RMSE by +4.20% while sMAPE regresses by
2.93%, so this evidence does not establish universal metric dominance.

The separate RTX 5090 BF16 batch-64 benchmark measures 100 training steps after
20 warm-up steps. Original provides 1.574x throughput, 36.46% lower mean step
latency, 88.29% lower peak allocated VRAM, and 98.92% fewer parameters.

The machine-readable inputs and decision checks are in
`artifacts/benchmarks/patchmixer_5090_multiseed_summary.json`.

### Structure-consolidation revalidation

Commit `acd65c5339acb57664ea7200728aa56769be4b81` moved the canonical Original
classes into `PatchMixer.py`, moved `PatchMixerOriginalConfig` into
`common/configs.py`, and retained `original.py` as a compatibility re-export.
The post-move validation used a clean detached checkout of that exact commit on
the RTX 5090 with Python 3.12.13, PyTorch 2.11.0+cu130, and the same Walmart
dataset SHA-256 shown above.

- Original parity/public API/registry preflight: 21 tests passed.
- Rolling MAE improvement across seeds 11/22/33: +8.378%, -0.257%, +10.539%
  (mean +6.220%; Original wins 2/3 seeds).
- Last-origin MAE improvement: +4.547%, -4.091%, +3.209%
  (mean +1.222%; Original wins 2/3 seeds).
- BF16 mean training step: Original 2.720 ms, Enhanced 4.281 ms.
- BF16 throughput: Original 23,529 samples/s, Enhanced 14,951 samples/s.
- Peak allocated VRAM: Original 22.78 MiB, Enhanced 194.53 MiB.

The generated `patchmixer_5090_multiseed_summary.json` has SHA-256
`fe65486e75c2908ce99e78b27d9e81b338e4ae47bd4027aa1c23bad4d09d3de8`.
All decision guardrails passed, so the consolidation is behavior-preserving for
the tested contract and does not change the capability defaults below.

## RTX 5090 gated-fusion validation

Commit `cf6cb5c99535e7716bc8f8e24edc58546624a5b0` separates the current
PatchMixer exogenous path into four controlled cases: target-only Endogenous,
past `z_gate` only, future output-shift only, and Full (`z_gate` plus future
shift). The run used a clean detached checkout, the same Walmart dataset, FP32
MSE training with seeds 11/22/33, at most 100 epochs with patience 15, and BF16
batch-64 performance measurements after 20 warm-up steps.

Positive accuracy values mean that the candidate has lower MAE than its
baseline.

| Candidate vs baseline | Rolling MAE mean | Last-origin MAE mean | Rolling seed wins | Last-origin seed wins |
|---|---:|---:|---:|---:|
| Past gate vs Endogenous | -0.817% | -3.456% | 1/3 | 0/3 |
| Future shift vs Endogenous | +0.520% | -2.040% | 2/3 | 1/3 |
| Full vs Endogenous | -1.814% | -3.904% | 1/3 | 0/3 |
| Full vs Future shift | -2.387% | -1.830% | 1/3 | 0/3 |

The trained Full model is sensitive to the past path, but that sensitivity is
not consistently beneficial. Replacing standardized past inputs with zero
changes forecast values by 12,484-15,144 units on average depending on the
seed, while mean rolling MAE improves by 0.492% and mean last-origin MAE
regresses by 1.466%. Removing future inputs changes predictions by only 11-21
units and changes MAE by less than 0.01%, so the future shift is effectively
ignored at the current target scale.

The latent gate is also strongly saturated. Across the three seeds its mean
activation is 0.510, but 38.36% of activations are below 0.05 and 40.25% are
above 0.95. This 78.62% saturation supports redesigning the pooling and gate
stabilization rather than promoting the current formulation.

| PatchMixer case | Parameters | BF16 training step | BF16 inference | Training peak VRAM | Inference peak VRAM |
|---|---:|---:|---:|---:|---:|
| Endogenous | 7,077,643 | 4.978 ms | 1.504 ms | 243.22 MiB | 157.75 MiB |
| Past gate | 7,892,107 | 5.455 ms | 1.718 ms | 252.96 MiB | 164.66 MiB |
| Future shift | 7,078,156 | 5.249 ms | 1.602 ms | 241.63 MiB | 158.00 MiB |
| Full | 7,892,620 | 5.555 ms | 1.764 ms | 253.01 MiB | 164.70 MiB |

Full fusion adds 814,977 parameters, 11.58% training-step latency, 17.24%
inference latency, 9.79 MiB training peak allocation, and 6.95 MiB inference
peak allocation over Endogenous. The comparison result JSON has SHA-256
`e3dea5475aa8bb6d94da0dbd3059e75b905ed48dfcf219db3314d4d5f3f9c507`.

This evidence does not promote the current gated fusion as an accuracy default.
`patchmixer_exogenous` remains the explicit PatchMixer capability route when a
caller requires exogenous inputs, but it must not be presented as more accurate
than Endogenous until a redesigned gate passes the same three-seed contract.
Future-feature results also assume every horizon value is available or forecast
at the prediction origin; upstream feature-forecast error is outside this test.

### b15aaa6 compatibility revalidation

The clip-policy checkpoint `b15aaa670fa3f7a9185ed44d68a99658a587bd91`
was revalidated in a clean detached worktree on the RTX 5090. It used the same
6,435-row, 45-series Walmart dataset (SHA-256
`950a9a9ccc9424d09bb652d908a224d8e225b95f6b48d0d05f79e16c2bb4685f`),
FP32 seeds 11/22/33, at most 100 epochs with patience 15, and the BF16
batch-64 20-warm-up/100-measured-step performance protocol.

The complete PatchMixer accuracy aggregate is exactly equal to the prior
`cf6cb5c` result, not merely equal after rounding. Full Exogenous again records
-1.814% mean rolling MAE improvement and -3.904% mean last-origin MAE
improvement relative to Endogenous, so the default-model decision is unchanged.
Zeroing future inputs changes Full-model rolling predictions by only 11.13,
20.90, and 17.99 target units for seeds 11, 22, and 33. Gate saturation remains
between 78.42% and 78.72%.

| PatchMixer case | Parameters | BF16 training step | BF16 inference | Training peak VRAM | Inference peak VRAM |
|---|---:|---:|---:|---:|---:|
| Endogenous | 7,077,643 | 4.964 ms | 1.470 ms | 243.22 MiB | 157.75 MiB |
| Past gate | 7,892,107 | 5.267 ms | 1.626 ms | 252.96 MiB | 164.66 MiB |
| Future output shift | 7,078,156 | 5.219 ms | 1.583 ms | 241.63 MiB | 158.00 MiB |
| Full Exogenous | 7,892,620 | 5.542 ms | 1.755 ms | 253.01 MiB | 164.70 MiB |

Full Exogenous adds 814,977 parameters, 11.66% training-step latency, 19.43%
inference latency, 9.79 MiB training peak allocation, and 6.95 MiB inference
peak allocation over Endogenous in this rerun. The result file
`patchmixer_exogenous_b15aaa6_5090.json` has SHA-256
`3c05ec17b85e2efc583a2241a1f4ea03049483ba677b69bac496be03bb68996b`.

## Exogenous implementation contract

The current Point and Quantile implementations inherit `_ExoMixin`. Their
active construction and forward path is:

1. `_init_exo` resolves the output-only `future_exo_shift_space` contract and
   registers the future MLP, categorical embeddings, latent projection, and
   latent gate when the configured widths require them.
2. `_validate_future_exo_contract` validates the future `[B,H,E]` tensor at the
   model boundary.
3. `_inject_past_exo_z_gate` calls `_pool_past_exo` and injects the pooled past
   features after the backbone.
4. Point and Quantile forward methods call `apply_exo_shift_linear_trainable`
   directly to add the future shift in output space. The packed Distribution
   path calls `apply_exo_shift_linear` directly for its location parameter.

`_apply_future_exo_shift` had no call site and was removed after this baseline
was established. It is not part of the behavioral contract.

The machine-enforced fixture uses lookback/horizon 8/2, `d_model=8`, one
Quantile branch with `fused_dim=8`, two past continuous features, two past
categorical features with embedding dimensions 3/4, and two future continuous
features.

| Variant | Total parameters | Exogenous parameters | State-dict keys |
|---|---:|---:|---:|
| Point Endogenous | 12,323 | 0 | 45 |
| Point Exogenous | 13,999 | 1,676 | 55 |
| Quantile Endogenous | 2,372 | 0 | 53 |
| Quantile Exogenous | 2,824 | 452 | 63 |
| Distribution Endogenous | 12,332 | 0 | 45 |
| Distribution Exogenous | 14,008 | 1,676 | 55 |

Both Exogenous variants register the same ten parameter names: four future-head
tensors, two categorical embedding tables, two latent-projection tensors, and
two latent-gate tensors. Their exact shapes, full state-schema hashes, output
values, and gradient reachability are pinned in
`tests/test_patchmixer_exogenous_mixin_contract.py`.

## Default model strategy

- Endogenous point forecasting: `patchmixer_original`
- Point forecasting with exogenous inputs: `patchmixer_exogenous` (explicit
  capability route; current gated fusion is not accuracy-promoted)
- Distribution forecasting: `patchmixer_base`
- Quantile forecasting: `patchmixer_quantile`
- Quantile forecasting with exogenous inputs: `patchmixer_quantile_exogenous`

Callers can resolve this policy with `get_patchmixer_default_model_key`. The
public `patchmixer` family still expands to `patchmixer_base` followed by
`patchmixer_quantile`; existing aliases and Enhanced checkpoints are unchanged.
This avoids routing unsupported exogenous or probabilistic requests into the
Original point-only implementation.

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
20 warm-up steps. Original provides 1.583x throughput, 36.84% lower mean step
latency, 88.30% lower peak allocated VRAM, and 98.92% fewer parameters.

The machine-readable inputs and decision checks are in
`artifacts/benchmarks/patchmixer_5090_multiseed_summary.json`.

## Default model strategy

- Endogenous point forecasting: `patchmixer_original`
- Point forecasting with exogenous inputs: `patchmixer_base`
- Distribution forecasting: `patchmixer_base`
- Quantile forecasting: `patchmixer_quantile`

Callers can resolve this policy with `get_patchmixer_default_model_key`. The
public `patchmixer` family still expands to `patchmixer_base` followed by
`patchmixer_quantile`; existing aliases and Enhanced checkpoints are unchanged.
This avoids routing unsupported exogenous or probabilistic requests into the
Original point-only implementation.

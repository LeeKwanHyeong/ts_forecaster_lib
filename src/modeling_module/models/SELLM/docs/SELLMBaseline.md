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

The official v3 checkpoint smoke artifact was 1.578 GiB. Strict restoration
reproduced the `[2, 26, 1]` output exactly with zero maximum absolute error.

## Semantic vocabulary capacity

The default semantic vocabulary is 512. It was selected by a controlled 5090
pilot over 256, 512, and 1024 prototypes using:

- target SHA-256 `f5abf27149a8408f5011b0735fb622aec430ccebcf89f7f4ce797a668aafb416`;
- train targets ending at 202435 and validation targets from 202436 through 202509;
- 256 seed-42 sampled series, 13,271 training windows, and 256 validation windows;
- two epochs, batch 128, learning rate `1e-4`, and MAE plus TSCC KL loss;
- non-negative clipping only for validation metrics.

| Prototypes | MAE | WAPE | sMAPE | Bias | Peak train VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 2.4759 | 0.4756 | 0.6508 | 0.9211 | 9.36 GiB |
| 512 | **2.3562** | **0.4526** | 0.7199 | **0.1915** | 11.55 GiB |
| 1024 | 2.9856 | 0.5735 | 0.7894 | 1.8519 | 15.93 GiB |

The 512 profile is the accuracy-oriented default because it produced the best
MAE, WAPE, and bias. The 256 profile remains the preferred lightweight option
when lower memory and better sMAPE matter more. The 1024 profile is not the
default because its additional parameters and memory did not improve pilot
accuracy. This pilot fixes the initial capacity policy; multi-seed full-data
qualification is still required before production promotion.

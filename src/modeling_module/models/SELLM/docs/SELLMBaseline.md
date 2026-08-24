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

The fallback backend verifies architecture and public API behavior only. Real Qwen loading,
GPU memory, throughput, accuracy, and full checkpoint artifact policy remain RTX 5090
qualification work.

# N-HiTS implementation notes

## Public identity

- Canonical artifact: `nhits_base`
- Family request: `nhits`
- Model class: `NHITSModel`
- Config class: `NHITSConfig`
- Public architecture override: `NHITSArchitectureConfig`

`models=["nhits"]` and `models=["nhits_base"]` both resolve to the same point artifact.

## Input and output

`NHITSModel` accepts a float tensor `x` with shape `[batch, lookback, 1]` and returns
`[batch, horizon, 1]`. `load_predictor(...).predict(x)` normalizes this to a dictionary containing
`point`.

The model is intentionally endogenous-only. Non-empty `future_exo`, `past_exo_cont`, and
`past_exo_cat` inputs are contract errors. Public training also rejects `use_exogenous_mode=True`.

## Architecture configuration

`NHITSArchitectureConfig` exposes stack structure without exposing optimizer or data settings:

- `stack_types`
- `n_blocks`
- `n_layers`
- `n_theta_hidden`
- `n_pool_kernel_size`
- `n_freq_downsample`
- `pooling_mode`
- `interpolation_mode`
- `activation`
- `initialization`
- `batch_normalization`
- `dropout_prob_theta`
- `shared_weights`

Every stack-indexed sequence must have the same length as `stack_types`. Each hidden-width sequence
must match its stack's `n_layers`. The current backbone supports identity stacks only.

## Training and artifacts

N-HiTS uses the shared staged `CommonTrainer`, point loss, optional intermittent weighting, and the
standard warm-up/spike stage controls. Distribution and quantile losses are not supported.

Saved checkpoints include:

- canonical `model_key=nhits_base` and `family_key=nhits`
- primitive `NHITSConfig` state
- `output_spec.mode=point`
- complete model state dict

Use `load_predictor(path, strict=True)` for supported checkpoints. The restored predictor uses the
checkpoint horizon by default.

## Legacy wrapper

`NHITS` remains available inside the implementation package for older dict-batch callers. It is not the
public registry artifact and its optimizer/training helper API is not part of the stable top-level API.

See [NHITSBaseline.md](NHITSBaseline.md) for source identity limits and regression evidence.

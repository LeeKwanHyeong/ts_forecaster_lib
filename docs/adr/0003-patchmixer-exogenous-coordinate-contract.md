# ADR 0003: PatchMixer exogenous coordinate contract

## Status

Accepted. The output-space compatibility contract and normalized-space forward
paths are implemented for Point, Quantile, and Distribution location outputs.

## Context

PatchMixer receives exogenous tensors that may already have been standardized by
the data pipeline. Historical fields named `exo_is_normalized_default` and
`exo_is_normalized` do not perform that standardization and do not change the
current model output. Reusing either flag to choose where the future-exogenous
head is applied would mix two different concerns:

- the coordinate system of the exogenous input features; and
- the target coordinate system of the learned forecast shift.

The future head emits one shift per horizon. Historically, Point, Quantile, and
Distribution location paths added it after RevIN denormalization, so the learned
value was interpreted directly in raw target units. The RTX 5090 gated-fusion
baseline showed that this path changed predictions by only 11-21 target units
and was effectively ignored at the tested demand scale.

## Decision

### Legacy normalization fields

For PatchMixer, `exo_is_normalized_default`, the inherited training config field
`exo_is_normalized`, and the forward argument `exo_is_normalized` are deprecated
accepted-and-ignored compatibility fields.

- Existing checkpoints continue to load and re-save these values.
- Existing callers may continue to pass the forward argument.
- The values must not select a model branch, transform a tensor, or change an
  output.
- No per-call warning is emitted because it would flood training logs.
- Removal requires a future major public-API and checkpoint-format transition.

Exogenous feature scaling belongs to the data layer. Its fitted statistics,
feature order, and schema identity must travel with the dataset or checkpoint
metadata rather than through a boolean model flag.

### Future shift space

The rollout was staged so the config could not silently select an unimplemented
forward path. Stage one fixed the legacy output behavior and missing-field
default; stage two completed the normalized forward paths. The final config
schema is `future_exo_shift_space: Literal["output", "normalized"] = "output"`.
Missing fields in legacy checkpoints still resolve to `output`. Point,
Quantile, and Distribution models accept both declared values, while unknown
values always fail.

| Value | Meaning | Compatibility |
| --- | --- | --- |
| `output` | The future head emits raw target-unit corrections added after target denormalization. | Default for missing fields and all legacy checkpoints. |
| `normalized` | The future head emits target RevIN-space corrections added before target denormalization. | Opt-in candidate requiring its own accuracy baseline. |

This field describes the output coordinate of the future-exogenous head. It
does not describe whether the exogenous inputs themselves were standardized.
A schema-declared value becomes executable only when model capability
validation confirms its complete forward implementation. A declared but
unimplemented value must never fall back silently to another mode.

### Target coordinate definition

PatchMixer Enhanced and Quantile currently construct target RevIN with
`affine=False`, `subtract_last=True`, and `use_std=True`. For forecast target
channel 0, which the current heads use for anchoring and denormalization, let:

```text
c_b       = x[b, -1, 0]
s_b       = sqrt(var_t(x[b, :, 0]) + eps)
N_b(y)    = (y - c_b) / s_b
D_b(z)    = z * s_b + c_b
r[b, h]   = the effective future-head residual for sample b and horizon h
```

The RevIN statistics come only from the observed target history, are computed
per sample, and are detached by the current RevIN implementation.

### Optional normalized residual bound

`future_exo_normalized_residual_limit: Optional[float] = None` is an opt-in
safety boundary for the normalized coordinate. A positive value `c` changes
only the normalized residual before it is added to the forecast:

```text
r_bounded = c * tanh(r / c)
```

The field requires `future_exo_shift_space="normalized"` and `use_revin=True`;
invalid, non-finite, non-positive, output-space, and RevIN-disabled
combinations fail during model construction. `None` preserves the previous
equation exactly. The bound creates no parameter or persistent buffer, so
state-dict schemas and legacy strict loading are unchanged.

- In `output` mode, `r` is in raw target units: `D_b(z) + r`.
- In `normalized` mode, `r` is in target RevIN units: `D_b(z + r)`.

Before clipping or a final non-negative transform, the raw effect of a
normalized shift is therefore:

```text
D_b(z + r) - D_b(z) = r * s_b
```

Coordinate selection happens after the future head and any established
branch-level multiplier. It must not change the future-head architecture,
parameter names, or state-dict schema.

The normalized residual is added to the normalized forecast; it is not passed
through full RevIN denormalization on its own. Denormalizing `r` alone would add
`c_b` and incorrectly turn a delta into an absolute target value. A post-denorm
implementation using a scale-only inverse could be algebraically equivalent
when no clipping is present, but pre-denorm insertion is the canonical path so
the corrected forecast is subject to the existing normalized-space clip.

## Application points

The exact operation order is part of the checkpoint behavior contract.

| Output mode | `normalized` pipeline | `output` pipeline |
| --- | --- | --- |
| Point | head -> last-value anchor -> shared horizon shift -> eval clip -> RevIN denorm -> final non-negative transform | head -> last-value anchor -> eval clip -> RevIN denorm -> shared horizon shift -> final non-negative transform |
| Quantile | head -> last-value anchor -> shared horizon shift broadcast to every quantile -> eval clip -> per-quantile RevIN denorm | head -> last-value anchor -> eval clip -> per-quantile RevIN denorm -> shared horizon shift broadcast to every quantile |
| Distribution | location head -> last-value anchor -> output-scale/bias and depthwise location refinement -> location shift -> RevIN denorm -> final non-negative transform | location head -> last-value anchor -> output-scale/bias and depthwise location refinement -> RevIN denorm -> location shift -> final non-negative transform |

The following boundaries are explicit:

- Point and Quantile shifts are inserted after anchoring, so they remain a
  residual correction rather than part of level estimation.
- The normalized shift is inserted before eval clipping. The clip therefore
  constrains the complete corrected normalized forecast, not only the
  endogenous forecast.
- Distribution shifts are inserted after all normalized-space location
  refinements so the future residual is not re-scaled or temporally mixed by
  the location head.
- Distribution `scale`, `df`, and any other non-location parameters are never
  shifted.
- Quantile uses one `[B,H]` shift broadcast across all quantiles. Addition,
  clipping, and RevIN denormalization are monotone and therefore preserve
  quantile ordering when it was present before the shift.
- The final non-negative transform remains after either shift mode. Its
  nonlinearity may make the observed raw-output difference smaller than the
  pre-transform residual.

When `use_revin=False`, target normalization is the identity. Both shift-space
values must use the current output insertion path and produce exactly the same
result for identical inputs and weights. The selected config value remains
serialized for reproducibility; it must not trigger synthetic statistics or an
extra scaling operation.

Very low-variance histories have `s_b` near `sqrt(eps)`, so normalized shifts
can have a much smaller raw effect than output shifts. Conversely, gradients to
the future head are scaled by `s_b` for high-variance histories. No separate
scale floor is introduced because that would define a third coordinate system;
these effects must be measured in the accuracy and stability baseline.

## Alternatives considered

1. Reinterpret `exo_is_normalized` as the shift-space selector. Rejected because
   existing values are no-ops and their name refers to the input, not the target.
2. Keep output-space shift as the only mode. Compatible, but poorly scaled for
   heterogeneous series and already shown to be nearly inactive.
3. Make normalized-space shift the new default. Rejected because it changes old
   checkpoint behavior and requires multi-seed evidence before promotion.
4. Denormalize the shift by itself and add it after the base forecast. Rejected
   because full RevIN denormalization adds the target center to a delta.
5. Convert the shift with a scale-only inverse and add it after denormalization.
   Rejected as the canonical path because it bypasses normalized-space clipping;
   it remains a useful algebraic oracle for tests where clipping is disabled.

## Implementation invariants

The normalized implementation is complete only when all of these hold:

1. `output` mode remains bit-for-bit identical to the frozen baseline.
2. With RevIN enabled and clipping/non-negative transforms disabled, a
   normalized shift has raw effect `r * s_b` for Point, every Quantile, and
   Distribution `loc`.
3. Quantile applies one shared shift and preserves pairwise quantile gaps before
   any clip saturation.
4. Distribution changes `loc` only; `scale`, `df`, and all other packed
   parameters remain identical.
5. With RevIN disabled, `output` and `normalized` are exactly equal.
6. Gradients reach the future-head parameters and future-exogenous input in both
   spaces.
7. State-dict keys and parameter counts are unchanged, and checkpoints without
   the config field still restore as `output`.

## Validation sequence

1. Preserve the legacy no-op and missing-config checkpoint contracts in tests.
2. Stabilize the two-value config schema while keeping `output` as the runtime
   behavior of checkpoints without the field.
3. Implement `normalized` as an explicit opt-in at the application points above
   without changing parameter names or state-dict shapes. Point, Quantile, and
   Distribution `loc` are complete.
4. Compare Endogenous, output-shift, and normalized-shift variants with the same
   data, split, loss, seeds, and RTX 5090 performance protocol.
5. Promote a new default only when both rolling and last-origin evidence justify
   the compatibility cost.

## Validation result

The clean RTX 5090 run at commit
`4d1ce99dfa1a53ee9a232b87c56f5b72a6d3c5d4` completed step 4 with seeds
11/22/33. The output path exactly reproduced the previous baseline's training
histories and prediction hashes. Relative to output, normalized had -0.805%
mean rolling MAE improvement and -3.629% mean last-origin MAE improvement. It
won 2/3 rolling seeds but only 1/3 last-origin seeds, with a 7.949% rolling
regression on seed 33.

The two shift modes have identical parameters and measured peak VRAM.
Normalized added 0.392% fixed-step training latency and 3.140% inference
latency over output in this run. The evidence therefore does not satisfy step
5: `output` remains the compatibility default, while `normalized` remains an
explicit opt-in. Endogenous remains preferred when future covariates are not a
required model input.

### Seed-33 diagnosis and bounded-candidate result

The follow-up diagnostic run at `f175ca1` showed that seed 33 was not driven by
the future head's zero-input bias. The feature-conditioned effect averaged
0.1204 history-standard-deviation units versus 0.0069 for zero-input bias. The
error-delta correlation with history standard deviation was 0.151, and every
horizon, test series, and history-scale quartile regressed relative to output.
Series 12 had the largest standardized shift and the largest relative
regression. Removing the shift from the trained normalized model worsened its
rolling MAE by 11.170%, demonstrating backbone/head co-adaptation rather than a
standalone inference-time spike.

The `c=0.15` soft-bound candidate was then evaluated at clean commit `213dd07`
with the same seeds and protocol. It preserved the legacy three strategies
exactly and added no parameters or peak VRAM. Against output it recorded
-2.063% mean rolling improvement and -6.598% mean last-origin improvement. It
lost all three last-origin seeds and regressed 13.203% on seed 33 rolling.
Training and inference latency were 1.705% and 2.372% higher than output in the
final 100-step run.

Consequently, a single global bound is rejected as an accuracy promotion. The
field remains explicit and disabled by default for safety experiments; no
default preset may enable it. Any next candidate must condition reliability on
target history or forecast state and must repeat both rolling and last-origin
multi-seed validation before promotion.

### Validation history-gate capacity result

The follow-up RTX 5090 run at clean commit
`d9b7bb69b6ff7dc53ac9c16616e7815d5d10e337` tested whether target history can
identify when the normalized future residual should be attenuated. The local
artifact is
`artifacts/benchmarks/patchmixer-shift-space-results/patchmixer_history_gate_d9b7bb6_5090.json`
with SHA-256
`8e422721f0ab364ff1d49827076b19ed0295e750bde057f45ac096c272e25e20`.
It used the same dataset hash, seeds 11/22/33, 100 epochs, patience 15, and
series-disjoint split as the bounded-candidate run. Each seed supplied 441
validation windows from seven validation series. All prior training histories
and rolling/last-origin prediction hashes reproduced exactly.

For the architecture-faithful analysis, `base` was the trained normalized
model with its future shift disabled and `full` was the same model with its
shift enabled. A gate produced `base + g * (full - base)`, with either one
scalar per window or one value per window and horizon. Gate inputs were nine
forecast-time history summaries: level, scale, last value, trend, recent
4/12-week deviations, 52-week gap, range, and zero fraction. Ridge and KNN
gates used nested leave-one-series-out fitting and hyperparameter selection.
No test target was read. Target-aware oracle rows are unattainable ceilings.

Positive values below mean lower validation rolling error relative to the
ungated normalized model (`g=1`):

| Gate | Scalar MAE | Scalar MSE | Horizon MAE | Horizon MSE |
| --- | ---: | ---: | ---: | ---: |
| Series-OOF constant | -0.119% | -1.040% | -0.258% | -1.474% |
| Nested series-OOF KNN | -0.261% | -0.413% | -0.514% | -1.106% |
| Nested series-OOF ridge | -2.569% | -2.039% | -1.315% | -1.579% |
| Target-aware MSE oracle | +6.596% | +7.851% | +22.855% | +21.779% |

The scalar oracle improved MAE in every seed by 8.913%, 8.068%, and 2.806%,
so useful target-dependent attenuation exists in principle. The scalar OOF KNN
gate improved only seed 22; its mean MSE oracle-gain capture was -9.698%.
Ridge captured -25.083%. Horizon-wise KNN and ridge captured -5.832% and
-8.067%. They also underperformed the output-space reference on rolling MAE.
On last-origin validation, scalar KNN regressed 0.954% MAE and 3.000% MSE
relative to `g=1`; horizon-wise KNN regressed 0.357% and 1.408%. The oracle
ceilings remained positive, confirming that the failure is predictability, not
the absence of useful gate values.

A separate blend analysis interpolated between two independently trained
output-space and normalized-space models. Its series-OOF scalar constant was
near 0.5 and improved rolling MAE over output in all seeds by 4.471%, 1.713%,
and 4.553% (3.579% mean); mean MSE improvement was 6.513%. History-conditioned
KNN was weaker than that constant. Last-origin behavior was not stable: seed 11
regressed 3.278% MAE and 18.058% MSE. This is an ensemble hypothesis with two
model executions, not evidence for an internal normalized-residual gate.

Therefore no history-conditioned residual gate is promoted from this feature
set, and `output` remains the PatchMixer default. The oracle gap does justify
future research with richer forecast-state representations, but any such gate
must be trained end to end or with genuine outer-fold model training. The
approximately 0.5 dual-model blend may be tested separately only after its
memory/latency cost and held-out test behavior are measured. The present OOF
analysis applies only to the post-hoc gate; base checkpoints were selected on
the complete validation split, so these numbers are optimistic capacity
characterization rather than held-out generalization estimates.

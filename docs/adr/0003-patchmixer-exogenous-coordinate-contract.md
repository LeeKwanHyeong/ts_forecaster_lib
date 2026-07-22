# ADR 0003: PatchMixer exogenous coordinate contract

## Status

Accepted for compatibility and implementation planning.

## Context

PatchMixer receives exogenous tensors that may already have been standardized by
the data pipeline. Historical fields named `exo_is_normalized_default` and
`exo_is_normalized` do not perform that standardization and do not change the
current model output. Reusing either flag to choose where the future-exogenous
head is applied would mix two different concerns:

- the coordinate system of the exogenous input features; and
- the target coordinate system of the learned forecast shift.

The current future head emits one shift per horizon. Point and Quantile models
add it after RevIN denormalization, so the learned value is interpreted directly
in raw target units. The RTX 5090 gated-fusion baseline showed that this path
changed predictions by only 11-21 target units and was effectively ignored at
the tested demand scale.

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

The rollout is staged so the config cannot silently select an unimplemented
forward path. Stage one exposes
`future_exo_shift_space: Literal["output"] = "output"`. Missing fields in
legacy checkpoints resolve to `output`, while `normalized` and unknown values
fail at model construction. Stage two will expand the field to exactly two
values when the normalized forward path lands:

| Value | Meaning | Compatibility |
| --- | --- | --- |
| `output` | The future head emits raw target-unit corrections added after target denormalization. | Default for missing fields and all legacy checkpoints. |
| `normalized` | The future head emits target RevIN-space corrections added before target denormalization. | Opt-in candidate requiring its own accuracy baseline. |

This field describes the output coordinate of the future-exogenous head. It
does not describe whether the exogenous inputs themselves were standardized.
An accepted value must always have a complete forward implementation;
unsupported and unknown values fail at model construction.

## Application points

| Output mode | `normalized` insertion point | `output` insertion point |
| --- | --- | --- |
| Point | After last-value anchoring and before eval clipping and RevIN denormalization. | After RevIN denormalization and before the final non-negative transform. |
| Quantile | Add the same horizon shift to every quantile after anchoring and before eval clipping and per-quantile denormalization. | Add the same shift to every quantile after per-quantile denormalization. |
| Distribution | Add to `loc` only, after normalized-space location refinements and before location denormalization. | Add to `loc` only after location denormalization. |

Distribution scale and other distribution parameters are never shifted. A
shared shift across quantiles preserves their ordering. When RevIN is disabled,
the two coordinate systems are numerically identical, but the selected path
remains explicit for checkpoint reproducibility.

## Alternatives considered

1. Reinterpret `exo_is_normalized` as the shift-space selector. Rejected because
   existing values are no-ops and their name refers to the input, not the target.
2. Keep output-space shift as the only mode. Compatible, but poorly scaled for
   heterogeneous series and already shown to be nearly inactive.
3. Make normalized-space shift the new default. Rejected because it changes old
   checkpoint behavior and requires multi-seed evidence before promotion.

## Validation sequence

1. Preserve the legacy no-op and missing-config checkpoint contracts in tests.
2. Expose the output-only field while keeping `output` as the behavior of
   checkpoints without it.
3. Implement `normalized` as an explicit opt-in without changing parameter
   names or state-dict shapes.
4. Compare Endogenous, output-shift, and normalized-shift variants with the same
   data, split, loss, seeds, and RTX 5090 performance protocol.
5. Promote a new default only when both rolling and last-origin evidence justify
   the compatibility cost.

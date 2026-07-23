# ADR 0004: PatchMixer public model consolidation

## Status

Accepted.

## Context

PatchMixer exposed a paper-faithful point model alongside project Enhanced,
distribution, quantile and exogenous variants. The overlapping names made the
family default unclear and forced retired output modes through the active trainer.
The paper model had already won the controlled three-seed baseline and required
substantially fewer parameters and GPU memory.

## Decision

The public PatchMixer registry contains exactly two trainable artifacts.

| Key | Responsibility |
|---|---|
| `patchmixer` | paper-faithful endogenous point forecast |
| `patchmixer_exo` | project exogenous point forecast |

`PatchMixerModel` and `PatchMixerConfig` are the canonical endogenous names.
`PatchMixerExogenousModel` owns its exogenous boundary and does not inherit the
retired Enhanced identity. Family expansion resolves to `patchmixer` only.

Enhanced endogenous, distribution and quantile training is removed. Their keys
remain in a private legacy registry only when required to identify and restore a
supported checkpoint schema. They are never returned by the public model list and
public training requests reject them.

## Compatibility

- `patchmixer_original` resolves to `patchmixer`.
- `patchmixer_exogenous` resolves to `patchmixer_exo`.
- historical class/config names remain hidden aliases where Python or checkpoint
  reconstruction requires them.
- supported v1/v2/v3 Enhanced/distribution/quantile schemas restore exactly.
- unversioned `BaseModel`/`QuantileModel` artifacts must exact-load; incompatible
  schemas fail even when the caller requests non-strict loading.

No automatic conversion is attempted between paper and project state dicts.

## Consequences

The active API is easier to read and model selection is explicit. This is a
training API breaking change for callers that requested retired PatchMixer keys.
Existing supported checkpoint inference remains available. Reintroducing a
probabilistic PatchMixer requires a new model responsibility and validation
contract rather than restoring branches in the active point trainer.

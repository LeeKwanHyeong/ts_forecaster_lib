# Ten-model private Wheel compatibility

## Decision

The SELLM-inclusive private Wheel built from commit
`ef59fd068d5254d7d4cc56a3c23a24d3193fafaf` is compatible with all ten
registered weekly L52/H26 checkpoints. This closes the Demand Engine handoff
requirement for an integrated Wheel compatibility receipt. It does not activate
the Wheel in the port 8011 runtime or authorize a database write.

## Verified Wheel

- Distribution: `modeling-module==0.2.0`
- Profile: `sellm`
- Filename: `modeling_module-0.2.0-1private-cp312-none-any.whl`
- SHA-256: `20803a0498e0a4cd37f4848eaf0910d6d38796ce7840eb49dad8cc62bfa5e78a`
- Runtime: CPython 3.12.13, PyTorch 2.11.0+cu130, CUDA 13.0
- GPU: NVIDIA GeForce RTX 5090

## Existing eight-model parity

The five endogenous and three exogenous checkpoints were loaded from the
sealed Demand Engine production registry. Each checkpoint was restored with
`strict=True` from both the previous non-SELLM Wheel and the new SELLM Wheel.
Both isolated installs received the same deterministic two-series L52 input.
Exogenous models additionally received their registered past and future
continuous widths.

All eight models passed:

- checkpoint SHA-256 validation;
- strict checkpoint restoration;
- identical state-dict key, shape and dtype schema;
- two complete W0-W25 outputs with no non-finite values;
- exact output parity with maximum absolute difference `0.0`.

The quantile PatchTST comparison includes point, q10, q50 and q90 outputs.
ExoTST and PatchTST Exogenous use 12 past and 12 future continuous features;
TimeXer uses 12 past continuous features and no future features.

## ICL model verification

SELLM and AutoTimes were already verified from the same Wheel in the sealed
`wheel-provenance.json` receipt. Both corrected checkpoints passed strict load,
finite H26 output and source-to-corrected prediction parity with maximum
absolute difference `0.0`. The ICL receipt remains authoritative for Qwen
local-path and Transformers dependency verification.

## Registry metadata warnings

The Wheel and checkpoint contracts are compatible, but the existing Demand
Engine registry contains two pre-existing parameter-count metadata differences:

| Model | Registry | Restored checkpoint |
| --- | ---: | ---: |
| PatchMixer | 125,070 | 124,866 |
| TimeMixer | 101,447 | 21,447 |

These differences are present independently of the new Wheel and did not alter
strict load, state-dict schema or predictions. Historical registry v1 and its
execution contracts must not be rewritten. A future combined registry version
should use the restored counts and receive a new registry seal.

## Evidence

- Integrated receipt: `docs/TenModelPrivateWheelCompatibility.json`
- Integrated receipt seal:
  `bf40d2a005e5d33507c2061f9f9651e9b7449b0a54e1f7f842dc7e1f584d1173`
- Legacy eight-model receipt seal:
  `f29b705eb9eae83b0e2d56a455fd6e18849e80c1a3905d9225167fc05c07abfc`
- ICL two-model Wheel receipt seal:
  `2280ead86421d9a5d68e49865a74a27d974a1235406f53bad9c007599ad72839`
- RTX 5090 artifact root:
  `/home/leekwanhyeong/artifacts/icl-operational/ef59fd0/wheel`

The verification did not install the Wheel into `ai_env`, modify the active
Demand Engine runtime, touch port 8011, or write to a database.

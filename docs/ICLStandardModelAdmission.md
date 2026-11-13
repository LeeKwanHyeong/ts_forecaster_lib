# Standard admission for AutoTimes and SELLM

Effective source policy: 2026-10-30, by explicit user instruction.

- Both current ModelSpec entries use `operational_admission_status="approved"`.
- The old exception notes are removed from the active model catalog.
- Future ICL production checkpoint exports read admission from ModelSpec.
- Model architecture, checkpoint loading, ICL/exogenous input and output validation are unchanged.
- DemandEngine decides default eligibility and readiness. Its exogenous-enabled default selection
  includes both models; endogenous-only execution continues to exclude ICL models.

Existing checkpoint metadata, qualification results and `ICLOperationalExceptionApproval.json`
remain historical sealed evidence. They are not rewritten or represented as new quality qualification.
The current catalog and exported metadata do not use that old exception as the admission state.

Verified offline: 11 AutoTimes/model-catalog tests and 2 small-fixture production metadata
save/load tests. No production training, checkpoint replacement, wheel build or shared Runtime
deployment was performed.

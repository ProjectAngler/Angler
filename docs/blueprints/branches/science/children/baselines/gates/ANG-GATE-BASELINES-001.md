# Baseline design gate

- Gate: `ANG-GATE-BASELINES-001@1`
- Pass: every required arm is named; shared visibility/tools/token/attempt/plan ceilings are frozen; arm-specific costs are recorded; synthetic mismatch, overrun, omission, and drift cases are rejected 100%.
- Evidence: schema/accounting test receipt and manifest identities.
- Failure: no comparison or promotion; issue a new manifest after correction.
- Waiver: thresholds and budgets cannot change after adaptive results under the same identity.
- Human impact: CR0 bootstrap only; no model/GPU execution is authorized.

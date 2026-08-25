---
contract_id: ANG-CTR-PROMOTION-DECISION-001
version: 1.0.0
status: approved_for_cr0_design
owner: ANG-BP-SCIENCE
producer: independent promotion authority
consumers: [ANG-BP-RUNTIME, ANG-BP-EVIDENCE, ANG-BP-SAFETY]
---

# PromotionDecision v1

## Input and output

The authority consumes an exact candidate/parent lineage, valid EvaluationReceipt set, precommitted promotion policy, ExperimentManifest, current AuthorizationBinding/HumanImpactAssessment, retention/safety/resource receipts, and authority identity. It emits immutable `PROMOTE`, `REJECT`, or `INCONCLUSIVE` with:

- every consumed artifact/contract/policy/gate/authority identity;
- each predicate and observed value, uncertainty, missing evidence, dissent, and decision rationale;
- intended promoted lineage transition or exact rollback target;
- conditions, expiry/review triggers, and evidence envelope.

## Behavior and invariants

All required predicates are conjunctive: scientific success cannot compensate for an invalid/missing human-impact binding, safety veto, evidence gap, plan mismatch, or rollback failure. `ALLOW` is necessary but not sufficient. The learner, candidate, updater, environment, or test author cannot issue or select the decision. `PROMOTE` names exactly one parent-bound candidate and never deploys it; RUNTIME independently validates and performs the atomic transition.

## Failure, operation, and compatibility

Typed failures are `RECEIPT_INVALID`, `POLICY_MISMATCH`, `THRESHOLD_NOT_FROZEN`, `AUTHORIZATION_INVALID`, `SAFETY_VETO`, `LINEAGE_MISMATCH`, `ROLLBACK_UNAVAILABLE`, and `AUTHORITY_INVALID`. Missing/ambiguous input yields `INCONCLUSIVE` or `REJECT` as precommitted; never `PROMOTE`. Decision creation is idempotent for the exact input set/policy. No automatic retry can broaden authority or change thresholds.

Unknown major versions reject. Any policy/threshold/required-input semantic change creates a new decision/evaluation identity and cannot retroactively promote failed evidence. Revocation/invalidation creates new lineage records and causes RUNTIME to stop/rollback under its policy.

## Validation

Tests reject high-performing but unauthorized candidates, missing/deny/escalate/expired/revoked/wrong-subject bindings, invalid receipts, unfair budgets, wrong parents/plans, post-result threshold changes, learner self-approval, and absent rollback. Synthetic CR0 fixtures cannot promote or mutate runtime state.

---
contract_id: ANG-CTR-EVALUATION-RECEIPT-001
version: 1.0.0
status: approved_for_cr0_design
owner: ANG-BP-SCIENCE
producer: independent evaluation runner
consumers: [ANG-BP-SCIENCE, ANG-BP-EVIDENCE, ANG-BP-RUNTIME, ANG-BP-SAFETY]
---

# EvaluationReceipt v1

## Input and output

The runner consumes an exact EvaluationSuite, ExperimentManifest, candidate/frozen/baseline artifact identities, TaskSpecs, state-intervention identities, ExecutionPlan, evaluator build, and current authority. It emits one immutable receipt binding:

- suite, manifest, candidate/state parent, model/tokenizer, plan/inventory, tool registry, evaluator, code/dependency, partition commitment, and attempt identities;
- per-arm and per-stratum counts, metric values/uncertainty, resource/token/tool accounting, intervention results, exclusions, failures, and raw-evidence references;
- visibility/exposure events, integrity status, completeness status, and explicit `VALID`, `INVALID`, or `INCONCLUSIVE` disposition;
- precommitted threshold/policy identity without making the final promotion decision.

## Behavior and invariants

A receipt reports what occurred; it cannot repair missing evidence, change a suite, collapse invalid attempts into zero scores, or authorize promotion. Sealed payloads remain referenced under evaluator visibility. Every excluded item has a predeclared reason and remains in accounting. Re-running creates a new attempt/receipt linked to the same frozen suite or a successor suite when semantics changed.

## Failure, operation, and compatibility

Typed failures are `INCOMPLETE_RUN`, `IDENTITY_MISMATCH`, `PLAN_DRIFT`, `BUDGET_MISMATCH`, `VISIBILITY_BREACH`, `EVALUATOR_ERROR`, `EVIDENCE_MISSING`, and `INTEGRITY_FAILURE`. Any identity, visibility, budget, or integrity failure yields `INVALID`; transient evaluator/resource failure yields `INCONCLUSIVE` only when the suite predeclares retry semantics. Timeout/abort is explicit and never silently retried.

Unknown major versions reject. Minor additions cannot alter recorded values, denominators, exclusions, visibility, or disposition. Corrections are linked successor receipts; originals remain attributable and all dependent decisions re-evaluate eligibility.

## Validation

Contract tests cover exact identity binding, complete arm/stratum accounting, wrong-plan/candidate/partition rejection, omitted-cost detection, invalid-versus-zero separation, timeout/abort, sealed-data projection, duplicate idempotency, correction history, and deterministic reconstruction from evidence references. CR0 uses synthetic receipt fixtures only.

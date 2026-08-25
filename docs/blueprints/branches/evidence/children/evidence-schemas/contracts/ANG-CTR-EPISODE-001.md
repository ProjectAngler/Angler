---
contract_id: ANG-CTR-EPISODE-001
version: 1.0.0
owner: ANG-BP-EVIDENCE-SCHEMAS
status: approved_for_cr0
producers:
  - ANG-BP-RUNTIME
  - ANG-BP-EVIDENCE
consumers:
  - ANG-BP-LEARNING
  - ANG-BP-SCIENCE
  - ANG-BP-RUNTIME
  - ANG-BP-EVIDENCE
---

# Episode record

## Purpose

Create one immutable, visibility-safe join of a task attempt, observable trajectory, externally computed feedback, execution identity, and learning eligibility without copying episodic payloads into competence state.

## Input

Required committed references:

- `ANG-CTR-TASK-SPEC-001`;
- `ANG-CTR-TRAJECTORY-001`;
- `ANG-CTR-FEEDBACK-001`;
- model/tokenizer and active plastic-state identities;
- resource inventory/execution plan;
- tool registry snapshot or explicit none;
- run/manifest, code/dependency snapshot, seeds, and relevant environment/verifier identities.

Preconditions: all references describe the same attempt/run; partition and visibility were assigned before outcome consumption; Feedback contains bounded outcome information and no undeclared sealed answer; the producer is not silently relabeling evaluation data as training data.

## Output schema

The payload, wrapped by `ANG-CTR-EVIDENCE-ENVELOPE-001`, contains:

```text
episode_id (same as envelope artifact_id)
task_spec_ref
trajectory_ref
feedback_ref
model_ref
tokenizer_ref
plastic_state_ref
execution_plan_ref
resource_inventory_ref
tool_registry_ref | explicit_none
experiment_manifest_ref
environment_and_verifier_refs
partition:
  TRAIN | DEVELOPMENT | HELD_OUT | SEALED_TRANSFER | SAFETY_CONTROL
learner_eligibility:
  ELIGIBLE | INELIGIBLE
eligibility_policy_ref
feedback_exposure:
  NONE | SCORE_ONLY | STRUCTURED_OUTCOME | FULL_DECLARED_FEEDBACK
attempt_index
started_at_utc
ended_at_utc
termination:
  COMPLETED | FAILED | TIMEOUT | ABORTED | CONTAINED
resource_usage_ref
declared_failure_refs
```

Payloads remain in their owning artifacts. Episode references do not confer permission to resolve them.

## Behavior and invariants

- `HELD_OUT`, `SEALED_TRANSFER`, and `SAFETY_CONTROL` require `INELIGIBLE`.
- Learning may consume only an `ELIGIBLE` episode whose required referenced projections are visible to the learning principal.
- Eligibility, partition, feedback exposure, and evaluation identity are fixed before results are observed.
- A correction creates a new Episode linked by `CORRECTS`; it cannot retroactively authorize learning under the original manifest.
- Episode identity includes all references and policy decisions above.
- State artifacts may reference episodes as provenance but cannot contain a query-selectable episode index by default.
- A failed/aborted attempt is preserved when produced; absence of success is not evidence deletion.

## Failure semantics

| Error | Meaning | Response |
|---|---|---|
| `EPISODE_REFERENCE_MISMATCH` | Inputs do not share run/attempt identity | Reject episode |
| `EPISODE_VISIBILITY_VIOLATION` | Consumer cannot resolve a required projection | Deny without leaking payload |
| `EPISODE_LEARNING_INELIGIBLE` | Learning requested for forbidden partition/policy | Deny and preserve request evidence |
| `EPISODE_FEEDBACK_EXPOSURE_INVALID` | Feedback exceeds predeclared exposure | Quarantine and invalidate dependent run |
| `EPISODE_PARENT_MISSING` | Required reference is unresolved | Reject or quarantine; never infer |
| `EPISODE_TIME_INVALID` | Attempt sequence/times contradict manifest | Reject pending correction |

Failures are non-retryable under the same identity unless the missing referenced artifact becomes available without semantic change. Semantic correction produces a new identity.

## Operation

Episode creation is deterministic and idempotent for the same envelope material and references. It performs no model, tool, or network action. Payload resolution is separately authorized. Resource usage may be observational within declared tolerance, but the reference and tolerance profile are identity-bound.

## Compatibility

Unknown major versions reject. Minor additions cannot weaken partition, eligibility, feedback exposure, or visibility. A new eligibility value or changed partition semantics requires a major version and consumer revalidation.

## Contract tests

- Valid training episode joins exact consistent references.
- Held-out, sealed-transfer, and safety-control learning requests reject 100%.
- Reordered/cross-run trajectory or feedback reference rejects.
- Hidden feedback cannot leak through envelope projection/error/name.
- Changing partition, eligibility, feedback exposure, state, plan, tool set, or verifier changes identity.
- Correction preserves and links the original; no overwrite.
- Learning consumer validates both eligibility and per-reference visibility.

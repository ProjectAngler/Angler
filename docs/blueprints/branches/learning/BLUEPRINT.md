---
blueprint_id: ANG-BP-LEARNING
title: Plasticity and competence learning
parent_id: ANG-BP-ROOT
revision: 1
tier: 1
design_status: draft
delivery_status: not_started
accountable_owner: unassigned
execution_owner: unassigned
updated_at: 2026-08-25
parent_revision: 2
required_children:
  - ANG-BP-FEEDBACK-UPDATE
  - ANG-BP-REPLAY-SELECTION
  - ANG-BP-META-UPDATER
  - ANG-BP-CONSOLIDATION
  - ANG-BP-RETENTION
  - ANG-BP-SKILL-COMPOSITION
depends_on:
  - ANG-BP-RUNTIME
  - ANG-BP-SCIENCE
  - ANG-BP-EVIDENCE
  - ANG-BP-RESOURCES
  - ANG-BP-WORLDS
contracts_in:
  - ANG-CTR-EPISODE-001
  - ANG-CTR-FEEDBACK-001
  - ANG-CTR-PLASTIC-STATE-001
  - ANG-CTR-EXECUTION-PLAN-001
contracts_out:
  - ANG-CTR-UPDATE-PROPOSAL-001
gate: ANG-GATE-LEARNING-DESIGN-001
---

# Plasticity and competence learning

## Context capsule

This branch turns experience into bounded candidate changes to the one active competence lineage and later learns the update rule itself. The first mechanism is an FTTT-inspired, gradient-based LoRA update. Replay, meta-plasticity, consolidation, retention, and composition are added only after the minimal causal learner works.

## Contribution to the root

RUNTIME makes state change possible and SCIENCE judges it; LEARNING supplies the adaptive mechanism between them. Its success criterion is transfer after update, not lower loss on the episode that generated the update.

## Inherited invariants

Applies: `ANG-INV-STABLE-BASE-001`, `ANG-INV-ONE-COMPETENCE-001`, `ANG-INV-EVIDENCE-SEPARATION-001`, `ANG-INV-CAUSAL-PROMOTION-001`, `ANG-INV-REVERSIBLE-UPDATES-001`, and `ANG-INV-ELASTIC-COMPUTE-001`.

## Scope

- Compute bounded candidate changes from episodes, feedback, replay, and a resource budget.
- Begin with ordinary gradient updates restricted to plastic parameters.
- Select replay that protects prior skills without exposing hidden evaluation data.
- Learn a meta-updater from accepted and rejected transaction histories.
- Consolidate several temporary gains into surface-neutral competence.
- Measure and manage sequential retention, interference, drift, and composition.

## Explicit non-goals

- Mutating the base model in early phases.
- Owning state promotion, rollback, hidden evaluation, or sealed partitions.
- Retrieving a different adapter for each query or retaining one adapter per episode.
- Adding tools, generated curricula, or code mutation to rescue a failed minimal learner.
- Treating model-generated critique as authoritative feedback.

## Contracts

LEARNING consumes an immutable parent `PlasticState`, visible `Episode`, externally computed `Feedback`, safe replay, and `ExecutionPlan` budget. It emits a parent-bound `UpdateProposal` containing candidate material, optimizer/update receipt, losses, norms, budget use, and provenance. It never receives promotion answers.

Conceptually:

```text
U(phi, parent_state, episode, replay, budget) → UpdateProposal
```

`phi` is fixed/hand-designed in the first learner, then trainable in the meta-plastic phase.

## Child branch map

| Child ID | Outcome | Gate | Status |
|---|---|---|---|
| `ANG-BP-FEEDBACK-UPDATE` | Minimal bounded FTTT-style LoRA update | `ANG-GATE-FEEDBACK-UPDATE-001` | stub; first active learning child |
| `ANG-BP-REPLAY-SELECTION` | Retention-aware sampling without evaluation leakage | `ANG-GATE-REPLAY-001` | stub |
| `ANG-BP-META-UPDATER` | Learned update rule beating fixed updates under equal compute | `ANG-GATE-META-UPDATER-001` | deferred until M3 corpus exists |
| `ANG-BP-CONSOLIDATION` | Distilled surface-neutral competence from accepted histories | `ANG-GATE-CONSOLIDATION-001` | deferred until M3 |
| `ANG-BP-RETENTION` | Sequential acquisition with bounded interference and drift | `ANG-GATE-RETENTION-001` | deferred until consolidation |
| `ANG-BP-SKILL-COMPOSITION` | Independently learned procedures cooperate on novel tasks | `ANG-GATE-COMPOSITION-001` | deferred until multiple skills exist |

## Dependencies and sequencing

The FEEDBACK-UPDATE child depends on stable episode, feedback, state, and plan contracts plus hidden evaluation that it cannot access. REPLAY may start as a minimal fixed baseline but cannot become adaptive until evidence/partition boundaries are proven. META-UPDATER begins only after a sufficiently varied corpus of accepted and rejected transactions. CONSOLIDATION, RETENTION, and COMPOSITION follow the first causal success.

## Acceptance gate and evidence

`ANG-GATE-LEARNING-DESIGN-001` passes when each child has a bounded learning objective, visible data policy, update authority, compute budget, failure behavior, and scientific gate mapping. It must be possible to implement FEEDBACK-UPDATE without importing later subsystems.

The first substantive gate is M3: across multiple families, one persistent candidate must improve held-out structural variants above frozen and fair-RAG controls; zeroing must remove the gain, swapping must transfer it, replay must reconstruct it, and retention must remain inside the precommitted bound.

## Testing and validation

- Parameter-scope tests proving only plastic weights change.
- Gradient-step, norm, time, memory, and optimizer budget enforcement.
- Parent binding and update-receipt reproducibility.
- No access to hidden promotion labels or suite internals.
- Shuffled feedback, random update, and no-update controls.
- Retention, interference, composition, and cross-family transfer.
- Equal-experience/compute comparisons for meta-updater claims.

## Risks and rollback

- `ANG-RISK-EPISODE-MEMORIZATION-001`: update encodes episode surface content. Reject under substitutions/fair-RAG.
- `ANG-RISK-GRADIENT-LEAKAGE-001`: hidden evaluator data reaches loss. Invalidate lineage and evaluation.
- `ANG-RISK-INTERFERENCE-001`: new competence erases old. Reject proposal; revise replay/update bounds.
- `ANG-RISK-UPDATER-OVERFIT-001`: learned updater encodes development generators. Require cross-generator/domain tests.
- `ANG-RISK-STATE-DRIFT-001`: repeated small accepted updates degrade global performance. Checkpoint, consolidate, or restore an earlier promoted head.

## Resource scaling

The update contract declares budgets rather than assuming hardware. Plans may change adapter rank, target layers, batch, steps, checkpointing, offload, parallelism, or meta-batch size. Scientific comparisons match plans; changing model/topology uses migration instead of silently continuing an incompatible lineage.

## Current status and next leaf

Design is draft. Expand `ANG-BP-FEEDBACK-UPDATE` only after episode, feedback, state, execution-plan, and M3 evaluation interfaces reach compatible drafts. Later children remain stubs.

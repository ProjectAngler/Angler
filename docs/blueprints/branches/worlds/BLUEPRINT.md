---
blueprint_id: ANG-BP-WORLDS
title: Environments and curriculum
parent_id: ANG-BP-ROOT
revision: 2
tier: 1
design_status: approved_for_cr0
delivery_status: blocked_by_cr0_predecessors
accountable_owner: project_owner
execution_owner: codex_or_designated_builder
updated_at: 2026-08-25
parent_revision: 2
required_children:
  - ANG-BP-ENVIRONMENT-PROTOCOL
  - ANG-BP-FIXED-FAMILIES
  - ANG-BP-OUTCOME-VERIFIERS
  - ANG-BP-ENVIRONMENT-VALIDATION
  - ANG-BP-CURRICULUM
  - ANG-BP-GENERATED-WORLDS
depends_on:
  - ANG-BP-SCIENCE
  - ANG-BP-EVIDENCE
  - ANG-BP-SAFETY
  - ANG-BP-RESOURCES
contracts_out:
  - ANG-CTR-TASK-SPEC-001
  - ANG-CTR-OBSERVATION-001
  - ANG-CTR-ACTION-001
  - ANG-CTR-FEEDBACK-001
gate: ANG-GATE-WORLDS-DESIGN-001
adrs:
  - ANG-ADR-0002
  - ANG-ADR-0004
---

# Environments and curriculum

## Context capsule

This branch supplies controllable experiences, observations, action transitions, and bounded outcome feedback. It begins with fixed hand-authored procedural worlds, then may schedule or generate challenges near the learner's frontier only after independent evaluation is credible.

## Contribution to the root

The learner cannot acquire methods without experiences that contain a procedure but do not leak the desired method. WORLDS supplies the causal laboratory while SCIENCE independently decides what the results prove.

## Inherited invariants

Applies: `ANG-INV-EVIDENCE-SEPARATION-001`, `ANG-INV-CAUSAL-PROMOTION-001`, `ANG-INV-OUTCOME-JUDGES-001`, `ANG-INV-ELASTIC-COMPUTE-001`, and `ANG-INV-EXTERNAL-AUTHORITY-001`.

## Scope

- Define reset, step, score, seed, version, capability, visibility, and budget protocols.
- Provide initial procedural task families with independently generated surface forms.
- Return observable outcome feedback without prescribing reasoning steps.
- Validate determinism bounds, solvability, structural diversity, leakage, and shortcuts.
- Later schedule tasks near measured competence under an external objective.
- Much later generate executable environments and submit them to independent validation.

## Explicit non-goals

- Owning hidden promotion partitions or deciding scientific claims.
- Encoding target chains of thought or answer-specific scripts in verifiers.
- Letting a generated environment be its own sole validator.
- Using adaptive curricula before fixed-family transfer is demonstrated.
- Granting generated code unrestricted execution authority.

## Contracts

WORLDS emits a versioned `TaskSpec`, learner-visible `Observation`, transitions, and bounded `Feedback`. The task declares action/observation spaces, tool permissions, limits, seed policy, verifier identity, and visibility classes. Feedback reports externally observable results and may include failing cases, constraint violations, or prediction errors allowed by the task contract; it excludes hidden transfer answers.

## Child branch map

| Child ID | Outcome | Gate | Status |
|---|---|---|---|
| `ANG-BP-ENVIRONMENT-PROTOCOL` | Stable reset/step/score/version/budget interface | `ANG-GATE-ENV-PROTOCOL-001` | approved; construction-ready |
| `ANG-BP-FIXED-FAMILIES` | Initial symbolic, constraint, counterexample, causal, repair, and tool-choice generators | `ANG-GATE-FIXED-FAMILIES-001` | stub |
| `ANG-BP-OUTCOME-VERIFIERS` | Outcome-only signals with no prescribed method | `ANG-GATE-VERIFIERS-001` | stub |
| `ANG-BP-ENVIRONMENT-VALIDATION` | Solvability, diversity, leakage, shortcut, and containment checks | `ANG-GATE-ENV-VALIDATION-001` | stub |
| `ANG-BP-CURRICULUM` | Externally bounded competence-frontier scheduler | `ANG-GATE-CURRICULUM-001` | deferred until M5 |
| `ANG-BP-GENERATED-WORLDS` | SPADE-style generated tasks that transfer externally | `ANG-GATE-GENERATED-WORLDS-001` | deferred until M7 |

## Initial fixed families

The initial set spans symbolic rule induction, constraint decomposition, counterexample search, causal intervention, program repair, and tool-choice reasoning. Each family separates latent procedure from names, facts, symbols, ordering, and wording. SCIENCE owns which partitions and transformations support claims; WORLDS owns correct generation and outcome execution.

## Dependencies and sequencing

ENVIRONMENT-PROTOCOL joins EVIDENCE schemas and SCIENCE visibility rules in slice 01. FIXED-FAMILIES, VERIFIERS, and VALIDATION enable slices 02–04. CURRICULUM waits for lifelong competence; GENERATED-WORLDS waits for an independent validator and sealed human-authored transfer suite.

## Acceptance gate and evidence

`ANG-GATE-WORLDS-DESIGN-001` passes when each environment child has deterministic boundaries, version/seed identity, visibility rules, outcome-only feedback, resource limits, validation, and failure semantics. At least two initial families must support independent structural variants before the first learner can be authorized.

## Testing and validation

- Reset/step/score contract and seed replay.
- Generator uniqueness, disjointness, and latent-procedure preservation.
- Known-solution solvability and deliberately invalid task rejection.
- Feedback leakage and shortcut adversarial tests.
- Verifier mutation tests.
- Sandbox/resource cleanup for executable tasks.
- Cross-plan determinism/tolerance tests.
- Generated curriculum transfer to sealed external families when activated.

## Risks and rollback

- `ANG-RISK-VERIFIER-PRESCRIPTION-001`: deterministic code encodes the solution procedure. Redesign verifier around outcome.
- `ANG-RISK-TASK-LEAKAGE-001`: surface cue predicts label/procedure. Invalidate generator version.
- `ANG-RISK-CURRICULUM-COLLUSION-001`: generator produces easy/leaky tasks. Independent validation and sealed transfer.
- `ANG-RISK-WORLD-NONREPLAY-001`: seed/version fails to reconstruct task. Exclude evidence and repair protocol.

## Resource scaling

Environment semantics remain fixed while execution may serialize locally or parallelize across devices/nodes. Scheduler throughput cannot change which tasks count as promotion evidence. Resource-intensive executable worlds declare requirements and receive a validated sub-plan.

## Current status and next leaf

The Tier-1 ownership boundary and ENVIRONMENT-PROTOCOL design are approved for Construction Release 0 schema/scaffold work. Fixed families, executable verifiers, and validation remain unimplemented, so no task execution or learner use is authorized. Next execute only the release-listed protocol leaf; subsequent leaves will implement the two selected families after independent validation design is ready.

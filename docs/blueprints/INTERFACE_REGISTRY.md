# Cross-branch interface registry

This registry is the canonical vocabulary for branch boundaries. It defines ownership and meaning, not implementation schemas. Detailed versioned schemas are created under the owning branch and linked here.

## Canonical lifecycle

```text
ResourceInventory → ExecutionPlan → TaskSpec → Observation ↔ Action/Observation
→ terminal Trajectory + Feedback → Episode
→ UpdateProposal → CandidateState content commitment → EvaluationReceipt
→ HumanImpactAssessment(ALLOW | DENY | ESCALATE)
→ final CandidateState envelope → AuthorizationBinding
→ PromotionDecision → PromotedState | exact rollback
```

Every persistent/evaluative item is wrapped by `ANG-CTR-EVIDENCE-ENVELOPE-001` and related through `ANG-CTR-ARTIFACT-LINEAGE-001`. The staged candidate/assessment/envelope/binding sequence prevents a circular artifact/assessment hash dependency.

## Interface cards

| Contract ID | Owner | Primary consumers | Meaning | Status |
|---|---|---|---|---|
| `ANG-CTR-EVIDENCE-ENVELOPE-001` | EVIDENCE | every persistent artifact producer/consumer | Canonical content commitment, semantic artifact identity, provenance, visibility, compatibility, and integrity envelope | approved for CR0 design; [spec](branches/evidence/children/evidence-schemas/contracts/ANG-CTR-EVIDENCE-ENVELOPE-001.md) |
| `ANG-CTR-ARTIFACT-LINEAGE-001` | EVIDENCE | all branches | Immutable typed parentage, correction, version, invalidation, revocation, promotion, rollback, and migration graph | approved for CR0 design; [spec](branches/evidence/children/artifact-lineage/contracts/ANG-CTR-ARTIFACT-LINEAGE-001.md) |
| `ANG-CTR-RESOURCE-INVENTORY-001` | RESOURCES | RESOURCES, EVIDENCE | Measured hardware, software, topology, uncertainty, and separate administrative capacity | approved design; [spec](branches/resources/contracts/ANG-CTR-RESOURCE-INVENTORY-001.md) |
| `ANG-CTR-EXECUTION-PLAN-001` | RESOURCES | RUNTIME, LEARNING, SCIENCE, EVIDENCE | Constraint-valid model/precision/placement/budget configuration with reserved headroom | approved design; [spec](branches/resources/contracts/ANG-CTR-EXECUTION-PLAN-001.md) |
| `ANG-CTR-TASK-SPEC-001` | WORLDS | RUNTIME, SCIENCE, EVIDENCE | Versioned task identity, observation/action spaces, budgets, permissions, and visibility policy | approved design; [spec](branches/worlds/contracts/ANG-CTR-TASK-SPEC-001.md) |
| `ANG-CTR-OBSERVATION-001` | WORLDS | RUNTIME | Learner-visible state with provenance, attempt/step identity, budget, and trust classification | approved design; [spec](branches/worlds/contracts/ANG-CTR-OBSERVATION-001.md) |
| `ANG-CTR-ACTION-001` | WORLDS (accepted environment input) | RUNTIME producer, WORLDS/EVIDENCE consumers | Instance/attempt/prior-observation-bound action and exactly-once step request with permission, budget, deadline, and idempotency semantics | approved for CR0 design; [spec](branches/worlds/contracts/ANG-CTR-ACTION-001.md) |
| `ANG-CTR-TRAJECTORY-001` | RUNTIME | WORLDS, LEARNING, EVIDENCE | Complete observable model/tool interaction for one attempt | conceptual |
| `ANG-CTR-FEEDBACK-001` | WORLDS | LEARNING, EVIDENCE, SCIENCE | Bounded externally computed outcome signal without hidden transfer answers or prescribed method | approved design; [spec](branches/worlds/contracts/ANG-CTR-FEEDBACK-001.md) |
| `ANG-CTR-EPISODE-001` | EVIDENCE | LEARNING, SCIENCE, RUNTIME | Immutable reference join of task, trajectory, feedback, eligibility, and identities | approved for CR0 design; [spec](branches/evidence/children/evidence-schemas/contracts/ANG-CTR-EPISODE-001.md) |
| `ANG-CTR-PLASTIC-STATE-001` | RUNTIME | LEARNING, SCIENCE, EVIDENCE, RESOURCES | Content-addressed state tied to a model and adapter topology | conceptual |
| `ANG-CTR-UPDATE-PROPOSAL-001` | LEARNING | RUNTIME, SCIENCE, EVIDENCE | Parent-bound candidate state plus bounded update receipt | conceptual |
| `ANG-CTR-EVALUATION-SUITE-001` | SCIENCE | RUNTIME, EVIDENCE | Hidden transfer, retention, causal, leakage, comparison-budget, and resource-controlled test identity | approved for CR0 design; [spec](branches/science/contracts/ANG-CTR-EVALUATION-SUITE-001.md) |
| `ANG-CTR-EVALUATION-RECEIPT-001` | SCIENCE | RUNTIME, EVIDENCE, SAFETY | Immutable identity-bound results, interventions, accounting, uncertainty, validity, and gate-relevant statistics | approved for CR0 design; [spec](branches/science/contracts/ANG-CTR-EVALUATION-RECEIPT-001.md) |
| `ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001` | SAFETY | all promotable-artifact owners, SCIENCE, EVIDENCE, human authority | Exact-subject purpose, affected people, benefits, harms, rights, agency, distribution, uncertainty, authority, conditions, monitoring, expiry, and rollback | active design; [spec](branches/safety/contracts/ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001.md) |
| `ANG-CTR-AUTHORIZATION-BINDING-001` | EVIDENCE structure; SAFETY authority | RUNTIME, SCIENCE, EVIDENCE, SAFETY | Sidecar binding joining a final artifact envelope to its exact current assessment, policy, authority, conditions, and subject tuple | approved for CR0 design; [spec](branches/evidence/children/artifact-lineage/contracts/ANG-CTR-AUTHORIZATION-BINDING-001.md) |
| `ANG-CTR-PROMOTION-DECISION-001` | SCIENCE | RUNTIME, EVIDENCE, SAFETY | Promote/reject/inconclusive decision under precommitted conjunctive policy; human-impact authorization and safety veto remain independent requirements | approved for CR0 design; [spec](branches/science/contracts/ANG-CTR-PROMOTION-DECISION-001.md) |
| `ANG-CTR-TRANSACTION-RECEIPT-001` | RUNTIME | EVIDENCE, SCIENCE | Atomic propose/evaluate/promote-or-restore outcome | conceptual |
| `ANG-CTR-TOOL-PACKAGE-001` | TOOLS | RUNTIME, SAFETY, EVIDENCE | Typed executable capability with provenance, limits, and tests | conceptual |
| `ANG-CTR-TOOL-RECEIPT-001` | TOOLS | RUNTIME, SAFETY, EVIDENCE | Independent sandbox and functional validation result | conceptual |
| `ANG-CTR-MIGRATION-PROPOSAL-001` | RESOURCES | RUNTIME, LEARNING, SCIENCE, EVIDENCE | Explicit replay/distillation of competence to a new model/topology | conceptual |
| `ANG-CTR-EXPERIMENT-MANIFEST-001` | EVIDENCE | all branches | Frozen identities, seeds/commitments, partitions, resource plan, gates, thresholds, stop/rollback, and intended claims | approved for CR0 design; [spec](branches/evidence/children/evidence-schemas/contracts/ANG-CTR-EXPERIMENT-MANIFEST-001.md) |

## Shared identity envelope

Every persistent or evaluative artifact carries the canonical EVIDENCE envelope. It separates:

- `content_id` or a protected commitment over canonical payload bytes;
- `artifact_id` over canonical semantic envelope material;
- attestations/signatures and storage locators, which cannot alter those identities;
- immutable authority/lineage records from rebuildable indexes and “current” projections.

Visibility classes are `LEARNER_VISIBLE`, `CONTROL_PLANE`, `SEALED_EVALUATION`, `HUMAN_AUTHORITY`, and future-only `RESTRICTED_PERSONAL`. CR0 prohibits real-person data. A small/enumerable sealed value cannot expose a raw public digest; it uses a keyed or authenticated-encryption commitment.

The semantic envelope carries, where applicable:

- code and dependency snapshot;
- foundation model and tokenizer signature;
- adapter topology and parent state;
- updater and optimizer identity;
- task generator and data-partition identity;
- tool registry snapshot;
- random seeds;
- resource inventory and execution plan;
- producing contract versions;
- evidence and gate identities.

Authorization is non-circular: SAFETY assesses the final content commitment plus parent/scope/plan/permission/context/duration tuple; the final artifact envelope records the assessment reference or requirement; `ANG-CTR-AUTHORIZATION-BINDING-001` then joins the completed artifact and assessment. Promotion consumes the current binding. An artifact-local `authorized`, `safe`, `promoted`, or `current` boolean is never authoritative.

Unknown major contract versions fail closed. Every affected producer/consumer must accept and test a contract before emitting or consuming admissible artifacts.

## Change rule

A branch may implement its owned contract but may not silently redefine it. A breaking change requires an ADR, a new contract version, an impact list, migration behavior, and `needs_revision` status on affected consumers.

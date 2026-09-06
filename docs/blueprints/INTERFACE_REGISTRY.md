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
| `ANG-CTR-SITUATED-RECALL-001` | EVIDENCE | RUNTIME, LEARNING, SCIENCE | Disposable retrieval candidate rejoined to canonical evidence with acquisition age, landmark relations, separate world-validity time, provenance, and visibility | experimental 0.1.0; [ADR](decisions/ANG-ADR-0006-MOVING-ORIGIN-COGNEE-SITUATED-MEMORY.md) |
| `ANG-CTR-COGNITIVE-MEMORY-001` | EVIDENCE structure; RUNTIME transaction | RUNTIME, LEARNING, SCIENCE | Immutable typed episodic, semantic, procedural, causal, self, or counterfactual record with epistemic status, provenance, temporal validity, relations, supersession, and rebuildable retrieval projections | experimental 0.1.0; [ADR](decisions/ANG-ADR-0007-UNIFIED-TEMPORAL-COGNITIVE-SYSTEM.md) |
| `ANG-CTR-PROSPECTIVE-COMMITMENT-001` | RUNTIME transaction; LEARNING content | RUNTIME, LEARNING, EVIDENCE | Immutable pre-execution learned prediction; not an action request, permission, or authority decision | experimental 0.1.0; [ADR](decisions/ANG-ADR-0008-EXPERIMENTAL-COGNITIVE-TRANSACTION-INTERFACES.md) |
| `ANG-CTR-COGNITIVE-EPISODE-001` | RUNTIME transaction; EVIDENCE canonical storage | RUNTIME, LEARNING, EVIDENCE | Experimental complete cognitive-turn record; distinct from and not admissible as approved `ANG-CTR-EPISODE-001` | experimental 0.1.0; [ADR](decisions/ANG-ADR-0008-EXPERIMENTAL-COGNITIVE-TRANSACTION-INTERFACES.md) |
| `ANG-CTR-COGNITIVE-GRAPH-PROJECTION-001` | RUNTIME projection; EVIDENCE canonical source | RUNTIME, EVIDENCE | Content-addressed rebuildable projection of a canonical cognitive episode and typed record; never canonical evidence or authority | experimental 0.1.0; [ADR](decisions/ANG-ADR-0008-EXPERIMENTAL-COGNITIVE-TRANSACTION-INTERFACES.md) |
| `ANG-CTR-PROSPECTIVE-TURN-RESERVATION-001` | RUNTIME | RUNTIME, LEARNING, EVIDENCE | Durable pre-effect decision aggregate binding exact execution context, agent/world references, pending learner bytes, and the sole external idempotency key | experimental 0.1.0; [ADR](decisions/ANG-ADR-0008-EXPERIMENTAL-COGNITIVE-TRANSACTION-INTERFACES.md) |
| `ANG-CTR-COGNITIVE-EXECUTION-REQUEST-001` | RUNTIME | RUNTIME, TOOLS, EVIDENCE | Exact claimed executor request derived from a reservation; grants no permission and is not approved `ANG-CTR-ACTION-001` | experimental 0.1.0; [ADR](decisions/ANG-ADR-0008-EXPERIMENTAL-COGNITIVE-TRANSACTION-INTERFACES.md) |
| `ANG-CTR-COGNITIVE-EXECUTION-RECEIPT-001` | RUNTIME | RUNTIME, LEARNING, EVIDENCE | Public executor result bound to one exact cognitive request; neither authorization nor objective outcome truth | experimental 0.1.0; [ADR](decisions/ANG-ADR-0008-EXPERIMENTAL-COGNITIVE-TRANSACTION-INTERFACES.md) |
| `ANG-CTR-OBJECTIVE-FEEDBACK-001` | RUNTIME binding; WORLDS outcome source | RUNTIME, LEARNING, EVIDENCE | Objective outcome bound to the exact completed cognitive context before learning; distinct from and not admissible as approved `ANG-CTR-FEEDBACK-001` | experimental 0.1.0; [ADR](decisions/ANG-ADR-0008-EXPERIMENTAL-COGNITIVE-TRANSACTION-INTERFACES.md) |
| `ANG-CTR-SITUATED-STATE-LINEAGE-001` | RUNTIME state transaction; LEARNING state content | RUNTIME, LEARNING, EVIDENCE | Evidence-bound operational agent/world/context and learned-state lineage; neither approved `ANG-CTR-PLASTIC-STATE-001` nor personhood, identity authority, external-world truth, or permission | experimental 0.1.0; [ADR](decisions/ANG-ADR-0009-PROSPECTIVE-ORIGIN-ACQUISITION-INTERFACES.md) |
| `ANG-CTR-PROSPECTIVE-DYNAMICS-BATCH-001` | RUNTIME transaction; LEARNING prediction content | RUNTIME, LEARNING, EVIDENCE | Immutable bounded sibling set of learned hypothetical predictions committed before claim; not an approved `ANG-CTR-ACTION-001`, permission, or observed fact | experimental 0.1.0; [ADR](decisions/ANG-ADR-0009-PROSPECTIVE-ORIGIN-ACQUISITION-INTERFACES.md) |
| `ANG-CTR-PROSPECTIVE-TURN-RESERVATION-002` | RUNTIME | RUNTIME, LEARNING, EVIDENCE | Successor compatibility binding of a prospective batch and selected branch to the exact legacy reservation and its sole executor idempotency key; not an approved `ANG-CTR-ACTION-001` or permission | experimental 0.1.0; [ADR](decisions/ANG-ADR-0009-PROSPECTIVE-ORIGIN-ACQUISITION-INTERFACES.md) |
| `ANG-CTR-PROSPECTIVE-RESOLUTION-001` | RUNTIME binding; WORLDS outcome source | RUNTIME, LEARNING, EVIDENCE | Append-only lifecycle and prediction-resolution record; objective outcome requires the exact experimental feedback record and remains distinct from approved `ANG-CTR-FEEDBACK-001` | experimental 0.1.0; [ADR](decisions/ANG-ADR-0009-PROSPECTIVE-ORIGIN-ACQUISITION-INTERFACES.md) |
| `ANG-CTR-COGNITIVE-EPISODE-002` | RUNTIME transaction; EVIDENCE canonical storage | RUNTIME, LEARNING, EVIDENCE | Observed-only experimental successor episode referencing a prospective resolution; distinct from and not admissible as approved `ANG-CTR-EPISODE-001` | experimental 0.1.0; [ADR](decisions/ANG-ADR-0009-PROSPECTIVE-ORIGIN-ACQUISITION-INTERFACES.md) |
| `ANG-CTR-COGNITIVE-ACQUISITION-001` | RUNTIME sequencing; EVIDENCE record semantics | RUNTIME, LEARNING, EVIDENCE | Canonical local acquisition-order envelope for one typed memory record; its ordinal conveys neither evidence admission, truth, authorization, promotion, nor global contiguity without store validation | experimental 0.1.0; [ADR](decisions/ANG-ADR-0009-PROSPECTIVE-ORIGIN-ACQUISITION-INTERFACES.md) |
| `ANG-CTR-COGNITIVE-GRAPH-PROJECTION-002` | RUNTIME projection; EVIDENCE canonical source | RUNTIME, EVIDENCE | Content-addressed rebuildable projection of a generic canonical acquisition and typed record; disposable and never canonical evidence, authorization, truth, or authority | experimental 0.1.0; [ADR](decisions/ANG-ADR-0009-PROSPECTIVE-ORIGIN-ACQUISITION-INTERFACES.md) |
| `ANG-CTR-JENNY-GENESIS-001` | RUNTIME lineage | RUNTIME, EVIDENCE | Immutable operational Jenny 2.0 genesis binding to exact Jenny 1.x archive/source revisions; not personhood, consciousness, or active imported state | experimental 0.1.0; [ADR](decisions/ANG-ADR-0010-JENNY-GENESIS-TEMPORAL-AUTONOMY.md) |
| `ANG-CTR-TEMPORAL-NOW-001` | RUNTIME clock integrity | RUNTIME, LEARNING, EVIDENCE | Trusted UTC/local/monotonic sample with uncertainty, jump evidence, and Moving-Origin ordinal; neither semantic age nor world truth | experimental 0.2.0; [ADR](decisions/ANG-ADR-0010-JENNY-GENESIS-TEMPORAL-AUTONOMY.md) |
| `ANG-CTR-TEMPORAL-V2-001` | RUNTIME temporal form; EVIDENCE semantics | RUNTIME, LEARNING, EVIDENCE | Separate event/acquired/recorded/verified/valid times with provenance and uncertainty; no field implies truth or authorization | experimental 0.2.0; [ADR](decisions/ANG-ADR-0010-JENNY-GENESIS-TEMPORAL-AUTONOMY.md) |
| `ANG-CTR-PERSISTENT-AUTONOMY-STATE-001` | RUNTIME | RUNTIME, LEARNING, EVIDENCE | Crash-recoverable supervisor control and state-head record; not an approved Action, Feedback, Episode, PlasticState, permission, promotion, or readiness claim | experimental 0.1.0; [ADR](decisions/ANG-ADR-0010-JENNY-GENESIS-TEMPORAL-AUTONOMY.md) |
| `ANG-CTR-JENNY-TOOL-DEFINITION-001` | RUNTIME tool bridge | RUNTIME, TOOLS, SAFETY, EVIDENCE | Content-addressed versioned tool schema and declared scope; grants no execution authority and is not equivalent to approved `ANG-CTR-ACTION-001`, `ANG-CTR-FEEDBACK-001`, or `ANG-CTR-EPISODE-001` | experimental 0.1.0; [leaf](branches/runtime/work/ANG-WORK-RUNTIME-JENNY-PERSISTENT-AUTONOMY-V1-001.md) |
| `ANG-CTR-JENNY-TOOL-MANIFEST-001` | RUNTIME tool bridge | RUNTIME, TOOLS, SAFETY, EVIDENCE | Content-addressed set of exact tool-definition versions; grants no execution authority and is not equivalent to approved `ANG-CTR-ACTION-001`, `ANG-CTR-FEEDBACK-001`, or `ANG-CTR-EPISODE-001` | experimental 0.1.0; [leaf](branches/runtime/work/ANG-WORK-RUNTIME-JENNY-PERSISTENT-AUTONOMY-V1-001.md) |
| `ANG-CTR-JENNY-TOOL-CALL-001` | RUNTIME tool-operation store | RUNTIME, TOOLS, EVIDENCE | Idempotent reservation of arguments against one exact manifest and definition; it is intent, not execution authority, and is not equivalent to approved `ANG-CTR-ACTION-001`, `ANG-CTR-FEEDBACK-001`, or `ANG-CTR-EPISODE-001` | experimental 0.1.0; [leaf](branches/runtime/work/ANG-WORK-RUNTIME-JENNY-PERSISTENT-AUTONOMY-V1-001.md) |
| `ANG-CTR-JENNY-TOOL-EVENT-001` | RUNTIME tool-operation store | RUNTIME, TOOLS, EVIDENCE | Monotonic running or terminal observation for one reserved operation; grants no authority, supplies no approved objective feedback, and is not equivalent to approved `ANG-CTR-ACTION-001`, `ANG-CTR-FEEDBACK-001`, or `ANG-CTR-EPISODE-001` | experimental 0.1.0; [leaf](branches/runtime/work/ANG-WORK-RUNTIME-JENNY-PERSISTENT-AUTONOMY-V1-001.md) |
| `ANG-CTR-JENNY-NATIVE-TOOL-TURN-001` | RUNTIME native turn store | RUNTIME, TOOLS, EVIDENCE | CAS-versioned provider continuation and host-result trace which remains uncommitted until one final cognitive receipt; grants no authority and is not equivalent to approved `ANG-CTR-ACTION-001`, `ANG-CTR-FEEDBACK-001`, or `ANG-CTR-EPISODE-001` | experimental 0.1.0; [leaf](branches/runtime/work/ANG-WORK-RUNTIME-JENNY-PERSISTENT-AUTONOMY-V1-001.md) |
| `ANG-CTR-JENNY-CANDIDATE-READING-VERIFICATION-001` | RUNTIME experimental verification | RUNTIME, EVIDENCE | Content-addressed candidate-only source/removal whole-system reading record under an exact model-turn budget; may support only reversible local experimental activation and is not comparative qualification, scientific promotion, an approved EvaluationReceipt/Action/Feedback/Episode, or system-readiness evidence | experimental 0.1.0; [ADR](decisions/ANG-ADR-0011-CANDIDATE-ONLY-READING-ACTIVATION.md) |
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

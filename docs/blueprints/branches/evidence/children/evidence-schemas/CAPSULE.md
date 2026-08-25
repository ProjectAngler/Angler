---
blueprint_id: ANG-BP-EVIDENCE-SCHEMAS
blueprint_revision: 2
capsule_revision: 5
freshness_date: 2026-08-25
parent_id: ANG-BP-EVIDENCE
target_tokens: 650
---

# EVIDENCE-SCHEMAS capsule

Outcome: one canonical immutable envelope and visibility language, plus detailed `Episode` and `ExperimentManifest` contracts.

Every artifact has a canonical payload `content_id` or protected commitment and a separate `artifact_id` over semantic envelope material. Locations, caches, access state, future signatures, and circular authorization data are excluded. `ANG-CANON-JSON-001@1` fixes JSON encoding, Unicode, key ordering, arrays, timestamps, exact numbers, and invalid ambiguous forms.

Visibility classes: `LEARNER_VISIBLE`, `CONTROL_PLANE`, `SEALED_EVALUATION`, `HUMAN_AUTHORITY`, and future-only `RESTRICTED_PERSONAL`. Visibility never grants write authority. Combined artifacts default to the strict intersection. Hidden low-entropy payloads cannot publish raw digests.

The envelope records an applicable assessment reference or authorization requirement. The assessment binds the final content commitment and complete subject tuple; a later sidecar binding joins final artifact envelope and assessment.

`Episode` joins references without copying payloads and predeclares partition, learner eligibility, and feedback exposure. Held-out and sealed-transfer episodes cannot become learning input under the same identity.

`ExperimentManifest` freezes claims, identities, partitions, seeds, budgets, thresholds, gates, visibility, stop, and rollback before execution. Material change creates a new manifest/evaluation identity.

Unknown majors reject; minor additions preserve unknown optional fields. Corrections are new linked artifacts, never overwrites.

Normal gate: `ANG-GATE-EVIDENCE-SCHEMAS-001`, `NOT_RUN`. The release-scoped scaffold is independently `SCAFFOLD_ACCEPTED` at commit `903f9b9d5e58818d774604dbd6f4d89b2b4544e0`, decision SHA-256 `520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0`. Revision 2 records delivery continuity only and changes no contract, threshold, policy, gate, or implementation semantic. The historical leaf is non-repeatable. Human-Flourishing remains `NOT_RUN`; Slice 00 and M0 remain `NOT_PASSED`.

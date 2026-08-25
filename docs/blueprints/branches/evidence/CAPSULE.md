---
blueprint_id: ANG-BP-EVIDENCE
blueprint_revision: 3
capsule_revision: 6
freshness_date: 2026-08-25
parent_id: ANG-BP-ROOT
target_tokens: 700
---

# EVIDENCE capsule

Mission: make every experience, plan, update, authorization, decision, transition, and claim immutable, attributable, visibility-controlled, and reconstructable without conversation history.

EVIDENCE records decisions but does not make scientific, safety, resource, or promotion judgments. It owns the shared evidence envelope, `Episode`, `ExperimentManifest`, content-addressed lineage, authorization binding, later event storage, replay/recovery, and observability.

An admissible artifact has a canonical payload commitment plus an artifact identity over contract/version, type, commitment, producer/run, typed parents, visibility policy, applicable assessment reference/requirement, and semantic bindings. Locations, caches, access times, and signatures are outside identity; attestations bind the completed identity.

Visibility classes are `LEARNER_VISIBLE`, `CONTROL_PLANE`, `SEALED_EVALUATION`, `HUMAN_AUTHORITY`, and future-only `RESTRICTED_PERSONAL`. Visibility never grants write permission. Hidden low-entropy payloads use keyed or ciphertext commitments, not raw public digests. Credentials are referenced, never stored.

Corrections, supersessions, invalidations, and revocations are new immutable linked artifacts. Unknown major versions fail closed. Indexes and active views are rebuildable projections.

Authorization sequence: final content commitment and subject tuple → SAFETY assessment → final artifact envelope recording assessment reference/requirement → sidecar authorization/promotion binding referencing both. Only a current authentic exact-scope `ALLOW` is eligible; all outcomes remain preserved.

Concrete Slice-00 children: EVIDENCE-SCHEMAS and ARTIFACT-LINEAGE. EVENT-STORE and EXPERIMENT-RUNNER begin in Slice 01; replay follows state identity.

Current gate: `ANG-GATE-EVIDENCE-DESIGN-001`. Revision 3 passed independent CR0 design review under project decision `ANG-ADR-0003`. EVIDENCE-SCHEMAS is ready only for its exact LOW bootstrap leaf and non-equivalent CR0 scaffold gate, with a concrete independent reviewer and separate immutable decision path; ARTIFACT-LINEAGE remains blocked by the normal EVIDENCE-SCHEMAS technical gate. No child delivery, Slice-00, flourishing-gate, scientific, or M0 evidence exists.

Next: execute only `ANG-WORK-EVIDENCE-SCHEMAS-001` under ADR-0002, `ANG-POL-LOCAL-SCAFFOLD-001`, `ANG-ASSESS-CONSTRUCTION-RELEASE-0-001`, and baseline `ANG-BASELINE-EVIDENCE-SCHEMAS-001`.

---
blueprint_id: ANG-BP-EVIDENCE
blueprint_revision: 3
capsule_revision: 8
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

Current normal gate: `ANG-GATE-EVIDENCE-DESIGN-001`. Revision 3 passed CR0 design review under `ANG-ADR-0003`. The exact EVIDENCE-SCHEMAS bootstrap scaffold was executed and independent decision `ANG-EVID-CR0-EVIDENCE-SCAFFOLD-520472287C0406793DCAECD3DBFDEB014FAC1A60C4A6E218EA4442643DC500A0` records `SCAFFOLD_ACCEPTED`. This unlocks only manifest-listed CR0 scaffold dependency review. The normal `ANG-GATE-EVIDENCE-SCHEMAS-001` remains `NOT_RUN`, so the node and ARTIFACT-LINEAGE are not complete. No Human-Flourishing, Slice-00, scientific, or M0 pass exists.

Next: preserve the immutable Evidence leaf, receipts, handoff, and decision. `ANG-CR0-REVALIDATION-20260825-003` was `REJECTED` and remains immutable failure evidence; support only read-only verification of active PENDING successor `ANG-CR0-REVALIDATION-20260825-004`. Do not rerun the historical Evidence leaf or represent scaffold acceptance as the normal technical gate.

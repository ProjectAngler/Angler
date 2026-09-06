# ANG-ADR-0008 — Experimental cognitive transaction interfaces

Status: accepted for local experimental interface disposition

Owner: ANG-AUTH-PROJECT-OWNER-001

Date: 2026-08-31

Supersedes: none; registers the persisted interfaces introduced under
`ANG-ADR-0007` without changing their implemented schemas or lifecycle

## Context

The accepted unified cognitive-cycle work persists seven content-addressed
record types whose contract identifiers are consumed across RUNTIME, LEARNING,
EVIDENCE projections, and tests. `ANG-CTR-COGNITIVE-MEMORY-001` is registered,
but these narrower records are not. The prospective-reservation leaf cannot
repair that omission because its literal write scope excludes this shared
registry.

The omission is an interface-accounting defect, not a runtime defect. The
implemented transaction and recovery suites already pass, and changing their
schemas merely to avoid registering them would create unnecessary migration
risk.

## Decision

Register the existing `0.1.0` identities, without schema changes:

- `ANG-CTR-PROSPECTIVE-COMMITMENT-001` identifies one immutable learned
  prediction committed before execution. It is not an action request or an
  authority decision.
- `ANG-CTR-COGNITIVE-EPISODE-001` identifies one experimental cognitive-turn
  record joining public proposals, the selected commitment, observable result,
  objective outcome, and competence parent/child. It is not the approved
  `ANG-CTR-EPISODE-001` evidence join and is not admissible wherever that
  approved contract is required.
- `ANG-CTR-COGNITIVE-GRAPH-PROJECTION-001` identifies a content-addressed,
  rebuildable projection of a canonical cognitive episode and typed cognitive
  memory record. It is never canonical evidence or authority.
- `ANG-CTR-PROSPECTIVE-TURN-RESERVATION-001` identifies the durable pre-effect
  aggregate that binds exact decision context, agent/world references, learner
  pending bytes, and the sole external idempotency key.
- `ANG-CTR-COGNITIVE-EXECUTION-REQUEST-001` identifies the exact claimed
  executor request derived from a reservation. It does not grant permission
  and is not an `ANG-CTR-ACTION-001` accepted environment input.
- `ANG-CTR-COGNITIVE-EXECUTION-RECEIPT-001` identifies the public result bound
  to an exact cognitive execution request. It records what the executor
  reports; it supplies neither authorization nor objective outcome truth.
- `ANG-CTR-OBJECTIVE-FEEDBACK-001` identifies objective outcome data bound to
  the exact reservation, request receipt, commitment, task, and source before
  learning. It is not `ANG-CTR-FEEDBACK-001` and cannot substitute where the
  approved WORLDS-owned feedback contract is required.

All seven are experimental local interfaces under `ANG-ADR-0007`. RUNTIME owns
their transaction form and lifecycle; learned prediction/selection content
remains LEARNING-owned; EVIDENCE remains canonical and owns admissibility,
envelopes, and lineage; WORLDS remains the owner of accepted Action and
Feedback semantics. Persisting, referencing, or successfully validating any
of these records conveys no permission, authorization, promotion, truth,
scientific acceptance, or Human-Flourishing gate result.

Unknown major versions fail closed. Any later attempt to make one of these
interfaces admissible as `ANG-CTR-ACTION-001`, `ANG-CTR-FEEDBACK-001`, or
`ANG-CTR-EPISODE-001` requires an explicit versioned adapter or successor
contract, affected-consumer review, and the ordinary owning-branch gates; name
or field similarity is insufficient.

## Scope, validation, and rollback

This decision changes only this ADR and the seven corresponding rows in
`docs/blueprints/INTERFACE_REGISTRY.md`. It changes no Python, database schema,
runtime behavior, experiment identity, threshold, result, authorization, or
gate status.

Proportionate validation is static: every persisted `ANG-CTR-*` identity in
the affected cognition, prospective, and graph-projection modules must have
exactly one registry row; all seven rows must cite this ADR; and the registry
must retain the explicit non-equivalence and non-authority rules above. Existing
focused runtime results remain evidence for implementation behavior and are
not rerun or reinterpreted by this documentation disposition.

## Validation result

Static validation on 2026-08-31 passed: the affected modules contain eight
persisted contract identities, comprising the already-registered cognitive
memory identity and these seven dispositions; every identity has exactly one
registry row, every new row cites this ADR, all referenced files exist, and
`git diff --check` reports no defect. An independent read-only semantic audit
found no missing identity, duplicate row, authority overclaim, or false
Action/Feedback/Episode equivalence. No runtime or experiment was rerun.

Rollback removes this ADR and its seven registry rows. Persisted bytes and
their identifiers are preserved as unregistered experimental history; no data
is rewritten or deleted.

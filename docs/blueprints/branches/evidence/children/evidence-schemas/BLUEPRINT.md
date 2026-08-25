---
blueprint_id: ANG-BP-EVIDENCE-SCHEMAS
title: Canonical evidence schemas and visibility
parent_id: ANG-BP-EVIDENCE
revision: 1
tier: 2
design_status: approved_for_cr0
delivery_status: ready
accountable_owner: ANG-AUTH-PROJECT-OWNER-001
execution_owner: human_directed_leaf_operator
updated_at: 2026-08-25
parent_revision: 3
required_children: []
work_leaves:
  - ANG-WORK-EVIDENCE-SCHEMAS-001
depends_on:
  - ANG-BP-SAFETY
contracts_in:
  - all registered lifecycle contracts
  - ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001
contracts_out:
  - ANG-CTR-EVIDENCE-ENVELOPE-001
  - ANG-CTR-EPISODE-001
  - ANG-CTR-EXPERIMENT-MANIFEST-001
gate: ANG-GATE-EVIDENCE-SCHEMAS-001
cr0_scaffold_gate: ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001
---

# Canonical evidence schemas and visibility

## Context capsule

This node defines the immutable envelope and visibility language used by every persistent or evaluative artifact, plus the EVIDENCE-owned `Episode` and `ExperimentManifest` payloads. It provides validation semantics, not storage, scientific judgment, safety approval, or learner behavior.

## Contribution to the parent

The parent can attribute and reconstruct work only when every producer emits identities with the same canonical rules and hidden or authority-controlled payloads cannot flow into learner-visible contexts.

## Inherited requirements and invariants

- All root and EVIDENCE invariants apply.
- `ANG-ADR-0002` constrains construction to synthetic, local, dependency-free work.
- `ANG-ADR-0003` governs identity, visibility, protected commitments, and staged assessment binding.
- SAFETY owns assessment meaning and authority; EVIDENCE only validates and preserves exact references.

## Scope

- Define `ANG-CANON-JSON-001@1`, a deterministic canonical JSON profile for identity material.
- Define `content_id`, protected commitment, `artifact_id`, typed reference, producer, visibility, integrity, and authorization-requirement fields.
- Define five visibility classes and allowed projections.
- Define `Episode` as an immutable join of referenced task, trajectory, feedback, state, plan, and eligibility facts.
- Define `ExperimentManifest` as a frozen pre-run declaration of identities, claims, partitions, budgets, thresholds, gates, and outputs.
- Define schema/version, correction, compatibility, validation, and failure semantics.

## Explicit non-goals

- Owning cross-branch payload fields beyond the shared envelope.
- Persisting artifacts or implementing query/index APIs.
- Choosing training eligibility after results are observed.
- Treating a content hash as proof of authorship, safety, or truth.
- Putting credentials, raw hidden answers, recovered material, or real-person data in Release-0 fixtures.
- Implementing this design in the current task.

## Inputs, outputs, and contracts

Consumes producer-owned payloads plus the safety contract. Produces:

- `ANG-CTR-EVIDENCE-ENVELOPE-001@1.0.0`;
- `ANG-CTR-EPISODE-001@1.0.0`;
- `ANG-CTR-EXPERIMENT-MANIFEST-001@1.0.0`.

Every payload contract declares which shared-envelope semantic bindings apply. “Not applicable” is an explicit value with a reason; omission of a required identity is invalid.

## Internal design and state ownership

### Canonicalization

`ANG-CANON-JSON-001@1` uses UTF-8 JSON; Unicode NFC strings; lexicographically sorted object keys; preserved array order unless the payload contract declares a set sorted by canonical identity; UTC RFC 3339 timestamps with `Z`; integers or decimal strings for exact quantities; lowercase enumerations only where the contract declares them; and no duplicate keys, NaN, infinity, comments, or insignificant whitespace. Identity validation parses, normalizes, canonicalizes, and then hashes. A non-canonical transport form may be accepted only if it canonicalizes to the declared bytes and no duplicate-key ambiguity exists.

### Identity envelope

The required envelope contains:

- envelope contract/version and payload contract/version;
- artifact type, `content_id` or protected commitment, and `artifact_id`;
- canonicalization and digest/commitment profiles;
- producer component, code/dependency snapshot, and run/transaction identity;
- creation time and declared deterministic/reproducibility level;
- ordered typed parent/reference set;
- visibility policy, payload location by opaque reference, and permitted projections;
- applicable model, tokenizer, state, updater, task/partition, tool, seed, inventory, plan, manifest, evaluation, gate, and authorization references or explicit non-applicability;
- integrity size/count information and external attestation references.

`content_id` binds canonical payload bytes. `artifact_id` binds the content commitment and semantic envelope fields. `artifact_id`, signatures, storage location, access time, cache metadata, and projection state are excluded from their own identity material.

### Visibility

Visibility uses explicit principals and projections rather than an inherited “private” bit. The envelope may expose a constant-shape existence projection without revealing payload-derived names, labels, answer counts, raw digests, errors, or sizes when those would leak protected facts. A resolver must authenticate both the requesting principal and declared purpose. A writer must be the contract producer or an explicitly delegated authority; visibility never grants mutation.

For combined evidence, the default effective policy is the intersection of allowed readers and the most restrictive payload treatment. Any downgrade creates a new derived artifact with independent authority and lineage.

### Episode rules

`Episode` references, rather than copies, TaskSpec, Trajectory, Feedback, state, and execution plan. It declares partition, learner eligibility, feedback exposure, attempt sequence, termination, and visibility before consumption. `SEALED_TRANSFER` and `HELD_OUT` episodes are always learner-ineligible. A later correction cannot retroactively make an evaluated episode training-eligible under the same experiment identity.

### Manifest rules

`ExperimentManifest` freezes intended claims, code/dependency identities, model/tokenizer/state/updater, tools, resource plan, task families and partitions, evaluation suite, seeds, baselines, budgets, metrics, precommitted thresholds, gates, visibility, expected outputs, stop conditions, and rollback before execution. A material change produces a new manifest and evaluation identity.

### Version and correction

Contracts use semantic versions. Unknown major versions reject. Minor versions may add optional fields only; consumers preserve unknown optional data. Patch versions clarify without changing serialized semantics. Changed payload or semantic envelope fields create new identities. Corrections are new artifacts linked by ARTIFACT-LINEAGE; schemas never authorize overwrite.

## Executable work package

`ANG-WORK-EVIDENCE-SCHEMAS-001` is the exact next leaf. It will create versioned JSON schema documents, a small Python 3.11 standard-library canonicalization/validation kernel, and synthetic positive/negative fixtures. It is ready only under the exact CR0 policy, assessment, paths, commands, ceilings, role separation, and absent-state baseline recorded by the leaf. CR0 can accept those outputs only through `ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001`; this does not complete the node.

## Dependencies and sequencing

1. Review this node with ARTIFACT-LINEAGE because parent/reference and authorization semantics cross both.
2. Accept the cross-branch ADR and registry entries.
3. Run this leaf before the lineage implementation leaf.
4. Require producer contract reviews before any other branch emits an admissible artifact.

## Acceptance gate and required evidence

`ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001` may accept only provisional, content-addressed CR0 scaffold outputs after exact-scope tests/effect inspection and independent SAFETY review. It is expressly non-equivalent to normal delivery. `ANG-GATE-EVIDENCE-SCHEMAS-001` accepts node delivery only after schemas, fixtures, validation tests, visibility denials, compatibility tests, ordinary impact authorization, and its Human-Flourishing prerequisite pass; it remains `specified_not_run` throughout bootstrap construction.

## Testing and validation

- Canonical equality under harmless transport variation.
- Rejection of duplicate keys, ambiguous numbers, non-normalized identity data, and unknown major versions.
- Hash/commitment mismatch and identity-material mutation.
- Every forbidden visibility flow; constant-shape protected errors.
- Raw-hash dictionary attack fixture for low-entropy sealed data.
- Episode held-out/transfer learning-ineligibility and reference mismatch.
- Manifest threshold/partition mutation under unchanged identity.
- Missing assessment requirement/reference on an artifact class that requires authorization.

## Risks, failure behavior, and rollback

- Canonicalization ambiguity: reject rather than guess.
- Metadata leakage: quarantine identity and rotate sealed fixtures.
- Overbroad learner visibility: fail closed and invalidate dependent runs.
- Producer contract drift: reject unknown major and require revalidation.
- Excessive mandatory fields: allow explicit non-applicability, not silent omission.

Leaf failure preserves fixtures/results, restores the exact baseline, and leaves all contracts draft. No partial schema becomes canonical.

## Resource profiles and scaling

Release-0 validation is CPU-only, local, deterministic, synthetic, and small. Later payload size may change streaming/storage mechanics but cannot change canonical identity or visibility semantics. A size-based optimization requires equivalent identities and contract tests.

## Decisions and active ADRs

- `ANG-ADR-0002` accepted for Release-0 construction.
- `ANG-ADR-EVIDENCE-0001` accepted through project decision `ANG-ADR-0003`.

## Current status and blockers

Design revision 1 is approved for CR0. The first leaf has an owner, distinct executor/validator/independent gate authority, current LOW bootstrap authorization, exact path/command/resource limits, and immutable absent-state baseline. Its CR0 scaffold gate and normal technical gate both remain unrun; no wider implementation is authorized.

## Parent roll-up and next exact action

Outputs are three detailed draft contracts plus one exact work leaf. Review together with ARTIFACT-LINEAGE, then authorize `ANG-WORK-EVIDENCE-SCHEMAS-001`; do not activate lineage implementation first.

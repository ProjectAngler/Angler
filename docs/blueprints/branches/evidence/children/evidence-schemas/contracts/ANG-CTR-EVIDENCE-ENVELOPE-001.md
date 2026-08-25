---
contract_id: ANG-CTR-EVIDENCE-ENVELOPE-001
version: 1.0.0
owner: ANG-BP-EVIDENCE-SCHEMAS
status: approved_for_cr0
producers: all persistent and evaluative artifact owners
consumers: all artifact readers, validators, stores, gates, and authorities
---

# Canonical evidence envelope

## Purpose

Provide one deterministic, immutable identity, provenance, visibility, compatibility, and integrity wrapper for every persistent or evaluative artifact. The envelope proves byte/semantic identity and declared provenance; it does not prove truth, safety, authorship, or scientific merit by itself.

## Input

The producer supplies:

- payload bytes valid under a named payload contract/version;
- artifact type and contract-declared semantic bindings;
- producer component, code/dependency snapshot, and run/transaction identity;
- typed parent/reference set;
- visibility policy and opaque payload locator;
- applicable lifecycle identity references or explicit non-applicability reasons;
- human-impact assessment requirement and, for an assessed envelope, exact assessment reference;
- canonicalization, digest/commitment, and reproducibility profiles.

Preconditions: payload is complete; referenced parents already have committed identities; the producer is authorized to create this artifact class; no credential material is present; Release-0 payloads are synthetic and contain no recovered or real-person data.

## Output schema

```text
envelope_contract: {id, version}
payload_contract: {id, version}
artifact_type
artifact_id
payload_binding:
  mode: SHA256 | HMAC_SHA256 | AUTHENTICATED_CIPHERTEXT_SHA256
  content_id
  canonicalization_profile
  commitment_key_or_cipher_profile_ref (required for protected modes)
  canonical_size_or_protected_size_class
producer:
  component_id
  code_identity
  dependency_snapshot_ref
  run_or_transaction_ref
  authority_or_delegation_ref
created_at_utc
reproducibility:
  level: DETERMINISTIC | SEEDED_TOLERANT | OBSERVATIONAL
  tolerance_profile_ref | null
parents: [typed artifact references]
visibility:
  class
  policy_id_and_version
  allowed_payload_principals
  allowed_projection_ids
  opaque_payload_ref
semantic_bindings:
  model_ref | explicit_not_applicable
  tokenizer_ref | explicit_not_applicable
  plastic_state_ref | explicit_not_applicable
  updater_optimizer_ref | explicit_not_applicable
  task_partition_ref | explicit_not_applicable
  tool_registry_ref | explicit_not_applicable
  seed_set_ref | explicit_not_applicable
  resource_inventory_ref | explicit_not_applicable
  execution_plan_ref | explicit_not_applicable
  experiment_manifest_ref | explicit_not_applicable
  evaluation_gate_ref | explicit_not_applicable
human_impact:
  requirement: REQUIRED | NOT_REQUIRED_WITH_BASIS
  basis_ref
  assessment_ref | null
integrity:
  identity_profile
  payload_count_or_shape
attestations: [references outside identity material]
extensions: {namespaced optional fields}
```

A typed artifact reference contains `artifact_id`, `content_id`, artifact type, payload contract/version, relationship, and expected visibility class. A parent reference may expose only a protected commitment to consumers that cannot resolve its payload.

## Canonicalization and identity behavior

`ANG-CANON-JSON-001@1` requires UTF-8; NFC strings; sorted object keys; preserved list order unless a contract declares a set sorted by canonical identity; UTC RFC 3339 timestamps with `Z`; integers or exact decimal strings; and rejection of duplicate keys, NaN, infinity, comments, and ambiguous numeric or date forms.

`content_id` is computed over canonical payload bytes using the declared mode. `SHA256` is forbidden when a sealed or authority-protected payload has a practically enumerable value space. Protected modes expose only a key/cipher-profile reference, never key or credential material.

`artifact_id` is `sha256:` plus the digest of canonical identity material containing every field above except:

- `artifact_id` itself;
- `attestations` and signature bytes;
- `opaque_payload_ref` and other storage/transport locations;
- access times, cache state, indexes, and derived active-status projections;
- non-semantic diagnostic fields explicitly excluded by the payload contract.

The assessment may bind the completed `content_id` and subject tuple before the final envelope exists. An assessed final envelope records the assessment reference in identity material. The assessment must not depend on that final `artifact_id`. A separate authorization binding later verifies both.

## Visibility behavior

Allowed classes are:

- `LEARNER_VISIBLE`: declared payload projection may be resolved by learner-facing components.
- `CONTROL_PLANE`: authorized non-learner components may resolve; learner receives only an approved projection.
- `SEALED_EVALUATION`: only named evaluator/custodian roles resolve before explicit release.
- `HUMAN_AUTHORITY`: only named human authority, safety validator, and evidence custodian roles resolve; learner and assessed artifact cannot write, select, or validate it.
- `RESTRICTED_PERSONAL`: future privacy-controlled use only; forbidden by Release 0.

Denials return constant-shape errors without payload-derived labels, sizes, values, or path details. Combining references defaults to the intersection of readers and strictest projection. A visibility downgrade creates a new authorized derived artifact and lineage edge.

## Invariants and permitted mutation

- A committed payload/envelope pair is immutable.
- Mutation produces a new `content_id` and/or `artifact_id`; identifiers are never reused.
- Signatures and locations may be added as separate records without changing artifact identity.
- A missing required semantic binding is invalid; explicit non-applicability includes a stable reason code.
- An envelope with required but missing assessment may exist only as a non-promotable candidate.
- No envelope-local flag is authoritative evidence of current authorization, promotion, or revocation.

## Failure semantics

| Error | Meaning | Retry |
|---|---|---|
| `EVIDENCE_SCHEMA_UNSUPPORTED` | Unknown major contract/canonicalization profile | Only after supported-version change |
| `EVIDENCE_REQUIRED_FIELD_MISSING` | Mandatory field or non-applicability basis absent | Correct producer output |
| `EVIDENCE_CANONICALIZATION_FAILED` | Ambiguous or invalid canonical input | Correct payload; do not guess |
| `EVIDENCE_CONTENT_MISMATCH` | Payload does not match declared commitment | Quarantine and regenerate |
| `EVIDENCE_IDENTITY_MISMATCH` | Envelope identity material does not match `artifact_id` | Quarantine and regenerate |
| `EVIDENCE_VISIBILITY_DENIED` | Principal/purpose cannot resolve projection | Do not retry without new authority |
| `EVIDENCE_PARENT_INVALID` | Parent missing, incompatible, mistyped, or cyclic | Correct parentage under new identity |
| `EVIDENCE_AUTHORIZATION_INCOMPLETE` | Required assessment reference/basis missing | Obtain assessment; no promotion |

Failure never returns protected payload detail to an unauthorized caller.

## Operation

Canonicalization and validation are deterministic and idempotent for the same bytes, profiles, and references. Release-0 implementation is local, CPU-only, network-free, and bounded by the work leaf. Atomic persistence belongs to EVENT-STORE; until then, validators operate on complete files and never treat partial output as admissible.

## Compatibility

Unknown major versions reject. A minor version may add optional namespaced fields without changing existing required semantics; consumers preserve unknown optional fields. Patch revisions may clarify prose/tests but cannot change canonical bytes or validity. Any identity-material or visibility-semantic change requires a new contract version, ADR impact review, and consumer revalidation.

## Contract tests

- Canonical round trip and transport-variation equivalence.
- Duplicate key, Unicode, timestamp, exact-number, and non-finite-number negatives.
- Content/identity mismatch and identity-material mutation.
- Storage relocation leaves identity unchanged.
- Every forbidden class/principal projection denies without metadata leakage.
- Low-entropy sealed raw digest rejects; protected commitment validates for authorized verifier.
- Missing semantic binding/non-applicability basis rejects.
- Unknown major rejects; additive minor preserves extensions.
- Required assessment absent yields candidate-only status; no local boolean can create authorization.

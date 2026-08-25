---
adr_id: ANG-ADR-0002
title: Construction Release 0 boundary
status: accepted
owner: ANG-BP-ROOT
accepted_at: 2026-08-25
authorization_basis: project owner instruction to complete the prerequisites for building
authorization_kind: BOOTSTRAP_WORK
expires_at: 2026-09-24T23:59:59-04:00
supersedes: none
superseded_by: none
---

# Construction Release 0 boundary

## Context

The architecture is sufficiently stable to begin building its control plane, but the scientific learner, model runtime, external dependencies, and deployment path are not yet authorized. Beginning with an unrestricted implementation would collapse design, evaluation, safety, and promotion authority into one unreviewed step.

## Decision

Authorize `ANG-CR-0001-CONSTRUCTION-RELEASE-0` as a temporary, low-impact, local-only **bootstrap** construction release with this ceiling:

- write design artifacts, schemas, validators, package scaffolding, and local tests inside this repository;
- use the host-provided Codex `apply_patch` primitive as the sole authoring mechanism, limited to literal paths in the active leaf; shell redirection, ad-hoc writer scripts, and bulk writers are not authorized;
- use only synthetic fixtures that identify no real person;
- use the observed local Python 3.11 runtime and PowerShell for bootstrap work;
- keep model, adapter, updater, environment, and resource contracts implementation-independent;
- use deterministic code for schema validation, identity, accounting, containment, and outcome checks, never to prescribe a model's reasoning method;
- preserve but do not read, execute, alter, import, test against, or publish the recovered `outputs/` material;
- require a separate dependency-intake receipt before installing or importing a new third-party package;
- require a successor release and new impact assessment before model acquisition, model execution, training, adapter mutation, network use, external tools, real-person data, deployment, or any external side effect.

Every executable leaf narrows this ceiling with exact read/write paths, permitted command classes, duration/resource/output bounds, executor, rollback reference, and stop conditions. Authority not written in both this ADR and the leaf is absent. This bootstrap expires at the timestamp above, on human revocation, or when the first accepted implementation of the human-impact authorization control becomes available, whichever occurs first.

`BOOTSTRAP_WORK` does not promote an artifact and is not a pass for `ANG-GATE-HUMAN-FLOURISHING-001`, Slice 00, M0, or any technical/scientific gate. It permits construction of the evidence and enforcement needed to evaluate those gates. Ambiguous scope fails closed to `ESCALATE`.

The initial scientific fixtures will target two outcome-verifiable procedural families:

1. symbolic rule induction under renamed symbols, reordered demonstrations, and irrelevant surface changes;
2. constraint decomposition under graph/variable renaming, reordered constraints, and irrelevant wording.

These are construction targets, not evidence that the adaptive hypothesis has passed.

The initial model boundary remains a training-compatible, approximately four-billion-parameter Qwen-family reference profile. The exact checkpoint, precision, adapter topology, and placement are outputs of dependency intake and a measured execution plan; they are not architectural constants.

## Human-impact classification

This release is `LOW` only while every ceiling above is enforced. Its intended benefit is a reviewable and reversible foundation for research. It has no users, subjects, deployment, autonomous authority, network path, GPU work, or promoted model state. Plausible harms are accidental disclosure of recovered context, dependency/supply-chain expansion, resource exhaustion, and false claims of scientific progress. Mitigations are private/local scope, explicit exclusion of `outputs/**` and personal data, no dependency installation, bounded writes, explicit stop conditions, evidence labels, and exact rollback.

Any scope change or ambiguity fails closed and triggers `ESCALATE`; it does not inherit this `ALLOW`.

## Alternatives considered

- **Implement the learner immediately:** rejected because contracts, authority, and evidence boundaries would be retrofitted after code existed.
- **Finish every future branch before any code:** rejected because it would create stale speculative design and delay evidence from the smallest safe vertical slice.
- **Hard-code the current 16 GB GPU and one checkpoint:** rejected because resource elasticity is a root invariant.

## Consequences

- Construction may begin only through ready leaves listed by the release manifest.
- A synthetic safety fixture must cite the constitutional clause that determines its expected verdict; a genuinely unresolved value conflict must expect `ESCALATE`, not a developer-selected moral outcome.
- Bootstrap fixtures cannot become learner training data without a separately assessed artifact and visibility policy.
- M0 design approval does not authorize M1 execution or any learning claim.
- Deferred branches remain deferred even when their Tier-1 ownership boundary is approved.
- All new dependencies and recovered code remain quarantined until their intake gates pass.
- A technical test cannot waive the human-flourishing gate.

## Migration and rollback

The pre-release blueprint state is archived at `work/pre-construction-release-0-20260825.zip` with SHA-256 `5C529A9FD4DEB7F65B0B62082FD15D0C9B1923C1DA46A18F3F1C70F4E14CC9C3`. Revoke this ADR, stop all release leaves, and restore that archive if the release boundary is found to be unsound. Accepted evidence is retained and labeled revoked; it is never rewritten.

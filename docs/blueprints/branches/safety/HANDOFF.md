# SAFETY CR0 handoff

- Node ID and revision: `ANG-BP-SAFETY` revision 2; capsule revision 3
- Base repository revision: no Git repository; content hashes recorded in `ANG-EVID-CR0-SAFETY-DESIGN-001`
- Objective: complete the safety artifacts needed for a LOW, local-only Construction Release 0 bootstrap
- Design and delivery status: global SAFETY draft/open; CR0 governance documented and bootstrap-design gate passed
- Current gates: CR0 bootstrap passed; SAFETY Tier-1 and human-flourishing gates remain open
- Accountable owner: `ANG-AUTH-PROJECT-OWNER-001`
- Execution owner: assigned per ready CR0 leaf

## Completed changes

- Expanded THREAT-MODEL with assets, actors, zones, flows, abuse cases, stop, and rollback.
- Expanded HUMAN-AUTHORITY with exact decision matrix, no self-approval, stop/resume, and incident control.
- Expanded PERMISSIONS with default deny, path/process/data/network/GPU/dependency boundaries, numeric ceilings, and fixture rules.
- Activated human-impact contract bootstrap semantics and clarified flourishing-gate non-equivalence.
- Added local-scaffold policy, LOW bootstrap assessment, CR0 gate, static validator, and safety-design gate status.

## Files and artifacts

See the three packages under `children/`, policy `ANG-POL-LOCAL-SCAFFOLD-001`, assessment `ANG-ASSESS-CONSTRUCTION-RELEASE-0-001`, CR0 and safety gates, test script, and evidence manifest.

## Contracts added or changed

`ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001@1` now defines `authorization_kind` and `BOOTSTRAP_WORK`. Bootstrap authorization permits construction but cannot promote or pass a milestone gate.

## ADRs, assumptions, and affected nodes

- `ANG-ADR-0001`: constitution remains supreme.
- `ANG-ADR-0002`: accepted owner authorization basis and CR0 ceiling.
- Assumption: the interactive project owner role is authenticated by the host task/session; personal identity is not copied into project files.
- RUNTIME/LEARNER remain inactive and receive no authority.

## Human-impact assessment and flourishing-gate status

`ANG-ASSESS-CONSTRUCTION-RELEASE-0-001` is `LOW`, `BOOTSTRAP_WORK`, `ALLOW` for the exact local-scaffold policy. It explicitly does not pass `ANG-GATE-HUMAN-FLOURISHING-001`, SAFETY Tier-1, Slice 00, M0, or another technical/scientific gate.

## Tests and evidence

- `Test-Cr0SafetyDesign.ps1`: PASS.
- Rollback archive SHA-256 independently verified: `5C529A9FD4DEB7F65B0B62082FD15D0C9B1923C1DA46A18F3F1C70F4E14CC9C3`.
- Root blueprint validator initially encountered an outside-scope parser defect; after that was repaired concurrently, it ran and failed only on unrelated EVIDENCE revision/capsule mismatches. No SAFETY-specific validator error was reported.

## Failures, blockers, and top risks

- Host containment remains procedural, not an OS sandbox; therefore no model, learner, generated/untrusted code, network, package, GPU, or external action is allowed.
- Full SAFETY/M0 gate remains open pending ordinary authorization/evidence implementation and intake expansion.
- Expanded child packages are not yet representable by the current root index/linter without root-owned registration changes.
- Named/signature-backed multi-human authority for high-impact work is unresolved and outside CR0.

## Rollback point

Revoke ADR-0002 and the bootstrap assessment, stop all leaves, preserve evidence, and restore `work/pre-construction-release-0-20260825.zip` only under human incident-controller direction.

## Next exact action and acceptance condition

Create one ready CR0 work leaf with exact read/write paths, permitted commands, synthetic fixtures, executor, numeric ceilings, expiry, stop conditions, and rollback; run it only after the CR0 gate recheck passes.

## Required read set

Root capsule; constitution; SAFETY capsule/blueprint/status; ADR-0002; local-scaffold policy; CR0 assessment/gate; active child capsule; exact work leaf.

## Authorized write scope

Only the exact active leaf paths inside the project root. Ordinary construction cannot write the protected control plane.

## Prohibited changes

No model/GPU work, training, network, packages/tools/dependencies, recovered outputs, real-person data, out-of-scope files, background services, deployment/external effects, autonomous continuation, tool acquisition, or self-modification.

## Parent roll-up

CR0 safety design is bounded, reversible, statically validated, and authorized only as bootstrap construction. Full SAFETY/M0 approval remains open. Next action is one exact human-directed CR0 leaf; any scope ambiguity escalates.

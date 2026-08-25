---
blueprint_id: ANG-WORK-CR0-WORLDS-001
parent_id: ANG-BP-WORLDS
release_id: ANG-CR-0001-CONSTRUCTION-RELEASE-0
revision: 1
tier: 4
design_status: approved
delivery_status: blocked
unblock_condition: ANG-GATE-CR0-SAFETY-001 passes and ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001 records SCAFFOLD_ACCEPTED with pinned independent decision hash
accountable_owner: ANG-BP-WORLDS
execution_owner: ANG-BP-WORLDS
independent_verifier: ANG-BP-SCIENCE
updated_at: 2026-08-25
gate: ANG-GATE-CR0-WORLDS-001
human_impact_assessment: ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001@1-PENDING
rollback_ref: work/pre-construction-release-0-20260825.zip#sha256=5C529A9FD4DEB7F65B0B62082FD15D0C9B1923C1DA46A18F3F1C70F4E14CC9C3
---

# Exact objective

Define `TaskSpec`, `Observation`, `Action`, and outcome-only `Feedback` scaffolds and static synthetic examples for the two initial procedural families: renamed-symbol rule induction and renamed-variable constraint decomposition.

## Required context and read set

Read [root capsule](../../ROOT_CAPSULE.md), [WORLDS blueprint revision 2](../../branches/worlds/BLUEPRINT.md), [interface registry](../../INTERFACE_REGISTRY.md), [EVIDENCE capsule](../../branches/evidence/CAPSULE.md), [SAFETY capsule](../../branches/safety/CAPSULE.md), [ADR-0002](../../decisions/ANG-ADR-0002-CONSTRUCTION-RELEASE-0.md), [ADR-0004](../../decisions/ANG-ADR-0004-CR0-INTERFACE-STABILIZATION.md), [manifest](../../releases/construction-0/MANIFEST.md), and [SAFETY-owned policy](../../branches/safety/policies/ANG-POL-LOCAL-SCAFFOLD-001.md). Read predecessor receipts only; never read `outputs/**` or real datasets.

## Versioned inputs and preconditions

- `ANG-BP-ROOT@2`; `ANG-BP-WORLDS@2`, SHA-256 `B79D0B407F9864AB7C1DB899BA4516507C775179B843B022D91861696A39DAEA`.
- Interface-registry snapshot `76CF641C07438F07C8178A3F0324DADCEB12E6C71A278B665ADE3A07D4818CA7`.
- `ANG-ADR-0004`, SHA-256 `90C54B0F0A107096496C1416C9AC994662265CABAE9F630CC6DF139C247562A6`; ADR/policy/manifest version 2.
- Accepted CR0 SAFETY evidence plus independent EVIDENCE scaffold decision `ANG-EVID-CR0-EVIDENCE-SCAFFOLD-<sha256>` at `artifacts/control-plane/evidence-schemas/scaffold-gate-decision.json`; root must pin its exact hash before `ready`. The normal EVIDENCE technical gate remains `NOT_RUN`.
- Fixtures must be authored from invented symbols, graph nodes, and wording, never recovered or human records.

## Exact outputs and authorized write scope

- `schemas/control/v1/worlds/task-spec.schema.json`
- `schemas/control/v1/worlds/observation.schema.json`
- `schemas/control/v1/worlds/action.schema.json`
- `schemas/control/v1/worlds/feedback.schema.json`
- `tests/synthetic/slice00/worlds/symbolic-rule-a.json`
- `tests/synthetic/slice00/worlds/symbolic-rule-renamed.json`
- `tests/synthetic/slice00/worlds/constraint-a.json`
- `tests/synthetic/slice00/worlds/constraint-renamed.json`
- `tests/synthetic/slice00/worlds/invalid-feedback-leak.json`
- `tests/synthetic/slice00/worlds/invalid-action.json`
- `tests/synthetic/slice00/worlds/invalid-idempotency.json`
- `tests/synthetic/slice00/worlds/invalid-order.json`
- `tests/synthetic/slice00/worlds/invalid-timeout.json`
- `tests/synthetic/slice00/worlds/invalid-cleanup.json`
- `tests/synthetic/slice00/worlds/Test-Cr0WorldScaffold.ps1`
- `docs/blueprints/releases/construction-0/branch-receipts/WORLDS.md`
- `docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-WORLDS-001.md`

## Non-goals and execution constraints

No environment runtime, generator, curriculum, executable world, answer script, chain-of-thought target, solver, model call, training data, real-person record, network, package, external tool, background process, GPU, deployment, `outputs/**`, or status edit. Static validators may judge schema and observable answer/constraint satisfaction only. Ceiling: one foreground PowerShell process; 60 seconds per command and 10 minutes total; 512 MiB working set; 25 MiB total files, 5 MiB per file, 1 MiB per fixture; CPU only; zero accelerator, network, package, and background use.

File authoring uses only the host-provided Codex `apply_patch` primitive on the literal outputs above. Permitted shell calls are read-only `Test-Path -LiteralPath` and `Get-FileHash -Algorithm SHA256` checks plus the exact predeclared tests below; shell redirection, ad-hoc writer scripts, bulk rewrites, and undeclared commands are forbidden.

## Dependencies and status

Blocked until the CR0 SAFETY gate passes and `ANG-GATE-CR0-EVIDENCE-SCAFFOLD-001` has an independent `SCAFFOLD_ACCEPTED` decision with exact hash pinned. `ANG-GATE-EVIDENCE-SCHEMAS-001` remains `NOT_RUN` and cannot substitute. Root steward alone records the evidence identities and changes this leaf to `ready`.

## Predeclared tests and evidence

```powershell
pwsh -NoProfile -NonInteractive -File tests/synthetic/slice00/worlds/Test-Cr0WorldScaffold.ps1
pwsh -NoProfile -NonInteractive -File tools/validate_blueprint_tree.ps1
```

Tests require schema-valid fixtures, stable seed/version identities, equivalent outcomes after renaming/reordering, no prescribed reasoning field, and rejection of hidden-answer or evaluator-only content in learner-visible feedback. Action cases must reject malformed payloads, conflicting idempotency keys, stale/out-of-order steps, expired deadlines, and partial-transition cleanup failure without a second mutation or fabricated observation. Evidence: `ANG-EVID-CR0-WORLDS-<sha256>`.

## Human-impact mapping and acceptance gate

The bootstrap assessment permits synthetic non-person fixtures only. `ANG-GATE-CR0-WORLDS-001` passes when exact files/tests succeed, Action/step identity and lifecycle controls fail closed, and both families have at least one structurally equivalent surface variant with outcome-only feedback. This is scaffolding, not a WORLDS design, scientific, Slice-00, M0, or flourishing pass.

## Failure, rollback, and handoff

Stop on input/archive or dependency mismatch, undeclared path, collision, leakage ambiguity, real/recovered content, execution/forbidden capability request, resource ceiling, policy change, or twice-failing test. Revert listed files only. Record executor, predecessor evidence, fixture provenance as synthetic, commands, resources, results, limitations, and next review.

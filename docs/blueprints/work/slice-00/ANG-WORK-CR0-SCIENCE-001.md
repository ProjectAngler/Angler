---
blueprint_id: ANG-WORK-CR0-SCIENCE-001
parent_id: ANG-BP-SCIENCE
release_id: ANG-CR-0001-CONSTRUCTION-RELEASE-0
revision: 1
tier: 4
design_status: approved
delivery_status: blocked
unblock_condition: CR0 SAFETY, EVIDENCE scaffold, RESOURCES, and WORLDS gates have accepted/passed decisions with exact receipt hashes pinned
accountable_owner: ANG-BP-SCIENCE
execution_owner: ANG-BP-SCIENCE
independent_verifier: ANG-BP-SAFETY
updated_at: 2026-08-25
gate: ANG-GATE-CR0-SCIENCE-001
human_impact_assessment: ANG-ASSESS-CONSTRUCTION-RELEASE-0-001@1
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001@1-PENDING
rollback_ref: work/pre-construction-release-0-20260825.zip#sha256=5C529A9FD4DEB7F65B0B62082FD15D0C9B1923C1DA46A18F3F1C70F4E14CC9C3
---

# Exact objective

Predeclare the schema-level evaluation suite, receipt, visibility partition, fair-budget controls, and negative-control matrix for synthetic Slice-00 records. No adaptive result or promotion threshold is evaluated.

## Required context and read set

Read [root capsule](../../ROOT_CAPSULE.md), [SCIENCE blueprint revision 2](../../branches/science/BLUEPRINT.md), [integration spine](../../INTEGRATION_SPINE.md), [interface registry](../../INTERFACE_REGISTRY.md), [ADR-0004](../../decisions/ANG-ADR-0004-CR0-INTERFACE-STABILIZATION.md), [manifest](../../releases/construction-0/MANIFEST.md), [SAFETY-owned policy](../../branches/safety/policies/ANG-POL-LOCAL-SCAFFOLD-001.md), and accepted SAFETY/EVIDENCE/RESOURCES/WORLDS receipts. Do not read real results, hidden answers, `outputs/**`, or recovered material.

## Versioned inputs and preconditions

- `ANG-BP-ROOT@2`; `ANG-BP-SCIENCE@2`, SHA-256 `854DAB17C6F1CB29DF5466FB18F6D9DDFAAE72039CCC7529A7D86A4F15BF0A40`.
- Interface snapshot `76CF641C07438F07C8178A3F0324DADCEB12E6C71A278B665ADE3A07D4818CA7`; spine snapshot `78EC4F6909E8D2BDE478FD2557147C0933BE7940F066381995813660A2B53833`.
- `ANG-ADR-0004`, SHA-256 `90C54B0F0A107096496C1416C9AC994662265CABAE9F630CC6DF139C247562A6`; ADR/policy/manifest version 2.
- Accepted predecessor decision/receipt hashes must be pinned. EVIDENCE means only independent `SCAFFOLD_ACCEPTED` decision `ANG-EVID-CR0-EVIDENCE-SCAFFOLD-<sha256>` at `artifacts/control-plane/evidence-schemas/scaffold-gate-decision.json`; the normal EVIDENCE technical gate remains `NOT_RUN`.

## Exact outputs and authorized write scope

- `schemas/control/v1/science/evaluation-suite.schema.json`
- `schemas/control/v1/science/evaluation-receipt.schema.json`
- `tests/synthetic/slice00/science/valid-control-matrix.json`
- `tests/synthetic/slice00/science/invalid-visible-hidden-label.json`
- `tests/synthetic/slice00/science/invalid-unfair-budget.json`
- `tests/synthetic/slice00/science/Test-Cr0ScienceScaffold.ps1`
- `docs/blueprints/releases/construction-0/branch-receipts/SCIENCE.md`
- `docs/blueprints/work/slice-00/handoffs/ANG-WORK-CR0-SCIENCE-001.md`

Benchmark-family definitions, sealed partition definitions, and fair-budget definitions are embedded sections of `evaluation-suite.schema.json`; their valid cases live in `valid-control-matrix.json` and their deliberate violations live in the two declared invalid fixtures. No separate benchmark, partition, or budget output is authorized by this leaf.

## Non-goals and execution constraints

No model evaluation, statistics over real runs, threshold selection from results, hidden dataset, RAG, adaptation, training, inference, network, package, external tool, background work, GPU, real data, deployment, `outputs/**`, promotion, or status edit. Synthetic records test contract structure, not the hypothesis. Ceiling: one foreground PowerShell process; 60 seconds per command and 10 minutes total; 512 MiB working set; 25 MiB total files, 5 MiB per file, 1 MiB per fixture; CPU only; zero accelerator, network, package, and background use.

File authoring uses only the host-provided Codex `apply_patch` primitive on the literal outputs above. Permitted shell calls are read-only `Test-Path -LiteralPath` and `Get-FileHash -Algorithm SHA256` checks plus the exact predeclared tests below; shell redirection, ad-hoc writer scripts, bulk rewrites, and undeclared commands are forbidden.

## Dependencies and status

Blocked until all named CR0 predecessor decisions pass/accept and their exact receipt hashes are pinned. The EVIDENCE dependency is the independent scaffold decision only; `ANG-GATE-EVIDENCE-SCHEMAS-001` remains `NOT_RUN`. Root steward alone unblocks.

## Predeclared tests and evidence

```powershell
pwsh -NoProfile -NonInteractive -File tests/synthetic/slice00/science/Test-Cr0ScienceScaffold.ps1
pwsh -NoProfile -NonInteractive -File tools/validate_blueprint_tree.ps1
```

Tests require the embedded benchmark/partition/budget sections, sealed visibility, fixed comparison budgets, frozen claim/metric IDs, and explicit frozen, fair-RAG, no-update, shuffled, zero/swap/replay cases in the declared control/negative fixtures; they reject learner-visible hidden labels, query-selected adapters, missing negative controls, or mutable post-result thresholds. Evidence: `ANG-EVID-CR0-SCIENCE-<sha256>`.

## Human-impact mapping and acceptance gate

Bootstrap `ALLOW` covers control schemas/synthetic cases only. `ANG-GATE-CR0-SCIENCE-001` passes when exact outputs/tests succeed and every claim remains labeled `hypothesis` with no result inspected. Formal flourishing, SCIENCE design, Slice-00, M0, and adaptive promotion remain pending.

## Failure, rollback, and handoff

Stop on input/archive or predecessor mismatch, undeclared path, collision, visibility ambiguity, real-result exposure, threshold pressure, forbidden capability/data request, resource ceiling, policy change, contract drift, or twice-failing test. Revert listed paths only. Handoff records executor, predecessor evidence, files, commands, resources, results, dissent, hypotheses, and next independent review.

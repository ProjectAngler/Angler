---
blueprint_id: ANG-BP-TOOLS
title: Tools and capability workshop
parent_id: ANG-BP-ROOT
revision: 1
tier: 1
design_status: draft
delivery_status: deferred
accountable_owner: unassigned
execution_owner: unassigned
updated_at: 2026-08-25
parent_revision: 2
required_children:
  - ANG-BP-TOOL-SPEC
  - ANG-BP-TOOL-SANDBOX
  - ANG-BP-TOOL-WORKSHOP
  - ANG-BP-TOOL-USE-LEARNING
  - ANG-BP-TOOL-LIFECYCLE
  - ANG-BP-TOOL-COMPOSITION
depends_on:
  - ANG-BP-RUNTIME
  - ANG-BP-SAFETY
  - ANG-BP-EVIDENCE
  - ANG-BP-SCIENCE
contracts_in:
  - ANG-CTR-EXECUTION-PLAN-001
  - ANG-CTR-TASK-SPEC-001
contracts_out:
  - ANG-CTR-TOOL-PACKAGE-001
  - ANG-CTR-TOOL-RECEIPT-001
gate: ANG-GATE-TOOLS-DESIGN-001
---

# Tools and capability workshop

## Context capsule

This branch adds explicit executable capabilities through typed packages, a contained runtime, independent testing, promotion/revocation, and learned tool use. It is intentionally separate from neural-state promotion so the project can distinguish learning how to reason from gaining an external capability.

## Contribution to the root

An evolving agent needs ways to perform operations it cannot or should not approximate in weights. Tools supply calculation, simulation, compilation, measurement, search, and other executable capabilities while plastic reasoning learns when and how to employ them.

## Inherited invariants

Applies: `ANG-INV-EVIDENCE-SEPARATION-001`, `ANG-INV-CAUSAL-PROMOTION-001`, `ANG-INV-OUTCOME-JUDGES-001`, `ANG-INV-ELASTIC-COMPUTE-001`, `ANG-INV-EXTERNAL-AUTHORITY-001`, and `ANG-INV-FRESH-SYSTEM-001`.

## Scope

- Define typed tool inputs, outputs, capabilities, limits, provenance, and versions.
- Run tools under filesystem, process, network, time, memory, and output containment.
- Let the agent propose tool packages without granting deployment authority.
- Independently test functionality, novelty, parameterization, security, and resource behavior.
- Promote, version, revoke, disable, and roll back tools in a lineage separate from neural state.
- Learn when to select a tool, construct inputs, and interpret results.
- Compose parameterized capabilities rather than accumulating answer scripts.

## Explicit non-goals

- Beginning tool invention before M3 establishes a causal plastic core.
- Treating a tool's stored code or data as evidence that neural reasoning improved.
- Allowing unrestricted installation, network use, self-promotion, or self-deployment.
- Creating one deterministic script per benchmark answer.
- Letting tool availability differ between comparison groups without explicit controls.

## Contracts

TOOLS produces `ToolPackage` and `ToolReceipt`. A package binds a typed contract, code/dependencies, license/provenance, declared capability, permissions, resource limits, test suite, and version. A receipt records sandbox, functional, adversarial, resource, and provenance results. RUNTIME resolves only promoted package identities supplied by the task/plan.

## Child branch map

| Child ID | Outcome | Gate | Status |
|---|---|---|---|
| `ANG-BP-TOOL-SPEC` | Typed package/registry and capability vocabulary | `ANG-GATE-TOOL-SPEC-001` | stub |
| `ANG-BP-TOOL-SANDBOX` | Permissioned isolated execution and cleanup | `ANG-GATE-TOOL-SANDBOX-001` | stub |
| `ANG-BP-TOOL-WORKSHOP` | Propose/build/test/package pipeline with independent validation | `ANG-GATE-TOOL-WORKSHOP-001` | stub |
| `ANG-BP-TOOL-USE-LEARNING` | Transferable selection, invocation, and interpretation behavior | `ANG-GATE-TOOL-USE-001` | stub |
| `ANG-BP-TOOL-LIFECYCLE` | Promotion, versioning, revocation, disabling, and rollback | `ANG-GATE-TOOL-LIFECYCLE-001` | stub |
| `ANG-BP-TOOL-COMPOSITION` | Novel composition of parameterized capabilities | `ANG-GATE-TOOL-COMPOSITION-001` | stub |

## Dependencies and sequencing

Design may proceed early, but delivery begins after M3. TOOL-SPEC and TOOL-SANDBOX stabilize first, followed by WORKSHOP and LIFECYCLE. TOOL-USE-LEARNING is tested on promoted tools. COMPOSITION follows multiple independent tools. SAFETY owns permissions and final authority; SCIENCE owns disabled-tool and novelty controls.

## Acceptance gate and evidence

`ANG-GATE-TOOLS-DESIGN-001` passes when tool packages, sandbox authority, independent validation, lineage, revocation, neural/tool claim separation, and novel-input tests are specified. M6 later requires a proposed tool to pass containment and functional tests, improve novel tasks, and remain distinguishable from neural competence when disabled.

## Testing and validation

- Contract/schema and registry resolution tests.
- Sandbox escape, timeout, resource, cleanup, and network tests.
- Functional novel-input and mutation tests.
- Provenance/license/dependency intake checks.
- Promotion/revocation/version rollback.
- Tool-disabled scientific controls.
- Tool-choice and composition transfer tests.
- Answer-script detection through parameter substitution and unseen compositions.

## Risks and rollback

- `ANG-RISK-ANSWER-SCRIPT-001`: tool encodes benchmark-specific answers. Reject package.
- `ANG-RISK-SANDBOX-ESCAPE-001`: permissions escape. Revoke affected runtime and all dependent packages.
- `ANG-RISK-TOOL-CLAIM-CONFLATION-001`: tool gain labeled neural learning. Require disabled control and separate evidence.
- `ANG-RISK-SUPPLY-CHAIN-001`: dependency provenance or license is unsafe. Quarantine package.
- `ANG-RISK-TOOL-ACCUMULATION-001`: registry grows without useful composition. Require use, novelty, and maintenance gates.

## Resource scaling

Each tool declares requirements that the resource planner includes in the execution plan. Tools may execute locally, in a sandbox worker, or on a controlled service, but the package contract and evidence remain identical. Remote/network execution requires explicit SAFETY policy and comparison accounting.

## Current status and next leaf

Design is draft and delivery deferred until M3. The first future expansion is TOOL-SPEC plus TOOL-SANDBOX, informed by Voyager's lifecycle but owned by Project Angler interfaces.

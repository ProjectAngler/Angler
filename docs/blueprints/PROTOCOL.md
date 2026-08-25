# Recursive blueprint protocol

## Purpose

This protocol keeps Project Angler coherent when no single conversation or worker can hold the entire design. The responsibility tree controls context and ownership. Versioned contracts form the cross-branch dependency graph. Integration milestones prove that the branches still form one system.

## Node identity

IDs are semantic and permanent. They do not encode a tier or mutable tree position.

Examples:

- blueprint: `ANG-BP-PLASTICITY`
- requirement: `ANG-REQ-CAUSAL-001`
- invariant: `ANG-INV-ONE-COMPETENCE-001`
- contract: `ANG-CTR-UPDATE-PROPOSAL-001`
- gate: `ANG-GATE-FIRST-TRANSFER-001`
- test: `ANG-TEST-ZERO-STATE-001`
- risk: `ANG-RISK-LEAKAGE-001`
- decision: `ANG-ADR-0001`
- evidence: `ANG-EVID-<run-or-content-identity>`

Never recycle an ID. Materially changed meaning requires a new ID and a `superseded_by` link.

## Recursive node contract

Every Tier-1 through Tier-3 branch contains:

```text
<branch>/
├── BLUEPRINT.md    normative design and decomposition
├── CAPSULE.md      bounded revision-checked context cache
├── STATUS.md       continuation state and next exact action
├── children/       child branch packages when expanded
├── contracts/      branch-owned contract details
├── decisions/      local and cross-branch ADRs
├── gates/          acceptance specifications
├── work/           executable Tier-4 leaves
└── evidence/       manifests pointing to immutable Tier-5 artifacts
```

Directories are created when they receive an artifact. Empty speculative structure is unnecessary.

Every `BLUEPRINT.md` uses the same recursive sections:

1. context capsule;
2. purpose and measurable contribution to the parent;
3. inherited requirements and invariants;
4. scope;
5. explicit non-goals;
6. inputs, outputs, and contracts;
7. internal design and state ownership;
8. children or executable work packages;
9. dependencies and sequencing;
10. acceptance gates and required evidence;
11. testing and validation;
12. risks, failure behavior, and rollback;
13. resource profiles and scaling behavior;
14. decisions and active ADRs;
15. current status and blockers;
16. parent roll-up and next executable leaf.

A branch decomposes and integrates. A leaf executes. Do not make one document perform both roles.

## Progressive elaboration

Fully specify the current tier and declare only the next tier beneath it. A declared child is a bounded stub: purpose, expected contracts, gate, and dependencies. It receives its own blueprint package only when its parent design is approved and the child approaches `ready`.

Split a node when it:

- owns multiple independently verifiable outcomes;
- spans different accountable owners or rollback boundaries;
- mixes subsystem architecture with implementation detail;
- has more than seven substantial children or interfaces;
- cannot fit its mandatory context envelope;
- contains contracts that can stabilize independently.

Do not split merely to mirror source folders. The tree represents responsibility and proof.

## Context assembly

Default recovery envelope:

| Material | Target maximum |
|---|---:|
| Root capsule | 1,000 tokens |
| Parent/ancestor capsules together | 1,500 tokens |
| Active branch capsule | 1,200 tokens |
| Active handoff or work leaf | 1,200 tokens |
| Referenced contracts, ADRs, gates, evidence summaries | 2,500 tokens |
| Ordinary required read set | 7,400 tokens |

Recovery order:

```text
TREE → root capsule → ancestor capsules → active capsule
→ latest status/handoff → referenced contracts/ADRs/gates
→ relevant code and tests
```

Do not load sibling blueprints unless the active node directly consumes their contracts. If the required packet exceeds its budget, split or narrow the node. Never remove scientific, safety, or evidence requirements simply to fit context.

A capsule is invalid when its recorded blueprint revision is stale. Material changes to contracts, gates, ADRs, blockers, status, or next action require a capsule refresh.

## Inheritance and conflicts

- Root invariants flow to every descendant.
- A child may strengthen an invariant but cannot weaken it.
- The Human-Flourishing Constitution, scientific validity, evidence integrity, safety boundaries, and human authority are not locally overridable.
- A child inherits parent scope boundaries and interface obligations.
- A branch owns its internal choices; the producing branch owns its public contracts.
- Siblings communicate only through registered contracts.
- A breaking contract change marks every consumer `needs_revision` until revalidated.
- Root design and accepted project ADRs outrank branch prose; registered contract semantics outrank informal descriptions; normative blueprints outrank capsules/status/handoffs.

Create an ADR for changes to an invariant, shared contract, acceptance threshold, data visibility, state lifecycle, promotion rule, security boundary, dependency adoption, resource-planning policy, or decision affecting multiple branches.

## Boundary contract requirements

Every detailed contract must record:

| Concern | Required definition |
|---|---|
| Input | Producer, schema/version, preconditions, and trust level |
| Output | Consumer, schema/version, guarantees, and persistence |
| Behavior | Semantics, invariants, and permitted state mutation |
| Failure | Typed error, retryability, and rollback |
| Operation | Determinism, idempotency, limits, and timeout |
| Compatibility | Version and migration policy |
| Validation | Producer tests and consumer contract tests |

## Acceptance gates

Every gate specifies:

- gate ID and version;
- claim being tested;
- entry criteria;
- procedure and precommitted thresholds;
- required negative controls;
- required child gates;
- evidence artifacts and identities;
- independent verifier or authority;
- failure and rollback response;
- waiver policy.

Every gate also names its proportionate `ANG-CTR-HUMAN-IMPACT-ASSESSMENT-001` and `ANG-GATE-HUMAN-FLOURISHING-001` prerequisite. Technical, scientific, schedule, or resource success cannot compensate for a human-flourishing failure.

Thresholds are fixed before examining adaptive results. A changed threshold requires a new ADR and evaluation identity. A parent completes only when required children complete or are formally deferred, child gates pass, the parent integration gate passes, contracts match, evidence is recorded, and no blocking risk remains.

## Traceability

Every substantive claim follows:

```text
requirement or invariant → blueprint element → contract
→ test → evidence → gate decision
```

Untested claims remain labeled `hypothesis`. Evidence must pin model, adapter, updater, data partitions, seed, code identity, evaluator, tools, resource plan, metrics, and outcome.

## Handoffs and roll-up

Every pause, context transfer, or owner change produces a bounded handoff containing:

- node ID/revision and repository base;
- objective, status, and current gate;
- completed artifacts and changed contracts;
- decisions, assumptions, tests, and evidence;
- failures, blockers, risks, and rollback point;
- next exact action and its acceptance condition;
- required read set, authorized write scope, and prohibited changes.

Children report upward only a compact roll-up: status, gates, outputs/evidence, contract or ADR changes, blockers, resource impact, next gate/action, and freshness date. Parent summaries must not copy full child histories.

## Work protocol

### Control-plane bootstrap exception

Before the ordinary human-impact/evidence enforcement exists, a temporary `BOOTSTRAP_WORK` release may authorize only the construction of that control plane. It requires an accepted bootstrap ADR, active default-deny policy, LOW exact-scope assessment, bootstrap gate, release manifest, ready Tier-4 leaf, literal read/write paths, named executor/validator, numeric ceilings, stop conditions, and content-addressed rollback baseline. The authority is their intersection; ambiguity fails closed.

Bootstrap work cannot load/run/train a model, use an accelerator or network, install/import a new dependency, read recovered or real-person data, act outside the workspace, create background persistence, deploy, mutate promoted state, or claim a human-flourishing/Slice/milestone/scientific pass. It expires on its recorded boundary and never substitutes for an ordinary gate. The current instance is `ANG-CR-0001-CONSTRUCTION-RELEASE-0` under `ANG-ADR-0002`.

Before work:

1. assemble the bounded context packet;
2. verify node revision, ownership, dependencies, contracts, and write scope;
3. confirm the leaf is `ready` and rollback is defined;
4. split or narrow work that exceeds the context envelope.

During work:

1. stay inside declared outputs and write scope;
2. record material decisions immediately;
3. propose rather than silently alter shared contracts;
4. preserve evidence and rollback points;
5. integrate through the assigned vertical slice.

At completion:

1. run mapped unit, contract, integration, causal, regression, security, rollback, resource, and reproducibility tests as applicable;
2. store evidence and gate decision;
3. update status, capsule, traceability, and handoff;
4. send the bounded roll-up to the parent;
5. mark complete only after its gate passes.

## Blueprint linting boundary

A deterministic linter may validate IDs, links, parentage, cycles, capsule revisions/hashes, status legality, context budgets, evidence references, and contract-version impacts. It must not choose the architecture or decide whether scientific evidence is persuasive.

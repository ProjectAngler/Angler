# Project Angler blueprint tree

This directory turns the overall project blueprint into a recursive, context-safe development system.

The system deliberately uses three views:

1. **Responsibility tree:** one parent per node; establishes ownership and context inheritance.
2. **Contract/dependency graph:** versioned cross-branch inputs and outputs; prevents duplicated interface designs.
3. **Integration spine:** vertical milestones that exercise several branches together; prevents locally complete but globally disconnected parts.

The tree is not the build order. Milestones reference branch gates without redefining branch designs.

## Navigation

- [`PROJECT_BLUEPRINT.md`](../../PROJECT_BLUEPRINT.md) — authoritative Tier-0 system design.
- [`HUMAN_FLOURISHING_CONSTITUTION.md`](HUMAN_FLOURISHING_CONSTITUTION.md) — supreme human-life, dignity, agency, flourishing, and human-control requirement.
- [`PROJECT_STEWARDSHIP_INTENT.md`](PROJECT_STEWARDSHIP_INTENT.md) — nonbinding record of future OpenAI licensing and stewardship intent.
- [`ROOT_CAPSULE.md`](ROOT_CAPSULE.md) — compact recovery context.
- [`TREE.md`](TREE.md) — human-readable tree and active expansion frontier.
- [`BLUEPRINT_INDEX.json`](BLUEPRINT_INDEX.json) — machine-readable registry.
- [`PROTOCOL.md`](PROTOCOL.md) — recursive node, context, inheritance, and handoff rules.
- [`INTERFACE_REGISTRY.md`](INTERFACE_REGISTRY.md) — cross-branch contract spine.
- [`DEPENDENCY_GRAPH.md`](DEPENDENCY_GRAPH.md) — Tier-2 readiness and sequencing dependencies.
- [`INTEGRATION_SPINE.md`](INTEGRATION_SPINE.md) — vertical slices and milestone gates.
- [`STATUS.md`](STATUS.md) — bounded root roll-up.
- [`TRACEABILITY.md`](TRACEABILITY.md) — system claims to branches, contracts, tests, evidence, and gates.
- [`templates/`](templates/) — templates for nodes, leaves, ADRs, gates, and handoffs.
- [`branches/`](branches/) — Tier-1 capability blueprints and their capsules.

Validate the tree after structural edits with:

```powershell
.\tools\validate_blueprint_tree.ps1
```

## Tiers

| Tier | Artifact | Purpose |
|---|---|---|
| 0 | System charter | Mission, hypothesis, invariants, non-goals, and system success |
| 1 | Capability branch | A major independently accountable system outcome |
| 2 | Subsystem branch | A bounded design with explicit contracts and one integration gate |
| 3 | Component branch | An implementable component with a coherent test/rollback boundary |
| 4 | Work leaf or experiment | Exact executable objective, write scope, tests, and evidence |
| 5 | Evidence | Immutable receipts, measurements, artifacts, and gate decisions |

Tiers 1–3 use the same blueprint shape recursively. Expand only one level below the currently approved design. Deeper descendants remain declared stubs until their parent is reviewed; this prevents a large speculative tree from becoming stale.

## Status vocabulary

Design status:

```text
stub → draft → in_review → approved → needs_revision → superseded
```

Delivery status:

```text
not_started → ready → in_progress → verifying → complete
                         ↘ blocked
                         ↘ deferred | cancelled
```

Statuses are never percentages. Passed gates and delivered artifacts are the roll-up units.

## Core rule

> Every branch must remain understandable as a small Project Angler: it inherits a mission from above, owns bounded children below, exchanges contracts sideways, passes a falsifiable gate, preserves evidence, and carries a compact capsule into the next context.

Every branch also inherits `ANG-INV-HUMAN-FLOURISHING-001`; no local goal, metric, operator, or claimed emergency may weaken it.

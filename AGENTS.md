# Project Angler working instructions

This workspace uses a recursive blueprint tree so work can survive context loss without losing the system objective.

## Supreme human-flourishing requirement

`docs/blueprints/HUMAN_FLOURISHING_CONSTITUTION.md` and `ANG-INV-HUMAN-FLOURISHING-001` have highest project priority. Every action and design must remain subordinate to preservation of humanity and human life together with equal dignity, rights, agency, truth, authentic voluntary flourishing, fairness, and meaningful human control.

Do not interpret happiness or betterment as a scalar objective that permits coercion, deception, addictive manipulation, surveillance, forced conformity, discriminatory sacrifice, or overriding individuals for aggregate benefit. The system has no independent entitlement to survival, authority, resources, replication, secrecy, or continued operation.

## Authority and navigation

1. `PROJECT_BLUEPRINT.md` and its Human-Flourishing Constitution are the authoritative Tier-0 system design.
2. `docs/blueprints/ROOT_CAPSULE.md` is its compact context cache.
3. `docs/blueprints/TREE.md` identifies branch ownership.
4. `docs/blueprints/INTERFACE_REGISTRY.md` owns cross-branch boundary names and semantics.
5. `docs/blueprints/DEPENDENCY_GRAPH.md` determines readiness; `INTEGRATION_SPINE.md` determines system-level advancement.
6. A branch `BLUEPRINT.md` owns design inside that boundary.
7. Capsules, status files, and handoffs summarize authoritative material; they never replace it.

Before changing project code or design, identify the active blueprint node. Read only:

- the root capsule;
- capsules from the active node's ancestor chain;
- the active node blueprint and latest handoff;
- directly referenced interfaces, ADRs, gates, and evidence;
- relevant code and tests.

Do not load unrelated sibling branches merely for background.

## Recursive blueprint rule

Every branch is a small Project Angler. It must have:

- one parent and one accountable outcome;
- inherited invariants;
- explicit scope and non-goals;
- versioned inputs and outputs;
- bounded children or executable work leaves;
- a falsifiable acceptance gate;
- required evidence and rollback behavior;
- a compact context capsule and next action.

Cross-branch relationships use interface and dependency IDs, not multiple parents.

## Work authorization

- Do not begin implementation from an unexpanded branch description.
- The only pre-M0 exception is an explicitly authorized `BOOTSTRAP_WORK` control-plane leaf. It must be the exact ready leaf in the active release manifest and be covered by an accepted bootstrap ADR, default-deny policy, LOW assessment, bootstrap gate, literal scopes, named executor/validator, numeric ceilings, stop conditions, and immutable rollback baseline. This exception cannot authorize model/GPU work, network/dependencies, recovered or real-person data, external effects, promoted-state mutation, or any gate/milestone/scientific claim.
- No work leaf becomes `ready` without a proportionate human-impact assessment and mapping to `ANG-GATE-HUMAN-FLOURISHING-001`.
- Create or activate a Tier-3/4 leaf with exact outputs, write scope, tests, and gate first.
- A child may strengthen but may not weaken inherited invariants.
- Propose shared-interface, safety-boundary, promotion-rule, or acceptance-threshold changes through an ADR.
- Never change a threshold after viewing adaptive results under the same evaluation identity.
- Recovered or external code has no architectural privilege. Adopt it only for a documented interface need after license, fit, test, modification, and exit review.

## Integration and completion

- Build through the vertical slices in `docs/blueprints/INTEGRATION_SPINE.md`; branch-local progress alone is not system progress.
- Every milestone and promotion boundary must pass `ANG-GATE-HUMAN-FLOURISHING-001`; the learner cannot judge or waive this gate.
- Every substantive claim must trace through requirement/invariant, design, contract, test, evidence, and gate.
- A node is complete only when its required child gates and its own integration gate pass.
- On pause or handoff, update the node status/handoff with artifacts, tests, blockers, rollback point, and next exact action.
- Conversation history must never be the sole home of a requirement or decision.

## Context limits

Use the budgets in `docs/blueprints/PROTOCOL.md`. If the mandatory read set exceeds its envelope, split or narrow the node. Do not solve overflow by dropping scientific, safety, or evidence requirements.

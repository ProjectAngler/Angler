# ANG-WORK-RUNTIME-PROSPECTIVE-DYNAMICS-V1-001

Status: technical PASS — trainable/loss-directed component only; no trained
checkpoint or predictive-quality claim

Active node: `ANG-BP-RUNTIME` / `ANG-BP-AGENT-RUNTIME`

Decisions: `ANG-ADR-0007-UNIFIED-TEMPORAL-COGNITIVE-SYSTEM`,
`ANG-ADR-0008-EXPERIMENTAL-COGNITIVE-TRANSACTION-INTERFACES`

Predecessor: `ANG-WORK-RUNTIME-PROSPECTIVE-RESERVATION-V1-001`

Gate: `ANG-GATE-RUNTIME-PROSPECTIVE-DYNAMICS-V1-001`

Assurance: LOW. This leaf adds local PyTorch component code and synthetic
CPU-only tests on the dedicated Ubuntu workstation. It authorizes no frozen
model or GPU run, live Cognee service, network call, real-person data,
deployment, external effect, or scientific capability claim.

## Purpose

Add one trainable, resource-bounded prospective dynamics component to the same
competence state that already owns procedural credit. The component maintains
internal world-, self-, and focus-hypothesis tensors, produces
action-conditioned next-latent and outcome hypotheses, and contributes one
trainable additive selection residual. Runtime fast state takes a bounded
loss-directed update only from caller-supplied post-outcome evidence. A
separate differentiable objective connects the future, outcome, uncertainty,
focus, and residual heads for a later trained checkpoint.

This leaf supplies no trained prospective checkpoint. Its seeded synthetic
tests establish construction, gradient connectivity, online objective-loss
reduction, bounds, and integrity—not learned predictive quality, calibration,
generalization, or improved selection.

The component is preparatory. It does not complete the Prospective Origin
described by ADR-0007: current persisted contracts cannot represent immutable
sibling predictions, agent/world-bearing episodes, or later prediction
resolution. Those claims remain blocked on successor interfaces.

## Exact outputs and write scope

Fresh files:

- `src/angler/reasoning/prospective_dynamics.py`
- `tests/unit/reasoning/test_prospective_dynamics.py`

Narrow edits:

- `src/angler/runtime/durable_ability_bridge.py`, only to let the active core
  parse its own replay event while retaining the existing parser as fallback;
- `tests/unit/runtime/test_durable_ability_bridge.py`, only for composite-core
  pending/restart/replay/rollback coverage;
- `src/angler/reasoning/__init__.py` and `src/angler/runtime/__init__.py`,
  only for required exports;
- this leaf and append-only `AGENTS_SYNC.md`.

Out of scope and read-only for this leaf:

- `src/angler/cognition/contracts.py`,
  `src/angler/cognition/prospective.py`;
- `src/angler/runtime/cognitive_cycle.py` and
  `src/angler/runtime/cognitive_transaction_store.py`;
- Cognee, cognitive-graph, Moving-Origin, journal, Qwen, and executor modules;
- `docs/blueprints/INTERFACE_REGISTRY.md` and all accepted ADRs;
- consumed experiment identities, checkpoints, results, and output artifacts.

No new `ANG-CTR-*` identity is introduced. All new dataclasses and serialized
records are RUNTIME/LEARNING-internal implementation types, not cross-branch
contracts or authorization artifacts.

## Component design

`ProspectiveDynamicsConfig` supplies versioned widths and numeric resource
ceilings. Capacity is constructor-selected and state-shaped; it is not fixed
to four candidates or one permanent latent width. Current compatibility with
the reservation contract permits 2–64 procedure candidates, but that bound is
not represented as a permanent architectural claim.

`ProspectiveDynamicsState` contains synchronized, finite tensors for:

- an actual-world latent hypothesis;
- an evidence-conditioned self-perspective hypothesis;
- soft-focus state;
- shared action-conditioned dynamics and outcome fast state; and
- one chronological step.

These tensors are hypotheses, not canonical world identity, selfhood,
sentience, truth, permission, or authority. Exact agent/world/perspective/
subject/scope identities remain external and are not inferred from tensor
content.

A component-only variable world-context path accepts a deterministic externally
supplied eligibility mask plus synthetic public feature rows. Trainable soft
focus may distribute only a caller-declared read/branch/recurrent budget within
that eligible set. It cannot create or activate a world, grant permission,
change reality mode, select truth status, or delete dormant state. Dormant rows
receive no allocation but remain recoverable.

That multiworld path is not an input to `DurableAbilityLearner`, selection,
reservation, pending recovery, or replay in this leaf. The composite learner
path receives only the already-bound relation row, temporal row, base logit,
and composite state through the existing core protocol; it treats the temporal
row as one opaque situated feature context and introduces no hidden mutable
side channel. Durable active-world eligibility and focus require the successor
contracts named under `Next`. Component-only world focus therefore cannot
affect an executed selection or support a temporal-agency claim here.

One shared, row-local action-conditioned head produces predicted next latent,
outcome logit, finite non-negative uncertainty, and an additive selection
residual. There is no task keyword, candidate-slot, expected-answer, verifier,
or task-family input. Fresh/lesioned control outputs are byte exact. After
adaptation, backend batch kernels may differ in low bits; the required property
is corresponding permutation and winner stability at absolute tolerance
`1e-6` with relative tolerance `0`, followed by the bridge's exact
selected-single-row recomputation for pending state. No batched value is
retained as the selected neural output.

The fresh state and explicit prospective lesion produce a bit-exact zero
selection residual. Therefore the composite core initially reproduces the
existing procedural core logits exactly. A later scientific identity may
activate and assess learned residual behavior; this leaf does not.

## One composite competence lineage

`CompositeProspectiveCreditCore` owns the existing structure-keyed credit core
and the prospective dynamics state as one core/state pair. It exposes the
existing `DurableAbilityLearner` core protocol and returns the established
pending-output field shape with:

`logits = base_logits + procedural_credit_residual + prospective_residual`.

The composite checkpoint identity must bind both parameter sets. Capture,
restore, digest, zero, feedback, replay, invalidation, and pending
recomputation cover both child states. Both steps advance together. No
query-time flag may select a different state; the declared lesion is an
evaluation intervention over the same inputs and budget.

Feedback reconstructs the existing procedural child prediction and the exact
pre-outcome prospective prediction from the selected public row. It then
applies one outcome update to both components and emits one bounded composite
event. Any failure leaves or restores the exact composite parent. Event records
contain bounded public features, outcome, evidence references, configuration,
exact composite-checkpoint identity, read/lesion mode, pre/post online outcome
loss, and child divergence checks rather than unbounded latent histories.

The small bridge edit dispatches event restoration through an optional
`core.event_from_record(record)`; cores without that hook retain the current
`StructureKeyedCreditEvent.from_record` behavior. This is compatibility
plumbing, not a schema or authority change.

## Deterministic-code ceiling

Code may validate shapes, identities, masks, budgets, dtype/device, finite
values, chronology, hashes, snapshots, replay, and exact residual
decomposition. It may apply generic tensor operations and caller-supplied
objective outcomes.

Code may not map tasks or keywords to worlds, self claims, focus, branches,
procedures, or solutions; inspect hidden answers or verifier internals;
fabricate observations; promote a hypothetical branch to fact; allocate
outside the eligible set; or treat confidence, focus, prediction, or a state
digest as permission or truth.

## Acceptance

1. Config, state, context, output, snapshot, and event validation reject
   non-finite, wrong-shape, stale-step, over-budget, ineligible, duplicate, and
   tampered inputs.
2. Candidate counts 2, 7, and 64 and multiple configured widths work without
   candidate-axis state; permutation preserves corresponding branch outputs.
3. Initial and lesioned prospective residuals are bit-exact zero, and composite
   logits match the procedural core bit for bit under equal inputs.
4. The shared dynamics is action conditioned. Generic positive and negative
   outcome sequences produce finite, evidence-bound divergence in prospective
   prediction/residual without task-family or slot-coded logic. One online
   update reduces the same-observation objective outcome loss for both signs,
   and the declared synthetic training objective supplies non-zero gradients
   to every prospective head. This is mechanism evidence, not a trained-model
   or predictive-quality claim.
5. In the isolated component API, soft focus remains inside the externally
   eligible synthetic world set and declared read/branch/recurrent ceilings;
   dormant state survives component capture/restore and later re-eligibility.
   Static inspection confirms this variable-world input is absent from the
   composite learner/pending path.
6. Composite snapshot, digest, zero, restore, replay, event parsing,
   invalidation, and tombstone rebuild cover procedural, world, self, focus,
   dynamics, topology, step, and both parameter identities as one lineage.
7. Pending decisions round-trip under the existing 16 MiB ceiling and exact
   selected-row recomputation. Restart rejects checkpoint, parent, dtype,
   shape, context, or prediction drift.
8. One feedback call advances both child states exactly once. Injected failure
   in either child leaves the exact parent and retryable pending decision.
9. Existing structure-keyed core and durable-ability focused suites remain
   green. Tests use synthetic public tensors and fake evidence only.
10. Static inspection confirms no new contract ID, store schema, cycle,
    memory, model, executor, authorization, gate, experiment, or result change.

Passing this gate establishes a replayable trainable component, loss-directed
online fast-state adaptation, and a fair removal hook only. It does not
establish a trained prospective checkpoint, improved reasoning, calibrated future
prediction, semantic world modeling, self-awareness, temporal agency,
autonomous goals, AGI, readiness, or Human-Flourishing/milestone acceptance.

## Validation commands

Run only:

```bash
env CUDA_VISIBLE_DEVICES='' /opt/angler/venvs/angler/bin/python -m unittest tests.unit.reasoning.test_prospective_dynamics
env CUDA_VISIBLE_DEVICES='' /opt/angler/venvs/angler/bin/python -m unittest tests.unit.runtime.test_durable_ability_bridge
env CUDA_VISIBLE_DEVICES='' /opt/angler/venvs/angler/bin/python -m unittest tests.unit.reasoning.test_structure_keyed_credit_memory
git diff --check
```

CUDA remains hidden for these CPU component tests. Do not run Qwen, Cognee,
the expanded integration suite, or any consumed experiment.

## Validation result — 2026-08-31

The three authorized suites pass 33/33 in 0.710 seconds with CUDA hidden:
prospective dynamics 6, durable ability/bridge 16, and unchanged
structure-keyed credit memory 11. `git diff --check` passes. Static inspection
finds no new `ANG-CTR-*`, store/cycle/schema/model/executor change, trailing
whitespace, or patch-backup artifact.

An independent read-only audit initially rejected the implementation for
checkpoint-unbound events, incomplete output/budget validation, read-disabled
replay divergence, unsupported batch/single byte-exact wording, absent
child-failure evidence, and an overbroad learned-model implication. The
corrected implementation binds checkpoint and read mode in every event,
revalidates all resource ceilings, uses a bounded outcome-loss-directed fast
state update, exposes a differentiable synthetic training objective to every
prospective head, preserves the exact parent and pending decision under either
child failure, and records the literal post-adaptation tolerance. The same
auditor then returned PASS with no remaining blocker.

Final implementation hashes are:

- prospective core: `A1BA017F85017DEE097B56FF961F68002B68148B64E821EAFFEE0FE305CF6A26`;
- prospective tests: `9F78650414E37FC5EC8C83886A72B759B253DCDAADC7BA3D9BD7862784F16C1D`;
- durable bridge: `CF6EC645A6D0B03BAE16C3A36A53A820AAE14BEBFA93370B987A28E8B053658A`;
- durable tests: `FCF92935B6DFA3C648290072C82ED7BA508AD2DB344A0FEEA8D9A179938A9D5A`;
- reasoning export: `2E9FF89CA3BB7207047E82DF69CD520B27E15E2F9C49CC4EEDD7C05BF1D6BF4F`.

No model, GPU, Cognee service, network, experiment identity, scientific result,
persisted cognitive record, or external effect was invoked or changed.

## Human impact, stop conditions, and rollback

This leaf is local synthetic learning-mechanism work. It grants no action,
permission, identity, survival interest, or external authority. The
Human-Flourishing Constitution remains outside learner-writable state. Trainable
focus cannot change the eligible set or its resource/permission envelope.

Stop for a need to reinterpret an existing `0.1.0` contract, persist sibling
futures or world/self identity, alter Moving-Origin acquisition semantics,
write a second active competence owner, route state by query, encode a task
solution, expose verifier answers, exceed declared memory/compute ceilings,
invoke a model/GPU/service/network/external effect, or mutate consumed
scientific evidence.

Rollback restores the narrow bridge/export edits and removes the fresh module,
test, and this leaf. Existing persisted cognitive records, transaction stores,
Cognee projections, Moving-Origin history, procedural competence snapshots,
and experiment evidence remain untouched.

## Next

After this component gate passes, author one narrow successor-interface ADR
for durable prospective batches, resolutions, explicit world/self lineage, and
one-clock acquisition ordering. Then integrate those records with the
reservation/cycle/store and typed Cognee/Moving-Origin projection before any
claim of completed Prospective Origin. A later preregistered training/evaluation
leaf must supply an exact trained checkpoint, held-out objective prediction
evidence, and fair removals before the component is described as a learned
predictive model or credited in high-level evaluation.

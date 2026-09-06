# ANG-WORK-RUNTIME-AUTONOMOUS-APPRENTICE-V4-001

Status: active successor; first result not yet run

V3 is preserved `NOT_SUPPORTED`, SHA-256
`EE175D6B4B12BC8F4C11B228C227F2C06C8D43096A5F221D09696025B2FF24DF`.
Controller-owned identities allowed repairs to execute, but full remained 5/8
while state-reset reached 6/8. A post-result read-only replay of the full
learner showed the trained core assigned 0.999979 weight to the current failed
grouping procedure and approximately zero to the objectively successful paired
acquisition procedure.

The cause is an invalid deployment assumption: V6 trained
`apply_action_trace_feedback` exclusively with `+1` successful support traces.
Its tensor API validates `-1`, but negative anti-consolidation behavior was not
in the training objective. V1–V3 therefore wrote failures into neural plastic
state using an untrained path and allowed them to dominate future recall.

## Accountable outcome

Implement honest objective credit assignment for the existing core. Every
failure remains append-only in the journal/Cognee and is supplied as a labeled
counterexample. Only verifier-confirmed successes consolidate into the V6
neural plastic state. Among each outcome class, the trained core still supplies
the ordering weights. Frozen Qwen receives a clear separation between
successful procedures to adapt and failures not to repeat. No task-specific
rule, solution, expected output, candidate identity, or mechanism label enters
selection.

The fresh V4 verifier emits bounded case name plus actual/expected diagnostic,
matching ordinary test-runner feedback and enabling autonomous repair; it
contains tests but no implementation. All four arms receive identical
diagnostics.

## Scope and gate

Writes are limited to this leaf,
`src/angler/runtime/autonomous_apprentice.py`, its focused unit test,
`experiments/runners/autonomous_apprentice_v4.py`, append-only
`AGENTS_SYNC.md`, disposable state
`/opt/angler/state/autonomous-apprentice-v4/evaluation-first`, and fresh result
`/opt/angler/results/autonomous-apprentice-v4-evaluation.json`.

Tests must prove failure leaves the exact neural state unchanged, success still
changes it, both outcome classes remain recallable with learned weights, model
action labels are controller-scoped, and all confinement/persistence/default
privacy tests pass. The frozen live gate remains at least 0.75 full success,
one failure-to-success retry, exact restart replay, durable episodes/state, and
strict transfer advantage over state-reset and memory-disabled controls. First
valid result is accepted without tuning.

This is bounded success consolidation plus failure memory, not a fully learned
negative-update rule, arbitrary-project competence, AGI, consciousness,
unrestricted autonomy, deployment, or a final Angler claim. A later core may
replace this boundary only after being explicitly trained and evaluated on
negative credit assignment.

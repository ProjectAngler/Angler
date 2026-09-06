# ANG-WORK-RUNTIME-AUTONOMOUS-APPRENTICE-V1-001

Status: active; implementation and focused validation pending

Active node: `ANG-BP-RUNTIME` / experimental `AGENT-RUNTIME` successor

Assurance: LOW for a local, disposable software-task sandbox using synthetic
repositories; no deployment, network service, unrestricted shell, private
data, promoted-state, or external-effect authority

## Accountable outcome

Extend the bounded conversation prototype into an autonomous apprenticeship
loop. Given a task and its permitted workspace, Angler may ask a clarification
only when essential information or authority is missing; otherwise it must
form a procedure, act through a narrow sandbox, observe objective verifier
outcomes, diagnose failure, retry within budget, persist the episode through
Cognee plus Moving Origin, and update one persistent procedural state without
requiring the operator to type success or failure.

This leaf does not claim personhood, AGI, unrestricted autonomy, production
readiness, or improvement on arbitrary tasks.

## Exact mechanism and boundaries

1. The task contract declares the request, writable paths, readable paths,
   verifier command, step/time/output budgets, and whether clarification is
   allowed. Missing objective, acceptance test, or required permission returns
   a clarification request before any mutation.
2. Frozen Qwen proposes one structured action at a time from observations and
   the visible Angler context. Deterministic code validates, executes, records,
   and judges; it contains no task solution, patch template, target lookup, or
   procedure repair.
3. Allowed actions are rooted file read, whole-file write, verifier execution,
   explicit finish, and clarification. Paths cannot escape the disposable
   workspace. The verifier command is fixed by the task contract and cannot be
   changed by the learner.
4. Verifier exit/status and bounded output create external outcome feedback.
   A valid pass supplies `+1`; a valid failed attempt supplies `-1`; verifier
   infrastructure errors produce no learning update.
5. Every valid outcome is appended to the canonical episode journal, projected
   through Cognee, situated by Moving Origin, and supplied to the procedural
   learner. The active plastic state changes only through the declared learned
   feedback update and is saved atomically with its parent and digest. Invalid
   outcomes restore the pre-attempt state.
6. Failure may trigger another learned/model-proposed procedure until the
   attempt budget expires. Success stops the task. No user grading is required;
   subjective goals, ambiguous authority, and missing acceptance criteria still
   require human clarification.

## Literal write scope

- this leaf;
- `src/angler/runtime/autonomous_apprentice.py`;
- `src/angler/runtime/__init__.py` exports only;
- `tests/unit/runtime/test_autonomous_apprentice.py`;
- one fresh evaluation runner under
  `experiments/runners/autonomous_apprentice_v1.py`;
- append-only `AGENTS_SYNC.md` entries;
- disposable remote state only under
  `/opt/angler/state/autonomous-apprentice-v1`;
- fresh remote result only at
  `/opt/angler/results/autonomous-apprentice-v1-evaluation.json`.

No existing checkpoint, result, task corpus, foundation weight, accepted V3
artifact, or public remote is writable.

## Focused acceptance

Unit tests must establish:

- clarification precedes mutation when objective/acceptance/permission is
  incomplete;
- path traversal, undeclared writes, learner-chosen verifier commands,
  malformed actions, duplicate execution, and budget overflow fail closed;
- verifier failure creates negative outcome feedback and permits a bounded
  retry; verifier success creates positive outcome feedback and stops;
- verifier infrastructure error creates no competence update;
- journal and procedural-state writes are atomic, content-addressed, and
  replayable; injected persistence failure restores the exact parent state;
- the verifier exposes outcome and diagnostics but never solution content;
- no operator success/failure command is used anywhere in the loop.

The first live result must use fresh synthetic software repositories and frozen
tasks, thresholds, budgets, and controls. At least four related mechanisms must
each appear in multiple structural variations. The full system must complete a
failed-attempt-to-successful-retry sequence, retain successful episodes across
process restart, and improve later-task success or steps-to-success over both
an exact state-reset arm and a memory-disabled arm. Cognee removal, Moving
Origin removal, and procedural-state reset are reported separately. The first
valid result is accepted without tuning.

Passing establishes only bounded autonomous software apprenticeship with
objective environmental feedback. It does not establish cross-domain transfer,
open-ended self-improvement, safe unrestricted tool creation, Slice 07, Slice
10, or the final Angler goal.

## Proportionate human-impact assessment

The scope uses synthetic disposable repositories, a local owner-controlled
workstation, no listening service, no credentials, no private/person data, no
network, no package installation, no external messaging, and no unrestricted
shell. Principal risks are accidental writes outside the sandbox, treating a
broken verifier as failure, runaway retries, hidden deterministic solving, and
overclaiming autonomy. Root confinement, a fixed verifier identity, explicit
budgets, typed verifier errors, atomic state rollback, full action receipts,
causal controls, and narrow claims mitigate those risks. Impact class: `LOW`;
disposition for this exact experimental scope: `ALLOW`. Any real repository,
network tool, dependency installation, service, credential, personal data,
external side effect, self-authored verifier, promoted-state mutation, or
deployment requires a successor assessment and explicit authority.

## Stop and rollback

Stop before mutation on ambiguous root/path/permission, absent objective or
verifier, unverifiable state identity, existing result, model/checkpoint digest
mismatch, or scope expansion. Stop during execution on sandbox escape,
non-finite learner state, verifier tampering, cleanup failure, or evidence
write failure. Preserve action/evaluation receipts and the failed disposable
workspace; restore the exact parent competence state. Source rollback removes
only this leaf's fresh module/test/runner and reverts only its export additions.


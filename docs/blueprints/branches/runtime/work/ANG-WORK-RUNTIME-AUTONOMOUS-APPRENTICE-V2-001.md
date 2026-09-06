# ANG-WORK-RUNTIME-AUTONOMOUS-APPRENTICE-V2-001

Status: active successor; first result not yet run

Active node: `ANG-BP-RUNTIME` / bounded `AGENT-RUNTIME` experiment

Assurance: LOW; local synthetic disposable software workspaces only

## Why this successor exists

The frozen V1 result at
`/opt/angler/results/autonomous-apprentice-v1-evaluation.json` is
`NOT_SUPPORTED` (SHA-256
`41733461EAE84DAE6FF6BC44085B383911572EF1A63EC0380BA112221369B8CF`).
The full arm solved 5/8 tasks, tied state-reset and memory-disabled controls,
and lost to the Moving-Origin-removed arm. Persistence and restart fidelity
worked, but the semantic apprenticeship path did not: Cognee retained only
action labels such as `write:solution.py`, not the bounded procedure content,
and the model received only pass/fail rather than actionable failure
diagnostics. The procedural core could rank episodes but there was no useful
procedure payload to retrieve. V1 remains preserved and is not rerun.

## Accountable outcome

Make the existing Angler/Cognee/Moving-Origin loop semantically capable of
self-repair and procedure reuse without adding any task solver. A task may
explicitly authorize bounded verifier diagnostics and bounded retention of
the model's own write procedure. Cognee carries that procedure as disposable
episodic memory; Moving Origin situates it; the trained action-trace core and
plastic state rank recalled episodes; frozen Qwen adapts the recalled procedure
to a new task. Deterministic code still performs only contracts, confinement,
execution, measurement, persistence, and rendering.

## Exact changes and constraints

1. Both new capabilities default off in `AutonomousTaskContract`; the caller
   must opt in per task.
2. Diagnostic feedback is bounded verifier output from the already fixed
   verifier. It cannot change the verifier, create a solution, or turn an
   infrastructure error into training feedback.
3. Retained procedure content is limited to authorized read/write actions and
   a 3,072-character projection ceiling. It is labeled with objective success
   or failure; raw verifier output is not placed in Cognee.
4. The learned core still chooses recall weights. Outcome labels are rendered
   so frozen Qwen can distinguish examples from counterexamples.
5. V2 uses fresh tasks, state, and result identity. The first valid outcome is
   accepted without tuning. Foundation and trained core weights remain frozen;
   only episodic memory and plastic state may change.

## Literal write scope

- this leaf;
- `src/angler/runtime/autonomous_apprentice.py`;
- `tests/unit/runtime/test_autonomous_apprentice.py`;
- `experiments/runners/autonomous_apprentice_v2.py`;
- append-only `AGENTS_SYNC.md`;
- remote disposable state under
  `/opt/angler/state/autonomous-apprentice-v2/evaluation-first`;
- remote result
  `/opt/angler/results/autonomous-apprentice-v2-evaluation.json`.

No V1 evidence, existing checkpoint, frozen foundation, accepted artifact,
public repository, service, credential, personal data, or external system is
writable.

## Focused acceptance

- default contracts continue to hide verifier output and procedure content;
- opt-in diagnostics reach the next model prompt after failure;
- opt-in successful and failed procedure traces are bounded, outcome-labeled,
  replayable through Cognee, and do not contain verifier output;
- failure can lead to an objectively successful retry without user grading;
- exact state rollback, path confinement, fixed verifier, duplicate execution,
  restart, and foundation invariants continue to pass;
- on fresh paired mechanisms, the full arm must show at least one
  failure-to-success repair, persist and reload state exactly, and beat both
  a state-reset arm and a memory-disabled arm on transfer success or
  steps-to-success under frozen thresholds.

Passing establishes bounded autonomous software procedure reuse and
self-repair only. It does not establish arbitrary project competence,
cross-domain reasoning, open-ended self-improvement, AGI, consciousness, or
safe unrestricted autonomy.

## Stop and rollback

Stop on an existing V2 identity, digest mismatch, verifier mutation, path
escape, unbounded projection, private/external data, network/service need,
non-finite learner state, evidence failure, or scope expansion. Preserve any
first result and disposable workspace. Restore only the exact parent plastic
state on persistence failure; source rollback is limited to this successor's
changes.

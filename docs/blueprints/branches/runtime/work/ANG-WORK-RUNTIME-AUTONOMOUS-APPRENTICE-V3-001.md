# ANG-WORK-RUNTIME-AUTONOMOUS-APPRENTICE-V3-001

Status: active successor; first result not yet run

Active node: `ANG-BP-RUNTIME`; LOW-impact local synthetic evaluation

## Evidence and accountable outcome

V2 is preserved `NOT_SUPPORTED` at SHA-256
`1E73FF87618324FF1E8FDC1A194BEA356BC691D2D5124AFA8272FDCECA272B84`.
Its semantic procedure and diagnostic interfaces were active, but full still
solved 5/8 and did not recover a failed attempt. A fresh read-only replay of
the first post-failure prompt showed Qwen proposed the correct repair, including
the missing import, but reused model action ID `1`. The sandbox rejected this
as conflicting with its earlier action ID, after which the same valid repair
was rejected until step exhaustion.

V3 makes execution identities controller-owned and turn-scoped while retaining
the model's label for traceability. This is generic idempotency plumbing: it
does not inspect or change paths, content, action kind, task, diagnostics, or
solution. Direct duplicate calls to `RootedSoftwareSandbox` remain exactly-once
and conflict-detecting. The accountable outcome is to let model-proposed
self-repairs actually execute, then evaluate the existing semantic
Angler/Cognee/Moving-Origin path on one fresh frozen task suite.

## Scope and acceptance

Literal writes are this leaf, `src/angler/runtime/autonomous_apprentice.py`,
its focused unit test, `experiments/runners/autonomous_apprentice_v3.py`, an
append-only `AGENTS_SYNC.md` entry, disposable remote state under
`/opt/angler/state/autonomous-apprentice-v3/evaluation-first`, and the fresh
result `/opt/angler/results/autonomous-apprentice-v3-evaluation.json`.

Tests must prove that a model may reuse its own label across write, verify,
repair, and reverify turns; the controller produces unique scoped identities;
the sandbox's direct duplicate protection still holds; and all V2 default-off,
diagnostic, retained-procedure, confinement, rollback, and persistence tests
remain green. The fresh result retains V1's frozen causal gate: at least 0.75
full success, at least one failed-attempt recovery, exact restart replay,
persistent episodes/state, and strict transfer benefit over both state-reset
and memory-disabled controls. First valid V3 result is accepted without tuning.

No foundation/core training, unrestricted shell, real repository, network,
service, external effect, private data, deployment, public push, AGI, or
open-ended self-improvement claim is authorized.

# ANG-WORK-RUNTIME-PROCEDURAL-EXPERIENCE-BRIDGE-V1-001

Status: implemented and live; focused/runtime validation passed

Active node: `ANG-BP-RUNTIME`

Outcome: make Jenny's observed experience reusable immediately, without a
foundation-model training pass or an expanding VRAM resident adapter set.

The leaf derives one content-addressed procedural case from each evaluated
canonical episode: condition, goal, selected action/capability revision,
prediction, observed result, feedback, and Moving-Origin provenance. Cognee
may retrieve the derived case, but canonical episodes remain authoritative.
Unevaluated model output is never promoted into a procedure.

For every scalar observed outcome, exact online count/mean/M2 vectors are
retained per affordance and exact capability revision. No dimension is reduced
to an arbitrary emotion or utility scalar. Full history remains reconstructible
from canonical episodes; the online view requires no replay and no GPU memory.

Acceptance: focused tests prove late feedback binds the original request,
unevaluated text is excluded, failures remain learnable evidence, case identity
is deterministic, Moving Origin is retained, and vector moments update exactly.
The existing runtime suite must remain green before live activation.

Result (2026-09-04):

- `tests.unit.runtime.test_procedural_experience`: 6/6 passed.
- Affected adapter/Cognee/runtime set: 77/77 passed.
- Complete runtime unit suite: 463/463 passed in 52.813 seconds with
  `PYTHONPATH=src:scripts`.
- Read-only replay of all 96 live canonical episodes produced 6 grounded
  procedural experiences, excluded 88 unevaluated exchanges, retained the
  existing model-authored artifact and legacy consolidation, and had zero
  projection failures. Maximum projected memory was 2,950 characters against
  the 4,096-character ceiling.
- The live Jenny service was restarted onto these source bytes and reported
  `api_ready`, `model_runtime_ready`, `life_loop_running`, and scheduler ready,
  with no retained life-loop or projection error.

Interpretation: this is immediate, bounded procedural plasticity. Evaluated
experience becomes retrievable evidence and exact per-procedure consequence
profiles update online without replay. It adds no model call, adapter swap,
training pass, or VRAM allocation. It does not establish emotional awareness,
subjective feeling, or broad capability improvement by itself.

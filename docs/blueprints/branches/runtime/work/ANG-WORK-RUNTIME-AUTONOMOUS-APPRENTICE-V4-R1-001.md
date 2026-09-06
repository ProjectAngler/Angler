# ANG-WORK-RUNTIME-AUTONOMOUS-APPRENTICE-V4-R1-001

Status: complete technical recovery; scientific result `NOT_SUPPORTED`

The original V4 execution stopped before a terminal result because an
identical failed procedure, unchanged neural parent, and identical verifier
receipt produced an already-recorded content-addressed episode. The canonical
journal correctly rejected the duplicate. Failed V4 state is preserved under
`/opt/angler/state/autonomous-apprentice-v4/evaluation-first`; no V4 result
exists and that identity will not be reused.

R1 changes only persistence idempotency: if an episode's artifact reference
already exists and the complete `JournalRecord` is byte-semantically equal,
recording is a no-op. A same-reference/different-record conflict still fails.
No procedure, outcome, learner state, retrieval weight, task, threshold, or
verifier changes. The exact V4 task suite and gate are reused under fresh
state/result identity. Writes are limited to this leaf, the recorder and its
focused test, `experiments/runners/autonomous_apprentice_v4_r1.py`, append-only
coordination, recovery state, and recovery result.

## Frozen result

The recovery ran once in 185.064 seconds and produced
`/opt/angler/results/autonomous-apprentice-v4-r1-evaluation.json`, SHA-256
`226E92BDA20DFA942B2DB4DEDA6CEE7124A5BCABDF9278EEBBC425FC75BF5123`.
Classification: `NOT_SUPPORTED`.

- full: 4/8 overall, 1/4 transfer, no failure-to-success recovery;
- state-reset: 5/8 overall, 2/4 transfer;
- memory-disabled: 4/8 overall, 1/4 transfer;
- Moving-Origin-removed: 5/8 overall, 2/4 transfer;
- full state changed, eight-task evidence persisted, and exact restart replay
  passed; causal performance gates failed.

Interpretation: runtime correctness, diagnostic feedback, procedure retention,
action identity, idempotency, objective persistence, and restart are now
demonstrated. The frozen V6 core is not a suitable apprenticeship learner: its
training distribution contains successful synthetic pipeline traces, not mixed
objective coding episodes, and its persistent state remains harmful relative
to reset/removal controls. Do not add another prompt or deterministic outcome
preference to claim success. The next valid work is a fresh LEARNING successor
that trains mixed success/failure credit assignment and tests held-out
apprenticeship transfer before returning to this runtime.

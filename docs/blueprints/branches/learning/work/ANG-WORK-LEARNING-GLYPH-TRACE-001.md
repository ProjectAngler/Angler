---
blueprint_id: ANG-WORK-LEARNING-GLYPH-TRACE-001
parent_id: ANG-BP-SKILL-COMPOSITION
revision: 1
tier: 4
design_status: approved
delivery_status: complete
human_authority: project owner direction, 2026-08-27
human_impact: LOW; contained local synthetic procedural learning
depends_on:
  - ANG-WORK-LEARNING-SKILL-LOCAL-MEMORY-001@1
---

# Glyph-machine trace precursor

## Objective

Replace the fixed five-item permutation action space with variable, typed
actions and an autoregressive learned procedure.  On a private family of small
reversible state machines, Angler must acquire transition procedures from
public traces, compose them to reach a public goal, and improve from scalar
terminal feedback.  This is the controlled precursor to GlyphMachine
Reconstruction, not the software-reconstruction result itself.

## Exact increment

- `experiments/evaluators/glyph_machine_trace_suite.py`
- `experiments/runners/phase5_glyph_machine_trace.py`
- `tests/unit/experiments/test_glyph_machine_trace_suite.py`
- `tests/unit/experiments/test_phase5_glyph_machine_trace.py`
- `outputs/phase5-glyph-machine-trace-result-summary.json` after a completed run

Reuse the existing typed `State`, `Goal`, `ActionSchema`, `GroundAction`,
`Transition`, and `Trace` records; routed fixed-capacity memory; reversible
transition; transactional feedback; scalar-preference learning; and frozen
checkpoint reload.  Do not reuse `_PERMUTATIONS`, five-item feature packing, or
the existing permutation policy.

Public tasks contain two to four opaque states, one to three opaque actions,
zero or more observed transition traces, an origin, exact goal, and a one-to-
four-step budget.  The committed output is a bounded tuple of declared
`GroundAction` values plus a stop decision.  The hidden evaluator owns the
transition table and mechanism commitment and returns only terminal `0.0` or
`1.0`; it exposes no reached state, distance, next action, target procedure, or
diagnostic.

The learner performs one neural autoregressive rollout.  It may encode public
records, route observed transitions into fixed-capacity memory, predict a
successor for each available action, compare predicted successors with the
public goal, select an action or STOP, and update from attempted procedures and
scalar outcomes.  It may not reconstruct or inspect the hidden table, enumerate
action sequences, run BFS/shortest-path/dynamic-programming search, or call a
domain solver.

## Scale and partitions

Canonicalize mechanisms under simultaneous state renaming and action
reordering.  Use semantic partitions of 64 train, 16 development, and 16 final
mechanisms; surface names and presentation order come from independent seeds.
The smoke profile may use an 8/4/4 subset.  Query tasks contain no observations,
so retained procedural state—not prompt replay—must carry acquisition.

The runner must accept a small smoke profile and a resource-measured graph
profile implemented by the same architecture and differing only in declared
capacity.  The graph profile must contain 20--30 million trainable parameters
(nominal target 25 million) across typed encoders, graph depth/width, compact
memory, causal prediction, and the learned updater.  It must use factorized
typed action pointers rather than enumerating complete plans.  The smoke
profile exists only for fast interface and gradient tests; it is not evidence
that the controller is large enough for reconstruction.  Qwen grounding and
code editing are deferred until the dynamic procedure interface works.  A
100--200 million parameter reconstruction controller is the next capacity
band if the graph profile retains the causal learning effect.

Angler's controller, compact procedural memory, causal model, learned updater,
and active plastic state remain device-resident throughout each online
reasoning and learning episode.  Future quantization or CPU offload may apply
only to immutable foundation-substrate weights or cold archival evidence; it
must not page the live Angler state or interrupt its update continuity.

## Acceptance and falsification

Tests must prove deterministic generation; semantic partition disjointness;
rename canonicalization; public/hidden separation; dynamic shapes; declared-
action-only rollouts; exact budget/STOP handling; one scalar per attempted
procedure; fixed memory capacity; rollback; checkpoint reload before final;
and an AST-level absence of hidden-field reads and sequence-search/solver APIs.

A bounded positive result requires correct public traces to outperform both
no-trace and wrong-trace controls by at least five percentage points on the
sealed final mechanisms, with a positive correct-over-no-trace effect on at
least ten of sixteen mechanisms.  Removing the learned reversible transition
must remove at least half of the acquired gain.  Report the full distribution,
not only the mean.  Failure changes the learned representation or updater; it
must never be patched with a deterministic machine solver.

Passing establishes only variable-action procedural acquisition and
composition in a small synthetic world.  It does not establish broad abstract
reasoning, human-level reasoning, software reconstruction, or AGI.

## First controller diagnostic — 2026-08-27

The initial dynamic controller and scalar-only experiment harness passed 21
focused tests.  Its smoke and resource profiles share one architecture and
measure 81,314 and 27,481,474 trainable parameters respectively; the resource
profile occupied 111 MB after loading and peaked at 243 MB for one traced
forward/backward step before optimizer state.  A 64-mechanism, five-epoch
smoke run performed 320 optimizer steps, 1,280 scalar attempts, and 640 public
trace losses, but produced exactly zero correct-trace advantage on all 16
development mechanisms.  The final partition remained unopened.

The failure is architectural rather than a capacity shortfall.  Trace writes
were keyed by successor state while inference queried a strict goal-keyed
route, leaving many reads exactly empty; the remaining signal was attenuated
through a successor softmax/convex mixture and overwhelmed by an unconditioned
action-pointer bypass.  The pre-acquisition trace objective also rewarded
surface memorization instead of evidence use.  Do not train the resource
profile in this form.

The successor keeps the scalable graph controller but replaces that path with
a fixed-capacity learned associative transition memory keyed by `(state,
action)`, one soft context per candidate action, post-write and
leave-one-observation-out recall, direct causal successor/goal scoring, and
whole-trajectory scalar preference credit.  It must first demonstrate a
material correct/no-trace/wrong-trace separation whose gain disappears under
transition removal; no deterministic transition lookup, search, or answer
head may be introduced.

## v3.2 learned-address repair and development result — 2026-08-27

The v3.1 resource run localized the remaining failure to associative
addressing rather than scale: correct/no-trace/wrong-trace/transition-removed
development accuracy was `31.25/28.125/31.25/31.25%`.  Raw public event keys
ranked their own event perfectly, but independent learned query/key projections
reduced exact binding to `12.4%`, and the projected successor-value path stayed
near chance even under oracle routing.

v3.2 preserves full public `(state, action)` structure with parameter-free
identity anchors plus orthogonal, smoothly bounded learned residuals.  The
same learned soft-address path is used for trace writes, lattice reads, and
rollout; stored successor values remain in the state-representation space and
are recalled without a learned rotation.  These anchors identify public
evidence but neither predict its value nor choose a procedure.  Hidden tables,
metadata IDs, exact transition lookup, plan enumeration, and search remain
outside the inference path.

The focused host and WSL verification passed `40` tests with the `7` tests that
open or enumerate final/sealed partitions deliberately excluded.  The resource
controller has `26,422,466` trainable parameters.  Its CUDA structural gate
measured minimum exact-pair cosine margin `0.433203`, minimum correct-slot
attention mass `0.999870`, and minimum correct-successor mass `0.786148`.

A fresh `64`-mechanism, `3`-epoch resource run used three supports, two queries,
and three observations per support.  It completed `192` optimizer steps and
`1,920` scalar attempts in `205.15` training seconds, with zero complete-plan
candidates and a `660,253,696`-byte peak CUDA allocation.  On all `16`
development mechanisms, the trace-only correct/no-trace/wrong-trace/transition-
removed accuracy was `84.375/40.625/37.5/18.75%`.  An expanded frozen-checkpoint
diagnostic using eight queries per mechanism produced
`72.65625/26.5625/22.65625/22.65625%`; correct evidence beat both absent and
wrong evidence on `16/16` mechanisms.  No retraining occurred between these
evaluations, and final remained unopened.

The frozen v3.2 checkpoint is
`/opt/angler/experiments/glyph-v3.2-resource-64x3-dev.pt`, SHA-256
`A532D6E8B86107D85FA07ED5E6F46E5EDB161B670F8A6564F0E2577B09FE0FE4`.
Development records are retained beside it with SHA-256
`19C6FFDAD9DAFBA0A95BD0CAE5C27E95834D3F8850366F238D350DA5A5D9A3FC`
and `7BD74A1BE2EF8422EBED3DEB26CAC9DDE53824DF8CABCAE75C4F4648C8F1E0BF`.

The one-shot final evaluation is now fixed: strict-reload that exact checkpoint;
use seed `109001`, all `16` final mechanisms, three supports, three observations
per support, and eight queries per mechanism; retain trace-only acceptance and
sequential adaptation as a separate diagnostic; apply the unchanged acceptance
and falsification rules above; and perform no tuning after final is opened.

## One-shot sealed final result — 2026-08-27

The exact frozen checkpoint above was strictly reloaded and evaluated once on
all `16` sealed final mechanisms with the fixed eight-query protocol.  Trace-
only correct/no-trace/wrong-trace/transition-removed accuracy was
`62.5/28.125/28.90625/23.4375%`.  Correct evidence therefore gained `34.375`
percentage points over no evidence and `33.59375` points over wrong evidence.
Correct beat no trace on `14/16` mechanisms and wrong trace on `13/16`.
Removing the learned transition path removed `39.0625` points, more than the
required half of the acquired gain.  Every fixed acceptance condition passed.

The complete distributions and nonclaims are recorded in
`outputs/phase5-glyph-machine-trace-result-summary.json`, SHA-256
`648E42054F55D14835332D3FC2620649305969F45C4E34763A68711A8D6B7D8F`.
The immutable raw final record remains at
`/opt/angler/experiments/glyph-v3.2-resource-64x3-final-q8.json`, SHA-256
`5B5AFF00A17953151289D5AAC3CDBD3C13D114B0CE9296EEA9D76AA42A11E04D`.
Final is now consumed and may not be used for tuning.

A post-result development-only coverage audit found that supports exposed a
mean `77.43%` of the small world's state/action edges and `119/128` development
queries were solvable using only observed edges.  The result therefore proves
learned associative routing and neural composition of retained transitions,
not cross-system procedural abstraction.  The successor software-pipeline
leaf addresses this directly with zero exact support/query address overlap and
never-observed motif compositions.

## Effects and rollback

Local synthetic data and foreground WSL2 compute only.  No network, service,
personal/recovered data, model-weight update, deployment, or external effect.
Rollback removes only the four additive code/test files and any failed-run
result artifact; prior checkpoints and evidence remain immutable.

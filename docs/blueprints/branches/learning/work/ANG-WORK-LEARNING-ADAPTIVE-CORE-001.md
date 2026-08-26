---
blueprint_id: ANG-WORK-LEARNING-ADAPTIVE-CORE-001
parent_id: ANG-BP-META-UPDATER
revision: 1
tier: 4
design_status: approved
delivery_status: complete
human_authority: project owner direction, 2026-08-25
human_impact: LOW; contained local synthetic learning research
depends_on: ANG-WORK-LEARNING-PERSISTENT-PLASTICITY-001
---

# Connect persistent plasticity to reasoning and feedback

## Objective

Make the accepted SRWM state participate causally in Angler's action policy and
let one attempted solution plus scalar outcome update that same state.  This is
the minimum bridge from a plastic matrix to a learner; it must not encode a
task-specific solution method.

## Exact outputs

- `src/angler/reasoning/recurrent_core.py`: extract a public decoder method
  without changing ordinary-core behavior.
- `src/angler/reasoning/adaptive_core.py`: compose the recurrent core with one
  SRWM state, generic query/action/outcome event encoders, read-only action,
  and explicit post-outcome write.
- `src/angler/reasoning/__init__.py`: export the new public types.
- `tests/unit/reasoning/test_recurrent_core.py`: prove decoder extraction is
  behavior-preserving.
- `tests/unit/reasoning/test_adaptive_core.py`: causal and meta-gradient tests.
- this work leaf.

## Hard boundaries

The bridge receives detached observation features, the attempted public action,
and one bounded scalar reward.  It receives no family/task-type ID, hidden
answer, failed-fact identity, evaluator internals, router key, adapter name, or
replay.  It contains no rank, sorting, permutation, relation, symbolic-rule, or
other task solver.  Foundation-model parameters remain outside it and frozen.

An evaluation action is a pure state read.  Only an explicit feedback call may
write.  A feedback event encodes what Angler observed, what it attempted, and
the outcome it received; otherwise scalar feedback would not identify the
behavior being credited.

## Acceptance gate

The WSL2 reasoning suite must prove ordinary action output is unchanged by the
decoder refactor; read-only evaluation does not mutate state; feedback changes
the fixed-size state; the changed state alters a later fresh action policy;
reset erases and swap transfers that alteration; later-query loss trains the
earlier feedback-update dynamics; state snapshot/restore remains exact; and
the public API has no forbidden task-specific channel.

Passing proves only that feedback can alter a learned generic reasoning state.
It does not prove useful adaptation.  The successor mechanism-stream leaf must
meta-train and test improvement prequentially across changing operator/program
structures, without relying on same-generator heldouts.

## Result

The recurrent core now exposes a behavior-preserving encoded decoder.  The
adaptive wrapper reads one SRWM state for action modulation and writes only
after an explicit attempted-action/scalar-outcome event.  Focused reasoning
tests passed 23/23; the complete WSL learning/reasoning/runtime/world unit set
passed 64/64; and the host Evidence suite remained 16/16.  Causal tests show
state change, reset erasure, swap transfer, unchanged slow weights, and a
nonzero later-query meta-gradient through the earlier feedback write.  No
useful task improvement is claimed yet.

## Effects and rollback

Local source and synthetic tests only; no network, package, service, external
effect, personal data, deployment, or foundation-weight mutation.  Roll back
to Git commit `b25813e` if the bridge fails its gate.

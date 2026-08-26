---
blueprint_id: ANG-WORK-LEARNING-BIDIRECTIONAL-PROCEDURE-001
parent_id: ANG-BP-META-UPDATER
revision: 1
tier: 4
design_status: approved
delivery_status: complete
human_authority: project owner direction, 2026-08-26
human_impact: LOW; contained local synthetic learning research
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001@1-NOT_RUN
depends_on:
  - ANG-WORK-WORLDS-REVERSIBLE-TRANSITION-001
  - ANG-WORK-LEARNING-PREQUENTIAL-META-001
---

# Bidirectional procedural-learning core

## Objective

Replace the failed direct policy-conditioning experiment with an explicit
learned procedure constructor.  A frozen language model may propose one or
more candidate destination states.  Angler receives only an origin, a
candidate destination, and generic primitive actions; it learns forward and
backward transition models from executed experience, joins exact symbolic
frontiers, commits a primitive procedure, and accepts success only after the
real environment executes that procedure forward.

```text
public task -> frozen-model candidate goal
                         |
              origin + goal + primitives
                         |
       learned forward/backward constructor
                         |
              committed primitive procedure
                         |
           independent forward execution/check
```

The constructor cannot receive the task rule, hidden target, evaluator state,
program identity, solution trace, SRWM state, or verifier calls.  Generic
search control flow is permitted; task semantics must come from learned
transition predictions.  A decoded-state equality is required to join
frontiers, and only environment execution can certify arrival.

## Exact outputs

- `src/angler/reasoning/bidirectional_procedure_core.py`
- `tests/unit/reasoning/test_bidirectional_procedure_core.py`
- `experiments/evaluators/bidirectional_procedure_suite.py`
- `experiments/runners/phase3_bidirectional_procedure.py`
- `tests/unit/experiments/test_phase3_bidirectional_procedure.py`
- this work leaf

## Borrowed foundations

The clean-room design combines MIT-licensed Backward Learning reversed
transition training and backward rollout, HER future-state relabeling, and an
SGT-PG-inspired shared-midpoint auxiliary target.  The midpoint head is trained
but is not yet used by the planner, so no causal planning contribution is
claimed for it.  No donor source is copied and no new dependency is introduced.
Stitch-style verified macro mining is explicitly deferred until primitive
planning has a causal success.

## Training and gate

Every actually observed transition trains `F(s,a)->s'`,
`B(s',a)->s`, and inverse-action prediction.  Every observed trajectory also
supplies future-goal, distance, first/last-action, and midpoint examples; no
benchmark solver trace is generated.  Learned action heads restrict each
frontier state to its top two primitives, while task-blind bounded search joins
the resulting forward and backward frontiers.  The result reports transition
accuracy, exact independently executed procedure success on unique tasks,
forward-only versus bidirectional search under equal expansion ceilings,
untrained/corrupted-backward/permuted-policy controls, and a goal-swap
choke-point check.

A positive result proves only that learned reversible dynamics and learned
action guidance can be composed by generic bounded bidirectional search into
verified procedures.  It does not prove general reasoning, AGI, continual
retention, a learned search algorithm, or a correct goal proposer.  The
frozen-model and structured-goal results remain separately attributable.

## Proportionate impact assessment and rollback

Impact class `LOW`, disposition `ALLOW` by the project owner for the exact
isolated WSL2 synthetic run.  Qwen remains frozen and local.  No network,
package change, service, personal/recovered data, external effect, deployment,
or promoted-state mutation is permitted.  Failed results remain evidence and
are not patched with a task-specific solver.  Rollback returns to commit
`66f982a47a1fd406645f5b1acb7a0a991d82e481` and removes only these additive
successor outputs.

## Result

The final seed-9107 run trained a 753,499-parameter Angler core from 2,600
unique executed transitions and 17,320 automatically derived future-goal
examples.  All 1,000 transition edges excluded from every supervision head
were predicted exactly in both directions; inverse-action accuracy was 100%.
No structured evaluation pair or intended Qwen connector pair entered the
future-goal training set.

On 20 unique held-out origin/destination cases spanning exact distances
2/4/6/8/10, Angler's learned bidirectional constructor committed procedures
that the independent world executed successfully 20/20.  Equal-budget
forward-only construction also reached 20/20 but required 77.2 mean expansions
versus 41.4 for bidirectional construction.  An untrained core reached 0/20;
permuting backward action semantics reduced success to 11/20; redirecting the
learned top-two action choices through a permutation reduced it to 7/20.
Swapping only the proposed goal caused the executed endpoint to follow that
swapped goal rather than the original, confirming the architectural choke
point.

With local frozen Qwen3-4B supplying semantic destination proposals, Qwen
proposed the correct destination on 7/8 cases.  Angler independently reached
all 8/8 proposed destinations; end-to-end task success was therefore 7/8, with
the sole failure attributable to Qwen's wrong destination rather than Angler's
procedure construction.

Evidence:

- `work/experiments/bidirectional-procedure-final-full-9107.json` SHA-256
  `8AC50FD4E93999BAA23ABC093840EEA9ACA5DE443D52DD5EBA51D8E8EC63A222`
- `work/experiments/bidirectional-procedure-final-qwen-full-9107.json` SHA-256
  `0BB9486A39C8763BFE3BF2863571CC1187427DFC1856BA6813ADF91AC3B6D1A8`
- learned state checkpoint SHA-256
  `58EFEBCA8F9C5FCC1F9E3BA08943000B13FE594992C4B62921077967262CD546`

The 23 focused tests, 99 non-Evidence WSL unit tests, and 16 host Evidence
tests pass.  This establishes the bounded claim above, not continual ability
acquisition or cross-domain procedural abstraction.  The next leaf must make
this mechanism learn online across changing action algebras and then retain
and compose verified procedures without replaying all prior experience.

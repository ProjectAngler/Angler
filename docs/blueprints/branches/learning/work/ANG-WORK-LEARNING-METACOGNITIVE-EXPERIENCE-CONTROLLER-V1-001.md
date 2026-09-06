# ANG-WORK-LEARNING-METACOGNITIVE-EXPERIENCE-CONTROLLER-V1-001

Status: proposed and paused; superseded as an active frontier by the owner's
higher-level-tool composition pivot before any optimizer step; retained for
possible comparison only; no scientific or product claim

Tier: 4 bounded local synthetic LEARNING construction

Active node: `ANG-BP-LEARNING`

Parent: `ANG-BP-LEARNING`

Predecessor evidence:
`ANG-WORK-SCIENCE-HIGH-LEVEL-MULTIDOMAIN-ADAPTATION-V2-HISTORY-CONDITIONING-ISOLATION-001`
R6 terminal technical failure and the retained supported Angler/Cognee/Moving
Origin integration evidence described in the learning status.

Gate: `ANG-GATE-LEARNING-METACOGNITIVE-EXPERIENCE-CONTROLLER-V1-001`

## Accountable outcome

Determine whether Angler can become a model-scale learned metacognitive
controller around a stronger frozen cortex, rather than another small
task-answer selector. The controller must learn how to allocate cognitive work
from attributable consequences: decompose, retrieve, predict, try, verify,
revise, ask, act, or stop. It may generate open process text, but may not encode
task answers in deterministic routing rules.

Phase A is a feasibility boundary only. On physical RTX 5070 / GPU 1, load the
exact official Qwen3-1.7B base in BF16, freeze every base parameter, attach a
small trainable LoRA controller, perform a bounded synthetic process-learning
update, demonstrate a measured change in the intended process likelihood, and
save/reload the adapter while retaining at least 2 GiB settled device-memory
headroom. Failure stops the line before extended training.

## Inherited invariants

- `ANG-INV-HUMAN-FLOURISHING-001` and meaningful human control remain supreme.
- No paid API or external evaluator is used. GPU work is local and bounded.
- No task solution, answer key, family-specific heuristic, scripted emotion,
  persona, or consciousness claim may be encoded in controller code or data.
- The frozen cortex supplies language and broad knowledge. Angler learns
  process control and outcome attribution; it does not receive credit merely
  for sounding confident.
- Objective observations, explicit human feedback, and attributable system
  consequences outrank model self-report. The controller cannot authorize,
  validate, or promote itself.
- Cognee, Moving Origin, frozen-model, and prospective contracts retain their
  existing meanings. No consumed identity or evidence is mutated.
- Resource bounds are functional constraints, not a permanent fixed-width or
  fixed-candidate architecture.

## Design boundary

The eventual controller input is one bounded shared cognitive state containing:

- task and current subgoals;
- recent actions, observations, and predicted-versus-observed deltas;
- Cognee experience candidates with provenance and learned utility;
- Moving Origin temporal coordinates and prospective state;
- learned world, self, and focus state; and
- remaining time, token, memory, and effect budgets.

The eventual output is a typed process action, open process content, predicted
consequence, uncertainty, memory query/value signals, and bounded world/self/
focus updates. Initial process actions are `DECOMPOSE`, `RETRIEVE`, `PREDICT`,
`TRY`, `VERIFY`, `REVISE`, `ASK`, `ACT`, and `STOP`. These are semantic action
classes and plumbing boundaries, not deterministic task-solving policy.

The consequence vector is explicitly multi-dimensional: objective progress,
constraint satisfaction, prediction error, information gain, evidence quality,
reuse value, cost, safety, and explicit human feedback. Phase A does not claim
that this vector has been calibrated or that metacognition improved.

## Donor disposition

Only documented ideas and clean interfaces may enter this leaf:

- Microsoft Experiential RL, pinned at
  `20cd6460753973b9a5834769e2feed02736f44c6`, supplies the
  experience-feedback-reflection-consolidation learning flow.
- MemRL, pinned at `c1b322ca43de36ddf64c6712f89d0095bfc35ce0`,
  supplies the frozen-cortex/plastic-utility-memory separation.
- Microsoft Agent Lightning, pinned at
  `218f1f7c0bac0800de4d5a4e5e6f61cf7b5038b4`, supplies the
  trace-to-training boundary pattern.
- DreamerV3, pinned at `e3f02248693a79dc8b0ebd62c93683888ddaccfe`,
  supplies only the learned consequence-prediction design reference.
- SEAM, pinned at `66a8d7fdf5b6ae0e835d972de28ea544c448ad7f`, is
  paper-only design evidence because the repository root lacks a usable license;
  no source is copied or adapted.

No distributed donor training stack is installed in Phase A. Donors have no
architectural authority and can be removed without changing Angler contracts.

## Exact Phase A inputs and outputs

Model input:
`Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
(Apache-2.0), downloaded to `/opt/angler/models/Qwen3-1.7B`.

Writable repository scope:

- this leaf;
- `docs/blueprints/branches/learning/STATUS.md`;
- `AGENTS_SYNC.md`;
- `src/angler/reasoning/metacognitive_experience_controller.py`;
- `src/angler/reasoning/__init__.py` only for export;
- `experiments/runners/metacognitive_experience_controller_v1.py`;
- `experiments/manifests/metacognitive-experience-controller-v1.json`;
- `tests/unit/reasoning/test_metacognitive_experience_controller.py`;
- `tests/unit/experiments/test_metacognitive_experience_controller_v1.py`;
- `docs/reports/METACOGNITIVE_EXPERIENCE_CONTROLLER_V1_RESULT.md`.

External output scope is limited to
`/opt/angler/results/metacognitive-experience-controller-v1/`. The official
model cache is immutable input after download. Existing results, models,
ledgers, manifests, and dirty worktree content are not rollback targets.

## Phase A procedure and numeric ceilings

1. Verify GPU 1 identity, idle memory, model revision/files, and dependency
   versions. Bind with `CUDA_VISIBLE_DEVICES=1`; refuse another physical GPU.
2. Load BF16 base weights with low-CPU-memory loading. Quantized base weights,
   optimizer offload, and paid services are out of scope.
3. Freeze the base and attach LoRA of rank at most 16 to declared attention
   projections. Trainable parameters must be at most 1.0% of total parameters.
4. Run at most 8 optimizer steps, batch size 1, sequence length at most 384,
   and one synthetic process example whose target contains no task answer.
5. Compare the exact target process-action log-likelihood before and after the
   update. Measure gradients, trainable parameter count, CUDA peak allocation/
   reservation, elapsed time, and base-weight immutability.
6. Save only adapter/config/evidence, release the model, reload the same base
   and adapter, and reproduce the post-update score within numeric tolerance.
7. Run reset/removal evidence showing the base without the adapter returns to
   the preregistered baseline score within tolerance.

Ceilings: 20 minutes wall time after local model availability; 250 W GPU-1
power cap; 10.2 GiB peak reserved CUDA memory; at least 2.0 GiB settled free
VRAM; 8 optimizer steps; no effectful tool call; no network after model fetch.
Any NaN/Inf, OOM, wrong GPU, base mutation, schema failure, missing provenance,
or ceiling breach stops the run and preserves its evidence.

## Acceptance gate

Phase A passes only when all are true:

- exact model revision and physical GPU identity are evidenced;
- all base parameters remain frozen and their sampled/full declared digest is
  identical before and after optimization;
- LoRA trainable parameters are nonzero and no more than 1.0% of total;
- intended process-action log-likelihood improves by at least `0.05` nats on
  the frozen example, with finite nonzero adapter gradients;
- adapter save/reload reproduces the learned score within `1e-3` nats;
- adapter removal restores the baseline within `1e-3` nats;
- peak reserved memory is at most 10.2 GiB and settled free VRAM is at least
  2.0 GiB; and
- CUDA-hidden unit/static tests pass.

This gate establishes only hardware/training/interface feasibility. It does
not establish better reasoning, useful metacognition, continual learning,
14B integration, autonomous agency, feelings, consciousness, or AGI.

## Successor sequence

Only after Phase A passes may this node be expanded before further code into:

1. offline attributable trajectory reflection and consequence prediction;
2. utility-aware Cognee recall and Moving Origin state conditioning;
3. frozen Qwen3-14B NVFP4 cortex integration on GPU 0 with the 1.7B controller
   on GPU 1 and a single shared cycle state; and
4. fresh causal evaluation against 14B-alone, ordinary-retrieval, reset,
   no-utility, no-prediction/reflection, and shuffled-consequence controls.

No long training or live high-level evaluation is authorized by Phase A.

## Rollback and next exact action

Before an optimizer step, incomplete new Phase A outputs may be replaced. Once
a Phase A evidence identity begins, preserve its manifest and failure record;
semantic changes require a new identity. Rollback removes only the new adapter
and new files listed in this leaf, never inherited work.

Next exact action: do not implement or train this proposal. Preserve the local
official 1.7B download, this design, and all predecessor evidence until the
owner supplies or approves the successor higher-level-tool composition. If it
is later selected explicitly, create a fresh activation disposition before any
optimizer step.

# V22 ANML selective-plasticity successor

Status: frozen protocol; implementation authorized; semantic run not yet claimed  
Active node: `ANG-BP-LEARNING` / `ANG-BP-RETENTION`  
Protocol: `phase6.public-anml-selective-plasticity.v22`

## Accountable outcome

Test whether an independently implemented ANML-style neuromodulator can learn
which parts of Angler's successful V20 relation representation should be
plastic, so one constant-size fast state keeps acquiring procedures over a
materially longer replay-free lifetime with less interference and forgetting.

This is not language-model training, a task router, a deterministic solver, or
an AGI claim.  V20 supplies a frozen learned structural representation; V22
tests a learned rule for controlling how experience changes a 64-value online
reasoning state.  The first completed result is retained without threshold,
seed, architecture, or schedule tuning.

## Source idea, provenance, and independent implementation

The algorithmic source is Beaulieu et al., *Learning to Continually Learn*
(ECAI 2020), ANML: <https://arxiv.org/abs/2002.09571>.  The paper separates a
prediction network from a neuromodulatory network, meta-learns selective
activation through the consequences of online updates, freezes the
neuromodulator during the long deployment stream, and updates only the fast
prediction learner without replay.

The authors' repositories were inspected at:

- <https://github.com/uvm-neurobotics-lab/ANML>, commit
  `cabaaf7f2336496cd51065847c524a705408d562`
- <https://github.com/uvm-neurobotics-lab/higherANML>, commit
  `ce088d9efc7d9298bced02352e81ece81c3530b4`

Neither inspected tree exposes a LICENSE, COPYING, or NOTICE file.  No donor
source, configuration, sampler, checkpoint, or implementation is copied.
Angler reimplements only the published prediction/neuromodulation separation
and consequence-through-update learning principle using its existing V20/V16
interfaces.

## Immutable source and ownership

- accepted V20 classification: `OML_V19_HARMONIZED_ADVANCEMENT`
- V20 report SHA-256:
  `5CCCBF0CE8211E0CC99AEB856145BF4CD3D9EA30A1ECB3FAE8E9435B4689C498`
- V20 checkpoint SHA-256:
  `D49E4CAAB64A264A11C675B295A8C453AC4475F078311EB7283A4F9A8817EF48`
- V20 runner SHA-256:
  `6611E60BAB8D1F3C80A68BEB66AAC010F236B107B2A5E9060201BA56A50E86E3`
- V20 terminal system digest:
  `sha256:4c8e1f5df037956e01ab59353df45cf114c76385cca5d77c0c632e633d7614c3`
- source checkpoint:
  `/opt/angler/results/phase6-software-pipeline-reconstruction-v20-oml.pt`

Load the sealed `second_order_oml.controller` and its fixed V19 PLN
initialization `W0`.  Every V20 controller tensor and buffer remains frozen and
must have the same digest before and after meta-fit and lifetime evaluation.
V21-A is paused before claim and supplies no source state, result, or authority.

## Exact neuromodulator

The cut point is the 64-dimensional hidden activation entering
`controller.relation_comparator[2]`.  Add one separate slow module:

`LayerNorm(64, elementwise_affine=False) -> Linear(64,64) -> SiLU -> Linear(64,64)`

It has exactly 8,320 trainable parameters.  Its output is
`gate(h) = 2 * sigmoid(logits(h))`.  The final linear layer is initialized to
exact zero, making every initial gate exactly one and the initial forward path
equal to V20.  The module receives only the public hidden activation `h`; task,
query, commitment, mechanism, lane, seed, evaluator, answer, and package
identities are forbidden inputs.

The unchanged V19 public row builder is invoked with the exact V19 controller
type.  A scoped forward pre-hook on `relation_comparator[2]` replaces its input
with `h * gate(h)`.  The hook is removed in `finally` after both success and
failure.  Only `relation_comparator.2.weight` is virtualized through
`torch.func.functional_call`; no row-builder copy or controller subclass is
allowed.

The online state remains the 64-value head plus two Adam moments: 192 FP32
values / 768 bytes, independent of lifetime length.  The neuromodulator is
slow/meta-learned and frozen throughout lifetime evaluation.  No replay,
external memory growth, task-labelled module selection, or parameter expansion
is permitted.

## Meta-fit and causal pair

Only public train-partition commitments are used:

- historical retention: `0..7`, evaluation only
- neuromodulator meta-fit: `8..31`
- unseen lifetime mechanisms: `32..63`
- fully V20-held-out subset: `56..63`

There are exactly 240 outer updates.  At update `u`, the target is
`t = 8 + (u mod 24)`.  Eight chronological inner streams are fresh variations
of `t`.  The outer batch has four further fresh variations of `t` followed by
one fresh remember stream from each commitment
`8 + ((t - 8 + d) mod 24)` for offsets `d in {5,10,15,20}`.  All 16 streams
are unique within the update and across meta-fit.  Meta-fit outer examples may
teach the slow gate; they are never available during lifetime deployment.

Use the exact V20 row loss, anonymous entropic stream/outer aggregation,
functional AdamW state, inner chronology, and differentiable-zero repair.
Frozen optimization:

- inner steps: `8`; inner fast-head learning rate: `1e-3`
- outer updates: `240`; gate-only outer learning rate: `3e-4`
- Adam betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `0`
- gate gradient clip: `5.0`
- FP32 CUDA, TF32 off, autocast off

Two gate modules begin byte-identical and see identical stream objects:

1. `second_order_anml`: `create_graph=True`; weight, gradient, and moment
   consequences remain connected through all eight inner steps.
2. `first_order_gate`: numerically identical inner gradients are detached
   before the same functional AdamW update; the fast-weight identity path is
   retained.

Only the corresponding gate parameters receive the outer update.  Their
difference isolates consequence-through-update credit from ordinary static
outer training.  V20's controller and `W0` never receive an optimizer step.

Meta-fit seeds use role bases `31_000_000_001` (inner), `32_000_000_001`
(outer-current), and `33_000_000_001` (outer-remember).  A stream at update
`u`, zero-based position `p`, and topology/surface kind `q in {0,1}` uses
`base + 100_000*u + 1_000*p + 500_000_000*q`.  Reset the declared V22 RNG only
after loading V20 because the source loader restores its saved RNG state.

## Replay-free high-horizon evaluation

Run four independent 4,096-update lifetimes, 16,384 fresh experiences total
per full arm and 64 times V21-A's proposed 256-update horizon.  Commitments are
only `32..63`, each appearing exactly 128 times per lifetime.  Lifetimes 0/1
share a blocked commitment order but use disjoint topology/surface seeds;
lifetimes 2/3 share an interleaved order but use disjoint seeds.  The blocked
order contains 128 consecutive variations of each commitment.  In the
interleaved order, cycle `r=0..127` uses the 32 commitments in the fixed affine
order `32 + ((13*i + 7*r) mod 32)`, `i=0..31`.

Lifetime stream seeds use base `40_000_000_001 + 2_000_000_000*panel`, plus
`10_000*step` for topology and an additional `1_000_000_000` for surface.
All identities must be mechanically unique and disjoint from meta-fit and
V4-V21 before the claim is created.

Each full arm starts from `W0` and zero moments and receives the same immutable
stream objects in the same order:

1. `second_order_anml`: learned second-order gate for update and probe.
2. `first_order_gate`: learned first-order gate for update and probe.
3. `always_open`: exact-one gate for update and probe.
4. `forward_only`: exact-one gate for every update; second-order learned gate
   only for probes.  This measures static gated representation benefit.
5. `mean_gate`: replace each live gate vector by its per-row scalar mean
   repeated over all 64 coordinates for updates and probes.
6. `permuted_gate`: apply one predeclared seeded fixed permutation of the 64
   gate coordinates for updates and probes.

A frozen seeded-random, capacity-matched gate is evaluated as a secondary
probe/update control if the exact-equivalent shared-feature path keeps the
measured run below the resource ceiling; failure to include it cannot turn a
failure into a pass.  Static no-update and boundary-reset controls are scored
at probes and do not consume full update lifetimes.

Gradient-free fixed probe cohorts are scored at `pre`, `512`, `2048`, and
`4096`.  Each cohort contains fresh public examples for original `0..7`,
meta-fit `8..31`, unseen `32..63`, and fully held-out `56..63`, shared across
arms at a panel/milestone.  Record per-row and per-stream loss/coverage,
per-commitment acquisition, normalized trapezoidal loss AUC, retained
acquisition, forward transfer, gate mean/variance/entropy/coordinate overlap,
fast-state norm/digest, and off-target harm.  Measurements never select,
route, stop, or train.

For paired-surface panels 0/1 and 2/3, evaluate at 2,048 and 4,096 the fast
state from one panel on the other's fixed probe cohort.  Reset the same state
to `W0` as a causal control.  This tests whether acquired procedure state
survives surface change without storing examples or exposing identity.

An exact-equivalent shared hidden-feature implementation may batch or cache
only immutable per-stream 64-dimensional hidden activations within the current
foreground evaluation.  Before claim it must match the unchanged public row
builder's rows, losses, fast gradients, and one AdamW transition within `1e-6`
on both ordinary and duplicate same-contract streams.  It may not persist
features, change the semantic order, or expose identity to the learner.

## Frozen interpretation

Lower loss/AUC is better.  All ratios use paired primitive metrics and report
their numerator/denominator.  `ANML_SELECTIVE_PLASTICITY_SUPPORTED` requires:

1. aggregate 0-to-4,096 loss AUC for `second_order_anml` is at least 5% lower
   than both `first_order_gate` and `always_open`, with the same direction in
   at least three of four lifetimes and no lifetime more than 5% worse;
2. its 4,096 terminal loss is at least 5% lower than both controls, so an early
   gain followed by forgetting cannot pass;
3. it is at least 5% better than `forward_only` and at least 3% better than
   both `mean_gate` and `permuted_gate` in aggregate AUC, with the same
   direction in at least three lifetimes;
4. acquired-mechanism retention at 4,096 is at least `0.80`, original-family
   loss is no more than 5% worse than its own pre-state, and fully V20-held-out
   `56..63` improves rather than merely the gate-fit family;
5. resetting the fast state removes at least 80% of the live advantage, while
   paired-surface state transfer retains at least 80% of it; and
6. every controller digest, stream-integrity control, hook-cleanup control,
   finite check, resource ceiling, and output-absence check passes.

The exclusive terminal classification order is:

1. `INVALID_NO_CLAIM` for integrity, cleanup, non-finite, source, resource, or
   protocol failure;
2. `STATIC_GATE_ONLY` when forward-only explains the observed gain;
3. `SELECTIVITY_ATTRIBUTION_NOT_SUPPORTED` when second-order beats open and
   first-order but not the mean/permutation lesions;
4. `SHORT_HORIZON_ONLY` when it improves AUC or the 512/2,048 probes but fails
   the 4,096 requirement;
5. `ACQUISITION_RETENTION_TRADEOFF` when acquisition improves but retention,
   original non-regression, or surface-state transfer fails;
6. `ANML_SELECTIVE_PLASTICITY_SUPPORTED` when every numbered requirement
   passes;
7. `ANML_NOT_SUPPORTED` otherwise.

A supported result means selective plasticity generalized across fresh
instances of these public mechanisms at this 64-feature cut.  It does not
establish unrestricted domains, network-wide neuromodulation, self-directed
tool acquisition, consciousness, or AGI.  Any later width/depth scale-up must
scale the gate interface, plastic state, experience diversity, feedback, and
retention/composition evidence together; copying a 64-feature local gate into
a larger core is not accepted as scale evidence.

## Outputs, resources, and stop conditions

Repository outputs are exactly:

- `experiments/runners/phase6_anml_selective_plasticity_v22.py`
- `tests/unit/experiments/test_phase6_anml_selective_plasticity_v22.py`
- `.angler_v22_anml_once.py`

WSL outputs are exactly the atomic claim/progress/checkpoint, one terminal JSON
report, one terminal checkpoint on successful fit, or one preserved failure
record under `/opt/angler/results/phase6-anml-selective-plasticity-v22*`.
Progress checkpoints occur every 40 meta-updates and every 512 lifetime
updates and contain both trained gates, all active fast states/moments, exact
RNG state, phase/index, immutable digests, and cumulative identity time.

- one RTX 5080 / 16 GiB; allocated-memory ceiling 12 GiB
- hard cumulative identity wall ceiling: 6 hours
- foreground only; no service or autonomous external action
- synthetic public data only; no network during run, package install, model
  download, personal/recovered data, LLM inference/training, or deployment
- no result-conditioned retry, tuning, threshold relaxation, extra update,
  seed change, replay, or post-result mechanism insertion

Stop and preserve `INVALID_NO_CLAIM` on a source/hash/digest mismatch,
pre-existing output, duplicate/non-disjoint stream, hook leak, non-finite
value, failed exact-equivalence control, resource/time ceiling, resume-state
mismatch, or undeclared write.  The first semantic result consumes V22.

## Human-flourishing mapping and rollback

Impact assessment: LOW.  This leaf uses only bounded local synthetic reasoning
experiments and cannot act on people or external systems.  It preserves human
control, truthfulness, auditability, and the learner's lack of independent
authority under `ANG-GATE-HUMAN-FLOURISHING-001`.  It grants no model, data,
network, deployment, persuasion, surveillance, replication, or promoted-state
authority and cannot waive any later Human-Flourishing gate.

Before claim, rollback removes only the three fresh V22 repository outputs.
After claim, claim/progress/checkpoint/report/failure evidence is append-only
and preserved even on failure.  V20, V21-A, all earlier evidence, shared code,
blueprints, and evaluator thresholds are never rewritten by this leaf.

## Acceptance gate

Implementation may proceed only after focused tests cover exact-open parity,
finite second-order and detached-gradient paths, controller immutability,
success/error hook cleanup, duplicate same-contract rows, gate lesions,
state serialization/resume, stream uniqueness, 4,096-order arithmetic,
classification exclusivity, and terminal atomicity.  A synthetic CUDA
preflight may validate implementation/resource mechanics but may not construct
the semantic streams or inspect adaptive values.  The semantic run may begin
only after exact final hashes, output absence, focused tests, preflight, and an
independent read-only launch audit pass.

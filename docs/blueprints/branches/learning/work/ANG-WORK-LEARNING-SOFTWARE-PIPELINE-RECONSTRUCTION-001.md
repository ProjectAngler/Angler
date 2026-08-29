---
blueprint_id: ANG-WORK-LEARNING-SOFTWARE-PIPELINE-RECONSTRUCTION-001
parent_id: ANG-BP-SKILL-COMPOSITION
revision: 1
tier: 4
design_status: approved
delivery_status: in_progress
human_authority: project owner continuation direction, 2026-08-27
human_impact: LOW; contained local synthetic software reconstruction
depends_on:
  - ANG-WORK-LEARNING-GLYPH-TRACE-001@1
  - ANG-WORK-LEARNING-SKILL-LOCAL-MEMORY-001@1
---

# Software-pipeline reconstruction precursor

## Objective

Make Angler reconstruct a missing executable pipeline in unfamiliar synthetic
Python micro-packages.  Each package exposes public typed component contracts,
public structural records, type-compatible distractors, a required input/output
contract, and observations from separate sibling packages.  Angler commits one
bounded sequence of grounded component applications plus `STOP`.  An
evaluator-owned integration batch executes the committed pipeline once and
returns only terminal `0.0` or `1.0`.

This is the next precursor to repository repair.  It tests whether the learned
memory and recurrent backward reasoner can acquire and compose software-like
procedures across new identities.  It does not generate free-text patches,
operate on a real repository, establish broad reasoning, or establish AGI.

## Why this leaf follows Glyph

Glyph v3.2 proved learned associative routing and neural multi-step
composition, but its development supports covered a mean `77.43%` of the
state/action edges and `119/128` queries were solvable using only observed
edges.  Scaling that exact-address setting would risk building a larger
episodic transition-table reasoner.

This leaf therefore requires zero support/query overlap in state identity,
component identity, grounded action, and exact pair address.  Every support and
query package uses fresh opaque names and digests.  Query compositions combine
two motifs that appeared separately but never together in evidence.

## Exact increment

- `experiments/evaluators/software_pipeline_reconstruction_suite.py`
- `experiments/runners/phase6_software_pipeline_reconstruction.py`
- `tests/unit/experiments/test_software_pipeline_reconstruction_suite.py`
- `tests/unit/experiments/test_phase6_software_pipeline_reconstruction.py`
- `outputs/phase6-software-pipeline-reconstruction-result-summary.json` only
  after a completed fixed evaluation

Reuse, without changing the completed Glyph result:

- typed records and action contracts from `src/angler/procedures/records.py`;
- `GlyphAssociativeMemory`, transactional state, scalar feedback,
  checkpointing patterns, and `GlyphBackwardProcedureReasoner` from the
  completed v3.2 runner;
- permutation-invariant record encoding and pointer-style candidate scoring
  from `src/angler/procedures/trunk.py`;
- `ConditionalReversibleTransition` from the skill-memory runner; and
- the private evaluator/commit/control-arm boundary from the Glyph evaluator.

Do not use `BidirectionalOperatorPlanner`, `search_teacher_plan`, deterministic
graph isomorphism, MDL operator induction, BFS, DFS, dynamic programming,
complete-pipeline enumeration, or a motif-class answer head.

## Public and hidden contracts

The learner receives a `PublicSoftwarePipelineTask` containing:

- an unordered tuple of public component schemas and grounded candidates;
- typed input/output/error/state contracts and public incidence records;
- an origin record state and an exact required output contract;
- zero or more public execution-observation traces; and
- a one-to-four-step public budget.

The evaluator retains component implementations, hidden integration inputs,
the reference motif construction, hidden failure location, partition identity,
and any reference pipeline.  Equivalent pipelines are accepted.  A committed
pipeline is immutable before the evaluator runs it.  The learner cannot query
the evaluator during rollout, execute hidden tests, or receive a reached
intermediate, failed assertion, target component, or distance signal.

Public action selection is factorized over `APPLY(component)` and `STOP`.
Grounded component candidates are declared for the current package; they are
not complete patch candidates or sequences.  Future patch realization may use
a frozen LLM only after this learned procedural boundary passes.

## Learned architecture

`SoftwareReconstructionState` contains two bounded device-resident lanes:

1. A pointer-fidelity lane retains v3.2 exact addresses for observations inside
   one package.  Because package identities never overlap, this lane cannot
   transfer a query procedure.
2. A procedure-role lane uses learned permutation-equivariant keys and
   successor-role values derived only from public component contracts, record
   incidence, state/goal structure, and observed outcomes.  It must learn
   rename-invariant equivalence across packages.

The identity lane may resolve local entities but cannot dominate the role
lane.  The role encoder may not consume task IDs, digests, evaluator labels,
hidden topology, canonical graph matches, or reference procedures.  Metadata
poisoning must leave inference logits bit-exact.

The existing recurrent backward reasoner consumes the learned role transition
lattice and emits one component or `STOP` at each step.  Public completed
attempts may write their public trajectory and one scalar terminal outcome.
There is one persistent slow lineage and one bounded fast state; no query-
conditioned model selection or replay of complete history is permitted.

## Motifs, partitions, and scale

Initial reusable repair motifs include missing data flow, incorrect execution
order, wrong branch routing, stale-state invalidation, and lost error
propagation.  Each motif appears repeatedly across unrelated package graphs.
Composition queries require two independently demonstrated motifs in one
pipeline.

- Smoke: 8 train / 4 development mechanisms, 3--9 states, 3--6 components,
  maximum 4 steps.  The first composed query uses two two-stage motifs, so the
  smoke budget must preserve all four causal decisions.
- Resource: 64 train / 16 development / 16 final mechanisms, 4--8 states,
  4--6 components, maximum 4 steps, eight distinct support packages and eight
  query packages per mechanism.
- Capacity pilot: keep the proven `width=512` approximately 26-million-
  parameter core until cross-package development passes.  Then run one
  approximately 50-million-parameter development pilot.  A 100-million-class
  profile is permitted only if the pilot shows a capacity trend rather than an
  addressing or representation failure.

FP32 live reasoning state remains GPU-resident.  Abort any widened run above
12 GiB allocated or 13 GiB reserved on the 16.3-GiB RTX 5080.  Do not use CPU
offload for the online reasoning state.

## Acceptance and falsification

Before opening final, a fixed development run across at least three seeds and
six held-out motif combinations must show:

- full-evidence exact hidden-test success at least `50%`;
- at least `+15` percentage points over no evidence, wrong evidence, and the
  strongest equal-budget episodic-retrieval control;
- at least `+10` points over the best A-only or B-only state on A+B queries;
- positive composition effect on at least five of six motif combinations;
- removing procedure-role memory erases at least half the acquired gain;
- removing backward reasoning erases at least half the composition gain;
- a semantically equivalent state swap preserves at least `80%` of the gain;
- alpha-renaming or independent rerendering changes success by at most five
  points; and
- old-motif loss after acquiring the next motif remains within five points.

Every query invokes the hidden judge exactly once.  Report all mechanism-level
distributions.  Correct, no-evidence, wrong-evidence, pointer-only, shuffled-
outcome, A-only, B-only, role-memory-removed, and backward-reasoning-removed
arms start from matched state and compute.

Tests must prove deterministic generation; partition disjointness; fresh
support/query identities; exact zero pair-address overlap; alpha-renaming and
presentation-order invariance; dynamic shapes; declared-component-only
rollout; budget and `STOP`; exact snapshot/restore and checkpoint reload; one
scalar per committed pipeline; and AST/dependency-closure absence of hidden
reads and search/solver APIs.  Permuting stored values must make behavior
follow evidence, never public metadata.

Failure changes the learned representation, role memory, or reasoner.  It must
not be patched with a deterministic pipeline solver, reference-patch label, or
larger model before the causal bottleneck is localized.

## Pre-training adversarial findings

Two evaluator drafts were rejected before any resource or sealed run:

- the first exposed motif/stage/twin identity through incidence counts, so a
  public count-plus-type-chain rule scored `128/128` development queries; and
- equalizing those counts was insufficient because support and query packages
  repeated the same anonymous graph signatures, allowing exact WL-signature
  retrieval plus type chaining to score `128/128` again; and
- a first fresh-graph transform draft still leaked transform polarity through
  candidate-only walk/degree graphlets, letting a public k-nearest-neighbor
  control score `82.23%` where composed binary-twin chance is `25%`.

Neither result is Angler evidence.  The successor evaluator must give every
support and query package different absolute graph signatures, balance each
candidate's marginal graph distribution across transform polarity, and make
the transferable object only a joint predecessor-to-candidate relation.  The learner may consume raw
public adjacency/equality through learned equivariant message passing, but it
may not contain a graph canonicalizer, transform detector, motif rule, or
pipeline constructor.  A-only/B-only controls must be evaluator-owned and
motif-pure; episodic retrieval must be measured as its own baseline rather
than aliased to a neural ablation.

## Audited smoke diagnosis and learned repair

The repaired evaluator passed the independent shortcut audit, including
exchangeable candidate-only marginals and a perfect evaluator-side joint-
relation oracle.  A bounded ten-epoch learned smoke then reduced its public
training loss from `1.0690` to `0.08136` while remaining `0/8` on fresh
development queries.  Diagnostic probes found the earliest failure before
memory or planning: the learned predecessor/candidate relation codes for the
two opposing transforms had cosine similarity approximately `1.0`, and their
role-memory attention margin was approximately zero.  Teacher-forced
successor prediction was top-one only `25%`; greedy rollouts repeated one
stage-zero action and then stopped.  More epochs, data, width, or GPU use are
therefore not an authorized remedy for this failure.

The next implementation increment is a generic learned representation repair:

- replace the lossy final node mean/max multiplex pool with a permutation-
  equivariant ordered-node-pair message-passing encoder over the two raw public
  adjacency channels, with no signature, canonicalization, transform label, or
  motif rule;
- key procedure memory by learned local public state-to-component relations,
  not by a globally contextualized absolute state embedding;
- store and compare learned public transition-effect relations between
  `before` and `after`, rather than anchoring a support package's absolute
  successor-state embedding into a query package;
- supervise relation separation and sibling-package transfer only from public
  observed transitions, including a leave-one-package-out acquisition path;
  and
- condition `STOP` on a learned local current-to-goal relation rather than the
  demonstrated support length.

Tests must establish alpha invariance, correct-versus-wrong relation
separation after learning, sibling-package memory preference, relative-effect
successor transfer, and four-step composition without exposing a deterministic
pipeline construction path.  The resource profile and final partition remain
blocked until this repaired smoke transfers and its causal controls pass.

The first repair implementation passed `60/60` focused evaluator/runner tests.
Its public-only leave-one-package-out smoke reduced loss from `1.390604` to
`0.139310` and learned high-confidence fresh-sibling successor effects, but the
four-mechanism composed development check remained `0/4`; wrong evidence was
`2/4`.  This is not acceptance evidence.  It localizes the next investigation
to multi-support fast-memory key/value binding and interference.  Freeze scale
and final access while testing slot attention, value geometry, and an isolated
correct-sibling memory control; do not widen the model or add a deterministic
selection rule.

The subsequent operator-preserving key, dense-null read, and public retrieval-
margin repair opened fresh correct-versus-wrong margins in the operator, role-
key, and evidence spaces without using motif labels.  End-to-end composition
nevertheless exposed an evaluator mismatch: the hidden judge accepts all six
interleavings of two independent two-step motifs, but the public query declared
only the five states on one serialized reference path.  Five judge-valid paths
therefore contained an unrepresentable public intermediate, and early single-
reference action accuracy was not a valid composition metric.

Before any new training signal, two-motif queries must declare the complete
variant-blind `3 x 3` progress lattice (nine states); single-motif supports stay
at three.  The state set is derived from public read/write incidence only and
must be identical for the two hidden completion twins.  It grants no learner-
side transition constructor, applicability mask, sequence enumeration, or
solver.  Tests must prove that every judge-valid interleaving has declared
intermediates, invalid stage order still fails, both twins have identical
public effects, query observations remain empty, and shortcut controls remain
at chance.

The repaired lattice passed those tests and made all `16/16` selected public
transitions applicable with the correct declared successor as top one.  It did
not establish evidence-conditioned reasoning.  A ten-epoch smoke scored `2/4`
with correct evidence and the same `2/4` with no evidence.  A subsequent
class-local causal calibration scored `1/4` versus `2/4` without role memory;
its mean role-on improvement was only `0.0000305` nat against the frozen `0.05`
target.  A same-forward diagnostic that detached only the production-action
gradient restored `2/4`, but every evidence and ablation arm also scored
`2/4`.  Individual-support and all-support public LOPO probes retained the
same chance twin sign, while combined memory retained approximately `96%` of
the best positive single-support magnitude.  These results falsify a larger
gate, more head calibration, joint-credit scheduling, and aggregation loss as
the next remedy.  The earliest remaining failure is transferable relation
identity before scalar evidence scoring.

The bounded successor exposes the frozen ordered-node-pair tensor before its
existing lossy pool and learns a separate permutation-invariant evidence pool
and symmetric relation comparator.  Observed relation codes occupy one tensor
aligned to the existing bounded role-trace slots; the proven transition,
successor, backward-reasoning, and STOP paths remain byte-preserved.  Training
uses matched public LOPO arms: original support observations select one
completion, the public wrong-evidence projection selects its effect-equivalent
twin, and the masked held-out task is otherwise byte-identical.  Both arm
orientations are balanced and jointly optimized, so an evidence-ignoring,
absolute-index, or presentation-order policy cannot satisfy the objective.
No motif, stage, variant, hidden solution, graph signature, canonicalization,
or deterministic inference rule enters this path.

Before another development outcome is observed, the relation-only learner
must pass a fresh public-train gate of 96 paired LOPO rows: mean original margin
at least `+0.10`, mean swapped margin at most `-0.10`, at least `77/96` rows
with both signs correct, and mean paired separation at least `0.20`.  Alpha
rerendering and state-equivalent rerendering must retain at least `80%` of the
separation and `90%` sign agreement; candidate/support order must be covariant;
empty/no-role evidence must be exactly zero; and all pre-existing parameters,
no-memory logits, transitions, successors, and STOP outputs must remain exact.
Failure stops this representation rather than triggering scale, extra epochs,
or threshold changes.

The frozen 64-step one-pass schedule failed that gate cleanly.  Across its 96
fresh rows, mean original and swapped margins were `+0.001870` and `-0.002594`,
mean separation was `0.004464`, and `0/96` rows reached both signed targets.
Permutation covariance, alpha rerendering, empty-memory behavior, and every
frozen legacy output remained exact.  No action calibration, development,
final, judge, checkpoint, or resource run followed the failure.

A bounded train-only capacity diagnostic then distinguished a dead
representation from an inadequate exposure schedule.  Repeating one four-row
public stream made the unchanged matcher cross all four signed targets by step
25; at step 100 its mean original/swapped margins were `+0.136366` and
`-0.137645` with separation `0.274011`.  The learned relation transferred
unchanged to an alpha rerender and to a fresh topology for the same opaque
commitment (`4/4` signed rows in both).  Twenty-five updates that each averaged
four fresh topology/surface streams also transferred to a fresh topology with
mean separation `0.297790` and `4/4` signed rows.  Thus the current matcher and
gradient path can learn a rename-invariant relation; the failed production
schedule presented each changing stream only once and left cross-commitment
acquisition or interference unresolved.

Before changing the encoder, run one finite rehearsal-free retention
diagnostic: learn a balanced varied batch from eight train commitments, measure
fresh-topology acquisition, then learn eight different train commitments with
zero replay of the first set and remeasure both sets.  This is a bounded test of
whether varied batching teaches one reusable procedure and whether later
learning erases it; it is not authorization for indefinite replay.  Keep the
development, final, judge, resource, and action-calibration boundaries closed.

That diagnostic ruled out forgetting as the primary failure.  Twenty-five
balanced updates over eight commitments produced mean separation `0.028643`
and only `4/32` signed rows.  A second 25-update stage learned eight different
commitments with zero replay of the first set.  It raised the first, second, and
unexposed third groups to separations `0.086175`, `0.065112`, and `0.022748`,
but none passed and stable commitment clusters retained reversed polarity.
Later learning therefore strengthened a partially shared direction rather
than erasing the first group.  The current single relation code plus global
`logsumexp` lacks enough public conditioning to distinguish the one relevant
same-motif demonstration from two other-motif demonstrations when transform
polarity varies across commitments.

The next representation increment must factor public context from public
relation.  A learned context head identifies which anonymous predecessor graph
is relevant; a separate learned relation head represents the directed
predecessor-to-candidate transform.  Retrieval first weights slots by context,
then compares relation values.  Both heads consume only the existing public
ordered-pair tensors and remain learned end to end; they receive no motif,
variant, commitment, signature, canonical order, or solver input.  Do not
unfreeze the shared role/reasoning encoder until a bounded factorized probe has
shown that frozen pair states cannot support joint fresh-topology transfer.

The factorized implementation passed `104/104` focused evaluator/runner tests
and preserved aligned atomic state, rollback, snapshot, digest, checkpoint,
successor, reasoning, STOP, and no-memory behavior.  Its fixed eight-commitment
probe nevertheless failed: after 25 balanced updates over 200 fresh streams,
primary fresh mean original/swapped margins were `+0.006860` and `-0.016980`,
mean separation was `0.023840`, and `0/32` rows reached both signs.  A second
fresh topology reached `4/32`; only one of eight commitments transferred.
Twin query contexts were byte-identical, all 96 positive/wrong context states
were identical, and all 96 paired relation-value states differed as intended.
This falsifies missing slot binding and frozen-H factorization alone.  The
remaining supported repair is a separate evidence-specific ordered-pair
encoder trained by the public paired objective; the shared role/reasoning
encoder remains frozen.

That dedicated four-round encoder passed `106/106` focused tests; all 52 trunk
tensors received finite nonzero public-paired gradients while every shared
role/reasoning parameter remained byte-exact.  A predeclared 100-update probe
then processed 800 fresh streams (3,200 rows) with separate encoder/head
learning rates.  It also failed without tuning: primary fresh margins were
`+0.010654` and `-0.019007`, separation `0.029661`, and `4/32` signed rows;
a second fresh topology reached separation `0.037840` and the same `4/32`.
One commitment learned and transferred strongly, two retained strongly reversed
signs, and the others remained weak.  Loss fell from `0.200011` to `0.173185`
and all 68 relation tensors changed.  This rules out a dead gradient path,
surface sensitivity, frozen representation alone, and merely extending the
same episodic objective.

Before another optimizer change, diagnose credit at the raw slot level.  For
each public paired row, measure each stored slot's pre-temperature,
pre-null/softplus relation witness, context mass on the strongest witness,
twin-code distance, context-only versus relation-only trunk gradients and their
cosine, and parameter-update/weight ratio.  Strong raw witnesses with weak
context mass justify a direct public multiple-instance relevance objective;
near-zero witnesses justify relation-representation work; materially opposing
context/relation gradients justify separate towers.  Do not choose among these
repairs from aggregate end-to-end accuracy alone.

The exact 25-update diagnostic found useful sparse evidence but severe credit
dilution.  Best per-slot witness averaged `0.031458` and was positive on `30/32`
rows, while the all-slot mean was `0.001772`.  The selector placed only
`0.153106` mass on the best slot with `0.118370` on null, effectively uniform
over six occupied slots.  Weighted raw and final separations collapsed to
`0.001637` and `0.001290`.  Relation-code twin distance was nonzero
(`0.081701`).  Relation-live trunk gradient norm was `0.0305724`; context-live
gradient norm was `0.00002368`, a roughly `1,291x` starvation ratio.  Across
commitments, `42.86%` of gradient pairs were negative; commitments 0/2 aligned
at cosine `+0.9421` while each opposed commitment 3 near `-0.95`.  The next
repair is therefore direct pre-aggregation public multiple-instance credit for
context relevance plus isolated context/relation trunks.  More epochs or a
larger undifferentiated encoder are not supported.

External technical red-team review challenged that repair at the correct
boundary.  Its valid challenge is to test commitments with coherent reversed
gradients in isolation before changing architecture or credit.  Its proposed
runtime `logsumexp` over paired witnesses is not admissible because a witness
requires both selected and counterfactual wrong-evidence arms, which exist only
during public training.  Likewise, per-commitment inference heads would defeat
unseen-commitment transfer, and the claim that the twins differ only by trace
order does not match this evaluator: their public discriminator is the directed
predecessor-to-candidate graph transform.  Witness aggregation remains eligible
as an auxiliary public loss only.  Pause the split-tower/MIL implementation and
first run fixed isolation probes on the coherent reversed cluster and the
successfully learned control commitment.

The fixed isolation study then resolved that challenge decisively.  Each of
commitments 0, 2, and 3 was trained independently for exactly 100 updates from
fresh public streams under three fixed initialization seeds.  All `9/9` runs
passed on 32 disjoint fresh streams per run: `1,150/1,152` rows had both signs
correct, mean original/swapped margins were approximately `+0.1853` and
`-0.1855`, and mean separation was `0.3708`.  Every run retained exact public
and evidence-order covariance and exact witness-slot transport.  Commitment
identity selected only the external diagnostic episode set and report row; it
never entered model inputs, parameters, or inference.  No code, checkpoint,
development partition, judge, or result artifact was changed or opened.

Thus the reversed joint results are not evidence that those procedures are
unlearnable or that the public benchmark is incompatible with the encoder.
The same representation learns and transfers each tested procedure robustly
when trained alone.  Combined with the nearly antiparallel cross-commitment
gradients and starved context selector, the supported defect is specifically
multi-procedure credit assignment and interference.

The bounded joint-credit successor uses only ordinary public traces.  For each
masked held-out transition, its observed component is contrasted with the one
other declared component having the same public static contract.  Each other
support contributes its observed relation and, when available, the analogous
declared alternative as a training-only contrast.  No wrong-evidence control
stream or counterfactual state enters runtime.  A bounded symmetric comparator
first learns a multiple-instance witness over relation slots; an independent
context tower then learns a detached soft responsibility over those slots; the
two paths are finally optimized together through the ordinary aggregate score.

The first fixed probe is 100 updates over eight opaque train commitments and
800 fresh streams: relation witness updates 1--40, context-credit updates
41--65, and joint updates 66--100.  Encoder learning rate is `3e-4`, head
learning rate is `1e-3`, AdamW weight decay is zero, gradient clipping is `5`,
and there is no extension after viewing an outcome.  A disjoint 32-row public
panel must pass after the relation stage before context training proceeds; a
second disjoint panel is reserved for the final joint check.  The final check
requires mean positive/negative margins at least `+0.10`/at most `-0.10`, mean
separation at least `0.20`, at least `26/32` signed rows, at least seven of
eight streams with three of four signed rows, context top-one agreement at
least `80%`, target-slot mass at least `0.60`, covariance within `1e-6` on
each declared order axis, exact empty-memory zero, and byte-exact shared
role/reasoning parameters.

The first relation-stage diagnostic stopped at its mandatory boundary.  Across
40 updates, 320 fresh streams, and 1,280 rows, the new public objective reduced
loss from `0.225863` to `0.074155` and produced strong slot witnesses: mean
best positive/alternative margins were `+0.295881` and `-0.298409`, mean
witness separation was `0.594290`, and mean best-versus-second gap was
`0.624315`.  It nevertheless reached only `22/32` confident rows and five of
eight streams with at least three confident rows, below the frozen `24/32` and
six-of-eight boundary.  Context and joint stages therefore did not run.  Every
non-relation parameter remained exact; there were no control-stream, wrong-
evidence, development, final, judge, checkpoint, or artifact effects.

Audit then found that the diagnostic reporter chose its target slot by maximum
raw separation while the learner assigned responsibility by minimum full
sign-balanced loss.  Those choices need not identify the same slot.  The
stopped numbers are therefore useful localization evidence but cannot activate
the next stage or validate the reporter.  A successor must align training and
reporting to the same loss-defined slot, preserve raw maximum-separation as a
separate diagnostic, retain per-stream/per-row failure details, and expose one
enforcing staged orchestrator.  It must also replace or explicitly retire the
still-exported v2 wrong-evidence fit path before claiming protocol v3 and bind
the changed fast-state semantics to a new digest identity.

The fixed successor is `phase6.public-relation-credit.v4`.  It is a clean
restart at torch initialization seed `2026082821`; it does not load or continue
the stopped diagnostic state.  Exactly the first eight opaque train
commitments appear once per update.  Update `u` and anonymous commitment
position `c` use topology seed `221000001 + 100000*u + 1000*c` and surface seed
`321000001 + 100000*u + 1000*c`, with `u=0..99` and `c=0..7`.  Relation uses
updates `0..39`, context uses `40..64`, and joint uses `65..99`.  These 800
seed pairs are unique and are disjoint from both fixed panels.

The relation/context panel uses one stream per commitment at topology seeds
`421000001 + 1000*c` and surface seeds `431000001 + 1000*c`.  Relation may
advance only with at least `24/32` loss-selected confident rows and six of eight
streams containing at least three confident rows.  A confident row has
loss-selected positive margin at least `+0.05`, alternative margin at most
`-0.05`, witness at least `0.10`, and best-versus-second slot-loss gap at least
`0.02`.  Context may advance only if the same frozen panel still passes that
relation boundary, context top-one agreement is at least `0.80`, and mean mass
on the loss-selected slot is at least `0.60`.

The final joint panel uses topology seeds `521000001 + 1000*c` and surface
seeds `531000001 + 1000*c`.  It passes only with aggregate mean positive and
alternative margins at least `+0.10` and at most `-0.10`, mean separation at
least `0.20`, at least `26/32` signed rows, seven of eight streams with at least
three signed rows, context top-one agreement at least `0.80`, mean target mass
at least `0.60`, covariance within `1e-6` independently for evidence order,
public presentation, and their combination, exact empty-memory zero, and
byte-exact shared role/reasoning parameters.  Every occupied runtime relation
slot is retained per row: three transferable slots, one from each unmasked
support package, with its positive margin, alternative margin, sign-balanced
loss, responsibility, and context weight.  Structurally empty relation slots
are excluded exactly as they are from the runtime read.  Raw
maximum-separation selection is reported separately and cannot satisfy a gate.

One orchestrator must generate these streams internally, enforce the exact
R40/C25/J35 order, and stop before the next stage on failure.  The permissive
single-stage helper remains unit-test machinery and cannot claim freshness or
an integrated result.  The v2 evaluator-owned wrong-evidence fitter is retained
only as private historical/audit machinery and is not an executable production
fit.  Each stage starts a fresh AdamW optimizer state; only learned controller
parameters cross R-to-C-to-J.  This v4 successor keeps the existing mean
objectives; an optimizer change would require a later fresh identity only if
the correctly aligned v4 relation gate still fails.

The exact v4 run then stopped at that relation boundary with no rerun or
tuning.  It executed 40 updates, 320 unique fresh train streams, and 1,280
public rows; loss fell from `0.225892` to `0.070678`.  On its fixed 32-row
panel, the loss-selected and raw maximum-separation slots agreed on every row.
Their mean observed/alternative margins were `+0.296170` and `-0.291771`, mean
witness was `0.587941`, and mean best-versus-second loss gap was `0.297158`.
Only `17/32` rows satisfied all four confident conditions and only three of
eight streams had at least three confident rows, below the frozen `24/32` and
six-of-eight gate.  Individually, `24/32` rows passed each sign and witness
condition while `22/32` passed the loss-gap condition.  Untrained context mass
was `0.245744` with `34.375%` top-one agreement, so aggregate margins remained
only `+0.041142` and `-0.037646`; context training correctly did not run.
Evidence-order, public-presentation, and combined covariance deltas were all
exactly zero, empty memory was exactly zero, and shared parameters were
byte-exact.  Context and joint stages, development/final partitions, judge,
control streams, checkpoints, resources, and artifacts were not accessed.
The complete ephemeral report had SHA-256
`419355DA8408A9A57826DAAF51A958A8259EE8BCF2D26DE9372B08EE2881A60C`.

This aligned result confirms that the failure is breadth across anonymous
streams rather than a dead relation representation or a reporter mismatch.
Any optimization successor may change only the fixed cross-stream aggregation
objective.  A demonstrably invalid reporter assumption may be corrected only
under a fresh protocol identity while preserving the failed historical result
and every numeric threshold.  The clean restart, R40/C25/J35 exposure budget,
public inputs, three-slot loss, panels, stage stops, and all denials remain
unchanged.  No successor may add commitment-conditioned state, replay, extra
epochs, threshold relaxation, a deterministic solver, or a development/final
query.

The fixed combined successor is `phase6.public-relation-credit.v5`.  It changes
no row loss, model input, parameter scope, stage length, or numeric threshold.
It corrects the v4 reporter's false assumption that a good relation must have
one unique evidence slot: v4 stream 3 had all three slots independently satisfy
the frozen sign/witness conditions on all four rows, and its ordinary aggregate
runtime margins were signed, yet the near-zero loss gaps marked all four rows
as failures.  Retrospective set scoring finds `23/32` supported v4 rows with
per-stream counts `[4,3,4,4,3,2,3,0]`; that is still below `24/32` and remains a
failed v4 result rather than a retroactive pass.

For each row, the valid witness set contains every one of its three slots whose
same-slot conjunction is observed margin at least `+0.05`, alternative margin
at most `-0.05`, and their difference at least `0.10`.  A relation row is
supported iff this detached public set is nonempty.  Loss-selected slot,
raw-best slot, responsibility, and loss gap remain diagnostics only.  Relation
still requires `24/32` supported rows and six of eight streams with at least
three supported rows.  Context set mass is the sum of all valid real-slot
weights, averaged over supported rows only.  Context top-one is true only when
the maximum valid weight is strictly greater than both the maximum invalid
real-slot weight and the null weight returned by the same runtime softmax; the
null is not reconstructed by floating-point subtraction.  Ties and empty sets
fail.  Context retains its `0.60` mass and `0.80` top-one thresholds.  Valid-set
cardinality and unconditional mass with empty rows contributing zero are also
reported.  Derived valid masks, support decisions, and strict top-one outcomes
must remain exact under every declared evidence/presentation-order
transformation in addition to the continuous `1e-6` covariance bound.  The
unchanged disjoint final aggregate sign gate remains the behavioral evidence
and cannot be rescued by perfect set metrics alone.

For stream `s`, let `L_s` be the mean over its four rows of the existing
relation instance loss plus `0.25` times the existing separation loss.
Relation updates minimize

```text
0.5 * mean_s(L_s)
+ 0.5 * 0.05 * (logsumexp_s(L_s / 0.05) - log(8))
```

so the gradient weight of stream `s` is
`0.5/8 + 0.5*softmax(L/0.05)_s`.  Every anonymous stream therefore retains at
least `6.25%` weight while harder streams receive more pressure.  The operator
is permutation-invariant and has no commitment ID, persistent group state,
replay, or query-conditioned parameters.  Context keeps its ordinary flat
detached-responsibility loss.  Joint updates apply the same anonymous robust
operator to each stream's mean existing joint row loss.

V5 is another clean restart at torch seed `2026082831`.  It retains R40/C25/J35
and eight streams per update.  Update `u=0..99`, anonymous commitment position
`c=0..7`, uses topology seed `241000001 + 100000*u + 1000*c` and surface seed
`341000001 + 100000*u + 1000*c`.  Its relation/context panel uses
`441000001 + 1000*c` and `451000001 + 1000*c`; its final panel uses
`541000001 + 1000*c` and `551000001 + 1000*c`.  All seed pairs are unique and
disjoint.  Every update records its eight anonymous stream losses, eight
gradient weights, flat mean, entropic term, effective stream count, objective,
and gradient norm.  Final aggregate, covariance, empty-state, and frozen-byte
gates remain unchanged.  Failure still stops at the boundary without extension
or reinterpretation.  The set reporter is measurement only; the existing
smooth detached-responsibility context loss remains unchanged and no hard-set
self-label is introduced into training.

The exact v5 run passed its relation boundary and then stopped at its context
boundary without extension or tuning.  Relation processed 40 updates, 320
fresh streams, and 1,280 rows; loss fell from `0.225913` to `0.101348`.  Its
fixed panel reached `25/32` supported rows and six of eight streams with at
least three supported rows, against requirements of `24/32` and six of eight.
The per-stream counts were `[4,3,4,4,2,3,3,2]`; the valid-set cardinality
histogram was `[7,16,5,4]` for zero through three valid slots.  Mean
loss-selected observed/alternative margins were `+0.189810` and `-0.225100`.

Context then processed its fixed 25 updates, 200 fresh streams, and 800 rows;
loss fell from `1.401044` to `1.164529`.  Supported-row valid-set mass improved
from `0.374410` to `0.475952`, and strict valid-set top-one improved from
`0/25` to `13/25` (`0.52`).  It remained below the frozen `0.60` and `0.80`
requirements, so joint training did not run.  Relation support remained exact,
all three order-covariance deltas were zero, valid-set decisions were exactly
covariant, empty memory remained exact zero, and shared role/reasoning
parameters remained byte-exact.  There was no development/final access,
wrong-evidence training stream, judge call, checkpoint, or deployment effect.
The ephemeral full report SHA-256 is
`7949D52188F1FE4C0F9749BBBEC5FF1AB4053B6A53647758E2BCCC233A4FB5EF`;
the durable bounded summary is
`outputs/phase6-software-pipeline-reconstruction-result-summary.json`.

The source-demonstrable v5 context defect is objective mismatch.  C trained
cross-entropy to a detached softmin over all real slots on all 32 rows, while
its gate rewards probability on any absolute sign-valid slot and evaluates
only supported rows.  Seven of the fixed panel rows were unsupported, and
multiple other rows had two or three equally valid witnesses.  More epochs or
lower temperature could therefore sharpen the wrong target and are not an
admissible remedy.  The observed C loss and disjoint improvements establish a
live learned path but do not establish that its objective, representation, or
exposure is sufficient.

The fixed causal successor is `phase6.public-relation-credit.v6`.  It is a
clean restart at torch seed `2026082841`; update `u=0..99`, anonymous
commitment position `c=0..7`, uses topology seed
`261000001 + 100000*u + 1000*c` and surface seed
`361000001 + 100000*u + 1000*c`.  Its relation/context panel uses
`461000001 + 1000*c` and `471000001 + 1000*c`; its final panel uses
`561000001 + 1000*c` and `571000001 + 1000*c`.  These identities are unique
and disjoint from v4, v5, and each other.

V6 preserves the model, public inputs, R40/C25/J35 budget, eight streams per
update, optimizer settings, numeric thresholds, relation objective, panels,
stage stops, and all access denials.  It changes only C's training target after
R passes.  With relation bytes frozen, each supported public row forms the
same detached anonymous valid set `V` used by the reporter and minimizes

```text
-log(sum(context_probability[j] for j in V))
```

The C update is the ordinary mean across every supported row in the fixed
eight-stream batch.  Unsupported rows contribute no loss or gradient; a batch
with zero supported rows stops.  This is a training-only multiple-instance
likelihood: it chooses no slot, persists no mask, has no runtime fallback, and
uses no commitment identity, control stream, hidden label, or deterministic
selector.  The actual null probability remains in the softmax denominator.
Joint training retains the existing smooth joint loss so v6 tests whether the
set-aligned C basin survives ordinary harmonization without sending a
thresholded mask through a moving relation tower.

Each C update records supported rows per anonymous stream, total supported
rows, valid-set loss, responsibility mass already on `V`, responsibility
argmax-in-`V`, null mass, `q(V)`, `q(V)/(1-q(null))`, strict top-one-in-`V`,
gradient norm, and the exact row-count contribution weight of each stream.
The disjoint after-C panel remains the generalization measurement.  If R
fails, C does not run.  If train set loss falls without disjoint context gain,
representation/generalization is limiting.  If real-normalized set mass is
good while total set mass is low, null calibration is implicated.  If C passes
but J regresses, the smooth joint objective is mismatched.  Strong terminal
slopes still stop at C25 and may inform a later prospectively sized successor;
they do not authorize extension or rerun.

## Effects, rollback, and next action

Local synthetic data and foreground WSL2 compute only.  No network, service,
real or recovered repository data, model download, deployment, background
process, or external effect.  The hidden evaluator executes only its in-memory
synthetic component library.

Rollback removes only this additive leaf, the four additive code/test files,
and a failed-run result artifact.  The completed Glyph implementation,
checkpoint, sealed result, and evidence remain immutable.

The exact v6 run stopped at its relation boundary without a rerun or threshold
change.  Relation processed 40 updates, 320 fresh streams, and 1,280 rows;
loss fell from `0.225882` to `0.114760` and reached a minimum of `0.079292`.
Its disjoint panel produced `22/32` supported rows and five of eight streams
with at least three supported rows, below the fixed `24/32` and six-of-eight
requirements.  Per-stream support was `[4,3,4,4,2,2,3,0]`, and valid-set
cardinality was `[10,17,1,4]` for zero through three witnesses.  All order,
valid-set covariance, empty-memory, and shared-parameter invariants remained
exact.  C and J did not run, so this result says nothing yet about the v6
set-likelihood mechanism.  It instead falsifies the assumption that the R40
foundation was stable across the fresh v6 initialization and identities.  The
ephemeral full report SHA-256 is
`5107345047B4BD540138D654E45BF9E3FE1D3DC2FBCA0CABCF683D853CADA408`;
the bounded summary is
`outputs/phase6-software-pipeline-reconstruction-v6-result-summary.json`.

The fixed learned-exposure successor is
`phase6.public-relation-credit.v7`.  It is a clean restart at torch seed
`2026082851` and changes only relation exposure from R40 to R80: 640 unique
anonymous public streams and 2,560 rows.  C remains C25 and J remains J35.
Update `u=0..139`, anonymous commitment position `c=0..7`, uses topology seed
`281000001 + 100000*u + 1000*c` and surface seed
`381000001 + 100000*u + 1000*c`.  Its relation/context panel uses
`481000001 + 1000*c` and `491000001 + 1000*c`; its final panel uses
`581000001 + 1000*c` and `591000001 + 1000*c`.  These identities are unique
and disjoint from v4 through v6 and from each other.

V7 preserves the exact model, public inputs, optimizer and learning rates,
anonymous entropic R objective, supported-valid-set C objective, smooth J
objective, thresholds, panels, stage stops, instrumentation, and access
denials.  The first 40 R updates may be recorded as ordinary training history
but are not a gate, checkpoint candidate, or selection point.  The only R gate
evaluates the terminal R80 state.  All additional R updates use new streams;
there is no replay.  If R fails, stop without extension or seed retry.  If R
passes, freeze the qualified relation bytes and execute the existing C25
set-likelihood learner; J35 runs only if C passes and retains R.

This successor is intended to establish whether broader learned exposure is
enough to qualify one fresh relation foundation and let the new C mechanism
receive a fair trial.  A C improvement or pass would be conditional mechanism
evidence, not proof of causal superiority over v5 and not a cross-identity
robustness claim.  The project deliberately omits an old-objective baseline
arm here because the immediate objective is functional integration rather than
comparative benchmarking.  Do not rerun v6, select a lucky seed, relax a
threshold, install a deterministic selector, or open development/final/judge
access.  Freeze resource scaling and legacy reasoning/action calibration until
a fixed fresh-train joint gate passes.

The exact v7 run passed 99 focused and 129 combined frozen tests plus two
independent static audits, then stopped at its terminal R80 relation gate
without a rerun.  Relation processed 640 unique streams and 2,560 rows; loss
fell from `0.225913` to `0.113038` and reached `0.052826`, but the disjoint
panel again reached only `22/32` supported rows and five of eight qualifying
streams.  Per-stream support was exactly `[4,3,4,4,2,2,3,0]`, the same pattern
as v6, while cardinality shifted only from `[10,17,1,4]` to `[10,16,2,4]`.
All invariants and access denials remained exact.  C and J did not run.  The
ephemeral full report SHA-256 is
`5D36609374E8F240A2CB3EE0AFB30CE12B967BAA229866210727BABDFD9717AE`;
the bounded summary is
`outputs/phase6-software-pipeline-reconstruction-v7-result-summary.json`.

V7 falsifies the exposure-only remedy: doubling unique learned R experience
did not change the earliest boundary or its per-stream coverage pattern.  The
repeated zero-of-four stream and the fixed concentration of misses are now
evidence of a structural relation representation or objective blind spot,
although they do not alone identify which internal feature is missing.  Do not
add more R epochs, replay, seed retries, threshold changes, or a deterministic
selector.

Fresh public-only diagnostics localize that blind spot.  The final
`[node,node,width]` tensors from `EvidenceOrderedPairEncoder` retain a strong
candidate-twin contrast for all five motif families (about `0.214` relative
difference), including the systematically weak motif-3 and motif-4 families.
The old relation readout then flattens those tensors into an unordered bag of
cells before two attention sums, mean, and maximum.  After that reduction,
motif-3 and motif-4 twin codes become nearly collinear (cosines approximately
`0.99999988` and `0.99999994`).  The missing information is therefore not in
the data or learned pair encoder: the readout erases which directed cells
share a node, precisely the row/column co-incidence needed by the broad
anonymous gap families.

The fixed structural successor is `phase6.public-relation-credit.v8`.  It
replaces only the relation readout with a learned two-axis, permutation-
equivariant set hierarchy.  For every anonymous node it separately learned-
pools the outgoing row and incoming column with two heads each, retains the
diagonal cell, and maps those five width-sized summaries to one node token.
One shared multi-head self-attention block exchanges information among the
unordered node tokens.  Four learned seed queries then pool that node set, and
the existing relation-code boundary projects and normalizes their concatenated
outputs to the unchanged width.  Simultaneously renaming both graph axes only
permutes intermediate node tokens and therefore cannot change the final code;
predecessor/candidate direction and within-node co-incidence remain available.

This is a learned Set-Transformer/DeepSets-style readout, not a graph
canonicalizer or hand-coded motif detector.  It receives only the existing
public pair tensor and contains no component, commitment, package, motif,
variant, or partition identity; no deterministic solver, topology signature,
control stream, replay, hard selector, or stored mask is added.  The context
tower stays byte-structurally unchanged.  The pair encoder, symmetric relation
comparator, anonymous entropic R objective, optimizer and learning rates,
R80/C25/J35 exposure, valid-set C objective, smooth J objective, stage stops,
and every numeric gate remain unchanged.

V8 is one clean restart at torch seed `2026082861`.  Update `u=0..139` and
anonymous commitment position `c=0..7` use topology seed
`301000001 + 100000*u + 1000*c` and surface seed
`401000001 + 100000*u + 1000*c`.  Its relation/context panel uses
`501000001 + 1000*c` and `511000001 + 1000*c`; its final panel uses
`601000001 + 1000*c` and `611000001 + 1000*c`.  These identities are unique
and disjoint from v4 through v7 and from each other.

Before the one GPU run, tests must prove simultaneous-node-renaming
invariance, sensitivity to row/column incidence even when the flattened cell
multiset is identical, finite nonzero gradient paths through both axis pools,
node self-attention, learned seeds, and final projection, dynamic node counts,
candidate-swap sensitivity, and dependency-closure exclusion of identity,
private, control, and solver material.  The frozen evaluator and its tests may
not change.  If the unchanged R gate fails again, stop without extension,
rerun, or reinterpretation; if it passes, freeze R and immediately exercise
the already specified C25 then J35 sequence.  Motif-family geometry is a
diagnostic, never an alternate pass condition.

V8 passed `101/101` focused and `131/131` combined frozen CPU tests plus three
independent static audits, then executed exactly once.  It stopped at R80 with
`0/32` supported rows and zero of eight qualifying streams.  Relation loss was
effectively flat (`0.225910` to `0.225905`, minimum `0.221979`) and mean
gradient norm was only `0.012229`; all panel margins were on the order of
`1e-6`.  This is qualitatively different from v6/v7's partial relation fit:
the replacement hierarchy is structurally capable of distinguishing incidence
in isolated tests but its initialized end-to-end public training path collapsed
to nearly constant relation codes.  Context and joint stages correctly did not
run.  Covariance, empty-memory, shared-parameter, and access-denial invariants
remained exact.  Runtime was `88.931` seconds with peak allocated/reserved GPU
memory `237601280`/`239075328` bytes.  Full ephemeral report SHA-256 is
`ABA7F115B469572B43A52BFC91E619E0E77582575ADC85EE6CCCB20A1823390C`;
the bounded result is
`outputs/phase6-software-pipeline-reconstruction-v8-result-summary.json`.

Do not rerun v8, add exposure, relax a gate, or interpret structural unit-test
separation as learned success.  The next diagnosis must measure public-code
dispersion and gradient attenuation through the seed residual, normalization,
and comparator using identities outside every protocol schedule.  A successor
must preserve the previously live global relation-learning path while adding
incidence structure, rather than replacing it with another unproven tower.

The fresh-identity diagnosis confirms an exact symmetry-collapse stationary
point.  Pair tensors retain mean relative twin separation `0.2657`, while the
v8 hierarchy reduces final twin-code L2 to `0.001412` and off-diagonal cosine
to approximately `0.999999`.  Its row, column, node, and seed attention are
almost uniform at initialization.  A shared constant code for every relation
reproduces loss `0.2259075`, all zero margins, and exactly zero code and
comparator gradients: positive/negative demands cancel, while the maximally
violated cosine-separation hinge also has zero derivative at equality.  The
seed-query residual is not the cause; removing it improves separation only
about five percent.  In contrast, the old global attended/mean/max feature
retains roughly `7.2x` more twin separation.  V8 also consumed its new module's
random draws before constructing the comparator and pair encoders, so replacing
the readout unintentionally discarded both the proven signal carrier and the
legacy initialization geometry.

The fixed function-preserving successor is
`phase6.public-relation-credit.v9`.  It restores the exact learned global
relation branch in its original construction order: two cell-attention sums,
cell mean, and cell maximum followed by the existing `4W -> W` projection.
After every legacy relation, context, and pair-encoder module has been
constructed, v9 appends one learned incidence residual.  It separately pools
each node's outgoing row and incoming column with two learned heads each,
retains the diagonal, maps the resulting `5W` vector to a node token, and pools
the unordered node-token set with two learned heads plus mean and maximum.
A bias-free `4W -> W` residual projection is initialized to exact zero:

```text
relation_code = normalize(global_precode + incidence_residual)
```

Consequently v9 begins as the hypothetical same-seed legacy learner bit-for-
bit; it cannot regress by random replacement.  The first gradient opens only
the zero projection, later gradients reach the anonymous axis/node features,
and the global branch remains trainable throughout.  This is a neural residual
expansion, not a frozen baseline, fallback, hard selector, or mixture of
models.  There is still one relation code and one active competence lineage.
The residual receives no identity, motif, signature, hidden label, control
stream, or deterministic rule.

V9 is one clean restart at torch seed `2026082871`.  Update `u=0..139` and
anonymous position `c=0..7` use topology seed
`1101000001 + 100000*u + 1000*c` and surface seed
`1201000001 + 100000*u + 1000*c`.  Its relation/context panel uses
`1301000001 + 1000*c` and `1311000001 + 1000*c`; its final panel uses
`1401000001 + 1000*c` and `1411000001 + 1000*c`.  All are unique and disjoint
from v4 through v8 and from the diagnostic identities.

The pair encoder, comparator, public inputs, anonymous robust relation
objective, optimizer and learning rates, R80/C25/J35 exposure, supported-set C
objective, smooth J objective, thresholds, stage stops, and access denials
remain unchanged.  Tests must prove exact initial equality with the legacy
global function, zero residual bytes, first-step projection opening and
second-step upstream gradient flow, simultaneous-renaming invariance,
same-cell-multiset incidence sensitivity, candidate-swap sensitivity, dynamic
node counts, exact stage ownership/freeze behavior, and dependency-closure
absence of private or identity inputs.  One fresh GPU execution is decisive:
the unchanged fused R gate passes or v9 stops.  If it passes, C25 and J35 run
immediately under their existing gates; no legacy-only score may rescue the
fused result.

V9 passed `104/104` focused and `134/134` combined frozen CPU tests plus three
independent static audits, then executed exactly once.  Its R80 relation loss
fell from `0.225931` to `0.122426` (minimum `0.072893`) with mean gradient norm
`0.738283`.  The disjoint public panel reached the required `24/32` supported
rows, but support was distributed `[4,3,4,4,3,2,2,2]`: only five of eight
streams contained at least three supported rows, below the fixed six-stream
requirement.  The effective stream count simultaneously contracted from
approximately `8.00` to `4.18`.  The fused relation gate therefore failed and
C25/J35 correctly did not run.  All permutation, presentation, empty-memory,
shared-parameter, and access-denial invariants remained exact.  Runtime was
`85.011` seconds with peak allocated/reserved GPU memory
`236243456`/`236978176` bytes.  Full ephemeral report SHA-256 is
`E91945562BCD91E7152F9BBD9B2186BCCD50397330E684DD190EC5FA8E2570F4`;
the bounded result is
`outputs/phase6-software-pipeline-reconstruction-v9-result-summary.json`.

V9 repairs v8's optimization collapse and improves fresh-panel coverage over
v7 from `22/32` to `24/32`, including eliminating the zero-support stream, but
it does not yet produce sufficiently broad competence.  The remaining failure
is distributional rather than aggregate: exactly enough rows qualify while
three streams remain at two-of-four.  Do not rerun v9, lower the six-stream
gate, add deterministic routing, or infer that the unexecuted context learner
works.  The next diagnosis must determine why the anonymous robust objective's
credit concentrates onto roughly four effective streams and whether the live
incidence residual is acquiring general endpoint structure or merely
amplifying already-easy relations, using fresh public-only identities outside
all protocol schedules.

The source-demonstrable v9 mismatch is now bounded to the reduction inside
each anonymous stream.  Its outer objective correctly gives larger gradients
to higher-loss streams while retaining at least `6.25%` credit for every
stream; an effective count of `4.18` is therefore concentration, not exclusion.
Inside each stream, however, four row losses were averaged uniformly even
though fresh success must be broad across variations.  Two already-easy rows
could consequently offset two persistently hard rows before the robust outer
learner saw that stream.  The slot-level soft minimum remains aligned with the
existential valid-witness contract and is not changed.

The fixed learned successor is `phase6.public-relation-credit.v10`.  It keeps
the complete v9 neural architecture, width, public inputs, parameter ownership,
optimizer, learning rates, R80/C25/J35 exposure, eight-stream outer entropic
objective, context objective, joint sequencing, numeric thresholds, panels,
stops, and every access denial.  It changes only the differentiable reduction
of the four existing public row losses inside R and prospective J.  For stream
`i`, with unchanged row losses `r_ij`, it minimizes the inner risk

```text
S_i = 0.5 * mean_j(r_ij)
    + 0.5 * 0.05 * (logsumexp_j(r_ij / 0.05) - log(4))
```

before applying the unchanged v9 outer objective to the eight values `S_i`.
The gradient weight of row `j` is
`0.5/4 + 0.5*softmax(r_i/0.05)_j`: every row retains at least `12.5%` of its
stream's credit while higher-loss variations receive more.  The operation is
permutation-invariant and consumes no stream, motif, component, package,
partition, or row identity.  It adds no routing rule, mask, top-k decision,
solver, replay, stored example, extra update, threshold change, or model
parameter.

V10 is one clean restart at torch seed `2026082881`.  Update `u=0..139` and
anonymous commitment position `c=0..7` use topology seed
`1501000001 + 100000*u + 1000*c` and surface seed
`1601000001 + 100000*u + 1000*c`.  Its relation/context panel uses
`1701000001 + 1000*c` and `1711000001 + 1000*c`; its final panel uses
`1801000001 + 1000*c` and `1811000001 + 1000*c`.  These identities are unique
and disjoint from v4 through v9 and every declared diagnostic range.

Pre-run tests must prove the exact formula and gradient, uniform `0.25` row
credit for equal losses, the `0.125` floor, harder-row preference, row-order
permutation covariance, equal-mean upper-tail discrimination, full per-update
row/stream accounting, unchanged outer objective and stage boundaries, seed
freshness, and continued hidden/control/solver exclusion.  The one fixed GPU
run must persist its complete report and terminal synthetic checkpoint inside
the isolated WSL environment whether the relation gate passes or fails.  The
bounded workspace summary binds their hashes.  Those artifacts are diagnostic
only: they cannot authorize a same-protocol rerun, act as an alternate pass,
or be selected into a promoted lineage.  A residual-zeroed panel may be
recorded only after the fused gate outcome, with exact byte restoration, as a
measurement of learned incidence contribution; it cannot rescue the fused
result.

The unchanged fused R gate remains decisive.  If it passes, C25 and J35 run in
the same invocation under their existing gates.  If it fails, v10 stops with
no extension or retry.  Broader coverage would support the row-credit
diagnosis; another concentrated failure with healthy inner credit would move
the next isolated test to bounded outer distributional robustness rather than
another representation rewrite or capacity increase.

The fixed v10 training and gate computation ran once, but its external evidence
harness then failed before emitting or persisting the returned report.  The
failure occurred while digesting the scalar `role_match_scale`: directly
viewing a zero-dimensional float tensor as bytes is invalid in PyTorch.  It was
strictly post-fit and post-panel, but the process terminated before the report,
terminal checkpoint, or compact result was written.  The in-memory gate result
and trained weights are therefore unavailable and no scientific outcome is
claimed.  Both declared WSL artifact paths remained absent.  V10 will not be
rerun; the exact failure is preserved in
`outputs/phase6-software-pipeline-reconstruction-v10-artifact-failure-summary.json`.

The recovery successor is `phase6.public-relation-credit.v11`.  It changes no
learning behavior from v10: architecture, initialization procedure, nested row
risk, outer stream risk, data scale, optimizer, exposure, gates, stage stops,
and access denials are identical.  It introduces only a scalar-safe model-state
digest that flattens each tensor before taking its byte view, with a unit test
that mutates and restores an actual scalar parameter.  The evidence harness
must use this tested function and persist checkpoint and full report before it
prints the compact result.

V11 is one clean restart at torch seed `2026082891`.  Update `u=0..139` and
anonymous position `c=0..7` use topology seed
`1901000001 + 100000*u + 1000*c` and surface seed
`2001000001 + 100000*u + 1000*c`.  Its relation/context panel uses
`2021000001 + 1000*c` and `2031000001 + 1000*c`; its final panel uses
`2041000001 + 1000*c` and `2051000001 + 1000*c`.  These identities are unique
and disjoint from v4 through the consumed v10 schedule.  V11 executes once
under the unchanged fused gates.  It is a fresh recovery identity, not a v10
seed retry, and v10 remains a preserved no-result infrastructure failure.

On its prospective bytes, v11 passes `106/106` focused CPU tests and the
combined frozen evaluator/runner surface passes `136/136`.  The added test
proves that the full model digest handles, detects mutation of, and exactly
restores the scalar parameter that broke v10 capture.  The frozen evaluator and
evaluator-test hashes remain `45D2282D...D0DFCE44` and
`CFB02F8D...86586D3`.

The one fixed v11 GPU execution is complete and preserved.  R80 loss fell from
`0.225905` to `0.119743` (minimum `0.086011`) across 640 fresh public streams
and 2,560 rows.  The nested objective remained live: effective stream count
ended at `5.19`, mean effective within-stream row count ended at `3.70`, and
the minimum observed row effective count was `2.29`.  Nevertheless, the fresh
panel supported only `21/32` rows and four of eight streams, with distribution
`[4,2,4,4,2,2,3,0]`, below the unchanged `24/32` and six-stream gate.  Context
and joint stages therefore did not run.

The terminal model and full report were durably written before compact output
and independently reloaded.  Their SHA-256 values are
`2CF650BA...F5F0694` and `EFDAC946...071AE43`; the bounded record is
`outputs/phase6-software-pipeline-reconstruction-v11-result-summary.json`.
A post-gate same-panel intervention zeroed only the learned incidence
projection and then restored it bit-exact.  The residual improved mean witness
by `0.04392`, reduced mean target loss by `0.00110`, and improved mean
separation by `0.00545`, but changed neither supported rows nor qualifying
streams.  Thus v11 falsifies the nested row-risk candidate as sufficient and
shows that incidence contributes quantitatively without resolving the breadth
failure.  Because v9 and v11 used different fixed identities, their score
difference is not claimed as a causal objective comparison.

Do not rerun v11, relax its gate, add a deterministic selector, or replay its
examples.  The next bounded action is a public-only causal diagnosis of the
recurring hard structural streams using the preserved report/checkpoint,
followed by one fresh learned representation or factorization successor.  The
working hypothesis is a missing transferable structural distinction rather
than simple row-credit starvation; it remains a hypothesis until the
intervention localizes it.

The fixed checkpoint-only diagnostic is
`phase6.public-structure-path-diagnostic.v11-d1`.  It may load only the exact
v11 terminal checkpoint above and may not update any parameter.  For each of
the same eight anonymous train-partition commitments, it constructs the full
2x2 factorial of topology bases `3001000001`/`3011000001` and surface bases
`3101000001`/`3111000001`, adding `1000*c` at commitment position `c`.  These
32 streams are public-only, fresh, and outside every v4-v11 training, panel,
and diagnostic identity.  It records fused and incidence-zeroed public panels;
same-contract candidate-twin contrast through pair tensor, global precode,
incidence feature/projection, fused code, and comparator outcome; and
per-stream gradient norms/cosines for the shared encoder, global pool,
incidence branch, and comparator.  Backpropagation is measurement-only:
optimizer creation, optimizer steps, checkpoint writes, evaluator calls,
wrong-evidence/control streams, development/final partitions, replay, and
parameter mutation are forbidden.  The model digest must be exact before and
after, and the incidence projection must be restored bit-exact after every
ablation.

The full diagnostic output is
`/opt/angler/results/phase6-software-pipeline-reconstruction-v11-d1-report.json`;
the bounded workspace record is
`outputs/phase6-software-pipeline-reconstruction-v11-d1-summary.json`.  The
diagnostic cannot pass v11 or choose a favorable identity.  Signal present in
the incidence features but absent from discrete fused decisions supports one
shared zero-initialized late structural-score residual; signal absent before
fusion requires a learned structural-encoder revision; signal preserved
through scoring but opposed across streams supports the bounded outer-DRO
control.  Any trained successor starts from a fresh identity, not the failed
v11 checkpoint.

The diagnostic completed without an optimizer, update, checkpoint write, or
parameter mutation and passed independent artifact review.  Surface
rerendering changed zero support decisions and only approximately `1e-7`
continuous values.  Topology changed the fused panel from `23/32` rows and
four qualifying streams to `29/32` and seven, with target witness improving
about `0.04988`.  The public pair tensor retained mean candidate-twin L2 near
`9.02`, the global code retained `0.46--0.50`, and fusion retained
`0.53--0.56`.  The incidence branch improved target witness by
`0.053--0.054` and crossed one support boundary per cell, but its own projected
twin difference was only `0.00030--0.00033`; it is currently a useful learned
amplifier rather than a distinct structural basis.

Cross-stream update geometry is the stronger blocker.  Across shared encoder,
global pool, incidence branch, and comparator, `53.6--64.3%` of off-diagonal
gradient cosines were negative and the worst reached `-0.948`; the conflict
reproduced under surface rerendering.  Scalar loss weighting alone cannot make
opposed vectors mutually constructive.  Full report SHA-256 is
`3191E1D9962A11BCCE9E8664E315D13CADB2F893C02994B7AA8B25233DF142BA`;
the bounded record is
`outputs/phase6-software-pipeline-reconstruction-v11-d1-summary.json`.

The next hypothesis is modular plasticity: preserve one competence lineage but
compare an anonymous conflict-aware single core with a capacity-matched
four-cell Angler cluster whose partial procedures and composition weights are
learned from public structure and outcome feedback.  Cells may not receive
fixed domain roles, commitment/package/motif identity, private or control
data, replay, a deterministic solver, or evaluator feedback during inference.
The comparison must equalize total trainable parameters, public exposure,
update count, and candidate-evaluation budget.  A cluster gain counts only if
cells acquire complementary reusable procedures and their learned composition
improves fresh topology transfer; gains explainable by extra capacity,
sampling, voting, or fixed orchestration are rejected.

Before attributing any effect to multiple cells, isolate the update mechanism
they would share.  The fixed successor is
`phase6.public-conflict-reconcile.single.v12`.  It retains the complete v11
controller, public row and stream losses, R80/C25/J35 sequence, optimizer
hyperparameters, relation/context/final gates, and every access denial.  It
changes only how the eight public stream gradients are combined during R and
prospective J.

For each stream and each of four anonymous relation parameter blocks (pair
encoder, global readout, incidence readout, and comparator), v12 measures the
public gradient norm and cosine Gram geometry.  Joint training adds one fifth
anonymous context block.  One small permutation-equivariant
set network consumes only detached standardized losses, existing outer
weights, log norms, and cosine summaries.  It emits a residual over the
existing anonymous weights independently for each block.  The residual output
is initialized to exact zero, so the first relation update is mathematically
the v11 direction; a fixed twin test compares its gradients and AdamW result
numerically under predeclared tolerance.  Joint training continues the learned
mixer without resetting it.  A `0.5` anchored mixture retains a positive contribution
from the existing weights.  No stream, commitment, package, component, motif,
topology, surface, partition, or cell identity is an input.

The mixer learns from eight symmetric leave-one-stream-out consequences per
fresh update.  Seven streams propose a blockwise direction; its first-order
cosine alignment with the withheld eighth stream supplies the meta-loss.
Every stream is withheld once, the mixer receives no evaluator result, and the
controller receives one all-eight-stream AdamW update.  This is a learned
update rule over experience geometry, not a per-batch deterministic projection
or a promoted PCGrad/CAGrad solver.  Raw norms, cosine geometry, loss
components, legacy and adjusted direction norms, and direction digests are
recorded from the same gradients; no second controller run is needed for that
mechanism comparison.  Controller, competence state, mixer configuration, and
mixer weights share one reloadable checkpoint lineage.

V12 uses controller seed `2026082931`, mixer seed `2026082932`, training
topology/surface bases `3301000001`/`3401000001`, relation-panel bases
`3501000001`/`3511000001`, and final-panel bases
`3601000001`/`3611000001`, with the existing `100000*u + 1000*c` schedule.
All identities are fresh and disjoint from v4--v11 and D1.  It runs once.
Tests must prove stream-permutation equivariance, zero-residual equality with
the legacy direction, finite normalized positive weights under zero gradients,
gradient reachability into the mixer, symmetric withheld coverage, exact
parameter-block ownership, first-update equality, frozen-parameter integrity,
fresh identities, and hidden/control/solver exclusion.  Failure stops v12
without rerun, threshold relaxation, or replay.

The capacity-matched cluster pilot is licensed by mechanistic integrity rather
than a favorable v12 score: one fixed invocation must complete R80 with finite
complete geometry, exact exposure and fold accounting, unchanged frozen
parameters, a changed terminal mixer digest, and at least one post-first
learned block-weight vector and applied controller direction different from
their inherited counterparts.  The fixed
twin and permutation tests must pass and the combined checkpoint must reload.
Relation/final pass or fail is recorded for interpretation but cannot select
whether the causal cluster comparison runs; doing so would bias that test.

After v12, the capacity-matched four-cell pilot uses the same frozen learned
update interface in both its monolithic and clustered arms.  The cluster may
advance only if learned fused behavior beats the equal-budget single core on
fresh structures, uses multiple cells causally, remains stable under surface
rerendering, and exceeds uniform fusion and every individual-cell lesion.

V12 executed exactly once on `cuda:0` and produced a valid combined lineage.
The relation stage completed R80 (`640` streams, `2,560` rows) and passed its
fresh gate at exactly `24/32` supported rows and `6/8` qualifying streams.
Every one of the `79` post-anchor controller directions differed from the
legacy direction, the mixer moved by L2 `1.32156`, optimizer ownership remained
exactly separated, and all runtime integrity observations passed.  The context
stage then completed C25 but stopped at its unchanged gate: supported-row
top-one was `0.58333` versus `0.80`, and valid-set mass was `0.45652` versus
`0.60`; the relation boundary remained intact.

This is a genuine stage advance, not yet end-to-end success.  The same-gradient
diagnostic means did not improve globally (negative-alignment fraction
`0.18789` legacy versus `0.18867` applied; cancellation ratio `0.68063` versus
`0.68031`), so the relation pass is not attributed solely to a global conflict
metric.  V11 and v12 also use different fixed identities, so their `21/4`
versus `24/6` outcomes are descriptive rather than a causal baseline claim.
`CLUSTER_PILOT_LICENSE_V1` is nevertheless issued from mechanistic integrity,
not favorable performance.  Full report SHA-256 is
`C0F579DD9CE3C6A8CC3AC352802484B2E2C8541F8EF39B8408DD8B98998428FB`;
combined checkpoint SHA-256 is
`B4DA4550D18C9F1480903DA087A8E7799341763F1EDD63061E8A04A7491BD62C`;
the bounded record is
`outputs/phase6-software-pipeline-reconstruction-v12-result-summary.json`.
Do not rerun or tune v12.  The next construction is the one fresh
capacity-matched monolithic-versus-four-cell causal pilot.

The fixed successor is `phase6.public-anonymous-cluster.paired.v13`.  It is a
relation-only causal comparison, not a runtime or AGI claim.  Three fresh
replicates each train an unchanged monolithic relation core and a four-cell
anonymous cluster for the same R80 batches (`640` streams / `2,560` rows per
arm), then evaluate the same two fresh topology panels and a surface rerender.
Both arms run regardless of score.  Context, joint training, development,
final, control arms, replay, and normal runtime memory remain closed.

The cluster contains four independent width-16 learned pair encoders and
comparators plus one learned permutation-invariant soft composer.  Every cell
sees every stream and retains at least `0.125` fusion weight; there are no cell
roles, IDs, sharding, voting, hard routing, solver, or padding parameters.  Its
complete controller-plus-updater has `269,031` parameters versus `269,010` for
the paired monolith (difference `+21`, `0.00781%`); active trainable capacity is
`65,387` versus `65,366` (difference `+21`, `0.03213%`).  Both arms use five
anonymous update blocks and byte-identical initial mixers and shared
non-relation state.  Learned fusion is compared read-only with uniform,
single-cell, and every leave-one-cell-out lesion.

The dedicated V13 suite passes `7/7`; the combined frozen evaluator/runner
suite passes `156/156` in `66.999s`.  Pre-run runner SHA-256 is
`F1045756E77D60A7968265867035CEA55BFFE8BF6E1A73AB50C12A719EC8B529` and
runner-test SHA-256 is
`2E6D844D24DB0A9326D84A19AEC56ED5BF6288B94C67AD5926AC05933FB6DF32`.
The evaluator and evaluator-test remain frozen at `45D2282D...CE44` and
`CFB02F8D...86D3`.  One CPU preflight update completed for both arms and every
cluster cell plus the composer received finite nonzero gradients.

The one fixed GPU pilot is complete and independently audited.  Across three
paired replicates the cluster reached `134` supported rows, `24` qualifying
streams, and mean target loss `0.0580147`, while the capacity-matched monolith
reached `137`, `27`, and `0.0550869`.  Only one replicate favored the cluster,
so the frozen result is `CLUSTER_HARMFUL` for this exact all-active design.
This does not reject modular reasoning in general.

The mechanistic result is more specific.  All four cells and the composer
received gradients and changed, cell outputs remained diverse, and the
composer's largest mean weight selected the best-loss individual cell in each
replicate.  Learned fusion beat uniform fusion in all three replicates
(`134/24/0.05801` versus `121/19/0.06642`) but failed against several
single-cell and leave-one-cell-out lesions.  The cluster also retained more
gradient conflict than the monolith: negative stream-gradient pairs were
`28.79%` versus `25.67%`, mean cosine was `0.323` versus `0.402`, and
cancellation ratio was `0.649` versus `0.721`.  Requiring every stream to
update every cell and retaining a `0.125` read floor diversified predictions
without isolating incompatible procedural updates.

The full report SHA-256 is
`781BAF57325882A281ED3A6860324711F6597BE25F4ABB54992A1A45A59F53C0`;
the reloadable checkpoint SHA-256 is
`734FCB369FF2573180BC8E6AF728917A5ED51A38270048C0570433B3EC644BA2`;
the bounded record is
`outputs/phase6-software-pipeline-reconstruction-v13-cluster-summary.json`.
V13 will not be rerun, tuned, or rescued by changing its fixed surface
tolerance.

The next learned hypothesis is an anonymous permutation-equivariant
counterfactual plasticity router.  It learns both read weights and per-cell
update weights from content-derived public evidence, anonymous cell responses,
and differentiable withheld-evidence marginal improvement.  It removes
compulsory per-query participation in favor of a soft across-experience
anti-collapse objective.  It may not use task/domain/cell identity, fixed
roles, hard-coded lesion winners, deterministic top-k, voting, gradient
surgery, replay, or solver rules.  Its purpose is to let incompatible
procedures occupy different plastic cells while retaining learned composition.

The fixed successor is
`phase6.public-counterfactual-plasticity-router.paired.v14`.  It tests one
mechanism only: whether feedback-trained allocation of plasticity reduces
interference between anonymous cells.  It does not compare with the historical
monolith or reuse a V13 checkpoint.  Three fresh paired replicates each run an
`UNIFORM_PLASTICITY` cluster and a byte-identically initialized
`LEARNED_PLASTICITY` cluster over the same R80 batches (`80` updates, `640`
streams, and `2,560` rows per arm).  Both arms run regardless of intermediate
score and both contain the same active router capacity.  The retired V12
conflict mixer is absent from both arms.

The four V13 relation cells and anchored learned read composer remain unchanged
so the causal variable is write allocation alone.  For every public stream,
each cell is evaluated alone and supplies a composer-independent public loss,
prediction-strength summary, and detached local gradient.  A shared
permutation-equivariant set network receives only anonymous cell-by-stream loss,
log-gradient norm, and within-cell compatibility summaries.  It emits a softmax
over cells for each stream, with no floor, role, identity, top-k rule, vote,
solver, or replay.  Cell encoders use explicit SGD at `3e-4`, cell heads use
explicit SGD at `1e-3`, and their jointly routed direction is clipped at `5.0`.
The composer remains separately owned by AdamW at `1e-3`; the router is
separately owned by AdamW at `1e-3`.

Every update performs all eight leave-one-stream-out folds.  For each fold the
router sees only the other seven streams.  Their detached cell-local gradients
form a differentiable virtual SGD cell update; the excluded stream is then
evaluated through those virtual cells and the unchanged current composer using
strict eager functional execution.  Gradient flows from actual held-out
post-update public loss only into the router, without a Hessian or
`create_graph=True`.  A `0.01` KL penalty balances mean cell use across
experiences but does not require uniform participation per stream.  Because the
real-step loss difference is numerically small, the router objective scales the
post-minus-pre difference by the fixed factor `1e4`; this changes gradient scale,
not the optimum or the virtual step.  The pre-meta full-batch allocation is
detached for the real update, and the router changes only after the controller
step, so feedback affects the next experience rather than circularly labeling
the current one.

Fresh replicate seeds are `(2026083201..204)`, `(2026083211..214)`, and
`(2026083221..224)`.  Training topology/surface bases are
`5001000001`/`5041000001`; panel A uses `5081000001`/`5121000001`, its surface
rerender uses `5161000001`, and panel B uses
`5201000001`/`5241000001`, with replicate stride `10000000`.  These identities
are disjoint from V12 and V13.

The router is supported only if its learned arm has lower aggregate fresh-panel
target loss, no aggregate supported-row or qualifying-stream regression, and
loss wins in at least two of three replicates.  One fresh post-training
adaptation diagnostic must also favor the correct learned allocation over both
uniform and cell-permuted allocation in aggregate and in at least two
replicates.  Every cell must retain at least `10%` aggregate allocation, and
normal learned read fusion must beat its forced-uniform read lesion without
losing rows or streams.  Cell/stream permutation equivariance, heldout-input
exclusion, strict optimizer ownership, checkpoint reload, and all access
denials must hold.  Performance without the allocation ablation is
inconclusive; regression on every aggregate task measure is harmful.  No
post-result rerun, role assignment, hard selection, or deterministic repair is
permitted.

V14 executed exactly once on `cuda:0` after `164/164` combined contract tests
passed (`1` CUDA skip).  The run covered `480` arm updates, `3,840` public
streams, `15,360` rows, and `3,840` leave-one-stream-out folds in
`10,104.637` seconds.  The report SHA-256 is
`83F4730E91A9600740BAFD66439493649014D4F077C2890D57B3B698867AB378`;
the reloadable checkpoint SHA-256 is
`5B64D851CEBC19138554A2B5AE5BAF89F4CE8973F5FADB05A224F593E901D552`.
An independent read-only audit passed `111,688` checks and reloaded all three
paired systems on CPU.  The bounded record is
`outputs/phase6-software-pipeline-reconstruction-v14-plasticity-summary.json`.

The frozen Boolean rule returned `PLASTICITY_ROUTER_SUPPORTED`, and that label
is preserved as historical protocol output.  It is not promoted as scientific
evidence.  Learned loss was `0.1953117092` versus `0.1953117319` for uniform
plasticity, only `2.2662e-8` (`0.116` ppm) lower.  Both arms produced `0`
supported rows and `0` qualifying streams.  Mean routing total variation from
uniform was only `0.000439`, normalized routing entropy was `0.99999947`, and
the meta consequences were quantized at the FP32 loss ULP with `33.8%` exact
zeros.  The effect was smaller than ordinary continuous rerender variation.
The durable interpretation is therefore
`FORMAL_THRESHOLD_PASS_PRACTICAL_EFFECT_NOT_ESTABLISHED`; V14 will not be
rerun, post-hoc rescued, or advanced as a working plasticity mechanism.

V14 did independently retain one substantive inherited result: learned read
composition lowered loss by `2.7367%` relative to forced-uniform read in all
three replicates.  The same effect existed with uniform write allocation, so it
validates the V13 composer rather than V14 plasticity, and it did not yield
discrete competence.

The successor must first restore nonzero procedural competence with matching
actual and virtual optimizer semantics.  It must then teach anonymous cells
from causal consequences on fresh paired variations of procedures without
exposing procedure, task, cell, stream, package, or motif identity and without
replay, fixed roles, deterministic selection, voting, gradient surgery, or a
solver.  A future success rule must require nonzero absolute competence,
nonzero discrete advantage, a loss margin materially above measured numerical
variation, observable nonuniform specialization with balanced aggregate use,
and composition that beats uniform read and every single/drop-one lesion.  A
zero-versus-zero comparison or floating-point crumb can never pass again.

## Fixed V15 cross-variation plasticity successor

The freeze-ready successor is
`phase6.public-anonymous-cross-variation-plasticity.paired.v15`.  It remains
relation-only, train-partition-only, local, and synthetic.  It tests whether
four anonymous cells can acquire different procedural competence from public
outcome consequences and then cooperate through the unchanged anchored V13
composer parameterization and objective.  It does not claim unseen-procedure
acquisition, general reasoning, repository repair, or AGI.

Three paired replicates compare `uniform_adamw_plasticity` with
`learned_episodic_plasticity`.  Each arm starts from byte-identical fresh V13
controller, four-cell, composer, router, and zero AdamW state and receives 80
updates, eight streams per update, and 2,560 rows.  The arm orders are uniform
then learned, learned then uniform, and uniform then learned.  The uniform arm
runs the same two virtual meta folds and router optimizer update as the learned
arm but applies exact `0.25` cell allocations to its real update; this preserves
equal experience and meta-compute without letting its learned scores affect the
control.

Every update contains four procedures, each rendered twice as fresh topology
and surface variations A and B.  The canonical 80-by-4 commitment schedule is:

```text
2567 0126 0457 1467 2467 1357 0167 1247 0236 0567
0134 1237 1347 0456 2374 2456 0173 0273 0136 3476
1035 1276 3456 3475 1075 2034 1056 1046 2065 2063
2014 2145 2173 4165 4012 4275 3605 5712 4705 3670
5612 3015 3604 5102 3152 3524 4602 5723 6713 4760
4613 7201 6701 6723 5762 6702 4612 6513 4531 7240
6753 7530 6324 7634 6541 7340 5324 7520 3201 5401
4321 5240 7651 5320 5340 7410 6321 7654 7541 6532
```

Its canonical JSON payload has 801 bytes and domain-separated SHA-256
`8B18860D42DB4DF6979EBA3148CE94E817CF98D2A25014C58E15D34D46F8F7D1`.
It contains every four-subset once plus ten balanced repeats.  Every commitment
appears 40 times and exactly ten times in every seed slot; pair co-occurrence is
17 for 24 pairs and 18 only for `01`, `23`, `45`, and `67`.  Commitment
incidence signatures are distinct with Hamming distance 44 or 46.  Lane A,
lane B, and the combined real batch receive independent domain-separated hash
orders.  None of those ordering keys reaches a learned tensor.

Training model seeds are `(2026083401..404)`, `(2026083411..414)`, and
`(2026083421..424)`.  With replicate stride `10000000`, train A
topology/surface bases are `6001000001`/`6041000001`, train B bases are
`6081000001`/`6121000001`, panel A bases are
`6201000001`/`6241000001`, its rerender surface base is `6281000001`, panel B
bases are `6321000001`/`6361000001`, adaptation A bases are
`6401000001`/`6441000001`, adaptation B bases are
`6481000001`/`6521000001`, and probe bases are
`6601000001`/`6641000001`.  The separate four-update adaptation schedule is
`[[0,1,2,3],[2,3,4,5],[4,5,6,7],[6,7,0,1]]`; every commitment appears twice
in distinct seed slots.  Its domain-separated SHA-256 is
`6B449614DC824EF71022B622FF8348D444942D2F00701E6810BD119965F1D04D`.
All identities are disjoint from V12--V14 and from evaluation panels.

The router retains V14's shared permutation-equivariant 28-to-48-to-48-to-1
set scorer and softmax across four cells.  Its seven detached local features
are single-cell public loss, log gradient norm, within-cell gradient cosine
mean and minimum, prediction strength, raw Adam-moment alignment, and log
hypothetical preconditioned-step norm.  Cell, lane, pair, update, seed,
commitment, task, package, motif, topology, surface, and partition identity are
forbidden.  The final scorer starts at exact zero: update 0 must route exactly
uniformly, its finite nonzero first meta gradient may reach only the scorer,
and after that router step the next fresh route must differ from uniform by
more than `1e-6` while upstream router gradients are reachable.  This is an
explicit one-step structural warm-up, not evidence of competence.

Each cell-local consequence is measured through a homogeneous anonymous
four-slot lift: the target learned cell temporarily occupies all four
anonymous slots, the ordinary public V13 relation objective is evaluated, and
the exact original module list is restored in `finally`.  This removes the
physical-order leakage of V14's heterogeneous single-cell lesion without
exposing a cell identity or the contents of another cell.  Only anonymous
four-cell mean, softmax, fused-sum, and separation-cosine reductions use FP64
accumulators for numerical permutation stability.  Learned parameters, codes,
logits, optimizer state, and every returned live tensor remain FP32; no
sorting, canonical cell role, quantization, or fixed routing is introduced.
Tests cover the full transitive learned closure and require the real cell,
composer, router, functional/committed AdamW state, and optimizer state to
remain equivariant under a full physical-cell permutation within `1e-6`.

Both actual and virtual cell changes call one pure functional AdamW
implementation with beta values `0.9`/`0.999`, epsilon `1e-8`, zero weight
decay, no AMSGrad/maximize/capturable/fused/foreach variant, encoder learning
rate `3e-4`, and head learning rate `1e-3`.  Step advances before bias
correction.  A missing gradient skips a parameter and its state; an explicit
zero gradient advances it.  One global four-cell direction clip at `5.0`
precedes moments.  The implementation must match ordinary AdamW within
`atol=1e-7`, `rtol=1e-6` at the first and nonzero-moment steps.  Composer and
router use separately owned AdamW at `1e-3`.  Virtual folds receive detached
copies of parameters and moments and may not mutate controller state, optimizer
state, buffers, RNG, or `.grad` fields.  Checkpoints preserve every controller,
router, optimizer state, seed plan, and digest.

The same router independently processes the four A streams and four B streams;
the two lane-local routes are merged for one eight-stream real update.  This
matches the set context used in the two virtual folds.  The public anonymous
entropic weight is computed over exactly eight pre-update ensemble stream risks
for the real direction and exactly four source-lane risks for a virtual
direction.  Fold A-to-B applies a virtual routed AdamW change from A and
measures the unchanged current composer on B; B-to-A is symmetric.  Each target
loss is the same anonymous entropic aggregate over exactly its four post-update
stream risks.  The router objective is their mean plus `0.01` KL of aggregate
cell use to uniform.  It optimizes post-update loss directly: no subtractive
FP32 crumb, `1e4` multiplier, entropy floor, sharpening, top-k, fixed role,
vote, gradient surgery, replay, or deterministic solver is allowed.  Router
updates affect only the next fresh batch.

Whole-lane A/B swap must only permute recorded routes and folds while leaving
the combined real direction, meta objective and gradients, and committed state
equal within the fixed numerical tolerance.  Within-lane stream permutation
and cell permutation must be equivariant.  Tests also enforce parameter/state
ownership, exact paired starts and exposure, seed disjointness, hidden-field
exclusion, strict checkpoint reload, frozen evaluator hashes, and unchanged
shared parameters.

Fresh evaluation pools two eight-stream panels per replicate: 64 rows and 16
streams.  Surface rerendering must preserve discrete results exactly and keep
continuous change at most `1e-6`.  Post-first learned routes must have mean
total variation from uniform at least `0.10`, median at least `0.075`, and each
cell must receive 10--40 percent aggregate use.  After undoing presentation
shuffles, matched A/B route distance is averaged over the four matching pairs;
unmatched distance is averaged over all twelve ordered nonmatching A-by-B
pairs.  Updates 1--79 are macro-averaged within replicate and then across the
three replicates.  Require `D_pair <= 0.75*D_unpaired`, a gap at least `0.025`,
and a per-replicate gap at least `0.015` in two of three replicates.

Absolute learned competence requires at least 120/192 supported rows and 18/48
qualifying streams, with every replicate at least 32/64 and 4/16.  Against the
uniform arm, learned routing must gain at least six rows and three streams,
reduce loss by at least `max(1e-4, 2% of uniform loss)`, win loss by at least one
percent in two replicates, and avoid row and stream regression in those same
replicates.

The four-update fresh adaptation ablation clones the complete terminal learned
state into `correct`, `uniform`, `cell_permuted`, and `no_update`.  The first
three continue exact cell AdamW state; no composer or router step occurs.
Correct must reach at least 48/96 rows and 6/24 streams, with every replicate at
least 12/32 and 1/8.  Against every control it must gain at least three rows and
one stream, reduce loss by at least `max(1e-4, 1% of control loss)`, and produce
the material loss win in at least two replicates.  Thus merely degrading less
than another updater cannot pass.

Normal learned read must beat forced-uniform, every single-cell, and every
drop-one read in aggregate by at least two rows, one stream, and
`max(1e-4, 1% of comparator loss)`.  For every comparator separately, at least
two replicates must have the same material loss win with row and stream
nonregression.  In at least two replicates, at least two distinct cells must
each be the unique best cell on at least two of sixteen streams, where a unique
winner is at least `1e-4` below the second-best cell loss.

Classification precedence is `INVALID_NO_CLAIM` for any integrity failure;
`PLASTICITY_ROUTER_HARMFUL` only when a competent uniform arm is materially
better; `NO_COMPETENCE` when learned absolute floors fail;
`PLASTICITY_ROUTER_SUPPORTED` only when every frozen functional, routing,
adaptation, harmonization, and integrity rule passes; and
`PLASTICITY_ROUTER_NULL` otherwise.  Parameter or digest change, uniform
routing, zero competence, a `0>=0` comparison, or a sub-margin loss ordering
can never support V15.

Implementation and all CPU structural/twin tests must pass before one GPU fit
invocation.  The fit contains three replicates, both arms, 480 real updates,
3,840 streams, 15,360 rows, and 960 virtual folds.  There is no early stopping,
threshold change, seed replacement, result-conditioned continuation, rescue
run, or historical checkpoint reuse.  A semantic result is preserved and not
rerun.  Expected runtime is 40--60 minutes with a conservative 90-minute
envelope; runtime is not a success criterion.

## V15 precondition abort and fixed V16 numerical successor

The one V15 invocation is consumed as
`INVALID_NO_CLAIM / PARTIAL_PRECONDITION_ABORT`.  It emitted `FIT_INVOKED` but
stopped at the combined update-zero warm-up predicate before any evaluation,
checkpoint, or terminal report.  The result targets remained absent.  Because
the runner emitted neither arm progress nor the individual failed predicate,
no scientific outcome may be inferred.  The append-only bounded record is
`outputs/phase6-software-pipeline-reconstruction-v15-plasticity-summary.json`.
V15 must not be rerun.

The fixed successor identity is
`phase6.public-anonymous-cross-variation-plasticity.paired.v16`.  It changes no
architecture, exposure, schedule, learning rate, optimizer hyperparameter,
scientific threshold, evaluator, or classification rule.  It retains the same
three paired replicates, 80 updates per arm, eight streams per update, balanced
commitment/adaptation schedules, uniform comparator, four-cell learned arm,
evaluation panels, adaptation controls, and harmonization lesions.  The V15
schedule digests remain exact.

V16 uses fresh model seeds `(2026083601..3604)`, `(2026083611..3614)`, and
`(2026083621..3624)`.  Every V15 topology and surface seed in training,
evaluation, adaptation, and probe records is increased by exactly
`1000000000`; all resulting identities must be disjoint from V12--V15.  Lane
and real-batch presentation-order tuples are copied byte-for-byte from the
corresponding V15 plan; they are not rehashed under the V16 protocol identity.
This isolates the numerical repair from an additional floating-point ordering
change.  The order tuples carry no learned identity.  Both arms in a replicate
remain byte-identical at start and receive the same transformed stream binding.

The only learning-path repair is the exact-zero derivative of functional
AdamW.  V15 computes `sqrt(exp_avg_sq)` directly; at an exactly zero direction
its backward path is singular even though the composite AdamW update has a
finite derivative.  V16 computes the square root from
`exp_avg_sq.clamp_min(torch.finfo(dtype).tiny)`.  For FP32 live state this must
remain bit-equal to the V15 functional forward update for zero and nonzero
gradients and retain the already frozen ordinary-AdamW parity of `atol=1e-7`,
`rtol=1e-6`; V15 and PyTorch are already known to differ by up to one FP32
rounding step, so three-way byte equality is not claimed.  The zero-direction
VJP must be finite on CPU and CUDA.  No multiplier, gradient floor, tolerance
relaxation, clipping change, or loss change is allowed.

Before any V16 fit, a no-update structural preflight runs both fresh twins for
all three replicates on CPU and CUDA.  It may build only update-zero public
training batches and records exact-uniform routes, routed-direction zero counts,
per-parameter finiteness, nonzero counts, maximum magnitude and FP64 norm, plus
before/after controller, router, cell-optimizer and owned-optimizer digests.
It performs no optimizer step, evaluation panel, classification, checkpoint,
or scientific claim.  Every gradient must be finite and the zero-initialized
scorer must leave every upstream gradient exactly zero.  Scorer magnitude and
the later update-one route deviation are diagnostics, not abort thresholds;
failure to learn enough routing is decided only by the unchanged terminal
support rules.  During the fit, every meta, composer, and cell-direction tensor
must be finite before its owning optimizer can step.

The semantic fit uses deterministic single-thread CPU execution because the
failed V15 run measured this small sequential graph as GPU-kernel-bound, while
the same no-update warm-up completed substantially faster on CPU.  CUDA
preflight retains backend portability evidence; choosing CPU changes compute
placement, not the model or evidence.  The V16 runner may emit arm-boundary
progress events that contain no adaptive metric.  One fresh V16 fit invocation
produces
`/opt/angler/results/phase6-software-pipeline-reconstruction-v16-plasticity.json`
and the matching `.pt` checkpoint, followed by the bounded local summary
`outputs/phase6-software-pipeline-reconstruction-v16-plasticity-summary.json`.
There is no early stopping, seed replacement, result-conditioned continuation,
or rescue run.  Expected CPU runtime is 45--90 minutes with a conservative
120-minute envelope; runtime is not a success criterion.

## V16 result and V12 champion retention

The one V16 semantic invocation completed all six arms, 480 optimizer updates,
3,840 streams, and 15,360 rows.  The exact-zero AdamW repair succeeded and all
three learned replicates reached the absolute competence floor in aggregate.
The frozen result is nevertheless `INVALID_NO_CLAIM`: five of six surface
rerenders exceeded the predeclared `1e-6` continuous-stability ceiling, learned
routing remained almost uniform (`0.00508` mean total variation versus `0.10`),
only one replicate beat uniform, and the paired-advantage, fresh-adaptation,
routing, and harmonization checks all failed.  Its strict report and checkpoint
reload passed with no lineage discrepancy.  Full hashes and measurements are
preserved in
`outputs/phase6-software-pipeline-reconstruction-v16-plasticity-summary.json`.
V16 is consumed and will not be rerun or promoted.

V12 remains the measured champion.  It passed relation at `24/32` supported
rows and `6/8` qualifying streams, then failed only context selection: `14/24`
supported rows placed a valid slot first (`0.58333` versus `0.80`) and valid-set
mass was `0.45652` versus `0.60`.  Its C25 objective did fall from `1.175082`
to `0.944844`, but real-slot weights remained nearly uniform: mean max-minus-min
spread was only `0.000164`.  Training mainly suppressed the null slot instead
of distinguishing the three real memories.  Relation representation and V12's
learned conflict updater are therefore retained; the next intervention is
context-only.

## Fixed V17 V12-champion context-residual successor

The fixed successor is
`phase6.public-v12-champion-context-residual.v17`.  It loads only terminal V12
checkpoint SHA-256
`B4DA4550D18C9F1480903DA087A8E7799341763F1EDD63061E8A04A7491BD62C`
and preserves every inherited controller tensor, the conflict mixer, and the
competence state byte-for-byte.  It overrides only `_context_pair_logits` in a
new additive runner; relation encoding, relation comparison, evidence
acquisition, state shape, action scoring, and all V12 source files remain
unchanged.

Four anonymous rank-8 context experts receive the existing 32-dimensional
context codes.  Expert `e` computes

```text
x_e = x + Up_e(SiLU(Down_e(x)))
```

with bias-free `32→8→32` projections, independently seeded `Down`, and exact
zero `Up`.  The same frozen V12 `relation_context_comparator` scores every
adapted query/stored pair.  The four logits are fused by the existing
FP64-stable V15 implementation of V13's anonymous soft composer at width 32,
hidden width 41, and anchor `0.5`.  No cell, stream, task, package, motif,
topology, surface, partition, or answer identity is an input.  There is no
hard routing, vote, deterministic solver, replay, stored example, V16 write
router, or second model.

Only these prefixes may train:

```text
context_residual_experts.0..3.{down,up}.weight
context_composer.{local_encoder,residual_scorer}.*
```

They contain exactly 17 tensors and 10,007 scalars.  The V12 controller remains
265,606 parameters, the frozen mixer remains 3,404, and the complete successor
is 279,017 parameters (`+3.72%`).  At step zero every expert output is exactly
zero, all four cell logits equal V12, the composer emits exact quarter weights,
and its FP64 four-quarter sum casts to the original FP32 V12 logit exactly.
Thus context/null weights, relation logits, final evidence scores, runtime
state, and action scores must be bit-identical before learning.  This is a
single expanded reasoning core, not a V12 fallback running beside a challenger.

The successor retains C25 exactly: 25 updates, eight fresh public streams per
update, four rows per stream, AdamW with zero weight decay, clip `5.0`, expert
learning rate `3e-4`, and composer learning rate `1e-3`.  Only public
relation-derived valid-set feedback is used; detached relation margins define
the valid set and never update the relation path.  Expert seeds are
`2026083701..2026083704` and composer seed is `2026083705`.  Training identity
`u=0..24`, anonymous commitment `c=0..7` uses topology/surface seeds
`8001000001 + 100000*u + 1000*c` and
`8041000001 + 100000*u + 1000*c`.  The one fresh panel uses
`8081000001 + 1000*c` / `8121000001 + 1000*c`; its surface rerender uses
`8161000001 + 1000*c`.  All are disjoint from V12--V16.

The successor checkpoint persists canonical name-keyed AdamW moments and exact
step counts for all 17 allowed tensors together with the learned weights.  This
lets a later successor inherit the acquired adaptation trajectory without
replay or an optimizer reset; it does not authorize a second V17 run.

Pre-run tests must prove strict V12 source migration, all inherited-key and RNG
exactness, complete step-zero context/null/relation/final-score equality, and
finite nonzero first credit into every expert `Up` projection.  The expert
`Down` projections and composer may have exact-zero first-step gradients because
the zero-output experts are initially identical; after the first expert update,
the first fresh post-divergence batch must provide finite nonzero composer credit.
Tests must also prove later expert divergence, an exact 17-tensor optimizer and
mutation allowlist with no inherited mutation, and finite reachability of the
staged composer.  Exact-zero symmetry can leave some deeper allowed tensors
byte-unchanged during a short opening test; every tensor need not change merely
to satisfy a count.  Tests also cover cell/query/slot permutation symmetry, V12
relation-boundary retention, strict successor checkpoint reload/tamper
rejection, and exclusion of identity, replay, solver, router, and V16
dependencies.

One fresh run follows the tests.  Integrity failure is `INVALID_NO_CLAIM`.
Both the fresh panel and its disjoint surface rerender must independently pass
the unchanged context gate (`>=0.80` supported-row top-one and `>=0.60`
supported-row valid-set mass), preserve identical relation-supported masks and
counts, and show a same-panel causal residual-on advantage over the analytically
available residual-off V12 path of at least three additional supported-row
top-one successes and `0.05` valid-set mass.  Their discrete context top-one
decisions must be exact across surfaces; corresponding context/null weights,
valid-set masses, and final evidence scores may differ by at most the inherited
`1e-6` ceiling.  Exact inherited V12 bytes and relation-boundary values are also
required.

On both surfaces, harmonization requires learned composition to nonregress both
context metrics versus forced uniform and improve either by at least two
top-one rows or `0.02` mass, plus at least two distinct drop-one expert lesions
that each lose one top-one row or `0.01` mass.  These rules are fixed before
execution.  Diagnostics may not rescue either failed context gate.  The run
stops after the context result: joint training remains a separate successor
decision.

The implementation surfaces are only
`experiments/runners/phase6_v12_champion_context_residual.py` and
`tests/unit/experiments/test_phase6_v12_champion_context_residual.py`, followed
by a one-shot harness and bounded result summary.  ACL/SRWM-style persistent
fast updater state is the next donor mechanism if this learned context capacity
works; adding it now would combine the observed read-selection defect with a
second untested write/retention change.

## V17 result and corrected bottleneck

The one V17 invocation is consumed as `INVALID_NO_CLAIM`.  Its implementation
and state lineage were sound: all 17 allowed tensors received finite credit,
changed, and restored at optimizer step 25, while all 338 inherited controller,
9 mixer, and 18 competence tensors remained byte-exact.  The added mechanism
was nevertheless causally inert.  On the base surface it produced `8/21`
supported-row top-one, `0.433023` valid-set mass, zero additional top-one rows
and only `0.000682` mass over residual-off.  Learned composition differed from
uniform by about `7e-9` mass, all four expert lesions lost zero top-one rows,
and their largest mass effect was `5.55e-5`.  The independent rerender repeated
the same failure.  Exact artifacts, hashes, and the independent reload audit
are preserved in
`outputs/phase6-software-pipeline-reconstruction-v17-context-residual-summary.json`.

V17 therefore falsifies additional transformations of the already-pooled
32-dimensional context code as the sufficient repair.  The upstream V12
context path still reduces each rich predecessor pair tensor to one code by
flattened attention/mean/max pooling.  Its C25 objective also admits a shortcut:
null mass fell from `0.272914` to `0.060819`, while real-normalized valid mass
fell from `0.47618` to `0.46668`; unsupported rows received no loss.  Unique-
valid real-slot rank remained essentially tied.  The next intervention must
both preserve anonymous row/column incidence before that pool and place direct
credit on real-slot discrimination.  More downstream experts, another mixer,
or SRWM behind the collapsed code is not authorized by this result.

V17's raw cross-surface array comparison also failed to undo a public candidate
permutation.  This is corrected prospectively by canonical alignment; it does
not reclassify V17 because both surfaces independently failed every causal
capacity and harmonization requirement.

## Fixed V18 incidence-aware contrastive context successor

The fresh successor is
`phase6.public-v12-champion-context-incidence.v18`.  It again loads only the
exact terminal V12 checkpoint SHA-256
`B4DA4550D18C9F1480903DA087A8E7799341763F1EDD63061E8A04A7491BD62C`.
Every inherited controller tensor, conflict-mixer tensor, competence value,
relation computation, action path, and existing context path remains frozen.
V17 weights are not inherited.

One context-specific instance of the existing `RelationAxisSetReadout` receives
the raw context pair tensor before flat pooling.  Its anonymous row attention,
column attention, diagonal retention, node projection, and node-set pooling
preserve shared endpoint incidence under simultaneous node renaming.  A new
bias-free `4*width -> width` projection is initialized exactly to zero and its
output is added to the existing context precode immediately before the existing
normalization.  This is the function-preserving V9 structural-residual pattern
reused on the context side, not a selector or hand-coded graph solver.  At
width 32 the new branch owns exactly 10 tensors and 16,992 scalars; the complete
system is 286,002 parameters (`+6.32%` over V12).  Step-zero production output
must be bit-exact V12.  The first public update must open the zero projection
and later updates must reach the incidence trunk.

V18 replaces the shortcut-prone C objective only for the new branch.  With
relation margins detached, let `S` be rows with at least one valid real slot,
`I` the informative rows with between one and `R-1` valid real slots, `U` the
subset of `I` with exactly one valid real slot, and `N` the unsupported rows.
An informative row uses

```text
L_rank = -log(sum(valid real-slot mass) / sum(all real-slot mass))
L_presence = -log(sum(all real-slot mass))
```

and an unsupported row uses `L_abstain = -log(null mass)`.  Each update minimizes
mean `L_rank` over `I`, plus `0.25` times mean `L_presence` over `S`, plus `0.25`
times mean `L_abstain` over `N`, omitting only a term whose set is empty.  Thus
raising every real logit cannot improve the primary term; all-valid rows neither
create nor dilute rank credit; unsupported rows finally teach abstention.  No
responsibility, hidden target, task identity, replay item, task-specific
deterministic solver/selector/relabeling result, or hard assignment enters the
branch.  Existing deterministic public tensorization and incidence-graph
construction remain shared representation plumbing, not a solution procedure.

The fixed fit contains 256 updates, eight unique public streams per update,
four rows per stream, no replay, AdamW with zero weight decay, clip `5.0`,
`3e-4` incidence-trunk learning rate, and `1e-3` output-projection learning
rate.  This is 2,048 unique streams / 8,192 rows rather than another C25 probe.
The context readout seed is `2026083801`.  Training update `u=0..255` and
anonymous stream `c=0..7` use topology/surface seeds
`9001000001 + 100000*u + 1000*c` and
`9101000001 + 100000*u + 1000*c`.  Fresh panel `p=0..3`, stream `c=0..7`
uses `9201000001 + 100000*p + 1000*c` and
`9301000001 + 100000*p + 1000*c`.  All identities are disjoint from V12--V17
and are part of the frozen plan digest.  No early stop, seed replacement,
adaptive threshold, rescue run, or second V18 identity is allowed.

Tests must prove exact V12 migration and step-zero equality; the exact ten-
tensor mutation/optimizer allowlist; staged gradient reach; finite strict
checkpoint continuation; simultaneous node-renaming and row/column permutation
invariance; and sensitivity to different endpoint incidence when the flattened
cell multiset is held fixed.  They must also reject IDs, V17/V16 mechanisms,
SRWM, replay, stored answers, routers, and task-specific deterministic solvers,
selectors, or relabeling procedures.

Evaluation uses four disjoint fresh topology panels and canonical candidate
alignment derived only from public candidate content inside the evaluator;
latent IDs, commitment indices, and alignment labels never enter the model.
Learned incidence and projection-zero lesion share the same live system and
evidence; every causal delta is learned minus that same-checkpoint lesion and
no separate frozen-model training arm is required.  Primary effect evidence is
strict real-slot argmax success over `U` with null excluded, real-normalized
valid mass pooled over the union of `I`, and the pooled `I` margin
`log(max valid real weight) - log(max invalid real weight)`, which is the
temperature-scaled pre-softmax real-logit margin and is invariant to null mass.
Null reduction alone and all-valid rows cannot establish or dilute selector
improvement.

A panel recurs positively when both the `U`-top-one and pooled-`I`-mass causal
deltas are nonnegative and either it adds at least one `U` top-one success or
`0.01` pooled-`I` real-normalized mass.
The fourth panel is nonregressed when its top-one delta is at least `-1` and its
mass delta at least `-0.01`.  A supported V18 component requires recurrence on
at least three panels, nonregression on all four, an aggregate sum across `U`
of at least 12 additional top-one successes, a pooled-union-`I`
real-normalized-mass gain of at least `0.05`, a learned pooled-`I` rank margin
above zero with a causal margin delta above zero, and a projection-zero lesion
that removes those material gains while inherited relation masks and values
remain exact.  Full context advancement additionally requires aggregate
inherited relation support of at least `96/128` rows and `24/32` qualifying
streams.  Its `0.80` top-one and `0.60` full valid-set-mass gates are pooled over
all relation-supported rows across the four panels; the latter includes null
normalization and is distinct from the causal real-normalized mass.  A
component-effect pass without those full-system floors is useful bounded
evidence but does not replace V12 as the system champion.

Only after V18 establishes a discriminating read is the existing Angler-native
ACL/SRWM state eligible for the next successor.  That later writer must use the
ACL boundary objective that queries old abilities immediately after each new
write and must make the state residual causally unavoidable; the earlier Phase-2
slow-path bypass may not recur.

## V18 result: paired comparison, not larger independent pooling

The single frozen V18 invocation completed as
`CONTEXT_INCIDENCE_NOT_SUPPORTED`.  The run executed exactly 256 updates over
2,048 streams / 8,192 rows, all ten new tensors received gradients and changed,
all inherited V12 controller/mixer/competence bytes remained exact, and the
terminal checkpoint passed strict reload.  Exact claim/report/checkpoint hashes
are `FB551D9B...4283`, `E68099CE...A7EE`, and `3FF20A47...5EEF`; the bounded
record is
`outputs/phase6-software-pipeline-reconstruction-v18-context-incidence-summary.json`
at SHA-256
`38C8C6C313789086288DFA2F394D878F6EA3F7A24C18A9D71D1AA34E9DDC506A`.

The negative result is decisive.  Real-slot rank loss moved from `1.057856` to
`1.062132`, unique-valid top-one was exact chance at `24/72` and one below the
same-checkpoint projection-zero lesion, and pooled informative real-normalized
mass was `0.346665` versus chance `0.346667`.  The apparent total-loss gain came
from a common-mode null compromise: abstention improved from `2.799811` to
`0.815993`, presence worsened from `0.062758` to `0.583757`, all real weights
became nearly equal, and null mass converged near `0.4422` on supported and
unsupported rows alike.  The learned residual reached mean norm `35.19` against
the inherited normalized code's norm `1.0`, became effectively rank one, and
collapsed final-code variation.  Live V18 therefore erased rather than extended
V12: full top-one fell from the lesion's `43/91` to `0/91`.

This was not missing public information.  A read-only structural diagnostic over
all 128 frozen-panel predecessor graphs found zero anonymous-topology signature
collisions and recovered the latent motif relation from topology in `128/128`;
those diagnostic signatures and labels never entered the model.  The failed
interface independently embedded each non-isomorphic graph, compressed it to one
vector, and delegated the relation between two graphs to a frozen point
comparator.  Increasing that same pool, adding more updates, retuning null loss,
or attaching SRWM behind it is not supported by the evidence.

## Fixed V19 paired graph-context successor

The fresh successor is
`phase6.public-v12-champion-paired-graph-context.v19`.  It restores only the
exact terminal V12 checkpoint SHA-256
`B4DA4550D18C9F1480903DA087A8E7799341763F1EDD63061E8A04A7491BD62C`;
V17 and V18 learned weights are excluded.  All 338 inherited controller, nine
conflict-mixer, and 18 competence tensors remain frozen.  V19 adapts the
cross-graph attention and mismatch-message principle from DeepMind's Graph
Matching Networks (Apache-2.0), rather than importing its TensorFlow/Sonnet
runtime.  The inspected donor is
`google-deepmind/deepmind-research@f5de0ede8430809180254ee957abf36ed62579ef`,
with GMN subtree last-change commit
`451d2964904a4e71d8d28ac45cdc5f33c1db1b19`.

V19 keeps each public predecessor graph as raw anonymous adjacency until the
query and one stored graph are compared.  Before any learned encoding, the
node mask selects the exact induced active adjacency; padding never enters the
pair encoder.  The frozen V12
`EvidenceOrderedPairEncoder(active_adjacency, zero_candidate)` first produces
its raw pair states at the true node count.  Resulting node tokens may then be
padded only for masked cross-attention and pooling.  A trainable context-axis node encoder applies two-head row and
column attention, diagonal retention, and a shared node projection but does not
pool the node set.  For query tokens `Q` and stored tokens `S`, one shared GMN
round computes bidirectional masked cross-attention and mismatch messages:

```text
A_qs = softmax(Q S^T / sqrt(W), over stored nodes)
A_sq = softmax(Q S^T / sqrt(W), over query nodes)
D_q  = Q - A_qs S
D_s  = S - A_sq^T Q
Q'   = Q + Update([Q, D_q])
S'   = S + Update([S, D_s])
```

Independent masked mean/max pools are formed only after this paired exchange.
A symmetric scorer receives `[0.5*(poolQ+poolS), abs(poolQ-poolS),
poolQ*poolS]` and emits one real-slot residual.  At width 32 / hidden width 64,
the context-axis node encoder is exactly `row-attention + column-attention +
LN(5W)->Linear(5W,H)->SiLU->Linear(H,W)` and owns eight tensors / 12,832
scalars.  The shared GMN update is exactly
`LN(2W)->Linear(2W,H)->SiLU->Linear(H,W)` and owns six / 6,368.  The symmetric
pair scorer is exactly
`LN(6W)->Linear(6W,H)->SiLU->Linear(H,W)->SiLU->Linear(W,1,bias=False)`
and owns seven / 14,848; its final bias-free linear is initialized exactly to
zero.  V19 therefore owns exactly 21 trainable tensors / 34,048 scalars and
303,058 complete parameters (`+12.66%` over V12).

The residual cannot repeat V18's null shortcut.  Let inherited real-context
logits be `l`, bounded paired residuals be `r = 0.5*tanh(raw)`, and inherited
context temperature be `tau = 0.25`.  Across all occupied real slots V19 uses

```text
c  = tau * (logsumexp((l+r)/tau) - logsumexp(l/tau))
l' = l + r - c
```

so `sum(exp(l'/tau))` equals `sum(exp(l/tau))`.  The inherited zero null logit,
null probability, and total real probability are therefore preserved, while
relative real-slot ranking can change.  One or zero occupied graph slots has an
exact-zero effective residual.  The primary lesion forces `r=0` and must
recover terminal V12 context, relation, evidence, and production action scores
exactly.

This is an online Angler capability, not an evaluator-only head.  V19 versions
task encoding with a padded raw graph and mask per action, and versions live
state with raw graph adjacency/masks aligned atomically to the same role-memory
trace slots as V12 context keys and relation values.  Raw adjacency is stored,
never learned node embeddings, so later matcher updates do not stale old
memories.  The smoke identity uses a declared 32-node capacity (current graphs
have 18 nodes), rejects rather than truncates node 33, and treats that value as
an execution-profile ceiling rather than a permanent architecture limit.
Resource profiles must declare their own capacity from available memory.
Terminal V12 trace memory is verified empty; any future nonempty legacy state
without raw graphs must fail rather than invent history.

Ordinary `score_actions` first obtains V12 scores, then uses the common paired
graph-read method and adds only the difference between graph-aware and V12
evidence-action contributions to real production actions; STOP, transition,
relation, and other inherited paths remain exact.  V19 acquisition writes raw
graphs to the exact accepted role ring slots or rolls back the complete state.
V19 public credit rows call the same paired graph-read method used by ordinary
production scoring.  Base V12 encoding/state or the V12 credit-row builder in a
V19 fit is a hard error, preventing an auxiliary-evaluator bypass.

The fixed fit contains 512 updates, eight unique public streams per update and
four rows per stream: 4,096 streams / 16,384 rows.  AdamW uses zero weight decay,
clip `5.0`, `3e-4` for the node encoder/GMN update and `1e-3` for the pair
scorer/head.  Seeds are `2026083901`, `2026083902`, and `2026083903` for those
modules.  Update `u=0..511`, stream `c=0..7` uses topology/surface seeds
`9401000001 + 100000*u + 1000*c` and
`9501000001 + 100000*u + 1000*c`; panel `p=0..3`, stream `c=0..7` uses
`9601000001 + 100000*p + 1000*c` and
`9701000001 + 100000*p + 1000*c`.  There is no replay, early stop, seed retry,
rescue identity, or adaptive threshold.

Only informative rows `I` with one through `R-1` valid real slots train V19:

```text
L_list = -log(sum(softmax(real_logits / 0.25)[valid]))
L_pair = mean_valid,invalid 0.05 *
         softplus((0.10 - (logit_valid-logit_invalid))/0.05)
L = mean_I(0.5*L_list + 0.5*L_pair)
```

Unsupported and all-valid rows contribute no V19 loss.  Presence, abstention,
null calibration, relation margins, evaluator identities, motif/gap signatures,
canonical matches, hard assignments, replay, and deterministic solver output
provide no gradient or model input.  The zero head must receive first-update
credit; later updates must reach all 21 tensors.

Evaluation retains V18's four-panel causal requirements: positive recurrence on
at least three panels, nonregression on all four, aggregate `U` top-one gain at
least `+12`, pooled-`I` real-normalized-mass gain at least `+0.05`, learned
pooled-`I` margin above zero, and causal margin gain at least `+0.05`.  The
evaluation margin retains V18's exact definition
`log(max valid real weight)-log(max invalid real weight)`; it is the
temperature-scaled real-logit margin rather than an unscaled raw-logit
threshold.  Relation masks/values must be exact under the zero-residual lesion.
Full V12 replacement still requires at least `96/128` relation-supported
rows, `24/32` qualifying streams, `0.80` supported-row full top-one and `0.60`
supported-row full valid-set mass.  Each attribution lesion independently--
uniform cross-graph attention and mismatch-zero--must remove at least half the
causal margin gain and either four top-one gains or `0.02` mass; otherwise any
component effect is recorded without claiming GMN-specific attribution.

Preflight is wiring-only: exact V12 migration/step-zero production equality,
one two-update staged-gradient check, independent query/stored node permutation
invariance, unequal node counts, padding, directed-edge sensitivity, slot and
candidate covariance, state ring alignment/rollback, common training-production
matcher use, production-logit lesion reach, exact 21-tensor mutation scope, and
strict snapshot/checkpoint continuation.  It may not report competence, tune a
threshold or seed, or decide execution from tiny-data accuracy.  No semantic
V19 fit begins until these properties and the frozen one-shot harness pass
independent review.

---
blueprint_id: ANG-WORK-LEARNING-OML-PERSISTENT-LIFELONG-V21-001
parent_id: ANG-BP-RETENTION
revision: 4
tier: 4
design_status: ready
delivery_status: not_started
human_authority: project owner continuation direction, 2026-08-29
human_impact: LOW; contained local synthetic continual-learning experiment
human_flourishing_gate: ANG-GATE-HUMAN-FLOURISHING-001@1-NOT_RUN
depends_on:
  - ANG-WORK-LEARNING-SOFTWARE-PIPELINE-RECONSTRUCTION-V20-OML-001@1
---

# V21-A persistent OML lifelong bridge

Protocol: `phase6.public-oml-persistent-lifelong.v21a`  
Active node: `ANG-BP-LEARNING` / `ANG-BP-RETENTION`  
Status: ready for implementation and preflight; semantic identity not run

## Accountable outcome

Test whether V20's confirmed OML representation can support **one fixed-size
persistent learned state** through a substantially longer, changing sequence:
acquire fresh public abilities, retain earlier acquired effects without replay,
and transfer to update-heldout two-motif compositions.

The foundation model and every slow Angler parameter remain frozen.  V21-A
adds no parameter, slow training, replay system, task router, ANML mechanism,
consolidator, deterministic solver, or new evaluator path.  It is the smallest
experimental bridge from the learned-updater evidence in Slice 05 toward the
retention/composition question in Slice 06.  Even a pass is bounded synthetic
evidence, not normal RETENTION delivery, Slice 06, M5, broad-domain reasoning,
or AGI.

## Frozen source and borrowed idea

V21-A reuses the independently implemented OML slow-representation / fast-head
split from Javed and White, *Meta-Learning Representations for Continual
Learning* (NeurIPS 2019):
<https://papers.nips.cc/paper_files/paper/2019/hash/f4dd765c12f2ef67f98f3558c282a9cd-Abstract.html>.
No donor source, framework code, sampler, configuration, or checkpoint is
copied.

Frozen Angler identities:

- V20 execution/continuation leaf path:
  `docs/blueprints/branches/learning/work/ANG-WORK-LEARNING-SOFTWARE-PIPELINE-RECONSTRUCTION-V20-OML-001.md`
- V20 frozen execution-leaf SHA-256 (the exact bytes that governed the
  consumed run):
  `BB2CCDDB80B25ACD5B79CB4DDC52F347ED77C8F843D972B7D4049C2B4546F257`
- V20 current post-result continuation-leaf SHA-256 (status/evidence bytes,
  not a replacement execution identity):
  `1C0130928F77D80F2AAA9047E44DD02B4DEDEE951CEB2E41837EAA2A086B66F5`
- V20 runner SHA-256:
  `6611E60BAB8D1F3C80A68BEB66AAC010F236B107B2A5E9060201BA56A50E86E3`
- V20 terminal checkpoint:
  `/opt/angler/results/phase6-software-pipeline-reconstruction-v20-oml.pt`
- V20 checkpoint SHA-256:
  `D49E4CAAB64A264A11C675B295A8C453AC4475F078311EB7283A4F9A8817EF48`
- V20 terminal report SHA-256:
  `5CCCBF0CE8211E0CC99AEB856145BF4CD3D9EA30A1ECB3FAE8E9435B4689C498`
- V20 terminal system digest:
  `sha256:4c8e1f5df037956e01ab59353df45cf114c76385cca5d77c0c632e633d7614c3`
- frozen V19/V12 runner SHA-256:
  `54A8E2E510424E485DE34A2975A82C927D22C87B5576EFE00537545158ECE5BE`
  / `F1045756E77D60A7968265867035CEA55BFFE8BF6E1A73AB50C12A719EC8B529`
- frozen V19 source checkpoint:
  `/opt/angler/results/phase6-software-pipeline-reconstruction-v19-paired-graph-context.pt`
- frozen V19 source-checkpoint SHA-256:
  `10BB6BAC9BD83F7F4EE0ABF2846CE4133D2133790C2B55113C9044930D2EBC7F`
- V16 functional-AdamW source SHA-256:
  `EB1A29AC78670C6A0ECDED943E17AA62B1CFB91BF58DAB1ADC9001A3B75D63AB`

The V20 first result is consumed and remains
`OML_V19_HARMONIZED_ADVANCEMENT`; V21-A does not rerun or tune it.  V20's
second-order arm supplies the candidate slow representation, its first-order
arm supplies one attribution control, and its untouched source-V19 state
supplies the other.  The supported V19 paired whole-graph context comparison
remains exact.  V11-D2 fixed `d2_same_module_overlap=false`, so ANML remains
unauthorized by this lineage.

## One persistent plastic state

The only online-writable candidate state is:

```text
fast relation_comparator.2.weight: 64 FP32 values
Adam first moment:                    64 FP32 values
Adam second moment:                   64 FP32 values
optimizer step:                        1 integer
```

The 192 floating values occupy 768 bytes and remain constant in size regardless
of experience count.  The fast weight begins at V20's frozen `W0`; moments and
step begin at exact zero.  After every online update, the weight, both moments,
and step are detached before the next experience.  No autograd history,
example, trace, task identity, or per-ability state persists.  Exact snapshot,
restore, digest, reset, swap, and checkpoint semantics are required.

Every slow tensor, buffer, V20 outer optimizer, V19 context tensor, base model,
and source checkpoint remains frozen and byte-identical.  Online loss requests
gradients only with respect to the current 64-value fast weight.  Each update
uses V16 `functional_adamw_step` with V20's unchanged settings: learning rate
`1e-3`, betas `(0.9, 0.999)`, epsilon `1e-8`, and zero weight decay.

## Four causal arms and no-update measurements

All arms consume the same immutable generated public streams in the same
chronology.  Sharing a stream object across matched arms is counterfactual
measurement, not learner replay.

1. `second_order_oml_persistent` — primary candidate: V20 second-order slow
   representation and one fast state carried without reset through A and B.
2. `first_order_meta_persistent` — V20 first-order slow representation and its
   own persistent fast state.
3. `source_v19_persistent` — untouched source-V19 representation and its own
   persistent fast state initialized from the same `W0`.
4. `second_order_boundary_reset` — byte-identical to the candidate throughout
   Stage A, then resets weight to `W0`, moments to zero, and step to zero exactly
   once at the A-to-B boundary.

`second_order_no_update` is a measurement control, not a fifth learned arm.  It
uses the candidate slow representation and fixed `W0`, performs no optimizer
step, and supplies matched AUC, loss, coverage, reset, and static-representation
metrics.  `first_order_no_update` and `source_v19_no_update` are likewise
stateless Stage-B AUC measurement controls matched respectively to
`first_order_meta_persistent` and `source_v19_persistent`; they are not learned
arms and create no persistent state.  No arm selects a state, adapter, model,
or update rule from query or mechanism identity.

## Public-only generalized row boundary

V21-A may add one successor-local generalized V19 public-row adapter solely
because frozen V19 restricts its production credit builder to the train
partition.  The adapter may consume only the same public task, raw anonymous
graph, public trace, public candidate, and public outcome material used by V19.
It receives no partition answer, semantic/mechanism index, commitment value,
motif identity, hidden implementation, judge result, reference pipeline,
solution route, hidden valid-set answer, or evaluator-private object.  The
unchanged public V19 credit labels remain permitted only where their provenance
is already public learner-visible feedback.

On train streams, the generalized builder must be bit-exact in row order,
values, masks, graph tensors, and duplicate handling, and gradient-exact for
the 64-value fast weight and every V20 RLN tensor, against frozen V19/V20.  It
must call the same frozen controller scoring methods used by ordinary V19
credit; it may not copy the controller class, replace `score_actions`, or make
an evaluator-only learned head.  A mismatch is an invalid implementation, not
an adaptive result.  No core or evaluator file may change.

## Frozen experience schedule

All streams use `supports_per_motif=2`, `queries=1`, `maximum_steps=4` and the
unchanged V20 public row/stream objective.  For each row:

```text
row_loss = _paired_relation_margin_loss(P, N)
         + 0.25 * _relation_instance_losses(p_q, n_q)[1]
```

The four row losses form the V20 anonymous `0.5` mean plus `0.5` temperature-
`0.05` entropic stream objective.  No detached valid mask, query identity, or
hidden result enters an update.

### Stage A — established-mechanism continual acquisition

Use train commitments `8..55`, four fresh variants per commitment, exactly 192
experiences.  For experience `e=0..191`, let `p=floor(e/48)` and `s=e mod 48`.
The commitment index is:

```text
8 + ((s + 13*p) mod 48)
```

Thus every commitment appears once in each pass and every experience is a
fresh topology/surface instance.  Topology and surface seeds are respectively:

```text
31_000_000_001 + 100_000*e
31_500_000_001 + 100_000*e
```

### Stage B — new-mechanism acquisition without Stage-A replay

Only even development commitment indices `(0,2,4,6,8,10,12,14)` may update.
Use eight fresh variants per commitment, exactly 64 experiences.  For
experience `e=0..63`, let `p=floor(e/8)` and `s=e mod 8`.  The development
commitment index is:

```text
2 * ((s + 3*p) mod 8)
```

Topology and surface seeds are respectively:

```text
32_000_000_001 + 100_000*e
32_500_000_001 + 100_000*e
```

No Stage-A commitment, stream, row, or stored history is read during Stage B.
Odd development indices `(1,3,5,7,9,11,13,15)` are update-heldout and may
appear only in no-gradient probes.  `development`, `train`, and schedule
identities are evaluator/control metadata and never learner inputs.

Before claim creation, the harness must mechanically prove only that the
declared update/diagnostic/probe schedule metadata and exact seed pairs are
unique where required and disjoint from each other and the frozen seed ranges
used by V4--V20 and V11-D1/D2.  It must not construct a V21 protocol stream,
package, task, graph, or row before the claim, and therefore makes no preclaim
claim about the runtime uniqueness of those objects.  After the claim, stream
construction and all runtime identity, public-boundary, row, and graph checks
run as each stream is materialized; any mismatch consumes the identity and
writes the bounded failure record.

Those postclaim checks keep a bounded, checkpointed control ledger containing
only order-sensitive digests and declared role/index keys. It must establish
`56` unique diagnostic identities, `256` unique update identities, and `80`
unique probe identities; the three probe passes must reproduce the same 80
literal identities, and all three role sets must otherwise be disjoint. The
ledger is never passed to a controller, loss, optimizer, gate operand, router,
or stream constructor and cannot serve as replay or task identity. It exists
only to make the construction claims above mechanically falsifiable across
resume.

## Sparse frozen probes

Probes are read-only, gradient-disabled, state-preserving, and occur at exactly
three boundaries: `pre`, `end_A`, and `end_B`.  There is no per-update retention
scan.  One immutable 80-stream cohort contains one stream for every ability in
these five groups and is reused byte-for-byte at all three boundaries so every
retention and recurrence comparison is paired:

- `original`: train commitments `0..7`;
- `v20_heldout`: train commitments `56..63`;
- `stage_a`: train commitments `8..55`;
- `dev_acquired`: development even indices `0..14`;
- `dev_unseen`: development odd indices `1..15`.

Let group `g` be `0..4` in the order above and member position `i` be zero-based
inside its group.  Probe topology and surface seeds are independent of boundary
and are:

```text
33_000_000_001 + 10_000_000*g + 10_000*i
34_000_000_001 + 10_000_000*g + 10_000*i
```

Probe bytes are immutable and shared across arms and boundaries.  Probe loss,
supported rows, qualifying streams, signed margins, and the exact paired
per-ability comparisons defined below are recorded.  Final, sealed, hidden
judge, action rollout, and evaluator result paths remain unopened in V21-A.
Composition here means transfer on the existing public two-motif
composed-query relation structure; it is not end-to-end pipeline execution.

## Descriptive initial gradient geometry

Before any update, use a disposable no-write clone at `W0` and one additional
fresh public stream for each of the 56 update-eligible abilities (48 Stage A
plus eight Stage B).  Diagnostic position `i=0..47` maps to train commitment
`8+i`; `i=48..55` maps in order to development indices
`(0,2,4,6,8,10,12,14)`.  Diagnostic topology/surface seeds are:

```text
30_000_000_001 + 100_000*i
30_500_000_001 + 100_000*i
```

For each ability, record the 64-dimensional stream-loss gradient with respect
to `W0`.  Report the `56 x 64` matrix rank using the standard
`max(shape)*eps*largest_singular_value` tolerance, its singular values,
pairwise cosine distribution, negative-cosine fraction, and Stage-A/Stage-B
within/between conflict summaries.  The diagnostic clone is then discarded
with the candidate state unchanged.  These values are descriptive only: they
cannot select a mode, seed, threshold, schedule, classification, or successor.

## Frozen observables and gates

For the 64 Stage-B update experiences, `online_pre_loss[a,e]` is the stream
loss immediately before arm `a` updates.  Define:

```text
AUC[a] = (0.5*l_0 + sum(l_1..l_62) + 0.5*l_63) / 63
```

`ProbeLoss[a,G,b]` is the mean stream loss for probe group `G` at boundary
`b`; `terminal_loss[a,G] = ProbeLoss[a,G,end_B]`.
`Rows[a,G,b]` and `Streams[a,G,b]` are V20 supported-row and qualifying-stream
counts at boundary `b`; a stream qualifies at three or more supported rows.
Every comparison uses the same probe bytes.

Let `L[a,G,i,b]` be arm `a`'s loss on the immutable probe for member `i` of
group `G` at boundary `b`.  All per-ability counts are literal paired loss
comparisons on that same stream:

```text
paired_better[a,c,G,b] = count_i(L[a,G,i,b] < L[c,G,i,b])
paired_retained[a,G,b2,b1,r] = count_i(L[a,G,i,b2] <= r*L[a,G,i,b1])
paired_ratio[a,c,G,b,r] = count_i(L[a,G,i,b] <= r*L[c,G,i,b])
```

These counts, rather than an unpaired mean or a vague recurrence label, define
every ability-recurrence requirement below.  For each updated arm `a`, its
matched no-update AUC control is the same slow representation at `W0` on the
same 64 Stage-B streams.  Define normalized fast gain:

```text
G[a] = 1 - AUC[a] / AUC[matched_no_update[a]]
```

Zero, missing, or non-finite denominators fail mechanical validity.

Define the following booleans exactly.

### `substantive_fast_acquisition` (`A`)

On `dev_acquired`, all must hold:

1. candidate Stage-B AUC is at most `0.95` times no-update AUC;
2. candidate terminal loss is at most `0.95` times no-update terminal loss;
3. candidate end-B coverage exceeds no-update by at least four supported rows
   **or** at least two qualifying streams; and
4. `paired_better[candidate, second_order_no_update, dev_acquired, end_B]`
   is at least `6/8`.

No absolute no-update competence can substitute for this causal fast-state
gain.

### `stage_a_acquired` (`S`)

On the Stage-A cohort at `end_A`, all must hold:

1. candidate aggregate loss is at most `0.95` times candidate no-update loss;
2. candidate coverage exceeds candidate no-update by at least four supported
   rows **or** at least two qualifying streams; and
3. `paired_better[candidate, second_order_no_update, stage_a, end_A]` is at
   least `36/48`.

### `oml_fast_attribution` (`T`)

Let the updated controls be `first_order_meta_persistent` and
`source_v19_persistent`, each with its correspondingly matched stateless
no-update AUC control.  Both of the following must hold for each control `c`:

```text
G[candidate] >= G[c] + 0.02
AUC[candidate] <= AUC[c]
```

This attributes useful normalized online plasticity to the V20 second-order
learned representation rather than merely to any linear online head or a
control's different static representation.

### `stage_a_retained` (`R`)

For Stage-A probes, define matched improvements over no-update:

```text
I_A = ProbeLoss[no_update, stage_a, end_A]
    - ProbeLoss[candidate, stage_a, end_A]
I_B = ProbeLoss[no_update, stage_a, end_B]
    - ProbeLoss[candidate, stage_a, end_B]
retained_fraction = I_B / I_A
```

`S` must pass, `I_A` must be positive, `retained_fraction >= 0.80`, and both
candidate supported-row and qualifying-stream coverage at end B must be at
least `0.95` of their end-A values.  In addition,
`paired_retained[candidate,stage_a,end_B,end_A,1.05]` must be at least `43/48`.
A zero denominator or missing/non-finite value fails.

### `unseen_development_transfer` (`U`)

On `dev_unseen`, candidate end-B terminal loss must be at most `0.95` times
no-update terminal loss and candidate coverage must exceed no-update by at
least two supported rows **or** at least one qualifying stream.  In addition,
`paired_better[candidate,second_order_no_update,dev_unseen,end_B]` must be at
least `6/8`.  Odd development mechanisms never update and cannot be used for
threshold choice.

### `persistent_boundary_nonregression` (`B`)

Against `second_order_boundary_reset` on acquired development mechanisms, the
candidate must have no higher Stage-B AUC, no higher end-B terminal loss, no
fewer supported rows, and no fewer qualifying streams.  On the immutable
Stage-A cohort at `end_B`, candidate aggregate loss must additionally be at
most `0.98` times boundary-reset aggregate loss, candidate supported rows and
qualifying streams must each be at least the corresponding boundary-reset
count, and
`paired_better[candidate,boundary_reset,stage_a,end_B] >= 36/48`.  This
directly requires a carried Stage-A-state advantage; Stage-B equality or
reacquisition after reset cannot establish persistence.

### `inherited_nonregression` (`N`)

For each of `original` and `v20_heldout` separately at `end_B`, all must hold:

1. candidate aggregate loss is at most `1.05` times second-order no-update;
2. candidate supported rows and qualifying streams are each at least `0.95`
   times second-order no-update; and
3. `paired_ratio[candidate,second_order_no_update,G,end_B,1.05]`, meaning the
   literal count of members satisfying
   `L[candidate,G,i,end_B] <= 1.05*L[second_order_no_update,G,i,end_B]`, is at
   least `7/8`.

### Mechanical causal validity

All are hard validity requirements:

- candidate and boundary-reset state/digest/metrics are exact through end A;
- swapping the terminal candidate fast state into a clean controller with the
  identical second-order RLN reproduces its state digest and all end-B probe
  logits and metrics exactly;
- resetting weight to `W0`, moments and step to zero reproduces all no-update
  end-B logits and metrics exactly;
- all probe and diagnostic operations leave state and slow digests exact;
- all slow tensors/buffers remain byte-identical and receive no gradient;
- candidate, first-order, and source arms each record
  `lifetime_updates=256` and terminal Adam `step=256`; the boundary arm records
  `lifetime_updates=256`, `reset_count=1`, and terminal Adam `step=64`; every
  no-update control records `lifetime_updates=0` and Adam `step=0`; and
- all state values, gradients, optimizer states, losses, metrics, and digests
  are finite and complete.

The A-to-B transition order is immutable: complete Stage-A update `191`; run
the `end_A` cohort and all through-A exactness checks; apply the boundary arm's
sole reset; atomically write the progress checkpoint with cursor `192` and
`boundary_reset_applied=true`; only then construct or consume Stage-B
experience `0`.  Resume at cursor `192` requires that flag and cannot repeat or
omit the reset.

## Exclusive first-result classification

The first completed result is classified without tuning by this priority tree:

1. any integrity, visibility, schedule, causal-validity, numerical, resource,
   or completeness failure -> `INVALID_NO_CLAIM`;
2. `A and S and T and R and U and B and N` ->
   `PERSISTENT_OML_TRANSFER_AND_RETENTION_SUPPORTED`;
3. `not S` -> `STAGE_A_NOT_ACQUIRED`;
4. `S and not R` -> `FAST_ACQUISITION_WITH_FORGETTING`;
5. `S and R and not N` -> `INHERITED_CAPABILITY_REGRESSION`;
6. `S and R and N and A and not T` ->
   `FAST_ACQUISITION_ATTRIBUTION_NOT_ESTABLISHED`;
7. `S and R and N and A and T and (not U or not B)` ->
   `FAST_ACQUISITION_WITHOUT_PERSISTENT_TRANSFER`;
8. `S and R and N and not A` and no-update end-B `dev_acquired` coverage is at least `24/32`
   supported rows and `6/8` qualifying streams ->
   `STATIC_REPRESENTATION_DOMINATES`;
9. otherwise -> `PERSISTENT_OML_NOT_SUPPORTED`.

The tree is exhaustive and exclusive.  A static V20 representation cannot
earn an acquisition or lifelong claim.  No result authorizes ANML, replay,
final/sealed access, promoted-state mutation, or deployment.

## Proportionate Human-Flourishing assessment

Impact class `LOW`, disposition `ALLOW` by the project owner for this exact
isolated WSL2 public-synthetic experiment, mapped to
`ANG-GATE-HUMAN-FLOURISHING-001@1-NOT_RUN`. It trains no language model, reads
no personal/recovered data, opens no network or service, takes no external
action, and changes no promoted state. The normal Human-Flourishing gate
remains `NOT_RUN`; this mapping authorizes only the bounded experimental work
and does not imply normal delivery or promotion.

## Resource, resume, and first-result envelope

- device: one CUDA GPU;
- numerical mode: FP32, TF32 off, autocast off, deterministic algorithms;
- allocated-memory ceiling: `2 GiB`;
- cumulative identity wall-time ceiling: `45 minutes`, measured from immutable
  post-preflight claim creation and never reset by resume;
- no slow optimizer, early stopping, seed retry, result-based extension,
  threshold adjustment, or resource-conditioned scientific change;
- atomic progress checkpoint every 32 completed experiences; the cursor-192
  checkpoint is written only after the ordered end-A probe/exactness/reset
  transition above;
- checkpoint includes all four fast states/moments/steps, stage and experience
  cursor, RNG state, frozen hashes/digests, boundary-reset status, and completed
  sparse probe records;
- resume restores the exact chronology and remaining claim-age budget without
  reading semantic metrics to choose behavior;
- checkpoints at most `16 MiB`; terminal JSON at most `4 MiB`;
- exactly one terminal report or failure record; the first result is retained
  without tuning.

Every exception after immutable claim creation, including resume-entry
integrity failure, publishes the bounded terminal classification
`INVALID_NO_CLAIM` with `HARNESS_ERROR_PRESERVED` retained only as its failure
subtype. No harness failure creates an additional scientific classification.

A synthetic CUDA preflight must prove detach/continuation equivalence,
functional-AdamW parity, checkpoint/resume identity, selected-state constant
capacity, and output absence before a claim.  It generates no protocol stream
or runtime package/task/graph/row and makes no scientific claim.  Preclaim
schedule validation is confined to arithmetic metadata and seed-range
uniqueness/disjointness; construction-dependent checks begin only after the
claim.

## Literal outputs and tests

Repository write scope:

- this leaf;
- `experiments/runners/phase6_oml_persistent_lifelong_v21.py`;
- `tests/unit/experiments/test_phase6_oml_persistent_lifelong_v21.py`;
- `.angler_v21_oml_persistent_lifelong_once.py`;
- append-only `AGENTS_SYNC.md` entries.

WSL artifact scope under `/opt/angler/results/`:

- `phase6-software-pipeline-reconstruction-v21a-oml-persistent-lifelong.claim.json`;
- `phase6-software-pipeline-reconstruction-v21a-oml-persistent-lifelong.progress.pt`;
- `phase6-software-pipeline-reconstruction-v21a-oml-persistent-lifelong.pt`;
- exactly one terminal `.json` or `.failure.json` with the same prefix.

Focused tests must cover source/report/checkpoint pins; exact four-arm
initialization; 192/64 schedule arithmetic and chronology; commitment/seed/
partition-metadata disjointness; one immutable 80-stream probe cohort reused
at all boundaries; constant 192-float-plus-step state; detach after every
update; no slow gradients or writes; functional AdamW and ordinary AdamW
parity including exact-zero VJP; candidate/boundary exactness through end A;
the exact update-191/probe/exactness/reset/checkpoint/Stage-B order and sole
boundary reset; row-builder value, ordering, mask, duplicate, and gradient
parity on frozen train streams; public-only development closure;
absence of mechanism/task/partition/query identity and hidden fields from the
learned dependency closure; probe timing and no-write/no-gradient behavior;
diagnostic no-write behavior and non-gating status; state reset/swap; no-update
equivalence; sparse metric arithmetic; every classification branch; finite
guards; progress/resume equivalence; exact terminal reload; ceilings; and
terminal/failure exclusivity.

Static tests must reject replay/history reads, core/evaluator edits, final or
sealed partition access, hidden judge/action rollout, solver/search APIs,
query-conditioned state selection, new trainable parameters, slow optimizers,
and imports outside the frozen source closure.

## Stop, rollback, and nonclaims

Stop before the claim on any source/result/checkpoint mismatch, occupied output,
schedule/seed-metadata overlap, unexpected parameter, failed test, or failed
synthetic preflight.  Row-builder, constructed-stream identity,
public-boundary, package/task/graph/row, and other runtime protocol checks
occur only after claim creation; any mismatch writes the bounded failure
record and consumes the identity.
Preserve claim/progress/checkpoint/terminal evidence on failure.  Repository
rollback removes only the four new V21-A leaf/implementation/test/harness
files.  Every append-only `AGENTS_SYNC.md` entry is preserved as evidence, and
`AGENTS_SYNC.md` is never restored, truncated, or rewritten.  No V20/V19/V16/
V12 byte may be edited.

Normal LEARNING, RETENTION, SKILL-COMPOSITION, Human-Flourishing, Slice 05,
Slice 06, M4, and M5 gates remain `NOT_RUN`/`NOT_PASSED`.  V21-A permits only
foreground local synthetic public learning in the isolated WSL GPU.  It grants
no model/LLM training or serving, network/package access, personal or recovered
data, tool growth, external effect, deployment, promotion, final/sealed access,
or autonomous authority.

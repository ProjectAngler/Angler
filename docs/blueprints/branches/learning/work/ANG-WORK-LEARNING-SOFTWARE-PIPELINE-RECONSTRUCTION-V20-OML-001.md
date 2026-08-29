# V20 OML relation-representation successor

Status: completed once; `OML_V19_HARMONIZED_ADVANCEMENT` independently confirmed  
Active node: `ANG-BP-LEARNING` / software-pipeline reconstruction  
Protocol: `phase6.public-oml-relation-representation.v20`

## Accountable outcome

Test whether an online-aware meta-learned relation representation lets one
small persistent prediction head acquire fresh public mechanisms with less
interference and forgetting, while retaining V19's supported learned paired
whole-graph context comparison byte-exact.

This is an evolving-reasoning component test, not LLM training and not an AGI
claim.  It meta-learns how the relation pathway should represent public
structure so later online updates are more useful and less destructive.  It
does not insert a deterministic task solver, mechanism router, answer lookup,
replay rescue, or fixed domain role.

## Borrowed idea and independent implementation

The algorithmic basis is Javed and White, *Meta-Learning Representations for
Continual Learning* (NeurIPS 2019), OML:
<https://papers.nips.cc/paper_files/paper/2019/hash/f4dd765c12f2ef67f98f3558c282a9cd-Abstract.html>.
Its slow representation / fast prediction-head objective is implemented anew
for Angler.

The author repository <https://github.com/kjaved0/mrcl> documents a July 2020
repair for incorrect meta-gradients and the result that a linear prediction
layer can match the neuromodulated variant on its benchmarks.  Current reviewed
head is `2855a6b7e820f171432981b58c49664fcdbf00ed`; repair lineage includes
`5d5626b0185bac528c3e6256fb85a1aae6874487`.  The repository exposes no
LICENSE/COPYING/NOTICE, so no donor source, legacy framework, data sampler,
configuration, checkpoint, or hyperparameter implementation may be copied.
Only the published algorithmic idea is reused.

Angler-owned `functional_adamw_step` and `AdamWSlot` semantics come from
`experiments/runners/phase6_cross_variation_plasticity_v16.py`, SHA-256
`EB1A29AC78670C6A0ECDED943E17AA62B1CFB91BF58DAB1ADC9001A3B75D63AB`.
V16 repaired the exact-zero square-root VJP while retaining forward/state
parity.  Its detached routing/meta wrappers must not be reused because they
would sever OML's representation gradient.

## Frozen source

- V19 runner SHA-256:
  `54A8E2E510424E485DE34A2975A82C927D22C87B5576EFE00537545158ECE5BE`
- inherited V12 runner SHA-256:
  `F1045756E77D60A7968265867035CEA55BFFE8BF6E1A73AB50C12A719EC8B529`
- terminal V19 checkpoint:
  `/opt/angler/results/phase6-software-pipeline-reconstruction-v19-paired-graph-context.pt`
- terminal checkpoint SHA-256:
  `10BB6BAC9BD83F7F4EE0ABF2846CE4133D2133790C2B55113C9044930D2EBC7F`
- accepted V19 recovery report SHA-256:
  `55592E9861EC16301603D0CD7BB2A104E596BAAA97BDC65D50DCC517951A0800`

The source checkpoint is loaded once into an exact
`V12ChampionPairedGraphContextController`.  V20 uses a successor-local source
binding and digest after relation meta-learning begins; it must not weaken or
call V19's source-lineage assertion as though inherited relation tensors were
still unchanged.  The source file/checkpoint/result remain immutable.

## Exact learned split

Slow representation learner (RLN), outer update only:

- `evidence_pair_encoder.*`
- `relation_pool_attention.*`
- `relation_pool_projection.*`
- `relation_incidence_readout.*`
- `relation_incidence_projection.*`
- `relation_comparator.0.*`

This is exactly 67 tensors / 61,898 parameters.  Existing
`relation_comparator[1]` is the SiLU representation activation.

Fast prediction learner (PLN), inner update only:

- `relation_comparator.2.weight`

This is one bias-free 64-to-1 linear tensor / 64 parameters, followed by the
existing tanh.  Its terminal V19/V12 value is the fixed initialization for
every meta-trajectory; the first pilot does not outer-update that initialization,
so a better learned starting head cannot masquerade as a better representation.

Frozen throughout:

- `evidence_context_encoder.*` and all `relation_context_*`
- all 21 `paired_graph_*` V19 tensors / 34,048 parameters
- action heads, memories, recurrent/backward reasoner, conflict mixer,
  competence state, and every other controller tensor or buffer

V19 production `public_paired_graph_credit_rows`, trace acquisition, and
`score_actions` remain the common path.  The controller keeps its exact V19
type; no row-builder copy or subclass is allowed.

## Meta-gradient and online update semantics

Each trajectory clones the fixed V19 PLN weight `W0` and fresh zero
`AdamWSlot` moments.  It then performs exactly eight chronological inner
updates `k=0..7`.  At every step, compute the stream loss defined below,
request gradients only with respect to the current functional PLN weight, and
call V16 `functional_adamw_step`.  `second_order_oml` uses
`create_graph=True` and preserves weight, gradient, and moment graphs.
`first_order_meta` uses the numerically identical gradient detached before the
same AdamW call; the fast-weight identity path is not detached.  After all
eight updates, aggregate the eight outer-stream losses under `W8` and
differentiate that objective into only the 61,898 RLN parameters.  The virtual
PLN and moments are then discarded.  At evaluation, the RLN is fixed while
PLN weight and moments persist across experiences with no replay.

For every V19 public row, let `P,N` be `positive_margin,negative_margin` and
`p_q,n_q` the paired slot-margin vectors.  Define:

`pair = _paired_relation_margin_loss(P, N)`

`slot = _relation_instance_losses(p_q, n_q)[1]`

`row_loss = pair + 0.25 * slot`

Thus `pair = relu(0.10-P) + relu(0.10+N) +
0.10*smooth_l1(P,-N)`, while `slot` is the existing normalized
temperature-`0.025` soft minimum of the per-slot losses.  For the four rows in
one stream:

`stream_loss = 0.5*mean(row_loss) +
0.5*0.05*(logsumexp(row_loss/0.05)-log(4))`

The outer objective applies the same formula over its eight stream losses,
replacing `4` by `8`.  An inner step uses one stream's `stream_loss`.  V19's
detached `valid_mask`, responsibilities, labels, and detached measurements are
forbidden as training input.
No query task, hidden solution, judge, commitment identity, package identity,
mechanism index, lane index, or evaluator result enters the learner.

## Frozen meta-training distribution

Only public `train`-partition commitments are used.

- original V19 retention set: commitment indices `0..7`, evaluation only
- meta-training set: indices `8..55`, arranged as six anonymous cohorts of
  eight (`8..15`, ..., `48..55`)
- completely held-out cross-mechanism set: indices `56..63`, evaluation only

There are exactly 240 outer updates.  For update `u`, define
`t_j=(4*u+j) mod 48` and commitment `c_j=8+t_j`, for `j=0..3`.  Inner step `k`
uses `pass=floor(k/4)`, `j=k mod 4`, and commitment `c_j`.  The exact order is:

`(pass0,c0),(pass0,c1),(pass0,c2),(pass0,c3),`
`(pass1,c0),(pass1,c1),(pass1,c2),(pass1,c3)`.

After step seven, outer streams are evaluated in this exact order:

`(same,c0),(same,c1),(same,c2),(same,c3),`
`(cross,d0),(cross,d1),(cross,d2),(cross,d3)`,

where `d_j=8+((t_j+24) mod 48)`.  Each meta-training mechanism is therefore a
target exactly 20 times and receives two fresh inner variations, for exactly
40 inner exposures per mechanism.  Each update has 16 unique streams, 64
public rows, eight inner steps, and one outer step.

The topology/surface seed-base pairs are respectively
`10_001_000_001/10_101_000_001` for `inner`,
`10_201_000_001/10_301_000_001` for `outer_same`, and
`10_401_000_001/10_501_000_001` for `outer_cross`.  For each role and update,
the seed formula is `role_base + 100000*u + 1000*position`, with `position`
equal to the exact zero-based position in that role's order.  The complete
incidence table and disjointness from V4-V19 and V11-D1/D2 identities must be
asserted before training.

Within an update, paired arms may share immutable generated public stream
objects solely to guarantee identical data, and both arms consume them in the
same frozen order.  No generated example crosses an outer update or becomes
persistent replay.  The report distinguishes unique stream/row encodings from
lane-specific loss uses.

## Paired causal arms

Two learned arms start byte-identical and receive the same public data,
optimizer, exposure, and update budget:

- `second_order_oml`: every inner gradient is requested with
  `create_graph=True`; V16 functional AdamW carries differentiable weight and
  moment state through all eight inner steps.
- `first_order_meta`: calculate the same numeric inner gradient, immediately
  detach that gradient, and then call the same functional AdamW.  Do not detach
  the fast parameter itself: the identity path from the fixed PLN
  initialization remains, while Hessian/meta-credit paths through inner
  gradients and moments are removed.

Their difference isolates second-order online-aware meta-credit from direct
outer representation training under an adapted head.  Outer examples,
ordering, learning rates, moments, update count, initialization, and outer
optimizer are identical.  The first-order arm may use less wall compute; no
sham graph is built merely to equalize GPU time.

Terminal evaluation also includes:

- untouched source V19 representation with identical online PLN AdamW;
- each learned arm's terminal representation with PLN updates disabled.

## Frozen optimization and resources

- outer updates per arm: 240
- unique streams per update shared by paired arms: 16
- inner steps per arm/update: 8
- inner PLN learning rate: `1e-3`
- outer RLN learning rate: `3e-4`
- Adam betas: `(0.9, 0.999)`; epsilon `1e-8`; weight decay `0`
- outer gradient clip: `5.0`
- numerical mode: FP32, CUDA, TF32 off, autocast off, exact second order
- device: one CUDA GPU; current target RTX 5080 16 GiB
- hard allocated-memory ceiling: 12 GiB
- hard semantic wall-time ceiling: 150 minutes
- no early stopping, seed retry, replay, threshold adjustment, or result-based
  extension

The implementation should batch relation feature computation where equivalence
is exact; the eight-step functional PLN unroll remains explicit.  A predeclared
fallback may split the eight outer streams into two groups of four with
gradient accumulation only if a synthetic/mechanical CUDA preflight exceeds
memory; the test must prove the same outer loss and gradient within `1e-6`
before any semantic values are read.  No other resource-conditioned algorithm
change is allowed.

Atomic progress checkpoints every 40 outer updates include both arm states,
outer optimizer states, RNG state, update index, and frozen digests.  Exact
crash continuation resumes the same identity and cannot inspect or select
semantic metrics.  A mathematical, integrity, or non-finite failure consumes
the identity and is preserved.

## Frozen evaluation

There are three four-panel families:

- `original` is historical retention only; every panel orders commitments
  `0..7`.
- `meta_seen` measures fresh instances of OML meta-training mechanisms.  Panel
  `p` orders `8 + ((6*i+p) mod 48)` for positions `i=0..7`; the four panels are
  disjoint and span all six training cohorts.
- `heldout` is cross-mechanism transfer; every panel orders never-meta-trained
  commitments `56..63`.

For family index `f` (`original=0`, `meta_seen=1`, `heldout=2`), role index `r`
(`update=0`, `probe=1`, `terminal_credit=2`), and kind `q`
(`topology=0`, `surface=1`), the exact seed base is:

`B(f,r,q)=20_000_000_001 + 1_000_000_000*f + 100_000_000*r + 50_000_000*q`.

For panel `p=0..3` and step/position `s,i=0..7`, update-stream seeds are
`B(f,0,q)+1_000_000*p+1_000*s`; probe-stream seeds are
`B(f,1,q)+1_000_000*p+100_000*s+1_000*i`; terminal-credit seeds are
`B(f,2,q)+1_000_000*p+1_000*i`.  These literal formulas are disjoint from
training and all prior identities and must be mechanically asserted.

For each panel and arm, start a fresh PLN/moments and take eight sequential
one-stream updates in the panel's exact commitment order with no replay.
Probe streams are generated once and are immutable/shared across arms: each
`probe[p,s,i]` is scored immediately before and after update `s`, and the same
target probe `probe[p,s,s]` is scored again after all eight updates.  Separate
terminal-credit streams supply relation coverage and context metrics after all
updates.  Report:

- online-loss area under the curve;
- immediate same-mechanism acquisition;
- forward transfer to not-yet-presented mechanisms;
- terminal retention of earlier mechanisms;
- relation-supported rows, qualifying streams, signed margins, and per-panel
  recurrence;
- a persistent eight-mechanism sequence comparing each mechanism's immediate
  post-acquisition loss with its final loss after later mechanisms.

Report the `meta_seen` versus `heldout` generalization gap explicitly, but do
not use it to retune a threshold or reinterpret a failed heldout gate.

The full frozen V19 context scorer runs on terminal relation states.  Its
`zero_residual` lesion is evaluation-only.  Every frozen context tensor and
the inherited V19 component's causal effect must remain exact/supported.

## Pre-registered classification

For family `h` (`original`, `meta_seen`, or `heldout`), panel `p`, online step
`s`, mechanism `i`, and arm `a`, record these exact loss observables on frozen
fresh probe bytes:

- `online_pre_loss[a,h,p,s]`: incoming update-stream loss before its update;
- `probe_pre_loss[a,h,p,s,i]` and `probe_post_loss[a,h,p,s,i]`: the same
  gradient-forbidden probe immediately before and after step `s`;
- `probe_terminal_loss[a,h,p,s]`: that step's target probe after all eight
  updates;
- terminal `supported_rows` and `qualifying_streams` under V19
  `_credit_rows_metrics` semantics.

Define, with `o_s=online_pre_loss[a,h,p,s]`:

`AUC[a,h,p]=(0.5*o_0 + sum(o_1..o_6) + 0.5*o_7)/7`

`AUC[a,h]=mean_p(AUC[a,h,p])`

`Rows[a,h]=sum_p(terminal_supported_rows[a,h,p])`

`Streams[a,h]=sum_{p,i}(1[terminal_supported_rows_for_stream_i >= 3])`

For the target shown at step `s`, define aggregate
`immediate_gain=mean(probe_pre_loss-probe_post_loss)` and
`terminal_gain=mean(probe_pre_loss-probe_terminal_loss)`.  Then
`retained_fraction=terminal_gain/immediate_gain`, valid only when
`immediate_gain>0`; otherwise the retention gate fails.  `forward_gain` is the
mean `probe_pre_loss-probe_post_loss` over mechanisms whose first presentation
is after step `s`; the empty set after the final step is omitted.  A panel
improves over its paired first-order panel exactly when its lexicographic
`(terminal_supported_rows, terminal_qualifying_streams)` pair is greater.

Let `S` (`SECOND_ORDER_OML_CREDIT_SUPPORTED`) be true exactly when, on heldout
commitments `56..63`:

1. `AUC[second_order_oml,heldout] < AUC[first_order_meta,heldout]` and is at
   most `0.95 * AUC[first_order_meta,heldout]`;
2. `Rows[second_order_oml,heldout]-Rows[first_order_meta,heldout] >= 4` or
   `Streams[second_order_oml,heldout]-Streams[first_order_meta,heldout] >= 2`;
3. at least three of four panels improve lexicographically, and every panel's
   second-order supported rows are at least `first_order_rows-1`;
4. second-order AUC is strictly lower than both untouched-source-V19 with the
   identical online PLN update and its own learned-representation no-update
   control.

Let `C` (`OML_CROSS_MECHANISM_ADVANCEMENT`) be true exactly when `S` is true,
`Rows[second_order_oml,heldout] >= 96`,
`Streams[second_order_oml,heldout] >= 24`, heldout
`retained_fraction >= 0.80`, and heldout `forward_gain >= 0`.

Let `H` (`OML_V19_HARMONIZED_ADVANCEMENT`) be true exactly when `C` is true and
the original panels also have at least `96/128` supported rows and `24/32`
qualifying streams, no original panel loses more than one supported row against
its paired source-V19-online control, and the original-panel terminal full
V19-context arm versus its `zero_residual` lesion retains all frozen V19 rules:
aggregate top-one gain at least `+12`, real-normalized valid-mass gain at least
`+0.05`, informative-margin gain at least `+0.05`, positive recurrence in at
least three of four panels, no panel top-one delta below `-1`, no panel
valid-mass delta below `-0.01`, and exact relation signatures under the lesion.

Define `fast_adaptation_supported` only when both learned arms have strictly
lower heldout AUC than both their corresponding no-update controls and the
source-V19 representation with identical online PLN updates.

For the heldout `second_order_oml` arm define
`post_update_loss_delta[p,s,i]=probe_post_loss[p,s,i]-probe_pre_loss[p,s,i]`,
`target_gain_mean=mean(-delta[p,s,target_s])`, and
`positive_off_target_harm_mean=mean(max(delta[p,s,i],0), i != target_s)`.
`off_target_harm_to_target_gain_ratio` is their ratio and is invalid when
`target_gain_mean<=0`.  `harmful_panel_step_fraction` is the fraction of panel
steps where target gain is positive and mean positive off-target harm is at
least `0.25` times that target gain.  `selective_plasticity_harm` is true only
when `target_gain_mean>0`, the ratio is at least `0.25`, and the harmful
panel-step fraction is at least `0.50`.

The sole V11-D2 input is
`/opt/angler/results/phase6-software-pipeline-reconstruction-v11-d2-representation-overlap.json`,
schema `angler.phase6-v11-d2-representation-overlap-report.v1`, SHA-256
`69D56232A4E70720AFD8428208A0F5ED4B4C2C75AED3D71DFF678F5BA10E6C9F`.
Its top-level `classification` is
`REPRESENTATION_OVERLAP_INTERFERENCE_NOT_SUPPORTED`; values are under
`evaluation.stages.relation_comparator_hidden.cells`.  No alternate artifact
or conversational summary may affect this gate.

`d2_same_module_overlap` is true only when that exact artifact's classification
is `REPRESENTATION_OVERLAP_INTERFERENCE_SUPPORTED` and, in both `t0_s0` and
`t0_s1`, comparator-hidden `easy_hard_exceeds_both_within_groups` is true and
`gradient_alignment.shared_comparator` has positive observed mean burden with
one-sided `p<=0.05`.  Missing or invalid D2 evidence makes this false.  Under
the immutable first result above, `d2_same_module_overlap=false`.

`anml_trigger = S and not H and d2_same_module_overlap and
selective_plasticity_harm`.

Classify the first completed result without tuning by this exclusive priority
tree:

1. `H` -> `OML_V19_HARMONIZED_ADVANCEMENT`;
2. `S and not H and anml_trigger` ->
   `OML_COMPONENT_SUPPORTED_NOT_INTEGRATED`;
3. `C and not H and not anml_trigger` ->
   `OML_CROSS_MECHANISM_ADVANCEMENT`;
4. `S and not C and not anml_trigger` ->
   `SECOND_ORDER_OML_CREDIT_SUPPORTED`;
5. `not S and fast_adaptation_supported` ->
   `FAST_ADAPTATION_SUPPORTED_OML_ATTRIBUTION_NOT_ESTABLISHED`;
6. otherwise -> `OML_NOT_SUPPORTED`.

Only `OML_COMPONENT_SUPPORTED_NOT_INTEGRATED` authorizes a narrow ANML
experiment.  That successor may gate the 64-dimensional activation after
`relation_comparator[1]`, keep the same linear PLN, and compare directly with
OML.  Any false/unavailable trigger operand denies ANML authorization from
V20; OML-specific credit absent means ANML is not justified by this result.

## Literal outputs and tests

Repository write scope:

- this leaf
- `experiments/runners/phase6_oml_relation_representation.py`
- `tests/unit/experiments/test_phase6_oml_relation_representation.py`
- `.angler_v20_oml_once.py`
- append-only `AGENTS_SYNC.md` entries

WSL artifacts:

- `phase6-software-pipeline-reconstruction-v20-oml.claim.json`
- `phase6-software-pipeline-reconstruction-v20-oml.progress.pt`
- `phase6-software-pipeline-reconstruction-v20-oml.pt`
- exactly one terminal `.json` or `.failure.json` under the same prefix

The terminal JSON ceiling is 4 MiB; checkpoints are each at most 16 MiB.

Tests must cover: the analytic missing-second-order-gradient regression;
two-or-more-step FP64 finite-difference agreement; V16 AdamW nonzero/zero/None
parity and finite zero VJP; exact RLN/PLN/frozen ownership; source and V19
context digests; no slow inner update; no fast outer write; full/first-order
exposure equality; full/first-order numeric forward equality before an owner step;
first-order zero dependency through every detached inner-gradient tensor;
schedule balance/disjointness; no identity/valid-mask/query leakage;
batched/unbatched and fallback equivalence; CPU/CUDA tiny-case parity; finite
guards before every owner step; progress resume equivalence; first-result
exclusivity; evaluation control separation; and exact frozen gate arithmetic.

Semantic launch requires all focused tests, one synthetic CUDA preflight, exact
source/result hashes, absent terminal identity, and an independent read-only
audit.  The first completed semantic result is preserved without tuning.

Human-Flourishing and normal Learning gates remain unchanged and NOT_RUN;
Slice-00/M0 remain NOT_PASSED.  This leaf permits only local synthetic public
train-partition learning on the isolated WSL GPU and writes its bounded local
artifacts.  It grants no model serving, network/package access, personal or
recovered-personal data, deployment, external effect, promoted-state change,
or autonomous authority.

## Accepted first result

The one frozen semantic identity completed on 2026-08-29 without tuning or
resume.  Independent recomputation from primitive panel metrics confirms the
exclusive classification `OML_V19_HARMONIZED_ADVANCEMENT`.

- terminal report: `/opt/angler/results/phase6-software-pipeline-reconstruction-v20-oml.json`
  (`5CCCBF0CE8211E0CC99AEB856145BF4CD3D9EA30A1ECB3FAE8E9435B4689C498`,
  2,529,486 bytes)
- terminal checkpoint: `/opt/angler/results/phase6-software-pipeline-reconstruction-v20-oml.pt`
  (`D49E4CAAB64A264A11C675B295A8C453AC4475F078311EB7283A4F9A8817EF48`,
  3,787,497 bytes)
- run claim: SHA-256
  `15CD27B64BE06F45044025ABADCD6E71DA982D399E9CDB054DB2BEADA282CB17`
- terminal system digest:
  `sha256:4c8e1f5df037956e01ab59353df45cf114c76385cca5d77c0c632e633d7614c3`
- elapsed identity wall time: `4,960.10` seconds; peak allocated CUDA memory:
  `403,186,176` bytes

On completely held-out commitments `56..63`, second-order OML achieved AUC
`0.0611586` versus `0.1006185` for the paired first-order arm, a `39.22%`
reduction.  It reached `120/128` supported rows and `30/32` qualifying streams
versus `88/128` and `12/32`, improved all four panels, retained fraction
`4.88999`, and forward gain `+0.0001881`.  On original commitments it retained
`123/128` rows and `30/32` streams.  The inherited V19 context component also
remained causal: full versus zero-residual gained `+63` unique-valid top-one,
`+0.31706` real-normalized valid mass, and `+1.68263` informative margin with
positive recurrence in `4/4` panels and exact relation signatures.

The evidence supports a meta-learned relation representation that generalizes
to mechanisms never used by OML training and harmonizes with the retained V19
whole-graph context comparison.  The largest gain is in the slow learned
representation: the second-order no-update control already reached `118/128`
rows and `30/32` streams, while online PLN adaptation added the final two rows
and reduced AUC from `0.0616766` to `0.0611586`.  This is therefore strong OML
representation evidence and modest additional fast-adaptation evidence, not a
claim that persistent lifelong acquisition is solved.

The immutable D2 operand remains false, selective-plasticity harm is false
(`0.0812` harm/target-gain ratio; harmful step fraction `0.1875`), and
`anml_trigger=false`.  V20 does not authorize ANML.  The result is bounded to
the frozen public synthetic mechanism families; it is not an AGI, unrestricted
domain-transfer, normal Learning-gate, Slice-00, or M0 claim.

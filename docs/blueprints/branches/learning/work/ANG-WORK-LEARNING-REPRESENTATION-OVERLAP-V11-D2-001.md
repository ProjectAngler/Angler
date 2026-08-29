# V11-D2 public representation-overlap diagnostic

Status: ready for one no-update diagnostic execution  
Active node: `ANG-BP-LEARNING` / software-pipeline reconstruction  
Protocol: `phase6.public-representation-overlap.v11-d2`

## Purpose

Localize the already-observed cross-stream gradient conflict before the first
OML successor.  This diagnostic asks whether the recurring hard public streams
occupy overlapping learned representation subspaces while requesting opposed
updates.  It cannot train, rescue, promote, or rerun V11 and it cannot decide
that OML or ANML succeeds.

## Frozen sources and inputs

- preserved Git ref: `refs/angler-preserved/v11-d1-source-tree`
- tree: `6d54d3fe66d7e27b30e550b65c94d3f82c22bb1f`
- runner blob: `cf7fe45fb31531435a2b51e4485ec3137d40ed4e`
- runner SHA-256: `305A083B4A108E5CA3784BD8834DDA74CA813D6E06C25CE08F75C61AE39D0B01`
- runner bytes: `308885`
- V11 checkpoint SHA-256:
  `2CF650BA5C9B62F1205CBA7F096CF9A078B752E699B98584FBC436FE1F5F0694`
- V11 report SHA-256:
  `EFDAC9461F34BE20226F54B718FB7A6F29375F74D9EFAD293D847867E071AE43`
- V11-D1 report SHA-256:
  `3191E1D9962A11BCCE9E8664E315D13CADB2F893C02994B7AA8B25233DF142BA`
- terminal model digest:
  `sha256:3833c9a01e986d5d7206802969b909747e34a5136266b54c72546f28436d9581`

The exact preserved tree may be exported only to a leaf-specific temporary
directory for execution.  It must not replace current repository files.  The
temporary copy is not authority and must be removed after the result or
preserved failure has been written.

## Identity and public input

Reuse the four fixed no-update D1 cells only:

- `t0_s0`: topology base `3001000001`, surface base `3101000001`
- `t0_s1`: topology base `3001000001`, surface base `3111000001`
- `t1_s0`: topology base `3011000001`, surface base `3101000001`
- `t1_s1`: topology base `3011000001`, surface base `3111000001`

Each cell contains the same eight anonymous train-partition commitments with
the D1 `1000*c` seed stride.  Total exposure is exactly 32 streams and 128
public support tasks.  No development/final/control stream, query task, answer,
private evaluator value, replay item, or identity feature may be accessed.

The exact V11 `after_relation` support pattern is
`[4, 2, 4, 4, 2, 2, 3, 0]`; the two D1 `t0` cells each record
`[4, 2, 4, 4, 2, 2, 3, 2]`.  Their shared recurrence defines easy streams
`E={0,2,3}`, hard streams `H={4,5,7}`, and intermediate streams `M={1,6}`.
These labels are used only in post-forward summaries; they must never enter a
model forward pass.

## Captured learned stages

For each stream, capture detached public activations from the unmodified V11
controller while executing `public_relation_credit_rows` under `torch.no_grad`:

1. `fused_relation_code` — the two public same-contract candidate relation
   codes from each support task, both retained in presentation order (width
   32; eight rows per stream).
2. `relation_comparator_hidden` — the `relation_comparator[1]` outputs for the
   same two candidates against every occupied real evidence slot in both
   public counterfactual arms (width 64; 48 rows per stream).

The unused root candidate is excluded because it never enters the measured row
loss and would dilute attribution to that loss.  Selecting the pair uses the
same public observed/alternative indices already consumed by the D1 gradient
objective; both twins are included without an answer-order reduction and no
selected activation leaves the process.

The temporary instance wrappers/hooks must be removed after every stream and
also on an injected exception.  Raw activations, component names, graph IDs,
commitments, answers, and public task contents must not be written.

## Frozen calculations

For an activation matrix `X`, calculate in FP64 after column centering:

`C = X.T @ X / max(rows - 1, 1)`

For each pair of streams, representation overlap is the Frobenius cosine of
their covariance matrices.  Also report descriptive mean row-wise Hoyer
sparsity, fraction `abs(x) <= 1e-6`, and covariance effective rank.  These
descriptive values do not pass or fail the diagnostic.

For each cell, stage, and each D1 gradient group (`shared_encoder`,
`global_pool`, `incidence_branch`, `shared_comparator`), define pair burden:

`burden(i,j) = overlap(i,j) * max(0, -gradient_cosine(i,j))`

The group-localization summary reports raw mean overlap for the nine `E-H`
pairs, the three `E-E` pairs, and the three `H-H` pairs.  The primary alignment
statistic `T` is mean burden over all 28 off-diagonal stream pairs.  Enumerate
all `8! = 40320` permutations of the overlap matrix's stream indices relative
to the fixed D1 gradient-cosine matrix.  The exact one-sided Mantel-style
p-value is the fraction whose `T` is at least the observed `T`.  Permuting one
matrix relative to the other is mandatory; jointly permuting both matrices or
permuting unused labels would leave the statistic unchanged and is invalid.
The observed threshold is the identity permutation evaluated through the same
fixed-order scalar reduction used for every null permutation, so a one-ULP
reduction-path difference cannot alter an exact tie or exclude the identity.

Classification is
`REPRESENTATION_OVERLAP_INTERFERENCE_SUPPORTED` only when, at the comparator
hidden stage, raw mean `E-H` overlap is greater than both mean `E-E` and mean
`H-H` overlap, and exact `p <= 0.05` for at least three of four gradient groups,
in both `t0_s0` and `t0_s1`.  Otherwise it is
`REPRESENTATION_OVERLAP_INTERFERENCE_NOT_SUPPORTED`.  `t1` cells are
descriptive topology-transfer measurements and cannot rescue the required
`t0` criterion.  The two `t0` cells are surface rerenders used as a determinism
control, not independent replication.  No threshold may change after values
are observed.

## Controls and invariants

- Reconstructed seed grid, commitments, stream count, row count, and D1
  gradient matrices must match the preserved D1 report.
- `t0_s0` versus `t0_s1` overlap matrices must differ by at most `1e-6` at
  each captured stage; this is a reported rerender control, not a hidden
  threshold adjustment.
- Reversing captured sample order must change every FP64 overlap by at most
  `1e-12`.
- Failure of either rerender or sample-order control makes the run
  `REPRESENTATION_OVERLAP_HARNESS_ERROR_PRESERVED` with no scientific claim;
  it may not be interpreted as either supported or not supported.
- The controller model digest, checkpoint hash, source-report hash, and D1
  report hash must be exact before and after.  No parameter or buffer may
  change and no parameter may retain a gradient.
- Device is CPU, torch threads are exactly one, optimizer creations/steps and
  parameter updates are zero, checkpoint writes are zero, and no GPU, model
  inference beyond this frozen small controller, package, network, or external
  effect is permitted.

## Literal outputs and bounds

Repository write scope:

- this leaf
- `experiments/evaluators/phase6_v11_representation_overlap.py`
- `tests/unit/experiments/test_phase6_v11_representation_overlap.py`
- `.angler_v11_representation_overlap_d2_once.py`
- append-only coordination entries in `AGENTS_SYNC.md`

Execution artifacts:

- `/opt/angler/results/phase6-software-pipeline-reconstruction-v11-d2-representation-overlap.claim.json`
- exactly one of:
  - `/opt/angler/results/phase6-software-pipeline-reconstruction-v11-d2-representation-overlap.json`
  - `/opt/angler/results/phase6-software-pipeline-reconstruction-v11-d2-representation-overlap.failure.json`

The terminal JSON ceiling is `262144` bytes.  The evaluator records the four
8x8 overlap matrices per stage, bounded descriptive summaries, burden
contrasts and p-values, source/integrity bindings, control results, and the
classification.  It writes no raw activation.

## Tests and execution gate

Unit tests must cover covariance/Frobenius overlap, zero-variance rejection,
sample-order invariance, Hoyer/effective-rank bounds, exact permutation count,
the relative-matrix permutation statistic (including rejection of a joint
permutation no-op), raw E/H group summaries, classification boundary, exact
8/48 captured row counts, hook cleanup after success and injected failure, no
raw activation serialization, and rejection of altered frozen identities.

Before the one execution: tests pass; source/checkpoint/report hashes match;
terminal output paths are absent; and an independent read-only reviewer finds
no answer leakage, update path, mutable-source substitution, or post-result
tuning channel.  Accept and preserve the first terminal classification.

## Interpretation and successor

A supported result localizes one reason OML is appropriate: the existing
representation exposes overlapping features to updates that point in opposed
directions.  It does not prove OML will solve lifelong learning.  A non-
supported result does not veto the owner's OML experiment; it requires the OML
leaf to treat its representation split as exploratory rather than as confirmed
localization.  ANML remains deferred until an OML result demonstrates a
specific residual need for learned activation/plasticity gating.

Human-Flourishing, normal Learning, Human-Flourishing, Slice-00, and M0 gates
remain unchanged and NOT_RUN/NOT_PASSED as applicable.  This leaf has no
deployment, network, personal-data, recovered-personal-data, model-serving, or
promotion authority.

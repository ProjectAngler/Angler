# ANG-WORK-LEARNING-BARLOW-SEMANTIC-INVARIANCE-V18-001

Status: ready for implementation and independent prelaunch audit. V17 and all
earlier identities are consumed and immutable. The first preflight-passing V18
result will be accepted without tuning.

## Outcome and bounded claim

Test whether a standard negative-free Barlow Twins objective over public frozen
V13 relation rows makes V17's shared semantic metric sufficiently invariant and
non-collapsed for causal structural correspondence to emerge from the unchanged
task-feedback objective.

A pass establishes only learned public-input augmentation invariance plus the
unchanged V17 causal correspondence/durability operands. It does not establish
cross-mechanism transfer, general reasoning, autonomous improvement, or that
Barlow Twins alone identifies correspondence.

The rejected alternative—an outcome-weighted anchor-assignment head—is not part
of V18. It would directly supervise correspondence with outcomes and encode the
benchmark-specific rule that a probe copies one of two anchor outcomes. No
deterministic task solution, outcome-weighted route, informative-row filter,
relation label, family/task ID, answer, regime, position, or metadata may enter
the V18 auxiliary objective or semantic metric.

## One learned-mechanism change

Reuse the exact V17 `SharedSemanticMetricCreditMemoryCore`, state, corpus,
task-feedback losses, optimizer, update schedule, controls, evaluator, and
gates. Add one auxiliary training objective only in the full arm:

1. At each update, collect exactly one copy of each of the 48 public detached
   64-D frozen V13 relation rows from the eight current counterfactual-twin
   pairs. Visit pairs in frozen schedule order, then `first` before `second`,
   then ascending event/row index. Public-row identity is the SHA-256 of
   `PublicContingencyEvent.to_prediction_payload()` serialized by
   `json.dumps(payload, sort_keys=True, separators=(",", ":"),
   ensure_ascii=False).encode("utf-8")`. Retain first occurrence order. A
   repeated digest must have byte-identical canonical payload and an exactly
   equal detached CPU contiguous float32 relation row, otherwise execution
   stops. This removes only the exact public counterfactual-twin copy; two
   distinct public events remain distinct even if their learned relation
   tensors happen to be equal. The operation must yield exactly 48 unique
   public event digests or execution stops. No row is selected using an outcome
   or evaluator sidecar. Record the ordered event digests and SHA-256 of their
   concatenated 32-byte binary digests for every update.
2. Before preflight or training, create one dedicated CPU
   `torch.Generator(device="cpu")`, seed it exactly with `2026083118`, and make
   exactly one float32 draw of shape `[96, 2, 48, 64]` using `torch.rand` on
   CPU. Threshold `<0.80`, convert to contiguous uint8, and make no further
   draws from that generator. Draw order is update, view, row, feature. The
   schedule must contain `472284` retained elements and its C-order uint8 bytes
   must hash to
   `8478D8C3883CA298FB3349B57DFA8FD2B8A8D01AF4F9FED94DB7C6BC46C2156D`
   under frozen workstation Torch `2.13.0+cu130`, or execution stops. Preflight
   reads copies from this precomputed schedule and cannot consume or change it.
   Full-arm views use the selected mask cast onto the relation tensor's device,
   with inverted-mask rescaling by `1/0.80`. The objective-off arm does not use
   the masks. Global/task RNG state must be byte-exact before and after schedule
   generation and all mask checks.
3. Pass both views through the existing shared semantic metric. For each view
   and each of its 32 code dimensions, compute the float32 batch mean and
   population variance `mean((x-mean)^2)` over the 48 rows; standardize with
   `(x-mean) / sqrt(variance + 1e-4)`. Exact-zero pre-epsilon variance is
   permitted, yields standardized zeros for that dimension, and is counted in
   diagnostics; the diagonal penalty supplies its learning signal. Any
   non-finite input, mean, variance, standardized code, correlation, or loss
   stops execution. Compute `C = z1.T @ z2 / 48`, and use standard Barlow Twins
   redundancy reduction:

   `L_barlow = (sum_i (1-C_ii)^2 + 0.0051 * sum_{i!=j} C_ij^2) / 32`.

4. The full arm uses the unchanged V17 task total plus `1.0 * L_barlow`. The
   paired objective-off arm uses only the unchanged task total. Arms begin from
   byte-identical initialization and receive the same task rows, task ordering,
   feedback, optimizer settings, and 96x8 update budget. No later capacity,
   schedule, mask, coefficient, or seed adjustment is permitted under V18.
   Paired-arm parity means byte-identical initial parameters and optimizer
   configuration plus identical encoded task rows, schedule, feedback,
   chronology, and task-loss implementation. At the same initial parameter
   state, task-only numeric losses and gradients must be exact. Numeric task
   losses/gradients are not required to remain equal after update zero because
   the auxiliary objective is expected to make the full parameters diverge.

There are no sample negatives, contrastive class labels, teacher embeddings,
nearest-neighbor targets, replay additions, or task-specific matching rules.
The objective may update only the existing shared semantic metric network; all
ordinary V17 task gradients and parameter scopes remain unchanged.

## Frozen inputs and identities

- experiment seed: `2026083118`;
- consumed V17 result SHA-256:
  `C74AA58CA0E78702366E024F6BFB928B94D5E627241986001133E8112DF72894`;
- consumed V17 checkpoint SHA-256:
  `5AA492D38555AFEBCDBB674EA31ECFA9474E0A8A25BF5AE579F5205FB989D182`;
- V17 core SHA-256:
  `A25D42E9A4B8697FC441FC4C55543BFABD661AD5BD7756C4BF87EA80EAB14790`;
- V17 runner SHA-256:
  `17AA47B7A37FE54A01481AEF5B875F0B85CF805937ED6BFE35EF784F895A1633`;
- V17 runner test SHA-256:
  `0A48D31F9418C8BF85B58915EB37757E040ADDDC68757CC8D71D3CF527D5C82A`;
- historical reasoning export must remain SHA-256
  `B06E4F87620974FD5C2F2AD75A433DBE55A9CBA799B0F442C5FE9A51785D5798`.

The V15 counterfactual-twin corpus, frozen Qwen/V13 substrate, partitions,
pair-once encoding, predict-before-feedback chronology, AdamW configuration,
true/deranged/detached-zero task losses, affine and removal controls, evaluator
sidecars, final seal, and every V17 scientific threshold remain byte-for-byte or
semantically exact as applicable.

## Owned outputs

This leaf owns only:

- `experiments/runners/barlow_semantic_invariance_v18.py`;
- `tests/unit/experiments/test_barlow_semantic_invariance_v18.py`;
- `/opt/angler/results/barlow-semantic-invariance-v18-full.pt`;
- `/opt/angler/results/barlow-semantic-invariance-v18-objective-off.pt`;
- `/opt/angler/results/barlow-semantic-invariance-v18-development.json`.

No new reasoning core or package export is owned. All consumed source and
artifacts are read-only. Result/checkpoint writes are atomic. A failed terminal
result is preserved; fresh partial artifacts may be removed individually.

## Required preflight and tests

Before training, tests must prove:

- V17 source/result/checkpoint identities match the frozen hashes;
- full and objective-off arms have byte-identical initial parameters and exact
  task-row/order parity;
- ordinary V17 per-row losses, gradients, chronology, state bounds, replay, and
  interventions remain exact before the auxiliary objective is enabled;
- the 48-row auxiliary batch is deduplicated only by canonical public-event
  payload identity; twin payloads and relation rows are exact, while distinct
  public events are never merged merely because encoded features collide;
- changing outcomes, labels, or all evaluator sidecars leaves auxiliary inputs
  and loss exact;
- simultaneous row permutation and matching mask permutation leave the loss
  invariant within `1e-6`;
- swapping the two views leaves the loss invariant within `1e-6`;
- the auxiliary gradient reaches every trainable semantic-metric parameter and
  no parameter outside that metric receives an auxiliary gradient;
- no sample-negative table, outcome-dependent filtering, or task-specific
  assignment head exists;
- checkpoint/result binding, source-chain hashing, device, version, memory,
  and atomic-write evidence remain complete.

## Frozen interpretation and gates

Report standard Barlow diagnostics without using them to tune V18: mean
cross-correlation diagonal, mean absolute off-diagonal, code standard deviation,
and effective rank for full and objective-off arms.

`BARLOW_SEMANTIC_OBJECTIVE_SUPPORTED` requires all of:

- every integrity and preflight operand passes;
- full minus objective-off corresponding-anchor top-1 is `>=0.10`;
- full minus objective-off mean corresponding-anchor read mass is `>=0.05`;
- full minus objective-off mean corresponding-anchor margin is `>=0.05`;
- full true-feedback balanced accuracy is not below objective-off;
- full true-feedback NLL is no more than `0.02` worse than objective-off;
- the full arm independently passes the entire frozen V17
  `SHARED_SEMANTIC_CORRESPONDENCE_SUPPORTED` component gate.

`FULL_DURABLE_BARLOW_SEMANTIC_CREDIT_SUPPORTED` additionally requires the full
arm to pass the entire frozen V17 durable-memory gate, including streaming
accuracy, breadth, retention, all causal removals, replay, and bounds. Even this
classification does not open final within V18: V18 owns no final output or
final phase. A supported development result may justify a separately authored,
audited successor final-evaluation leaf; until then the final partition remains
sealed.

An attribution improvement that does not pass the frozen V17 component gate is
reported as directional evidence only. If V18 fails, preserve it and diagnose
from its recorded arms; do not tune keep probability, coefficient, seed,
thresholds, or masks under the consumed identity.

## Human impact, resources, and stop conditions

Local synthetic learning only. No external action, procedure execution,
service, personal/recovered data, deployment, promotion, Qwen/V13 mutation, or
human-outcome judgment. Human stop/deletion control is absolute. Qwen remains
on `cuda:0`; V13 and V18 run on `cuda:1`. Maximum Angler allocation is 12 GiB
and maximum elapsed time after model load is 60 minutes.

Stop before training on any identity mismatch, source drift, missing consumed
artifact, preflight failure, output collision, device mismatch, non-finite
value, auxiliary gradient outside the semantic metric, sidecar/outcome leakage,
or inability to reproduce the paired-arm initialization and task stream.

## Consumed first result

The single preflight-passing run completed in `289.886147` seconds as
`DEVELOPMENT_NOT_SUPPORTED`. V18 is consumed without tuning and final remains
sealed:

- development result SHA-256
  `8289750FF87394493E0E23681BC52128CC55429F291A84488E700179F9FE87A9`;
- full checkpoint SHA-256
  `BBCD20BCE682936181BDB5FA89A72674BE92F08A54515A35024EE6776F39AD80`;
- objective-off checkpoint SHA-256
  `27212FB2B531CCA13B29D1F8CC69343486BC16998E6242875F23B768B931ED2C`;
- peak Angler allocation `160120832` bytes;
- all identity, preflight, mask, canonical-row, replay, chronology, bounds,
  source, and final-seal checks passed.

The auxiliary objective learned its narrow same-instance invariance target:
full versus objective-off mean cross-correlation diagonal rose from `0.5132`
to `0.6698`, and Barlow loss fell from `0.2713` to `0.1416`. That gain did not
create cross-event correspondence. Corresponding-anchor top-1 fell from
`0.2865` to `0.2448`, matching mass fell from `0.2167` to `0.2099`, and margin
changed only from `-0.0091` to `-0.0050`. True-feedback BA fell from `0.5807`
to `0.5547`, while NLL worsened from `0.7090` to `0.7331`, exceeding the frozen
`0.02` non-regression allowance. Fresh matched-state BA fell from `0.7266` to
`0.6589`; paired direction fell from `0.6563` to `0.6406`; opposite-twin
reversal fell from `0.4844` to `0.3490`; and worst retention drop remained
`0.50` in both arms.

V18 therefore falsifies same-row augmentation invariance as the missing
correspondence signal in this setting. It may improve augmentation statistics
while competing with task-relevant geometry. The successor must learn
cross-event structural equivalence from public inputs without outcomes,
relation/family/task labels, evaluator sidecars, or a deterministic assignment
rule. V18 masks, weight, seed, and thresholds will not be tuned or rerun.

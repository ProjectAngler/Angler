# ANG-WORK-LEARNING-SHARED-SEMANTIC-METRIC-CREDIT-V17-001

Status: consumed; first result `DEVELOPMENT_NOT_SUPPORTED`. V15 and V16 are consumed and
immutable. V17 isolates one addressing change; no result-dependent tuning of a
consumed identity is permitted.

## Outcome and falsifiable claim

Test whether a shared, relation-only semantic metric can bind objective
feedback to the corresponding procedural structure. V16 proved that strict
outcome-only values retain a small causal signal, but independent query/key
networks over relation plus time behaved like a broad outcome mixture: swapping
the relation query or occupied values was nearly inert. V17 must learn content
correspondence; it may not encode a task, family, relation class, position rule,
answer, or target procedure.

## One compound learned-mechanism change

Create `SharedSemanticMetricCreditMemoryCore` as a small successor of the
consumed V16 strict core. Preserve its 512x32 state, fresh least-unused
allocation, outcome-only value embedding/transform, value-only bias-free
residual decoder, predict-before-feedback chronology, bounds, replay, and
failure behavior.

Replace V16's independent temporal-conditioned query and key networks with one
shared Siamese metric encoder. It receives only detached frozen 64-D V13
relation features, uses shared weights for write keys and read queries, and
returns normalized 32-D semantic codes. The same relation tensor must produce
the exact same code whether used for reading or writing. Public Moving-Origin
temporal features may continue to affect only read/write strength, inactive
fresh-slot erase, and generic slot-recency bias; time may not enter the semantic
code or residual decoder.

This is intentionally one compound semantic-address mechanism: shared q/k
weights, exclusion of time from content identity, and normalized metric codes
change together. V17 may establish the compound's effect but cannot attribute
that effect among those three subproperties. No subcomponent attribution is
claimed.

No contrastive/correspondence labels or loss are added in V17. The exact V15
causal objective must train the shared metric through downstream feedback. This
keeps the intervention to address geometry. If the shared metric is still not
causal, preserve V17 and only then consider a separately preregistered
contrastive correspondence objective.

## Owned outputs

This leaf owns only:

- `src/angler/reasoning/shared_semantic_metric_credit_memory.py`;
- `tests/unit/reasoning/test_shared_semantic_metric_credit_memory.py`;
- `experiments/runners/shared_semantic_metric_credit_v17.py`;
- `tests/unit/experiments/test_shared_semantic_metric_credit_v17.py`;
- one fresh V17 workstation checkpoint and train-development result.

Consumed V13/V14/V15/V16 sources, checkpoints, results, corpus, and tests are
read-only. Public package export is deferred until a supported sealed result so
historical source-chain hashes remain exact.

## Frozen training and evaluation

Reuse the exact V15 counterfactual-twin corpus, train/development partitions,
pair-once encoding, 96x8 schedule, AdamW settings, seed discipline, float32
determinism, true/deranged/detached-zero losses, paired-twin ranking, equal
1/1/1 weights, development order, affine control, and final seal. Reuse V16's
strict outcome-blind writer and query-only relation lesion. No additional
capacity, update, data, optimizer, loss, temperature, top-k selection,
consolidation, eviction, or replay training is permitted.

Use fresh seed `2026083117`. Pin consumed V16 result SHA-256
`5092DC199123606CE2F987C0DC91F322E271FEB1DEDD2BD36D5B7388F8B21C00`
and checkpoint SHA-256
`503F305D3702F639774BEB8CA777B94B62FAA0ECADED8F131541BF92DB8CF373`,
including their recorded source chain. The current V15/V16 imported scientific
source hashes must match those preserved packets before training.

Run and report every frozen V15/V16 gate without changing a threshold. Compute
two non-interchangeable booleans. `SHARED_SEMANTIC_CORRESPONDENCE_SUPPORTED`
requires the component clauses below but may coexist with streaming breadth or
retention failure. `FULL_DURABLE_SEMANTIC_CREDIT_SUPPORTED` requires both the
component gate and the entire frozen V16 durable-memory gate. Only the full
durable classification may authorize sealed final evaluation.

The component gate requires:

- exact architecture evidence that one shared relation-only encoder owns both
  read-query and write-key content and receives nonzero downstream gradient;
- after each probe prediction, use evaluator sidecars only to identify the
  unique current-pair acquisition anchor with the same relation class. Among
  all occupied slots, a top-1 success requires its read weight to be the unique
  strict maximum; exact ties fail. Overall top-1 fraction must be `>=0.75`,
  mean read mass on that anchor `>=0.30`, and mean mass margin versus the larger
  of the current opposite-relation anchor and every older same-relation anchor
  `>=0.10`. Top-1 fraction must also be `>=0.65` separately for every probe
  position and transition. Report every family stratum without a family-level
  threshold because each has few observations. Relation-class/family/position
  sidecars are collected only after prediction for measurement and are
  forbidden from tensorization, routing, learner input, or state;
- the query-only opposite-relation swap reduces matched accuracy by `>=0.10`
  **or** worsens NLL by `>=0.02`, with target read strength, time, base, labels,
  state, keys, and values exact;
- occupied key/value mismatch reduces matched accuracy by `>=0.10` **or**
  worsens NLL by `>=0.02`;
- fresh matched state beats each of zero and unrelated state by `>=0.10`
  balanced accuracy **and** `>=0.02` NLL, and true feedback beats deranged
  feedback by both of the same operands;
- true feedback beats post-acquisition reset, outcome-blind value writing, and
  the equal-budget affine calibrator by the unchanged V16 `>=0.10` balanced
  accuracy removal threshold;
- paired direction `>=0.75`, paired margin `>=0.20`, and opposite-state
  reversal `>=0.70` with reversed-label gain retention `>=0.80`;
- blind-writer integrity, replay, source/foundation identity, finite gradients,
  state bounds, and all intervention-integrity operands pass.

The full inherited durable-memory gate, including `>=0.70` streaming accuracy,
family/transition breadth, and `<=0.05` worst retention drop, is reported
separately and remains required for `FULL_DURABLE_SEMANTIC_CREDIT_SUPPORTED`.
A component pass with retention failure authorizes only a later consolidation
experiment; it does not authorize final evaluation and is not conversation
readiness, general reasoning, or promotion.

## Human impact, resources, and rollback

Local synthetic learning only. No external action, procedure execution,
service, personal/recovered data, deployment, promotion, Qwen/V13 mutation, or
human-outcome judgment. Human stop/deletion control is absolute. Qwen remains
on `cuda:0`; frozen V13 and V17 run on `cuda:1`; maximum allocation is 12 GiB
and maximum elapsed time after load is 60 minutes. Outputs are atomic. Rollback
removes only fresh V17 files/artifacts while preserving terminal evidence.

## Consumed first result

The single preflight-passing run completed in 180.842 seconds (183.72 seconds
command wall) and is consumed without tuning:

- result SHA-256
  `C74AA58CA0E78702366E024F6BFB928B94D5E627241986001133E8112DF72894`;
- checkpoint SHA-256
  `5AA492D38555AFEBCDBB674EA31ECFA9474E0A8A25BF5AE579F5205FB989D182`;
- peak Angler allocation `143129600` bytes;
- identity, architecture, event lineage, replay, bounds, source chain, and
  evaluator-sidecar isolation all passed.

The shared metric improved causal use but did not learn the claimed
correspondence. True streaming BA/NLL was `0.5677/0.7034` versus deranged
`0.4323/0.9096` and zero `0.5000/0.8586`. Fresh matched-state BA reached
`0.7240`; occupied key/value mismatch fell to `0.6458` and worsened NLL by
`0.0723`. Ten of twelve families improved, with at least two in every
transition. These are improvements over V16.

However evaluator-only corresponding-anchor top-1 was only `0.2917`, mean
matching mass `0.2267`, and mean margin `-0.0071`. The query-only opposite-
relation lesion reduced BA by only `0.0182` and worsened NLL by `0.0192`, below
the frozen gate. Paired direction was `0.6146`, opposite-state reversal
`0.4792`, and worst retention drop remained `0.50`. Component and full durable
booleans are both false; final remains sealed.

V17 therefore shows that a shared relation-only geometry helps causal binding
but the outer feedback objective remains assignment-underdetermined. The
preregistered successor is now justified: add one explicit learned
correspondence objective derived only from public/frozen structural inputs,
with no outcome, answer, relation-class, task/family, position, or regime value
entering learner inputs or inference. Do not add consolidation until that
objective makes correspondence causally identifiable.

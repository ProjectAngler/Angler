# ANG-WORK-LEARNING-GROUPED-PUBLIC-RELATION-GEOMETRY-V20-001

Status: ready for independent prelaunch audit. V19 and all earlier identities
are consumed. The first preflight-passing V20 result is accepted without
tuning.

## Outcome and claim boundary

Test the cheapest prerequisite exposed by V19's failure: whether explicit
outcome-blind grouping of **distinct public task instances** can train a shared
representation in which the same public procedure relation corresponds across
new surfaces and generator families.

The maximum claim is `PUBLIC_RELATION_CORRESPONDENCE_GEOMETRY_SUPPORTED`.
This is not operator execution or transfer, target-attempt construction,
outcome credit, memory, continual learning, Cognee or Moving Origin benefit,
Qwen improvement, general reasoning, or AGI. It is Stage A only. Only a
supported result may justify a separate leaf that freezes this representation
and tests `A:B::C:?`; only that later result may justify memory integration.

Deterministic code may generate, group, schedule, hash, lesion, and score
synthetic public pairs. It may not emit relation features, choose a prediction,
prescribe a procedure, or expose the group label at inference.

## Frozen data

Reuse the consumed V13 public paired-natural-trace corpus without modification.
The five evaluator-only public relation groups are `paraphrase`, `reorder`,
`replacement`, `omission`, and `insertion`. A learner row contains only public
reference and attempt text. Outcome, relation label, mechanism/family identity,
structure signature, renderer plan, position, temporal coordinate, and every
other sidecar are rejected by the learned interface.

Materialize exactly:

- train: 384 V13 inner mechanisms x 6 rows = 2,304 public relation instances;
- development: 48 generator-disjoint mechanisms x 6 rows = 288 instances;
- final: 48 mechanisms remain sealed and unmaterialized.

Canonical public-payload and evaluator-sidecar SHA-256 digests are recorded
separately. Train/development raw public pairs and structural signatures must be
disjoint. Duplicate canonical public pairs fail closed.

Every positive pair in training and evaluation must be two different mechanism
identities and two different raw public pairs. Within a training batch, positive
members of one relation also use different generator families; if a scheduled
batch cannot satisfy that condition, preflight fails rather than substituting.
No same-row augmented view, counterfactual twin, duplicated text, or same-task
positive is permitted.

## Frozen representation and objective

Encode every public `(reference, attempt)` pair once with the exact consumed
frozen Qwen/V13 pair-once path. Each detached finite contiguous float32
64-vector is then processed by:

```text
LayerNorm(64)
-> Linear(64, 128, bias=False)
-> SiLU
-> Linear(128, 64, bias=False)
-> L2 normalize
```

The full arm trains only this map with the supervised contrastive loss of
Khosla et al. (NeurIPS 2020), temperature `0.10`. For anchor `i`, positives are
all other batch rows with the same evaluator-only public relation group;
denominator members are every other row. Self-comparisons are masked. The loss
is the mean negative log positive probability, first averaged over each
anchor's positives and then uniformly over anchors. Float32 math and epsilon
`1e-8` are fixed. Relation labels create the loss mask only; they never enter a
module tensor, checkpoint input, embedding call, or evaluation query.

Three arms begin byte-identical:

1. `grouped`: the exact relation grouping above;
2. `group_deranged`: identical public rows, order, optimizer, and updates, but
   its group-mask assignment changes the equivalence relation inside every
   batch after row scheduling. All 24 failure rows are assigned to one
   deranged `paraphrase` group. The 24 original paraphrase rows, in canonical
   row rank, are split into four consecutive blocks of six assigned to
   deranged `reorder`, `replacement`, `omission`, and `insertion` groups;
3. `objective_off`: no optimizer and no update.

The deranged mask is frozen before encoding, changes the label of every row,
and preserves the exact `24,6,6,6,6` per-batch group counts. Its same-group
positive-mask SHA-256 must differ from the grouped mask SHA-256. It changes no
row, outcome, order, batch, or optimizer operand.

## Frozen schedule and optimizer

- seed `2026083120` for schedule; seed `2026083121` for initialization;
- exactly 96 updates, one batch per update, 48 rows per batch;
- exactly 24 paraphrase rows and six rows from each of reorder, replacement,
  omission, and insertion per batch, matching the frozen corpus's exact
  4:1:1:1:1 relation frequency;
- rows of one relation within one batch come from distinct mechanisms and
  distinct generator families; the 24 selected failure rows jointly also come
  from 24 distinct mechanisms and 24 distinct generator families;
- the complete 4,608-row schedule is generated once on CPU, SHA-256 bound, and
  reused byte-for-byte by grouped and deranged arms;
- every one of the 2,304 train rows appears exactly twice;
- AdamW `lr=3e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `1e-4`,
  global gradient-norm clip `1.0`, float32, one step per update;
- no task/outcome loss, replay, scheduler, warmup, early stopping, resume,
  checkpoint selection, model inference inside an update, or adaptive batch.

A disposable one-update clone must give every trainable parameter finite
nonzero gradient without changing launch arm, optimizer, or RNG bytes.

## Development geometry evaluation

Use only the 288 development rows, no outcomes and no gradients.

1. `distinct-instance top-1`: for each query, compare cosine similarity to a
   gallery containing one row from each relation group. The matching exemplar
   is always from a different mechanism, family, and raw pair. Use 32 frozen
   balanced galleries per query; candidate relation-position counts differ by
   at most one, and the actual correct-target-position histogram is bound in
   the gallery-schedule record. Exact ties are wrong.
2. `all-pairs retrieval`: rank all other development rows after excluding the
   query's mechanism and family. Report same-relation recall@1, mean reciprocal
   rank, and mean same-relation mass under temperature `0.10`.
3. Report every metric overall and independently for all five relation groups,
   plus per-dimension variance, effective rank, and off-diagonal cosine.

Evaluate grouped, group-deranged, and objective-off checkpoints on identical
galleries. On the grouped checkpoint also run:

- `reference_deranged`: rotate reference texts by one canonical row within
  relation, holding attempt, gallery membership, positions, and target labels;
- `attempt_deranged`: the symmetric attempt-only rotation;
- `pair_direction_reversed`: swap reference and attempt in every query and
  gallery pair;
- `embedding_dimension_permuted`: apply one fixed 64-D permutation jointly to
  every query/gallery embedding; all metrics must remain exact within `1e-7`.

The three semantic lesions are fixed-point-free, selected before embeddings or
metrics are read, and branch independently from the same immutable public rows.
They may not train or feed another lesion. Operand digests prove only the named
text field changed.

The gallery schedule is constructed before any public text is encoded or any
metric is read. For canonical development query index `q`, repeat `g` in
`0..31`, and candidate relation order is the five-relation canonical order
rotated by `(q + g) mod 5`. For each candidate slot in that order, sort eligible
rows by canonical public-pair SHA-256, exclude the query mechanism/family and
every mechanism/family already selected in that gallery, hash
`b"angler.grouped-public-relation-geometry.v20.gallery\x00" +
ascii(q) + b":" + ascii(g) + b":" + relation`, interpret the first eight
digest bytes as one unsigned big-endian integer, and select modulo the eligible
count. The same-relation selection is the sole correct exemplar. This makes all
five gallery members mutually mechanism/family-distinct and query-disjoint.
The complete ordered `(query, repeat, candidate row indices, target position)`
table is canonical-JSON serialized and SHA-256 bound before preflight.

Representation statistics use exactly the grouped checkpoint's `[288,64]`
development embeddings in canonical row order. Convert once to float64 for
statistics. Per-dimension standard deviation is population standard deviation
with correction zero. Effective rank centers each dimension, forms
`C = X_centered.T @ X_centered / 288`, computes `eigvalsh(C)`, clamps negative
roundoff eigenvalues to zero, sets `p = eigenvalue / sum(eigenvalues)`, and
reports `exp(-sum(p * log(p)))` over strictly positive `p`; a zero sum or any
non-finite operand fails. Mean absolute off-diagonal cosine is the arithmetic
mean of `abs(x_i @ x_j)` over every upper-triangle pair `i < j` after each row
is L2 normalized with epsilon `1e-8`; the diagonal is excluded.

## Frozen classification

`PUBLIC_RELATION_CORRESPONDENCE_GEOMETRY_SUPPORTED` requires:

- all identity, corpus, distinct-positive, balance, no-label-input, pair-once,
  arm-parity, gradient, mutation, resource, and seal checks pass;
- grouped distinct-instance top-1 `>=0.65` (chance `0.20`) and all-pairs
  recall@1 `>=0.55`;
- grouped exceeds objective-off by `>=0.15` top-1 and `>=0.12` recall@1;
- grouped exceeds group-deranged by `>=0.12` top-1 and `>=0.10` recall@1;
- grouped distinct-instance top-1 is `>=0.50` for each of all five relations;
- reference derangement and attempt derangement each reduce top-1 by `>=0.10`
  or recall@1 by `>=0.08`;
- direction reversal reduces top-1 by `>=0.08` or recall@1 by `>=0.06`;
- the joint dimension permutation is metric-exact within `1e-7`;
- at least 48 embedding dimensions have development standard deviation
  `>=1e-3`, effective rank is `>=24`, and mean absolute off-diagonal cosine is
  `<=0.90`.

Integrity failure is `INVALID_NO_CLAIM`; otherwise any missed scientific clause
is `DEVELOPMENT_NOT_SUPPORTED`. The categories are exclusive. The first
preflight-passing result consumes V20. Do not tune the corpus, grouping,
architecture, temperature, optimizer, schedule, seed, lesions, or gates under
this identity. V20 has no final phase; a development pass authorizes only a new
audited final leaf.

## Required mechanics and preflight

Tests and a real encoded-tensor preflight must prove:

- exact counts, relation balance, partition separation, distinct-instance
  positives, group-derangement counts, unequal grouped/deranged positive-mask
  hashes, schedule hash, and sealed final;
- group labels and every V13 sidecar are absent from the learned API and
  checkpoint input; mutating them cannot alter embeddings;
- support/public row order permutations produce corresponding embedding
  permutations and unchanged SupCon loss within `1e-7`;
- grouped/deranged initial modules and optimizers are byte-identical; on one
  noncollapsed mechanics batch their positive masks, losses, and gradients
  differ while public rows/order and optimizer configuration remain exact;
  objective-off remains initial-byte exact;
- Qwen/V13 are immutable; relation vectors are detached and fixed before fit;
- consumed V19 bindings, source closure, environment/device versions, output
  absence, atomic checkpoint/result writes, and no-final parser are exact.

## Frozen donors and sources

- V19 result:
  `263B14ED0A73C7B487DE5616D8AF6BFEB4C4AB11C8C31A5855B00483C69A0A1F`;
- V19 full/unary checkpoints:
  `8FAB693FB88DCC86BE81E0BBDEF5B8DBF591D64FADEF2241C7382F5AB4FE234B`,
  `8C9D79B8D1411EE2C861492D04765625B3F92A0D280158F5008F712CE4DF6593`;
- V19 runner:
  `D9820298598B3044DBAADDB3AF5CB7DC2FB19D9EBB4D8D642F84FF4AFF55B063`;
- V13 corpus:
  `79B770EABF217663882873827D98F5F17B0365DF82CDC4C5DAE45FD42BD129B7`;
- V13 learned representation:
  `3FF543F15EC70BDC2103501CC21F35A7D6ACBDE6601069360E554D97257F7E22`;
- historical reasoning export:
  `B06E4F87620974FD5C2F2AD75A433DBE55A9CBA799B0F442C5FE9A51785D5798`.

The objective adapts the public algorithmic idea in Khosla et al., *Supervised
Contrastive Learning* (NeurIPS 2020); no external code is copied. V13 supplies
Angler's frozen public pair representation. Cognee and Moving Origin are not
used because persistence and time cannot establish relation geometry and would
confound this prerequisite.

## Owned outputs

- `src/angler/reasoning/grouped_public_relation_geometry.py`;
- `tests/unit/reasoning/test_grouped_public_relation_geometry.py`;
- `experiments/runners/grouped_public_relation_geometry_v20.py`;
- `tests/unit/experiments/test_grouped_public_relation_geometry_v20.py`;
- `/opt/angler/results/grouped-public-relation-geometry-v20-grouped.pt`;
- `/opt/angler/results/grouped-public-relation-geometry-v20-deranged.pt`;
- `/opt/angler/results/grouped-public-relation-geometry-v20-objective-off.pt`;
- `/opt/angler/results/grouped-public-relation-geometry-v20-development.json`.

No corpus, package export, `__init__.py`, consumed source/result/checkpoint,
Cognee store, Moving Origin state, Qwen weight, service, deployment artifact,
or final artifact may change.

## Impact, resources, stop, rollback

Local synthetic representation learning only. No external action, executable
procedure, personal/recovered data, network, service, deployment, promotion,
model mutation, or human judgment. Qwen remains on `cuda:0`; V13/V20 use
`cuda:1`. Maximum V20 allocation is 12 GiB and elapsed time after model load is
30 minutes. Human stop/deletion control is absolute.

Stop before fit on any hash mismatch, output collision, final access, duplicate
positive, balance failure, label/sidecar leakage, arm inequality outside the
declared mask, device mismatch, nonfinite value, model mutation, or resource
breach. Rollback removes only fresh V20 files/artifacts; terminal result and its
bound checkpoints remain preserved consumed evidence.

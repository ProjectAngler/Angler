# ANG-WORK-LEARNING-PAIRED-PUBLIC-RELATION-CREDIT-V19-001

Status: ready for implementation and independent prelaunch audit. V18 and all
earlier identities are consumed and immutable. The first preflight-passing V19
result will be accepted without tuning.

## Outcome and bounded claim

Test whether a learned joint comparison between the current public relation and
each stored public relation supplies the cross-event structural address that
V17's unary shared metric and V18's same-row augmentation objective did not.

The claim is deliberately `learned paired public-relation comparison`, not
semantic equivalence, assignment, graph matching, general reasoning, or durable
continual learning unless the corresponding causal gates pass.

V18 proved that same-row mask invariance can improve its own statistics while
hurting cross-event retrieval. V19 removes that auxiliary loss entirely. It
borrows the supported pooled pair-comparison pattern from Angler's earlier
Phase-6 V19/V20 work, not the unsupported GMN cross-attention attribution:

`[0.5*(q+s), abs(q-s), q*s] -> shared MLP -> bias-free scalar residual`.

This is an Angler-owned adaptation of the Graph Matching Network whole-object
comparison idea and the project's OML-shaped relation comparator. It also
reuses V11's outcome-blind public-sidecar state pattern, V17's strict
factorized outcome values, DNC-style least-used allocation, and Moving Origin's
existing temporal-strength/recency role. Cognee remains retrieval/persistence
infrastructure and cannot select or route memory slots.

## One learned-mechanism change

Create `PairedPublicRelationCreditMemoryCore` as a V17 successor.

- Preserve V17's shared normalized 32-D semantic q/k metric, 512x32 key/value
  state, outcome-only value writer, value-only bias-free decoder, allocation,
  temporal strengths/recency, chronology, replay, bounds, and failure behavior.
- Add one detached outcome-blind `[512,64]` public relation sidecar, one row for
  every memory slot. Feedback exact-overwrites the already-observed relation row into the
  exact same freshly allocated slot as its semantic key and outcome-derived
  value, independently of write strength and erase strength; every unused slot
  remains exact zero. It may
  not receive the outcome, value, slot index, family, relation class, position,
  regime, task/episode ID, evaluator metadata, or any Cognee backend identity.
- For each query and occupied slot, form only the symmetric public-pair features
  `0.5*(q+s)`, `abs(q-s)`, and `q*s`. A shared
  `LayerNorm(192) -> Linear(192,64) -> SiLU -> Linear(64,1,bias=False)` scorer
  emits a scalar. Its final weight is initialized to exact zero. Add
  `2.0*tanh(raw)` to the inherited semantic-content logit before the unchanged
  temporal bias and usage mask. The pair scorer sees neither Moving Origin nor
  state/value metadata. Unused slots receive exact-zero residual.
- The scorer is trained only through the unchanged V15/V17 causal task loss.
  No auxiliary objective, pair label, positive/negative table, explicit
  correspondence target, reconstruction target, clustering, outcome-weighted
  route, or deterministic assignment rule is added.

The full and unary arms instantiate byte-identical V19 cores. Full enables the
pair residual; unary disables it for every train/evaluation call while keeping
the same architecture, task rows, feedback, optimizer configuration, 96x8
updates, and evaluator. At initialization both predictions and task gradients
outside the zero-initialized scorer are exact. The full scorer head must receive
a nonzero first-update gradient; all scorer parameters must receive finite
nonzero gradients after a bounded two-update mechanics smoke. Numeric equality
is not required after the learned residual makes the arms diverge.

The two-update mechanics smoke runs only on disposable cloned cores and fresh
disposable optimizers, which are destroyed afterward. It may not update either launch arm. After the smoke,
the actual full and unary launch model digests and optimizer states must still
equal their recorded initial values exactly; their initial states and the
global CPU/CUDA RNG digests must also remain exact before official update 0.

## Frozen identity and donors

- experiment seed: `2026083119`;
- consumed V18 development result SHA-256:
  `8289750FF87394493E0E23681BC52128CC55429F291A84488E700179F9FE87A9`;
- consumed V18 full checkpoint SHA-256:
  `BBCD20BCE682936181BDB5FA89A72674BE92F08A54515A35024EE6776F39AD80`;
- consumed V18 objective-off checkpoint SHA-256:
  `27212FB2B531CCA13B29D1F8CC69343486BC16998E6242875F23B768B931ED2C`;
- consumed V18 runner SHA-256:
  `77382FA24C76BD9A1B221F3B2F808C0484F2BFCC1DC65BF433193963293B49F2`;
- consumed V18 runner test SHA-256:
  `D26042AB559CD4D0B127B9BE1D80FB78655057AB93DEF696022391D7F2855E35`;
- V17 core SHA-256:
  `A25D42E9A4B8697FC441FC4C55543BFABD661AD5BD7756C4BF87EA80EAB14790`;
- Phase-6 paired comparator donor SHA-256:
  `54A8E2E510424E485DE34A2975A82C927D22C87B5576EFE00537545158ECE5BE`;
- Phase-6 OML relation donor SHA-256:
  `6611E60BAB8D1F3C80A68BEB66AAC010F236B107B2A5E9060201BA56A50E86E3`;
- V11 public-sidecar donor SHA-256:
  `99C2452B6F42E6241CC2B4CC4CA86DA8803E4B08924E3D1376A0813460070C7D`;
- historical reasoning export SHA-256:
  `B06E4F87620974FD5C2F2AD75A433DBE55A9CBA799B0F442C5FE9A51785D5798`.

The consumed V18 result must be read as JSON and its recorded source map,
classification, preflight, identity, mask, final seal, and checkpoint hashes
must match. V19 starts two fresh byte-identical cores; it does not load or
continue either V18 checkpoint, mutate a consumed artifact, or inherit V18's
failed Barlow objective.

Reuse the exact V15 counterfactual-twin corpus, partitions, pair-once frozen
Qwen/V13 encoding, 96x8 schedule, task optimizer/loss, predict-before-feedback
chronology, evaluator order, sidecar timing, affine control, resource checks,
and final seal. Every V17 component and durable threshold remains unchanged.

## Owned outputs

This leaf owns only:

- `src/angler/reasoning/paired_public_relation_credit_memory.py`;
- `tests/unit/reasoning/test_paired_public_relation_credit_memory.py`;
- `experiments/runners/paired_public_relation_credit_v19.py`;
- `tests/unit/experiments/test_paired_public_relation_credit_v19.py`;
- `/opt/angler/results/paired-public-relation-credit-v19-full.pt`;
- `/opt/angler/results/paired-public-relation-credit-v19-unary.pt`;
- `/opt/angler/results/paired-public-relation-credit-v19-development.json`.

No consumed source, package export, checkpoint, result, or final artifact may
change. Public export remains deferred until a separately supported result.

## Required mechanics and preflight

Tests and real encoded-tensor preflight must prove:

- the public sidecar is detached, finite, outcome-independent, slot-aligned,
  replay-exact, snapshot/restore-exact, zeroable, and bounded by fixed shape;
  capture, restore, state digest, zero-state construction, replay event
  reconstruction, and state cloning must all include its exact bytes;
- sidecar writes remain exact when outcomes, relation classes, families,
  positions, regimes, IDs, and evaluator metadata change;
- initial full/unary parameters, predictions, ordinary task operands, non-scorer
  gradients, optimizer configuration, data, feedback, and chronology are exact;
- scorer inputs contain only the current and stored public relation rows and
  are invariant to changing keys, values, positive usage magnitudes while the
  occupied bitmask remains fixed, temporal state, outcomes, labels, and
  evaluator sidecars. Changing usage across zero is excluded because the
  occupied bitmask must exact-zero the residual of an unused slot. `positions`
  in these checks means evaluator metadata only, never event order or public
  temporal coordinates;
- joint permutation of keys, values, usage, acquisition, and public relation
  sidecars produces an exactly corresponding permutation of read weights and
  invariant logits within `1e-6`;
- public-sidecar-only occupied-slot derangement is frozen as an ascending
  occupied-slot rotate-by-one (and is valid only with at least two occupied
  slots). It is selected before outcomes, labels, families, or metric values
  are read, has zero fixed points within every evaluated occupied set, and
  leaves keys, values, usage, acquisition, time, labels, and semantic queries
  exact;
- a pair-query-only swap is frozen within every episode as `2 <-> 3` and
  `4 <-> 5`, equivalently destination order `[3,2,5,4]` for source probe order
  `[2,3,4,5]`. It is selected before outcomes, labels, families, or metric
  values are read; it must have zero fixed points, preserve the complete raw
  query-row multiset and identical family/transition/probe-position exposure,
  and mechanically confirm that every scored row receives a different raw
  pair query. It changes the raw pair query while holding the semantic
  query, read strength, temporal features, base logits, labels, keys, values,
  usage, acquisition, and stored public rows exact;
- the residual-zero path is exact V17 addressing for the same V19 parameters;
- the full scorer head has nonzero first-update gradient and every scorer
  parameter has finite nonzero gradient by the second mechanics update; unary
  scorer parameters receive no gradient;
- consumed hashes, source closure, environment/device/version evidence, output
  absence, atomic writes, dual-checkpoint/result bindings, and no-final parser
  are exact.

Evaluator compatibility may use a tightly scoped evaluator adapter for the
new sidecar state, but deterministic code may only preserve/lesion state and
measure outcomes. It may not select a slot or prescribe an answer.

Residual-zero, public-sidecar-deranged, and pair-query-swapped probes each
branch from the same exact post-acquisition/pre-probe state snapshot. No lesion
receives probe feedback, writes memory, steps an optimizer, or becomes the
input to another lesion. Each branch's full state digest immediately after
prediction must equal that same branch's digest immediately before prediction,
and the source digest remains exact. Residual-zero and query-swap branch
digests equal the source. The sidecar-deranged branch differs from the source
only in the declared public-sidecar bytes, while a non-target-state digest that
excludes only those declared bytes remains source-exact. `matched` BA/NLL means
the exact unchanged V17 matched state-control operands, not a newly selected
comparison.

## Frozen development interpretation

Report full and unary metrics separately plus pair residual magnitude,
nonzero-residual fraction, changed-top-read fraction, and every existing V17
retrieval/task/retention operand.

`PAIRED_PUBLIC_RELATION_COMPARISON_SUPPORTED` requires all of:

- every identity, mechanics, sidecar, replay, permutation, gradient, chronology,
  resource, and final-seal check passes;
- full minus unary corresponding-anchor top-1 is `>=0.10`;
- full minus unary mean corresponding-anchor mass is `>=0.05`;
- full minus unary mean corresponding-anchor margin is `>=0.05`;
- full true-feedback BA is not below unary;
- full true-feedback NLL is no more than `0.02` worse than unary;
- the full arm independently passes every frozen V17
  `SHARED_SEMANTIC_CORRESPONDENCE_SUPPORTED` component clause;
- on the full trained core, residual-zero, public-sidecar derangement, and
  pair-query-only swap each reduce matched BA by `>=0.10` or worsen NLL by
  `>=0.02`, with each lesion's declared non-target operands exact;
- residual-zero eliminates the full-minus-unary corresponding-anchor advantage
  to below `0.02` top-1 and `0.02` matching mass;
- at least half of development probe rows have a nonzero pair residual and at
  least `0.10` of probe rows change their strict top-read slot versus residual
  zero; exact ties do not count.

`FULL_DURABLE_PAIRED_PUBLIC_RELATION_CREDIT_SUPPORTED` additionally requires
the entire unchanged V17 durable-memory gate, including streaming accuracy,
family/transition breadth, retention, causal removals, replay, and bounds. V19
owns no final phase: even a full durable development pass can only justify a
new audited final-evaluation leaf.

If the pair scorer fails to beat unary or survives its public-row lesions,
preserve V19 without tuning and move to the preregistered larger alternative:
explicitly grouped, distinct public demonstrations with a staged representation
objective and balanced deranged grouping. Do not tune scorer size, residual
bound, seed, update count, thresholds, or task loss under V19.

## Human impact, resources, and rollback

Local synthetic learning only. No external action, procedure execution,
service, personal/recovered data, deployment, promotion, Qwen/V13 mutation, or
human-outcome judgment. Human stop/deletion control is absolute. Qwen remains
on `cuda:0`; V13 and V19 run on `cuda:1`. Maximum Angler allocation is 12 GiB
and maximum elapsed time after model load is 60 minutes.

Stop before training on any source/artifact mismatch, preflight failure, output
collision, device mismatch, non-finite value, sidecar misalignment, scorer
leakage, slot-permutation failure, unary scorer gradient, failure to reproduce
the paired initialization/task stream, or final-partition access. Rollback
removes only fresh V19 files/artifacts while preserving terminal evidence.

## Consumed first result — 2026-08-31

The first preflight-passing identity is consumed as `DEVELOPMENT_NOT_SUPPORTED`.
It is not eligible for tuning, rerun, final evaluation, export, or promotion.

- terminal result SHA-256:
  `263B14ED0A73C7B487DE5616D8AF6BFEB4C4AB11C8C31A5855B00483C69A0A1F`;
- full checkpoint SHA-256:
  `8FAB693FB88DCC86BE81E0BBDEF5B8DBF591D64FADEF2241C7382F5AB4FE234B`;
- unary checkpoint SHA-256:
  `8C9D79B8D1411EE2C861492D04765625B3F92A0D280158F5008F712CE4DF6593`;
- elapsed time `404.85918548000336` seconds; peak Angler allocation
  `293636096` bytes; preflight, identity, protocol, final partition, and both
  checkpoint seals passed.

The full arm modestly improved true-feedback behavior over unary: balanced
accuracy `0.6171875` versus `0.5885416666666667`, and NLL
`0.6777545374158459` versus `0.6953705947380513`. It did not learn the required
cross-event correspondence: anchor top-1 improved only `+0.0104167`, matching
mass only `+0.0192316`, and matching margin worsened `-0.0110146`, all below
the frozen gates.

The pair path was active on every probe and changed strict top-read on
`0.1197917` of probes. Public-sidecar derangement and pair-query swap each
worsened NLL by more than `0.02`, but exact residual zero left matched balanced
accuracy unchanged and worsened NLL only `0.0059834`. Therefore the trained
pair inputs had some causal influence, but the direct paired residual did not
cause the task/correspondence benefit required by the claim; co-adaptation of
the inherited path remains a viable explanation.

Per the frozen interpretation, the next eligible direction is not V19 tuning.
It is a fresh staged successor using explicitly grouped, distinct public
demonstrations and balanced deranged grouping to teach transferable public
relation/operator structure before causal task credit. V19 remains immutable
negative evidence.

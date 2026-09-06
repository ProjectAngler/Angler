# ANG-WORK-LEARNING-PAIRED-LATENT-CONTINGENCY-CREDIT-V15-001

Status: ready for bounded implementation. This is a fresh successor to the
consumed V14 `STRUCTURE_REPLICATED_NOT_ADAPTIVE` result. It may not alter,
rerun, relabel, or tune V13/V14 evidence.

## Outcome and falsifiable claim

Determine whether the unchanged V14 learned memory can acquire a hidden
procedural contingency from objective feedback and apply it to fresh surfaces.
The current probe input alone must be insufficient: paired counterfactual
episodes have byte-identical public inputs and base logits but opposite correct
probe outcomes. The only model-visible difference before a probe is earlier
objective feedback. This tests learned adaptation, not static classification,
confidence calibration, retrieval alone, or a deterministic solver.

## Frozen inputs and owned outputs

Freeze Qwen3-4B, the complete V13 protected procedural representation and its
consumed evidence, and the V14 memory source architecture. Instantiate a fresh
V15-seeded V14 core; train only its shared neural parameters. No regime, task,
family, renderer, corruption, transition, or expected-answer identifier may
enter learner features.

This leaf owns only:

- `experiments/corpora/paired_latent_contingency_credit_v15.py`;
- `experiments/runners/paired_latent_contingency_credit_v15.py`;
- `tests/unit/experiments/test_paired_latent_contingency_credit_v15_corpus.py`;
- `tests/unit/experiments/test_paired_latent_contingency_credit_v15.py`;
- one fresh V15 checkpoint and one train-development result on the workstation.

Final data remains sealed unless every development gate passes. Deterministic
code may counterbalance, schedule, check integrity, and measure results; it may
not decode a latent regime or choose an answer for the learner.

## Corpus and identifiability contract

Each episode contains two acquisition anchors followed by four fresh probes.
Two abstract procedural-relation classes and four balanced latent two-bit
regimes determine objective outcomes. Every public reference, attempt, base
logit, temporal sequence, and frozen encoded relation has an exact
counterfactual twin under the opposite regime. Matched twins have opposite
probe outcomes. Train and development families/surfaces are disjoint.
All four scored probes branch independently from the exact same state after
the two acquisition writes; no scored probe outcome may be written before any
other scored probe. Optional online probe-feedback behavior is secondary and
cannot contribute to the development gate.

Mechanical preflight must establish byte-exact twin input tensor hashes,
pre-feedback logit equality, one bit of label entropy for each matched probe
input, balanced outcomes at every position and across family/transition
groups, exact balance of relation class and outcome polarity at every anchor
and probe position, absence of public regime tokens, and zero train/development overlap.
Acquisition labels may update state but do not count as probe success.

## Training protocol

Use the frozen V14 optimizer configuration and exactly 96 updates with eight
base episodes per update; schedule every declared training episode equally.
Each state spans exactly one deterministic eight-pair block (16 anchor writes
per twin side) and resets only at the next block. Development uses the same
eight-pair horizon in six fixed blocks; it may not expose a longer state horizon
than training. Retention compares the first acquired ability in each block
before and after the other seven pairs, then reports the worst drop.
For each counterfactual pair, begin true, balanced-deranged, and zero/no-write
states from byte-identical starts on identical public inputs. Predict before
every feedback write. Train only on probe predictions using equal-weight:

1. ordinary objective outcome loss on the true-history probes;
2. the mean of `relu(0.05 + L_true - L_control)` for deranged and zero
   controls, where every loss uses the same true probe labels;
3. mean matched-twin `relu(0.20 - y*(logit_true-logit_opposite))` requiring
   outcome-directed paired probe separation.

True and deranged losses must remain differentiable through their own two-write
chronological state paths. Zero/no-write is intentionally a fixed detached
baseline inside the separation term. Fixed weights are `1/1/1`; no coefficient, step, seed, corpus,
threshold, or schedule changes are permitted after the first result begins.

Balanced derangement is a fixed-point-free permutation of complete two-bit
acquisition histories across matched episodes/regimes, preserving every global
per-position label count. Every scored episode's deranged acquisition history
must differ from true before probe one; within-episode anchor reordering alone
is not an admissible derangement.

## Development controls and gate

Evaluate fresh held-out twins with true feedback, balanced deranged feedback,
zero/no-read, post-acquisition reset before the four probes, an equal-budget train-only affine base-logit
calibrator, an outcome-blind writer, opposite-twin state swaps, genuinely
unrelated-family state, key/value mismatch, exact event replay, and early
ability retention without replay. All scored twin probes use identical state
checkpoints and matched current inputs.
The outcome-blind writer must preserve keys, gates, allocation, and write count
while replacing only outcome content with one shared outcome-independent token.
An unrelated state must come from a different held-out family and a different
two-anchor history. Key/value mismatch permutes only occupied values among
occupied keys, leaving usage, acquisition, write count, and unoccupied slots
unchanged; it may not collapse into a zero-state intervention.

`PAIRED_LATENT_PROCEDURAL_CREDIT_SUPPORTED` requires all of:

- identifiability/integrity preflight is exact and frozen V13 structural
  evidence, representation, Qwen, and all source hashes remain exact;
- true-history development probe balanced accuracy is at least `0.70`;
- true beats both deranged and zero by at least `0.10` balanced accuracy **and**
  `0.02` mean probe NLL;
- paired-twin directional accuracy is at least `0.75` with mean
  outcome-directed logit margin at least `0.20`;
- swapping opposite-twin states changes an originally correct outcome-directed
  prediction into the opposite preference on at least `0.70` of probes and
  preserves at least `0.80` of full causal gain under the reversed labels;
  both gains are signed and must be positive—absolute-value scoring is invalid;
- true beats post-acquisition reset, affine calibration, and outcome-blind writer
  by at least `0.10` balanced accuracy;
- at least 9/12 held-out families improve over deranged history, with at least
  two improved families in every transition group;
- true-versus-deranged feedback changes candidate ordering on at least `0.25`
  of informative matched probes;
- terminal zero, genuinely unrelated, and key/value-mismatched states each
  lose at least `0.10` balanced accuracy or `0.02` NLL versus matched state;
- replay reconstructs state and logits within `1e-6`, early-family retention
  drop is no more than `0.05`, and every state/gradient/read/write score is
  finite and within its declared bound.

If identifiability fails, stop before training. If the first complete result
fails, preserve it and isolate the already-proposed strict key/value feedback
bottleneck as the next architectural successor; do not tune or rerun V15.

## Human impact, resources, and rollback

Local synthetic experimentation only. No procedure execution, autonomous
external action, network service, personal/recovered data, Qwen/V13 mutation,
deployment, promotion, or human-outcome judgment is authorized. Human stop and
deletion control are absolute. Use Qwen on `cuda:0` and the V13/V15 components
on `cuda:1`, float32, no autocast/TF32, at most 12 GiB allocated on the Angler
device, and at most 60 minutes after model load. Outputs are atomic. Rollback
removes only fresh V15 files/artifacts while preserving terminal evidence.

## Pre-training recovery note — 2026-08-31

The initial launch stopped at identifiability preflight before optimizer
creation, training, or artifact output. Identical public twins had been encoded
as separate GPU rows; their public payload hashes matched but encoded tensor
hashes and pre-feedback logits were not bit-identical. All other preflight
conditions passed. Recovery encodes each public pair once and reuses the exact
frozen relation/base/temporal tensors for both supervision twins. This preserves
the intended experimental input, strengthens byte equality, reduces duplicate
work, and changes no corpus, loss, schedule, seed, gate, or threshold. The
recovered runner requires fresh hashes and the complete preflight/test sequence
before training.

## Consumed result — 2026-08-31

After two preserved pre-training stops (duplicate-encoding equality and scalar
margin shape), the first identity to pass preflight and complete all 96 updates
finished in 166.328 seconds as `DEVELOPMENT_NOT_SUPPORTED`. Result SHA-256 is
`E2DBEE07C91189736382CB2D3EBCBE1162653F19C84336894C9F5F2FD1F6C0C3`;
checkpoint SHA-256 is
`DE63B238125C307ED745FD3998362CE845A47875A199A70FB2B7146FC9C4F5B1`.
Final remains sealed and V15 must not be rerun or tuned.

V15 establishes genuine feedback-specific acquisition. True history achieved
0.6042 balanced accuracy / 0.6453 NLL, versus deranged history 0.3958 / 0.8283
and zero/reset 0.5000 / 0.8586. Outcome-blind writing remained 0.5000; all 12
held-out families improved; all 48 pair orderings changed; paired directional
accuracy was 0.7240 with margin 0.3660; a fresh matched state reached 0.7370,
while unrelated state fell to 0.5130. Exact replay and all bounds passed.

The full gate failed because true accuracy was below 0.70, paired direction was
below 0.75, outcome-directed swap reversal was 0.5625 rather than 0.70, and
retention after seven later pairs dropped by as much as 0.75. Key/value mismatch
retained 0.7057 accuracy, showing insufficient binding despite a 0.025 NLL
penalty. The evidence supports causal fast ability acquisition but not durable,
precisely bound continual ability memory. The preregistered next isolated change
is a strict key/value feedback bottleneck on the unchanged V15 corpus.

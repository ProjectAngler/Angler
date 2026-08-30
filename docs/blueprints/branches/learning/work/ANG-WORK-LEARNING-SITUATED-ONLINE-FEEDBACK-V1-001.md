# ANG-WORK-LEARNING-SITUATED-ONLINE-FEEDBACK-V1-001

Status: complete — `NOT_SUPPORTED`

Assurance: bounded synthetic/local GPU experiment; no promotion authority

## Question

Can Angler use externally verified Qwen outcomes to improve future situated
evidence selection without changing Qwen, replaying old examples, or replacing
the sealed situated reader?

## Mechanism

Add one constant-size fast residual policy over the sealed reader's candidate
scores. It consumes only detached query/candidate representations and trusted
Moving Origin coordinates. The base scores are row-normalized without changing
their order. A zero-initialized bounded residual starts with the exact same
argmax as the sealed reader. Each observed success reinforces the selected
candidate; each failure suppresses it through a policy-gradient loss. There is
no answer label, action lookup, task identity, replay buffer, or deterministic
solver input.

The implementation borrows the OML separation already evidenced in Angler:
retain a slow learned representation and adapt a small fast prediction policy.
It also applies the useful part of the ANML line by bounding where plasticity
can act, without claiming second-order ANML attribution.

## Required component evidence

Focused tests must establish zero-initialized selection equivalence, frozen
base-reader parameters, outcome-direction causality, bounded residuals,
constant state size, snapshot/reset, and retention on an unrelated query.

The first GPU stream must use fresh generated instances and the same frozen
Qwen in live, frozen, reset, and shuffled-outcome arms. Every expected mechanism
must recur under several surface variations. Record early/late Qwen accuracy,
selection accuracy, prior-family retention, state causality, parameter counts,
device/version data, hashes, and runtime. Thresholds and the complete stream
must be frozen before the first semantic update. Accept the first result
without tuning.

## Frozen first stream

The stream has four environment-specific software procedure families and four
procedure names. Target mappings exist only in the evaluator. Ninety-six
unique query variations are presented once: 24 alternating family-0/1 cases,
24 alternating family-2/3 cases, then 48 interleaved cases across all families.
No example or outcome is replayed. Thirty-two additional unique probe variants
are never update inputs.

Three byte-identical initial fast policies run against the same frozen Qwen and
sealed reader: live true-outcome updates, no-update frozen, and one-step-lagged
wrong-family outcome updates. Rank is `32`, residual bound `4.0`, AdamW learning
rate `0.003`, zero weight decay, gradient clip `2.0`, and one update follows
each generated response. The exact Qwen response-first contract and parser use
the four procedure words `trace`, `stage`, `invert`, and `seal`.

`ONLINE_SITUATED_FEEDBACK_SUPPORTED` requires all of:

- final 48-row live Qwen accuracy at least `0.65`;
- live final Qwen accuracy at least `0.20` above frozen and shuffled-feedback;
- live terminal held-out probe selection at least `0.65` and at least `0.20`
  above frozen;
- reset removes at least `0.20` of terminal live probe accuracy;
- family-0/1 terminal probe accuracy is no more than `0.15` below its post-stage
  A value;
- Qwen parameters, sealed-reader parameters, stream identities, state size,
  and all finite/bounded checks remain exact.

## Nonclaims

A pass is evidence for bounded outcome-driven situated-selection adaptation.
It is not unrestricted cross-domain reasoning, persistent lifelong learning,
general software repair, AGI, consciousness, or a normal LEARNING gate.

## First result

The first identity completed once without tuning. Result SHA-256:
`6513AB17B21BC9CCBED4C77ECF3992BE1AF436906EA8E7AE2D69F461E69D53F9`.

The final 48-row live Qwen accuracy was `0.2083`, below frozen and shuffled
controls at `0.2292`. Terminal probe selection was `0.1875` live versus
`0.15625` frozen/reset; reset removed only `0.03125`. The fast state changed
and Qwen output exactly tracked the selected procedure, while Qwen and reader
digests remained exact. Thus the failure lies in transferable selection credit,
not the Qwen publication interface.

The identity is consumed. Do not tune or rerun it. The next action is an
artifact-bound reconstruction diagnostic using its recorded outcomes only, to
separate insufficient update magnitude, base-score domination, representation
misalignment, and cross-family interference before choosing a successor.

The exact reconstruction diagnostic completed at SHA-256
`195B9AE5371D174D91F011F6018E39EF1F4F8EC729BEE00E2707E13F1FFBDBC`.
Mean gradient norm was `0.3454` with zero clipping; mean maximum residual was
`0.5295` against mean normalized base margin `0.5804`, and terminal residual
reached `1.3986`. It reproduced the terminal fast-state digest exactly. This
rules out simply increasing gradient/residual scale and supports replacing the
all-plastic tiny adapter with the scaled slow-representation/plastic-memory
architecture.

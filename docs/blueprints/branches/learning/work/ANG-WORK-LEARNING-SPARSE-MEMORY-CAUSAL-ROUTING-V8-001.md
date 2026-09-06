# ANG-WORK-LEARNING-SPARSE-MEMORY-CAUSAL-ROUTING-V8-001

Status: development complete; `DEVELOPMENT_NOT_SUPPORTED`; final sealed

Active node: `ANG-BP-LEARNING` / sparse memory-dependent causal routing

Assurance: LOW; local synthetic software episodes, frozen foundation, no
deployment or external effects

## Evidence-driven need

V7 is preserved `DEVELOPMENT_NOT_SUPPORTED`, result SHA-256
`DE35BE6F857D435C03E5283A338913D0EF712B09666537521318CC52188D81E8`.
It improved full development success to 4/8 and retained bounded state, but all
neural controls also scored 4/8. True-minus-deranged target probability was
0.00074 and true-minus-zeroed was -0.0464. Writes used 15.96 effective slots;
challenge reads used roughly 13–16 effective slots. The state was not collapsed
to one slot, but routing was nearly uniform and therefore did not specialize.
The V7 score network also had a direct semantic-only path, allowing learned
residuals with empty memory; memory-disabled equaled full.

## Accountable outcome and borrowed mechanism

Use generic sparse top-2 routing borrowed from sparse Mixture-of-Experts
systems for both content reads and writes, while retaining the bounded
NTM/DNC-style state. Remove the semantic-only residual bypass: the learned
residual must be a bias-free function of read memory and semantic×memory
interaction, and must be exactly zero for an empty memory. Frozen Qwen semantic
similarity remains the transparent base score.

Top-2 routing is architecture, not a procedure solver: slot indices are chosen
only from learned content/allocation logits shared across every task. No
outcome sign, family identity, target, procedure rule, or answer enters routing.

## Scope and frozen protocol

Fresh scope is this leaf, one sparse V8 core and focused test, reasoning
exports, one thin V8 runner and focused test, append-only `AGENTS_SYNC.md`, and
fresh remote V8 checkpoint/results. V7 and earlier artifacts, Qwen, V6 final,
public remote, and owner workspaces remain immutable.

Reuse the seal-safe V6 64/8/8 corpus, safe encoder, exact 256-pair/64-step
schedule, stop-gradient causal references, loss weights, state/removal/swap
metrics, and V7/V6 acceptance thresholds. Initialize only the outcome-blind
query/candidate/temporal/semantic projections from V7 byte-exactly; sparse
routing, outcome, memory writer/reader, and memory-dependent score start fresh.

Additional pre-run invariants:

- each read/content-write/allocation distribution has at most two nonzero
  slots and sums to one before write strength;
- empty-memory residuals are exactly zero for all candidates;
- non-empty memory permits gradients through selected routing weights, keys,
  values, outcome embeddings, writer, and scorer;
- aggregate writes across a development stream still use at least four
  effective slots and no slot exceeds 0.50 aggregate mass.

First valid development/final results are accepted without seed retry,
threshold change, temperature tuning, top-k change, or post-result rescue.
Passing is bounded synthetic outcome-causal memory evidence, not arbitrary
reasoning, AGI, consciousness, deployment safety, or production readiness.

## Stop and rollback

Stop on leakage, non-finite state/gradient, bound/sparsity/empty-memory
violation, final access before gate pass, result reuse, foundation mutation,
deterministic solution logic, or evidence failure. Preserve terminal results;
roll back only fresh V8 artifacts.

## Preserved first result

The single frozen development run completed in 62.50 seconds. Result SHA-256 is
`3B91E66B5C7EC168F86F98F0501004D5A369608F24D10F2EDD8694D262F11D8C`;
checkpoint SHA-256 is
`196BF0EC74D716442882742C7282342265D1420131431E0FD97A14799BBCE2F0`.
Full solved 3/8 and every removal/control arm also solved 3/8. Writes were
sparse and bounded, but aggregate effective write use was 3.9637, below the
frozen four-slot floor. True outcomes beat zeroed outcomes by 0.05415 mean
target probability, while true outcomes beat shuffled outcomes by only
0.000586. The final partition was not opened.

A read-only checkpoint diagnostic found that true and shuffled histories had
mean memory-key cosine 0.99595, mean memory-value cosine 0.99179, and identical
top-2 challenge slot identities on all eight mechanisms. Outcome embeddings
remained distinct. The failure is therefore not absence of outcome input:
event-specific feedback is being compressed into almost the same shared
memory state, destroying the binding between a procedure trace and its result.

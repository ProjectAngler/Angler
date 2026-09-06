# ANG-WORK-RUNTIME-PLASTIC-WORKING-SET-V1-001

Status: active implementation

Active node: `ANG-BP-RUNTIME`

Assurance: local learned-state research; no external effects

## Outcome

Replace the single indefinitely frozen Jenny adapter assumption with one
bounded, expandable plastic competence lineage.  Learned low-rank increments
may accumulate in an immutable RAM/NVMe archive while one uniformly active
working-set snapshot remains within a 1 GiB total adapter budget.

## Architectural boundary

- Qwen foundation weights remain frozen.
- Cognee stores exact facts and episodes; it is not the learned behavior state.
- Moving Origin supplies temporal evidence; it does not assign temporal meaning.
- Expert relevance and expected contribution are learned from representation
  and observed outcome evidence.  Topic names, keywords, emotions, and fixed
  task tables are prohibited router inputs.
- Deterministic code may validate identities, account bytes, solve placement,
  and atomically load/rollback a snapshot.  It may not decide what an expert
  means or whether a memory is current.
- Working-set changes happen only at explicit consolidation boundaries.  The
  same snapshot is used uniformly between boundaries; there is no per-prompt
  adapter/personality switching.

## Donor basis

The first continual-learning candidate adapts the replay-free orthogonal
low-rank principle from O-LoRA (Wang et al., EMNLP Findings 2023; MIT-licensed
reference implementation).  Angler's existing exact rank-concatenation and
content-addressed binding machinery remains the artifact path.  Donor results
are hypotheses to test, not inherited evidence.

## First vertical slice

1. Implement a learned working-set coordinator whose scores predict the
   contribution of archived plastic increments from numeric representations.
2. Enforce the 1 GiB active ceiling independently of those learned scores.
   The envelope covers all GPU ranks together and subtracts measured SGLang
   pool, routing, cross-GPU communication, and atomic-staging reservations
   before computing usable adapter capacity.
3. Preserve mandatory shared-trunk components and emit one content-addressed,
   auditable working-set decision.
4. Train the first new increment on temporal-semantic revision: support,
   contradiction, supersession, closure, reopening, and continuing relevance.
   Counterexamples must reject newest-wins, age-decay, and keyword shortcuts.
5. Evaluate fresh multi-turn trajectories, old-capability retention, component
   removal, and state swap before any live activation.

## Initial acceptance

- Coordinator loss decreases on held-out numeric routing observations and its
  chosen working set improves held-out contribution under the same byte budget.
- Every plan is at most 1,073,741,824 bytes and retains all mandatory shared
  components or fails without a plan.
- Adapter bytes plus measured backend, routing, communication, and staging
  reserves fit the same envelope; file size alone is never treated as VRAM.
- No semantic string is accepted by the learned router.
- Identical learned state and numeric evidence reproduce the same plan.
- Removing learned coordinator state removes the measured routing advantage.
- No live adapter is replaced until the temporal-semantic increment also passes
  retention, removal, and exact rollback checks.

## Stop conditions

Stop before activation if the adapter archive, current binding, budget, or
SGLang hot-load identity differs; if a new increment requires deterministic
semantic routing; if retained abilities regress; or if an atomic rollback
cannot restore the current content-addressed adapter exactly.

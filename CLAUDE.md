# CLAUDE.md — Jenny 2.0 session rules (Claude Code)

AGENTS.md is the authority. Read it completely before any change; this
file supplements it and never overrides it. Then read the latest
AGENTS_SYNC.md entries for active claims.

## Active program

JENNY_EMOTION_AGENCY_INTEGRATION_PLAN_V1.md is the live emotion/agency
roadmap. Current milestone: leaf E0 (substrate atlas, merged with the
takeover doc's suggested read-only audit), then leaf E1 (causal-work
evaluation), which gates all container/label/level/coupling work.

SUPERSEDED: JENNY_INTEROCEPTION_SPEC.md Phase 1 ("build the interoception
logger"). `src/angler/runtime/grounded_appraisal.py` IS the substrate.
Never build a parallel logger, second substrate, second procedural model,
or adapter-swap daemon.

## Determinism law (absolute)

Deterministic code is allowed for identity, provenance, schemas, bounds,
permissions, transactions, crash recovery, orchestration, and measurement.
It must not encode task answers, topic-specific behavior, synthetic
emotions, or hidden decision policy. Measurement code measures — never
synthesizes or smooths. New behavior arises from model-authored choices
conditioned on observed evidence and persistent learned state.

## Epistemic rules

Emotion labels are revisable hypotheses with alternative explanations,
never trusted measurements. Retrieval alone is not learning. Unevaluated
conversation earns no credited competence. No claims of consciousness,
feelings, or personhood — functional evidence only, uncertainty preserved,
in code comments and diary entries alike.

## Standing hazards

- Git is the largest data risk: substantial valuable uncheckpointed work
  exists. Never run `git clean`, destructive reset, or broad
  checkout/restore. Untracked does not mean disposable.
- Never expose or browser-connect `127.0.0.1:30000` (raw model API).
- Never print, copy, or commit credentials; read the API token only at
  execution time per the takeover doc.
- Never mutate canonical stores with ad hoc SQL. Cognee projections are
  rebuildable; canonical history is not.
- No additional foreground model calls; no discarded-response cascades.

## Change protocol (per AGENTS.md — reminder only)

Claim in AGENTS_SYNC.md naming exact files → diagnose with existing
traces/tests → apply_patch edits → focused tests → complete runtime suite
for cross-boundary changes → `git diff --check` → restart only with
`pending_operation: false`, recording state ref and ordinal before and
after → append exact results, limitations, and next action.

# ANG-WORK-RUNTIME-GROUNDED-APPRAISAL-DYNAMICS-V1-001

Status: implemented and live; runtime validation passed

Active node: `ANG-BP-RUNTIME`

Outcome: give Jenny a bounded plastic appraisal substrate whose state is
learned from her own grounded event history rather than assigned emotion
labels or scripted reactions.

Each closed event contributes only mechanically available signals: model
uncertainty, alternatives, retrieval distances, commitment state, receipt
closure, observation/source presence, and exact declared consequence
dimensions. Exact online moments establish each signal's learned baseline;
pairwise online covariance establishes which signals actually move together.
The latest deviations and strongest supported relations become compact
advisory evidence at existing model decisions.

This leaf adds no emotion vocabulary, valence/arousal formula, scalar utility,
action rule, extra inference, replay, adapter operation, gradient step, or GPU
memory. It does not claim subjective feeling. Its learned state may inform
model judgment but cannot itself force behavior.

Acceptance: focused tests prove deterministic online updates, exact moments,
data-derived cross-signal relations, provenance retention, bounded context,
absence of emotion/utility labels, and safe handling of unevaluated turns. The
existing runtime suite must remain green before live activation.

Result (2026-09-04):

- Focused and affected runtime set: 89/89 passed.
- Complete runtime unit suite: 469/469 passed in 52.504 seconds.
- Read-only replay over all 97 existing live episodes: zero failures; 22
  observed signal dimensions, 213 co-observed relations, a 40,996-character
  learned state, and a 2,003-character model-facing context.
- Replay cost was 40.158 ms total, averaging 0.414 ms per episode.
- `git diff --check` passed. The live service restarted onto the new source and
  reports API, model runtime, life loop, scheduler, and projection healthy.

Interpretation: Jenny now carries a continuously changing empirical appraisal
graph. It learns what signals deviate from her own history and which signals
co-vary, then presents only a bounded advisory view to her existing decisions.
This supplies grounded plastic evidence for later semantic/emotional learning;
it neither assigns an emotion nor establishes subjective feeling.

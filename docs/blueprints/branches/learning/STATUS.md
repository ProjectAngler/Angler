# LEARNING status

Freshness: 2026-08-27
Design: draft  
Delivery: not_started  
Current gate: `ANG-GATE-LEARNING-DESIGN-001`

Completed: Tier-1 learning sequence, child map, boundaries, and update contract drafted.  
Blocker: input contracts and first scientific gate are not approved.  
Next: expand FEEDBACK-UPDATE only; keep meta-learning, consolidation, and lifelong features deferred.  
Evidence: none.  
Rollback: remove branch package; Tier-0 blueprint remains unchanged.

Experimental successor: the failed prequential SRWM candidate is preserved,
and `ANG-WORK-LEARNING-BIDIRECTIONAL-PROCEDURE-001` is complete.  Its learned
forward/backward dynamics produced independently executed procedures on 20/20
unique held-out cases.  Equal-budget forward-only also reached 20/20 but used
77.2 mean expansions versus 41.4 bidirectionally; causal controls were
untrained 0/20, corrupted-backward 11/20, and permuted learned guidance 7/20.
Frozen Qwen proposed correct goals on 7/8 semantic cases and Angler reached
every proposed goal 8/8.  This is bounded experimental evidence, not a normal
LEARNING gate, an evolving learner, or a milestone pass.

The causal-operator successor now has a clean oracle-free v5 result: 114/120
on one untouched three-domain partition, with 55/60 four-step compositions and
zero proposal-time world execution.  Selection-only Boxes adaptation improved
18/40 to 33/40 while prior Tokens/Files scores were unchanged; Files adaptation
changed the permitted parameters but did not improve its fixed adaptation set.
The next experimental leaf is skill-local procedural memory, learned routing,
and consolidation without replaying all past abilities.  Normal LEARNING and
milestone gates remain unchanged.

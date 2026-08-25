# RESOURCE-PROBES status

Freshness: 2026-08-25  
Design: approved_for_cr0  
Delivery: deferred_to_successor_leaf  
Gate: `ANG-GATE-RESOURCE-PROBES-001`  
Blocker: every real probe needs a successor assessment and exact command/resource leaf.  
Next: after CR0 inventory/plan delivery, authorize a separate synthetic ProbeSpec/result schema leaf; actual probing remains separately gated.  
Rollback: retain prior inventory; ADR-0002 archive for design files.

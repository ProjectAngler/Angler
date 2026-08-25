# PERMISSIONS status

Freshness: 2026-08-25  
Revision: 1  
Design: approved_for_cr0  
Delivery: documented  
Current gate: `ANG-GATE-PERMISSIONS-001`

Completed: default-deny capability matrix, canonical path boundary, command classes, synthetic-fixture rule, resource ceilings, stop conditions, and rollback.  
Blocked outside CR0: OS-enforced sandboxing, learner/model execution, dependencies, network, real-person data, deployment, and external effects.  
Next: run the CR0 safety validator and attach a scoped bootstrap assessment to each ready leaf.  
Rollback: revoke ADR-0002 and the CR0 assessment; restore only under human direction.


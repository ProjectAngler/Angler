# Inventory design gate

- Gate: `ANG-GATE-INVENTORY-001@1`
- Pass: all required categories support source/unit/freshness/uncertainty/visibility/authority; missing/unknown, stale, topology-change, permission-below-capacity, and redaction fixtures behave as specified.
- Failure: required unknowns make dependent plans infeasible; never infer permission.
- Evidence: schema/fixture test receipt and content identities.
- Human impact: local synthetic bootstrap only.

# Resource-probe design gate

- Gate: `ANG-GATE-RESOURCE-PROBES-001@1`
- Pass: schema freezes bounds/authority/cleanup; synthetic timeout, overrun, abort, partial write, and cleanup failures cannot expand inventory; retry is idempotent.
- Evidence: schema/fixture test receipt.
- Failure: preserve prior inventory and mark result unusable.
- Human impact: CR0 never executes a real probe.

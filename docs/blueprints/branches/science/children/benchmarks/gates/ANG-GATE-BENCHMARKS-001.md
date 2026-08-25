# Benchmark design gate

- Gate: `ANG-GATE-BENCHMARKS-001@1`
- Pass: both family specifications name a procedure, grammar, transformations, outcome-only verifier, budget class, leakage hypotheses, version, and partition compatibility; synthetic tests include known-valid, known-invalid, transformed, and leaky controls.
- Fail: a surface cue predicts the answer/procedure, a verifier requires a preferred method, or a transformed instance changes the latent procedure.
- Evidence: schema-validation and fixture-test receipt, content identities, reviewer.
- Rollback: retire the family version and invalidate dependent suites.
- Human impact: `ANG-POL-LOCAL-SCAFFOLD-001`; bootstrap evidence is not a flourishing-gate or milestone pass.

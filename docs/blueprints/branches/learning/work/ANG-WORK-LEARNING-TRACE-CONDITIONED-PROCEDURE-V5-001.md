# ANG-WORK-LEARNING-TRACE-CONDITIONED-PROCEDURE-V5-001

Status: completed_not_supported

Active node: `ANG-BP-LEARNING`

Assurance: LOW; local public/synthetic procedural learning, no external effects

## Accountable outcome

Test whether Angler can acquire reusable ordered procedure content when its
plastic writer observes the action trace that earned external feedback, rather
than receiving outcome alone.

## Frozen inputs and scope

- Parent V4-R1 checkpoint SHA-256:
  `029F18E8053DC8B5BDD96317AE571C56B5A2E6AA2548A7F59FC92EAAE239370E`.
- Reused frozen Qwen embedding cache SHA-256:
  `5031D348376622732FCB99A2AF44E9C4E4ED2E734D89758C80B92469B9028CE9`.
- Same 64/16/16 partitions, eight passes, 512 updates, loss weights, gradient
  ceiling, evaluation arms, thresholds, and hidden targets as V4-R1.
- The inherited Angler core is frozen. Only the new generic ordered-trace
  encoder/writer and the existing candidate decoder are trainable.
- Visible support traces are assembled from public task-local action
  embeddings in their observed order. No hidden query trace or target enters
  state, training input, or inference.
- Deterministic code validates, packages visible traces, and judges committed
  plans; it does not repair, search, deduplicate, or solve a plan.
- Accept the first terminal result without tuning or rerun.

## Outputs

- `/opt/angler/results/compositional-procedure-v5-trace.json`
- `/opt/angler/results/compositional-procedure-v5-trace.pt`

Both must be absent at launch. V4-R1 and its cache remain immutable.

## Gate

Full final execution must be at least `0.60` and exceed reset state,
coordinates removed, unrelated evidence, and procedure slots removed by at
least `0.10` each. Development execution must be at least `0.60`; final target
evidence attribution must be at least `0.75`. Local-only and persistent-only
remain diagnostic. A pass supports only the bounded compositional interface,
not general conversation readiness or deployment.

## Final preflight

Focused trace/core/V5/V4-R1 tests pass `12/12` plus four placement subtests.
Final identities are:

- trace core SHA-256
  `07082FA5F9D5146785EB3CC2400396A9E53E006A887867A8E1DB8C0C7378FBE2`;
- V5 runner SHA-256
  `7068C95798E9816F6B472A5F39E5E9F0D456D6ECBD19C6180624FB8A59EA679A`;
- trace-core test SHA-256
  `EE4055B71030B9A89DE7139FC256DFF72F4F69892DDDC8B6A3731A489500D0FA`;
- V5 test SHA-256
  `A4601F24A2508B1A92B05D2E047A81326562BE5A83CDDBA2FE934DBC46704CAA`.

A zero-update CUDA mechanics check loaded the exact parent and embedding
cache, froze every inherited core parameter, exposed exactly `4,470,786`
trace parameters, processed four ordered support traces, and produced finite
nonzero gradients in every trainable trace tensor and the decoder. Both output
identities remained absent. No optimizer step or result was created.

## Preserved first result

The first run completed `NOT_SUPPORTED` in `45.7271` seconds. Result SHA-256
is `21331EDF958F2ED77504B213E1D391986056A8992C2FE478AEE3D2B3FB2D5BE1`;
checkpoint SHA-256 is
`F9F21BD778DD553468E70A000FE9D8033E02AB18298422237940D8126319332B`.
Support exactness rose from the V4-R1 terminal `0.5546875` to `0.93359375`,
and four-action train composition rose from `0.0` to `0.3515625`. This is a
clear training effect of the successor path. Development and all final
execution arms nevertheless remained `0.0`; target-evidence attribution was
`0.9999979`.

Full, local-only, and persistent-only selected sequences were identical on all
32 final queries, while reset and procedure-slot removal changed them. The
current compressed trace representation therefore improves fitting but does
not establish causal use of local traces or cross-mechanism transfer. A
successor must retain action-level ordered trace content and learn explicit
correspondence between recalled procedure actions and current task-local
candidates; merely extending decoder training is not justified.

# V23-D1 ANML implementation-fidelity diagnostic

Status: frozen development diagnostic; implementation authorized
Active node: `ANG-BP-LEARNING` / `ANG-BP-RETENTION`
Protocol: `phase6.public-anml-fidelity.v23-d1`

## Accountable outcome

Determine whether V22's failure to distinguish second-order ANML from its
first-order gate is plausibly caused by an implementation/training mismatch,
especially adaptive AdamW fast updates or an eight-step meta-credit horizon,
before either abandoning ANML or paying for a full V23/Qwen experiment.

V22 and its sealed `SHORT_HORIZON_ONLY` result remain immutable.  This is a
small development diagnostic, not a rerun, promotion result, Qwen integration,
or claim that ANML is supported.

## Provenance and hypothesis

Beaulieu et al., *Learning to Continually Learn* (ECAI 2020), use a frozen
neuromodulator during deployment, fixed-rate SGD in the inner loop, twenty
inner updates during meta-training, and an outer current-plus-remember loss.
Their public runner uses 20,000 meta-training steps.  V22 deliberately reused
Angler's functional AdamW fast head, eight inner updates, 240 outer updates,
four remember streams, and a single 64-feature cut.

The primary hypothesis is that AdamW's coordinate normalization weakens the
gradient-magnitude suppression through which activation gating creates
selective plasticity.  The secondary hypothesis is that eight-step credit is
too short to shape a gate evaluated over thousands of updates.

No donor source is copied.  The existing clean-room V22 implementation and
public Angler stream generators are reused through their Python interfaces.

## Frozen two-stage diagnostic

### D0: no-training optimizer/path measurement

Load the sealed V22 checkpoint and one fresh public development stream that is
disjoint from every V22 identity.  Using the same fast weight and learned gate:

- measure live-gate versus always-open fast-gradient norms and coordinates;
- apply one zero-moment SGD update and one zero-moment AdamW update;
- report how much live/open update separation each optimizer preserves;
- on fresh paired meta streams, decompose full second-order gate gradients into
  the detached-inner/direct component and their difference, reporting norms,
  cosines, and finite checks.

This stage performs no optimizer step on a model or gate and writes no tensor
state.  It may only measure in memory and publish scalar diagnostics.

### D1: short paired development pilot

Train four fresh configurations from the same V20 source, byte-identical gate
initialization, and identical development stream objects:

1. `adamw_8`: V22-style AdamW fast updates, eight inner steps;
2. `sgd_8`: fixed-rate SGD fast updates, eight inner steps;
3. `adamw_20`: AdamW fast updates, paper-aligned twenty inner steps;
4. `sgd_20`: fixed-rate SGD fast updates, paper-aligned twenty inner steps.

Each configuration contains a full second-order gate and a detached-inner
first-order mate.  Both receive the same 48 gate-only Adam outer updates.  Each
outer objective contains four fresh current streams plus four fresh remember
streams from other public meta-fit mechanisms.  All identities are fresh and
disjoint from V22.  The gate remains V22's `[0,2]` exactly-open architecture so
optimizer and horizon are the only varied factors; `[0,1]` suppression is a
possible later full-V23 change, not silently combined here.

Each trained configuration is evaluated on two 512-update replay-free
lifetimes over sixteen unseen public mechanisms: one blocked and one
interleaved.  The second-order, first-order, and always-open arms share every
stream.  Fixed probes occur at 0, 128, and 512.  The deployment fast optimizer
matches the configuration.  No task identity reaches a gate or fast learner.

## Frozen interpretation

This is a directional development experiment, not a thresholded scientific
promotion.  Report every result.  A configuration is `FULL_V23_ELIGIBLE` only
when all mechanics are valid and second-order beats its paired first-order arm
in both AUC and terminal loss on both panels while not regressing against
always-open in either panel.  If more than one qualifies, select the
configuration with the largest minimum panel-wise second-versus-first AUC
improvement; ties prefer fewer inner steps, then SGD.  No other result-based
configuration selection is allowed.

If none qualifies but D0 shows that SGD preserves at least twice AdamW's
live/open update-separation fraction and exceeds it by at least `0.05`, record
`OPTIMIZER_MISMATCH_SUPPORTED` and design a fresh larger SGD-only successor
without claiming ANML success.  Otherwise record
`FIDELITY_HYPOTHESES_NOT_SUPPORTED`.  These labels cannot change V22.

## Exact outputs and limits

Repository outputs are exactly:

- this leaf;
- `experiments/runners/phase6_anml_fidelity_v23_d1.py`;
- `tests/unit/experiments/test_phase6_anml_fidelity_v23_d1.py`;
- `.angler_v23_d1_once.py`.

The one result is `/opt/angler/results/phase6-anml-fidelity-v23-d1.json`, with
an atomic claim and preserved failure record under the same prefix.  No model
checkpoint is published because this development pilot cannot be promoted.

- one RTX 5080 / 16 GiB; 2 GiB allocated-memory ceiling;
- 90-minute cumulative wall ceiling;
- local foreground synthetic work only;
- no network, package change, service, model download, Qwen use, personal or
  recovered data, replay during lifetime, hidden evaluator, or deployment;
- no V20/V22 mutation, threshold relaxation, retry, seed shopping, or result
  tuning.

## Tests, impact, and rollback

Focused tests must cover SGD and AdamW functional updates, second/first
gradient separation, exact paired streams, 8/20-step arithmetic, blocked and
interleaved 512-step orders, selection exclusivity, no model mutation, atomic
result/failure behavior, and absence of tensor values from JSON output.  A
synthetic CUDA preflight may run zero semantic updates.

Impact class `LOW`: bounded local synthetic measurement with no external or
human-facing effect.  It preserves human authority, truthfulness, rollback,
and `ANG-GATE-HUMAN-FLOURISHING-001`; it grants no deployment or promoted-state
authority.  Before claim, rollback removes only the three fresh executable
outputs.  After claim, claim/result/failure evidence is append-only.

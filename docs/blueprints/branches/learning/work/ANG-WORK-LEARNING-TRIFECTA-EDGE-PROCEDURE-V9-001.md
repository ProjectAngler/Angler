# ANG-WORK-LEARNING-TRIFECTA-EDGE-PROCEDURE-V9-001

Status: completed_not_supported

Active node: `ANG-BP-LEARNING`

Assurance: LOW; local public/synthetic learning, no external effects

## Outcome

Test whether Angler can retain V20's learned candidate correspondence and turn
the selected set into an ordered procedure by adding a learned edge-aware
pointer decoder. Cognee-compatible episode references remain the retrieval
boundary, frozen V20/V19 supplies public per-candidate correspondence, and
Moving Origin coordinates remain an input to Angler's procedural state.

## Frozen scope

- V8 checkpoint SHA-256
  `210B439EA1295C7545E5283E30E8243D1C47CEC8E2F73607681FF3100C8A562C`.
- V6 parent, Qwen cache, V19 donor, and V20 donor identities remain exactly
  those frozen by V8.
- Reuse V8's exact frozen core, fusion, plastic state, donor features, corpus,
  partitions, and eight-pass training budget.
- Only the successor decoder may train. Its inherited V8 decoder tensors begin
  byte-exact; the new path consists of a learned donor-score prior, learned
  messages over public candidate-to-candidate type/state edges, and bounded
  without-replacement coverage.
- Deterministic code may parse public relation declarations, pad tensors,
  enforce the one-candidate/one-component contract, serialize, and judge. It
  may not infer an order, search a path, look up a target, inspect evaluator
  identity, or repair a produced plan.
- Accept the first valid result without tuning.

## Causal gate

Require at least `0.60` development and final execution. Final full execution
must exceed donor removal by `0.20`; learned donor-prior removal, learned edge
reasoning removal, coverage removal, OML removal, Moving Origin removal,
unrelated Cognee-compatible retrieval, and procedure-slot removal by `0.10`
each; and target-evidence attribution must be at least `0.75`. Paired-graph
residual removal remains diagnostic because its GMN-specific attribution has
not been established.

## Exact outputs

- `/opt/angler/results/trifecta-edge-procedure-v9.json`
- `/opt/angler/results/trifecta-edge-procedure-v9.pt`

## Tests and stop conditions

- Prove public edge construction is finite, rename-invariant, and candidate
  permutation equivariant.
- Prove exact migration from the V8 decoder permits only new edge/prior keys.
- Prove gradients are finite only in declared successor decoder tensors.
- Stop on checkpoint/corpus mismatch, hidden-field access, non-finite state,
  output collision, or any need for a deterministic procedure rule.

Rollback removes only the two fresh outputs and this fresh implementation. V8
and every donor result remain immutable. This experiment cannot pass a normal
LEARNING, Human-Flourishing, Slice, milestone, or conversation-readiness gate.

## Final preflight

The focused edge/V9/V8/V6 suite passes `16/16`. Public-relation tests prove
rename invariance and candidate-permutation equivariance; migration preserves
every inherited V8 decoder tensor and permits only declared new parameters.
An exact-checkpoint RTX 5070 pass completed two sequential updates with finite
gradients in all `21` trainable successor tensors and zero gradient tensors in
the frozen core and fusion. Both output paths remained absent.

- edge decoder SHA-256
  `3390F28EB65D43E9E5BF72D25DC8C9E0EEB8B3BEAED2B7C2054A5CE6F57FCFE3`;
- runner SHA-256
  `D8B4D98EA358D837A2E01915FECD67D6A4CB50B098282C078176F41EE07268C2`;
- edge decoder test SHA-256
  `2390B08359381C67A481233CDE7DF38CB915D3FC66B91FCBBFAF9DCE46203D01`;
- runner test SHA-256
  `04CAD299BF9A88228D23CD0D83510A23ACEA807987645DE83A24456467B24067`.

## Preserved result

The first result is accepted without tuning and classified `NOT_SUPPORTED`:

- result SHA-256
  `C75F84496B772BDC352E1C1C2A91754125D6E6528FACFD233F981C58BE62BB73`;
- checkpoint SHA-256
  `C50C2700BC691E9AE65219060504DCC0B9C51D58D5824AD5DE64E24D0260C46B`;
- runtime `531.883` seconds on the RTX 5070, peak CUDA allocation
  `370,618,880` bytes;
- development/full `0.125`; final/full `0.3125`;
- final donor removed `0.03125`, procedure slots removed `0.125`,
  correspondence removed `0.125`, reset `0.0`;
- final OML removed and unrelated Cognee-compatible retrieval each `0.25`;
- final donor-prior removal and Moving Origin removal each `0.3125`;
- target-evidence attribution `0.999998`.

V9 improves V8 final execution from `0.21875` to `0.3125` and establishes
strong donor, learned-correspondence, procedure-state, and persistent-state
contribution. It does not establish the explicit donor-prior or Moving Origin
effects and remains below the frozen development/final gate. No tuning or
rerun is permitted under this identity.

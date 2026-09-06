# ANG-WORK-LEARNING-TRIFECTA-CORRESPONDENCE-COMPOSITION-V8-001

Status: completed_not_supported

Active node: `ANG-BP-LEARNING`

Assurance: LOW; local public/synthetic learning, no external effects

## Outcome

Test the first genuinely cooperative Angler/Cognee/Moving-Origin composition:
Cognee-compatible episode references supply bounded candidate experiences;
Moving Origin supplies temporal self-location; frozen V20/V19 supplies learned
public candidate correspondence; and Angler's existing larger action-trace core
must learn to order the candidates into a procedure.

## Frozen scope

- V6 parent checkpoint SHA-256
  `08A966424CA678E9742A8D80B985BA48FC0C0FF8E3DF4BC86C0262B9C53FAD7A`.
- Qwen cache SHA-256
  `5031D348376622732FCB99A2AF44E9C4E4ED2E734D89758C80B92469B9028CE9`.
- V19 donor checkpoint SHA-256
  `10BB6BAC9BD83F7F4EE0ABF2846CE4133D2133790C2B55113C9044930D2EBC7F`.
- V20 donor checkpoint SHA-256
  `D49E4CAAB64A264A11C675B295A8C453AC4475F078311EB7283A4F9A8817EF48`.
- Frozen Qwen, Angler core, V20/V19 donors, corpus, partitions, and eight
  training passes. Train only one donor-feature fusion and the existing V6
  action-trace decoder.
- The donor receives canonical public tasks selected by episode references.
  It receives no hidden evaluator field, mechanism identity, target label,
  digest feature, search result, or deterministic ordering rule.
- Accept the first valid result without tuning.

## Causal gate

Require at least `0.60` development and final execution, final at least `0.20`
above donor removal, at least `0.10` above OML removal, Moving Origin removal,
unrelated Cognee-compatible episode retrieval, and procedure-slot removal, and
target-evidence attribution at least `0.75`. V19 paired-graph removal,
correspondence removal, reset, local-only, and persistent-only remain reported
diagnostics. A zero-residual gain without a mismatch-zero-specific gain is not
represented as proof of GMN-specific correspondence.

## Outputs

- `/opt/angler/results/trifecta-correspondence-composition-v8.json`
- `/opt/angler/results/trifecta-correspondence-composition-v8.pt`

## Final preflight

The focused fusion/V8/V6 suite passes `5/5` on the workstation. The exact
source identities are:

- fusion `877D2599197252E3AFCED36832888746349F9C57C30C38BCA6FA0AC02195B271`;
- runner `CB308B62052C5115684EFF7C1F4DF9F5D997505CF66AE40608B6769CB718E682`;
- fusion tests `BCAC3CD097969D2F8254035D6A61EA0FE0385590EC7C3799BDA1FE9068C11360`;
- runner tests `87FF629973BB1E53502B2FACFD247D3BF45E4BB1B58588EC4AD706093D17B761`.

An exact-checkpoint, zero-output CUDA pass built canonical episode-reference
features, processed one complete support/query episode, and backpropagated a
finite loss. All `12` fusion and `33` decoder gradient tensors were finite;
the frozen Angler core received zero gradient tensors and both V8 outputs
remained absent.

## Preserved result and recovery

The original run completed training and evaluation and wrote its checkpoint,
then failed to write the JSON because a local variable shadowed the output
path. Training was not repeated. Evaluation-only recovery R1 loaded the exact
checkpoint and wrote the result:

- checkpoint SHA-256
  `210B439EA1295C7545E5283E30E8243D1C47CEC8E2F73607681FF3100C8A562C`;
- recovered result SHA-256
  `E5198522656930F5F57B8FCA7A67DF53692E0F9EC59F9ECE5824F86428F22EDC`;
- development full execution `0.03125`;
- final full execution `0.21875`;
- donor removed `0.0`, correspondence removed `0.0625`, reset `0.0`;
- unrelated Cognee-compatible episodes `0.1875`;
- OML removed, paired residual removed, and Moving Origin removed each
  `0.21875`;
- target-evidence attribution `0.999998`.

V8 therefore establishes real donor, learned-correspondence, and persistent
state contribution but does not establish OML-, V19-residual-, Moving-Origin-,
or full-trifecta attribution. A diagnostic found only `21.875%` of final rows
preserved the donor's correct target set, despite the frozen donor itself
recovering that set on `90.625%`. The learned projection diluted candidate-set
credit and the decoder lacks explicit cross-candidate dependency edges and
without-replacement coverage. The narrow successor is a learned edge-aware
pointer decoder over the same frozen features, not a deterministic planner.

## Post-V9 interpretation correction

A read-only evaluator-side decomposition after V9 showed that the earlier
`0.90625` V20 “target set” statistic described its supported
evidence/correspondence candidate boundary, not exact action-set recovery.
For actual final action candidates, top-four selection from the isolated
paired-graph scalar is `0.50`; the complete frozen V20 rollout repeats actions
on every row and recovers no exact four-action set. Therefore V8's donor
features remain a useful learned representation, but V20 must not be described
as a `0.90625` action selector. This correction changes no V8 result bytes.

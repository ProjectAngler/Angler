# ANG-WORK-LEARNING-OML-PRODUCTION-BRIDGE-V21B-001

Status: completed_not_supported

Active node: `ANG-BP-LEARNING`

Assurance: LOW; local public/synthetic procedural learning, no external effects

## Outcome

Test the missing behavioral integration directly: whether the preserved V20
OML relation representation and V19 paired whole-graph comparator can drive
Angler's already-existing transition, action, backward-reasoning, and STOP
path after that narrow production bridge is trained.

## Frozen scope

- V19 checkpoint SHA-256
  `10BB6BAC9BD83F7F4EE0ABF2846CE4133D2133790C2B55113C9044930D2EBC7F`.
- V20 checkpoint SHA-256
  `D49E4CAAB64A264A11C675B295A8C453AC4475F078311EB7283A4F9A8817EF48`.
- Select V20's second-order OML arm before result access.
- Freeze every V20 RLN/PLN tensor, every V19 paired-graph tensor, all context
  encoders, and all auxiliary competence/checkpoint state.
- Train the exact `212`-tensor, `158,674`-parameter inherited production path
  for one chronological pass over all `64` train-partition mechanisms using
  public leave-one-package-out support
  traces. Query targets and hidden evaluator fields are never used.
- Accept the first valid result without tuning.

## Causal controls and gate

Evaluate the same trained bridge with its V19 paired-graph residual removed,
with OML removed but the bridge weights copied exactly, with role memory
removed, and with backward reasoning removed. Also record the untouched
pre-bridge controller.

Support requires development and final accuracy at least `0.50`, final at
least `0.25` above its own pre-bridge result, and final at least `0.10` above
both paired-graph-removed and OML-removed controls. A failure is preserved and
cannot be repaired by changing these thresholds or rerunning the identity.

This leaf isolates representation-to-action integration. Cognee candidate
recall and Moving Origin temporal self-location remain independently tested
runtime components and are not falsely credited by this component result.

## Frozen first result

The first valid result is preserved `OML_PRODUCTION_BRIDGE_NOT_SUPPORTED`:

- result SHA-256
  `F727BA4087BD885EC38852A2A5463E335484B2E73C4EB637619B4AFC513B5484`;
- checkpoint SHA-256
  `86B14BF563B7FC4CC2FB8F69B394632104D77D343B5A400F19F1C4FDEA658C7A`;
- one `64`-mechanism pass completed in `55.419` seconds;
- loss fell from `0.932603` to `0.292541` (`68.63%`);
- all donor tensors remained exact;
- development and final execution both remained `0/16`.

The result confirms that the inherited production state machine cannot compose
the donor's useful correspondence into the required four-action behavior. A
separate read-only diagnostic on the current 32-query development and final
corpora found the frozen V20 donor identifies the exact four-action target set
on `90.625%` of both partitions and places a target first on `100%`; V19 alone
reaches `50%` development and `37.5%` final. The useful representation exists,
but this bridge cannot order it. The successor must feed V20's paired public
candidate features into Angler's larger action-trace procedural core, retaining
Cognee episode recall and Moving Origin coordinates as causal inputs.

## Outputs

- `/opt/angler/results/oml-production-bridge-v21b.json`
- `/opt/angler/results/oml-production-bridge-v21b.pt`

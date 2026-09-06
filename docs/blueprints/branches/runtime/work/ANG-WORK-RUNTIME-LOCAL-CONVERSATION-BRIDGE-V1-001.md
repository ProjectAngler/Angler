# ANG-WORK-RUNTIME-LOCAL-CONVERSATION-BRIDGE-V1-001

Status: implementation complete; no readiness or gate claim

Active node: `ANG-BP-RUNTIME` / bounded local conversation prototype

Assurance: LOW; local synthetic/unit-test scaffold, no promotion or deployment authority

## Accountable outcome

Create a checkpoint-agnostic local bridge that can load a caller-supplied,
supported `ScalableProceduralCore` plus `CandidateProcedureDecoder` checkpoint,
join existing `SituatedMemory` recall with Moving Origin coordinates, decode an
explicit sequence over caller-supplied task-local action descriptions, ask one
already-loaded frozen `LocalQwenIO` to publish a response from that visible
plan, and package only externally supplied feedback.

## Inputs and preconditions

- The checkpoint is local, is supplied explicitly by the caller, and contains
  core/decoder configuration and state dictionaries plus one plastic state.
- `SituatedMemory` remains the validated boundary around canonical evidence,
  disposable Cognee candidates, and `MovingOriginIndex` coordinates.
- `LocalQwenIO` is already loaded and frozen; this leaf neither chooses nor
  downloads a foundation model.
- Candidate actions are non-empty, task-local descriptions supplied on each
  turn. They are never loaded from a global action table or inferred from task
  identity.

## Deliverables and write scope

- `src/angler/runtime/conversation_bridge.py`
- `tests/unit/runtime/test_conversation_bridge.py`
- this leaf, updated at handoff with the focused test result

No other runtime, reasoning, memory, experiment, installer, or test file is in
write scope. In particular, do not edit `compositional_procedure_v4.py` or its
tests.

## Execution constraints and non-goals

- CPU-only local unit tests with synthetic tensors and injected memory/Qwen
  doubles; no GPU, live Cognee, live model, network, package installation,
  external effect, or real-person data.
- No optimizer, checkpoint mutation, competence promotion, adapter routing,
  deterministic task solver, hidden answer table, response judge, autonomous
  feedback invention, or remote push.
- The bridge must fail closed on malformed/incompatible checkpoints, empty
  recall, candidate overflow, invalid decoder indices, empty generation, and
  foundation/checkpoint identity mismatch.
- Support is conditional on an explicitly supplied compatible checkpoint. This
  leaf cannot claim conversation readiness, a normal RUNTIME/LEARNING gate,
  Human-Flourishing gate, slice, milestone, or scientific result.

## Proportionate human-impact assessment

The assessed scope is an isolated local code-and-unit-test scaffold using only
synthetic data and caller-injected doubles. It does not contact people, deploy,
choose permissions, act on a generated plan, judge human outcomes, or write a
promoted state. Foreseeable harms are misleading capability claims, accidental
use of hidden task logic, untrusted checkpoint deserialization, and feedback
fabrication. Mitigations are explicit nonclaims, `weights_only` checkpoint
loading, task-local descriptions only, transparent plan/provenance output,
caller-owned feedback, bounded inputs, and fail-closed validation. Rollback is
deletion of the three scoped files. Impact class: `LOW`. Disposition for this
exact local construction and test scope: `ALLOW`; any live model/Cognee use,
real-person data, deployment, state update, tool execution, or scope expansion
requires a successor assessment. This assessment maps to
`ANG-GATE-HUMAN-FLOURISHING-001` but does not pass or waive that gate.

## Tests and acceptance gate

The focused unit suite must prove:

1. a supported synthetic checkpoint round-trips into frozen core/decoder
   instances and its plastic-state identity is preserved;
2. one turn consumes validated recall/Moving Origin coordinates and task-local
   descriptions, exposes evidence attribution and an explicit bounded plan,
   and sends that exact plan to the injected frozen-Qwen boundary;
3. STOP truncates the published plan and no absent/padded candidate is emitted;
4. externally supplied feedback is packaged without judging or inventing it;
5. incompatible foundation identity, malformed checkpoint, empty recall,
   candidate overflow, and empty generation fail closed.

Acceptance is a clean focused CPU test run and diff check. Passing establishes
only the local bridge contract under injected doubles.

## Failure, rollback, and handoff

On any failed check, preserve the prior repository state, report the failure,
and do not broaden scope. Rollback removes only the three files listed above.
At completion, record exact files, commands, results, nonclaims, and next
required live-integration prerequisite here and report them to the parent.

## Handoff result — 2026-08-30

Implemented only the declared module and focused test file. The bridge loads
caller-selected checkpoints with `torch.load(..., weights_only=True)`, validates
core/decoder/state topology and frozen-Qwen identity, keeps the loaded neural
components frozen, uses `SituatedMemory.recall()` and the memory-owned live
Moving Origin, decodes only over the turn's padded task-local candidates,
publishes the explicit plan and evidence attribution in the Qwen prompt, and
packages feedback text/outcome supplied by the caller. It never executes the
plan or changes the checkpoint plastic state.

Verification on CPU with Python 3.12 / PyTorch 2.11.0:

- `py -3.12 -m pytest tests/unit/runtime/test_conversation_bridge.py -q`:
  `6 passed`;
- focused bridge plus adjacent situated-Qwen/core/decoder tests: `19 passed`;
- `py -3.12 -m pytest tests/unit/runtime -q`: `23 passed`;
- both new Python files compile; `git diff --check` passes.

The default Python 3.11 installation lacked pytest and torch, so it was not a
valid test runtime; no dependency was installed. No GPU, live Cognee, live
Qwen, network, model download, checkpoint training/mutation, installer staging,
remote push, or external effect occurred. No supported trained checkpoint was
supplied or exercised, so conversation capability/readiness remains untested.
The next integration prerequisite is an explicitly supplied checkpoint whose
foundation digest and declared feature widths match the already-loaded local
Qwen and Situated Memory configuration, followed by a separately authorized
live integration/removal evaluation.

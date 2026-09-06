"""Model-agnostic runtime bridge for durable V14 procedural ability.

The bridge deliberately keeps four responsibilities separate:

* an attached frozen language model proposes and executes text procedures;
* a caller-supplied relation adapter exposes frozen V13 relation features;
* :class:`DurableAbilityLearner` owns the one persistent V14 neural state; and
* the canonical conversation journal owns replayable outcome evidence while
  Cognee remains only a disposable retrieval projection.

No verifier result, task-family label, target index, or solution is accepted by
the candidate encoder.  Objective feedback enters only after a candidate has
been selected and an immutable evidence identity has already been assigned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import io
import json
import math
from pathlib import Path
import zlib
import binascii
from typing import Any, Literal, Protocol, runtime_checkable

import torch

from angler.memory import (
    MemoryProjection,
    MovingOriginIndex,
    RecallBatch,
    SituatedMemory,
    encode_projection,
    projection_id,
)
from angler.reasoning.prospective_dynamics import (
    ProspectiveComponentStateIntegrity,
    ProspectiveFocusOutput,
    ProspectiveResourceBudget,
)
from angler.runtime.autonomous_apprentice import (
    AtomicCompetenceStore,
    OutcomeContext,
)
from angler.runtime.situated_conversation import ConversationJournal, JournalRecord


_SNAPSHOT_VERSION = "angler.durable-ability-runtime.v1"
_PENDING_SNAPSHOT_VERSION = "angler.durable-ability-pending.v1"
_PENDING_PROSPECTIVE_SNAPSHOT_VERSION = "angler.durable-ability-pending.v2"
_PENDING_OUTPUT_VERSION = "angler.structure-keyed-credit-output.v1"
_PENDING_PROSPECTIVE_VERSION = "angler.pending-prospective-material.v1"
_MAXIMUM_PENDING_STATE_BYTES = 16 * 1024 * 1024
_MAXIMUM_PENDING_SAFE_ITEMS = 32_768
_PENDING_TENSOR_FIELDS = (
    "logits",
    "base_logits",
    "residuals",
    "read_queries",
    "read_weights",
    "read_values",
    "temporal_hidden",
    "write_keys",
    "write_strengths",
    "observed_relations",
    "observed_temporal",
)
_PROSPECTIVE_OUTPUT_TENSOR_FIELDS = (
    "future_latents",
    "outcome_logits",
    "uncertainties",
    "selection_residuals",
    "focus_weights",
    "focused_mask",
)
_PROSPECTIVE_INPUT_TENSOR_FIELDS = (
    "relation_features",
    "temporal_features",
    "base_logits",
)
_EVENT_CONTEXT_COUNT_KEY = "ability_event_chunks"
_EVENT_CONTEXT_PREFIX = "ability_event_"
_PROCEDURE_CONTEXT_KEY = "procedure_trace"
_MAXIMUM_PROCEDURE_CHARS = 1_024
_MAXIMUM_TASK_CHARS = 16_384
_MAXIMUM_RESPONSE_CHARS = 16_384
_MINIMUM_CANDIDATES = 2
_MAXIMUM_CANDIDATES = 64


def _text(value: str, label: str, *, maximum: int) -> str:
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} must be non-empty text of at most {maximum} characters")
    return value.strip()


@runtime_checkable
class FrozenProcedureTextAdapter(Protocol):
    """Replaceable frozen-model boundary; Angler exchanges only text."""

    def propose_procedure_traces(
        self,
        task: str,
        recalled_evidence: tuple[str, ...],
        *,
        count: int = 4,
    ) -> Sequence[str]: ...

    def execute_with_procedure(
        self,
        task: str,
        selected_trace: str,
        observations: tuple[str, ...],
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class EncodedProcedureCandidate:
    """One label-free V13 relation row aligned with a public proposal."""

    trace: str
    relation_features: torch.Tensor
    temporal_features: torch.Tensor
    base_logit: torch.Tensor
    supporting_evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.trace, "candidate trace", maximum=_MAXIMUM_PROCEDURE_CHARS)
        if self.relation_features.ndim != 1:
            raise ValueError("relation_features must be one vector")
        if self.temporal_features.ndim != 1:
            raise ValueError("temporal_features must be one vector")
        if self.base_logit.shape not in ((), (1,)):
            raise ValueError("base_logit must be scalar")
        for name, value in (
            ("relation_features", self.relation_features),
            ("temporal_features", self.temporal_features),
            ("base_logit", self.base_logit),
        ):
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} must be finite")
        if type(self.supporting_evidence_refs) is not tuple:
            raise TypeError("supporting_evidence_refs must be a tuple")
        if any(type(item) is not str or not item for item in self.supporting_evidence_refs):
            raise ValueError("supporting evidence references must be non-empty strings")
        if len(set(self.supporting_evidence_refs)) != len(self.supporting_evidence_refs):
            raise ValueError("supporting evidence references must be unique")


@runtime_checkable
class ProcedureRelationAdapter(Protocol):
    """Frozen feature boundary; it receives no outcome or evaluator metadata."""

    def encode_candidates(
        self,
        task: str,
        proposals: tuple[str, ...],
        recall: RecallBatch,
    ) -> Sequence[EncodedProcedureCandidate]: ...


@dataclass(frozen=True, slots=True)
class AbilityDecision:
    task_id: str
    request: str
    proposals: tuple[str, ...]
    selected_index: int
    selected_trace: str
    logits: tuple[float, ...]
    evidence_ref: str
    supporting_evidence_refs: tuple[str, ...]
    parent_state_digest: str


@dataclass(frozen=True, slots=True)
class DurableAbilityTurn:
    decision: AbilityDecision
    recall: RecallBatch


@dataclass(frozen=True, slots=True)
class CompletedAbilityTurn:
    turn: DurableAbilityTurn
    response: str


@dataclass(frozen=True, slots=True)
class PendingProspectiveMaterial:
    """Bounded internal material for one unresolved successor prediction."""

    decision: AbilityDecision
    selected_output: Any
    prospective_output: ProspectiveFocusOutput
    relation_features: torch.Tensor
    temporal_features: torch.Tensor
    base_logits: torch.Tensor
    candidate_traces: tuple[str, ...]
    candidate_supporting_evidence_refs: tuple[tuple[str, ...], ...]
    recalled_refs: tuple[str, ...]
    context_refs: tuple[str, ...]
    eligibility_rows: tuple[tuple[int, ...], ...]
    resource_budget: ProspectiveResourceBudget
    parent_competence_digest: str
    checkpoint_ref: str
    config_ref: str
    component_state: ProspectiveComponentStateIntegrity
    batch_ref: str | None
    prospective_read_enabled: bool


@dataclass(slots=True)
class _PendingDecision:
    public: AbilityDecision
    output: Any
    prospective: PendingProspectiveMaterial | None = None


def _core_method(core: object, name: str):
    method = getattr(core, name, None)
    if not callable(method):
        raise TypeError(f"V14 core is missing required method {name}")
    return method


def _snapshot_mapping(snapshot: object) -> dict[str, object]:
    if is_dataclass(snapshot):
        return {field.name: getattr(snapshot, field.name) for field in fields(snapshot)}
    if isinstance(snapshot, Mapping):
        return dict(snapshot)
    raise TypeError("V14 core snapshot must be a dataclass or mapping")


def _event_refs(event: object) -> tuple[str, ...]:
    raw = getattr(event, "evidence_refs", None)
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        refs = tuple(raw)
        if all(type(item) is str and item for item in refs):
            return refs
    raise ValueError("V14 event has invalid evidence references")


def _jsonable(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if type(value) is float and not math.isfinite(value):
        raise ValueError("pending ability primitive floats must be finite")
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"event record contains unsupported value {type(value).__name__}")


def _event_context(event: object) -> dict[str, str]:
    to_record = getattr(event, "to_record", None)
    if not callable(to_record):
        raise TypeError("V14 event does not provide to_record")
    encoded = json.dumps(
        _jsonable(to_record()),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    import base64

    result = base64.b64encode(zlib.compress(encoded, level=9)).decode("ascii")
    chunks = tuple(result[offset : offset + 960] for offset in range(0, len(result), 960))
    if not chunks or len(chunks) > 8:
        raise ValueError("compressed V14 event exceeds the projection context ceiling")
    context = {_EVENT_CONTEXT_COUNT_KEY: str(len(chunks))}
    context.update(
        {f"{_EVENT_CONTEXT_PREFIX}{index:03d}": chunk for index, chunk in enumerate(chunks)}
    )
    return context


def _event_from_context(context: Mapping[str, str], *, core: object | None = None) -> object:
    import base64

    try:
        count = int(context[_EVENT_CONTEXT_COUNT_KEY])
        if not 1 <= count <= 8:
            raise ValueError("ability event chunk count is outside its ceiling")
        value = "".join(
            context[f"{_EVENT_CONTEXT_PREFIX}{index:03d}"] for index in range(count)
        )
        encoded = zlib.decompress(base64.b64decode(value, validate=True))
        record = json.loads(encoded.decode("utf-8"))
    except (
        KeyError,
        ValueError,
        binascii.Error,
        zlib.error,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        raise ValueError("ability event context is invalid") from exc
    return _event_from_record(record, core=core)


def _update_digest(digest: "hashlib._Hash", value: object) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\x00" + str(tensor.dtype).encode() + b"\x00")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(b"\x00" + tensor.view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\x00")
        for key in sorted(value, key=lambda item: str(item)):
            _update_digest(digest, str(key))
            _update_digest(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        digest.update(b"sequence\x00")
        for item in value:
            _update_digest(digest, item)
        return
    material = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    digest.update(type(value).__name__.encode() + b"\x00" + material.encode() + b"\x00")


def _sha256_ref(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 reference")
    return value


def _validate_pending_safe_value(
    value: object,
    *,
    depth: int = 0,
    counter: list[int] | None = None,
) -> None:
    """Accept only the tensor/primitive tree supported by weights-only loading."""

    if counter is None:
        counter = [0]
    counter[0] += 1
    if counter[0] > _MAXIMUM_PENDING_SAFE_ITEMS or depth > 8:
        raise ValueError("pending ability payload structure exceeds its ceiling")
    if isinstance(value, torch.Tensor):
        if (
            value.layout != torch.strided
            or value.is_quantized
            or value.device.type != "cpu"
            or value.requires_grad
        ):
            raise ValueError("pending ability tensors must be plain detached CPU tensors")
        return
    if value is None or type(value) in (str, int, float, bool):
        return
    if type(value) is list:
        for item in value:
            _validate_pending_safe_value(item, depth=depth + 1, counter=counter)
        return
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("pending ability mapping keys must be strings")
        for item in value.values():
            _validate_pending_safe_value(item, depth=depth + 1, counter=counter)
        return
    raise ValueError(
        f"pending ability payload contains unsupported {type(value).__name__}"
    )


def _pending_content_digest(body: Mapping[str, object]) -> str:
    digest = hashlib.sha256(b"project-angler.durable-ability-pending.v1\x00")
    _update_digest(digest, body)
    return "sha256:" + digest.hexdigest()


def _pending_prospective_content_digest(body: Mapping[str, object]) -> str:
    digest = hashlib.sha256(b"project-angler.durable-ability-pending.v2\x00")
    _update_digest(digest, body)
    return "sha256:" + digest.hexdigest()


def _detached_output_clone(output: object) -> object:
    mapping = _snapshot_mapping(output)
    return type(output)(
        **{
            name: value.detach().contiguous().clone()
            for name, value in mapping.items()
            if isinstance(value, torch.Tensor)
        },
        **{
            name: value
            for name, value in mapping.items()
            if not isinstance(value, torch.Tensor)
        },
    )


def _detached_prospective_clone(
    output: ProspectiveFocusOutput,
) -> ProspectiveFocusOutput:
    if type(output) is not ProspectiveFocusOutput:
        raise TypeError("prospective output must be ProspectiveFocusOutput")
    return ProspectiveFocusOutput(
        future_latents=output.future_latents.detach().contiguous().clone(),
        outcome_logits=output.outcome_logits.detach().contiguous().clone(),
        uncertainties=output.uncertainties.detach().contiguous().clone(),
        selection_residuals=(
            output.selection_residuals.detach().contiguous().clone()
        ),
        focus_weights=output.focus_weights.detach().contiguous().clone(),
        focused_mask=output.focused_mask.detach().contiguous().clone(),
        recurrent_steps=output.recurrent_steps,
        state_step=output.state_step,
    )


def _replace_prospective_row(
    batched: ProspectiveFocusOutput,
    selected: ProspectiveFocusOutput,
    selected_index: int,
) -> ProspectiveFocusOutput:
    if type(batched) is not ProspectiveFocusOutput or type(selected) is not ProspectiveFocusOutput:
        raise TypeError("prospective outputs must be exact ProspectiveFocusOutput values")
    if type(selected_index) is not int or not 0 <= selected_index < batched.outcome_logits.shape[0]:
        raise ValueError("selected prospective row index is invalid")
    if (
        selected.outcome_logits.shape != (1,)
        or selected.recurrent_steps != batched.recurrent_steps
        or selected.state_step != batched.state_step
    ):
        raise ValueError("selected prospective output does not align with its batch")

    def replace_row(parent: torch.Tensor, child: torch.Tensor) -> torch.Tensor:
        if parent.shape[1:] != child.shape[1:] or child.shape[0] != 1:
            raise ValueError("selected prospective tensor row does not align")
        result = parent.detach().contiguous().clone()
        result[selected_index : selected_index + 1] = child.detach()
        return result

    return ProspectiveFocusOutput(
        future_latents=replace_row(batched.future_latents, selected.future_latents),
        outcome_logits=replace_row(batched.outcome_logits, selected.outcome_logits),
        uncertainties=replace_row(batched.uncertainties, selected.uncertainties),
        selection_residuals=replace_row(
            batched.selection_residuals, selected.selection_residuals
        ),
        focus_weights=replace_row(batched.focus_weights, selected.focus_weights),
        focused_mask=replace_row(batched.focused_mask, selected.focused_mask),
        recurrent_steps=batched.recurrent_steps,
        state_step=batched.state_step,
    )


def _clone_pending_prospective_material(
    material: PendingProspectiveMaterial,
) -> PendingProspectiveMaterial:
    return replace(
        material,
        selected_output=_detached_output_clone(material.selected_output),
        prospective_output=_detached_prospective_clone(material.prospective_output),
        relation_features=material.relation_features.detach().contiguous().clone(),
        temporal_features=material.temporal_features.detach().contiguous().clone(),
        base_logits=material.base_logits.detach().contiguous().clone(),
    )


def _tensor_bundle_record(
    values: Mapping[str, torch.Tensor],
    expected_fields: tuple[str, ...],
) -> dict[str, object]:
    if set(values) != set(expected_fields):
        raise ValueError("pending tensor bundle fields are malformed")
    tensors = {
        name: values[name].detach().cpu().contiguous().clone()
        for name in expected_fields
    }
    return {
        "tensors": tensors,
        "tensor_specs": {
            name: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in tensors.items()
        },
    }


def _tensor_bundle_from_record(
    record: object,
    expected_fields: tuple[str, ...],
    *,
    reference: torch.Tensor,
    boolean_fields: frozenset[str] = frozenset(),
) -> dict[str, torch.Tensor]:
    if type(record) is not dict or set(record) != {"tensors", "tensor_specs"}:
        raise ValueError("pending tensor bundle envelope is malformed")
    tensors = record["tensors"]
    specs = record["tensor_specs"]
    if (
        type(tensors) is not dict
        or set(tensors) != set(expected_fields)
        or type(specs) is not dict
        or set(specs) != set(expected_fields)
    ):
        raise ValueError("pending tensor bundle set is malformed")
    restored: dict[str, torch.Tensor] = {}
    for name in expected_fields:
        value = tensors[name]
        spec = specs[name]
        if not isinstance(value, torch.Tensor) or type(spec) is not dict or set(spec) != {
            "dtype",
            "shape",
        }:
            raise ValueError(f"pending tensor bundle {name} metadata is malformed")
        shape = spec["shape"]
        expected_dtype = torch.bool if name in boolean_fields else reference.dtype
        if (
            type(spec["dtype"]) is not str
            or type(shape) is not list
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
            or str(value.dtype) != spec["dtype"]
            or list(value.shape) != shape
            or value.dtype != expected_dtype
            or (name not in boolean_fields and not value.is_floating_point())
            or (value.is_floating_point() and not bool(torch.isfinite(value).all().item()))
        ):
            raise ValueError(f"pending tensor bundle {name} dtype/shape/value is invalid")
        restored[name] = value.detach().to(device=reference.device).contiguous().clone()
    return restored


class DurableAbilityLearner:
    """CompetenceLearner-compatible owner of one V14 state and event lineage."""

    def __init__(
        self,
        core: object,
        state: object,
        *,
        checkpoint_identity: str,
        events: Sequence[object] = (),
        tombstoned_evidence_refs: Sequence[str] = (),
        prospective_lesion: bool = False,
    ) -> None:
        if type(prospective_lesion) is not bool:
            raise TypeError("prospective_lesion must be bool")
        self.core = core
        self.prospective_lesion = prospective_lesion
        self.checkpoint_identity = _text(
            checkpoint_identity, "checkpoint_identity", maximum=512
        )
        _core_method(core, "validate_state")(state)
        requires_grad = getattr(core, "requires_grad_", None)
        evaluate = getattr(core, "eval", None)
        if callable(requires_grad):
            requires_grad(False)
        if callable(evaluate):
            evaluate()
        self.state = state
        self._snapshot_type = type(_core_method(core, "capture_state")(state))
        self.events = tuple(events)
        self.tombstoned_evidence_refs = frozenset(tombstoned_evidence_refs)
        if any(type(item) is not str or not item for item in self.tombstoned_evidence_refs):
            raise ValueError("tombstoned evidence references must be non-empty strings")
        for event in self.events:
            if self.tombstoned_evidence_refs.intersection(_event_refs(event)):
                raise ValueError("active event depends on tombstoned evidence")
        self._pending: _PendingDecision | None = None
        self._last_event: object | None = None

    @property
    def pending_decision(self) -> AbilityDecision | None:
        return None if self._pending is None else self._pending.public

    @property
    def pending_prospective_material(self) -> PendingProspectiveMaterial | None:
        if self._pending is None or self._pending.prospective is None:
            return None
        return _clone_pending_prospective_material(self._pending.prospective)

    @property
    def last_event(self) -> object | None:
        return self._last_event

    def component_state_integrity(self) -> ProspectiveComponentStateIntegrity:
        method = _core_method(self.core, "component_state_integrity")
        result = method(self.state)
        if type(result) is not ProspectiveComponentStateIntegrity:
            raise TypeError("core returned malformed component-state integrity")
        if result.checkpoint_ref != self.checkpoint_identity:
            raise ValueError("learner checkpoint and prospective component differ")
        return result

    def bind_pending_prospective_batch(
        self,
        batch_ref: str,
    ) -> PendingProspectiveMaterial:
        batch_ref = _sha256_ref(batch_ref, "prospective batch reference")
        if self._pending is None or self._pending.prospective is None:
            raise RuntimeError("no prospective material is awaiting batch binding")
        current = self._pending.prospective
        if current.batch_ref is not None and current.batch_ref != batch_ref:
            raise ValueError("pending prospective material is bound to another batch")
        if current.batch_ref is None:
            current = replace(current, batch_ref=batch_ref)
            self._pending.prospective = current
        return _clone_pending_prospective_material(current)

    def _decision_record(self, decision: AbilityDecision) -> dict[str, object]:
        if not isinstance(decision, AbilityDecision):
            raise TypeError("pending public decision must be AbilityDecision")
        record: dict[str, object] = {
            "task_id": decision.task_id,
            "request": decision.request,
            "proposals": list(decision.proposals),
            "selected_index": decision.selected_index,
            "selected_trace": decision.selected_trace,
            "logits": list(decision.logits),
            "evidence_ref": decision.evidence_ref,
            "supporting_evidence_refs": list(decision.supporting_evidence_refs),
            "parent_state_digest": decision.parent_state_digest,
        }
        if self._decision_from_record(record) != decision:
            raise ValueError("pending public decision does not round-trip exactly")
        return record

    def _decision_from_record(self, record: object) -> AbilityDecision:
        expected = {
            "task_id",
            "request",
            "proposals",
            "selected_index",
            "selected_trace",
            "logits",
            "evidence_ref",
            "supporting_evidence_refs",
            "parent_state_digest",
        }
        if type(record) is not dict or set(record) != expected:
            raise ValueError("pending public decision envelope is malformed")
        task_id = record["task_id"]
        request = record["request"]
        if _text(task_id, "pending task_id", maximum=256) != task_id:
            raise ValueError("pending task_id is not canonical")
        if _text(request, "pending request", maximum=_MAXIMUM_TASK_CHARS) != request:
            raise ValueError("pending request is not canonical")
        raw_proposals = record["proposals"]
        if (
            type(raw_proposals) is not list
            or not _MINIMUM_CANDIDATES <= len(raw_proposals) <= _MAXIMUM_CANDIDATES
        ):
            raise ValueError("pending decision must contain 2 through 64 proposals")
        proposals = tuple(raw_proposals)
        for proposal in proposals:
            _text(proposal, "pending proposal", maximum=_MAXIMUM_PROCEDURE_CHARS)
        if len(set(proposals)) != len(proposals):
            raise ValueError("pending proposals must be text-distinct")
        selected_index = record["selected_index"]
        if type(selected_index) is not int or not 0 <= selected_index < len(proposals):
            raise ValueError("pending selected_index is invalid")
        selected_trace = record["selected_trace"]
        if type(selected_trace) is not str or selected_trace != proposals[selected_index]:
            raise ValueError("pending selected trace does not match its proposal")
        raw_logits = record["logits"]
        if (
            type(raw_logits) is not list
            or len(raw_logits) != len(proposals)
            or any(type(value) is not float or not math.isfinite(value) for value in raw_logits)
        ):
            raise ValueError("pending decision logits must align as finite floats")
        raw_supporting = record["supporting_evidence_refs"]
        if type(raw_supporting) is not list:
            raise ValueError("pending supporting evidence must be a list")
        supporting = tuple(raw_supporting)
        if (
            any(type(value) is not str or not value for value in supporting)
            or len(set(supporting)) != len(supporting)
        ):
            raise ValueError("pending supporting evidence references are invalid")
        parent_digest = _sha256_ref(
            record["parent_state_digest"], "pending parent-state digest"
        )
        current_parent = self.state_digest()
        if parent_digest != current_parent:
            raise ValueError("pending decision belongs to another parent state")
        evidence_ref = _sha256_ref(record["evidence_ref"], "pending evidence reference")
        material = json.dumps(
            {
                "checkpoint": self.checkpoint_identity,
                "parent": parent_digest,
                "request": request,
                "selected_index": selected_index,
                "selected_trace": selected_trace,
                "step": int(getattr(self.state, "step")),
                "task_id": task_id,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        expected_evidence = "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()
        if evidence_ref != expected_evidence:
            raise ValueError("pending evidence reference does not bind the public decision")
        return AbilityDecision(
            task_id=task_id,
            request=request,
            proposals=proposals,
            selected_index=selected_index,
            selected_trace=selected_trace,
            logits=tuple(raw_logits),
            evidence_ref=evidence_ref,
            supporting_evidence_refs=supporting,
            parent_state_digest=parent_digest,
        )

    def _validate_pending_output(
        self,
        output: object,
        decision: AbilityDecision,
        *,
        prospective_read_enabled: bool | None = None,
    ) -> None:
        if prospective_read_enabled is not None and type(prospective_read_enabled) is not bool:
            raise TypeError("prospective_read_enabled must be bool or None")
        try:
            _core_method(self.core, "_validate_output")(output)
        except Exception as exc:
            raise ValueError("pending neural output is invalid") from exc
        mapping = _snapshot_mapping(output)
        if set(mapping) != {*_PENDING_TENSOR_FIELDS, "state_step", "read_enabled"}:
            raise ValueError("pending neural output fields are malformed")
        if mapping["state_step"] != int(getattr(self.state, "step")):
            raise ValueError("pending neural output belongs to another parent step")
        if mapping["read_enabled"] is not True:
            raise ValueError("pending neural output must preserve the selected read path")
        for name in _PENDING_TENSOR_FIELDS:
            value = mapping[name]
            if (
                not isinstance(value, torch.Tensor)
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all().item())
            ):
                raise ValueError(f"pending neural output {name} is not finite floating data")
        if mapping["logits"].shape != (1,):
            raise ValueError("pending neural output must contain exactly one selected row")
        selected_logit = float(mapping["logits"].detach().cpu().item())
        if selected_logit != decision.logits[decision.selected_index]:
            raise ValueError("pending neural output does not match the public selected logit")
        if not torch.equal(
            mapping["logits"], mapping["base_logits"] + mapping["residuals"]
        ):
            raise ValueError("pending neural output logit decomposition is invalid")
        with torch.no_grad():
            if prospective_read_enabled is None:
                recomputed = _core_method(self.core, "predict")(
                    mapping["observed_relations"],
                    mapping["observed_temporal"],
                    mapping["base_logits"],
                    state=self.state,
                    read_enabled=True,
                )
            else:
                recomputed, _prospective = _core_method(
                    self.core, "predict_with_prospective"
                )(
                    mapping["observed_relations"],
                    mapping["observed_temporal"],
                    mapping["base_logits"],
                    state=self.state,
                    read_enabled=True,
                    prospective_read_enabled=prospective_read_enabled,
                )
        recomputed_mapping = _snapshot_mapping(recomputed)
        for name in _PENDING_TENSOR_FIELDS:
            if not torch.equal(mapping[name], recomputed_mapping[name]):
                raise ValueError(
                    f"pending neural output {name} does not match the frozen parent prediction"
                )
        if (
            mapping["state_step"] != recomputed_mapping["state_step"]
            or mapping["read_enabled"] != recomputed_mapping["read_enabled"]
        ):
            raise ValueError("pending neural output metadata does not match prediction")

    def _output_record(
        self,
        output: object,
        decision: AbilityDecision,
        *,
        prospective_read_enabled: bool | None = None,
    ) -> dict[str, object]:
        self._validate_pending_output(
            output,
            decision,
            prospective_read_enabled=prospective_read_enabled,
        )
        mapping = _snapshot_mapping(output)
        tensors = {
            name: mapping[name].detach().cpu().contiguous().clone()
            for name in _PENDING_TENSOR_FIELDS
        }
        specs = {
            name: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
            for name, value in tensors.items()
        }
        return {
            "version": _PENDING_OUTPUT_VERSION,
            "tensors": tensors,
            "tensor_specs": specs,
            "state_step": mapping["state_step"],
            "read_enabled": mapping["read_enabled"],
        }

    def _output_from_record(
        self,
        record: object,
        decision: AbilityDecision,
        *,
        prospective_read_enabled: bool | None = None,
    ) -> object:
        if type(record) is not dict or set(record) != {
            "version",
            "tensors",
            "tensor_specs",
            "state_step",
            "read_enabled",
        }:
            raise ValueError("pending neural output envelope is malformed")
        if record["version"] != _PENDING_OUTPUT_VERSION:
            raise ValueError("pending neural output version is unsupported")
        tensors = record["tensors"]
        specs = record["tensor_specs"]
        if (
            type(tensors) is not dict
            or set(tensors) != set(_PENDING_TENSOR_FIELDS)
            or type(specs) is not dict
            or set(specs) != set(_PENDING_TENSOR_FIELDS)
        ):
            raise ValueError("pending neural output tensor set is malformed")
        reference = next(self.core.parameters())
        restored: dict[str, torch.Tensor] = {}
        for name in _PENDING_TENSOR_FIELDS:
            value = tensors[name]
            spec = specs[name]
            if not isinstance(value, torch.Tensor) or type(spec) is not dict or set(spec) != {
                "dtype",
                "shape",
            }:
                raise ValueError(f"pending neural output {name} metadata is malformed")
            shape = spec["shape"]
            if (
                type(spec["dtype"]) is not str
                or type(shape) is not list
                or any(type(dimension) is not int or dimension < 0 for dimension in shape)
                or str(value.dtype) != spec["dtype"]
                or list(value.shape) != shape
                or value.dtype != reference.dtype
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all().item())
            ):
                raise ValueError(f"pending neural output {name} dtype/shape/value is invalid")
            restored[name] = value.detach().to(device=reference.device).clone()
        state_step = record["state_step"]
        read_enabled = record["read_enabled"]
        if type(state_step) is not int or type(read_enabled) is not bool:
            raise ValueError("pending neural output scalar metadata is invalid")
        try:
            from angler.reasoning.structure_keyed_credit_memory import (
                StructureKeyedCreditOutput,
            )

            output = StructureKeyedCreditOutput(
                **restored,
                state_step=state_step,
                read_enabled=read_enabled,
            )
        except Exception as exc:
            raise ValueError("pending neural output cannot be reconstructed") from exc
        self._validate_pending_output(
            output,
            decision,
            prospective_read_enabled=prospective_read_enabled,
        )
        return output

    def _validate_pending_prospective_material(
        self,
        material: PendingProspectiveMaterial,
        *,
        require_batch: bool,
    ) -> None:
        if type(material) is not PendingProspectiveMaterial:
            raise TypeError("pending prospective material has the wrong type")
        if material.decision.parent_state_digest != self.state_digest():
            raise ValueError("pending prospective material belongs to another parent")
        if material.parent_competence_digest != material.decision.parent_state_digest:
            raise ValueError("pending prospective competence binding drifted")
        component = self.component_state_integrity()
        if material.component_state != component:
            raise ValueError("pending prospective component state drifted")
        if (
            material.checkpoint_ref != component.checkpoint_ref
            or material.config_ref != component.config_ref
        ):
            raise ValueError("pending prospective checkpoint/config binding drifted")
        if material.prospective_read_enabled is not (not self.prospective_lesion):
            raise ValueError("pending prospective removal condition drifted")
        if require_batch and material.batch_ref is None:
            raise ValueError("pending prospective material has no batch binding")
        if material.batch_ref is not None:
            _sha256_ref(material.batch_ref, "pending prospective batch reference")

        count = len(material.decision.proposals)
        if material.candidate_traces != material.decision.proposals:
            raise ValueError("pending prospective candidate traces drifted")
        if (
            type(material.candidate_supporting_evidence_refs) is not tuple
            or len(material.candidate_supporting_evidence_refs) != count
        ):
            raise ValueError("pending prospective candidate evidence is misaligned")
        if (
            type(material.recalled_refs) is not tuple
            or not 1 <= len(material.recalled_refs) <= 256
            or len(set(material.recalled_refs)) != len(material.recalled_refs)
        ):
            raise ValueError("pending prospective recalled refs are invalid")
        for value in material.recalled_refs:
            _sha256_ref(value, "pending prospective recalled reference")
        recalled = set(material.recalled_refs)
        for refs in material.candidate_supporting_evidence_refs:
            if (
                type(refs) is not tuple
                or not refs
                or refs != tuple(sorted(refs))
                or len(set(refs)) != len(refs)
            ):
                raise ValueError("pending prospective branch evidence is not canonical")
            for value in refs:
                _sha256_ref(value, "pending prospective supporting evidence")
            if not set(refs).issubset(recalled):
                raise ValueError("pending prospective branch cites absent recall")
        if (
            material.decision.supporting_evidence_refs
            != material.candidate_supporting_evidence_refs[
                material.decision.selected_index
            ]
        ):
            raise ValueError("selected prospective evidence drifted")

        if type(material.context_refs) is not tuple or len(material.context_refs) != 1:
            raise ValueError("the first prospective lane requires one context ref")
        _sha256_ref(material.context_refs[0], "pending prospective context reference")
        if (
            type(material.eligibility_rows) is not tuple
            or len(material.eligibility_rows) != count
            or any(row != (0,) for row in material.eligibility_rows)
        ):
            raise ValueError("the first prospective lane requires eligibility (0,)")
        budget = material.resource_budget
        if (
            type(budget) is not ProspectiveResourceBudget
            or budget.world_reads != 1
            or budget.branches != count
            or budget.recurrent_steps != 1
        ):
            raise ValueError("pending prospective resource budget drifted")

        reference = next(self.core.parameters())
        expected_inputs = (
            ("relation_features", material.relation_features, (count, int(getattr(self.core, "relation_width", -1)))),
            ("temporal_features", material.temporal_features, (count, int(getattr(self.core, "temporal_width", -1)))),
            ("base_logits", material.base_logits, (count,)),
        )
        for name, value, shape in expected_inputs:
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != shape
                or value.device != reference.device
                or value.dtype != reference.dtype
                or not value.is_floating_point()
                or value.requires_grad
                or not value.is_contiguous()
                or not bool(torch.isfinite(value).all().item())
            ):
                raise ValueError(f"pending prospective {name} is invalid")

        self._validate_pending_output(
            material.selected_output,
            material.decision,
            prospective_read_enabled=material.prospective_read_enabled,
        )
        prospective = material.prospective_output
        if type(prospective) is not ProspectiveFocusOutput:
            raise TypeError("pending prospective output has the wrong type")
        eligible = torch.ones(
            count,
            1,
            device=reference.device,
            dtype=torch.bool,
        )
        _core_method(self.core, "_validate_prospective_output")(
            prospective,
            batch=count,
            worlds=1,
            eligible_mask=eligible,
            state=self.state,
            budget=budget,
        )
        for value in (
            prospective.future_latents,
            prospective.outcome_logits,
            prospective.uncertainties,
            prospective.selection_residuals,
            prospective.focus_weights,
            prospective.focused_mask,
        ):
            if value.requires_grad or not value.is_contiguous():
                raise ValueError("pending prospective outputs must be detached contiguous clones")

        with torch.no_grad():
            combined, batch_prospective = _core_method(
                self.core, "predict_with_prospective"
            )(
                material.relation_features,
                material.temporal_features,
                material.base_logits,
                state=self.state,
                read_enabled=True,
                prospective_read_enabled=material.prospective_read_enabled,
            )
            batch_logits = [
                float(value) for value in combined.logits.detach().cpu().tolist()
            ]
            selected_index = max(range(count), key=lambda index: batch_logits[index])
            if selected_index != material.decision.selected_index:
                raise ValueError("pending prospective winner drifted")
            selected_output, selected_prospective = _core_method(
                self.core, "predict_with_prospective"
            )(
                material.relation_features[selected_index : selected_index + 1],
                material.temporal_features[selected_index : selected_index + 1],
                material.base_logits[selected_index : selected_index + 1],
                state=self.state,
                read_enabled=True,
                prospective_read_enabled=material.prospective_read_enabled,
            )
        batch_logits[selected_index] = float(selected_output.logits.detach().cpu().item())
        if max(range(count), key=lambda index: batch_logits[index]) != selected_index:
            raise ValueError("exact prospective selected row changed the winner")
        if tuple(batch_logits) != material.decision.logits:
            raise ValueError("pending prospective public logits drifted")
        for name in _PENDING_TENSOR_FIELDS:
            if not torch.equal(
                getattr(material.selected_output, name),
                getattr(selected_output, name),
            ):
                raise ValueError("pending selected prospective output drifted")
        expected_prospective = _replace_prospective_row(
            batch_prospective,
            selected_prospective,
            selected_index,
        )
        for name in _PROSPECTIVE_OUTPUT_TENSOR_FIELDS:
            if not torch.equal(
                getattr(prospective, name), getattr(expected_prospective, name)
            ):
                raise ValueError(f"pending prospective {name} drifted")
        if (
            prospective.recurrent_steps != expected_prospective.recurrent_steps
            or prospective.state_step != expected_prospective.state_step
        ):
            raise ValueError("pending prospective scalar metadata drifted")

    def _prospective_material_record(
        self,
        material: PendingProspectiveMaterial,
    ) -> dict[str, object]:
        self._validate_pending_prospective_material(material, require_batch=True)
        component = material.component_state
        prospective = material.prospective_output
        return {
            "version": _PENDING_PROSPECTIVE_VERSION,
            "batch_ref": material.batch_ref,
            "candidate_traces": list(material.candidate_traces),
            "candidate_supporting_evidence_refs": [
                list(refs) for refs in material.candidate_supporting_evidence_refs
            ],
            "recalled_refs": list(material.recalled_refs),
            "context_refs": list(material.context_refs),
            "eligibility_rows": [list(row) for row in material.eligibility_rows],
            "resource_budget": {
                "world_reads": material.resource_budget.world_reads,
                "branches": material.resource_budget.branches,
                "recurrent_steps": material.resource_budget.recurrent_steps,
            },
            "parent_competence_digest": material.parent_competence_digest,
            "checkpoint_ref": material.checkpoint_ref,
            "config_ref": material.config_ref,
            "component_state": {
                "step": component.step,
                "checkpoint_ref": component.checkpoint_ref,
                "config_ref": component.config_ref,
                "world_state_digest": component.world_state_digest,
                "self_state_digest": component.self_state_digest,
                "focus_state_digest": component.focus_state_digest,
                "outcome_state_digest": component.outcome_state_digest,
            },
            "prospective_read_enabled": material.prospective_read_enabled,
            "candidate_inputs": _tensor_bundle_record(
                {
                    "relation_features": material.relation_features,
                    "temporal_features": material.temporal_features,
                    "base_logits": material.base_logits,
                },
                _PROSPECTIVE_INPUT_TENSOR_FIELDS,
            ),
            "prospective_output": {
                **_tensor_bundle_record(
                    {
                        name: getattr(prospective, name)
                        for name in _PROSPECTIVE_OUTPUT_TENSOR_FIELDS
                    },
                    _PROSPECTIVE_OUTPUT_TENSOR_FIELDS,
                ),
                "recurrent_steps": prospective.recurrent_steps,
                "state_step": prospective.state_step,
            },
        }

    def _prospective_material_from_record(
        self,
        record: object,
        *,
        decision: AbilityDecision,
        selected_output: object,
    ) -> PendingProspectiveMaterial:
        expected = {
            "version",
            "batch_ref",
            "candidate_traces",
            "candidate_supporting_evidence_refs",
            "recalled_refs",
            "context_refs",
            "eligibility_rows",
            "resource_budget",
            "parent_competence_digest",
            "checkpoint_ref",
            "config_ref",
            "component_state",
            "prospective_read_enabled",
            "candidate_inputs",
            "prospective_output",
        }
        if type(record) is not dict or set(record) != expected:
            raise ValueError("pending prospective material envelope is malformed")
        if record["version"] != _PENDING_PROSPECTIVE_VERSION:
            raise ValueError("pending prospective material version is unsupported")
        reference = next(self.core.parameters())
        inputs = _tensor_bundle_from_record(
            record["candidate_inputs"],
            _PROSPECTIVE_INPUT_TENSOR_FIELDS,
            reference=reference,
        )
        output_record = record["prospective_output"]
        if type(output_record) is not dict or set(output_record) != {
            "tensors",
            "tensor_specs",
            "recurrent_steps",
            "state_step",
        }:
            raise ValueError("pending prospective output envelope is malformed")
        output_tensors = _tensor_bundle_from_record(
            {
                "tensors": output_record["tensors"],
                "tensor_specs": output_record["tensor_specs"],
            },
            _PROSPECTIVE_OUTPUT_TENSOR_FIELDS,
            reference=reference,
            boolean_fields=frozenset({"focused_mask"}),
        )
        recurrent_steps = output_record["recurrent_steps"]
        state_step = output_record["state_step"]
        if type(recurrent_steps) is not int or type(state_step) is not int:
            raise ValueError("pending prospective output metadata is malformed")
        prospective_output = ProspectiveFocusOutput(
            **output_tensors,
            recurrent_steps=recurrent_steps,
            state_step=state_step,
        )
        budget_record = record["resource_budget"]
        if type(budget_record) is not dict or set(budget_record) != {
            "world_reads",
            "branches",
            "recurrent_steps",
        }:
            raise ValueError("pending prospective budget is malformed")
        budget = ProspectiveResourceBudget(**budget_record)
        component_record = record["component_state"]
        component_fields = {
            "step",
            "checkpoint_ref",
            "config_ref",
            "world_state_digest",
            "self_state_digest",
            "focus_state_digest",
            "outcome_state_digest",
        }
        if type(component_record) is not dict or set(component_record) != component_fields:
            raise ValueError("pending prospective component state is malformed")
        component = ProspectiveComponentStateIntegrity(**component_record)
        raw_traces = record["candidate_traces"]
        raw_support = record["candidate_supporting_evidence_refs"]
        raw_recalled = record["recalled_refs"]
        raw_context = record["context_refs"]
        raw_eligibility = record["eligibility_rows"]
        if not all(
            type(value) is list
            for value in (
                raw_traces,
                raw_support,
                raw_recalled,
                raw_context,
                raw_eligibility,
            )
        ) or any(type(value) is not list for value in (*raw_support, *raw_eligibility)):
            raise ValueError("pending prospective sequence fields are malformed")
        flag = record["prospective_read_enabled"]
        if type(flag) is not bool:
            raise ValueError("pending prospective removal flag is malformed")
        material = PendingProspectiveMaterial(
            decision=decision,
            selected_output=selected_output,
            prospective_output=prospective_output,
            relation_features=inputs["relation_features"],
            temporal_features=inputs["temporal_features"],
            base_logits=inputs["base_logits"],
            candidate_traces=tuple(raw_traces),
            candidate_supporting_evidence_refs=tuple(
                tuple(values) for values in raw_support
            ),
            recalled_refs=tuple(raw_recalled),
            context_refs=tuple(raw_context),
            eligibility_rows=tuple(tuple(values) for values in raw_eligibility),
            resource_budget=budget,
            parent_competence_digest=_sha256_ref(
                record["parent_competence_digest"],
                "pending prospective parent competence",
            ),
            checkpoint_ref=_sha256_ref(
                record["checkpoint_ref"], "pending prospective checkpoint"
            ),
            config_ref=_sha256_ref(
                record["config_ref"], "pending prospective configuration"
            ),
            component_state=component,
            batch_ref=_sha256_ref(
                record["batch_ref"], "pending prospective batch reference"
            ),
            prospective_read_enabled=flag,
        )
        self._validate_pending_prospective_material(material, require_batch=True)
        return material

    def capture_pending_state(self) -> bytes:
        """Capture the exact unresolved public decision and selected neural output."""

        if self._pending is None:
            raise RuntimeError("no ability decision is awaiting objective feedback")
        decision_record = self._decision_record(self._pending.public)
        material = self._pending.prospective
        if material is None:
            body: dict[str, object] = {
                "version": _PENDING_SNAPSHOT_VERSION,
                "checkpoint_identity": self.checkpoint_identity,
                "parent_state_digest": self._pending.public.parent_state_digest,
                "public_decision": decision_record,
                "selected_output": self._output_record(
                    self._pending.output, self._pending.public
                ),
            }
            content_digest = _pending_content_digest(body)
        else:
            body = {
                "version": _PENDING_PROSPECTIVE_SNAPSHOT_VERSION,
                "checkpoint_identity": self.checkpoint_identity,
                "parent_state_digest": self._pending.public.parent_state_digest,
                "public_decision": decision_record,
                "selected_output": self._output_record(
                    self._pending.output,
                    self._pending.public,
                    prospective_read_enabled=material.prospective_read_enabled,
                ),
                "prospective_material": self._prospective_material_record(material),
            }
            content_digest = _pending_prospective_content_digest(body)
        payload = {**body, "content_sha256": content_digest}
        _validate_pending_safe_value(payload)
        stream = io.BytesIO()
        torch.save(payload, stream)
        encoded = stream.getvalue()
        if len(encoded) > _MAXIMUM_PENDING_STATE_BYTES:
            raise ValueError("pending ability snapshot exceeds 16 MiB")
        return encoded

    def restore_pending_state(self, payload: bytes) -> AbilityDecision:
        """Restore a pending decision without invoking unrestricted deserialization."""

        if self._pending is not None:
            raise RuntimeError("an ability decision is already awaiting outcome")
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("pending ability snapshot must be non-empty bytes")
        if len(payload) > _MAXIMUM_PENDING_STATE_BYTES:
            raise ValueError("pending ability snapshot exceeds 16 MiB")
        try:
            decoded = torch.load(
                io.BytesIO(payload), map_location="cpu", weights_only=True
            )
        except Exception as exc:
            raise ValueError("pending ability snapshot cannot be decoded safely") from exc
        _validate_pending_safe_value(decoded)
        if type(decoded) is not dict:
            raise ValueError("pending ability snapshot envelope is malformed")
        version = decoded.get("version")
        if version == _PENDING_SNAPSHOT_VERSION:
            body_fields = (
                "version",
                "checkpoint_identity",
                "parent_state_digest",
                "public_decision",
                "selected_output",
            )
            expected_fields = {*body_fields, "content_sha256"}
            digest_method = _pending_content_digest
        elif version == _PENDING_PROSPECTIVE_SNAPSHOT_VERSION:
            body_fields = (
                "version",
                "checkpoint_identity",
                "parent_state_digest",
                "public_decision",
                "selected_output",
                "prospective_material",
            )
            expected_fields = {*body_fields, "content_sha256"}
            digest_method = _pending_prospective_content_digest
        else:
            raise ValueError("pending ability snapshot version is unsupported")
        if set(decoded) != expected_fields:
            raise ValueError("pending ability snapshot envelope is malformed")
        if decoded["checkpoint_identity"] != self.checkpoint_identity:
            raise ValueError("pending ability snapshot belongs to another checkpoint")
        parent_digest = _sha256_ref(
            decoded["parent_state_digest"], "pending parent-state digest"
        )
        if parent_digest != self.state_digest():
            raise ValueError("pending ability snapshot belongs to another parent state")
        body = {key: decoded[key] for key in body_fields}
        expected_digest = digest_method(body)
        content_digest = _sha256_ref(
            decoded["content_sha256"], "pending content digest"
        )
        if content_digest != expected_digest:
            raise ValueError("pending ability snapshot content digest is invalid")
        decision = self._decision_from_record(decoded["public_decision"])
        if decision.parent_state_digest != parent_digest:
            raise ValueError("pending public decision parent binding is inconsistent")
        if version == _PENDING_SNAPSHOT_VERSION:
            output = self._output_from_record(decoded["selected_output"], decision)
            material = None
        else:
            prospective_record = decoded["prospective_material"]
            if type(prospective_record) is not dict:
                raise ValueError("pending prospective material envelope is malformed")
            flag = prospective_record.get("prospective_read_enabled")
            if type(flag) is not bool:
                raise ValueError("pending prospective removal flag is malformed")
            output = self._output_from_record(
                decoded["selected_output"],
                decision,
                prospective_read_enabled=flag,
            )
            material = self._prospective_material_from_record(
                prospective_record,
                decision=decision,
                selected_output=output,
            )
        self._pending = _PendingDecision(decision, output, material)
        return decision

    def select(
        self,
        *,
        task_id: str,
        request: str,
        candidates: Sequence[EncodedProcedureCandidate],
    ) -> AbilityDecision:
        if self._pending is not None:
            raise RuntimeError("an ability decision is already awaiting outcome")
        task_id = _text(task_id, "task_id", maximum=256)
        request = _text(request, "request", maximum=_MAXIMUM_TASK_CHARS)
        rows = tuple(candidates)
        if not _MINIMUM_CANDIDATES <= len(rows) <= _MAXIMUM_CANDIDATES:
            raise ValueError("2 through 64 encoded procedure candidates are required")
        if len({row.trace for row in rows}) != len(rows):
            raise ValueError("encoded procedure candidates must be text-distinct")
        relation_width = int(getattr(self.core, "relation_width", -1))
        temporal_width = int(getattr(self.core, "temporal_width", -1))
        for row in rows:
            if row.relation_features.shape != (relation_width,):
                raise ValueError("candidate relation width does not match V14 core")
            if row.temporal_features.shape != (temporal_width,):
                raise ValueError("candidate temporal width does not match V14 core")
        relation_batch = torch.stack(tuple(row.relation_features for row in rows))
        temporal_batch = torch.stack(tuple(row.temporal_features for row in rows))
        base_batch = torch.cat(tuple(row.base_logit.reshape(1) for row in rows))
        output = _core_method(self.core, "predict")(
            relation_batch,
            temporal_batch,
            base_batch,
            state=self.state,
            read_enabled=True,
        )
        logit_tensor = getattr(output, "logits", None)
        if not isinstance(logit_tensor, torch.Tensor) or logit_tensor.shape != (len(rows),):
            raise ValueError("V14 prediction logits must align with every candidate")
        if not bool(torch.isfinite(logit_tensor).all().item()):
            raise ValueError("V14 prediction is non-finite")
        logits = [float(value) for value in logit_tensor.detach().cpu().tolist()]
        selected_index = max(range(len(rows)), key=lambda index: logits[index])
        selected = rows[selected_index]
        # Preserve the established byte-exact pending/replay proof.  Batched
        # matrix kernels can differ by a few low bits from a single-row replay,
        # so rank all candidates in one batch, then retain one exact single-row
        # output for the only candidate that may later receive feedback.
        selected_output = _core_method(self.core, "predict")(
            selected.relation_features.unsqueeze(0),
            selected.temporal_features.unsqueeze(0),
            selected.base_logit.reshape(1),
            state=self.state,
            read_enabled=True,
        )
        selected_logit_tensor = getattr(selected_output, "logits", None)
        if (
            not isinstance(selected_logit_tensor, torch.Tensor)
            or selected_logit_tensor.shape != (1,)
            or not bool(torch.isfinite(selected_logit_tensor).all().item())
        ):
            raise ValueError("selected V14 prediction must expose one finite logit")
        logits[selected_index] = float(selected_logit_tensor.detach().cpu().item())
        if max(range(len(rows)), key=lambda index: logits[index]) != selected_index:
            raise ValueError("batched and exact selected-row predictions disagree on the winner")
        public_logits = tuple(logits)
        parent_digest = self.state_digest()
        material = json.dumps(
            {
                "checkpoint": self.checkpoint_identity,
                "parent": parent_digest,
                "request": request,
                "selected_index": selected_index,
                "selected_trace": selected.trace,
                "step": int(getattr(self.state, "step")),
                "task_id": task_id,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        evidence_ref = "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()
        decision = AbilityDecision(
            task_id=task_id,
            request=request,
            proposals=tuple(row.trace for row in rows),
            selected_index=selected_index,
            selected_trace=selected.trace,
            logits=public_logits,
            evidence_ref=evidence_ref,
            supporting_evidence_refs=selected.supporting_evidence_refs,
            parent_state_digest=parent_digest,
        )
        self._validate_pending_output(selected_output, decision)
        self._pending = _PendingDecision(decision, selected_output)
        return decision

    def select_prospective(
        self,
        *,
        task_id: str,
        request: str,
        candidates: Sequence[EncodedProcedureCandidate],
        recalled_refs: Sequence[str],
        context_refs: Sequence[str],
        eligibility_rows: Sequence[Sequence[int]],
        resource_budget: ProspectiveResourceBudget,
    ) -> AbilityDecision:
        """Select once while retaining the complete same-call prospective view."""

        if self._pending is not None:
            raise RuntimeError("an ability decision is already awaiting outcome")
        task_id = _text(task_id, "task_id", maximum=256)
        request = _text(request, "request", maximum=_MAXIMUM_TASK_CHARS)
        rows = tuple(candidates)
        if not _MINIMUM_CANDIDATES <= len(rows) <= _MAXIMUM_CANDIDATES:
            raise ValueError("2 through 64 encoded procedure candidates are required")
        if any(type(row) is not EncodedProcedureCandidate for row in rows):
            raise TypeError("prospective candidates must be EncodedProcedureCandidate values")
        if len({row.trace for row in rows}) != len(rows):
            raise ValueError("encoded procedure candidates must be text-distinct")
        recalled = tuple(recalled_refs)
        if (
            not 1 <= len(recalled) <= 256
            or len(set(recalled)) != len(recalled)
        ):
            raise ValueError("prospective recall must contain 1 through 256 unique refs")
        for value in recalled:
            _sha256_ref(value, "prospective recalled reference")
        recalled_set = set(recalled)
        supporting = tuple(row.supporting_evidence_refs for row in rows)
        for refs in supporting:
            if not refs or refs != tuple(sorted(refs)):
                raise ValueError("every prospective candidate requires canonical support")
            for value in refs:
                _sha256_ref(value, "prospective candidate evidence")
            if not set(refs).issubset(recalled_set):
                raise ValueError("prospective candidate cited evidence outside recall")
        contexts = tuple(context_refs)
        if len(contexts) != 1:
            raise ValueError("the first prospective lane requires one context ref")
        _sha256_ref(contexts[0], "prospective context reference")
        eligibility = tuple(tuple(row) for row in eligibility_rows)
        if len(eligibility) != len(rows) or any(row != (0,) for row in eligibility):
            raise ValueError("the first prospective lane requires eligibility (0,)")
        if (
            type(resource_budget) is not ProspectiveResourceBudget
            or resource_budget.world_reads != 1
            or resource_budget.branches != len(rows)
            or resource_budget.recurrent_steps != 1
        ):
            raise ValueError("the first prospective lane requires its exact resource budget")

        relation_width = int(getattr(self.core, "relation_width", -1))
        temporal_width = int(getattr(self.core, "temporal_width", -1))
        for row in rows:
            if row.relation_features.shape != (relation_width,):
                raise ValueError("candidate relation width does not match prospective core")
            if row.temporal_features.shape != (temporal_width,):
                raise ValueError("candidate temporal width does not match prospective core")
        relation_batch = (
            torch.stack(tuple(row.relation_features for row in rows))
            .detach()
            .contiguous()
        )
        temporal_batch = (
            torch.stack(tuple(row.temporal_features for row in rows))
            .detach()
            .contiguous()
        )
        base_batch = (
            torch.cat(tuple(row.base_logit.reshape(1) for row in rows))
            .detach()
            .contiguous()
        )
        prospective_read_enabled = not self.prospective_lesion
        with torch.no_grad():
            output, prospective = _core_method(
                self.core, "predict_with_prospective"
            )(
                relation_batch,
                temporal_batch,
                base_batch,
                state=self.state,
                read_enabled=True,
                prospective_read_enabled=prospective_read_enabled,
            )
        logit_tensor = getattr(output, "logits", None)
        if (
            not isinstance(logit_tensor, torch.Tensor)
            or logit_tensor.shape != (len(rows),)
            or not bool(torch.isfinite(logit_tensor).all().item())
        ):
            raise ValueError("prospective prediction logits must align and be finite")
        logits = [float(value) for value in logit_tensor.detach().cpu().tolist()]
        selected_index = max(range(len(rows)), key=lambda index: logits[index])
        selected = rows[selected_index]
        with torch.no_grad():
            selected_output, selected_prospective = _core_method(
                self.core, "predict_with_prospective"
            )(
                relation_batch[selected_index : selected_index + 1],
                temporal_batch[selected_index : selected_index + 1],
                base_batch[selected_index : selected_index + 1],
                state=self.state,
                read_enabled=True,
                prospective_read_enabled=prospective_read_enabled,
            )
        selected_logit = getattr(selected_output, "logits", None)
        if (
            not isinstance(selected_logit, torch.Tensor)
            or selected_logit.shape != (1,)
            or not bool(torch.isfinite(selected_logit).all().item())
        ):
            raise ValueError("selected prospective prediction must expose one finite logit")
        logits[selected_index] = float(selected_logit.detach().cpu().item())
        if max(range(len(rows)), key=lambda index: logits[index]) != selected_index:
            raise ValueError("batched and exact prospective predictions disagree on winner")
        full_prospective = _replace_prospective_row(
            prospective,
            selected_prospective,
            selected_index,
        )
        parent_digest = self.state_digest()
        material = json.dumps(
            {
                "checkpoint": self.checkpoint_identity,
                "parent": parent_digest,
                "request": request,
                "selected_index": selected_index,
                "selected_trace": selected.trace,
                "step": int(getattr(self.state, "step")),
                "task_id": task_id,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        evidence_ref = "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()
        decision = AbilityDecision(
            task_id=task_id,
            request=request,
            proposals=tuple(row.trace for row in rows),
            selected_index=selected_index,
            selected_trace=selected.trace,
            logits=tuple(logits),
            evidence_ref=evidence_ref,
            supporting_evidence_refs=selected.supporting_evidence_refs,
            parent_state_digest=parent_digest,
        )
        selected_clone = _detached_output_clone(selected_output)
        component = self.component_state_integrity()
        pending_material = PendingProspectiveMaterial(
            decision=decision,
            selected_output=selected_clone,
            prospective_output=_detached_prospective_clone(full_prospective),
            relation_features=relation_batch.clone(),
            temporal_features=temporal_batch.clone(),
            base_logits=base_batch.clone(),
            candidate_traces=decision.proposals,
            candidate_supporting_evidence_refs=supporting,
            recalled_refs=recalled,
            context_refs=contexts,
            eligibility_rows=eligibility,
            resource_budget=resource_budget,
            parent_competence_digest=parent_digest,
            checkpoint_ref=component.checkpoint_ref,
            config_ref=component.config_ref,
            component_state=component,
            batch_ref=None,
            prospective_read_enabled=prospective_read_enabled,
        )
        self._validate_pending_prospective_material(
            pending_material,
            require_batch=False,
        )
        self._pending = _PendingDecision(
            decision,
            selected_clone,
            pending_material,
        )
        return decision

    def apply_outcome(self, outcome: OutcomeContext) -> None:
        if self._pending is None:
            raise RuntimeError("objective outcome has no pending ability decision")
        pending = self._pending
        if pending.prospective is not None:
            self._validate_pending_prospective_material(
                pending.prospective,
                require_batch=True,
            )
        if outcome.task_id != pending.public.task_id or outcome.request != pending.public.request:
            raise ValueError("objective outcome does not match the pending ability decision")
        if pending.public.evidence_ref in self.tombstoned_evidence_refs:
            raise ValueError("pending ability evidence has been tombstoned")
        value = 1.0 if outcome.disposition == "success" else -1.0
        reference = getattr(pending.output, "logits")
        outcomes = torch.tensor([value], device=reference.device, dtype=reference.dtype)
        parent = self.state
        try:
            state, event = _core_method(self.core, "apply_feedback")(
                parent,
                pending.output,
                outcomes,
                evidence_refs=pending.public.evidence_ref,
                detach_state=True,
            )
            _core_method(self.core, "validate_state")(state)
            if pending.public.evidence_ref not in _event_refs(event):
                raise ValueError("V14 event does not bind the pending evidence reference")
        except Exception:
            self.state = parent
            raise
        self.state = state
        self.events = (*self.events, event)
        self._last_event = event
        self._pending = None

    def capture_state(self) -> bytes:
        snapshot = _snapshot_mapping(_core_method(self.core, "capture_state")(self.state))
        event_records = []
        for event in self.events:
            to_record = getattr(event, "to_record", None)
            if not callable(to_record):
                raise TypeError("V14 event does not provide to_record")
            event_records.append(_jsonable(to_record()))
        payload = {
            "version": _SNAPSHOT_VERSION,
            "checkpoint_identity": self.checkpoint_identity,
            "core_snapshot": snapshot,
            "events": event_records,
            "tombstoned_evidence_refs": sorted(self.tombstoned_evidence_refs),
        }
        stream = io.BytesIO()
        torch.save(payload, stream)
        return stream.getvalue()

    def restore_state(self, state: bytes) -> None:
        if not isinstance(state, bytes) or not state:
            raise ValueError("competence snapshot must be non-empty bytes")
        try:
            payload = torch.load(io.BytesIO(state), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError("competence snapshot cannot be decoded") from exc
        if not isinstance(payload, Mapping) or set(payload) != {
            "version",
            "checkpoint_identity",
            "core_snapshot",
            "events",
            "tombstoned_evidence_refs",
        }:
            raise ValueError("competence snapshot envelope is malformed")
        if payload["version"] != _SNAPSHOT_VERSION:
            raise ValueError("competence snapshot version is unsupported")
        if payload["checkpoint_identity"] != self.checkpoint_identity:
            raise ValueError("competence snapshot belongs to another checkpoint")
        raw_snapshot = payload["core_snapshot"]
        if not isinstance(raw_snapshot, Mapping):
            raise ValueError("competence snapshot core state is malformed")
        try:
            snapshot = self._snapshot_type(**dict(raw_snapshot))
            candidate_state = _core_method(self.core, "restore_state")(snapshot)
            _core_method(self.core, "validate_state")(candidate_state)
            events = tuple(
                _event_from_record(record, core=self.core)
                for record in payload["events"]
            )
            tombstones = frozenset(payload["tombstoned_evidence_refs"])
        except Exception as exc:
            raise ValueError("competence snapshot content is invalid") from exc
        if any(type(item) is not str or not item for item in tombstones):
            raise ValueError("competence snapshot tombstones are invalid")
        if any(tombstones.intersection(_event_refs(event)) for event in events):
            raise ValueError("competence snapshot retains tombstoned evidence")
        replayed, _ = _core_method(self.core, "replay")(events, detach_state=True)
        if _core_method(self.core, "state_digest")(replayed) != _core_method(
            self.core, "state_digest"
        )(candidate_state):
            raise ValueError("competence snapshot state does not match its event lineage")
        self.state = candidate_state
        self.events = events
        self.tombstoned_evidence_refs = tombstones
        self._pending = None
        self._last_event = None

    def state_digest(self) -> str:
        digest = hashlib.sha256(b"project-angler.durable-ability-runtime.v1\x00")
        _update_digest(digest, self.checkpoint_identity)
        _update_digest(digest, _snapshot_mapping(_core_method(self.core, "capture_state")(self.state)))
        for event in self.events:
            _update_digest(digest, _jsonable(event.to_record()))
        _update_digest(digest, sorted(self.tombstoned_evidence_refs))
        return "sha256:" + digest.hexdigest()

    def zero(self) -> None:
        self.state = _core_method(self.core, "zero_state_like")(self.state)
        _core_method(self.core, "validate_state")(self.state)
        self.events = ()
        self._pending = None
        self._last_event = None

    def invalidate_evidence(self, evidence_refs: Sequence[str]) -> str:
        refs = frozenset(evidence_refs)
        if not refs or any(type(item) is not str or not item for item in refs):
            raise ValueError("evidence_refs must contain non-empty strings")
        tombstones = self.tombstoned_evidence_refs.union(refs)
        retained = tuple(
            event for event in self.events if not refs.intersection(_event_refs(event))
        )
        state, _ = _core_method(self.core, "replay")(retained, detach_state=True)
        _core_method(self.core, "validate_state")(state)
        self.state = state
        self.events = retained
        self.tombstoned_evidence_refs = tombstones
        self._pending = None
        self._last_event = None
        return self.state_digest()

    def rebuild_from_events(self, events: Sequence[object]) -> str:
        accepted = tuple(events)
        if any(
            self.tombstoned_evidence_refs.intersection(_event_refs(event))
            for event in accepted
        ):
            raise ValueError("rebuild includes tombstoned evidence")
        state, _ = _core_method(self.core, "replay")(accepted, detach_state=True)
        _core_method(self.core, "validate_state")(state)
        self.state = state
        self.events = accepted
        self._pending = None
        self._last_event = None
        return self.state_digest()


def _event_from_record(record: object, *, core: object | None = None) -> object:
    parser = getattr(core, "event_from_record", None)
    if callable(parser):
        return parser(record)

    try:
        from angler.reasoning.structure_keyed_credit_memory import (
            StructureKeyedCreditEvent,
        )
    except ImportError as exc:
        raise RuntimeError("V14 structure-keyed credit core is unavailable") from exc
    return StructureKeyedCreditEvent.from_record(record)


class DurableAbilityBridge:
    """Recall, select, execute, and durably commit one bounded procedure turn."""

    def __init__(
        self,
        *,
        text_adapter: FrozenProcedureTextAdapter,
        relation_adapter: ProcedureRelationAdapter,
        learner: DurableAbilityLearner,
        memory: SituatedMemory,
        journal: ConversationJournal,
        state_store: AtomicCompetenceStore,
        recall_limit: int = 12,
        proposal_count: int = 4,
    ) -> None:
        if not isinstance(text_adapter, FrozenProcedureTextAdapter):
            raise TypeError("text_adapter does not implement the frozen text contract")
        if not isinstance(relation_adapter, ProcedureRelationAdapter):
            raise TypeError("relation_adapter does not implement the relation contract")
        if not isinstance(memory, SituatedMemory):
            raise TypeError("memory must be SituatedMemory")
        if type(recall_limit) is not int or recall_limit < 1:
            raise ValueError("recall_limit must be a positive integer")
        if (
            type(proposal_count) is not int
            or not _MINIMUM_CANDIDATES <= proposal_count <= _MAXIMUM_CANDIDATES
        ):
            raise ValueError("proposal_count must be an integer from 2 through 64")
        self.text_adapter = text_adapter
        self.relation_adapter = relation_adapter
        self.learner = learner
        self.memory = memory
        self.journal = journal
        self.state_store = state_store
        self.recall_limit = recall_limit
        self.proposal_count = proposal_count

    async def begin_turn(
        self,
        task_id: str,
        task: str,
        *,
        world_time: int | None = None,
    ) -> DurableAbilityTurn:
        task_id = _text(task_id, "task_id", maximum=256)
        task = _text(task, "task", maximum=_MAXIMUM_TASK_CHARS)
        recall = await self.memory.recall(
            task, limit=self.recall_limit, world_time=world_time
        )
        raw = self.text_adapter.propose_procedure_traces(
            task, tuple(item.text for item in recall.items), count=self.proposal_count
        )
        proposals = tuple(
            _text(item, "procedure proposal", maximum=_MAXIMUM_PROCEDURE_CHARS)
            for item in raw
        )
        if (
            len(proposals) != self.proposal_count
            or len(set(proposals)) != self.proposal_count
        ):
            raise ValueError(
                "the frozen model must return the requested number of distinct traces"
            )
        encoded = tuple(self.relation_adapter.encode_candidates(task, proposals, recall))
        if (
            len(encoded) != self.proposal_count
            or tuple(row.trace for row in encoded) != proposals
        ):
            raise ValueError("relation adapter output does not align with proposals")
        recalled_refs = {item.artifact_ref for item in recall.items}
        if any(
            not set(row.supporting_evidence_refs).issubset(recalled_refs) for row in encoded
        ):
            raise ValueError("relation adapter cited evidence outside the validated recall")
        decision = self.learner.select(
            task_id=task_id,
            request=task,
            candidates=encoded,
        )
        return DurableAbilityTurn(decision, recall)

    def execute_turn(
        self,
        turn: DurableAbilityTurn,
        *,
        observations: Sequence[str] = (),
    ) -> CompletedAbilityTurn:
        if self.learner.pending_decision != turn.decision:
            raise ValueError("turn is not the learner's pending decision")
        bounded_observations = tuple(
            _text(item, "observation", maximum=4_096) for item in observations
        )
        response = self.text_adapter.execute_with_procedure(
            turn.decision.request,
            turn.decision.selected_trace,
            bounded_observations,
        )
        response = _text(response, "model response", maximum=_MAXIMUM_RESPONSE_CHARS)
        return CompletedAbilityTurn(turn, response)

    async def record_outcome(
        self,
        completed: CompletedAbilityTurn,
        *,
        outcome: Literal["success", "failure"],
        source_ref: str,
        feedback_text: str,
        world_valid_from: int | None = None,
        world_valid_until: int | None = None,
    ) -> JournalRecord:
        if outcome not in ("success", "failure"):
            raise ValueError("outcome must be objective success or failure")
        decision = completed.turn.decision
        if self.learner.pending_decision != decision:
            raise ValueError("completed turn is not awaiting objective feedback")
        source_ref = _text(source_ref, "source_ref", maximum=512)
        feedback_text = _text(feedback_text, "feedback_text", maximum=2_048)
        parent = self.learner.capture_state()
        try:
            self.learner.apply_outcome(
                OutcomeContext(
                    decision.task_id,
                    decision.request,
                    (),
                    outcome,
                    source_ref,
                )
            )
            event = self.learner.last_event
            if event is None:
                raise RuntimeError("V14 learner did not produce a replay event")
            event_context = _event_context(event)
            projection = MemoryProjection.from_mapping(
                artifact_ref=decision.evidence_ref,
                text=(
                    f"Objective software attempt. Task: {decision.request} "
                    f"Procedure: {decision.selected_trace}. Response: {completed.response}. "
                    f"Objective outcome: {outcome}. Feedback: {feedback_text}"
                )[:4_096].strip(),
                source_ref=source_ref,
                context={
                    **event_context,
                    "ability_state_parent": decision.parent_state_digest,
                    "feedback_origin": "objective_environment",
                    "outcome": outcome,
                    _PROCEDURE_CONTEXT_KEY: decision.selected_trace,
                },
            )
            record = JournalRecord(
                projection,
                world_valid_from=world_valid_from,
                world_valid_until=world_valid_until,
                current_regime=outcome == "success",
            )
            self.journal.append(record)
            await self.memory.backend.remember(encode_projection(projection))
            self.memory.origin = self.journal.build_origin()
            self.state_store.save(self.learner.capture_state())
        except Exception:
            self.learner.restore_state(parent)
            raise
        return record

    def rebuild_ability_from_journal(self) -> str:
        events = []
        for record in self.journal.records():
            artifact_ref = record.projection.artifact_ref
            if artifact_ref in self.learner.tombstoned_evidence_refs:
                continue
            context = dict(record.projection.context)
            if _EVENT_CONTEXT_COUNT_KEY not in context:
                continue
            event = _event_from_context(context, core=self.learner.core)
            if artifact_ref not in _event_refs(event):
                raise ValueError("journal ability event does not bind its artifact reference")
            events.append(event)
        return self.learner.rebuild_from_events(events)

    async def rebuild_projection(self) -> None:
        """Rebuild disposable recall while excluding durable tombstones."""

        records = tuple(
            record
            for record in self.journal.records()
            if record.projection.artifact_ref
            not in self.learner.tombstoned_evidence_refs
        )
        await self.memory.forget_projection()
        documents = tuple(encode_projection(record.projection) for record in records)
        bulk = getattr(self.memory.backend, "remember_many", None)
        if documents and callable(bulk):
            await bulk(documents)
        else:
            for document in documents:
                await self.memory.backend.remember(document)
        origin = MovingOriginIndex()
        for record in records:
            origin.append(
                record.projection.artifact_ref,
                projection_id(record.projection),
                world_valid_from=record.world_valid_from,
                world_valid_until=record.world_valid_until,
            )
        current = next((record for record in reversed(records) if record.current_regime), None)
        if current is not None:
            target = current.projection.artifact_ref
            material = f"{Path(self.journal.path).resolve()}:{target}:current_regime"
            origin.designate_landmark(
                "current_regime",
                target_event_ref=target,
                designation_event_ref="sha256:"
                + hashlib.sha256(material.encode()).hexdigest(),
                projection_id="sha256:"
                + hashlib.sha256((material + ":projection").encode()).hexdigest(),
            )
        self.memory.origin = origin


__all__ = [
    "AbilityDecision",
    "CompletedAbilityTurn",
    "DurableAbilityBridge",
    "DurableAbilityLearner",
    "DurableAbilityTurn",
    "EncodedProcedureCandidate",
    "FrozenProcedureTextAdapter",
    "PendingProspectiveMaterial",
    "ProcedureRelationAdapter",
]

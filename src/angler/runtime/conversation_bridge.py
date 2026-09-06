"""Local bridge from situated recall and learned procedures to frozen Qwen.

The bridge exposes what Angler planned.  It never executes a plan, judges a
response, invents feedback, selects a checkpoint from query identity, or
contains task-specific solution logic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from angler.memory import MemoryProjection, RecallBatch, SituatedMemory
from angler.reasoning.candidate_procedure_decoder import (
    CandidateProcedureDecode,
    CandidateProcedureDecoder,
)
from angler.reasoning.scalable_procedural_core import (
    PlasticProcedureState,
    ProceduralCoreConfig,
    ScalableProceduralCore,
    plastic_state_digest,
)
from angler.reasoning.situated_reader import SituatedFeatureSpec, encode_situated_features

from .qwen_peft import foundation_tensor_digest
from .situated_qwen import LocalQwenIO


_CHECKPOINT_FIELDS = frozenset(
    {
        "identity",
        "foundation_digest",
        "core_config",
        "core_state_dict",
        "decoder_config",
        "decoder_state_dict",
        "plastic_state",
    }
)


def _text(value: str, label: str, *, maximum: int) -> str:
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} must be non-empty text of at most {maximum} characters")
    if value != value.strip():
        raise ValueError(f"{label} must not have surrounding whitespace")
    return value.strip()


@dataclass(frozen=True, slots=True)
class TaskLocalAction:
    candidate_ref: str
    description: str

    def __post_init__(self) -> None:
        _text(self.candidate_ref, "candidate_ref", maximum=128)
        _text(self.description, "candidate description", maximum=2_048)


@dataclass(frozen=True, slots=True)
class AnglerPlanStep:
    step: int
    candidate_index: int
    candidate_ref: str
    description: str


@dataclass(frozen=True, slots=True)
class AnglerPlan:
    steps: tuple[AnglerPlanStep, ...]
    stopped: bool
    raw_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EvidenceAttribution:
    artifact_ref: str
    weight: float
    acquired_ordinal: int
    age: int
    world_valid_at_query: bool | None


@dataclass(frozen=True, slots=True)
class LocalConversationTurn:
    query: str
    candidates: tuple[TaskLocalAction, ...]
    recall: RecallBatch
    evidence_attribution: tuple[EvidenceAttribution, ...]
    plan: AnglerPlan
    prompt: str
    response: str
    checkpoint_identity: str
    plastic_state_identity: str


@dataclass(frozen=True, slots=True)
class LoadedConversationCheckpoint:
    identity: str
    foundation_digest: str
    core: ScalableProceduralCore
    decoder: CandidateProcedureDecoder
    plastic_state: PlasticProcedureState
    plastic_state_identity: str


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _state_dict(value: Any, label: str) -> Mapping[str, torch.Tensor]:
    mapping = _mapping(value, label)
    if not mapping or any(type(name) is not str or not isinstance(tensor, torch.Tensor) for name, tensor in mapping.items()):
        raise ValueError(f"{label} must contain named tensors")
    return mapping


def load_conversation_checkpoint(
    checkpoint_path: str | Path,
    qwen: LocalQwenIO,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> LoadedConversationCheckpoint:
    """Load one caller-selected supported checkpoint without mutating it."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"conversation checkpoint does not exist: {path}")
    if any(parameter.requires_grad for parameter in qwen.model.parameters()):
        raise ValueError("LocalQwenIO foundation parameters must remain frozen")
    sealed = torch.load(path, map_location="cpu", weights_only=True)
    mapping = _mapping(sealed, "checkpoint")
    missing = _CHECKPOINT_FIELDS.difference(mapping)
    if missing:
        raise ValueError(f"checkpoint is missing required fields: {', '.join(sorted(missing))}")
    identity = _text(mapping["identity"], "checkpoint identity", maximum=512)
    expected_foundation = _text(
        mapping["foundation_digest"],
        "foundation digest",
        maximum=128,
    )
    observed_foundation = foundation_tensor_digest(qwen.model)
    if observed_foundation != expected_foundation:
        raise ValueError("checkpoint binds a different frozen Qwen foundation")

    try:
        core_config = ProceduralCoreConfig(**dict(_mapping(mapping["core_config"], "core_config")))
        decoder_config = dict(_mapping(mapping["decoder_config"], "decoder_config"))
        decoder = CandidateProcedureDecoder(**decoder_config)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint component configuration is unsupported") from exc
    if decoder.content_width != core_config.content_width:
        raise ValueError("decoder content width does not match the procedural core")
    if decoder.procedure_width != core_config.model_width:
        raise ValueError("decoder procedure width does not match the procedural core")

    core = ScalableProceduralCore(core_config)
    try:
        core.load_state_dict(_state_dict(mapping["core_state_dict"], "core_state_dict"), strict=True)
        decoder.load_state_dict(
            _state_dict(mapping["decoder_state_dict"], "decoder_state_dict"),
            strict=True,
        )
    except RuntimeError as exc:
        raise ValueError("checkpoint tensor topology is incompatible") from exc
    target_device = torch.device(device)
    core.to(device=target_device, dtype=dtype)
    decoder.to(device=target_device, dtype=dtype)
    core.requires_grad_(False).eval()
    decoder.requires_grad_(False).eval()

    state_mapping = _mapping(mapping["plastic_state"], "plastic_state")
    if set(state_mapping) != {"keys", "values", "strengths", "step"}:
        raise ValueError("plastic_state fields are invalid")
    reference = next(core.parameters())
    tensors = {}
    for name in ("keys", "values", "strengths"):
        value = state_mapping[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"plastic_state {name} must be a tensor")
        tensors[name] = value.detach().to(device=reference.device, dtype=reference.dtype)
    step = state_mapping["step"]
    state = PlasticProcedureState(step=step, **tensors)
    expected_shape = (core_config.plastic_slots, core_config.model_width)
    if state.keys.shape != expected_shape or state.values.shape != expected_shape:
        raise ValueError("plastic-state key/value topology is incompatible")
    if state.strengths.shape != (core_config.plastic_slots,):
        raise ValueError("plastic-state strength topology is incompatible")
    if type(state.step) is not int or state.step < 0:
        raise ValueError("plastic-state step must be a non-negative integer")
    if any(not bool(torch.isfinite(value).all().item()) for value in tensors.values()):
        raise ValueError("plastic-state tensors must be finite")
    state = state.detached_clone()
    return LoadedConversationCheckpoint(
        identity=identity,
        foundation_digest=expected_foundation,
        core=core,
        decoder=decoder,
        plastic_state=state,
        plastic_state_identity=plastic_state_digest(state),
    )


class LocalConversationBridge:
    """Recall, plan, and publish one local turn without executing the plan."""

    def __init__(
        self,
        checkpoint: LoadedConversationCheckpoint,
        memory: SituatedMemory,
        qwen: LocalQwenIO,
        *,
        feature_spec: SituatedFeatureSpec,
        _foundation_already_verified: bool = False,
    ) -> None:
        if checkpoint.core.config.temporal_width != feature_spec.width:
            raise ValueError("Moving Origin feature width does not match the checkpoint")
        if (
            not _foundation_already_verified
            and foundation_tensor_digest(qwen.model) != checkpoint.foundation_digest
        ):
            raise ValueError("runtime Qwen does not match the loaded checkpoint")
        self.checkpoint = checkpoint
        self.memory = memory
        self.qwen = qwen
        self.feature_spec = feature_spec

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        memory: SituatedMemory,
        qwen: LocalQwenIO,
        *,
        feature_spec: SituatedFeatureSpec,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> "LocalConversationBridge":
        checkpoint = load_conversation_checkpoint(
            checkpoint_path,
            qwen,
            device=device,
            dtype=dtype,
        )
        return cls(
            checkpoint,
            memory,
            qwen,
            feature_spec=feature_spec,
            _foundation_already_verified=True,
        )

    async def answer(
        self,
        query: str,
        candidates: Sequence[TaskLocalAction],
        *,
        recall_limit: int = 12,
        world_time: int | None = None,
        response_constraint: str | None = None,
    ) -> LocalConversationTurn:
        query = _text(query, "query", maximum=16_384)
        if type(recall_limit) is not int or recall_limit < 1:
            raise ValueError("recall_limit must be a positive integer")
        actions = tuple(candidates)
        if not actions or any(not isinstance(action, TaskLocalAction) for action in actions):
            raise ValueError("candidates must be a non-empty sequence of TaskLocalAction values")
        if len(actions) > self.checkpoint.decoder.maximum_actions:
            raise ValueError("task-local candidates exceed the checkpoint decoder capacity")
        refs = tuple(action.candidate_ref for action in actions)
        if len(set(refs)) != len(refs):
            raise ValueError("task-local candidate references must be unique")
        if response_constraint is not None:
            response_constraint = _text(
                response_constraint,
                "response_constraint",
                maximum=4_096,
            )

        recall = await self.memory.recall(
            query,
            limit=recall_limit,
            world_time=world_time,
        )
        if not recall.items:
            raise LookupError("no validated situated evidence was recalled")
        texts = (
            query,
            *(item.text for item in recall.items),
            *(action.description for action in actions),
        )
        encoded = self.qwen.embed(texts)
        content_width = self.checkpoint.core.config.content_width
        if (
            not isinstance(encoded, torch.Tensor)
            or encoded.ndim != 2
            or encoded.shape != (len(texts), content_width)
            or not bool(torch.isfinite(encoded).all().item())
        ):
            raise ValueError("LocalQwenIO embeddings do not match checkpoint content width")

        reference = next(self.checkpoint.core.parameters())
        encoded = encoded.detach().to(device=reference.device, dtype=reference.dtype)
        recall_count = len(recall.items)
        query_features = encoded[:1]
        recall_features = encoded[1 : 1 + recall_count].unsqueeze(0)
        action_rows = encoded[1 + recall_count :]
        temporal, recall_mask = encode_situated_features(
            (recall,),
            now=(self.memory.origin.now,),
            spec=self.feature_spec,
            device=reference.device,
            dtype=reference.dtype,
        )
        with torch.inference_mode():
            core_output = self.checkpoint.core(
                query_features,
                recall_features,
                temporal,
                recall_mask,
                plastic_state=self.checkpoint.plastic_state,
            )
            padded_actions = torch.zeros(
                (1, self.checkpoint.decoder.maximum_actions, content_width),
                device=reference.device,
                dtype=reference.dtype,
            )
            padded_actions[0, : len(actions)] = action_rows
            action_mask = torch.zeros(
                (1, self.checkpoint.decoder.maximum_actions),
                device=reference.device,
                dtype=torch.bool,
            )
            action_mask[0, : len(actions)] = True
            decoded = self.checkpoint.decoder(
                core_output.procedure_slots,
                padded_actions,
                action_mask,
            )

        plan = _explicit_plan(decoded, actions, self.checkpoint.decoder.stop_index)
        weights = core_output.candidate_weights[0].detach().cpu().tolist()
        attribution = tuple(
            EvidenceAttribution(
                artifact_ref=item.artifact_ref,
                weight=float(weights[index]),
                acquired_ordinal=item.acquired_ordinal,
                age=item.age,
                world_valid_at_query=item.world_valid_at_query,
            )
            for index, item in enumerate(recall.items)
        )
        prompt = build_conversation_prompt(
            query,
            recall=recall,
            attribution=attribution,
            plan=plan,
            response_constraint=response_constraint,
        )
        response = self.qwen.generate(prompt)
        if type(response) is not str or not response.strip():
            raise RuntimeError("LocalQwenIO returned an empty response")
        return LocalConversationTurn(
            query=query,
            candidates=actions,
            recall=recall,
            evidence_attribution=attribution,
            plan=plan,
            prompt=prompt,
            response=response.strip(),
            checkpoint_identity=self.checkpoint.identity,
            plastic_state_identity=self.checkpoint.plastic_state_identity,
        )

    def package_feedback(
        self,
        turn: LocalConversationTurn,
        *,
        artifact_ref: str,
        source_ref: str,
        outcome: Literal["success", "failure"],
        feedback_text: str,
        family: str,
    ) -> MemoryProjection:
        """Package caller-supplied feedback without evaluating the response."""

        if outcome not in ("success", "failure"):
            raise ValueError("outcome must be externally supplied as success or failure")
        feedback_text = _text(feedback_text, "feedback_text", maximum=2_048)
        family = _text(family, "family", maximum=128)
        plan_text = " -> ".join(step.description for step in turn.plan.steps) or "STOP"
        text = (
            f"Prior {family} conversation. Query: {turn.query} "
            f"Angler plan: {plan_text}. Response: {turn.response}. "
            f"External outcome: {outcome}. External feedback: {feedback_text}"
        )
        return MemoryProjection.from_mapping(
            artifact_ref=artifact_ref,
            text=text,
            source_ref=source_ref,
            context={
                "checkpoint_identity": turn.checkpoint_identity,
                "family": family,
                "feedback_origin": "external",
                "outcome": outcome,
            },
        )

    async def record_feedback(
        self,
        turn: LocalConversationTurn,
        *,
        artifact_ref: str,
        source_ref: str,
        outcome: Literal["success", "failure"],
        feedback_text: str,
        family: str,
        world_valid_from: int | None = None,
        world_valid_until: int | None = None,
    ) -> int:
        projection = self.package_feedback(
            turn,
            artifact_ref=artifact_ref,
            source_ref=source_ref,
            outcome=outcome,
            feedback_text=feedback_text,
            family=family,
        )
        return await self.memory.remember(
            projection,
            world_valid_from=world_valid_from,
            world_valid_until=world_valid_until,
        )


def _explicit_plan(
    decoded: CandidateProcedureDecode,
    candidates: tuple[TaskLocalAction, ...],
    stop_index: int,
) -> AnglerPlan:
    if decoded.selected_indices.shape[0] != 1 or decoded.selected_indices.ndim != 2:
        raise RuntimeError("decoder must return one sequence for one conversation turn")
    raw = tuple(int(value) for value in decoded.selected_indices[0].detach().cpu().tolist())
    steps = []
    stopped = False
    for index in raw:
        if index == stop_index:
            stopped = True
            break
        if index < 0 or index >= len(candidates):
            raise RuntimeError("decoder selected an absent task-local candidate")
        action = candidates[index]
        steps.append(
            AnglerPlanStep(
                step=len(steps) + 1,
                candidate_index=index,
                candidate_ref=action.candidate_ref,
                description=action.description,
            )
        )
    return AnglerPlan(steps=tuple(steps), stopped=stopped, raw_indices=raw)


def build_conversation_prompt(
    query: str,
    *,
    recall: RecallBatch,
    attribution: tuple[EvidenceAttribution, ...],
    plan: AnglerPlan,
    response_constraint: str | None = None,
) -> str:
    """Render the learned plan and its evidence as transparent Qwen context."""

    if len(recall.items) != len(attribution):
        raise ValueError("evidence attribution does not align with recall")
    evidence_lines = []
    for item, evidence in zip(recall.items, attribution, strict=True):
        validity = (
            "unknown"
            if evidence.world_valid_at_query is None
            else str(evidence.world_valid_at_query).lower()
        )
        evidence_lines.append(
            f"- [{item.artifact_ref}] weight={evidence.weight:.6f}; "
            f"acquired={evidence.acquired_ordinal}; age={evidence.age}; "
            f"world_valid={validity}: {item.text}"
        )
    plan_lines = [
        f"{step.step}. [{step.candidate_ref}] {step.description}" for step in plan.steps
    ]
    if not plan_lines:
        plan_lines.append("0. STOP; no task-local action was selected.")
    sections = [
        "Answer the user's current request using your frozen language knowledge and the visible Angler context below.",
        "Angler's plan is advisory text only. Do not claim it was executed, and do not hide uncertainty.",
        "Validated situated evidence with learned attribution:\n" + "\n".join(evidence_lines),
        "Explicit Angler plan over this turn's task-local candidates:\n" + "\n".join(plan_lines),
        "Current user request:\n" + query,
    ]
    if response_constraint is not None:
        sections.append("Response requirement:\n" + response_constraint)
    return "\n\n".join(sections)


__all__ = [
    "AnglerPlan",
    "AnglerPlanStep",
    "EvidenceAttribution",
    "LoadedConversationCheckpoint",
    "LocalConversationBridge",
    "LocalConversationTurn",
    "TaskLocalAction",
    "build_conversation_prompt",
    "load_conversation_checkpoint",
]

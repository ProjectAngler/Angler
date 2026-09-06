"""Bounded autonomous apprenticeship over objective software-task outcomes.

The controller is deliberately solution-agnostic.  A model proposes structured
actions; the sandbox validates and executes only task-authorized operations;
and a task-fixed verifier supplies outcome feedback.  Verifier diagnostics and
procedure-content retention are explicit task-contract capabilities and remain
disabled by default.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import io
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Literal, Protocol

import torch

from angler.memory import MemoryProjection, SituatedMemory, encode_projection
from angler.reasoning.action_trace_matching import ActionTraceProceduralCore
from angler.reasoning.scalable_procedural_core import (
    PlasticProcedureState,
    ProceduralCoreConfig,
    plastic_state_digest,
)
from angler.reasoning.situated_reader import SituatedFeatureSpec, encode_situated_features

from .situated_conversation import ConversationJournal, JournalRecord
from .situated_qwen import LocalQwenIO
from .qwen_peft import foundation_tensor_digest


ActionKind = Literal["read", "write", "verify", "finish", "ask"]
RunStatus = Literal["SUCCEEDED", "EXHAUSTED", "CLARIFICATION_REQUIRED", "INVALID"]
VerifierDisposition = Literal["PASS", "FAIL", "ERROR"]


def _nonnegative_int(value: int, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: int, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _normalized_relative(value: str, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty relative text")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must remain relative to the task workspace")
    normalized = path.as_posix()
    if normalized in {"", "."}:
        raise ValueError(f"{label} cannot name the workspace root")
    return normalized


@dataclass(frozen=True, slots=True)
class AutonomousTaskContract:
    task_id: str
    request: str
    success_description: str
    workspace_root: Path
    readable_paths: tuple[str, ...]
    writable_paths: tuple[str, ...]
    verifier_command: tuple[str, ...]
    allow_mutation: bool
    max_steps: int = 24
    max_verifier_runs: int = 6
    verifier_timeout_seconds: float = 30.0
    max_file_bytes: int = 262_144
    max_observation_chars: int = 16_384
    reveal_verifier_diagnostics: bool = False
    retain_procedure_content: bool = False
    max_learning_trace_chars: int = 3_072

    def __post_init__(self) -> None:
        if type(self.task_id) is not str or not self.task_id.strip():
            raise ValueError("task_id must be non-empty text")
        if not isinstance(self.workspace_root, Path):
            raise TypeError("workspace_root must be a pathlib.Path")
        _positive_int(self.max_steps, "max_steps")
        _positive_int(self.max_verifier_runs, "max_verifier_runs")
        _positive_int(self.max_file_bytes, "max_file_bytes")
        _positive_int(self.max_observation_chars, "max_observation_chars")
        _positive_int(self.max_learning_trace_chars, "max_learning_trace_chars")
        if type(self.reveal_verifier_diagnostics) is not bool:
            raise TypeError("reveal_verifier_diagnostics must be boolean")
        if type(self.retain_procedure_content) is not bool:
            raise TypeError("retain_procedure_content must be boolean")
        if not isinstance(self.verifier_timeout_seconds, (int, float)) or self.verifier_timeout_seconds <= 0:
            raise ValueError("verifier_timeout_seconds must be positive")
        readable = tuple(_normalized_relative(item, "readable path") for item in self.readable_paths)
        writable = tuple(_normalized_relative(item, "writable path") for item in self.writable_paths)
        if len(set(readable)) != len(readable) or len(set(writable)) != len(writable):
            raise ValueError("task paths must be unique")
        if not all(type(item) is str and item for item in self.verifier_command):
            if self.verifier_command:
                raise ValueError("verifier_command must contain non-empty arguments")
        object.__setattr__(self, "readable_paths", readable)
        object.__setattr__(self, "writable_paths", writable)

    def clarification_questions(self) -> tuple[str, ...]:
        questions = []
        if not self.request.strip():
            questions.append("What exact change or outcome should I produce?")
        if not self.success_description.strip() or not self.verifier_command:
            questions.append("How can I objectively verify that the task is complete?")
        if self.writable_paths and not self.allow_mutation:
            questions.append("May I modify the declared writable files in the disposable workspace?")
        return tuple(questions)


@dataclass(frozen=True, slots=True)
class ProposedAction:
    action_id: str
    kind: ActionKind
    path: str | None = None
    content: str | None = None
    question: str | None = None


@dataclass(frozen=True, slots=True)
class ActionReceipt:
    action_id: str
    kind: ActionKind
    status: Literal["OK", "REJECTED"]
    observation: str
    mutated: bool = False


@dataclass(frozen=True, slots=True)
class VerifierReceipt:
    disposition: VerifierDisposition
    exit_code: int | None
    duration_seconds: float
    output_sha256: str
    output_chars: int
    controller_excerpt: str

    @property
    def learner_observation(self) -> str:
        if self.disposition == "PASS":
            return "Objective verifier passed."
        if self.disposition == "FAIL":
            return "Objective verifier failed; revise the procedure and try again."
        return "Verifier infrastructure error; the attempt is invalid and cannot teach the learner."

    def observation_for(self, *, reveal_diagnostics: bool, maximum: int) -> str:
        """Return bounded task feedback without changing verifier authority."""

        base = self.learner_observation
        if self.disposition != "FAIL" or not reveal_diagnostics:
            return base
        diagnostic = self.controller_excerpt.strip()
        if not diagnostic:
            return base
        return f"{base}\nBounded verifier diagnostic:\n{diagnostic[:maximum]}"


@dataclass(frozen=True, slots=True)
class OutcomeContext:
    task_id: str
    request: str
    action_trace: tuple[ProposedAction, ...]
    disposition: Literal["success", "failure"]
    verifier_ref: str


@dataclass(frozen=True, slots=True)
class AutonomousEpisode:
    task_id: str
    request: str
    action_trace: tuple[ProposedAction, ...]
    verifier: VerifierReceipt
    outcome: Literal["success", "failure"]
    state_parent: str
    retain_procedure_content: bool = False
    max_learning_trace_chars: int = 3_072


@dataclass(frozen=True, slots=True)
class AutonomousRunResult:
    status: RunStatus
    task_id: str
    receipts: tuple[ActionReceipt, ...]
    verifier_receipts: tuple[VerifierReceipt, ...]
    clarification_questions: tuple[str, ...]
    initial_state_digest: str
    final_state_digest: str


class ActionPolicy(Protocol):
    def __call__(self, prompt: str) -> str: ...


class CompetenceLearner(Protocol):
    def capture_state(self) -> bytes: ...
    def restore_state(self, state: bytes) -> None: ...
    def state_digest(self) -> str: ...
    def apply_outcome(self, outcome: OutcomeContext) -> None: ...


class EpisodeRecorder(Protocol):
    def record(self, episode: AutonomousEpisode) -> None | Awaitable[None]: ...


class AtomicCompetenceStore:
    """Atomically persist already-validated learner snapshots."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, state: bytes) -> str:
        if not isinstance(state, bytes) or not state:
            raise ValueError("competence snapshot must be non-empty bytes")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_bytes(state)
        os.replace(temporary, self.path)
        return "sha256:" + hashlib.sha256(state).hexdigest()


class SituatedEpisodeRecorder:
    """Append an autonomous outcome to canonical history and Cognee projection."""

    def __init__(self, journal: ConversationJournal, memory: SituatedMemory) -> None:
        self.journal = journal
        self.memory = memory

    async def record(self, episode: AutonomousEpisode) -> None:
        if episode.retain_procedure_content:
            procedure = "\n".join(
                _learning_action_text(action) for action in episode.action_trace
            )
        else:
            procedure = " -> ".join(
                _action_summary(action) for action in episode.action_trace
            )
        procedure = (procedure or "no executable action")[
            : min(episode.max_learning_trace_chars, 3_072)
        ]
        material = json.dumps(
            {
                "task_id": episode.task_id,
                "request": episode.request,
                "procedure": procedure,
                "outcome": episode.outcome,
                "verifier": episode.verifier.output_sha256,
                "parent": episode.state_parent,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        artifact_ref = "sha256:" + hashlib.sha256(material.encode()).hexdigest()
        projection_text = (
            f"Autonomous software attempt. Task: {episode.request} "
            f"Procedure: {procedure}. Objective outcome: {episode.outcome}."
        )[:4_096].strip()
        projection = MemoryProjection.from_mapping(
            artifact_ref=artifact_ref,
            text=projection_text,
            source_ref=episode.verifier.output_sha256,
            context={
                "feedback_origin": "objective_environment",
                "outcome": episode.outcome,
                "state_parent": episode.state_parent,
                "task_id": episode.task_id,
            },
        )
        record = JournalRecord(projection, current_regime=episode.outcome == "success")
        existing = next(
            (
                item
                for item in self.journal.records()
                if item.projection.artifact_ref == artifact_ref
            ),
            None,
        )
        if existing is not None:
            if existing != record:
                raise ValueError("episode artifact_ref conflicts with existing journal evidence")
            return
        self.journal.append(record)
        await self.memory.backend.remember(encode_projection(projection))
        self.memory.origin = self.journal.build_origin()


class ActionTraceCompetenceLearner:
    """Use the trained action-trace core as persistent online competence.

    The frozen core maps detached Qwen representations into procedure and
    memory writes.  Only ``PlasticProcedureState`` changes after objective
    outcomes.  Recalled Cognee episodes are situated with Moving Origin before
    the same state contributes to the next task's visible procedure context.
    """

    def __init__(
        self,
        core: ActionTraceProceduralCore,
        state: PlasticProcedureState,
        memory: SituatedMemory,
        qwen: LocalQwenIO,
        *,
        feature_spec: SituatedFeatureSpec,
        checkpoint_identity: str,
        recall_limit: int = 12,
        use_moving_origin: bool = True,
    ) -> None:
        if core.config.temporal_width != feature_spec.width:
            raise ValueError("Moving Origin feature width does not match the procedural core")
        _positive_int(recall_limit, "recall_limit")
        core._validate_state(state)
        core.requires_grad_(False).eval()
        if any(parameter.requires_grad for parameter in qwen.model.parameters()):
            raise ValueError("Qwen foundation parameters must remain frozen")
        self.core = core
        self.state = state.detached_clone()
        self.memory = memory
        self.qwen = qwen
        self.feature_spec = feature_spec
        self.checkpoint_identity = checkpoint_identity
        self.recall_limit = recall_limit
        self.use_moving_origin = bool(use_moving_origin)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        memory: SituatedMemory,
        qwen: LocalQwenIO,
        *,
        feature_spec: SituatedFeatureSpec,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        recall_limit: int = 12,
        use_moving_origin: bool = True,
    ) -> "ActionTraceCompetenceLearner":
        sealed = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
        required = {
            "identity",
            "foundation_digest",
            "core_type",
            "core_config",
            "maximum_trace_steps",
            "core_state_dict",
            "plastic_state",
        }
        if not isinstance(sealed, Mapping) or not required.issubset(sealed):
            raise ValueError("action-trace checkpoint is incomplete")
        if sealed["core_type"] != "action-trace-matching-v1":
            raise ValueError("checkpoint is not an action-trace procedural core")
        if foundation_tensor_digest(qwen.model) != sealed["foundation_digest"]:
            raise ValueError("checkpoint binds a different frozen Qwen foundation")
        config = ProceduralCoreConfig(**dict(sealed["core_config"]))
        core = ActionTraceProceduralCore(
            config,
            maximum_trace_steps=int(sealed["maximum_trace_steps"]),
        )
        core.load_state_dict(dict(sealed["core_state_dict"]), strict=True)
        target = torch.device(device)
        core.to(device=target, dtype=dtype)
        reference = next(core.parameters())
        raw_state = sealed["plastic_state"]
        if not isinstance(raw_state, Mapping) or set(raw_state) != {"keys", "values", "strengths", "step"}:
            raise ValueError("checkpoint plastic state is malformed")
        state = PlasticProcedureState(
            keys=raw_state["keys"].detach().to(reference),
            values=raw_state["values"].detach().to(reference),
            strengths=raw_state["strengths"].detach().to(reference),
            step=int(raw_state["step"]),
        )
        return cls(
            core,
            state,
            memory,
            qwen,
            feature_spec=feature_spec,
            checkpoint_identity=str(sealed["identity"]),
            recall_limit=recall_limit,
            use_moving_origin=use_moving_origin,
        )

    def capture_state(self) -> bytes:
        buffer = io.BytesIO()
        torch.save(
            {
                "checkpoint_identity": self.checkpoint_identity,
                "keys": self.state.keys.detach().cpu(),
                "values": self.state.values.detach().cpu(),
                "strengths": self.state.strengths.detach().cpu(),
                "step": self.state.step,
            },
            buffer,
        )
        return buffer.getvalue()

    def restore_state(self, state: bytes) -> None:
        if not isinstance(state, bytes) or not state:
            raise ValueError("competence snapshot must be non-empty bytes")
        payload = torch.load(io.BytesIO(state), map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or set(payload) != {
            "checkpoint_identity",
            "keys",
            "values",
            "strengths",
            "step",
        }:
            raise ValueError("competence snapshot is malformed")
        if payload["checkpoint_identity"] != self.checkpoint_identity:
            raise ValueError("competence snapshot belongs to a different checkpoint")
        reference = next(self.core.parameters())
        candidate = PlasticProcedureState(
            keys=payload["keys"].detach().to(reference),
            values=payload["values"].detach().to(reference),
            strengths=payload["strengths"].detach().to(reference),
            step=int(payload["step"]),
        )
        self.core._validate_state(candidate)
        self.state = candidate.detached_clone()

    def state_digest(self) -> str:
        return plastic_state_digest(self.state)

    async def experience(self, request: str) -> str:
        if self.memory.origin.size == 0:
            return ""
        recall = await self.memory.recall(request, limit=self.recall_limit)
        if not recall.items:
            return ""
        texts = (request, *(item.text for item in recall.items))
        encoded = self.qwen.embed(texts)
        reference = next(self.core.parameters())
        if not isinstance(encoded, torch.Tensor) or encoded.shape != (
            len(texts),
            self.core.config.content_width,
        ):
            raise ValueError("Qwen embeddings do not match the procedural core")
        encoded = encoded.detach().to(reference)
        temporal, mask = encode_situated_features(
            (recall,),
            now=(self.memory.origin.now,),
            spec=self.feature_spec,
            device=reference.device,
            dtype=reference.dtype,
        )
        if not self.use_moving_origin:
            temporal.zero_()
        with torch.inference_mode():
            output = self.core(
                encoded[:1],
                encoded[1:].unsqueeze(0),
                temporal,
                mask,
                plastic_state=self.state,
            )
        weights = output.candidate_weights[0].detach().cpu()
        order = torch.argsort(weights, descending=True).tolist()
        successes = [
            index
            for index in order
            if dict(recall.items[index].context).get("outcome") == "success"
        ][:2]
        failures = [
            index
            for index in order
            if dict(recall.items[index].context).get("outcome") == "failure"
        ][:2]
        unclassified = [
            index
            for index in order
            if dict(recall.items[index].context).get("outcome") not in {"success", "failure"}
        ][:1]

        def render(label: str, indices: list[int]) -> list[str]:
            if not indices:
                return []
            return [
                f"{label} (adapt, do not copy blindly):",
                *(
                    f"- learned_weight={float(weights[index]):.6f}: {recall.items[index].text}"
                    for index in indices
                ),
            ]

        rows = render("OBJECTIVELY SUCCESSFUL PROCEDURES", successes)
        if failures:
            rows.extend(
                [
                    "OBJECTIVELY FAILED PROCEDURES (counterexamples; do not repeat):",
                    *(
                        f"- learned_weight={float(weights[index]):.6f}: {recall.items[index].text}"
                        for index in failures
                    ),
                ]
            )
        if unclassified:
            rows.extend(render("UNCLASSIFIED PRIOR EVIDENCE", unclassified))
        return "\n".join(rows)

    def apply_outcome(self, outcome: OutcomeContext) -> None:
        # V6 learned its action-trace write rule only from successful support
        # traces.  A negative sign is accepted by the tensor API but was never
        # trained as anti-consolidation; deploying it caused failed procedures
        # to dominate later recall.  Failures remain durable Cognee
        # counterexamples, while neural competence consolidates only verified
        # success until an outcome-aware learned successor exists.
        if outcome.disposition == "failure":
            return
        trace = tuple(
            _learning_action_text(action)
            for action in outcome.action_trace
            if action.kind in {"read", "write"}
        )
        if not trace:
            raise ValueError("objective feedback requires an executed read/write procedure")
        trace = trace[-self.core.maximum_trace_steps :]
        texts = (outcome.request, *trace)
        encoded = self.qwen.embed(texts)
        reference = next(self.core.parameters())
        if not isinstance(encoded, torch.Tensor) or encoded.shape != (
            len(texts),
            self.core.config.content_width,
        ):
            raise ValueError("Qwen action embeddings do not match the procedural core")
        encoded = encoded.detach().to(reference)
        count = len(trace)
        candidates = encoded[1:].unsqueeze(0)
        candidate_mask = torch.ones((1, count), device=reference.device, dtype=torch.bool)
        temporal = torch.zeros(
            (1, count, self.core.config.temporal_width),
            device=reference.device,
            dtype=reference.dtype,
        )
        with torch.inference_mode():
            output = self.core(
                encoded[:1],
                candidates,
                temporal,
                candidate_mask,
                plastic_state=self.state,
            )
            padded = torch.zeros(
                (1, self.core.maximum_trace_steps, self.core.config.content_width),
                device=reference.device,
                dtype=reference.dtype,
            )
            padded[0, :count] = candidates[0]
            trace_mask = torch.zeros(
                (1, self.core.maximum_trace_steps),
                device=reference.device,
                dtype=torch.bool,
            )
            trace_mask[0, :count] = True
            self.state = self.core.apply_action_trace_feedback(
                self.state,
                output,
                padded,
                trace_mask,
                torch.ones(1, device=reference.device, dtype=reference.dtype),
            )


class RootedSoftwareSandbox:
    """Exact-path file operations plus one immutable verifier command."""

    def __init__(self, contract: AutonomousTaskContract) -> None:
        self.contract = contract
        self.root = contract.workspace_root.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("workspace_root must be an existing directory")
        self._receipts: dict[str, tuple[str, ActionReceipt | VerifierReceipt]] = {}
        self.verifier_runs = 0

    def execute(self, action: ProposedAction) -> ActionReceipt | VerifierReceipt:
        fingerprint = _action_fingerprint(action)
        previous = self._receipts.get(action.action_id)
        if previous is not None:
            if previous[0] != fingerprint:
                raise ValueError("duplicate action_id has conflicting content")
            return previous[1]
        if action.kind == "read":
            path = self._authorized_path(action.path, writable=False)
            if not path.is_file():
                raise FileNotFoundError(f"declared readable file does not exist: {action.path}")
            data = path.read_bytes()
            if len(data) > self.contract.max_file_bytes:
                raise ValueError("read exceeds the task file-size ceiling")
            receipt = ActionReceipt(
                action.action_id,
                action.kind,
                "OK",
                data.decode("utf-8", errors="replace")[: self.contract.max_observation_chars],
            )
        elif action.kind == "write":
            if not self.contract.allow_mutation:
                raise PermissionError("task mutation is not authorized")
            path = self._authorized_path(action.path, writable=True)
            if type(action.content) is not str:
                raise ValueError("write action requires text content")
            encoded = action.content.encode("utf-8")
            if len(encoded) > self.contract.max_file_bytes:
                raise ValueError("write exceeds the task file-size ceiling")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(encoded)
            receipt = ActionReceipt(
                action.action_id,
                action.kind,
                "OK",
                f"Wrote {len(encoded)} bytes to {action.path}.",
                mutated=True,
            )
        elif action.kind in {"verify", "finish"}:
            verifier = self.verify()
            self._receipts[action.action_id] = (fingerprint, verifier)
            return verifier
        elif action.kind == "ask":
            if type(action.question) is not str or not action.question.strip():
                raise ValueError("ask action requires a non-empty question")
            receipt = ActionReceipt(action.action_id, action.kind, "OK", action.question.strip())
        else:  # pragma: no cover - parse_proposed_action rejects unknown kinds
            raise ValueError("unsupported action kind")
        self._receipts[action.action_id] = (fingerprint, receipt)
        return receipt

    def verify(self) -> VerifierReceipt:
        if self.verifier_runs >= self.contract.max_verifier_runs:
            raise RuntimeError("verifier-run budget exhausted")
        self.verifier_runs += 1
        started = time.monotonic()
        try:
            completed = subprocess.run(
                self.contract.verifier_command,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=float(self.contract.verifier_timeout_seconds),
                shell=False,
                check=False,
            )
            output = (completed.stdout or "") + (completed.stderr or "")
            disposition: VerifierDisposition = "PASS" if completed.returncode == 0 else "FAIL"
            exit_code = completed.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            output = f"{type(exc).__name__}: {exc}"
            disposition = "ERROR"
            exit_code = None
        duration = time.monotonic() - started
        encoded = output.encode("utf-8", errors="replace")
        return VerifierReceipt(
            disposition=disposition,
            exit_code=exit_code,
            duration_seconds=duration,
            output_sha256="sha256:" + hashlib.sha256(encoded).hexdigest(),
            output_chars=len(output),
            controller_excerpt=output[: self.contract.max_observation_chars],
        )

    def _authorized_path(self, raw: str | None, *, writable: bool) -> Path:
        if raw is None:
            raise ValueError("file action requires a path")
        relative = _normalized_relative(raw, "action path")
        allowed = self.contract.writable_paths if writable else (
            self.contract.readable_paths + self.contract.writable_paths
        )
        if relative not in allowed:
            raise PermissionError(f"path is outside the declared {'write' if writable else 'read'} scope")
        unresolved = self.root / relative
        if unresolved.is_symlink():
            raise PermissionError("symbolic links are not permitted in task paths")
        target = unresolved.resolve(strict=False)
        if not target.is_relative_to(self.root):
            raise PermissionError("resolved path escapes the task workspace")
        probe = target if target.exists() else target.parent
        while probe != self.root and not probe.exists():
            probe = probe.parent
        if probe.is_symlink():
            raise PermissionError("symbolic links are not permitted in task paths")
        return target


class AutonomousApprentice:
    """Run one bounded request-to-outcome-to-learning cycle."""

    def __init__(
        self,
        policy: ActionPolicy,
        learner: CompetenceLearner,
        recorder: EpisodeRecorder,
        state_store: AtomicCompetenceStore,
        *,
        experience: Callable[[str], str | Awaitable[str]] | None = None,
    ) -> None:
        self.policy = policy
        self.learner = learner
        self.recorder = recorder
        self.state_store = state_store
        self.experience = experience

    async def run(self, contract: AutonomousTaskContract) -> AutonomousRunResult:
        questions = contract.clarification_questions()
        initial_digest = self.learner.state_digest()
        if questions:
            return AutonomousRunResult(
                "CLARIFICATION_REQUIRED",
                contract.task_id,
                (),
                (),
                questions,
                initial_digest,
                initial_digest,
            )
        sandbox = RootedSoftwareSandbox(contract)
        history: list[str] = []
        attempt_actions: list[ProposedAction] = []
        receipts: list[ActionReceipt] = []
        verifier_receipts: list[VerifierReceipt] = []
        mutated = False
        prior_experience = ""
        if self.experience is not None:
            prior_experience = await _maybe_await(self.experience(contract.request))
        for step in range(contract.max_steps):
            prompt = build_action_prompt(contract, history, prior_experience, step=step + 1)
            try:
                proposed = parse_proposed_action(self.policy(prompt))
                # Idempotency identities belong to the controller, not to a
                # small language model that may reuse a convenient label such
                # as "1" on every turn.  The model label remains visible in the
                # scoped identity while each bounded turn is unambiguous.
                action = replace(
                    proposed,
                    action_id=f"turn-{step + 1}:{proposed.action_id}",
                )
                if action.kind == "ask" and mutated:
                    raise ValueError("clarification cannot be requested after workspace mutation")
                result = sandbox.execute(action)
            except (ValueError, PermissionError, FileNotFoundError, RuntimeError) as exc:
                history.append(f"Action rejected: {type(exc).__name__}: {exc}")
                continue
            attempt_actions.append(action)
            if isinstance(result, ActionReceipt):
                receipts.append(result)
                if result.kind == "ask":
                    return AutonomousRunResult(
                        "CLARIFICATION_REQUIRED",
                        contract.task_id,
                        tuple(receipts),
                        tuple(verifier_receipts),
                        (result.observation,),
                        initial_digest,
                        self.learner.state_digest(),
                    )
                mutated = mutated or result.mutated
                history.append(f"{result.kind.upper()}: {result.observation}")
                continue

            verifier_receipts.append(result)
            history.append(
                result.observation_for(
                    reveal_diagnostics=contract.reveal_verifier_diagnostics,
                    maximum=contract.max_observation_chars,
                )
            )
            if result.disposition == "ERROR":
                return AutonomousRunResult(
                    "INVALID",
                    contract.task_id,
                    tuple(receipts),
                    tuple(verifier_receipts),
                    (),
                    initial_digest,
                    self.learner.state_digest(),
                )
            outcome: Literal["success", "failure"] = (
                "success" if result.disposition == "PASS" else "failure"
            )
            has_procedure = any(
                item.kind in {"read", "write"} for item in attempt_actions
            )
            if has_procedure:
                await self._commit_outcome(
                    contract,
                    tuple(attempt_actions),
                    result,
                    outcome,
                )
            elif outcome == "failure":
                history.append(
                    "No read/write procedure preceded this failure, so it produced no learning update."
                )
            attempt_actions.clear()
            if outcome == "success":
                return AutonomousRunResult(
                    "SUCCEEDED",
                    contract.task_id,
                    tuple(receipts),
                    tuple(verifier_receipts),
                    (),
                    initial_digest,
                    self.learner.state_digest(),
                )
        return AutonomousRunResult(
            "EXHAUSTED",
            contract.task_id,
            tuple(receipts),
            tuple(verifier_receipts),
            (),
            initial_digest,
            self.learner.state_digest(),
        )

    async def _commit_outcome(
        self,
        contract: AutonomousTaskContract,
        actions: tuple[ProposedAction, ...],
        verifier: VerifierReceipt,
        outcome: Literal["success", "failure"],
    ) -> None:
        parent = self.learner.capture_state()
        parent_digest = self.learner.state_digest()
        episode = AutonomousEpisode(
            task_id=contract.task_id,
            request=contract.request,
            action_trace=actions,
            verifier=verifier,
            outcome=outcome,
            state_parent=parent_digest,
            retain_procedure_content=contract.retain_procedure_content,
            max_learning_trace_chars=contract.max_learning_trace_chars,
        )
        try:
            await _maybe_await(self.recorder.record(episode))
            self.learner.apply_outcome(
                OutcomeContext(
                    contract.task_id,
                    contract.request,
                    actions,
                    outcome,
                    verifier.output_sha256,
                )
            )
            candidate = self.learner.capture_state()
            self.state_store.save(candidate)
        except Exception:
            self.learner.restore_state(parent)
            raise


def parse_proposed_action(raw: str) -> ProposedAction:
    if type(raw) is not str or len(raw) > 1_000_000:
        raise ValueError("model action must be bounded JSON text")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("model action is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("model action must be a JSON object")
    allowed = {"action_id", "kind", "path", "content", "question"}
    if set(payload).difference(allowed) or not {"action_id", "kind"}.issubset(payload):
        raise ValueError("model action has invalid fields")
    action_id = payload["action_id"]
    kind = payload["kind"]
    if type(action_id) is not str or not action_id.strip() or len(action_id) > 128:
        raise ValueError("action_id must be non-empty bounded text")
    if kind not in {"read", "write", "verify", "finish", "ask"}:
        raise ValueError("action kind is unsupported")
    for name in ("path", "content", "question"):
        if name in payload and payload[name] is not None and type(payload[name]) is not str:
            raise ValueError(f"{name} must be text when present")
    return ProposedAction(
        action_id=action_id.strip(),
        kind=kind,
        path=payload.get("path"),
        content=payload.get("content"),
        question=payload.get("question"),
    )


def build_action_prompt(
    contract: AutonomousTaskContract,
    history: Sequence[str],
    experience: str,
    *,
    step: int,
) -> str:
    """Render only public task state, prior experience, and outcome feedback."""

    observations = "\n".join(history[-12:]) or "No actions have been taken."
    prior = experience.strip() or "No relevant prior experience was recalled."
    return (
        "You are proposing the next action for a bounded software task. Return exactly one JSON object.\n"
        'Allowed forms: {"action_id":"...","kind":"read","path":"..."}, '
        '{"action_id":"...","kind":"write","path":"...","content":"..."}, '
        '{"action_id":"...","kind":"verify"}, '
        '{"action_id":"...","kind":"finish"}, or '
        '{"action_id":"...","kind":"ask","question":"..."}.\n'
        "Do not invent paths, commands, results, or permissions. The verifier command is fixed and hidden from your control.\n\n"
        f"Task: {contract.request}\n"
        f"Success means: {contract.success_description}\n"
        f"Readable paths: {', '.join(contract.readable_paths) or '(none)'}\n"
        f"Writable paths: {', '.join(contract.writable_paths) or '(none)'}\n"
        f"Step: {step}/{contract.max_steps}\n\n"
        f"Relevant prior experience:\n{prior[: contract.max_observation_chars]}\n\n"
        f"Current observations:\n{observations[-contract.max_observation_chars:]}"
    )


async def _maybe_await(value):
    return await value if inspect.isawaitable(value) else value


def _action_fingerprint(action: ProposedAction) -> str:
    payload = json.dumps(
        {
            "action_id": action.action_id,
            "kind": action.kind,
            "path": action.path,
            "content": action.content,
            "question": action.question,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _action_summary(action: ProposedAction) -> str:
    if action.kind in {"read", "write"}:
        return f"{action.kind}:{action.path}"
    return action.kind


def _learning_action_text(action: ProposedAction) -> str:
    if action.kind == "read":
        return f"Read declared file {action.path}."
    if action.kind == "write":
        content = (action.content or "")[:4_096]
        return f"Write declared file {action.path} with content:\n{content}"
    return action.kind


__all__ = [
    "ActionReceipt",
    "ActionTraceCompetenceLearner",
    "AtomicCompetenceStore",
    "AutonomousApprentice",
    "AutonomousEpisode",
    "AutonomousRunResult",
    "AutonomousTaskContract",
    "OutcomeContext",
    "ProposedAction",
    "RootedSoftwareSandbox",
    "SituatedEpisodeRecorder",
    "VerifierReceipt",
    "build_action_prompt",
    "parse_proposed_action",
]

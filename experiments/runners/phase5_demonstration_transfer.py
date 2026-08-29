"""Teach Angler a third procedure family from public demonstrations.

The retained V51 transition, decoder, base writers, and accepted precedence
adapter are frozen.  A typed correspondence encoder and a small generic,
zero-initialized latent reader learn together.  They see public input/output
examples on support tasks and are optimized through ordinary bounded
competence writes into later demonstration-free queries.  Training receives
only attempted public permutations and scalar outcomes; no target, solver,
procedure identity, or old-family replay is used.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

import torch
from torch import nn

from angler.procedures.skill_memory import (
    PublicEvidenceLatentReader,
    PublicEvidenceResidualWriter,
    differentiable_zero_public_evidence_skill_content,
    public_evidence_skill_content,
    procedural_skill_state_digest,
    snapshot_procedural_skill_state,
    zero_public_evidence_skill_content,
)
from experiments.evaluators.relational_procedure_transfer_suite import (
    PublicRelationalProcedureTask,
    make_relational_procedure_transfer_stream,
    score_relational_procedure_answer,
)
from experiments.evaluators.skill_memory_suite import (
    make_skill_memory_composition_curriculum,
    score_skill_memory_answer,
)
from experiments.evaluators.symbolic_procedure_transfer_suite import (
    PublicDemonstrationProcedureTask,
    PublicSymbolicDemonstration,
    demonstration_permutation_partition,
    make_demonstration_procedure_transfer_stream,
    score_demonstration_procedure_answer,
)
from experiments.runners import phase5_skill_memory_stream as phase5
from experiments.runners.phase5_cross_family_transfer import (
    SharedPublicFactAdapter,
    _acquire_pairs,
    _score_pairs,
    _state_element_count,
    _summary,
)


_REPORT_VERSION = "angler.phase5-demonstration-transfer.v11"
_DEMONSTRATION_ADAPTER_PREFIX = "public_fact_adapter.demonstration_adapter."
_PUBLIC_EVIDENCE_WRITER_PREFIX = "composition_memory.public_evidence_writer."
_PUBLIC_EVIDENCE_READER_PREFIX = "composition_memory.public_evidence_reader."


def _is_demonstration_trainable(name: str) -> bool:
    return name.startswith(_DEMONSTRATION_ADAPTER_PREFIX) or name.startswith(
        _PUBLIC_EVIDENCE_READER_PREFIX
    )


def _public_delta_preference_alignment(
    public_delta: torch.Tensor,
    candidate_indices: tuple[int, ...],
    scalar_scores: tuple[float, ...],
) -> tuple[int, int]:
    """Count reward-aligned public-delta edges without changing training."""

    row = public_delta.detach().reshape(-1)
    if row.numel() != len(phase5._PERMUTATIONS) or not bool(
        torch.isfinite(row).all().item()
    ):
        raise ValueError("public-delta alignment requires finite action logits")
    if (
        len(candidate_indices) != len(scalar_scores)
        or len(candidate_indices) < 2
        or len(set(candidate_indices)) != len(candidate_indices)
        or any(not 0 <= index < row.numel() for index in candidate_indices)
        or any(not math.isfinite(float(score)) for score in scalar_scores)
    ):
        raise ValueError("public-delta alignment inputs are invalid")
    aligned = 0
    edges = 0
    for left in range(len(candidate_indices)):
        for right in range(left + 1, len(candidate_indices)):
            reward_difference = float(scalar_scores[left]) - float(
                scalar_scores[right]
            )
            if reward_difference == 0.0:
                continue
            edges += 1
            logit_difference = float(
                (
                    row[candidate_indices[left]]
                    - row[candidate_indices[right]]
                ).item()
            )
            aligned += int(reward_difference * logit_difference > 0.0)
    return aligned, edges


def _root_public_transition_gate_mean_absolute(scores: Any) -> float | None:
    """Read the candidate-blind root gate for passive causal reporting."""

    roots = tuple(node for node in scores.nodes if tuple(node.path) == ())
    if len(roots) != 1:
        raise RuntimeError("policy scores must contain exactly one root node")
    gate = roots[0].memory_read.public_transition_gate
    if gate is None:
        return None
    detached = gate.detach()
    if detached.numel() < 1 or not bool(torch.isfinite(detached).all().item()):
        raise RuntimeError("public transition gate diagnostic is invalid")
    return float(detached.abs().mean().item())


def _causal_pass_bounds(
    meta_steps: int,
    mechanism_pass_size: int,
) -> tuple[tuple[int, int], ...]:
    """Return one-based full-pass bounds plus a final partial pass."""

    if (
        isinstance(meta_steps, bool)
        or not isinstance(meta_steps, int)
        or meta_steps < 1
        or isinstance(mechanism_pass_size, bool)
        or not isinstance(mechanism_pass_size, int)
        or mechanism_pass_size < 1
    ):
        raise ValueError("causal pass sizes must be positive integers")
    return tuple(
        (
            pass_index * mechanism_pass_size + 1,
            min((pass_index + 1) * mechanism_pass_size, meta_steps),
        )
        for pass_index in range(math.ceil(meta_steps / mechanism_pass_size))
    )


def _alpha_rename_demonstrations(
    task: PublicDemonstrationProcedureTask,
    *,
    salt: int,
) -> PublicDemonstrationProcedureTask:
    """Create a public-only surface rename without inspecting a mechanism."""

    if not task.demonstrations:
        return task
    symbols = sorted(
        {
            symbol
            for demonstration in task.demonstrations
            for symbol in demonstration.input_symbols
        }
    )
    renamed = {
        symbol: "renamed_"
        + hashlib.sha256(
            (
                "project-angler.public-demo-rename.v1\x00"
                f"{salt}\x00{symbol}"
            ).encode("utf-8")
        ).hexdigest()[:24]
        for symbol in symbols
    }
    demonstrations = tuple(
        PublicSymbolicDemonstration(
            tuple(renamed[symbol] for symbol in demonstration.input_symbols),
            tuple(renamed[symbol] for symbol in demonstration.output_symbols),
        )
        for demonstration in task.demonstrations
    )
    return replace(task, demonstrations=demonstrations)


class SymbolicDemonstrationAdapter(nn.Module):
    """Learn correspondence structure from raw public input/output examples.

    The adapter never compares symbol strings or constructs a permutation.
    A shared learned projection and soft cosine attention relate opaque input
    and output entities; its values carry only public output positions.  No
    string equality test or hard match is performed.  The resulting relation
    tokens are reduced into evidence for a later memory write and have no
    direct path to current-query logits.
    """

    _ENTITY_WIDTH = 64
    _POSITION_COUNT = 5

    def __init__(
        self,
        feature_width: int = 14,
        hidden_width: int = 64,
        evidence_width: int = 64,
    ) -> None:
        super().__init__()
        if (
            feature_width <= 0
            or hidden_width <= 0
            or evidence_width <= 0
            or hidden_width % 4
        ):
            raise ValueError("demonstration-adapter widths are invalid")
        self.feature_width = feature_width
        self.hidden_width = hidden_width
        self.evidence_width = evidence_width
        self.entity_encoder = nn.Sequential(
            nn.LayerNorm(self._ENTITY_WIDTH, elementwise_affine=False),
            nn.Linear(self._ENTITY_WIDTH, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
        )
        self.entity_norm = nn.LayerNorm(hidden_width)
        self.output_position_encoder = nn.Linear(
            self._POSITION_COUNT,
            hidden_width,
            bias=False,
        )
        self.correspondence_projection = nn.Linear(
            hidden_width,
            hidden_width,
            bias=False,
        )
        self.correspondence_logit_scale = nn.Parameter(
            torch.tensor(math.log(math.sqrt(hidden_width)), dtype=torch.float32)
        )
        self.relation_encoder = nn.Sequential(
            nn.LayerNorm(hidden_width + self._POSITION_COUNT),
            nn.Linear(hidden_width + self._POSITION_COUNT, 2 * hidden_width),
            nn.SiLU(),
            nn.Linear(2 * hidden_width, hidden_width),
        )
        self.relation_attention_norm = nn.LayerNorm(hidden_width)
        self.relation_attention = nn.MultiheadAttention(
            hidden_width,
            num_heads=4,
            batch_first=True,
        )
        self.relation_feed_forward_norm = nn.LayerNorm(hidden_width)
        self.relation_feed_forward = nn.Sequential(
            nn.Linear(hidden_width, 2 * hidden_width),
            nn.SiLU(),
            nn.Linear(2 * hidden_width, hidden_width),
        )
        self.evidence_norm = nn.LayerNorm(hidden_width)
        self.evidence_projection = nn.Linear(
            hidden_width,
            evidence_width,
            bias=False,
        )
        nn.init.normal_(self.evidence_projection.weight, mean=0.0, std=1.0e-3)

    def encode_public_task(
        self,
        public_task: PublicDemonstrationProcedureTask,
        public_features: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(public_task, PublicDemonstrationProcedureTask):
            raise TypeError("demonstration adapter received the wrong public schema")
        if public_features.shape != (5, self.feature_width):
            raise ValueError("demonstration adapter requires five feature rows")
        # Demonstrations are evidence for a state write, never a direct
        # distortion of the current answer geometry.
        return public_features

    def feedback_evidence(
        self,
        public_task: PublicDemonstrationProcedureTask,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(public_task, PublicDemonstrationProcedureTask):
            raise TypeError("demonstration evidence received the wrong public schema")
        if (
            reference.ndim != 2
            or reference.shape[-1] != self.evidence_width
            or reference.shape[0] not in (1, 3)
        ):
            raise ValueError("demonstration evidence reference has the wrong shape")
        if reference.shape[0] != 1 and public_task.demonstrations:
            raise ValueError(
                "batched demonstration evidence is supported only for queries"
            )
        if not public_task.demonstrations:
            return torch.zeros_like(reference)
        raw_entities = self._raw_public_entities(
            public_task,
            device=reference.device,
            dtype=reference.dtype,
        )
        demonstrations = raw_entities.shape[0]
        encoded = self.entity_norm(self.entity_encoder(raw_entities))
        correspondence_entities = torch.nn.functional.normalize(
            self.correspondence_projection(encoded),
            dim=-1,
            eps=1.0e-8,
        )
        input_entities = correspondence_entities[:, 0]
        output_entities = correspondence_entities[:, 1]
        position_basis = torch.eye(
            self._POSITION_COUNT,
            device=reference.device,
            dtype=reference.dtype,
        )
        output_positions = self.output_position_encoder(position_basis)
        output_positions = output_positions.unsqueeze(0).expand(
            demonstrations,
            -1,
            -1,
        )
        correspondence_logits = torch.einsum(
            "dih,doh->dio",
            input_entities,
            output_entities,
        ) * self.correspondence_logit_scale.exp().clamp(1.0, 16.0)
        correspondence_weights = torch.softmax(
            correspondence_logits,
            dim=-1,
        )
        matched_output_positions = torch.einsum(
            "dio,doh->dih",
            correspondence_weights,
            output_positions,
        )
        input_positions = position_basis.unsqueeze(0).expand(
            demonstrations,
            -1,
            -1,
        )
        relations = self.relation_encoder(
            torch.cat((matched_output_positions, input_positions), dim=-1)
        )
        normalized = self.relation_attention_norm(relations)
        attended, _ = self.relation_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        relations = relations + attended
        relations = relations + self.relation_feed_forward(
            self.relation_feed_forward_norm(relations)
        )
        pooled = relations.mean(dim=1).mean(dim=0, keepdim=True)
        projected = self.evidence_projection(self.evidence_norm(pooled))
        centered = projected - projected.mean(dim=-1, keepdim=True)
        norm = torch.linalg.vector_norm(centered, dim=-1, keepdim=True)
        return torch.where(
            norm > torch.finfo(centered.dtype).eps,
            centered / norm.clamp_min(torch.finfo(centered.dtype).eps),
            torch.zeros_like(centered),
        )

    @staticmethod
    def _raw_public_entities(
        public_task: PublicDemonstrationProcedureTask,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Tokenize symbols independently without comparing or aligning them."""

        encoded_demonstrations: list[list[list[list[float]]]] = []
        for demonstration in public_task.demonstrations:
            if set(demonstration.input_symbols) != set(
                demonstration.output_symbols
            ):
                raise ValueError("demonstration entities do not match")
            sides: list[list[list[float]]] = []
            for symbols in (
                demonstration.input_symbols,
                demonstration.output_symbols,
            ):
                sides.append(
                    [
                        SymbolicDemonstrationAdapter._opaque_symbol_vector(symbol)
                        for symbol in symbols
                    ]
                )
            encoded_demonstrations.append(sides)
        return torch.tensor(encoded_demonstrations, device=device, dtype=dtype)

    @staticmethod
    def _opaque_symbol_vector(symbol: str) -> list[float]:
        """Map one token independently; equal-token discovery remains learned."""

        if not isinstance(symbol, str) or not symbol:
            raise ValueError("public entity symbol must be non-empty text")
        digest = hashlib.sha256(
            b"project-angler.public-entity-token.v1\x00" + symbol.encode("utf-8")
        ).digest()
        scale = 1.0 / math.sqrt(SymbolicDemonstrationAdapter._ENTITY_WIDTH)
        return [
            scale if digest[index // 8] & (1 << (index % 8)) else -scale
            for index in range(SymbolicDemonstrationAdapter._ENTITY_WIDTH)
        ]


class TypedPublicFactPorts(nn.Module):
    """Public-schema dispatch only; every port feeds the same Angler core."""

    def __init__(
        self,
        precedence_adapter: SharedPublicFactAdapter,
        demonstration_adapter: SymbolicDemonstrationAdapter,
    ) -> None:
        super().__init__()
        self.precedence_adapter = precedence_adapter
        self.demonstration_adapter = demonstration_adapter

    @staticmethod
    def applies_to(public_task: Any) -> bool:
        return isinstance(
            public_task,
            (PublicRelationalProcedureTask, PublicDemonstrationProcedureTask),
        )

    def encode_public_task(
        self,
        public_task: Any,
        public_features: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(public_task, PublicRelationalProcedureTask):
            return self.precedence_adapter(public_features)
        if isinstance(public_task, PublicDemonstrationProcedureTask):
            return self.demonstration_adapter.encode_public_task(
                public_task,
                public_features,
            )
        raise TypeError("no typed public-fact port matches the task schema")

    def feedback_evidence(
        self,
        public_task: Any,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(public_task, PublicDemonstrationProcedureTask):
            return self.demonstration_adapter.feedback_evidence(
                public_task,
                reference,
            )
        return torch.zeros_like(reference)

    @staticmethod
    def reads_public_evidence_state(public_task: Any) -> bool:
        """Gate the auxiliary state by public schema, never by task identity."""

        return isinstance(public_task, PublicDemonstrationProcedureTask)


def _attach_public_evidence_writer(
    policy: phase5.SkillMemoryPolicy,
) -> PublicEvidenceResidualWriter:
    existing = getattr(policy.composition_memory, "public_evidence_writer", None)
    if existing is not None:
        if not isinstance(existing, PublicEvidenceResidualWriter):
            raise TypeError("composition public-evidence writer has the wrong type")
        return existing
    reference = next(policy.parameters())
    writer = PublicEvidenceResidualWriter(policy.composition_memory.width).to(
        device=reference.device,
        dtype=reference.dtype,
    )
    policy.composition_memory.public_evidence_writer = writer
    return writer


def _public_evidence_writer_state(
    policy: phase5.SkillMemoryPolicy,
) -> dict[str, torch.Tensor]:
    _attach_public_evidence_writer(policy)
    state = {
        name: value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
        if name.startswith(_PUBLIC_EVIDENCE_WRITER_PREFIX)
    }
    if not state:
        raise RuntimeError("public-evidence writer state is empty")
    return state


def _load_public_evidence_writer_state(
    policy: phase5.SkillMemoryPolicy,
    state: dict[str, Any],
) -> None:
    if not isinstance(state, dict) or not state:
        raise RuntimeError("public-evidence writer checkpoint is incomplete")
    _attach_public_evidence_writer(policy)
    current = policy.state_dict()
    expected = {
        name for name in current if name.startswith(_PUBLIC_EVIDENCE_WRITER_PREFIX)
    }
    if set(state) != expected:
        raise RuntimeError("public-evidence writer checkpoint keys are invalid")
    with torch.no_grad():
        for name in sorted(expected):
            value = state[name]
            target = current[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != target.shape
                or value.dtype != target.dtype
            ):
                raise RuntimeError(
                    "public-evidence writer checkpoint tensor is incompatible"
                )
            target.copy_(value.to(device=target.device))


def _attach_public_evidence_reader(
    policy: phase5.SkillMemoryPolicy,
) -> PublicEvidenceLatentReader:
    if getattr(policy.composition_memory, "public_evidence_writer", None) is not None:
        raise RuntimeError("v6 public latent reader cannot coexist with the v5 writer")
    existing = getattr(policy.composition_memory, "public_evidence_reader", None)
    if existing is not None:
        if not isinstance(existing, PublicEvidenceLatentReader):
            raise TypeError("composition public-evidence reader has the wrong type")
        return existing
    reference = next(policy.parameters())
    reader = PublicEvidenceLatentReader(policy.composition_memory.width).to(
        device=reference.device,
        dtype=reference.dtype,
    )
    policy.composition_memory.public_evidence_reader = reader
    return reader


def _public_evidence_reader_state(
    policy: phase5.SkillMemoryPolicy,
) -> dict[str, torch.Tensor]:
    _attach_public_evidence_reader(policy)
    state = {
        name: value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
        if name.startswith(_PUBLIC_EVIDENCE_READER_PREFIX)
    }
    if not state:
        raise RuntimeError("public-evidence reader state is empty")
    return state


def _load_public_evidence_reader_state(
    policy: phase5.SkillMemoryPolicy,
    state: dict[str, Any],
) -> None:
    if not isinstance(state, dict) or not state:
        raise RuntimeError("public-evidence reader checkpoint is incomplete")
    _attach_public_evidence_reader(policy)
    current = policy.state_dict()
    expected = {
        name for name in current if name.startswith(_PUBLIC_EVIDENCE_READER_PREFIX)
    }
    if set(state) != expected:
        raise RuntimeError("public-evidence reader checkpoint keys are invalid")
    with torch.no_grad():
        for name in sorted(expected):
            value = state[name]
            target = current[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != target.shape
                or value.dtype != target.dtype
            ):
                raise RuntimeError(
                    "public-evidence reader checkpoint tensor is incompatible"
                )
            target.copy_(value.to(device=target.device))


def _load_precedence_adapter(
    path: str | Path,
    *,
    expected_base_sha256: str,
) -> tuple[SharedPublicFactAdapter, dict[str, Any]]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"precedence adapter is missing: {checkpoint_path}")
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("runner") != "angler.phase5-cross-family-transfer.v3":
        raise RuntimeError("precedence adapter runner identity is invalid")
    if payload.get("base_checkpoint_sha256") != expected_base_sha256:
        raise RuntimeError("precedence adapter binds a different V51 core")
    model = payload.get("adapter_model")
    if not isinstance(model, dict):
        raise RuntimeError("precedence adapter payload is incomplete")
    adapter = SharedPublicFactAdapter()
    adapter.load_state_dict(model, strict=True)
    adapter.eval()
    adapter.requires_grad_(False)
    return adapter, {
        "path": str(checkpoint_path),
        "sha256": digest,
        "runner": payload["runner"],
        "base_checkpoint_sha256": payload["base_checkpoint_sha256"],
        "source_result_digest": payload["result_digest"],
    }


def _load_demonstration_adapter(
    path: str | Path,
    *,
    expected_base_sha256: str,
    expected_precedence_sha256: str,
) -> tuple[
    SymbolicDemonstrationAdapter,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"demonstration adapter is missing: {checkpoint_path}"
        )
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_runner = payload.get("runner")
    if source_runner != _REPORT_VERSION:
        raise RuntimeError("demonstration adapter runner identity is invalid")
    if source_runner == _REPORT_VERSION and payload.get("stage") != "train_only":
        raise RuntimeError("demonstration interface is not a train-only artifact")
    if payload.get("base_checkpoint_sha256") != expected_base_sha256:
        raise RuntimeError("demonstration adapter binds a different V51 core")
    if payload.get("precedence_adapter_sha256") != expected_precedence_sha256:
        raise RuntimeError("demonstration adapter binds a different precedence port")
    model = payload.get("demonstration_adapter_model")
    reader_model = payload.get("public_evidence_reader_model")
    training = payload.get("training")
    if (
        not isinstance(model, dict)
        or not isinstance(reader_model, dict)
        or not isinstance(training, dict)
    ):
        raise RuntimeError("demonstration interface payload is incomplete")
    adapter = SymbolicDemonstrationAdapter()
    adapter.load_state_dict(model, strict=True)
    adapter.eval()
    adapter.requires_grad_(False)
    return adapter, reader_model, training, {
        "path": str(checkpoint_path),
        "sha256": digest,
        "runner": payload["runner"],
        "stage": payload.get("stage", "legacy_post_evaluation"),
        "base_checkpoint_sha256": payload["base_checkpoint_sha256"],
        "precedence_adapter_sha256": payload["precedence_adapter_sha256"],
        "source_result_digest": payload.get("result_digest"),
        "protected_fingerprint": payload.get("protected_fingerprint"),
        "sensory_initialization": payload.get("sensory_initialization"),
    }


def _save_train_only_adapter(
    path: str | Path,
    *,
    policy: phase5.SkillMemoryPolicy,
    seed: int,
    base_checkpoint_sha256: str,
    precedence_adapter_sha256: str,
    training: dict[str, Any],
    sensory_initialization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if checkpoint_path.exists():
        raise FileExistsError(
            f"train-only adapter checkpoint already exists: {checkpoint_path}"
        )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    adapter = policy.public_fact_adapter.demonstration_adapter
    torch.save(
        {
            "runner": _REPORT_VERSION,
            "stage": "train_only",
            "seed": seed,
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "precedence_adapter_sha256": precedence_adapter_sha256,
            "demonstration_adapter_model": adapter.state_dict(),
            "public_evidence_reader_model": _public_evidence_reader_state(policy),
            "training": training,
            "sensory_initialization": sensory_initialization,
            "protected_fingerprint": _protected_state_fingerprint(policy),
        },
        checkpoint_path,
    )
    return {
        "path": str(checkpoint_path),
        "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "runner": _REPORT_VERSION,
        "stage": "train_only",
    }


def _load_v5_sensory_initialization(
    path: str | Path,
    *,
    expected_base_sha256: str,
    expected_precedence_sha256: str,
) -> tuple[SymbolicDemonstrationAdapter, dict[str, Any]]:
    """Reuse only V5's learned public correspondence port, never its writer."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"v5 sensory initialization is missing: {checkpoint_path}"
        )
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("runner") != "angler.phase5-demonstration-transfer.v5" or (
        payload.get("stage") != "train_only"
    ):
        raise RuntimeError("sensory initialization is not the frozen v5 artifact")
    if payload.get("base_checkpoint_sha256") != expected_base_sha256:
        raise RuntimeError("v5 sensory initialization binds a different V51 core")
    if payload.get("precedence_adapter_sha256") != expected_precedence_sha256:
        raise RuntimeError("v5 sensory initialization binds a different precedence port")
    model = payload.get("demonstration_adapter_model")
    if not isinstance(model, dict):
        raise RuntimeError("v5 sensory initialization is incomplete")
    adapter = SymbolicDemonstrationAdapter()
    adapter.load_state_dict(model, strict=True)
    return adapter, {
        "path": str(checkpoint_path),
        "sha256": digest,
        "runner": payload["runner"],
        "stage": payload["stage"],
        "writer_reused": False,
    }


def _logit_intervention_summary(
    policy: phase5.SkillMemoryPolicy,
    acquired_state: Any,
    reset_state: Any,
    pairs: Any,
) -> dict[str, Any]:
    acquired_digest = procedural_skill_state_digest(acquired_state)
    reset_digest = procedural_skill_state_digest(reset_state)
    reset_differences: list[torch.Tensor] = []
    removed_differences: list[torch.Tensor] = []
    public_zero_differences: list[torch.Tensor] = []
    reset_argmax_changes = 0
    removed_argmax_changes = 0
    public_zero_argmax_changes = 0
    count = 0
    public_zero_state = zero_public_evidence_skill_content(acquired_state)
    with torch.inference_mode():
        for pair in pairs:
            full = policy.score_task(pair.learner, acquired_state).logits
            reset = policy.score_task(pair.learner, reset_state).logits
            removed = policy.score_task(
                pair.learner,
                acquired_state,
                include_reversible_transition=False,
            ).logits
            public_zero = policy.score_task(
                pair.learner,
                public_zero_state,
            ).logits
            reset_differences.append((full - reset).detach().abs().flatten())
            removed_differences.append((full - removed).detach().abs().flatten())
            public_zero_differences.append(
                (full - public_zero).detach().abs().flatten()
            )
            reset_argmax_changes += int(
                (full.argmax(dim=-1) != reset.argmax(dim=-1)).item()
            )
            removed_argmax_changes += int(
                (full.argmax(dim=-1) != removed.argmax(dim=-1)).item()
            )
            public_zero_argmax_changes += int(
                (full.argmax(dim=-1) != public_zero.argmax(dim=-1)).item()
            )
            count += 1
    if procedural_skill_state_digest(acquired_state) != acquired_digest or (
        procedural_skill_state_digest(reset_state) != reset_digest
    ):
        raise RuntimeError("logit diagnostics mutated persistent competence")

    def summarize(
        differences: list[torch.Tensor],
        argmax_changes: int,
    ) -> dict[str, float | int]:
        flattened = torch.cat(differences)
        return {
            "queries": count,
            "mean_absolute_logit_difference": float(flattened.mean().item()),
            "maximum_absolute_logit_difference": float(flattened.max().item()),
            "argmax_changes": argmax_changes,
            "argmax_change_fraction": argmax_changes / count,
        }

    return {
        "acquired_vs_reference": summarize(
            reset_differences,
            reset_argmax_changes,
        ),
        "acquired_vs_reversible_removed": summarize(
            removed_differences,
            removed_argmax_changes,
        ),
        "acquired_vs_public_evidence_zeroed": summarize(
            public_zero_differences,
            public_zero_argmax_changes,
        ),
    }


def _acquire_matched_demonstration_arms(
    policy: phase5.SkillMemoryPolicy,
    initial_state: Any,
    correct_pairs: Any,
    no_demonstration_pairs: Any,
    wrong_demonstration_pairs: Any,
) -> tuple[tuple[Any, dict[str, Any]], ...]:
    """Acquire three causal arms with identical actions and scalar outcomes."""

    states = [initial_state, initial_state, initial_state]
    scores = [[], [], []]
    accepted = [0, 0, 0]
    core_accepted = [0, 0, 0]
    delta_norms = [[], [], []]
    incoming_elements = _state_element_count(initial_state)
    for correct, absent, wrong in zip(
        correct_pairs,
        no_demonstration_pairs,
        wrong_demonstration_pairs,
        strict=True,
    ):
        proposal = phase5.propose_task(
            policy,
            correct.learner,
            states[0],
            greedy=False,
            temperature=1.25,
        )
        proposals = (
            proposal,
            phase5._proposal_for_candidate(
                policy,
                absent.learner,
                states[1],
                proposal.candidate_index,
            ),
            phase5._proposal_for_candidate(
                policy,
                wrong.learner,
                states[2],
                proposal.candidate_index,
            ),
        )
        pairs = (correct, absent, wrong)
        reward = score_demonstration_procedure_answer(
            correct.learner,
            correct.hidden,
            proposal.answer,
        )
        for index, (pair, arm) in enumerate(
            zip(pairs, proposals, strict=True)
        ):
            feedback = phase5.apply_transactional_feedback(
                policy,
                pair.learner,
                arm,
                reward,
                states[index],
            )
            states[index] = feedback.state
            scores[index].append(float(reward))
            accepted[index] += int(feedback.accepted)
            core_accepted[index] += int(feedback.core_accepted)
            delta_norms[index].append(float(feedback.delta_norm))
            if _state_element_count(states[index]) != incoming_elements:
                raise RuntimeError("matched acquisition changed fixed state capacity")

    records: list[tuple[Any, dict[str, Any]]] = []
    for index in range(3):
        window = max(1, len(scores[index]) // 4)
        records.append(
            (
                states[index],
                {
                    "presentations": len(scores[index]),
                    "accepted_transactions": accepted[index],
                    "core_accepted_transactions": core_accepted[index],
                    "first_quarter_mean": (
                        sum(scores[index][:window]) / window
                    ),
                    "last_quarter_mean": (
                        sum(scores[index][-window:]) / window
                    ),
                    "trajectory_gain": (
                        sum(scores[index][-window:]) / window
                        - sum(scores[index][:window]) / window
                    ),
                    "mean_delta_norm": (
                        sum(delta_norms[index]) / len(delta_norms[index])
                    ),
                    "state_element_count": incoming_elements,
                },
            )
        )
    return tuple(records)


def _public_content_causality_diagnostic(
    policy: phase5.SkillMemoryPolicy,
    *,
    seed: int,
) -> dict[str, Any]:
    """Require learned write content to distinguish a public rule alteration."""

    permutation = demonstration_permutation_partition("train")[0]
    correct_stream = make_demonstration_procedure_transfer_stream(
        seed,
        supports_per_procedure=1,
        queries_per_procedure=1,
        position_permutation=permutation,
        mechanism_partition="train",
    )
    wrong_stream = make_demonstration_procedure_transfer_stream(
        seed,
        supports_per_procedure=1,
        queries_per_procedure=1,
        position_permutation=permutation,
        mechanism_partition="train",
        rotate_demonstration_outputs=1,
    )
    correct_task = next(
        pair.learner
        for pair in correct_stream.supports
        if pair.learner.demonstrations_visible
    )
    wrong_task = next(
        pair.learner
        for pair in wrong_stream.supports
        if pair.learner.demonstrations_visible
    )
    renamed_task = _alpha_rename_demonstrations(correct_task, salt=seed + 1)
    no_demonstration_task = replace(correct_task, demonstrations=())
    tasks = (correct_task, renamed_task, wrong_task, no_demonstration_task)
    state = policy.initial_state(1)
    proposals = tuple(
        phase5._proposal_for_candidate(policy, task, state, 0) for task in tasks
    )
    reference_logits = proposals[0].scores.logits
    if not all(
        torch.equal(reference_logits, proposal.scores.logits)
        for proposal in proposals[1:]
    ):
        raise RuntimeError("public demonstrations changed current logits")
    writes = tuple(
        phase5.propose_differentiable_feedback(
            policy,
            proposal,
            0.5,
            state,
        )
        for proposal in proposals
    )
    if len({write.write_slot for write in writes}) != 1:
        raise RuntimeError("public demonstration content changed the write address")
    evidence = tuple(proposal.scores.public_feedback_evidence for proposal in proposals)

    def rms(left: torch.Tensor, right: torch.Tensor) -> float:
        return float(torch.sqrt(torch.mean((left - right).square())).detach().item())

    same_evidence_distance = rms(evidence[0], evidence[1])
    wrong_evidence_distance = rms(evidence[0], evidence[2])
    absent_evidence_distance = rms(evidence[0], evidence[3])
    write_slot = writes[0].write_slot
    base_slots = tuple(
        write.candidate_state.slot_latents[:, write_slot, :] for write in writes
    )
    if not all(torch.equal(base_slots[0], value) for value in base_slots[1:]):
        raise RuntimeError("public content leaked into the retained base slot state")
    selected_public_codes = tuple(
        public_evidence_skill_content(write.candidate_state)[0][
            :, write_slot, :
        ]
        for write in writes
    )
    same_state_distance = rms(selected_public_codes[0], selected_public_codes[1])
    wrong_state_distance = rms(selected_public_codes[0], selected_public_codes[2])
    absent_state_distance = rms(selected_public_codes[0], selected_public_codes[3])
    gate_pass = (
        wrong_evidence_distance > same_evidence_distance
        and wrong_state_distance > same_state_distance
        and wrong_state_distance > 0.0
    )
    return {
        "public_augmentation_only": True,
        "current_logits_bit_exact_across_arms": True,
        "base_slot_latents_bit_exact_across_arms": True,
        "write_slot": write_slot,
        "shared_scalar_reward": 0.5,
        "same_mechanism_rename_evidence_rms": same_evidence_distance,
        "wrong_public_rule_evidence_rms": wrong_evidence_distance,
        "correct_vs_absent_evidence_rms": absent_evidence_distance,
        "same_mechanism_rename_public_code_rms": same_state_distance,
        "wrong_public_rule_public_code_rms": wrong_state_distance,
        "correct_vs_absent_public_code_rms": absent_state_distance,
        "maximum_absolute_evidence": max(
            float(value.abs().max().detach().item()) for value in evidence
        ),
        "gate_pass": gate_pass,
    }


def _evaluate_mechanism_panel(
    policy: phase5.SkillMemoryPolicy,
    *,
    seed: int,
    partition: str,
    supports_per_procedure: int,
    queries_per_procedure: int,
    permutations: tuple[tuple[int, ...], ...] | None = None,
) -> dict[str, Any]:
    """Evaluate learned induction on an evaluator-only mechanism partition."""

    mechanisms = (
        demonstration_permutation_partition(partition)  # type: ignore[arg-type]
        if permutations is None
        else permutations
    )
    if not mechanisms:
        raise ValueError("mechanism panel cannot be empty")
    records: list[dict[str, Any]] = []
    for index, permutation in enumerate(mechanisms):
        stream_seed = seed + 100_003 * (index + 1)
        stream = make_demonstration_procedure_transfer_stream(
            stream_seed,
            supports_per_procedure=supports_per_procedure,
            queries_per_procedure=queries_per_procedure,
            position_permutation=permutation,
            mechanism_partition=partition,
        )
        no_demonstration = make_demonstration_procedure_transfer_stream(
            stream_seed,
            supports_per_procedure=supports_per_procedure,
            queries_per_procedure=queries_per_procedure,
            position_permutation=permutation,
            mechanism_partition=partition,
            expose_transform_demonstrations=False,
        )
        wrong_demonstration = make_demonstration_procedure_transfer_stream(
            stream_seed,
            supports_per_procedure=supports_per_procedure,
            queries_per_procedure=queries_per_procedure,
            position_permutation=permutation,
            mechanism_partition=partition,
            rotate_demonstration_outputs=1,
        )
        phase5._seed_reproducible_stage(
            stream_seed,
            f"demonstration-{partition}-panel",
            next(policy.parameters()).device,
        )
        initial = policy.initial_state(1)
        with torch.inference_mode():
            arms = _acquire_matched_demonstration_arms(
                policy,
                initial,
                stream.supports,
                no_demonstration.supports,
                wrong_demonstration.supports,
            )
            acquired, acquisition = arms[0]
            no_state, no_acquisition = arms[1]
            wrong_state, wrong_acquisition = arms[2]
            full = _summary(
                _score_pairs(
                    policy,
                    acquired,
                    stream.queries,
                    score_demonstration_procedure_answer,
                )
            )
            no_score = _summary(
                _score_pairs(
                    policy,
                    no_state,
                    stream.queries,
                    score_demonstration_procedure_answer,
                )
            )
            wrong_score = _summary(
                _score_pairs(
                    policy,
                    wrong_state,
                    stream.queries,
                    score_demonstration_procedure_answer,
                )
            )
            reset = _summary(
                _score_pairs(
                    policy,
                    initial,
                    stream.queries,
                    score_demonstration_procedure_answer,
                )
            )
        records.append(
            {
                "mechanism_commitment": stream.mechanism_commitment,
                "full_mean": float(full["mean"]),
                "reset_mean": float(reset["mean"]),
                "acquired_state_gain": float(full["mean"]) - float(reset["mean"]),
                "correct_demo_gain_over_no_demo": (
                    float(full["mean"]) - float(no_score["mean"])
                ),
                "correct_demo_gain_over_wrong_demo": (
                    float(full["mean"]) - float(wrong_score["mean"])
                ),
                "no_demo_mean": float(no_score["mean"]),
                "wrong_demo_mean": float(wrong_score["mean"]),
                "accepted_transactions": acquisition["accepted_transactions"],
                "no_demo_accepted_transactions": no_acquisition[
                    "accepted_transactions"
                ],
                "wrong_demo_accepted_transactions": wrong_acquisition[
                    "accepted_transactions"
                ],
            }
        )
    gains = [float(record["acquired_state_gain"]) for record in records]
    full_means = [float(record["full_mean"]) for record in records]
    reset_means = [float(record["reset_mean"]) for record in records]
    no_demo_gains = [
        float(record["correct_demo_gain_over_no_demo"]) for record in records
    ]
    wrong_demo_gains = [
        float(record["correct_demo_gain_over_wrong_demo"]) for record in records
    ]
    return {
        "partition": partition,
        "mechanisms": len(records),
        "supports_per_procedure": supports_per_procedure,
        "queries_per_procedure": queries_per_procedure,
        "full_mean": sum(full_means) / len(full_means),
        "reset_mean": sum(reset_means) / len(reset_means),
        "acquired_state_gain_mean": sum(gains) / len(gains),
        "positive_gain_mechanisms": sum(gain > 0.0 for gain in gains),
        "negative_gain_mechanisms": sum(gain < 0.0 for gain in gains),
        "correct_demo_gain_over_no_demo_mean": (
            sum(no_demo_gains) / len(no_demo_gains)
        ),
        "correct_demo_gain_over_wrong_demo_mean": (
            sum(wrong_demo_gains) / len(wrong_demo_gains)
        ),
        "correct_beats_no_demo_mechanisms": sum(gain > 0.0 for gain in no_demo_gains),
        "correct_beats_wrong_demo_mechanisms": sum(
            gain > 0.0 for gain in wrong_demo_gains
        ),
        "records": records,
    }


def _protected_state_fingerprint(policy: phase5.SkillMemoryPolicy) -> str:
    return phase5._named_state_fingerprint(
        policy,
        include=lambda name: not _is_demonstration_trainable(name),
        domain=b"project-angler.demonstration-transfer-protected.v2",
    )


def _protected_parameter_identity(policy: phase5.SkillMemoryPolicy) -> str:
    digest = hashlib.sha256(b"project-angler.demonstration-protected-identity.v2\x00")
    selected = 0
    for name, parameter in policy.named_parameters():
        if _is_demonstration_trainable(name):
            continue
        encoded = (
            f"{name}\x00{id(parameter)}\x00{parameter.data_ptr()}\x00"
            f"{tuple(parameter.shape)}\x00{parameter.dtype}\x00{parameter.device}"
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        selected += 1
    if not selected:
        raise RuntimeError("protected identity selected no parameters")
    return "sha256:" + digest.hexdigest()


def _train_demonstration_adapter(
    policy: phase5.SkillMemoryPolicy,
    *,
    seed: int,
    meta_steps: int,
    supports_per_procedure: int,
    queries_per_procedure: int,
    learning_rate: float,
) -> dict[str, Any]:
    if isinstance(meta_steps, bool) or not isinstance(meta_steps, int) or meta_steps < 2:
        raise ValueError("meta_steps must be an integer of at least two")
    training_permutations = list(demonstration_permutation_partition("train"))
    selected: list[tuple[int, ...]] = []
    pass_index = 0
    while len(selected) < meta_steps:
        pass_permutations = list(training_permutations)
        random.Random(seed + 1_000_003 * pass_index).shuffle(pass_permutations)
        selected.extend(pass_permutations[: meta_steps - len(selected)])
        pass_index += 1
    selected_permutations = tuple(selected)
    ports = getattr(policy, "public_fact_adapter", None)
    if not isinstance(ports, TypedPublicFactPorts):
        raise RuntimeError("typed public-fact ports are not attached")
    for name, parameter in policy.named_parameters():
        parameter.requires_grad_(_is_demonstration_trainable(name))
    trainable = tuple(parameter for parameter in policy.parameters() if parameter.requires_grad)
    expected = {
        name for name, _ in policy.named_parameters() if _is_demonstration_trainable(name)
    }
    actual = {name for name, parameter in policy.named_parameters() if parameter.requires_grad}
    if actual != expected:
        raise RuntimeError("demonstration training exposed an undeclared parameter")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    protected_before = _protected_state_fingerprint(policy)
    identity_before = _protected_parameter_identity(policy)
    adapter_before = phase5._named_state_fingerprint(
        policy,
        include=lambda name: _is_demonstration_trainable(name),
        domain=b"project-angler.demonstration-interface.v2",
    )
    losses: list[float] = []
    task_losses: list[float] = []
    causal_margin_losses: list[float] = []
    no_demo_margin_losses: list[float] = []
    wrong_demo_margin_losses: list[float] = []
    representation_losses: list[float] = []
    representation_gaps: list[float] = []
    write_representation_losses: list[float] = []
    write_representation_gaps: list[float] = []
    gradient_norms: list[float] = []
    mechanism_pass_size = len(training_permutations)
    causal_pass_accumulators = [
        {
            "pass_index": pass_index + 1,
            "start_meta_step": bounds[0],
            "end_meta_step": bounds[1],
            "correct_aligned_edges": 0,
            "correct_edges": 0,
            "wrong_aligned_edges": 0,
            "wrong_edges": 0,
            "correct_logit_delta_sum": 0.0,
            "wrong_logit_delta_sum": 0.0,
            "logit_delta_queries": 0,
            "correct_gate_sum": 0.0,
            "wrong_gate_sum": 0.0,
            "gate_queries": 0,
        }
        for pass_index, bounds in enumerate(
            _causal_pass_bounds(meta_steps, mechanism_pass_size)
        )
    ]
    gradient_reach = {
        "entity_encoder": False,
        "correspondence": False,
        "relation_encoder": False,
        "evidence_projection": False,
        "public_reader_hidden": False,
        "public_reader_output": False,
        "public_transition_gate": False,
    }
    gradient_prefixes = {
        "entity_encoder": _DEMONSTRATION_ADAPTER_PREFIX + "entity_encoder.",
        "correspondence": _DEMONSTRATION_ADAPTER_PREFIX + "correspondence",
        "relation_encoder": _DEMONSTRATION_ADAPTER_PREFIX + "relation_",
        "evidence_projection": (
            _DEMONSTRATION_ADAPTER_PREFIX + "evidence_projection."
        ),
        "public_reader_hidden": _PUBLIC_EVIDENCE_READER_PREFIX + "hidden.",
        "public_reader_output": _PUBLIC_EVIDENCE_READER_PREFIX + "output.",
        "public_transition_gate": (
            _PUBLIC_EVIDENCE_READER_PREFIX + "transition_output."
        ),
    }
    informative_steps = 0
    ports.demonstration_adapter.train()
    policy.composition_memory.public_evidence_reader.train()
    mechanism_commitments: list[str] = []
    for step, position_permutation in enumerate(selected_permutations):
        episode_seed = seed + 100_003 * (step + 1)
        phase5._seed_reproducible_stage(
            episode_seed,
            "demonstration-adapter-training",
            next(policy.parameters()).device,
        )
        stream = make_demonstration_procedure_transfer_stream(
            episode_seed,
            supports_per_procedure=supports_per_procedure,
            queries_per_procedure=queries_per_procedure,
            position_permutation=position_permutation,
            mechanism_partition="train",
        )
        no_demonstration_stream = make_demonstration_procedure_transfer_stream(
            episode_seed,
            supports_per_procedure=supports_per_procedure,
            queries_per_procedure=queries_per_procedure,
            position_permutation=position_permutation,
            mechanism_partition="train",
            expose_transform_demonstrations=False,
        )
        wrong_demonstration_stream = make_demonstration_procedure_transfer_stream(
            episode_seed,
            supports_per_procedure=supports_per_procedure,
            queries_per_procedure=queries_per_procedure,
            position_permutation=position_permutation,
            mechanism_partition="train",
            rotate_demonstration_outputs=1,
        )
        mechanism_commitments.append(stream.mechanism_commitment)
        state = policy.initial_state(1)
        no_demonstration_state = policy.initial_state(1)
        wrong_demonstration_state = policy.initial_state(1)
        for pair, no_pair, wrong_pair in zip(
            stream.supports,
            no_demonstration_stream.supports,
            wrong_demonstration_stream.supports,
            strict=True,
        ):
            proposal = phase5.propose_task(
                policy,
                pair.learner,
                state,
                greedy=False,
                temperature=1.25,
            )
            reward = score_demonstration_procedure_answer(
                pair.learner,
                pair.hidden,
                proposal.answer,
            )
            state = phase5.propose_differentiable_feedback(
                policy,
                proposal,
                reward,
                state,
            ).candidate_state
            with torch.no_grad():
                no_proposal = phase5._proposal_for_candidate(
                    policy,
                    no_pair.learner,
                    no_demonstration_state,
                    proposal.candidate_index,
                )
                no_demonstration_state = phase5.propose_differentiable_feedback(
                    policy,
                    no_proposal,
                    reward,
                    no_demonstration_state,
                ).candidate_state
            wrong_proposal = phase5._proposal_for_candidate(
                policy,
                wrong_pair.learner,
                wrong_demonstration_state,
                proposal.candidate_index,
            )
            wrong_demonstration_state = phase5.propose_differentiable_feedback(
                policy,
                wrong_proposal,
                reward,
                wrong_demonstration_state,
            ).candidate_state
        query_losses: list[torch.Tensor] = []
        no_query_losses: list[torch.Tensor] = []
        wrong_query_losses: list[torch.Tensor] = []
        zeroed_state = differentiable_zero_public_evidence_skill_content(state)
        wrong_zeroed_state = differentiable_zero_public_evidence_skill_content(
            wrong_demonstration_state
        )
        informative = False
        causal_pass = causal_pass_accumulators[step // mechanism_pass_size]
        for query_index, (pair, no_pair, wrong_pair) in enumerate(
            zip(
                stream.queries,
                no_demonstration_stream.queries,
                wrong_demonstration_stream.queries,
                strict=True,
            )
        ):
            if pair.learner.demonstrations_visible:
                raise RuntimeError("training query unexpectedly exposes demonstrations")
            scores = policy.score_task(pair.learner, state)
            zeroed_scores = policy.score_task(pair.learner, zeroed_state)
            candidates = phase5._on_policy_reward_candidate_set(
                scores.logits,
                step,
                query_index,
            )
            rewards = phase5._scalar_attempt_scores(
                pair,
                candidates,
                score_demonstration_procedure_answer,
            )
            public_delta = scores.logits - zeroed_scores.logits
            public_loss, observed_edges = phase5._scalar_multi_preference_loss(
                public_delta,
                candidates,
                rewards,
            )
            query_losses.append(public_loss)
            zero_loss, _ = phase5._scalar_multi_preference_loss(
                torch.zeros_like(public_delta),
                candidates,
                rewards,
            )
            no_query_losses.append(zero_loss)
            with torch.no_grad():
                no_scores = policy.score_task(
                    no_pair.learner,
                    no_demonstration_state,
                )
            if not torch.equal(zeroed_scores.logits.detach(), no_scores.logits):
                raise RuntimeError(
                    "public-zero counterfactual differs from the matched absent arm"
                )
            wrong_scores = policy.score_task(
                wrong_pair.learner,
                wrong_demonstration_state,
            )
            wrong_zeroed_scores = policy.score_task(
                wrong_pair.learner,
                wrong_zeroed_state,
            )
            if not torch.equal(
                wrong_zeroed_scores.logits.detach(),
                no_scores.logits,
            ):
                raise RuntimeError(
                    "wrong public-zero counterfactual differs from the absent arm"
                )
            wrong_public_delta = wrong_scores.logits - wrong_zeroed_scores.logits
            correct_aligned, correct_edges = _public_delta_preference_alignment(
                public_delta,
                tuple(candidates),
                tuple(rewards),
            )
            wrong_aligned, wrong_edges = _public_delta_preference_alignment(
                wrong_public_delta,
                tuple(candidates),
                tuple(rewards),
            )
            correct_gate = _root_public_transition_gate_mean_absolute(scores)
            wrong_gate = _root_public_transition_gate_mean_absolute(wrong_scores)
            if (correct_gate is None) != (wrong_gate is None):
                raise RuntimeError(
                    "matched public transition gate diagnostics are inconsistent"
                )
            causal_pass["correct_aligned_edges"] += correct_aligned
            causal_pass["correct_edges"] += correct_edges
            causal_pass["wrong_aligned_edges"] += wrong_aligned
            causal_pass["wrong_edges"] += wrong_edges
            causal_pass["correct_logit_delta_sum"] += float(
                public_delta.detach().abs().mean().item()
            )
            causal_pass["wrong_logit_delta_sum"] += float(
                wrong_public_delta.detach().abs().mean().item()
            )
            causal_pass["logit_delta_queries"] += 1
            if correct_gate is not None and wrong_gate is not None:
                causal_pass["correct_gate_sum"] += correct_gate
                causal_pass["wrong_gate_sum"] += wrong_gate
                causal_pass["gate_queries"] += 1
            wrong_public_loss, wrong_observed_edges = (
                phase5._scalar_multi_preference_loss(
                    wrong_public_delta,
                    candidates,
                    rewards,
                )
            )
            wrong_query_losses.append(wrong_public_loss)
            informative = (
                informative
                or observed_edges > 0
                or wrong_observed_edges > 0
            )
        task_loss = torch.stack(query_losses).mean()
        no_task_loss = torch.stack(no_query_losses).mean()
        wrong_task_loss = torch.stack(wrong_query_losses).mean()
        no_demo_margin_loss = torch.relu(
            task_loss - no_task_loss + 0.05
        )
        wrong_demo_margin_loss = torch.relu(
            task_loss - wrong_task_loss + 0.05
        )
        causal_margin_loss = no_demo_margin_loss + wrong_demo_margin_loss

        transform_tasks = [
            pair.learner
            for pair in stream.supports
            if pair.learner.demonstrations_visible
        ]
        wrong_transform_tasks = [
            pair.learner
            for pair in wrong_demonstration_stream.supports
            if pair.learner.demonstrations_visible
        ]
        if len(transform_tasks) < 1 or len(wrong_transform_tasks) < 1:
            raise RuntimeError("contrastive training requires a transform support")
        reference = next(policy.parameters()).new_zeros((1, 64))
        anchor = ports.demonstration_adapter.feedback_evidence(
            transform_tasks[0],
            reference,
        )
        positive_task = _alpha_rename_demonstrations(
            transform_tasks[0],
            salt=episode_seed,
        )
        positive = ports.demonstration_adapter.feedback_evidence(
            positive_task,
            reference,
        )
        negative = ports.demonstration_adapter.feedback_evidence(
            wrong_transform_tasks[0],
            reference,
        )
        positive_distance = 1.0 - torch.nn.functional.cosine_similarity(
            anchor,
            positive,
        ).mean()
        negative_distance = 1.0 - torch.nn.functional.cosine_similarity(
            anchor,
            negative,
        ).mean()
        representation_loss = 0.25 * positive_distance + 0.05 * (
            torch.nn.functional.softplus(
                (0.20 + positive_distance - negative_distance) / 0.05
            )
        )
        representation_gap = negative_distance - positive_distance

        contrastive_state = policy.initial_state(1)
        no_demonstration_task = replace(
            transform_tasks[0],
            demonstrations=(),
        )
        contrastive_tasks = (
            transform_tasks[0],
            positive_task,
            wrong_transform_tasks[0],
            no_demonstration_task,
        )
        contrastive_proposals = tuple(
            phase5._proposal_for_candidate(
                policy,
                task,
                contrastive_state,
                0,
            )
            for task in contrastive_tasks
        )
        if not all(
            torch.equal(
                contrastive_proposals[0].scores.logits,
                proposal.scores.logits,
            )
            for proposal in contrastive_proposals[1:]
        ):
            raise RuntimeError("contrastive public content changed current logits")
        contrastive_writes = tuple(
            phase5.propose_differentiable_feedback(
                policy,
                proposal,
                0.5,
                contrastive_state,
            )
            for proposal in contrastive_proposals
        )
        if len({write.write_slot for write in contrastive_writes}) != 1:
            raise RuntimeError("contrastive public content changed the write address")
        write_slot = contrastive_writes[0].write_slot
        base_written_slots = tuple(
            write.candidate_state.slot_latents[:, write_slot, :]
            for write in contrastive_writes
        )
        if not all(
            torch.equal(base_written_slots[0], value)
            for value in base_written_slots[1:]
        ):
            raise RuntimeError("public content leaked into retained slot latents")
        public_write_codes = tuple(
            public_evidence_skill_content(write.candidate_state)[0][
                :, write_slot, :
            ]
            for write in contrastive_writes[:3]
        )
        write_positive_distance = 1.0 - torch.nn.functional.cosine_similarity(
            public_write_codes[0],
            public_write_codes[1],
        ).mean()
        write_negative_distance = 1.0 - torch.nn.functional.cosine_similarity(
            public_write_codes[0],
            public_write_codes[2],
        ).mean()
        write_representation_loss = 0.25 * write_positive_distance + 0.05 * (
            torch.nn.functional.softplus(
                (
                    0.20
                    + write_positive_distance
                    - write_negative_distance
                )
                / 0.05
            )
        )
        write_representation_gap = (
            write_negative_distance - write_positive_distance
        )
        loss = (
            task_loss
            + causal_margin_loss
            + representation_loss
            + 0.5 * write_representation_loss
        )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("demonstration training produced non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        step_has_gradient = False
        for name, parameter in policy.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            if not bool(torch.isfinite(parameter.grad).all().item()):
                raise RuntimeError("demonstration training produced a non-finite gradient")
            nonzero = bool(parameter.grad.detach().count_nonzero())
            step_has_gradient = step_has_gradient or nonzero
            if nonzero:
                for group, prefix in gradient_prefixes.items():
                    if name.startswith(prefix):
                        gradient_reach[group] = True
        if not step_has_gradient:
            raise RuntimeError("later scalar outcomes did not reach demonstration input")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            5.0,
            error_if_nonfinite=True,
        )
        optimizer.step()
        losses.append(float(loss.detach().item()))
        task_losses.append(float(task_loss.detach().item()))
        causal_margin_losses.append(float(causal_margin_loss.detach().item()))
        no_demo_margin_losses.append(float(no_demo_margin_loss.detach().item()))
        wrong_demo_margin_losses.append(float(wrong_demo_margin_loss.detach().item()))
        representation_losses.append(float(representation_loss.detach().item()))
        representation_gaps.append(float(representation_gap.detach().item()))
        write_representation_losses.append(
            float(write_representation_loss.detach().item())
        )
        write_representation_gaps.append(
            float(write_representation_gap.detach().item())
        )
        gradient_norms.append(float(gradient_norm.detach().item()))
        informative_steps += int(informative)
    ports.demonstration_adapter.eval()
    policy.composition_memory.public_evidence_reader.eval()
    if not all(gradient_reach.values()):
        missing = sorted(group for group, reached in gradient_reach.items() if not reached)
        raise RuntimeError(
            "demonstration objective did not reach every learned block: "
            + ", ".join(missing)
        )
    policy.requires_grad_(False)
    adapter_after = phase5._named_state_fingerprint(
        policy,
        include=lambda name: _is_demonstration_trainable(name),
        domain=b"project-angler.demonstration-interface.v2",
    )
    if _protected_state_fingerprint(policy) != protected_before:
        raise RuntimeError("demonstration training changed protected prior capability")
    if _protected_parameter_identity(policy) != identity_before:
        raise RuntimeError("demonstration training replaced a protected parameter")
    if adapter_after == adapter_before:
        raise RuntimeError("demonstration adapter did not change")
    causal_pass_metrics: list[dict[str, Any]] = []
    for accumulator in causal_pass_accumulators:
        query_count = int(accumulator["logit_delta_queries"])
        gate_count = int(accumulator["gate_queries"])
        correct_edges = int(accumulator["correct_edges"])
        wrong_edges = int(accumulator["wrong_edges"])
        if query_count < 1 or correct_edges < 1 or wrong_edges < 1:
            raise RuntimeError("causal pass diagnostic has no informative queries")
        if gate_count not in (0, query_count):
            raise RuntimeError("causal pass gate coverage is incomplete")
        correct_alignment = (
            int(accumulator["correct_aligned_edges"]) / correct_edges
        )
        wrong_alignment = int(accumulator["wrong_aligned_edges"]) / wrong_edges
        causal_pass_metrics.append(
            {
                "pass_index": int(accumulator["pass_index"]),
                "start_meta_step": int(accumulator["start_meta_step"]),
                "end_meta_step": int(accumulator["end_meta_step"]),
                "mechanisms": (
                    int(accumulator["end_meta_step"])
                    - int(accumulator["start_meta_step"])
                    + 1
                ),
                "queries": query_count,
                "correct_public_delta_aligned_edges": int(
                    accumulator["correct_aligned_edges"]
                ),
                "correct_public_delta_preference_edges": correct_edges,
                "correct_public_delta_preference_alignment": correct_alignment,
                "wrong_public_delta_aligned_edges": int(
                    accumulator["wrong_aligned_edges"]
                ),
                "wrong_public_delta_preference_edges": wrong_edges,
                "wrong_public_delta_preference_alignment": wrong_alignment,
                "correct_minus_wrong_preference_alignment": (
                    correct_alignment - wrong_alignment
                ),
                "correct_public_logit_delta_mean_absolute": (
                    float(accumulator["correct_logit_delta_sum"]) / query_count
                ),
                "wrong_public_logit_delta_mean_absolute": (
                    float(accumulator["wrong_logit_delta_sum"]) / query_count
                ),
                "correct_public_transition_gate_mean_absolute": (
                    None
                    if gate_count == 0
                    else float(accumulator["correct_gate_sum"]) / gate_count
                ),
                "wrong_public_transition_gate_mean_absolute": (
                    None
                    if gate_count == 0
                    else float(accumulator["wrong_gate_sum"]) / gate_count
                ),
            }
        )
    return {
        "meta_steps": meta_steps,
        "fresh_opaque_mapping_episodes": meta_steps,
        "unique_training_mechanisms": len(set(mechanism_commitments)),
        "effective_partition_passes": meta_steps / len(training_permutations),
        "causal_pass_metrics": causal_pass_metrics,
        "mechanism_partition": "train",
        "mechanism_partition_size": len(
            demonstration_permutation_partition("train")
        ),
        "mechanism_commitments_digest": "sha256:"
        + hashlib.sha256(
            json.dumps(
                mechanism_commitments,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "supports_per_procedure_per_mapping": supports_per_procedure,
        "demonstration_free_queries_per_mapping": queries_per_procedure,
        "attempted_outputs_per_query": 4,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "minimum_loss": min(losses),
        "first_task_loss": task_losses[0],
        "last_task_loss": task_losses[-1],
        "last_causal_margin_loss": causal_margin_losses[-1],
        "last_no_demo_margin_loss": no_demo_margin_losses[-1],
        "last_wrong_demo_margin_loss": wrong_demo_margin_losses[-1],
        "first_representation_loss": representation_losses[0],
        "last_representation_loss": representation_losses[-1],
        "first_representation_gap": representation_gaps[0],
        "last_representation_gap": representation_gaps[-1],
        "first_write_representation_loss": write_representation_losses[0],
        "last_write_representation_loss": write_representation_losses[-1],
        "first_write_representation_gap": write_representation_gaps[0],
        "last_write_representation_gap": write_representation_gaps[-1],
        "mean_gradient_norm": sum(gradient_norms) / len(gradient_norms),
        "gradient_reach": dict(sorted(gradient_reach.items())),
        "informative_reward_steps": informative_steps,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "trainable_sensory_parameter_count": sum(
            parameter.numel()
            for name, parameter in policy.named_parameters()
            if name.startswith(_DEMONSTRATION_ADAPTER_PREFIX)
        ),
        "trainable_public_reader_parameter_count": sum(
            parameter.numel()
            for name, parameter in policy.named_parameters()
            if name.startswith(_PUBLIC_EVIDENCE_READER_PREFIX)
        ),
        "adapter_fingerprint_before": adapter_before,
        "adapter_fingerprint_after": adapter_after,
        "protected_fingerprint_before": protected_before,
        "protected_fingerprint_after": _protected_state_fingerprint(policy),
        "training_signal": (
            "evaluator-only paired public-residual learning subtracts a "
            "differentiable zero-public counterfactual, then uses matched "
            "correct/absent/wrong supports, later query attempts with scalar "
            "outcomes, and public-only sensory/write anti-collapse losses"
        ),
        "wrong_demonstration_gradient_contrast": True,
        "counterfactual_public_reader": True,
        "paired_public_residual_objective": True,
        "post_saturation_public_transition_gate": True,
        "matched_control_candidate_and_reward": True,
        "evaluator_calls_per_unique_support": 1,
        "matched_controls_reuse_single_scalar_feedback": True,
        "public_augmentation_only_contrastive_sampler": True,
        "target_order_used": False,
        "deterministic_solver_used": False,
        "prior_family_replay_used": False,
    }


def run(
    *,
    seed: int = 95_001,
    device: str | torch.device = "cpu",
    initial_checkpoint: str | Path,
    precedence_adapter_checkpoint: str | Path,
    compiler_checkpoint: str | Path = phase5._PHASE4_CHECKPOINT,
    meta_steps: int = 64,
    meta_supports_per_procedure: int = 8,
    meta_queries_per_procedure: int = 8,
    learning_rate: float = 8.0e-4,
    online_supports_per_procedure: int = 64,
    online_queries: int = 40,
    sensory_initialization_checkpoint: str | Path | None = None,
    trained_adapter_checkpoint: str | Path | None = None,
    checkpoint: str | Path | None = None,
    state_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    target_device = torch.device(device)
    random.seed(seed)
    torch.manual_seed(seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    settings = phase5._PROFILES["composition"]
    compiler, compiler_record = phase5._load_phase4_compiler(compiler_checkpoint)
    policy = phase5.SkillMemoryPolicy(settings, compiler).to(
        device=target_device,
        dtype=torch.float32,
    )
    initialization = phase5._load_initial_policy_checkpoint(
        policy,
        initial_checkpoint,
        settings,
    )
    if not bool(policy.reversible_transition_mode.item()) or (
        initialization.get("source_stage") != "reversible_transition_acquisition"
    ):
        raise RuntimeError("demonstration transfer requires the retained V51 core")
    precedence_adapter, precedence_record = _load_precedence_adapter(
        precedence_adapter_checkpoint,
        expected_base_sha256=initialization["sha256"],
    )
    if (
        trained_adapter_checkpoint is not None
        and sensory_initialization_checkpoint is not None
    ):
        raise ValueError(
            "trained adapter and sensory initialization checkpoints are exclusive"
        )
    if trained_adapter_checkpoint is None:
        if sensory_initialization_checkpoint is None:
            demonstration_adapter = SymbolicDemonstrationAdapter()
            adapter_initialization: dict[str, Any] = {
                "mode": "trained_from_fresh_sensory_port"
            }
        else:
            (
                demonstration_adapter,
                sensory_initialization,
            ) = _load_v5_sensory_initialization(
                sensory_initialization_checkpoint,
                expected_base_sha256=initialization["sha256"],
                expected_precedence_sha256=precedence_record["sha256"],
            )
            adapter_initialization = {
                "mode": "trained_from_v5_sensory_port",
                **sensory_initialization,
            }
        loaded_reader_state = None
        loaded_training = None
    else:
        (
            demonstration_adapter,
            loaded_reader_state,
            loaded_training,
            loaded_record,
        ) = _load_demonstration_adapter(
            trained_adapter_checkpoint,
            expected_base_sha256=initialization["sha256"],
            expected_precedence_sha256=precedence_record["sha256"],
        )
        adapter_initialization = {"mode": "loaded", **loaded_record}
    policy.public_fact_adapter = TypedPublicFactPorts(
        precedence_adapter,
        demonstration_adapter,
    ).to(device=target_device, dtype=torch.float32)
    _attach_public_evidence_reader(policy)
    if loaded_reader_state is not None:
        _load_public_evidence_reader_state(policy, loaded_reader_state)
    policy.requires_grad_(False)
    protected_before = _protected_state_fingerprint(policy)
    protected_identity_before = _protected_parameter_identity(policy)
    if loaded_training is None:
        training = _train_demonstration_adapter(
            policy,
            seed=seed + 3_000_003,
            meta_steps=meta_steps,
            supports_per_procedure=meta_supports_per_procedure,
            queries_per_procedure=meta_queries_per_procedure,
            learning_rate=learning_rate,
        )
        if checkpoint is None:
            train_checkpoint_record = None
            adapter_initialization = {
                "mode": "trained_in_memory_without_immutable_boundary"
            }
        else:
            saved_record = _save_train_only_adapter(
                checkpoint,
                policy=policy,
                seed=seed,
                base_checkpoint_sha256=initialization["sha256"],
                precedence_adapter_sha256=precedence_record["sha256"],
                training=training,
                sensory_initialization=adapter_initialization,
            )
            (
                reloaded_adapter,
                reloaded_reader_state,
                reloaded_training,
                reloaded_record,
            ) = _load_demonstration_adapter(
                checkpoint,
                expected_base_sha256=initialization["sha256"],
                expected_precedence_sha256=precedence_record["sha256"],
            )
            if reloaded_training != training or (
                reloaded_record["sha256"] != saved_record["sha256"]
            ):
                raise RuntimeError("train-only adapter round trip changed evidence")
            policy.public_fact_adapter = TypedPublicFactPorts(
                precedence_adapter,
                reloaded_adapter,
            ).to(device=target_device, dtype=torch.float32)
            _load_public_evidence_reader_state(policy, reloaded_reader_state)
            policy.requires_grad_(False)
            adapter_initialization = {
                "mode": "trained_and_reloaded_before_evaluation",
                **reloaded_record,
            }
            train_checkpoint_record = reloaded_record
    else:
        if checkpoint is not None:
            raise ValueError(
                "checkpoint cannot be supplied when loading a trained adapter"
            )
        training = loaded_training
        train_checkpoint_record = dict(adapter_initialization)
        if adapter_initialization.get("protected_fingerprint") != protected_before:
            raise RuntimeError(
                "loaded demonstration adapter binds different protected capability"
            )
    loaded_adapter_fingerprint = phase5._named_state_fingerprint(
        policy,
        include=lambda name: _is_demonstration_trainable(name),
        domain=b"project-angler.demonstration-interface.v2",
    )
    if training.get("adapter_fingerprint_after") != loaded_adapter_fingerprint:
        raise RuntimeError("demonstration adapter fingerprint is invalid")
    if (
        training.get("mechanism_partition") != "train"
        or training.get("target_order_used") is not False
        or training.get("deterministic_solver_used") is not False
    ):
        raise RuntimeError("demonstration training record is invalid")
    if _protected_state_fingerprint(policy) != protected_before:
        raise RuntimeError("adapter freeze boundary changed protected capability")
    policy.eval()
    training_content_causality = _public_content_causality_diagnostic(
        policy,
        seed=seed + 3_300_003,
    )
    if not training_content_causality["gate_pass"]:
        raise RuntimeError(
            "learned demonstration content failed the pre-final causal gate"
        )
    training_diagnostic_panel = _evaluate_mechanism_panel(
        policy,
        seed=seed + 3_500_003,
        partition="train",
        supports_per_procedure=meta_supports_per_procedure,
        queries_per_procedure=meta_queries_per_procedure,
        permutations=demonstration_permutation_partition("train")[:16],
    )

    development_permutations = demonstration_permutation_partition("development")
    adaptive_development_panel = _evaluate_mechanism_panel(
        policy,
        seed=seed + 3_700_003,
        partition="development",
        supports_per_procedure=meta_supports_per_procedure,
        queries_per_procedure=meta_queries_per_procedure,
        permutations=development_permutations,
    )

    old_curriculum = make_skill_memory_composition_curriculum(
        seed + 1_000_003,
        encounters_per_primitive=8,
        cases_per_component_probe=8,
        cases_per_composition=8,
    )
    relational = make_relational_procedure_transfer_stream(
        seed + 2_000_003,
        supports_per_procedure=64,
        queries_per_procedure=40,
    )
    evaluation_permutation = development_permutations[
        seed % len(development_permutations)
    ]
    demonstration = make_demonstration_procedure_transfer_stream(
        seed + 4_000_003,
        supports_per_procedure=online_supports_per_procedure,
        queries_per_procedure=online_queries,
        position_permutation=evaluation_permutation,
        mechanism_partition="development",
    )
    no_demo_control = make_demonstration_procedure_transfer_stream(
        seed + 4_000_003,
        supports_per_procedure=online_supports_per_procedure,
        queries_per_procedure=online_queries,
        position_permutation=evaluation_permutation,
        mechanism_partition="development",
        expose_transform_demonstrations=False,
    )
    wrong_demo_control = make_demonstration_procedure_transfer_stream(
        seed + 4_000_003,
        supports_per_procedure=online_supports_per_procedure,
        queries_per_procedure=online_queries,
        position_permutation=evaluation_permutation,
        mechanism_partition="development",
        rotate_demonstration_outputs=1,
    )
    phase5._seed_reproducible_stage(
        seed + 5_000_003,
        "three-family-online-evaluation",
        target_device,
    )
    state = policy.initial_state(1)
    initial_elements = _state_element_count(state)
    state, old_acquisition = _acquire_pairs(
        policy,
        state,
        old_curriculum.component_supports,
        score_skill_memory_answer,
    )
    old_after_old_acquisition = _summary(
        _score_pairs(
            policy,
            state,
            old_curriculum.composition_queries,
            score_skill_memory_answer,
        )
    )
    state, relational_acquisition = _acquire_pairs(
        policy,
        state,
        relational.supports,
        score_relational_procedure_answer,
    )
    relational_before = _summary(
        _score_pairs(
            policy,
            state,
            relational.queries,
            score_relational_procedure_answer,
        )
    )
    old_before = _summary(
        _score_pairs(
            policy,
            state,
            old_curriculum.composition_queries,
            score_skill_memory_answer,
        )
    )
    pre_demonstration_state = state
    state_before_demonstration = procedural_skill_state_digest(
        pre_demonstration_state
    )
    phase5._seed_reproducible_stage(
        seed + 6_000_003,
        "demonstration-online-acquisition",
        target_device,
    )
    demonstration_arms = _acquire_matched_demonstration_arms(
        policy,
        pre_demonstration_state,
        demonstration.supports,
        no_demo_control.supports,
        wrong_demo_control.supports,
    )
    state, demonstration_acquisition = demonstration_arms[0]
    no_demo_state, no_demo_acquisition = demonstration_arms[1]
    wrong_demo_state, wrong_demo_acquisition = demonstration_arms[2]
    state_after_demonstration = procedural_skill_state_digest(state)
    if _state_element_count(state) != initial_elements:
        raise RuntimeError("three-family stream changed competence capacity")

    full_values = _score_pairs(
        policy,
        state,
        demonstration.queries,
        score_demonstration_procedure_answer,
    )
    pre_acquisition_values = _score_pairs(
        policy,
        pre_demonstration_state,
        demonstration.queries,
        score_demonstration_procedure_answer,
    )
    fresh_state = policy.initial_state(1)
    fresh_values = _score_pairs(
        policy,
        fresh_state,
        demonstration.queries,
        score_demonstration_procedure_answer,
    )
    removed_values = _score_pairs(
        policy,
        state,
        demonstration.queries,
        score_demonstration_procedure_answer,
        include_reversible_transition=False,
    )
    public_code_zero_values = _score_pairs(
        policy,
        zero_public_evidence_skill_content(state),
        demonstration.queries,
        score_demonstration_procedure_answer,
    )
    no_demo_values = _score_pairs(
        policy,
        no_demo_state,
        demonstration.queries,
        score_demonstration_procedure_answer,
    )
    wrong_demo_values = _score_pairs(
        policy,
        wrong_demo_state,
        demonstration.queries,
        score_demonstration_procedure_answer,
    )
    old_after = _summary(
        _score_pairs(
            policy,
            state,
            old_curriculum.composition_queries,
            score_skill_memory_answer,
        )
    )
    relational_after = _summary(
        _score_pairs(
            policy,
            state,
            relational.queries,
            score_relational_procedure_answer,
        )
    )
    full = _summary(full_values)
    pre_acquisition = _summary(pre_acquisition_values)
    fresh = _summary(fresh_values)
    removed = _summary(removed_values)
    public_code_zero = _summary(public_code_zero_values)
    no_demo = _summary(no_demo_values)
    wrong_demo = _summary(wrong_demo_values)
    logit_interventions = _logit_intervention_summary(
        policy,
        state,
        pre_demonstration_state,
        demonstration.queries,
    )
    _, public_write_counts = public_evidence_skill_content(state)
    result: dict[str, Any] = {
        "report_version": _REPORT_VERSION,
        "seed": seed,
        "device": str(target_device),
        "initial_checkpoint": {
            "path": str(initial_checkpoint),
            "sha256": initialization["sha256"],
            "result_digest": initialization["result_digest"],
        },
        "precedence_adapter_checkpoint": precedence_record,
        "demonstration_adapter_initialization": adapter_initialization,
        "train_only_adapter_checkpoint": train_checkpoint_record,
        "compiler_checkpoint": compiler_record,
        "training": training,
        "training_content_causality": training_content_causality,
        "training_diagnostic_panel": training_diagnostic_panel,
        "adaptive_development_panel": adaptive_development_panel,
        "evaluation_identity": {
            "single_permutation_s5_mechanisms_are_adaptive_evidence": True,
            "development_mechanisms": len(development_permutations),
            "scientific_successor_identity": (
                "precommitted conditional two-permutation bindings"
            ),
            "scientific_successor_opened": False,
        },
        "old_family": {
            "acquisition": old_acquisition,
            "after_old_family_acquisition": old_after_old_acquisition,
            "before_demonstration_family": old_before,
            "after_demonstration_family": old_after,
            "precedence_family_interaction_delta": (
                float(old_before["mean"])
                - float(old_after_old_acquisition["mean"])
            ),
            "retention_delta": float(old_after["mean"]) - float(old_before["mean"]),
        },
        "precedence_family": {
            "acquisition": relational_acquisition,
            "before_demonstration_family": relational_before,
            "after_demonstration_family": relational_after,
            "retention_delta": (
                float(relational_after["mean"]) - float(relational_before["mean"])
            ),
        },
        "demonstration_family": {
            "acquisition": demonstration_acquisition,
            "identity_support_demonstrations": 0,
            "transform_support_demonstrations": 2,
            "demonstrations_per_query": 0,
            "support_mix": {
                "identity_supports": online_supports_per_procedure,
                "transform_supports": online_supports_per_procedure,
                "aggregate_metrics_include_both": True,
            },
            "mechanism_partition": demonstration.mechanism_partition,
            "mechanism_commitment": demonstration.mechanism_commitment,
            "mechanism_excluded_from_meta_training_partition": True,
            "evaluation_is_engineering_diagnostic": True,
            "full": full,
            "pre_acquisition_state": pre_acquisition,
            "fresh_state": fresh,
            "reversible_transition_removed": removed,
            "public_evidence_code_zeroed": public_code_zero,
            "no_demonstration_control": {
                "acquisition": no_demo_acquisition,
                "score": no_demo,
            },
            "wrong_demonstration_control": {
                "acquisition": wrong_demo_acquisition,
                "score": wrong_demo,
                "uses_fixed_public_output_rotation": True,
                "uses_hidden_mechanism_identity": False,
            },
            "acquired_state_gain": (
                float(full["mean"]) - float(pre_acquisition["mean"])
            ),
            "reversible_transition_gain": (
                float(full["mean"]) - float(removed["mean"])
            ),
            "public_evidence_code_gain": (
                float(full["mean"]) - float(public_code_zero["mean"])
            ),
            "demonstration_content_gain_over_no_demo": (
                float(full["mean"]) - float(no_demo["mean"])
            ),
            "correct_demo_gain_over_wrong_demo": (
                float(full["mean"]) - float(wrong_demo["mean"])
            ),
            "logit_interventions": logit_interventions,
        },
        "state": {
            "before_demonstration_digest": state_before_demonstration,
            "after_demonstration_digest": state_after_demonstration,
            "element_count": initial_elements,
            "constant_capacity": _state_element_count(state) == initial_elements,
            "occupied_slots": int(state.occupied.sum().item()),
            "total_writes": int(state.write_counts.sum().item()),
            "write_counts": [
                int(value)
                for value in state.write_counts.detach().cpu().flatten().tolist()
            ],
            "public_evidence_write_counts": [
                int(value)
                for value in public_write_counts.detach().cpu().flatten().tolist()
            ],
        },
        "integrity": {
            "protected_fingerprint_before": protected_before,
            "protected_fingerprint_after": _protected_state_fingerprint(policy),
            "protected_state_unchanged": (
                _protected_state_fingerprint(policy) == protected_before
            ),
            "protected_parameter_objects_unchanged": (
                _protected_parameter_identity(policy) == protected_identity_before
            ),
        },
        "claims": {
            "foundation_reversible_transition_base_writer_query_decoder_or_precedence_adapter_training": False,
            "typed_sensory_and_public_latent_reader_training": True,
            "prior_family_replay_during_demonstration_training_or_acquisition": False,
            "online_feedback_per_support": "one sampled permutation plus one scalar outcome",
            "demonstration_free_queries_receive_feedback_or_writes": False,
            "deterministic_solver": False,
            "hidden_target_used_by_learner": False,
            "agi_or_broad_domain_transfer_proven": False,
            "train_only_interface_frozen_before_adaptive_diagnostic": bool(
                train_checkpoint_record
                and train_checkpoint_record.get("stage") == "train_only"
            ),
        },
    }
    if not result["integrity"]["protected_state_unchanged"] or not result[
        "integrity"
    ]["protected_parameter_objects_unchanged"]:
        raise RuntimeError("demonstration work changed protected prior capability")
    result["result_digest"] = "sha256:" + hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()
    if state_checkpoint is not None:
        checkpoint_path = Path(state_checkpoint)
        if checkpoint_path.exists():
            raise FileExistsError(
                f"acquired-state checkpoint already exists: {checkpoint_path}"
            )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "runner": _REPORT_VERSION,
                "stage": "three_family_acquired_state",
                "seed": seed,
                "base_checkpoint_sha256": initialization["sha256"],
                "precedence_adapter_sha256": precedence_record["sha256"],
                "train_only_adapter_sha256": (
                    None
                    if train_checkpoint_record is None
                    else train_checkpoint_record.get("sha256")
                ),
                "acquired_procedural_state": snapshot_procedural_skill_state(state),
                "acquired_procedural_state_digest": procedural_skill_state_digest(
                    state
                ),
                "acquired_state_context": {
                    "seed": seed,
                    "mechanism_commitment": demonstration.mechanism_commitment,
                    "old_support_presentations": old_acquisition["presentations"],
                    "precedence_support_presentations": relational_acquisition[
                        "presentations"
                    ],
                    "demonstration_support_presentations": (
                        demonstration_acquisition["presentations"]
                    ),
                },
                "protected_fingerprint": _protected_state_fingerprint(policy),
                "result_digest": result["result_digest"],
            },
            checkpoint_path,
        )
        result["acquired_state_checkpoint"] = str(checkpoint_path)
        result["acquired_state_checkpoint_sha256"] = hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=95_001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--precedence-adapter-checkpoint", required=True)
    parser.add_argument("--compiler-checkpoint", default=str(phase5._PHASE4_CHECKPOINT))
    parser.add_argument("--meta-steps", type=int, default=64)
    parser.add_argument("--meta-supports-per-procedure", type=int, default=8)
    parser.add_argument("--meta-queries-per-procedure", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--online-supports-per-procedure", type=int, default=64)
    parser.add_argument("--online-queries", type=int, default=40)
    parser.add_argument("--sensory-initialization-checkpoint")
    parser.add_argument("--trained-adapter-checkpoint")
    parser.add_argument("--checkpoint")
    parser.add_argument("--state-checkpoint")
    parser.add_argument("--result-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        seed=args.seed,
        device=args.device,
        initial_checkpoint=args.initial_checkpoint,
        precedence_adapter_checkpoint=args.precedence_adapter_checkpoint,
        compiler_checkpoint=args.compiler_checkpoint,
        meta_steps=args.meta_steps,
        meta_supports_per_procedure=args.meta_supports_per_procedure,
        meta_queries_per_procedure=args.meta_queries_per_procedure,
        learning_rate=args.learning_rate,
        online_supports_per_procedure=args.online_supports_per_procedure,
        online_queries=args.online_queries,
        sensory_initialization_checkpoint=args.sensory_initialization_checkpoint,
        trained_adapter_checkpoint=args.trained_adapter_checkpoint,
        checkpoint=args.checkpoint,
        state_checkpoint=args.state_checkpoint,
    )
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)
    if args.result_json:
        destination = Path(args.result_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()

"""Qwen-facing runtime boundary for learned situated evidence selection.

Angler selects experience with learned attention.  Qwen consumes the selected
evidence as working context.  Deterministic code here only validates tensors,
assembles transparent prompts, and packages externally judged outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch

from angler.memory import MemoryProjection, RecallBatch, SituatedMemory
from angler.reasoning import (
    LearnedSituatedMemoryReader,
    SituatedFeatureSpec,
    encode_situated_features,
)


@dataclass(frozen=True, slots=True)
class SituatedEvidenceSelection:
    artifact_ref: str
    text: str
    candidate_index: int
    attention: float
    attention_distribution: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SituatedQwenTurn:
    query: str
    selected: SituatedEvidenceSelection
    prompt: str
    response: str


class LocalQwenIO:
    """Use one already-loaded frozen Qwen model for embeddings and generation."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        *,
        embedding_batch_size: int = 32,
        max_new_tokens: int = 256,
        enable_thinking: bool = False,
    ) -> None:
        if embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be positive")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Qwen foundation parameters must be frozen")
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_batch_size = embedding_batch_size
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking

    def embed(self, texts: Sequence[str]) -> torch.Tensor:
        """Return detached foundation representations for reader input."""

        from .qwen_knowledge import encode_detached_segments

        return encode_detached_segments(
            self.model,
            self.tokenizer,
            texts,
            batch_size=self.embedding_batch_size,
            storage_dtype=torch.bfloat16,
        ).float()

    def generate(self, prompt: str) -> str:
        """Generate one response without modifying Qwen or Angler state."""

        if type(prompt) is not str or not prompt.strip():
            raise ValueError("prompt must be non-empty text")
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        encoded = self.tokenizer(rendered, return_tensors="pt")
        device = next(self.model.parameters()).device
        encoded = encoded.to(device)
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[:, encoded["input_ids"].shape[1] :]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


class SituatedQwenCoordinator:
    """Run recall -> learned selection -> Qwen -> externally judged feedback."""

    def __init__(
        self,
        memory: SituatedMemory,
        reader: LearnedSituatedMemoryReader,
        *,
        spec: SituatedFeatureSpec,
        embed: Callable[[Sequence[str]], torch.Tensor],
        generate: Callable[[str], str],
    ) -> None:
        if not callable(embed) or not callable(generate):
            raise TypeError("embed and generate must be callable")
        self.memory = memory
        self.reader = reader
        self.spec = spec
        self._embed = embed
        self._generate = generate

    async def answer(
        self,
        query: str,
        *,
        now: int,
        recall_limit: int = 6,
        world_time: int | None = None,
        response_constraint: str | None = None,
    ) -> SituatedQwenTurn:
        """Answer one query using a concrete learned-selected prior experience."""

        recall = await self.memory.recall(
            query,
            limit=recall_limit,
            world_time=world_time,
        )
        if not recall.items:
            raise LookupError("no validated situated evidence was recalled")
        texts = (query, *(item.text for item in recall.items))
        encoded = self._embed(texts)
        if encoded.ndim != 2 or encoded.shape[0] != len(texts):
            raise ValueError("Qwen encoder output does not align with recalled texts")
        selection = select_situated_evidence(
            self.reader,
            query_features=encoded[0],
            candidate_features=encoded[1:],
            recall=recall,
            now=now,
            spec=self.spec,
        )
        prompt = build_qwen_prompt(
            query,
            selected=selection,
            response_constraint=response_constraint,
        )
        response = self._generate(prompt)
        if type(response) is not str or not response.strip():
            raise RuntimeError("Qwen returned an empty response")
        return SituatedQwenTurn(
            query=query,
            selected=selection,
            prompt=prompt,
            response=response.strip(),
        )

    async def record_feedback(
        self,
        turn: SituatedQwenTurn,
        *,
        artifact_ref: str,
        source_ref: str,
        outcome: Literal["success", "failure"],
        procedure_note: str,
        family: str,
        world_valid_from: int | None = None,
        world_valid_until: int | None = None,
    ) -> int:
        """Persist caller-observed feedback and advance the Moving Origin."""

        projection = package_outcome_feedback(
            turn,
            artifact_ref=artifact_ref,
            source_ref=source_ref,
            outcome=outcome,
            procedure_note=procedure_note,
            family=family,
        )
        return await self.memory.remember(
            projection,
            world_valid_from=world_valid_from,
            world_valid_until=world_valid_until,
        )


def select_situated_evidence(
    reader: LearnedSituatedMemoryReader,
    *,
    query_features: torch.Tensor,
    candidate_features: torch.Tensor,
    recall: RecallBatch,
    now: int,
    spec: SituatedFeatureSpec,
) -> SituatedEvidenceSelection:
    """Select one retrieved experience using the learned reader attribution."""

    if not recall.items:
        raise ValueError("situated evidence selection requires recalled candidates")
    if query_features.ndim != 1:
        raise ValueError("query_features must have shape [content_width]")
    if candidate_features.ndim != 2:
        raise ValueError("candidate_features must have shape [candidates, content_width]")
    if candidate_features.shape[0] != len(recall.items):
        raise ValueError("candidate features do not align with recalled evidence")
    device = next(reader.parameters()).device
    temporal, mask = encode_situated_features(
        (recall,),
        now=(now,),
        spec=spec,
        device=device,
        dtype=next(reader.parameters()).dtype,
    )
    reader.eval()
    with torch.inference_mode():
        output = reader(
            query_features.unsqueeze(0).to(device),
            candidate_features.unsqueeze(0).to(device),
            temporal,
            mask,
        )
    weights = output.weights[0, : len(recall.items)].detach().cpu()
    selected_index = int(weights.argmax().item())
    selected = recall.items[selected_index]
    return SituatedEvidenceSelection(
        artifact_ref=selected.artifact_ref,
        text=selected.text,
        candidate_index=selected_index,
        attention=float(weights[selected_index].item()),
        attention_distribution=tuple(float(value) for value in weights.tolist()),
    )


def build_qwen_prompt(
    query: str,
    *,
    selected: SituatedEvidenceSelection | None = None,
    fair_rag: RecallBatch | None = None,
    response_constraint: str | None = None,
) -> str:
    """Build a visible prompt for baseline, fair-RAG, or Angler-selected use."""

    if type(query) is not str or not query.strip():
        raise ValueError("query must be non-empty text")
    if selected is not None and fair_rag is not None:
        raise ValueError("selected evidence and fair RAG are mutually exclusive")
    sections = [
        "Solve the problem using your own knowledge and the evidence supplied below.",
        "Do not claim the evidence is current unless it is marked Angler-selected.",
    ]
    if selected is not None:
        sections.extend(
            (
                "Angler-selected relevant prior experience:",
                selected.text,
            )
        )
    elif fair_rag is not None:
        sections.append("Semantically retrieved prior experiences (unordered; current regime unknown):")
        sections.extend(f"- {item.text}" for item in fair_rag.items)
    else:
        sections.append("No prior experience was supplied.")
    sections.extend(("Current problem:", query.strip()))
    if response_constraint is not None:
        if type(response_constraint) is not str or not response_constraint.strip():
            raise ValueError("response_constraint must be non-empty text")
        sections.extend(("Response requirement:", response_constraint.strip()))
    return "\n\n".join(sections)


def package_outcome_feedback(
    turn: SituatedQwenTurn,
    *,
    artifact_ref: str,
    source_ref: str,
    outcome: Literal["success", "failure"],
    procedure_note: str,
    family: str,
) -> MemoryProjection:
    """Package caller-supplied outcome feedback for later situated recall.

    The caller owns the outcome and procedure note.  This function neither
    judges the answer nor invents a corrective procedure.
    """

    if outcome not in ("success", "failure"):
        raise ValueError("outcome must be success or failure")
    if type(procedure_note) is not str or not procedure_note.strip():
        raise ValueError("procedure_note must be non-empty text")
    if type(family) is not str or not family.strip():
        raise ValueError("family must be non-empty text")
    text = (
        f"Prior {family.strip()} attempt. Query: {turn.query.strip()} "
        f"Procedure: {procedure_note.strip()} Result {outcome}; "
        f"status {outcome}. Response: {turn.response.strip()}"
    )
    return MemoryProjection.from_mapping(
        artifact_ref=artifact_ref,
        text=text,
        source_ref=source_ref,
        context={
            "family": family.strip(),
            "selected_evidence": turn.selected.artifact_ref,
        },
    )


__all__ = [
    "LocalQwenIO",
    "SituatedQwenCoordinator",
    "SituatedEvidenceSelection",
    "SituatedQwenTurn",
    "build_qwen_prompt",
    "package_outcome_feedback",
    "select_situated_evidence",
]

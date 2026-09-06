"""Qwen-facing runtime boundary for learned situated evidence selection.

Angler selects experience with learned attention.  Qwen consumes the selected
evidence as working context.  Deterministic code here only validates tensors,
assembles transparent prompts, and packages externally judged outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence
import hashlib
import re
from typing import Any, Literal, Mapping, Self

import torch

from angler.episodes.canonical import canonical_bytes
from angler.memory import MemoryProjection, RecallBatch, SituatedMemory
from angler.reasoning import (
    LearnedSituatedMemoryReader,
    SituatedFeatureSpec,
    encode_situated_features,
)


_SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_LOCAL_QWEN_GENERATION_SCHEMA = "angler.local-qwen-generation.v1"


def _optional_sha256_ref(value: object, label: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or _SHA256_REF.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase sha256 reference or None")
    return value


def _positive_integer(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive exact integer")
    return value


def _content_ref(payload: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(payload)).hexdigest()


def local_qwen_generation_config_ref(
    *,
    max_input_tokens: int | None,
    max_new_tokens: int,
    enable_thinking: bool,
) -> str:
    """Identify the fixed greedy generation settings used by ``LocalQwenIO``."""

    if max_input_tokens is not None:
        _positive_integer(max_input_tokens, "max_input_tokens")
    _positive_integer(max_new_tokens, "max_new_tokens")
    if type(enable_thinking) is not bool:
        raise TypeError("enable_thinking must be bool")
    return _content_ref(
        {
            "batch_size": 1,
            "do_sample": False,
            "enable_thinking": enable_thinking,
            "max_input_tokens": max_input_tokens,
            "max_new_tokens": max_new_tokens,
            "schema": _LOCAL_QWEN_GENERATION_SCHEMA,
            "use_cache": True,
        }
    )


@dataclass(frozen=True, slots=True)
class FrozenQwenGenerationRecord:
    """Auditable output from one bounded greedy call to a frozen local Qwen."""

    model_ref: str | None
    tokenizer_ref: str | None
    generation_config_ref: str
    prompt_ref: str
    prompt_tokens: int
    generated_token_ids: tuple[int, ...]
    response: str

    def __post_init__(self) -> None:
        _optional_sha256_ref(self.model_ref, "model_ref")
        _optional_sha256_ref(self.tokenizer_ref, "tokenizer_ref")
        if (self.model_ref is None) != (self.tokenizer_ref is None):
            raise ValueError("model_ref and tokenizer_ref must be present together")
        for label, value in (
            ("generation_config_ref", self.generation_config_ref),
            ("prompt_ref", self.prompt_ref),
        ):
            if type(value) is not str or _SHA256_REF.fullmatch(value) is None:
                raise ValueError(f"{label} must be a lowercase sha256 reference")
        _positive_integer(self.prompt_tokens, "prompt_tokens")
        if type(self.generated_token_ids) is not tuple:
            raise TypeError("generated_token_ids must be an immutable tuple")
        if not self.generated_token_ids:
            raise ValueError("generated_token_ids must not be empty")
        if any(
            type(value) is not int or value < 0
            for value in self.generated_token_ids
        ):
            raise ValueError("generated_token_ids must be non-negative exact integers")
        if type(self.response) is not str:
            raise TypeError("response must be text")

    @property
    def generation_ref(self) -> str:
        return _content_ref(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "generated_token_ids": list(self.generated_token_ids),
            "generation_config_ref": self.generation_config_ref,
            "model_ref": self.model_ref,
            "prompt_ref": self.prompt_ref,
            "prompt_tokens": self.prompt_tokens,
            "response": self.response,
            "schema": _LOCAL_QWEN_GENERATION_SCHEMA,
            "tokenizer_ref": self.tokenizer_ref,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> Self:
        expected = {
            "generated_token_ids",
            "generation_config_ref",
            "model_ref",
            "prompt_ref",
            "prompt_tokens",
            "response",
            "schema",
            "tokenizer_ref",
        }
        if type(payload) is not dict or set(payload) != expected:
            raise ValueError("generation payload fields are not exact")
        if payload["schema"] != _LOCAL_QWEN_GENERATION_SCHEMA:
            raise ValueError("generation payload schema is not supported")
        raw_tokens = payload["generated_token_ids"]
        if type(raw_tokens) is not list:
            raise TypeError("generated_token_ids payload must be a list")
        return cls(
            model_ref=payload["model_ref"],  # type: ignore[arg-type]
            tokenizer_ref=payload["tokenizer_ref"],  # type: ignore[arg-type]
            generation_config_ref=payload["generation_config_ref"],  # type: ignore[arg-type]
            prompt_ref=payload["prompt_ref"],  # type: ignore[arg-type]
            prompt_tokens=payload["prompt_tokens"],  # type: ignore[arg-type]
            generated_token_ids=tuple(raw_tokens),  # type: ignore[arg-type]
            response=payload["response"],  # type: ignore[arg-type]
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
        max_input_tokens: int | None = None,
        model_ref: str | None = None,
        tokenizer_ref: str | None = None,
    ) -> None:
        _positive_integer(embedding_batch_size, "embedding_batch_size")
        _positive_integer(max_new_tokens, "max_new_tokens")
        if max_input_tokens is not None:
            _positive_integer(max_input_tokens, "max_input_tokens")
        if type(enable_thinking) is not bool:
            raise TypeError("enable_thinking must be bool")
        resolved_model_ref = _optional_sha256_ref(model_ref, "model_ref")
        resolved_tokenizer_ref = _optional_sha256_ref(tokenizer_ref, "tokenizer_ref")
        if (resolved_model_ref is None) != (resolved_tokenizer_ref is None):
            raise ValueError("model_ref and tokenizer_ref must be present together")
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Qwen foundation parameters must be frozen")
        # Legacy callers did not explicitly place a frozen module in eval mode.
        # Normalizing that state is backward compatible while ensuring greedy
        # calls cannot retain dropout merely because no optimizer exists.
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_batch_size = embedding_batch_size
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.max_input_tokens = max_input_tokens
        self.model_ref = resolved_model_ref
        self.tokenizer_ref = resolved_tokenizer_ref
        self._generation_calls = 0

    @property
    def generation_config_ref(self) -> str:
        return local_qwen_generation_config_ref(
            max_input_tokens=self.max_input_tokens,
            max_new_tokens=self.max_new_tokens,
            enable_thinking=self.enable_thinking,
        )

    @property
    def has_exact_identity(self) -> bool:
        return self.model_ref is not None and self.tokenizer_ref is not None

    @property
    def generation_calls(self) -> int:
        """Return the number of actual foundation ``generate`` invocations."""

        return self._generation_calls

    def _assert_frozen_eval(self) -> None:
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("Qwen foundation parameters are no longer frozen")
        if self.model.training:
            raise RuntimeError("Qwen foundation model is not in evaluation mode")

    def assert_exact_identity(
        self,
        *,
        model_ref: str,
        tokenizer_ref: str,
        generation_config_ref: str,
    ) -> None:
        """Fail closed unless this IO is the exact frozen configured runtime."""

        self._assert_frozen_eval()
        if not self.has_exact_identity:
            raise ValueError("exact Qwen use requires model and tokenizer identities")
        if (
            self.model_ref != model_ref
            or self.tokenizer_ref != tokenizer_ref
            or self.generation_config_ref != generation_config_ref
        ):
            raise ValueError("local Qwen identity or generation configuration drifted")

    def _token_lengths(self, texts: Sequence[str]) -> tuple[int, ...]:
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )
        try:
            mask = encoded["attention_mask"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Qwen tokenizer did not return an attention mask") from exc
        if not isinstance(mask, torch.Tensor) or mask.ndim != 2:
            raise RuntimeError("Qwen tokenizer attention mask must be rank two")
        return tuple(int(value) for value in mask.sum(dim=1).tolist())

    def _assert_text_token_bounds(self, texts: Sequence[str]) -> None:
        if self.max_input_tokens is None:
            return
        lengths = self._token_lengths(texts)
        if any(value < 1 or value > self.max_input_tokens for value in lengths):
            raise ValueError("Qwen input exceeds the fixed token ceiling")

    def embed(self, texts: Sequence[str]) -> torch.Tensor:
        """Return detached foundation representations for reader input."""

        from .qwen_knowledge import encode_detached_segments

        self._assert_frozen_eval()
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence of complete segments")
        segments = tuple(texts)
        if not segments or any(
            type(value) is not str or not value.strip() for value in segments
        ):
            raise ValueError("texts must contain non-empty text segments")
        self._assert_text_token_bounds(segments)
        result = encode_detached_segments(
            self.model,
            self.tokenizer,
            segments,
            batch_size=self.embedding_batch_size,
            storage_dtype=torch.bfloat16,
        ).float()
        if (
            result.ndim != 2
            or result.shape[0] != len(segments)
            or not bool(torch.isfinite(result).all().item())
            or result.requires_grad
            or result.grad_fn is not None
        ):
            raise RuntimeError("Qwen knowledge representations are not detached finite rows")
        return result

    def generate_record(self, prompt: str) -> FrozenQwenGenerationRecord:
        """Generate and retain the exact bounded public token identity."""

        if type(prompt) is not str or not prompt.strip():
            raise ValueError("prompt must be non-empty text")
        self._assert_frozen_eval()
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        encoded = self.tokenizer(rendered, return_tensors="pt")
        try:
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Qwen tokenizer omitted generation tensors") from exc
        if (
            not isinstance(input_ids, torch.Tensor)
            or not isinstance(attention_mask, torch.Tensor)
            or input_ids.ndim != 2
            or attention_mask.shape != input_ids.shape
            or input_ids.shape[0] != 1
        ):
            raise RuntimeError("Qwen generation tokenizer output must be one aligned row")
        prompt_tokens = int(attention_mask.sum().item())
        if prompt_tokens < 1:
            raise RuntimeError("Qwen tokenizer produced an empty prompt")
        if self.max_input_tokens is not None and prompt_tokens > self.max_input_tokens:
            raise ValueError("Qwen input exceeds the fixed token ceiling")
        device = next(self.model.parameters()).device
        encoded = encoded.to(device)
        with torch.inference_mode():
            self._generation_calls += 1
            output = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        if (
            not isinstance(output, torch.Tensor)
            or output.ndim != 2
            or output.shape[0] != 1
        ):
            raise RuntimeError("Qwen generation output must be one token row")
        generated = output[:, encoded["input_ids"].shape[1] :]
        if generated.shape[1] < 1 or generated.shape[1] > self.max_new_tokens:
            raise RuntimeError("Qwen generated token count violates its fixed ceiling")
        token_ids = tuple(int(value) for value in generated[0].detach().cpu().tolist())
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        if (
            not isinstance(decoded, (tuple, list))
            or len(decoded) != 1
            or type(decoded[0]) is not str
        ):
            raise RuntimeError("Qwen tokenizer did not decode one response")
        return FrozenQwenGenerationRecord(
            model_ref=self.model_ref,
            tokenizer_ref=self.tokenizer_ref,
            generation_config_ref=self.generation_config_ref,
            prompt_ref=_content_ref({"prompt": prompt}),
            prompt_tokens=prompt_tokens,
            generated_token_ids=token_ids,
            response=decoded[0],
        )

    def generate(self, prompt: str) -> str:
        """Generate one response without modifying Qwen or Angler state."""

        return self.generate_record(prompt).response


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
    "FrozenQwenGenerationRecord",
    "LocalQwenIO",
    "SituatedQwenCoordinator",
    "SituatedEvidenceSelection",
    "SituatedQwenTurn",
    "build_qwen_prompt",
    "local_qwen_generation_config_ref",
    "package_outcome_feedback",
    "select_situated_evidence",
]

"""Persistent local conversation over Cognee, Moving Origin, and frozen Qwen.

The journal is canonical for this prototype; Cognee is a disposable retrieval
projection.  Neural weights are never updated here and outcome feedback always
comes from the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Literal

import torch

from angler.memory import (
    CogneeProjectionBackend,
    MemoryProjection,
    MovingOriginIndex,
    RecallBatch,
    SituatedMemory,
    encode_projection,
    projection_id,
)
from angler.reasoning import LearnedSituatedMemoryReader, SituatedFeatureSpec

from .situated_qwen import (
    SituatedEvidenceSelection,
    build_qwen_prompt,
    select_situated_evidence,
)


_JOURNAL_VERSION = "angler.situated-conversation-journal.v1"


class ChunkOnlyCogneeProjectionBackend(CogneeProjectionBackend):
    """Index validated records with Cognee's non-generative chunk pipeline.

    Conversation feedback already carries Angler provenance. Cognee still
    supplies scoped datasets, chunking, local embeddings, vector retrieval,
    and deletion, but a second language model does not reinterpret the record
    before it becomes searchable.
    """

    async def _remember_documents(self, documents: tuple[str, ...]) -> None:
        if not documents or any(type(item) is not str or not item for item in documents):
            raise ValueError("documents must be a non-empty tuple of text")
        cognee = self._module()
        self._require_telemetry_authority()
        self._require_model_authority()
        from cognee.eval_framework.corpus_builder.task_getters.get_default_tasks_by_indices import (
            get_just_chunks_tasks,
        )
        from cognee.modules.run_custom_pipeline import run_custom_pipeline

        # The public chunk tasks consume Cognee DataItems, so preserve
        # Cognee's supported two-stage contract: add raw text to the scoped
        # dataset, then project those records through the non-generative
        # chunk/vector pipeline.
        await cognee.add(
            list(documents),
            dataset_name=self.dataset_name,
            incremental_loading=False,
            data_cache=False,
        )
        result = await run_custom_pipeline(
            tasks=await get_just_chunks_tasks(),
            dataset=self.dataset_name,
            pipeline_name="angler_chunk_projection",
            use_pipeline_cache=False,
            incremental_loading=False,
            data_cache=False,
            # The selected tasks perform local embeddings but no LLM calls.
            # Their actual write is the connectivity test; the generic setup
            # probe would unnecessarily wake the configured Ollama LLM.
            skip_connection_test=True,
        )
        if not isinstance(result, dict) or not result:
            raise RuntimeError("Cognee chunk projection returned no pipeline result")
        failures = [
            str(getattr(run_info, "status", "UNKNOWN"))
            for run_info in result.values()
            if "Errored" in str(getattr(run_info, "status", ""))
        ]
        if failures:
            raise RuntimeError(f"Cognee chunk projection failed: {failures}")

    async def remember(self, document: str) -> str | None:
        await self._remember_documents((document,))
        return None

    async def remember_many(self, documents: tuple[str, ...]) -> tuple[str, ...]:
        await self._remember_documents(documents)
        return ()


class HybridChunkCogneeProjectionBackend(ChunkOnlyCogneeProjectionBackend):
    """Union Cognee semantic and lexical candidates before learned selection."""

    _PER_RETRIEVER_LIMIT = 6

    async def search(self, query: str, *, limit: int):
        branch_limit = min(self._PER_RETRIEVER_LIMIT, limit)
        vector_hits = await super().search(query, limit=branch_limit)
        cognee = self._module()
        from angler.memory.cognee_adapter import _chunk_hit, _cognee_result_entries

        raw_lexical = await cognee.search(
            query_text=query,
            query_type=cognee.SearchType.CHUNKS_LEXICAL,
            datasets=[self.dataset_name],
            top_k=branch_limit,
        )
        lexical_hits = tuple(
            hit
            for item in _cognee_result_entries(raw_lexical)
            if (hit := _chunk_hit(item)) is not None
        )
        combined = []
        seen_documents: set[str] = set()
        for hit in (*vector_hits, *lexical_hits):
            if hit.document in seen_documents:
                continue
            seen_documents.add(hit.document)
            combined.append(hit)
            if len(combined) == limit:
                break
        return tuple(combined)


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    role: Literal["user", "assistant"]
    text: str

    def __post_init__(self) -> None:
        if self.role not in ("user", "assistant"):
            raise ValueError("conversation role must be user or assistant")
        if type(self.text) is not str or not self.text.strip():
            raise ValueError("conversation text must be non-empty")


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    query: str
    selected: SituatedEvidenceSelection | None
    recalled: RecallBatch
    prompt: str
    response: str


@dataclass(frozen=True, slots=True)
class JournalRecord:
    projection: MemoryProjection
    world_valid_from: int | None = None
    world_valid_until: int | None = None
    current_regime: bool = False

    def __post_init__(self) -> None:
        if type(self.current_regime) is not bool:
            raise TypeError("current_regime must be boolean")
        if (
            self.world_valid_from is not None
            and self.world_valid_until is not None
            and self.world_valid_until < self.world_valid_from
        ):
            raise ValueError("world validity interval is reversed")


class ConversationJournal:
    """Append-only canonical projection history for one named workspace."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def records(self) -> tuple[JournalRecord, ...]:
        if not self.path.exists():
            return ()
        records = []
        seen: set[str] = set()
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"journal line {line_number} is invalid JSON") from exc
            records.append(_record_from_payload(payload, line_number=line_number))
            artifact_ref = records[-1].projection.artifact_ref
            if artifact_ref in seen:
                raise ValueError(f"journal line {line_number} repeats an artifact_ref")
            seen.add(artifact_ref)
        return tuple(records)

    def append(self, record: JournalRecord) -> None:
        existing = self.records()
        if any(item.projection.artifact_ref == record.projection.artifact_ref for item in existing):
            raise ValueError("artifact_ref already exists in the conversation journal")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            _record_payload(record), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded + "\n")

    def digest(self) -> str | None:
        if not self.path.exists():
            return None
        return "sha256:" + hashlib.sha256(self.path.read_bytes()).hexdigest()

    def build_origin(self) -> MovingOriginIndex:
        records = self.records()
        origin = MovingOriginIndex()
        for record in records:
            origin.append(
                record.projection.artifact_ref,
                projection_id(record.projection),
                world_valid_from=record.world_valid_from,
                world_valid_until=record.world_valid_until,
            )
        current = next((item for item in reversed(records) if item.current_regime), None)
        if current is not None:
            target = current.projection.artifact_ref
            material = f"{self.path.resolve()}:{target}:current_regime"
            designation_ref = "sha256:" + hashlib.sha256(material.encode()).hexdigest()
            designation_projection = "sha256:" + hashlib.sha256(
                (material + ":projection").encode()
            ).hexdigest()
            origin.designate_landmark(
                "current_regime",
                target_event_ref=target,
                designation_event_ref=designation_ref,
                projection_id=designation_projection,
            )
        return origin

    async def rebuild_projection(
        self,
        memory: SituatedMemory,
        *,
        forget_existing: bool = True,
    ) -> None:
        records = self.records()
        if forget_existing:
            await memory.forget_projection()
        documents = tuple(encode_projection(item.projection) for item in records)
        if documents:
            bulk = getattr(memory.backend, "remember_many", None)
            if callable(bulk):
                await bulk(documents)
            else:
                for document in documents:
                    await memory.backend.remember(document)
        memory.origin = self.build_origin()


class SituatedConversationSession:
    """One bounded multi-turn session with transparent evidence selection."""

    def __init__(
        self,
        memory: SituatedMemory,
        journal: ConversationJournal,
        reader: LearnedSituatedMemoryReader,
        *,
        spec: SituatedFeatureSpec,
        embed: Callable[[Sequence[str]], torch.Tensor],
        generate: Callable[[str], str],
        max_history_messages: int = 8,
    ) -> None:
        if max_history_messages < 0:
            raise ValueError("max_history_messages cannot be negative")
        self.memory = memory
        self.journal = journal
        self.reader = reader
        self.spec = spec
        self._embed = embed
        self._generate = generate
        self.max_history_messages = max_history_messages
        self.history: list[ConversationMessage] = []
        self.last_turn: ConversationTurn | None = None

    async def answer(
        self,
        query: str,
        *,
        recall_limit: int = 6,
        world_time: int | None = None,
        response_constraint: str | None = None,
    ) -> ConversationTurn:
        if type(query) is not str or not query.strip():
            raise ValueError("query must be non-empty text")
        # A brand-new named workspace has no Cognee database yet. Its empty
        # canonical journal is definitive, so avoid querying an uninitialized
        # disposable projection and take the visible Qwen-alone fallback. The
        # first explicit feedback write initializes Cognee for later turns.
        recall = (
            RecallBatch(())
            if not self.journal.records()
            else await self.memory.recall(
                query, limit=recall_limit, world_time=world_time
            )
        )
        selection = None
        if recall.items:
            texts = (query, *(item.text for item in recall.items))
            encoded = self._embed(texts)
            if encoded.ndim != 2 or encoded.shape[0] != len(texts):
                raise ValueError("encoder output does not align with recalled texts")
            selection = select_situated_evidence(
                self.reader,
                query_features=encoded[0],
                candidate_features=encoded[1:],
                recall=recall,
                now=self.memory.origin.now,
                spec=self.spec,
            )
        prompt = build_conversation_prompt(
            query,
            history=self.history[-self.max_history_messages :]
            if self.max_history_messages
            else (),
            selected=selection,
            response_constraint=response_constraint,
        )
        response = self._generate(prompt)
        if type(response) is not str or not response.strip():
            raise RuntimeError("Qwen returned an empty response")
        turn = ConversationTurn(
            query=query.strip(),
            selected=selection,
            recalled=recall,
            prompt=prompt,
            response=response.strip(),
        )
        self.history.extend(
            (ConversationMessage("user", turn.query), ConversationMessage("assistant", turn.response))
        )
        self.last_turn = turn
        return turn

    async def record_feedback(
        self,
        *,
        outcome: Literal["success", "failure"],
        procedure_note: str,
        family: str,
        source_ref: str,
        artifact_ref: str | None = None,
        world_valid_from: int | None = None,
        world_valid_until: int | None = None,
    ) -> JournalRecord:
        if self.last_turn is None:
            raise RuntimeError("feedback requires a completed conversation turn")
        if outcome not in ("success", "failure"):
            raise ValueError("outcome must be success or failure")
        if type(procedure_note) is not str or not procedure_note.strip():
            raise ValueError("procedure_note must be non-empty text")
        if type(family) is not str or not family.strip():
            raise ValueError("family must be non-empty text")
        selected_ref = (
            self.last_turn.selected.artifact_ref
            if self.last_turn.selected is not None
            else "NONE"
        )
        text = (
            f"Prior {family.strip()} attempt. Query: {self.last_turn.query} "
            f"Procedure: {procedure_note.strip()} Result {outcome}; status {outcome}. "
            f"Response: {self.last_turn.response}"
        )
        if artifact_ref is None:
            material = json.dumps(
                {
                    "query": self.last_turn.query,
                    "response": self.last_turn.response,
                    "outcome": outcome,
                    "procedure_note": procedure_note.strip(),
                    "family": family.strip(),
                    "journal": self.journal.digest(),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            artifact_ref = "sha256:" + hashlib.sha256(material.encode()).hexdigest()
        projection = MemoryProjection.from_mapping(
            artifact_ref=artifact_ref,
            text=text,
            source_ref=source_ref,
            context={"family": family.strip(), "selected_evidence": selected_ref},
        )
        record = JournalRecord(
            projection,
            world_valid_from=world_valid_from,
            world_valid_until=world_valid_until,
            current_regime=outcome == "success",
        )
        # Canonical journal first: a failed disposable projection can be rebuilt.
        self.journal.append(record)
        await self.memory.backend.remember(encode_projection(projection))
        self.memory.origin = self.journal.build_origin()
        return record

    async def rebuild_memory(self) -> None:
        await self.journal.rebuild_projection(self.memory)


def build_conversation_prompt(
    query: str,
    *,
    history: Sequence[ConversationMessage] = (),
    selected: SituatedEvidenceSelection | None = None,
    fair_rag: RecallBatch | None = None,
    response_constraint: str | None = None,
) -> str:
    """Extend the existing visible Qwen prompt with bounded prior turns."""

    base = build_qwen_prompt(
        query,
        selected=selected,
        fair_rag=fair_rag,
        response_constraint=response_constraint,
    )
    if not history:
        return base
    rendered = "\n".join(f"{item.role.upper()}: {item.text.strip()}" for item in history)
    marker = "\n\nCurrent problem:"
    if marker not in base:
        raise RuntimeError("base Qwen prompt has an unexpected structure")
    return base.replace(marker, f"\n\nRecent conversation (context only):\n{rendered}{marker}", 1)


def _record_payload(record: JournalRecord) -> dict[str, object]:
    projection = record.projection
    return {
        "version": _JOURNAL_VERSION,
        "projection": {
            "artifact_ref": projection.artifact_ref,
            "text": projection.text,
            "source_ref": projection.source_ref,
            "visibility": projection.visibility,
            "context": [list(pair) for pair in projection.context],
        },
        "projection_id": projection_id(projection),
        "world_valid_from": record.world_valid_from,
        "world_valid_until": record.world_valid_until,
        "current_regime": record.current_regime,
    }


def _record_from_payload(payload: object, *, line_number: int) -> JournalRecord:
    if not isinstance(payload, dict) or set(payload) != {
        "version",
        "projection",
        "projection_id",
        "world_valid_from",
        "world_valid_until",
        "current_regime",
    }:
        raise ValueError(f"journal line {line_number} has invalid fields")
    if payload["version"] != _JOURNAL_VERSION or not isinstance(payload["projection"], dict):
        raise ValueError(f"journal line {line_number} has an invalid version or projection")
    raw = payload["projection"]
    if set(raw) != {"artifact_ref", "text", "source_ref", "visibility", "context"}:
        raise ValueError(f"journal line {line_number} has invalid projection fields")
    context = raw["context"]
    if not isinstance(context, list) or any(
        not isinstance(pair, list) or len(pair) != 2 for pair in context
    ):
        raise ValueError(f"journal line {line_number} has invalid context")
    projection = MemoryProjection(
        artifact_ref=raw["artifact_ref"],
        text=raw["text"],
        source_ref=raw["source_ref"],
        visibility=raw["visibility"],
        context=tuple((pair[0], pair[1]) for pair in context),
    )
    if payload["projection_id"] != projection_id(projection):
        raise ValueError(f"journal line {line_number} projection identity mismatches")
    return JournalRecord(
        projection,
        world_valid_from=payload["world_valid_from"],
        world_valid_until=payload["world_valid_until"],
        current_regime=payload["current_regime"],
    )


__all__ = [
    "ChunkOnlyCogneeProjectionBackend",
    "ConversationJournal",
    "ConversationMessage",
    "ConversationTurn",
    "JournalRecord",
    "HybridChunkCogneeProjectionBackend",
    "SituatedConversationSession",
    "build_conversation_prompt",
]

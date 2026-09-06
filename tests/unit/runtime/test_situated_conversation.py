from __future__ import annotations

import asyncio
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from torch import nn

from angler.memory import MemoryHit, MemoryProjection, SituatedMemory, decode_projection
from angler.reasoning import SituatedFeatureSpec
from angler.runtime import (
    ConversationJournal,
    ConversationMessage,
    HybridChunkCogneeProjectionBackend,
    JournalRecord,
    SituatedConversationSession,
    build_conversation_prompt,
)


class _Backend:
    def __init__(self) -> None:
        self.documents: list[str] = []
        self.forgotten = 0
        self.searches = 0

    async def remember(self, document: str) -> str:
        self.documents.append(document)
        return str(len(self.documents))

    async def remember_many(self, documents: tuple[str, ...]) -> tuple[str, ...]:
        self.documents.extend(documents)
        return tuple(str(index) for index in range(len(documents)))

    async def search(self, query: str, *, limit: int) -> tuple[MemoryHit, ...]:
        self.searches += 1
        return tuple(MemoryHit(item, score=0.8) for item in self.documents[-limit:])

    async def forget_dataset(self) -> None:
        self.forgotten += 1
        self.documents.clear()


class _FirstReader(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, query, candidates, temporal, mask):
        weights = torch.zeros(candidates.shape[:2], device=candidates.device)
        weights[:, 0] = 1.0
        return type("Output", (), {"weights": weights})()


class _CogneeSearchDouble:
    class SearchType:
        CHUNKS = "chunks"
        CHUNKS_LEXICAL = "chunks_lexical"

    def __init__(self) -> None:
        self.calls = []

    async def search(self, *, query_text, query_type, datasets, top_k):
        self.calls.append((query_text, query_type, tuple(datasets), top_k))
        if query_type == self.SearchType.CHUNKS:
            return [
                {"text": "shared vector record", "id": "v0"},
                {"text": "vector-only record", "id": "v1"},
            ]
        return [
            {"text": "shared vector record", "id": "l0"},
            {"text": "lexical-only record", "id": "l1"},
            {"text": "later lexical record", "id": "l2"},
        ]


def _projection(index: int, *, text: str | None = None) -> MemoryProjection:
    return MemoryProjection.from_mapping(
        artifact_ref=f"sha256:event-{index}",
        text=text or f"Prior relevant procedure {index} succeeded.",
        source_ref=f"sha256:source-{index}",
        context={"family": "test"},
    )


class SituatedConversationTests(unittest.TestCase):
    def test_hybrid_cognee_recall_unions_deduplicates_and_caps_candidates(self) -> None:
        async def exercise():
            cognee = _CogneeSearchDouble()
            backend = HybridChunkCogneeProjectionBackend(
                "hybrid-test",
                cognee_module=cognee,
                local_models_configured=True,
            )
            prior = os.environ.get("TELEMETRY_DISABLED")
            os.environ["TELEMETRY_DISABLED"] = "1"
            try:
                hits = await backend.search("combined query", limit=3)
            finally:
                if prior is None:
                    os.environ.pop("TELEMETRY_DISABLED", None)
                else:
                    os.environ["TELEMETRY_DISABLED"] = prior
            self.assertEqual(
                tuple(hit.document for hit in hits),
                ("shared vector record", "vector-only record", "lexical-only record"),
            )
            self.assertEqual(
                tuple(call[1] for call in cognee.calls),
                (cognee.SearchType.CHUNKS, cognee.SearchType.CHUNKS_LEXICAL),
            )
            self.assertTrue(all(call[3] == 3 for call in cognee.calls))

        asyncio.run(exercise())

    def test_journal_round_trip_rebuilds_moving_origin_and_rejects_tampering(self) -> None:
        with TemporaryDirectory() as root:
            path = Path(root) / "journal.jsonl"
            journal = ConversationJournal(path)
            journal.append(JournalRecord(_projection(0)))
            journal.append(JournalRecord(_projection(1), current_regime=True))
            records = journal.records()
            origin = journal.build_origin()

            self.assertEqual(len(records), 2)
            self.assertEqual(origin.size, 3)
            self.assertEqual(dict(origin.position("sha256:event-1").landmark_relations)["current_regime"], "BEFORE")
            path.write_text(path.read_text().replace("procedure 1", "procedure X"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity mismatches"):
                journal.records()

    def test_history_is_bounded_and_visible_without_entering_retrieval(self) -> None:
        prompt = build_conversation_prompt(
            "new question",
            history=(
                ConversationMessage("user", "first question"),
                ConversationMessage("assistant", "first answer"),
            ),
        )
        self.assertIn("Recent conversation", prompt)
        self.assertIn("USER: first question", prompt)
        self.assertTrue(prompt.endswith("new question"))

    def test_feedback_is_human_supplied_persistent_and_recalled_next_turn(self) -> None:
        async def exercise(path: Path) -> None:
            backend = _Backend()
            journal = ConversationJournal(path)
            journal.append(JournalRecord(_projection(0), current_regime=True))
            memory = SituatedMemory(backend, origin=journal.build_origin())
            await journal.rebuild_projection(memory)
            seen_prompts: list[str] = []

            def embed(texts):
                return torch.arange(len(texts) * 4, dtype=torch.float32).reshape(len(texts), 4)

            def generate(prompt: str) -> str:
                seen_prompts.append(prompt)
                return "a generated answer"

            session = SituatedConversationSession(
                memory,
                journal,
                _FirstReader(),
                spec=SituatedFeatureSpec(("current_regime",), include_acquired_ordinal=False),
                embed=embed,
                generate=generate,
                max_history_messages=2,
            )
            first = await session.answer("how should this test proceed?")
            self.assertIsNotNone(first.selected)
            record = await session.record_feedback(
                outcome="success",
                procedure_note="use the verified narrow interface",
                family="runtime testing",
                source_ref="sha256:human-judge",
            )
            self.assertIn("Result success", record.projection.text)
            self.assertEqual(len(journal.records()), 2)
            self.assertEqual(dict(memory.origin.position(record.projection.artifact_ref).landmark_relations)["current_regime"], "BEFORE")

            second = await session.answer("verified narrow interface runtime testing")
            self.assertTrue(any(item.artifact_ref == record.projection.artifact_ref for item in second.recalled.items))
            self.assertIn("Recent conversation", second.prompt)
            self.assertEqual(len(seen_prompts), 2)

            await session.rebuild_memory()
            self.assertEqual(backend.forgotten, 2)
            self.assertEqual(len(backend.documents), 2)
            decoded, _ = decode_projection(backend.documents[-1])
            self.assertEqual(decoded.artifact_ref, record.projection.artifact_ref)

        with TemporaryDirectory() as root:
            asyncio.run(exercise(Path(root) / "journal.jsonl"))

    def test_empty_recall_falls_back_to_qwen_alone(self) -> None:
        async def exercise(path: Path) -> None:
            backend = _Backend()
            session = SituatedConversationSession(
                SituatedMemory(backend),
                ConversationJournal(path),
                _FirstReader(),
                spec=SituatedFeatureSpec(("current_regime",), include_acquired_ordinal=False),
                embed=lambda texts: torch.zeros((len(texts), 4)),
                generate=lambda prompt: "plain response" if "No prior experience" in prompt else "wrong",
            )
            turn = await session.answer("hello")
            self.assertIsNone(turn.selected)
            self.assertEqual(turn.response, "plain response")
            self.assertEqual(backend.searches, 0)

        with TemporaryDirectory() as root:
            asyncio.run(exercise(Path(root) / "journal.jsonl"))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from angler.memory import MemoryHit, MemoryProjection, RecallBatch, SituatedMemory, SituatedRecall
from angler.reasoning import LearnedSituatedMemoryReader, SituatedFeatureSpec
from angler.runtime.situated_qwen import (
    SituatedEvidenceSelection,
    SituatedQwenCoordinator,
    SituatedQwenTurn,
    build_qwen_prompt,
    package_outcome_feedback,
    select_situated_evidence,
)


class _Backend:
    def __init__(self) -> None:
        self.documents: list[str] = []

    async def remember(self, document: str) -> str:
        self.documents.append(document)
        return f"memory-{len(self.documents)}"

    async def search(self, query: str, *, limit: int) -> tuple[MemoryHit, ...]:
        return tuple(MemoryHit(document) for document in self.documents[-limit:])

    async def forget_dataset(self) -> None:
        self.documents.clear()


class _FirstReader(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, query, candidates, temporal, mask):
        weights = torch.zeros(candidates.shape[:2], device=candidates.device)
        weights[:, 0] = 1.0
        return SimpleNamespace(weights=weights)


def _item(index: int) -> SituatedRecall:
    return SituatedRecall(
        artifact_ref=f"sha256:event-{index}",
        text=f"procedure evidence {index}",
        source_ref=f"sha256:source-{index}",
        context=(("family", "test"),),
        visibility="LEARNER_VISIBLE",
        acquired_ordinal=index,
        age=1 - index,
        landmark_relations=(("current", "AT" if index else "BEFORE"),),
        world_valid_from=None,
        world_valid_until=None,
        world_valid_at_query=None,
        backend_score=None,
        backend_ref=f"chunk-{index}",
    )


class SituatedQwenRuntimeTests(unittest.TestCase):
    def test_reader_attribution_selects_recalled_evidence(self) -> None:
        torch.manual_seed(7)
        reader = LearnedSituatedMemoryReader(
            content_width=4,
            temporal_width=11,
            hidden_width=8,
            action_count=2,
        )
        recall = RecallBatch((_item(0), _item(1)))

        selected = select_situated_evidence(
            reader,
            query_features=torch.randn(4),
            candidate_features=torch.randn(2, 4),
            recall=recall,
            now=2,
            spec=SituatedFeatureSpec(("current",), include_acquired_ordinal=False),
        )

        self.assertIn(selected.candidate_index, (0, 1))
        self.assertEqual(selected.artifact_ref, recall.items[selected.candidate_index].artifact_ref)
        self.assertAlmostEqual(sum(selected.attention_distribution), 1.0, places=6)

    def test_prompt_modes_are_explicit_and_mutually_exclusive(self) -> None:
        selected = SituatedEvidenceSelection(
            artifact_ref="sha256:event-1",
            text="selected procedure",
            candidate_index=0,
            attention=0.9,
            attention_distribution=(0.9, 0.1),
        )
        baseline = build_qwen_prompt("problem")
        augmented = build_qwen_prompt("problem", selected=selected)
        rag = build_qwen_prompt("problem", fair_rag=RecallBatch((_item(0), _item(1))))

        self.assertIn("No prior experience", baseline)
        self.assertIn("Angler-selected", augmented)
        self.assertIn("unordered", rag)
        with self.assertRaises(ValueError):
            build_qwen_prompt(
                "problem",
                selected=selected,
                fair_rag=RecallBatch((_item(0),)),
            )

    def test_external_feedback_is_packaged_without_judging_it(self) -> None:
        selected = SituatedEvidenceSelection(
            artifact_ref="sha256:event-1",
            text="selected procedure",
            candidate_index=0,
            attention=1.0,
            attention_distribution=(1.0,),
        )
        turn = SituatedQwenTurn("query", selected, "prompt", "response")

        projection = package_outcome_feedback(
            turn,
            artifact_ref="sha256:new-event",
            source_ref="sha256:judge",
            outcome="success",
            procedure_note="compare the current successful precedent",
            family="interface adaptation",
        )

        self.assertIn("Result success", projection.text)
        self.assertIn(("selected_evidence", "sha256:event-1"), projection.context)

    def test_coordinator_closes_recall_generation_feedback_loop(self) -> None:
        async def exercise() -> None:
            backend = _Backend()
            memory = SituatedMemory(backend)
            seed = MemoryProjection.from_mapping(
                artifact_ref="sha256:seed",
                text="Prior repair used procedure cobalt and succeeded.",
                source_ref="sha256:observation",
                context={"family": "repair"},
            )
            await memory.remember(seed)
            prompts: list[str] = []

            def embed(texts):
                return torch.arange(len(texts) * 4, dtype=torch.float32).reshape(len(texts), 4)

            def generate(prompt: str) -> str:
                prompts.append(prompt)
                return "cobalt"

            coordinator = SituatedQwenCoordinator(
                memory,
                _FirstReader(),
                spec=SituatedFeatureSpec((), include_acquired_ordinal=False),
                embed=embed,
                generate=generate,
            )
            turn = await coordinator.answer("repair this interface", now=1)
            self.assertEqual(turn.response, "cobalt")
            self.assertEqual(turn.selected.artifact_ref, "sha256:seed")
            self.assertIn("Angler-selected", prompts[0])

            ordinal = await coordinator.record_feedback(
                turn,
                artifact_ref="sha256:new",
                source_ref="sha256:judge",
                outcome="success",
                procedure_note="reuse the compatible interface procedure",
                family="repair",
            )
            self.assertEqual(ordinal, 1)
            self.assertEqual(len(backend.documents), 2)

        import asyncio

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()

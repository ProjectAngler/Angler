from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch
from torch import nn

from angler.memory import MemoryHit, MemoryProjection, MovingOriginIndex, SituatedMemory
from angler.reasoning.candidate_procedure_decoder import (
    CandidateProcedureDecode,
    CandidateProcedureDecoder,
)
from angler.reasoning.scalable_procedural_core import (
    ProceduralCoreConfig,
    ScalableProceduralCore,
    plastic_state_digest,
)
from angler.reasoning.situated_reader import SituatedFeatureSpec
from angler.runtime.conversation_bridge import (
    LocalConversationBridge,
    TaskLocalAction,
    load_conversation_checkpoint,
)
from angler.runtime.qwen_peft import foundation_tensor_digest
from angler.runtime.situated_qwen import LocalQwenIO


class _Backend:
    def __init__(self) -> None:
        self.documents: list[str] = []

    async def remember(self, document: str) -> str:
        self.documents.append(document)
        return f"memory-{len(self.documents)}"

    async def search(self, query: str, *, limit: int) -> tuple[MemoryHit, ...]:
        return tuple(
            MemoryHit(document=document, score=1.0 / (index + 1), backend_ref=f"hit-{index}")
            for index, document in enumerate(reversed(self.documents[-limit:]))
        )

    async def forget_dataset(self) -> None:
        self.documents.clear()


class _FrozenFoundation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.arange(8, dtype=torch.float32), requires_grad=False)


def _qwen() -> LocalQwenIO:
    qwen = LocalQwenIO(_FrozenFoundation(), object())

    def embed(texts):
        rows = []
        for text in texts:
            seed = sum(text.encode("utf-8")) % 97
            rows.append(torch.tensor([(seed + offset) / 100.0 for offset in range(8)]))
        return torch.stack(rows)

    qwen.embed = Mock(side_effect=embed)
    qwen.generate = Mock(return_value="Frozen Qwen response")
    return qwen


def _write_checkpoint(path: Path, qwen: LocalQwenIO) -> str:
    torch.manual_seed(31)
    config = ProceduralCoreConfig(
        content_width=8,
        temporal_width=8,
        model_width=8,
        depth=1,
        heads=2,
        procedure_tokens=3,
        plastic_slots=2,
        feedforward_multiplier=2,
    )
    core = ScalableProceduralCore(config)
    decoder_config = {
        "content_width": 8,
        "procedure_width": 8,
        "hidden_width": 8,
        "heads": 2,
        "maximum_actions": 4,
        "maximum_steps": 3,
    }
    decoder = CandidateProcedureDecoder(**decoder_config)
    state = core.initial_plastic_state()
    state = type(state)(
        keys=state.keys,
        values=state.values,
        strengths=torch.tensor([0.5, 0.0]),
        step=7,
    )
    torch.save(
        {
            "identity": "synthetic-supported-checkpoint",
            "foundation_digest": foundation_tensor_digest(qwen.model),
            "core_config": config.__dict__ if hasattr(config, "__dict__") else {
                field: getattr(config, field)
                for field in config.__dataclass_fields__
            },
            "core_state_dict": core.state_dict(),
            "decoder_config": decoder_config,
            "decoder_state_dict": decoder.state_dict(),
            "plastic_state": {
                "keys": state.keys,
                "values": state.values,
                "strengths": state.strengths,
                "step": state.step,
            },
        },
        path,
    )
    return plastic_state_digest(state)


class ConversationCheckpointTests(unittest.TestCase):
    def test_supported_checkpoint_loads_frozen_components_and_state(self) -> None:
        qwen = _qwen()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            expected_state = _write_checkpoint(path, qwen)
            loaded = load_conversation_checkpoint(path, qwen)

        self.assertEqual(loaded.identity, "synthetic-supported-checkpoint")
        self.assertEqual(loaded.plastic_state_identity, expected_state)
        self.assertEqual(loaded.plastic_state.step, 7)
        self.assertFalse(loaded.core.training)
        self.assertFalse(loaded.decoder.training)
        self.assertFalse(any(parameter.requires_grad for parameter in loaded.core.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in loaded.decoder.parameters()))

    def test_foundation_mismatch_and_malformed_checkpoint_fail_closed(self) -> None:
        first = _qwen()
        second = _qwen()
        with torch.no_grad():
            second.model.weight.add_(1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            _write_checkpoint(path, first)
            with self.assertRaisesRegex(ValueError, "different frozen Qwen"):
                load_conversation_checkpoint(path, second)
            malformed = Path(directory) / "malformed.pt"
            torch.save({"identity": "incomplete"}, malformed)
            with self.assertRaisesRegex(ValueError, "missing required fields"):
                load_conversation_checkpoint(malformed, first)


class LocalConversationBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.qwen = _qwen()
        self.temporary = tempfile.TemporaryDirectory()
        self.checkpoint_path = Path(self.temporary.name) / "checkpoint.pt"
        _write_checkpoint(self.checkpoint_path, self.qwen)
        self.backend = _Backend()
        self.memory = SituatedMemory(self.backend, origin=MovingOriginIndex())
        await self.memory.remember(
            MemoryProjection.from_mapping(
                artifact_ref="episode-old",
                text="Earlier evidence about validating inputs.",
                source_ref="sha256:source-old",
            )
        )
        await self.memory.remember(
            MemoryProjection.from_mapping(
                artifact_ref="episode-new",
                text="Recent evidence about preserving rollback.",
                source_ref="sha256:source-new",
            )
        )
        self.bridge = LocalConversationBridge.from_checkpoint(
            self.checkpoint_path,
            self.memory,
            self.qwen,
            feature_spec=SituatedFeatureSpec(),
        )

    async def asyncTearDown(self) -> None:
        self.temporary.cleanup()

    async def test_turn_exposes_recall_coordinates_plan_and_exact_qwen_prompt(self) -> None:
        self.bridge.checkpoint.decoder.forward = Mock(
            return_value=CandidateProcedureDecode(
                logits=torch.zeros(1, 3, 5),
                selected_indices=torch.tensor([[1, 0, 4]]),
            )
        )
        turn = await self.bridge.answer(
            "How should I approach this local change?",
            (
                TaskLocalAction("inspect", "Inspect the bounded inputs"),
                TaskLocalAction("test", "Run the focused tests"),
            ),
            response_constraint="Be concise.",
        )

        self.assertEqual([step.candidate_ref for step in turn.plan.steps], ["test", "inspect"])
        self.assertTrue(turn.plan.stopped)
        self.assertEqual(turn.response, "Frozen Qwen response")
        self.assertEqual(len(turn.evidence_attribution), 2)
        self.assertEqual([item.age for item in turn.evidence_attribution], [0, 1])
        self.assertIn("1. [test] Run the focused tests", turn.prompt)
        self.assertIn("2. [inspect] Inspect the bounded inputs", turn.prompt)
        self.assertIn("acquired=1; age=0", turn.prompt)
        self.qwen.generate.assert_called_once_with(turn.prompt)
        self.assertEqual(self.qwen.embed.call_count, 1)
        embedded_texts = self.qwen.embed.call_args.args[0]
        self.assertEqual(embedded_texts[-2:], ("Inspect the bounded inputs", "Run the focused tests"))

    async def test_external_feedback_is_preserved_and_can_advance_origin(self) -> None:
        self.bridge.checkpoint.decoder.forward = Mock(
            return_value=CandidateProcedureDecode(
                logits=torch.zeros(1, 3, 5),
                selected_indices=torch.tensor([[0, 4, 4]]),
            )
        )
        turn = await self.bridge.answer(
            "Prepare a safe patch.",
            (TaskLocalAction("patch", "Apply the scoped patch"),),
        )
        projection = self.bridge.package_feedback(
            turn,
            artifact_ref="feedback-1",
            source_ref="human:local-review",
            outcome="failure",
            feedback_text="The test expectation was incomplete.",
            family="code-review",
        )
        self.assertIn("External outcome: failure", projection.text)
        self.assertIn("The test expectation was incomplete.", projection.text)
        self.assertEqual(dict(projection.context)["feedback_origin"], "external")
        ordinal = await self.bridge.record_feedback(
            turn,
            artifact_ref="feedback-2",
            source_ref="human:local-review",
            outcome="success",
            feedback_text="The bounded tests now pass.",
            family="code-review",
        )
        self.assertEqual(ordinal, 2)
        self.assertEqual(self.memory.origin.now, 2)

    async def test_empty_recall_candidate_overflow_and_empty_generation_fail_closed(self) -> None:
        empty_memory = SituatedMemory(_Backend())
        empty_bridge = LocalConversationBridge.from_checkpoint(
            self.checkpoint_path,
            empty_memory,
            self.qwen,
            feature_spec=SituatedFeatureSpec(),
        )
        action = TaskLocalAction("one", "One local action")
        with self.assertRaisesRegex(LookupError, "no validated"):
            await empty_bridge.answer("query", (action,))
        with self.assertRaisesRegex(ValueError, "exceed"):
            await self.bridge.answer(
                "query",
                tuple(TaskLocalAction(str(index), f"action {index}") for index in range(5)),
            )
        self.bridge.checkpoint.decoder.forward = Mock(
            return_value=CandidateProcedureDecode(
                logits=torch.zeros(1, 3, 5),
                selected_indices=torch.tensor([[0, 4, 4]]),
            )
        )
        self.qwen.generate.return_value = ""
        with self.assertRaisesRegex(RuntimeError, "empty response"):
            await self.bridge.answer("query", (action,))

    async def test_absent_candidate_selection_fails_closed(self) -> None:
        self.bridge.checkpoint.decoder.forward = Mock(
            return_value=CandidateProcedureDecode(
                logits=torch.zeros(1, 3, 5),
                selected_indices=torch.tensor([[3, 4, 4]]),
            )
        )
        with self.assertRaisesRegex(RuntimeError, "absent task-local"):
            await self.bridge.answer(
                "query",
                (TaskLocalAction("only", "Only available action"),),
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn

from angler.memory import (
    MemoryHit,
    MemoryProjection,
    MovingOriginIndex,
    SituatedMemory,
    decode_projection,
)
from angler.reasoning.action_trace_matching import ActionTraceProceduralCore
from angler.reasoning.scalable_procedural_core import ProceduralCoreConfig
from angler.reasoning.situated_reader import SituatedFeatureSpec
from angler.runtime.autonomous_apprentice import (
    ActionTraceCompetenceLearner,
    AtomicCompetenceStore,
    AutonomousApprentice,
    AutonomousEpisode,
    AutonomousTaskContract,
    OutcomeContext,
    ProposedAction,
    RootedSoftwareSandbox,
    SituatedEpisodeRecorder,
    VerifierReceipt,
    parse_proposed_action,
)
from angler.runtime.situated_conversation import ConversationJournal
from angler.runtime.situated_qwen import LocalQwenIO
from experiments.runners import autonomous_apprentice_v1 as runner
from experiments.runners import autonomous_apprentice_v2 as runner_v2
from experiments.runners import autonomous_apprentice_v3 as runner_v3
from experiments.runners import autonomous_apprentice_v4 as runner_v4


class _Learner:
    def __init__(self) -> None:
        self.value = 0
        self.outcomes = []

    def capture_state(self) -> bytes:
        return str(self.value).encode()

    def restore_state(self, state: bytes) -> None:
        self.value = int(state.decode())

    def state_digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.capture_state()).hexdigest()

    def apply_outcome(self, outcome) -> None:
        self.outcomes.append(outcome)
        self.value += 1 if outcome.disposition == "success" else -1


class _Recorder:
    def __init__(self, *, fail: bool = False) -> None:
        self.episodes = []
        self.fail = fail

    async def record(self, episode) -> None:
        self.episodes.append(episode)
        if self.fail:
            raise RuntimeError("injected recorder failure")


class _FailingStore(AtomicCompetenceStore):
    def save(self, state: bytes) -> str:
        raise OSError("injected state-store failure")


class _SequencedPolicy:
    def __init__(self, actions) -> None:
        self.actions = iter(actions)
        self.prompts = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps(next(self.actions))


class _MemoryBackend:
    def __init__(self) -> None:
        self.documents = []

    async def remember(self, document: str) -> str:
        self.documents.append(document)
        return str(len(self.documents))

    async def search(self, query: str, *, limit: int):
        return tuple(MemoryHit(item, score=1.0) for item in self.documents[-limit:])

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
            seed = sum(text.encode("utf-8")) % 31
            rows.append(torch.tensor([(seed + offset) / 32.0 for offset in range(8)]))
        return torch.stack(rows)

    qwen.embed = embed
    return qwen


def _contract(
    root: Path,
    *,
    command=None,
    allow_mutation=True,
    request="Repair value.txt",
    **contract_options,
):
    command = command or (
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import sys; "
            "ok=Path('value.txt').read_text()=='correct\\n'; "
            "print('PRIVATE_EXPECTED_VALUE=correct' if not ok else 'PASS'); "
            "sys.exit(0 if ok else 1)"
        ),
    )
    return AutonomousTaskContract(
        task_id="task-1",
        request=request,
        success_description="the fixed objective verifier passes",
        workspace_root=root,
        readable_paths=("value.txt",),
        writable_paths=("value.txt",),
        verifier_command=tuple(command),
        allow_mutation=allow_mutation,
        max_steps=8,
        max_verifier_runs=3,
        **contract_options,
    )


class AutonomousApprenticeTests(unittest.IsolatedAsyncioTestCase):
    async def test_clarification_precedes_model_call_and_workspace_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "value.txt"
            target.write_text("original\n")
            policy = _SequencedPolicy([])
            learner = _Learner()
            recorder = _Recorder()
            contract = AutonomousTaskContract(
                task_id="unclear",
                request="",
                success_description="",
                workspace_root=root,
                readable_paths=("value.txt",),
                writable_paths=("value.txt",),
                verifier_command=(),
                allow_mutation=False,
            )
            result = await AutonomousApprentice(
                policy,
                learner,
                recorder,
                AtomicCompetenceStore(root / "state.bin"),
            ).run(contract)

            self.assertEqual(result.status, "CLARIFICATION_REQUIRED")
            self.assertEqual(len(result.clarification_questions), 3)
            self.assertEqual(target.read_text(), "original\n")
            self.assertFalse(policy.prompts)
            self.assertFalse(recorder.episodes)
            self.assertEqual(learner.value, 0)

    async def test_failed_attempt_retries_succeeds_and_learns_from_verifier_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "value.txt").write_text("broken\n")
            policy = _SequencedPolicy(
                [
                    {"action_id": "a1", "kind": "read", "path": "value.txt"},
                    {"action_id": "a2", "kind": "write", "path": "value.txt", "content": "still-wrong\n"},
                    {"action_id": "a3", "kind": "verify"},
                    {"action_id": "a4", "kind": "write", "path": "value.txt", "content": "correct\n"},
                    {"action_id": "a5", "kind": "finish"},
                ]
            )
            learner = _Learner()
            recorder = _Recorder()
            state_path = root / ".angler" / "state.bin"
            result = await AutonomousApprentice(
                policy,
                learner,
                recorder,
                AtomicCompetenceStore(state_path),
                experience=lambda request: "Earlier attempts benefited from inspecting before editing.",
            ).run(_contract(root))

            self.assertEqual(result.status, "SUCCEEDED")
            self.assertEqual([item.disposition for item in result.verifier_receipts], ["FAIL", "PASS"])
            self.assertEqual([item.outcome for item in recorder.episodes], ["failure", "success"])
            self.assertEqual([item.disposition for item in learner.outcomes], ["failure", "success"])
            self.assertEqual(learner.value, 0)
            self.assertEqual(state_path.read_bytes(), b"0")
            self.assertEqual((root / "value.txt").read_text(), "correct\n")
            self.assertIn("Objective verifier failed", policy.prompts[3])
            self.assertNotIn("PRIVATE_EXPECTED_VALUE", "\n".join(policy.prompts))
            self.assertNotIn("/success", "\n".join(policy.prompts))
            self.assertNotIn("/failure", "\n".join(policy.prompts))

    async def test_verifier_infrastructure_error_never_updates_competence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "value.txt").write_text("broken\n")
            policy = _SequencedPolicy([{"action_id": "v1", "kind": "verify"}])
            learner = _Learner()
            recorder = _Recorder()
            result = await AutonomousApprentice(
                policy,
                learner,
                recorder,
                AtomicCompetenceStore(root / "state.bin"),
            ).run(_contract(root, command=(str(root / "missing-verifier"),)))

            self.assertEqual(result.status, "INVALID")
            self.assertEqual(result.verifier_receipts[0].disposition, "ERROR")
            self.assertEqual(learner.value, 0)
            self.assertFalse(learner.outcomes)
            self.assertFalse(recorder.episodes)
            self.assertFalse((root / "state.bin").exists())

    async def test_opt_in_diagnostics_enable_self_repair_without_manual_grading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "value.txt").write_text("broken\n")
            policy = _SequencedPolicy(
                [
                    {"action_id": "1", "kind": "write", "path": "value.txt", "content": "wrong\n"},
                    {"action_id": "1", "kind": "verify"},
                    {"action_id": "1", "kind": "write", "path": "value.txt", "content": "correct\n"},
                    {"action_id": "1", "kind": "verify"},
                ]
            )
            result = await AutonomousApprentice(
                policy,
                _Learner(),
                _Recorder(),
                AtomicCompetenceStore(root / "state.bin"),
            ).run(_contract(root, reveal_verifier_diagnostics=True))

            self.assertEqual(result.status, "SUCCEEDED")
            self.assertIn("PRIVATE_EXPECTED_VALUE=correct", policy.prompts[2])

    async def test_opt_in_episode_retains_bounded_procedure_content_for_recall(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "value.txt").write_text("broken\n")
            backend = _MemoryBackend()
            memory = SituatedMemory(backend, origin=MovingOriginIndex())
            recorder = SituatedEpisodeRecorder(
                ConversationJournal(root / "journal.jsonl"),
                memory,
            )
            result = await AutonomousApprentice(
                _SequencedPolicy(
                    [
                        {"action_id": "w1", "kind": "write", "path": "value.txt", "content": "correct\n"},
                        {"action_id": "v1", "kind": "verify"},
                    ]
                ),
                _Learner(),
                recorder,
                AtomicCompetenceStore(root / "state.bin"),
            ).run(_contract(root, retain_procedure_content=True))

            self.assertEqual(result.status, "SUCCEEDED")
            projection, _ = decode_projection(backend.documents[0])
            self.assertIn("content:\ncorrect", projection.text)
            self.assertEqual(dict(projection.context)["outcome"], "success")

    async def test_identical_failed_episode_record_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = _MemoryBackend()
            recorder = SituatedEpisodeRecorder(
                ConversationJournal(root / "journal.jsonl"),
                SituatedMemory(backend, origin=MovingOriginIndex()),
            )
            episode = AutonomousEpisode(
                task_id="same-failure",
                request="repair the value",
                action_trace=(
                    ProposedAction("one", "write", path="value.txt", content="bad\n"),
                    ProposedAction("two", "verify"),
                ),
                verifier=VerifierReceipt("FAIL", 1, 0.01, "sha256:failure", 5, "failed"),
                outcome="failure",
                state_parent="sha256:parent",
                retain_procedure_content=True,
            )
            await recorder.record(episode)
            await recorder.record(episode)

            self.assertEqual(len(recorder.journal.records()), 1)
            self.assertEqual(len(backend.documents), 1)

    async def test_state_persistence_failure_restores_exact_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "value.txt").write_text("correct\n")
            learner = _Learner()
            parent = learner.state_digest()
            apprentice = AutonomousApprentice(
                _SequencedPolicy([
                    {"action_id": "r1", "kind": "read", "path": "value.txt"},
                    {"action_id": "v1", "kind": "verify"},
                ]),
                learner,
                _Recorder(),
                _FailingStore(root / "state.bin"),
            )
            with self.assertRaisesRegex(OSError, "state-store"):
                await apprentice.run(_contract(root))
            self.assertEqual(learner.state_digest(), parent)
            self.assertEqual(learner.value, 0)

    async def test_recorder_failure_prevents_learning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "value.txt").write_text("correct\n")
            learner = _Learner()
            apprentice = AutonomousApprentice(
                _SequencedPolicy([
                    {"action_id": "r1", "kind": "read", "path": "value.txt"},
                    {"action_id": "v1", "kind": "verify"},
                ]),
                learner,
                _Recorder(fail=True),
                AtomicCompetenceStore(root / "state.bin"),
            )
            with self.assertRaisesRegex(RuntimeError, "recorder"):
                await apprentice.run(_contract(root))
            self.assertEqual(learner.value, 0)
            self.assertFalse((root / "state.bin").exists())


class RootedSoftwareSandboxTests(unittest.TestCase):
    def test_path_escape_undeclared_write_and_conflicting_duplicate_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "value.txt").write_text("original\n")
            sandbox = RootedSoftwareSandbox(_contract(root))
            with self.assertRaisesRegex(ValueError, "relative"):
                sandbox.execute(ProposedAction("escape", "read", path="../outside.txt"))
            with self.assertRaisesRegex(PermissionError, "outside"):
                sandbox.execute(ProposedAction("other", "write", path="other.txt", content="x"))
            first = sandbox.execute(ProposedAction("same", "write", path="value.txt", content="one\n"))
            duplicate = sandbox.execute(ProposedAction("same", "write", path="value.txt", content="one\n"))
            self.assertEqual(first, duplicate)
            self.assertEqual((root / "value.txt").read_text(), "one\n")
            with self.assertRaisesRegex(ValueError, "conflicting"):
                sandbox.execute(ProposedAction("same", "write", path="value.txt", content="two\n"))

    def test_parser_rejects_commands_and_malformed_actions(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid fields"):
            parse_proposed_action('{"action_id":"x","kind":"verify","command":"erase"}')
        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            parse_proposed_action("not-json")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            parse_proposed_action('{"action_id":"x","kind":"shell"}')


class ActionTraceCompetenceLearnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_objective_action_trace_changes_replays_and_influences_recall_context(self) -> None:
        torch.manual_seed(17)
        config = ProceduralCoreConfig(
            content_width=8,
            temporal_width=8,
            model_width=8,
            depth=1,
            heads=2,
            procedure_tokens=2,
            plastic_slots=3,
            feedforward_multiplier=2,
        )
        core = ActionTraceProceduralCore(config, maximum_trace_steps=3)
        backend = _MemoryBackend()
        memory = SituatedMemory(backend, origin=MovingOriginIndex())
        await memory.remember(
            MemoryProjection.from_mapping(
                artifact_ref="episode-1",
                text="A prior related repair succeeded after inspecting the input.",
                source_ref="objective-verifier-1",
            )
        )
        qwen = _qwen()
        learner = ActionTraceCompetenceLearner(
            core,
            core.initial_plastic_state(),
            memory,
            qwen,
            feature_spec=SituatedFeatureSpec(),
            checkpoint_identity="tiny-action-trace",
        )
        initial = learner.state_digest()
        learner.apply_outcome(
            OutcomeContext(
                task_id="trace-task",
                request="repair the declared value",
                action_trace=(
                    ProposedAction("r", "read", path="value.txt"),
                    ProposedAction("w", "write", path="value.txt", content="fixed\n"),
                    ProposedAction("v", "verify"),
                ),
                disposition="success",
                verifier_ref="sha256:verifier",
            )
        )
        terminal = learner.state_digest()
        snapshot = learner.capture_state()

        self.assertNotEqual(initial, terminal)
        self.assertEqual(learner.state.step, 1)
        context = await learner.experience("repair another declared value")
        self.assertIn("prior related repair", context)
        restored = ActionTraceCompetenceLearner(
            ActionTraceProceduralCore(config, maximum_trace_steps=3),
            core.initial_plastic_state(),
            memory,
            qwen,
            feature_spec=SituatedFeatureSpec(),
            checkpoint_identity="tiny-action-trace",
        )
        restored.restore_state(snapshot)
        self.assertEqual(restored.state_digest(), terminal)
        self.assertEqual(restored.state.step, 1)

    async def test_failed_trace_is_counterexample_without_neural_consolidation(self) -> None:
        torch.manual_seed(19)
        config = ProceduralCoreConfig(
            content_width=8,
            temporal_width=8,
            model_width=8,
            depth=1,
            heads=2,
            procedure_tokens=2,
            plastic_slots=3,
            feedforward_multiplier=2,
        )
        core = ActionTraceProceduralCore(config, maximum_trace_steps=3)
        learner = ActionTraceCompetenceLearner(
            core,
            core.initial_plastic_state(),
            SituatedMemory(_MemoryBackend(), origin=MovingOriginIndex()),
            _qwen(),
            feature_spec=SituatedFeatureSpec(),
            checkpoint_identity="success-only-consolidation",
        )
        parent = learner.state_digest()
        learner.apply_outcome(
            OutcomeContext(
                task_id="failed",
                request="repair",
                action_trace=(ProposedAction("w", "write", path="value.txt", content="bad"),),
                disposition="failure",
                verifier_ref="sha256:failure",
            )
        )
        self.assertEqual(learner.state_digest(), parent)
        self.assertEqual(learner.state.step, 0)


class AutonomousApprenticeRunnerTests(unittest.TestCase):
    def test_frozen_tasks_repeat_four_mechanisms_across_acquisition_and_transfer(self) -> None:
        self.assertEqual(len(runner.TASKS), 8)
        mechanisms = {}
        for task in runner.TASKS:
            mechanisms.setdefault(task.mechanism, []).append(task.phase)
            self.assertNotIn(task.verifier_source, task.request)
            self.assertIn("solution.py", task.request)
        self.assertEqual(len(mechanisms), 4)
        self.assertTrue(all(sorted(phases) == ["acquisition", "transfer"] for phases in mechanisms.values()))

    def test_json_policy_extracts_structured_action_without_adding_fields(self) -> None:
        class _Generator:
            def generate(self, prompt):
                return '```json\n{"action_id":"one","kind":"verify"}\n```'

        policy = runner.QwenJsonPolicy(_Generator())
        self.assertEqual(
            json.loads(policy("prompt")),
            {"action_id": "one", "kind": "verify"},
        )
        self.assertEqual(policy.calls, 1)

    def test_causal_advantage_requires_success_or_half_step_gain(self) -> None:
        full = {"transfer_success_rate": 0.75, "transfer_mean_steps": 2.0}
        self.assertTrue(runner._strictly_better(full, {"transfer_success_rate": 0.5, "transfer_mean_steps": 1.0}))
        self.assertTrue(runner._strictly_better(full, {"transfer_success_rate": 0.75, "transfer_mean_steps": 2.5}))
        self.assertFalse(runner._strictly_better(full, {"transfer_success_rate": 0.75, "transfer_mean_steps": 2.25}))

    def test_v2_is_fresh_and_enables_only_declared_semantic_capabilities(self) -> None:
        self.assertNotEqual(runner_v2.IDENTITY, runner.IDENTITY)
        self.assertEqual(len(runner_v2.TASKS), 8)
        with tempfile.TemporaryDirectory() as directory:
            contract = runner_v2._task_contract(
                runner_v2.TASKS[0],
                Path(directory),
            )
        self.assertTrue(contract.reveal_verifier_diagnostics)
        self.assertTrue(contract.retain_procedure_content)
        self.assertEqual(contract.max_learning_trace_chars, 3_072)

    def test_v3_is_fresh_and_repeats_four_mechanisms(self) -> None:
        self.assertNotIn(runner_v3.IDENTITY, {runner.IDENTITY, runner_v2.IDENTITY})
        mechanisms = {}
        for task in runner_v3.TASKS:
            mechanisms.setdefault(task.mechanism, []).append(task.phase)
        self.assertEqual(len(runner_v3.TASKS), 8)
        self.assertEqual(len(mechanisms), 4)
        self.assertTrue(
            all(sorted(phases) == ["acquisition", "transfer"] for phases in mechanisms.values())
        )

    def test_v4_has_fresh_identity_and_informative_fixed_verifiers(self) -> None:
        self.assertNotIn(
            runner_v4.IDENTITY,
            {runner.IDENTITY, runner_v2.IDENTITY, runner_v3.IDENTITY},
        )
        self.assertEqual(len(runner_v4.TASKS), 8)
        self.assertTrue(
            all("got={got!r}; expected={expected!r}" in task.verifier_source for task in runner_v4.TASKS)
        )


if __name__ == "__main__":
    unittest.main()

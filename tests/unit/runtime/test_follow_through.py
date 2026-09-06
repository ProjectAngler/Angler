"""Follow-through: a person's turn continues while she judges it unfinished.

The judgment is hers; the runtime only orchestrates, binds provenance, and
bounds the loop by time. These tests cover the envelope, the person-turn view,
the judgment decoder, the runtime loop, and the console's joined transcript."""
import json
import os
import unittest
from dataclasses import dataclass

from angler.runtime import higher_level_autonomy_adapter as adapter
from angler.runtime.higher_level_autonomy_adapter import (
    FOLLOW_THROUGH_CONTINUATION_CONTRACT,
    FOLLOW_THROUGH_STATE_KEY,
    PersonTurnView,
    _follow_through_envelope,
    _follow_through_origin,
    _judge_follow_through,
    _person_turn_view,
)
from angler.runtime.jenny2_runtime import Jenny2Runtime, _canonical_json
from angler.runtime.persistent_autonomy import CycleObservation, SupervisorResult

REF = "sha256:" + "0" * 64


def _envelope(step=1, **overrides):
    value = {
        "contract": FOLLOW_THROUGH_CONTINUATION_CONTRACT,
        "human_trigger_ref": "chat:one",
        "human_observation_ref": REF,
        "human_message": "Speaker: Becca.\n\nWrite the standard and tell me the reference.",
        "undertaking": "Write Honest Reporting at my desk.",
        "step": step,
        "done_so_far": [{"affordance_id": "cortex.respond", "output": "I will write it now."}],
    }
    value.update(overrides)
    return value


class EnvelopeAndViewTest(unittest.TestCase):
    def test_only_the_follow_through_contract_is_recognized(self):
        self.assertIsNone(_follow_through_envelope("Speaker: Becca.\n\nHello"))
        self.assertIsNone(_follow_through_envelope(json.dumps({"contract": "other"})))
        self.assertEqual(_follow_through_envelope(json.dumps(_envelope()))["step"], 1)
        with self.assertRaises(ValueError):
            _follow_through_envelope(json.dumps(_envelope(step=0)))
        with self.assertRaises(ValueError):
            _follow_through_envelope(json.dumps({**_envelope(), "extra": 1}))
        with self.assertRaises(ValueError):
            _follow_through_envelope(json.dumps(_envelope(undertaking="  ")))

    def test_person_turn_view_shows_her_own_turn_continued(self):
        human = CycleObservation("chat:one", "HUMAN", "Speaker: Becca.\n\nHello")
        self.assertIs(_person_turn_view(human), human)
        continuation = CycleObservation("ft:1", "CONTINUATION", json.dumps(_envelope()))
        view = _person_turn_view(continuation)
        self.assertIsInstance(view, PersonTurnView)
        self.assertEqual(view.source, "HUMAN")
        self.assertEqual(view.trigger_ref, "ft:1")
        self.assertEqual(view.observation_ref, continuation.observation_ref)
        self.assertTrue(view.content.startswith("Speaker: Jenny; this is your own follow-through, step 1"))
        self.assertIn("Write the standard and tell me the reference.", view.content)
        self.assertIn("- cortex.respond: I will write it now.", view.content)
        self.assertIn("You said you would now: Write Honest Reporting at my desk.", view.content)
        other = CycleObservation("lib:1", "CONTINUATION", json.dumps({"contract": "other"}))
        self.assertIs(_person_turn_view(other), other)

    def test_origin_names_the_persons_turn_for_every_person_facing_source(self):
        human = CycleObservation("chat:one", "HUMAN", "Speaker: Becca.\n\nHello")
        origin = _follow_through_origin(human)
        self.assertEqual(origin["human_trigger_ref"], "chat:one")
        self.assertEqual(origin["step"], 0)
        continuation = CycleObservation("ft:1", "CONTINUATION", json.dumps(_envelope(step=2)))
        origin = _follow_through_origin(continuation)
        self.assertEqual(origin["human_trigger_ref"], "chat:one")
        self.assertEqual(origin["step"], 2)
        self.assertIsNone(_follow_through_origin(CycleObservation("tick", "SCHEDULER", "")))
        library = {
            "contract": adapter.LIBRARY_RESPONSE_CONTINUATION_CONTRACT,
            "human_observation_ref": REF,
            "human_trigger_ref": "chat:one",
            "library_choice_ref": REF,
            "library_episode_ref": REF,
            "library_event_ref": REF,
            "library_observation_ref": None,
            "library_receipt_ref": REF,
        }
        origin = _follow_through_origin(
            CycleObservation("lib:1", "CONTINUATION", _canonical_json(library)),
            request_text="Speaker: Becca.\n\nCheck the library.",
        )
        self.assertEqual(origin["human_trigger_ref"], "chat:one")
        self.assertEqual(origin["human_message"], "Speaker: Becca.\n\nCheck the library.")


class _Backend:
    model_ref = "test-backend"

    def __init__(self, raw):
        self.raw = raw
        self.calls = []

    def generate(self, *, system, user, max_new_tokens):
        self.calls.append((system, user, max_new_tokens))
        if isinstance(self.raw, Exception):
            raise self.raw
        return self.raw


class JudgmentTest(unittest.TestCase):
    def _origin(self):
        return _follow_through_origin(CycleObservation("chat:one", "HUMAN", "Speaker: Becca.\n\nWrite it."))

    def test_her_unfinished_judgment_carries_her_undertaking(self):
        backend = _Backend(json.dumps({"turn_complete": False, "undertaking": "Write it at my desk now.", "why": "I said I would."}))
        record = _judge_follow_through(backend, origin=self._origin(), affordance_id="cortex.respond", what_happened="I will write it.", named_states=[], commitments=[])
        self.assertFalse(record["turn_complete"])
        self.assertEqual(record["undertaking"], "Write it at my desk now.")
        self.assertEqual(record["human_trigger_ref"], "chat:one")
        self.assertEqual(record["model_ref"], "test-backend")
        self.assertNotIn("error", record)
        system, user, _ = backend.calls[0]
        self.assertIn("no 'next turn'", system.lower())
        self.assertIn("what_you_just_did", user)

    def test_complete_string_booleans_and_failures_end_the_turn(self):
        record = _judge_follow_through(_Backend(json.dumps({"turn_complete": "true", "undertaking": "", "why": "done"})), origin=self._origin(), affordance_id="cortex.respond", what_happened="Done.", named_states=[], commitments=[])
        self.assertTrue(record["turn_complete"])
        record = _judge_follow_through(_Backend(json.dumps({"turn_complete": False, "undertaking": ""})), origin=self._origin(), affordance_id="cortex.respond", what_happened="x", named_states=[], commitments=[])
        self.assertTrue(record["turn_complete"])
        self.assertIn("undertaking", record["error"])
        record = _judge_follow_through(_Backend(RuntimeError("model down")), origin=self._origin(), affordance_id="cortex.respond", what_happened="x", named_states=[], commitments=[])
        self.assertTrue(record["turn_complete"])
        self.assertIn("RuntimeError", record["error"])


class _FakeSupervisor:
    """Commits one episode per ingress and lets the test script her judgments."""

    def __init__(self, judgments):
        self.judgments = list(judgments)  # per committed cycle, in order
        self.ordinal = 10
        self.episodes = {}
        self.ingresses = []
        self.state = {}

    def _commit(self, trigger_ref, affordance, output):
        self.ordinal += 1
        episode_ref = f"sha256:{self.ordinal:064d}"
        self.episodes[episode_ref] = (affordance, output)
        judgment = self.judgments.pop(0) if self.judgments else None
        if judgment is not None:
            self.state[FOLLOW_THROUGH_STATE_KEY] = {**judgment, "human_trigger_ref": trigger_ref, "moving_origin_ordinal": self.ordinal}
        return SupervisorResult("COMMITTED", trigger_ref, affordance, episode_ref, None, self.ordinal, "ok")

    def human_ingress(self, trigger_ref, content):
        self.ingresses.append(("HUMAN", trigger_ref, content))
        return self._commit(trigger_ref, "cortex.respond", "I will write it now.")

    def continuation_ingress(self, trigger_ref, content):
        self.ingresses.append(("CONTINUATION", trigger_ref, content))
        envelope = json.loads(content)
        if envelope["step"] == 1:
            return self._commit(trigger_ref, "internal.authored-artifact", "Written at your desk and kept: sha256:abc")
        return self._commit(trigger_ref, "cortex.respond", "Done; the reference is sha256:abc.")

    def state_bytes(self):
        return json.dumps(self.state).encode("utf-8")


class _RuntimeShell:
    """Jenny2Runtime's follow-through methods over a fake supervisor."""

    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.projections = 0

    def turn_output(self, episode_ref):
        affordance, output = self.supervisor.episodes[episode_ref]
        return output, "COMPLETED_UNEVALUATED"

    def schedule_pending_projections(self):
        self.projections += 1

    _library_response_continuation = Jenny2Runtime._library_response_continuation
    _follow_through = Jenny2Runtime._follow_through
    _follow_through_judgment = Jenny2Runtime._follow_through_judgment
    _append_turn_chain = Jenny2Runtime._append_turn_chain
    _turn_chain_store = Jenny2Runtime._turn_chain_store
    turn_chain = Jenny2Runtime.turn_chain
    turn_chain_output = Jenny2Runtime.turn_chain_output
    chat = Jenny2Runtime.chat


class RuntimeLoopTest(unittest.TestCase):
    def test_she_continues_until_she_judges_the_turn_complete(self):
        supervisor = _FakeSupervisor([
            {"turn_complete": False, "undertaking": "Write Honest Reporting at my desk."},
            {"turn_complete": False, "undertaking": "Tell Becca the reference."},
            {"turn_complete": True, "undertaking": ""},
        ])
        runtime = _RuntimeShell(supervisor)
        content = "Speaker: Becca.\n\nWrite it and give me the reference."
        result = runtime.chat("chat:one", content)
        self.assertEqual(result.status, "COMMITTED")
        self.assertEqual(result.selected_affordance_id, "cortex.respond")
        self.assertEqual([kind for kind, _, _ in supervisor.ingresses], ["HUMAN", "CONTINUATION", "CONTINUATION"])
        first = json.loads(supervisor.ingresses[1][2])
        self.assertEqual(first["contract"], FOLLOW_THROUGH_CONTINUATION_CONTRACT)
        self.assertEqual(first["human_trigger_ref"], "chat:one")
        self.assertEqual(first["human_observation_ref"], CycleObservation("chat:one", "HUMAN", content).observation_ref)
        self.assertEqual(first["undertaking"], "Write Honest Reporting at my desk.")
        self.assertEqual(first["done_so_far"], [{"affordance_id": "cortex.respond", "output": "I will write it now."}])
        second = json.loads(supervisor.ingresses[2][2])
        self.assertEqual(second["step"], 2)
        self.assertEqual([item["affordance_id"] for item in second["done_so_far"]], ["cortex.respond", "internal.authored-artifact"])
        chain = runtime.turn_chain("chat:one")
        self.assertEqual([item["affordance_id"] for item in chain], ["cortex.respond", "internal.authored-artifact", "cortex.respond"])
        output, status = runtime.turn_chain_output("chat:one")
        self.assertEqual(output, "I will write it now.\n\nWritten at your desk and kept: sha256:abc\n\nDone; the reference is sha256:abc.")
        self.assertEqual(status, "COMPLETED_UNEVALUATED")
        self.assertEqual(runtime.turn_chain_output("chat:one"), ("", ""))
        self.assertEqual(runtime.projections, 1)

    def test_a_complete_judgment_or_none_leaves_the_turn_as_one_episode(self):
        for judgments in ([{"turn_complete": True, "undertaking": ""}], [], [{"turn_complete": False, "undertaking": "   "}]):
            supervisor = _FakeSupervisor(judgments)
            runtime = _RuntimeShell(supervisor)
            runtime.chat("chat:one", "Speaker: Becca.\n\nHello")
            self.assertEqual(len(supervisor.ingresses), 1)
            self.assertEqual(runtime.turn_chain_output("chat:one"), ("", ""))

    def test_the_only_bound_is_the_time_budget_and_it_is_reported_in_her_words(self):
        supervisor = _FakeSupervisor([{"turn_complete": False, "undertaking": "Keep going."}])
        runtime = _RuntimeShell(supervisor)
        previous = os.environ.get("JENNY2_FOLLOW_THROUGH_SECONDS")
        os.environ["JENNY2_FOLLOW_THROUGH_SECONDS"] = "0"
        try:
            runtime.chat("chat:one", "Speaker: Becca.\n\nHello")
        finally:
            if previous is None:
                del os.environ["JENNY2_FOLLOW_THROUGH_SECONDS"]
            else:
                os.environ["JENNY2_FOLLOW_THROUGH_SECONDS"] = previous
        self.assertEqual(len(supervisor.ingresses), 1)
        output, _ = runtime.turn_chain_output("chat:one")
        self.assertIn("follow-through budget", output)
        self.assertIn("Keep going.", output)


class ConsoleTranscriptTest(unittest.TestCase):
    def test_transcript_joins_her_chain_into_one_visible_turn(self):
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location(
            "jenny2_chat_for_follow_through_test",
            os.path.join(os.path.dirname(adapter.__file__), "..", "..", "..", "scripts", "jenny2_chat.py"),
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        @dataclass
        class _Episode:
            ordinal: int
            episode_ref: str
            payload_json: str

        @dataclass
        class _Head:
            moving_origin_ordinal: int

        def episode(ordinal, source, trigger, content, affordance, output):
            return _Episode(ordinal, f"sha256:{ordinal:064d}", json.dumps({
                "observation": {"trigger_ref": trigger, "source": source, "content": content},
                "choice": {"selected_affordance_id": affordance},
                "receipt": {"status": "COMPLETED_UNEVALUATED", "output": output},
            }))

        human = "Speaker: Becca.\n\nWrite it."
        chat_ref = "sha256:" + "1" * 64
        ft1 = "sha256:" + "2" * 64
        ft2 = "sha256:" + "3" * 64
        env1 = _canonical_json(_envelope(step=1, human_message=human, human_trigger_ref=chat_ref))
        env2 = _canonical_json(_envelope(step=2, human_message=human, human_trigger_ref=chat_ref))
        episodes = (
            episode(1, "HUMAN", chat_ref, human, "cortex.respond", "I will write it now."),
            episode(2, "CONTINUATION", ft1, env1, "internal.authored-artifact", "Written at your desk and kept: sha256:abc"),
            episode(3, "CONTINUATION", ft2, env2, "cortex.respond", "Done; sha256:abc."),
        )

        class _Supervisor:
            def state_head(self):
                return _Head(3)

            def episode_items(self, *, after_ordinal, limit):
                return episodes

        class _Runtime:
            supervisor = _Supervisor()

        turns = module._recent_owner_conversation(_Runtime())
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["response"], "I will write it now.\n\nWritten at your desk and kept: sha256:abc\n\nDone; sha256:abc.")
        self.assertEqual(turns[0]["response_episode_ref"], f"sha256:{3:064d}")
        self.assertEqual(turns[0]["status"], "COMPLETED_UNEVALUATED")


if __name__ == "__main__":
    unittest.main()


class DeskRefusalAndStepFailureTest(unittest.TestCase):
    def test_a_desk_refusal_is_an_error_receipt_she_can_read(self):
        from angler.runtime.authored_artifact import (
            AuthoredArtifactExecutor,
            desk_refusal_payload,
            desk_refusal_reason,
        )
        from angler.runtime.persistent_autonomy import AffordanceRequest
        payload = desk_refusal_payload('{"title": "x"}', "version 1 may not supersede another artifact")
        self.assertEqual(desk_refusal_reason(payload), "version 1 may not supersede another artifact")
        self.assertIsNone(desk_refusal_reason('{"title": "x"}'))
        executor = AuthoredArtifactExecutor()
        request = AffordanceRequest(
            idempotency_key=REF, trigger_ref=REF, affordance_id=executor.AFFORDANCE_ID,
            observation_ref=REF, state_head_ref=REF, action_payload=payload,
        )
        receipt = executor(request)
        self.assertEqual(receipt.status, "ERROR")
        self.assertIn("version 1 may not supersede", receipt.output)
        self.assertIsNone(receipt.observable_consequence)
        malformed = AffordanceRequest(
            idempotency_key=REF, trigger_ref=REF, affordance_id=executor.AFFORDANCE_ID,
            observation_ref=REF, state_head_ref=REF, action_payload='{"contract": "nope"}',
        )
        self.assertEqual(executor(malformed).status, "ERROR")

    def test_a_failed_step_ends_the_loop_and_keeps_the_committed_turn(self):
        class _Failing(_FakeSupervisor):
            def continuation_ingress(self, trigger_ref, content):
                self.ingresses.append(("CONTINUATION", trigger_ref, content))
                raise ValueError("adaptive next internal request must name an unfinished pattern")

        supervisor = _Failing([{"turn_complete": False, "undertaking": "Retire the combined work."}])
        runtime = _RuntimeShell(supervisor)
        result = runtime.chat("chat:one", "Speaker: Becca.\n\nHello")
        self.assertEqual(result.status, "COMMITTED")
        self.assertEqual(len(supervisor.ingresses), 2)
        output, status = runtime.turn_chain_output("chat:one")
        self.assertTrue(output.startswith("I will write it now."))
        self.assertIn("follow-through step 1 failed", output)
        self.assertIn("Retire the combined work.", output)
        self.assertEqual(status, "COMPLETED_UNEVALUATED")

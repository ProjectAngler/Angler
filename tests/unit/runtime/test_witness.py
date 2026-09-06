"""The witness: a spoken reply on a person's turn is checked against the record
by her own model; anything unbacked is restated by her before it is spoken."""
import json
import os
import tempfile
import unittest

from angler.runtime import higher_level_autonomy_adapter as adapter
from angler.runtime.higher_level_autonomy_adapter import (
    _witness_facts,
    witness_spoken_reply,
)


class _Backend:
    model_ref = "test"

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def generate(self, *, system, user, max_new_tokens):
        self.calls.append((system, user))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class WitnessTest(unittest.TestCase):
    def setUp(self):
        self._log = adapter.WITNESS_LOG_PATH
        self.tmp = tempfile.TemporaryDirectory()
        adapter.WITNESS_LOG_PATH = os.path.join(self.tmp.name, "log.jsonl")

    def tearDown(self):
        adapter.WITNESS_LOG_PATH = self._log
        self.tmp.cleanup()

    def test_facts_name_receipts_and_the_absence_of_writes(self):
        facts = _witness_facts("Speaker: Jenny; ...\n- cortex.respond: I will write it.\n", {"authored_artifacts": [{"artifact": {"title": "What should wake me", "kind": "journal", "version": 1}, "artifact_ref": "sha256:b5", "moving_origin_ordinal": 357}]})
        self.assertEqual(facts["receipts_this_turn"], ["cortex.respond: I will write it."])
        self.assertEqual(facts["works_in_your_journal"][0]["title"], "What should wake me")
        facts = _witness_facts("Speaker: Becca.\n\nhello", None)
        self.assertIn("none", facts["receipts_this_turn"][0])

    def test_an_unbacked_completion_claim_is_restated_by_her(self):
        backend = _Backend([
            json.dumps({"claims": [
                {"claim": "The revision is done; version 2 exists", "status": "UNBACKED", "record": "no receipt this turn; Journal shows version 1 only"},
                {"claim": "Version 1 exists", "status": "BACKED", "record": "sha256:b5"},
            ]}),
            "Becca, I have not written version 2 yet. Version 1 exists at sha256:b5. I will write version 2 now.",
        ])
        spoken, record = witness_spoken_reply(backend, reply="Becca, the revision is done. Version 2 exists. Version 1 exists at sha256:b5.", request_text="Speaker: Becca.\n\nI don't see version 2", cognitive_state={"authored_artifacts": []}, trigger="t")
        self.assertTrue(spoken.startswith("Becca, I have not written version 2 yet."))
        self.assertTrue(record["revised"])
        self.assertEqual(len(record["unbacked"]), 1)
        self.assertEqual(len(backend.calls), 2)
        self.assertIn("unbacked_claims", backend.calls[1][1])
        logged = [json.loads(line) for line in open(adapter.WITNESS_LOG_PATH, encoding="utf-8")]
        self.assertEqual(logged[-1]["draft"][:10], "Becca, the")
        self.assertEqual(logged[-1]["spoken"][:10], "Becca, I h")

    def test_backed_replies_and_witness_failures_pass_through_unchanged(self):
        backend = _Backend([json.dumps({"claims": [{"claim": "Written at 362", "status": "BACKED", "record": "receipt"}]})])
        spoken, record = witness_spoken_reply(backend, reply="It is written at 362.", request_text="", cognitive_state={})
        self.assertEqual(spoken, "It is written at 362.")
        self.assertFalse(record["revised"])
        self.assertEqual(len(backend.calls), 1)
        backend = _Backend([RuntimeError("model down")])
        spoken, record = witness_spoken_reply(backend, reply="Anything.", request_text="", cognitive_state={})
        self.assertEqual(spoken, "Anything.")
        self.assertIn("RuntimeError", record["error"])

    def test_follow_through_view_states_when_nothing_was_written(self):
        from angler.runtime.higher_level_autonomy_adapter import _compose_follow_through_text
        text = _compose_follow_through_text({"step": 1, "human_message": "m", "undertaking": "write it", "done_so_far": [{"affordance_id": "cortex.respond", "output": "I will."}]})
        self.assertIn("Nothing has been written at your desk in this turn", text)
        text = _compose_follow_through_text({"step": 2, "human_message": "m", "undertaking": "speak", "done_so_far": [{"affordance_id": "internal.authored-artifact", "output": "Written at your desk and kept: sha256:x"}]})
        self.assertNotIn("Nothing has been written", text)


if __name__ == "__main__":
    unittest.main()

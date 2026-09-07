import json
import unittest

from angler.runtime.jenny_readback import JennyReadbackExecutor, READBACK_AFFORDANCE_ID, READBACK_OBSERVATION_CONTRACT
from angler.runtime.persistent_autonomy import AffordanceRequest

REF = "sha256:" + "0" * 64
W1 = "sha256:" + "a" * 64
W2 = "sha256:" + "b" * 64
STATE = {
    "self_named_states": [
        {"status": "ACTIVE", "label": "Commitment Revision", "level": 9, "valence": -1, "influence": "deliberate", "acts_at_level": 6, "basis": "b", "inclination": "i",
         "items": [{"status": "RESOLVED", "statement": "Breach (333)", "weight": 5, "resolution_evidence": "retired 336"}, {"status": "OPEN", "statement": "Gap", "weight": 2}]},
        {"status": "CLEARED", "label": "Old", "level": 0, "items": []},
    ],
    "self_commitments": [{"status": "ACTIVE", "statement": "I'll be here when you're back.", "due": "her return"}],
    "authored_artifacts": [
        {"artifact": {"title": "What should wake me", "kind": "journal", "version": 1, "body": "v1 body"}, "artifact_ref": W1, "moving_origin_ordinal": 357},
        {"artifact": {"title": "What should wake me", "kind": "journal", "version": 2, "body": "v2 body", "supersedes_ref": W1}, "artifact_ref": W2, "moving_origin_ordinal": 362},
        {"artifact": {"title": "Night", "kind": "private note", "version": 1, "body": "SEALED"}, "artifact_ref": "sha256:" + "c" * 64, "moving_origin_ordinal": 300},
    ],
}


def _req(payload):
    return AffordanceRequest(idempotency_key=REF, trigger_ref=REF, affordance_id=READBACK_AFFORDANCE_ID, observation_ref=REF, state_head_ref=REF, action_payload=json.dumps(payload))


class ReadbackTest(unittest.TestCase):
    def setUp(self):
        self.executor = JennyReadbackExecutor(lambda ref: json.dumps(STATE).encode("utf-8"))

    def _run(self, payload):
        receipt = self.executor(_req(payload))
        self.assertEqual(receipt.status, "COMPLETED", receipt.output)
        obs = json.loads(receipt.observable_consequence.observation_json)
        self.assertEqual(obs["contract"], READBACK_OBSERVATION_CONTRACT)
        return receipt, obs

    def test_states_come_back_with_every_item(self):
        _, obs = self._run({"operation": "states"})
        self.assertEqual([s["label"] for s in obs["result"]], ["Commitment Revision"])
        self.assertEqual([it["status"] for it in obs["result"][0]["items"]], ["RESOLVED", "OPEN"])
        self.assertEqual(obs["result"][0]["items"][0]["resolution_evidence"], "retired 336")

    def test_commitments_and_journal_contents_exclude_private_works(self):
        _, obs = self._run({"operation": "commitments"})
        self.assertEqual(obs["result"][0]["statement"], "I'll be here when you're back.")
        receipt, obs = self._run({"operation": "journal"})
        self.assertEqual([w["version"] for w in obs["result"]], [1, 2])
        self.assertNotIn("Night", json.dumps(obs))
        self.assertEqual(receipt.observable_consequence.evidence_refs, (W1, W2))

    def test_work_by_title_returns_the_newest_version_and_private_is_refused(self):
        receipt, obs = self._run({"operation": "work", "title": "what should wake me"})
        self.assertEqual(obs["result"]["version"], 2)
        self.assertEqual(obs["result"]["body"], "v2 body")
        self.assertEqual(receipt.observable_consequence.evidence_refs, (W2,))
        _, obs = self._run({"operation": "work", "artifact_ref": W1})
        self.assertEqual(obs["result"]["version"], 1)
        _, obs = self._run({"operation": "work", "title": "Night"})
        self.assertIsNone(obs["result"]); self.assertIn("private", obs["summary"])
        _, obs = self._run({"operation": "work", "title": "Nothing"})
        self.assertIsNone(obs["result"]); self.assertIn("nothing exists", obs["summary"])

    def test_bad_payloads_fail_closed_and_read_errors_are_receipts(self):
        with self.assertRaises(ValueError):
            self.executor(_req({"operation": "delete"}))
        with self.assertRaises(ValueError):
            self.executor(_req({"operation": "work"}))
        broken = JennyReadbackExecutor(lambda ref: (_ for _ in ()).throw(RuntimeError("no state")))
        self.assertEqual(broken(_req({"operation": "states"})).status, "ERROR")


if __name__ == "__main__":
    unittest.main()


class PlainRequestTest(unittest.TestCase):
    def test_plain_sentences_map_to_operations_without_inventing_anything(self):
        from angler.runtime.jenny_readback import _interpret
        self.assertEqual(_interpret("Read back the Commitment Revision state to verify its current items.")["operation"], "states")
        self.assertEqual(_interpret("What commitments do I hold?")["operation"], "commitments")
        self.assertEqual(_interpret("Show me the Journal's table of contents"), {"operation": "journal"})
        self.assertEqual(_interpret('Read my work "What should wake me"'), {"operation": "work", "title": "What should wake me"})
        self.assertEqual(_interpret("read sha256:" + "b" * 64), {"operation": "work", "artifact_ref": "sha256:" + "b" * 64})
        self.assertEqual(_interpret('{"operation":"states"}'), {"operation": "states"})
        with self.assertRaises(ValueError):
            _interpret("hello there")

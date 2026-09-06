import json
import unittest
from dataclasses import dataclass

from angler.runtime.jenny_recall import (
    JennyRecallExecutor,
    RECALL_AFFORDANCE_ID,
    RECALL_OBSERVATION_CONTRACT,
    RECALL_SOURCE_REF,
)
from angler.runtime.persistent_autonomy import AffordanceRequest

REF = "sha256:" + "0" * 64


@dataclass
class _Memory:
    record_ref: str
    content: str
    semantic_distance: float
    acquired_ordinal: int
    event_time_utc: str | None = None


class _FakeMemory:
    def __init__(self, items):
        self.items = items
        self.calls = []

    def recall(self, request, *, limit):
        self.calls.append((request, limit))
        return self.items[:limit]


def _request(payload):
    return AffordanceRequest(
        idempotency_key=REF,
        trigger_ref=REF,
        affordance_id=RECALL_AFFORDANCE_ID,
        observation_ref=REF,
        state_head_ref=REF,
        action_payload=json.dumps(payload, sort_keys=True, separators=(",", ":")),
    )


class RecallExecutorTest(unittest.TestCase):
    def test_recall_returns_bounded_source_bound_results(self):
        memory = _FakeMemory([
            _Memory("sha256:" + "a" * 64, "Becca said the workstation is her home.", 0.12, 264),
            _Memory("sha256:" + "b" * 64, "x" * 5000, 0.30, 100),
        ])
        executor = JennyRecallExecutor(memory, now_ordinal=lambda: 320)
        receipt = executor(_request({"limit": 5, "operation": "recall", "query": "her home", "recall_purpose": "catalog"}))
        self.assertEqual(receipt.status, "COMPLETED")
        self.assertEqual(memory.calls, [("her home", 5)])
        payload = json.loads(receipt.observable_consequence.observation_json)
        self.assertEqual(payload["contract"], RECALL_OBSERVATION_CONTRACT)
        self.assertEqual(payload["result_count"], 2)
        self.assertEqual(payload["results"][0]["ordinals_ago"], 56)
        self.assertEqual(len(payload["results"][1]["content"]), 2000)
        self.assertEqual(receipt.observable_consequence.source_ref, RECALL_SOURCE_REF)
        self.assertEqual(receipt.observable_consequence.evidence_refs, ("sha256:" + "a" * 64, "sha256:" + "b" * 64))

    def test_bad_payloads_fail_closed_and_memory_errors_become_error_receipts(self):
        executor = JennyRecallExecutor(_FakeMemory([]))
        with self.assertRaises(ValueError):
            executor(_request({"limit": 99, "operation": "recall", "query": "x", "recall_purpose": ""}))
        with self.assertRaises(ValueError):
            executor(_request({"limit": 3, "operation": "search", "query": "x", "recall_purpose": ""}))

        class _Broken:
            def recall(self, request, *, limit):
                raise RuntimeError("cognee down")

        receipt = JennyRecallExecutor(_Broken())(_request({"limit": 3, "operation": "recall", "query": "x", "recall_purpose": ""}))
        self.assertEqual(receipt.status, "ERROR")
        self.assertIsNone(receipt.observable_consequence)


if __name__ == "__main__":
    unittest.main()

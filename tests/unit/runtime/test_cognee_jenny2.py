from __future__ import annotations

import hashlib
import json
import unittest

from angler.runtime.cognee_jenny2 import (
    CogneeJennyCapabilityBackend,
    CogneeJennyReferenceBackend,
)


class _Wire:
    def __init__(self, **values):
        self.__dict__.update(values)


class _Bindings:
    DataPoint = _Wire
    Edge = _Wire
    NodeSet = _Wire
    node_set_name = "jenny-2-test-records"

    def __init__(self):
        self.points = []

    async def add_data_points(self, points):
        self.points.extend(points)
        return str(points[0].id)

    async def search_references(self, query, *, limit):
        point = self.points[0]
        return [
            {
                "search_result": [
                    {"score": 0.125, "payload": {"record_ref": point.record_ref}}
                ]
            }
        ]

    async def close(self):
        return None


class CogneeJenny2Tests(unittest.IsolatedAsyncioTestCase):
    async def test_projects_only_reference_point_and_returns_only_canonical_ref(self):
        bindings = _Bindings()
        backend = CogneeJennyReferenceBackend(bindings)  # type: ignore[arg-type]
        episode_ref = "sha256:" + "a" * 64
        event_ref = "sha256:" + "b" * 64
        backend_ref = await backend.project(
            record_ref=episode_ref,
            content="Fallible consolidation proposal.",
            provenance_refs=(episode_ref, event_ref),
            acquired_ordinal=3,
        )
        self.assertTrue(backend_ref)
        point = bindings.points[0]
        self.assertEqual(point.record_ref, episode_ref)
        self.assertEqual(point.source_ref, event_ref)
        self.assertEqual(point.epistemic_status, "PROPOSED")
        self.assertEqual(point.memory_kind, "EPISODIC")
        self.assertEqual(point.references, [])
        hits = await backend.search("proposal", limit=4)
        self.assertEqual(hits[0].record_ref, episode_ref)
        self.assertEqual(hits[0].semantic_distance, 0.125)

    async def test_procedural_case_keeps_observed_type_and_feedback_edge(self):
        bindings = _Bindings()
        backend = CogneeJennyReferenceBackend(bindings)  # type: ignore[arg-type]
        episode_ref = "sha256:" + "a" * 64
        event_ref = "sha256:" + "b" * 64
        target_episode_ref = "sha256:" + "c" * 64
        core = {
            "condition": "Why did the earlier action occur?",
            "contract": "jenny.procedural-experience.v1",
            "epistemic_status": "OBSERVED_PROCEDURAL_OUTCOME",
            "feedback": {
                "feedback_source_ref": "sha256:" + "d" * 64,
                "feedback_text": "Identify the initiating request.",
                "target_episode_ref": target_episode_ref,
                "target_choice_ref": "sha256:" + "e" * 64,
                "target_receipt_ref": "sha256:" + "f" * 64,
            },
            "goal": "Explain the exact causal antecedent.",
            "memory_kind": "PROCEDURAL_EXPERIENCE",
            "outcome": {},
            "prediction": "The correction will improve causal attribution.",
            "procedure": {},
            "provenance": {
                "event_ref": event_ref,
                "choice_ref": "sha256:" + "1" * 64,
                "receipt_ref": "sha256:" + "2" * 64,
                "parent_state_ref": "sha256:" + "3" * 64,
                "child_state_ref": "sha256:" + "4" * 64,
            },
            "temporal": {"moving_origin_ordinal": 7},
        }
        case_ref = "sha256:" + hashlib.sha256(
            json.dumps(core, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        content = json.dumps(
            {**core, "case_ref": case_ref},
            separators=(",", ":"),
            sort_keys=True,
        )

        await backend.project(
            record_ref=episode_ref,
            content=content,
            provenance_refs=(episode_ref, event_ref),
            acquired_ordinal=7,
        )

        point = bindings.points[0]
        self.assertEqual(point.memory_kind, "PROCEDURAL")
        self.assertEqual(
            point.epistemic_status, "OBSERVED_PROCEDURAL_OUTCOME"
        )
        self.assertEqual(point.adjacent_record_refs, [target_episode_ref])
        self.assertEqual(len(point.references), 1)
        edge, target = point.references[0]
        self.assertEqual(edge.relationship_type, "FEEDBACK_ON")
        self.assertEqual(edge.properties, {"target_ref": target_episode_ref})
        self.assertEqual(target.record_ref, target_episode_ref)

    async def test_capability_backend_uses_a_procedural_reference_point(self):
        bindings = _Bindings()
        backend = CogneeJennyCapabilityBackend(bindings)  # type: ignore[arg-type]
        capability_ref = "sha256:" + "c" * 64
        state_ref = "sha256:" + "d" * 64
        backend_ref = await backend.project(
            record_ref=capability_ref,
            content='{"capability_key":"learned:test","procedure":"Verify it."}',
            provenance_refs=(capability_ref, state_ref),
            acquired_ordinal=7,
        )
        self.assertTrue(backend_ref)
        point = bindings.points[0]
        self.assertEqual(point.record_ref, capability_ref)
        self.assertEqual(point.source_ref, state_ref)
        self.assertEqual(point.memory_kind, "PROCEDURAL")
        self.assertEqual(
            point.epistemic_status,
            "MODEL_PROPOSAL_WITH_OBSERVED_CONSEQUENCE",
        )
        hits = await backend.search("verify", limit=4)
        self.assertEqual(hits[0].capability_ref, capability_ref)
        self.assertEqual(hits[0].semantic_distance, 0.125)


if __name__ == "__main__":
    unittest.main()

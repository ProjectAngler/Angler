"""Focused tests: typed retrieval views for read knowledge and notes."""

from __future__ import annotations

import hashlib
import json
import unittest

from angler.runtime.reading_experience import (
    READING_EPISTEMIC_STATUS,
    READING_EXPERIENCE_CONTRACT,
    reading_content,
    reading_projection_metadata,
)


def _ref(tag: str) -> str:
    return "sha256:" + hashlib.sha256(tag.encode()).hexdigest()


def _episode(passage: str = "Yet the life of the group goes on.") -> dict:
    observation = {
        "artifact_ref": _ref("artifact"),
        "author": "John Dewey",
        "catalog_ref": _ref("catalog"),
        "content": passage,
        "contract": "jenny.library.passage-observation.v1",
        "eof": False,
        "item_path": "05-learning-and-adaptation/pg852_dewey.txt",
        "next_cursor": 8192,
        "normalized_span": {"end": 8192, "start": 0},
        "reading_purpose": "retrieve the requested line",
        "title": "Democracy and Education",
    }
    return {
        "event_ref": _ref("event"),
        "receipt": {
            "status": "COMPLETED",
            "output": "read",
            "consequence": [],
            "observable_consequence": {
                "version": "jenny.observable-consequence.v1",
                "source_kind": "WORLD",
                "source_ref": _ref("source"),
                "request_ref": _ref("request"),
                "artifact_refs": [],
                "evidence_refs": [],
                "observation_json": json.dumps(observation),
            },
        },
        "temporal": {"moving_origin_ordinal": 213},
    }


class ReadingContentTest(unittest.TestCase):
    def test_library_read_becomes_a_typed_bounded_view(self) -> None:
        content = reading_content(_episode())
        self.assertIsNotNone(content)
        view = json.loads(content)
        self.assertEqual(view["contract"], READING_EXPERIENCE_CONTRACT)
        self.assertEqual(view["memory_kind"], "LIBRARY_PASSAGE")
        self.assertEqual(view["epistemic_status"], READING_EPISTEMIC_STATUS)
        self.assertEqual(view["title"], "Democracy and Education")
        self.assertIn("life of the group", view["passage"])
        self.assertEqual(view["moving_origin_ordinal"], 213)
        self.assertLessEqual(len(content), 4_096)

    def test_long_passages_are_excerpted_within_the_bound(self) -> None:
        content = reading_content(_episode("word " * 3_000))
        self.assertLessEqual(len(content), 4_096)
        self.assertTrue(json.loads(content)["passage"].endswith("…"))

    def test_non_reading_episodes_return_none(self) -> None:
        episode = _episode()
        episode["receipt"]["observable_consequence"]["source_kind"] = "SELF"
        self.assertIsNone(reading_content(episode))
        self.assertIsNone(reading_content({"receipt": {"status": "COMPLETED_UNEVALUATED"}}))

    def test_malformed_passage_claims_fail_closed(self) -> None:
        episode = _episode()
        observed = episode["receipt"]["observable_consequence"]
        bad = json.loads(observed["observation_json"])
        bad["normalized_span"] = "not a span"
        observed["observation_json"] = json.dumps(bad)
        with self.assertRaises(ValueError):
            reading_content(episode)


class ReadingProjectionMetadataTest(unittest.TestCase):
    def test_round_trip_yields_reading_kind(self) -> None:
        metadata = reading_projection_metadata(reading_content(_episode()))
        self.assertEqual(metadata.memory_kind, "READING")
        self.assertEqual(metadata.epistemic_status, READING_EPISTEMIC_STATUS)
        self.assertEqual(metadata.adjacent_episode_refs, ())

    def test_tampered_view_fails_closed(self) -> None:
        content = reading_content(_episode())
        view = json.loads(content)
        view["passage"] = "something she never read"
        tampered = json.dumps(view, separators=(",", ":"), sort_keys=True)
        with self.assertRaises(ValueError):
            reading_projection_metadata(tampered)

    def test_authored_note_content_is_typed(self) -> None:
        note = json.dumps(
            {
                "memory_kind": "MODEL_AUTHORED_PRIVATE_ARTIFACT",
                "epistemic_status": "MODEL_AUTHORED_PRIVATE_ARTIFACT_UNVERIFIED",
                "title": "On substance",
            },
            separators=(",", ":"), sort_keys=True,
        )
        metadata = reading_projection_metadata(note)
        self.assertEqual(metadata.memory_kind, "AUTHORED_NOTE")

    def test_other_content_is_untouched(self) -> None:
        self.assertIsNone(reading_projection_metadata("plain consolidation prose"))
        self.assertIsNone(reading_projection_metadata(json.dumps({"memory_kind": "EPISODIC_EXCHANGE"})))


if __name__ == "__main__":
    unittest.main()

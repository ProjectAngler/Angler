from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import unittest

from angler.cognition.contracts import (
    CognitiveEpisode,
    CognitiveMemoryKind,
    CognitiveMemoryRecord,
    CognitiveRelation,
    EpistemicStatus,
    ProspectiveCommitment,
    RelationType,
)


def ref(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def record(
    kind: CognitiveMemoryKind = CognitiveMemoryKind.EPISODIC,
    status: EpistemicStatus = EpistemicStatus.OBSERVED,
) -> CognitiveMemoryRecord:
    tombstones = (ref("old"),) if status is EpistemicStatus.RETRACTED else ()
    return CognitiveMemoryRecord(
        kind=kind,
        epistemic_status=status,
        content="A public, evidence-linked memory.",
        provenance_refs=tuple(sorted((ref("evidence-a"), ref("evidence-b")))),
        visibility="LEARNER_VISIBLE",
        producer_id="ANG-TEST-PRODUCER-001",
        producer_checkpoint_ref=ref("checkpoint"),
        competence_ref=ref("competence"),
        acquired_ordinal=7,
        world_valid_from=3,
        world_valid_until=11,
        relations=(CognitiveRelation(RelationType.SUPPORTS, ref("claim")),),
        supersedes_refs=(ref("superseded"),),
        tombstone_refs=tombstones,
        confidence_ppm=750_000,
    )


def commitment() -> ProspectiveCommitment:
    return ProspectiveCommitment(
        parent_event_ref=ref("prior-event"),
        task_id="task-001",
        candidate_index=2,
        candidate_trace="proposal c",
        predicted_score=0.625,
        uncertainty=0.125,
        horizon=2,
        competence_state_digest=ref("parent-state"),
    )


def episode() -> CognitiveEpisode:
    support = tuple(sorted((ref("support-a"), ref("support-b"))))
    return CognitiveEpisode(
        task_id="task-001",
        request="Solve the public task.",
        recalled_refs=(ref("recall-b"), *support, ref("recall-a")),
        proposals=("proposal a", "proposal b", "proposal c", "proposal d"),
        selected_index=2,
        commitment=commitment(),
        response="Observable response",
        observations=("executor returned a bounded result",),
        outcome="success",
        feedback_text="Objective evaluator accepted the result.",
        feedback_source_ref=ref("feedback"),
        parent_state_digest=ref("parent-state"),
        child_state_digest=ref("child-state"),
        model_ref=ref("model"),
        encoder_ref=ref("encoder"),
        supporting_evidence_refs=support,
    )


class CognitiveMemoryContractTests(unittest.TestCase):
    def test_all_kinds_and_statuses_round_trip_canonically(self) -> None:
        for kind in CognitiveMemoryKind:
            for status in EpistemicStatus:
                with self.subTest(kind=kind, status=status):
                    value = record(kind, status)
                    encoded = value.canonical_bytes()
                    restored = CognitiveMemoryRecord.from_json(encoded)
                    self.assertEqual(restored, value)
                    self.assertEqual(restored.canonical_bytes(), encoded)
                    self.assertEqual(restored.record_ref, value.record_ref)
                    self.assertRegex(value.record_ref, r"^sha256:[0-9a-f]{64}$")

    def test_record_is_immutable_and_strictly_content_addressed(self) -> None:
        value = record()
        with self.assertRaises(FrozenInstanceError):
            value.content = "changed"  # type: ignore[misc]
        altered = CognitiveMemoryRecord.from_payload(
            {**value.to_payload(), "content": "Different public memory."}
        )
        self.assertNotEqual(altered.record_ref, value.record_ref)
        with self.assertRaises(ValueError):
            CognitiveMemoryRecord.from_json(value.canonical_json() + "\n")

    def test_record_rejects_malformed_boundaries(self) -> None:
        base = record().to_payload()
        cases = (
            {**base, "visibility": "PUBLIC"},
            {**base, "provenance_refs": []},
            {**base, "provenance_refs": list(reversed(sorted((ref("z"), ref("a")))))},
            {**base, "world_valid_from": 12, "world_valid_until": 11},
            {**base, "relations": [{"relation_type": "CHOSES", "target_ref": ref("x")}]},
            {**base, "supersedes_refs": [ref("x"), ref("x")]},
            {**base, "contract": "ANG-CTR-COGNITIVE-MEMORY-001@9.0.0"},
        )
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises((TypeError, ValueError)):
                    CognitiveMemoryRecord.from_payload(payload)

    def test_retraction_and_tombstone_states_are_consistent(self) -> None:
        base = record().to_payload()
        with self.assertRaises(ValueError):
            CognitiveMemoryRecord.from_payload(
                {**base, "epistemic_status": "RETRACTED", "tombstone_refs": []}
            )
        with self.assertRaises(ValueError):
            CognitiveMemoryRecord.from_payload(
                {**base, "epistemic_status": "OBSERVED", "tombstone_refs": [ref("old")]}
            )

    def test_lineage_relations_use_only_dedicated_canonical_fields(self) -> None:
        base = record().to_payload()
        for relation_type in ("SUPERSEDES", "TOMBSTONES"):
            with self.subTest(relation_type=relation_type):
                with self.assertRaisesRegex(ValueError, "dedicated canonical ref fields"):
                    CognitiveMemoryRecord.from_payload(
                        {
                            **base,
                            "relations": [
                                {
                                    "relation_type": relation_type,
                                    "target_ref": ref("lineage-target"),
                                }
                            ],
                        }
                    )


class CognitiveEpisodeContractTests(unittest.TestCase):
    def test_commitment_round_trip_preserves_floats_exactly(self) -> None:
        value = commitment()
        encoded = value.canonical_bytes()
        restored = ProspectiveCommitment.from_json(encoded)
        self.assertEqual(restored, value)
        self.assertEqual(restored.predicted_score.hex(), value.predicted_score.hex())
        self.assertEqual(restored.uncertainty.hex(), value.uncertainty.hex())  # type: ignore[union-attr]
        self.assertEqual(restored.commitment_ref, value.commitment_ref)
        self.assertIn(b'"predicted_score_hex":"0x1.4000000000000p-1"', encoded)

    def test_commitment_candidate_bound_is_owned_by_enclosing_contract(self) -> None:
        value = ProspectiveCommitment(
            parent_event_ref=ref("prior-event"),
            task_id="task-001",
            candidate_index=63,
            candidate_trace="proposal 63",
            predicted_score=0.5,
            uncertainty=None,
            horizon=1,
            competence_state_digest=ref("parent-state"),
        )
        self.assertEqual(
            ProspectiveCommitment.from_json(value.canonical_bytes()), value
        )

    def test_episode_round_trip_and_identity(self) -> None:
        value = episode()
        encoded = value.canonical_bytes()
        restored = CognitiveEpisode.from_json(encoded)
        self.assertEqual(restored, value)
        self.assertEqual(restored.canonical_bytes(), encoded)
        self.assertEqual(restored.episode_ref, value.episode_ref)
        self.assertEqual(restored.commitment.commitment_ref, value.commitment.commitment_ref)

    def test_episode_requires_bounded_distinct_proposals_and_bound_commitment(self) -> None:
        base = episode().to_payload()
        for proposals in (
            ["a"],
            [f"proposal {index}" for index in range(65)],
            ["a", "b", "c", "c"],
        ):
            with self.subTest(proposals=proposals):
                with self.assertRaises(ValueError):
                    CognitiveEpisode.from_payload({**base, "proposals": proposals})
        proposals = [f"proposal {index}" for index in range(7)]
        selected = ProspectiveCommitment(
            parent_event_ref=ref("prior-event"),
            task_id="task-001",
            candidate_index=6,
            candidate_trace=proposals[6],
            predicted_score=0.5,
            uncertainty=None,
            horizon=1,
            competence_state_digest=ref("parent-state"),
        )
        expanded = CognitiveEpisode.from_payload(
            {
                **base,
                "proposals": proposals,
                "selected_index": 6,
                "commitment": selected.to_payload(),
            }
        )
        self.assertEqual(len(expanded.proposals), 7)
        self.assertEqual(expanded.selected_index, 6)
        wrong = ProspectiveCommitment(
            parent_event_ref=ref("prior-event"),
            task_id="task-001",
            candidate_index=1,
            candidate_trace="proposal b",
            predicted_score=0.5,
            uncertainty=None,
            horizon=1,
            competence_state_digest=ref("parent-state"),
        )
        with self.assertRaises(ValueError):
            CognitiveEpisode.from_payload({**base, "commitment": wrong.to_payload()})

        wrong_state = ProspectiveCommitment(
            parent_event_ref=ref("prior-event"),
            task_id="task-001",
            candidate_index=2,
            candidate_trace="proposal c",
            predicted_score=0.5,
            uncertainty=None,
            horizon=1,
            competence_state_digest=ref("not-the-parent-state"),
        )
        with self.assertRaisesRegex(ValueError, "competence state"):
            CognitiveEpisode.from_payload(
                {**base, "commitment": wrong_state.to_payload()}
            )

    def test_episode_rejects_invalid_outcome_visibility_and_refs(self) -> None:
        base = episode().to_payload()
        for change in (
            {"outcome": "unknown"},
            {"visibility": "PUBLIC"},
            {"feedback_source_ref": "feedback"},
            {"supporting_evidence_refs": list(reversed(sorted((ref("z"), ref("a")))))},
        ):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    CognitiveEpisode.from_payload({**base, **change})

    def test_supporting_evidence_must_have_been_recalled(self) -> None:
        value = episode()
        with self.assertRaisesRegex(ValueError, "subset of recalled_refs"):
            CognitiveEpisode(
                task_id=value.task_id,
                request=value.request,
                recalled_refs=value.recalled_refs,
                proposals=value.proposals,
                selected_index=value.selected_index,
                commitment=value.commitment,
                response=value.response,
                observations=value.observations,
                outcome=value.outcome,
                feedback_text=value.feedback_text,
                feedback_source_ref=value.feedback_source_ref,
                parent_state_digest=value.parent_state_digest,
                child_state_digest=value.child_state_digest,
                model_ref=value.model_ref,
                encoder_ref=value.encoder_ref,
                supporting_evidence_refs=(ref("never-recalled"),),
                visibility=value.visibility,
            )

    def test_nonfinite_prediction_and_noncanonical_json_fail_closed(self) -> None:
        common = dict(
            parent_event_ref=None,
            task_id="task-001",
            candidate_index=0,
            candidate_trace="proposal a",
            uncertainty=None,
            horizon=1,
            competence_state_digest=ref("state"),
        )
        for score in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(score=score):
                with self.assertRaises(ValueError):
                    ProspectiveCommitment(predicted_score=score, **common)
        with self.assertRaises(ValueError):
            ProspectiveCommitment.from_json(commitment().canonical_json() + " ")


if __name__ == "__main__":
    unittest.main()

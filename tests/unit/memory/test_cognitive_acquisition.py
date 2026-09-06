from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import unittest

from angler.cognition.contracts import (
    CognitiveEpisode,
    CognitiveMemoryKind,
    CognitiveMemoryRecord,
    EpistemicStatus,
    ProspectiveCommitment,
)
from angler.episodes.canonical import canonical_bytes
from angler.memory.cognitive_acquisition import (
    CognitiveAcquisition,
    CognitiveGraphProjectionV2,
    PROSPECTIVE_DYNAMICS_BATCH_CONTRACT,
    PROSPECTIVE_RESOLUTION_CONTRACT,
)
from tests.unit.cognition.test_prospective_origin import (
    batch as _prospective_batch,
    observed_resolution as _observed_resolution,
)


def _ref(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _episode() -> CognitiveEpisode:
    parent = _ref("parent-state")
    support = (_ref("support"),)
    commitment = ProspectiveCommitment(
        parent_event_ref=None,
        task_id="task-001",
        candidate_index=1,
        candidate_trace="proposal b",
        predicted_score=0.25,
        uncertainty=0.5,
        horizon=1,
        competence_state_digest=parent,
    )
    return CognitiveEpisode(
        task_id="task-001",
        request="Solve the bounded public task.",
        recalled_refs=(_ref("recall"), *support),
        proposals=("proposal a", "proposal b"),
        selected_index=1,
        commitment=commitment,
        response="Observable response",
        observations=("bounded observation",),
        outcome="success",
        feedback_text="Objective evaluator accepted the result.",
        feedback_source_ref=_ref("feedback"),
        parent_state_digest=parent,
        child_state_digest=_ref("child-state"),
        model_ref=_ref("model"),
        encoder_ref=_ref("encoder"),
        supporting_evidence_refs=support,
    )


def _record(
    source_ref: str,
    ordinal: int,
    *,
    kind: CognitiveMemoryKind,
    status: EpistemicStatus,
) -> CognitiveMemoryRecord:
    return CognitiveMemoryRecord(
        kind=kind,
        epistemic_status=status,
        content="A bounded public view of one canonical acquisition.",
        provenance_refs=tuple(sorted((source_ref, _ref("supporting-evidence")))),
        visibility="LEARNER_VISIBLE",
        producer_id="ANG-TEST-ACQUISITION-001",
        producer_checkpoint_ref=_ref("checkpoint"),
        competence_ref=_ref("competence"),
        acquired_ordinal=ordinal,
    )


def _acquisition_chain() -> tuple[
    tuple[object, CognitiveAcquisition],
    tuple[object, CognitiveAcquisition],
    tuple[object, CognitiveAcquisition],
]:
    episode = _episode()
    episode_acquisition = CognitiveAcquisition.from_source(
        episode,
        ordinal=0,
        predecessor_acquisition_ref=None,
        record=_record(
            episode.episode_ref,
            0,
            kind=CognitiveMemoryKind.EPISODIC,
            status=EpistemicStatus.OBSERVED,
        ),
    )
    batch = _prospective_batch(count=2, latent_width=2)
    batch_acquisition = CognitiveAcquisition.from_source(
        batch,
        ordinal=1,
        predecessor_acquisition_ref=episode_acquisition.acquisition_ref,
        record=_record(
            batch.batch_ref,
            1,
            kind=CognitiveMemoryKind.COUNTERFACTUAL,
            status=EpistemicStatus.PROPOSED,
        ),
    )
    resolution = _observed_resolution()[0]
    resolution_acquisition = CognitiveAcquisition.from_source(
        resolution,
        ordinal=2,
        predecessor_acquisition_ref=batch_acquisition.acquisition_ref,
        record=_record(
            resolution.resolution_ref,
            2,
            kind=CognitiveMemoryKind.EPISODIC,
            status=EpistemicStatus.OBSERVED,
        ),
    )
    return (
        (episode, episode_acquisition),
        (batch, batch_acquisition),
        (resolution, resolution_acquisition),
    )


class CognitiveAcquisitionContractTests(unittest.TestCase):
    def test_sources_and_projections_round_trip_byte_exactly(self) -> None:
        for source, acquisition in _acquisition_chain():
            with self.subTest(source_contract=acquisition.source_contract):
                acquisition.assert_source(source)
                encoded = acquisition.canonical_bytes()
                restored = CognitiveAcquisition.from_json(encoded)
                self.assertEqual(restored, acquisition)
                self.assertEqual(restored.canonical_bytes(), encoded)
                self.assertEqual(restored.acquisition_ref, acquisition.acquisition_ref)
                self.assertRegex(acquisition.acquisition_ref, r"^sha256:[0-9a-f]{64}$")

                projection = CognitiveGraphProjectionV2.from_acquisition(acquisition)
                projection.assert_acquisition(acquisition)
                projected = projection.canonical_bytes()
                restored_projection = CognitiveGraphProjectionV2.from_json(projected)
                self.assertEqual(restored_projection, projection)
                self.assertEqual(restored_projection.canonical_bytes(), projected)
                self.assertEqual(
                    restored_projection.projection_ref,
                    projection.projection_ref,
                )

        acquisition = _acquisition_chain()[0][1]
        with self.assertRaises(FrozenInstanceError):
            acquisition.ordinal = 4  # type: ignore[misc]
        self.assertEqual(
            CognitiveAcquisition.__slots__,
            (
                "ordinal",
                "predecessor_acquisition_ref",
                "source_contract",
                "source_ref",
                "record",
            ),
        )

    def test_local_ordinal_predecessor_record_and_provenance_rules(self) -> None:
        _source, acquisition = _acquisition_chain()[1]
        with self.assertRaises(ValueError):
            replace(acquisition, ordinal=0)
        with self.assertRaises(ValueError):
            replace(acquisition, predecessor_acquisition_ref=None)
        with self.assertRaises(ValueError):
            replace(acquisition, ordinal=3)
        with self.assertRaises(ValueError):
            replace(
                acquisition,
                record=_record(
                    _ref("other-source"),
                    1,
                    kind=CognitiveMemoryKind.COUNTERFACTUAL,
                    status=EpistemicStatus.PROPOSED,
                ),
            )

        first = _acquisition_chain()[0][1]
        with self.assertRaises(ValueError):
            replace(first, predecessor_acquisition_ref=_ref("impossible-predecessor"))

    def test_source_contract_owns_kind_and_epistemic_status(self) -> None:
        episode, episode_acquisition = _acquisition_chain()[0]
        batch, batch_acquisition = _acquisition_chain()[1]
        resolution, resolution_acquisition = _acquisition_chain()[2]
        for source, acquisition, wrong_kind, wrong_status in (
            (
                episode,
                episode_acquisition,
                CognitiveMemoryKind.COUNTERFACTUAL,
                EpistemicStatus.PROPOSED,
            ),
            (
                batch,
                batch_acquisition,
                CognitiveMemoryKind.EPISODIC,
                EpistemicStatus.OBSERVED,
            ),
            (
                resolution,
                resolution_acquisition,
                CognitiveMemoryKind.COUNTERFACTUAL,
                EpistemicStatus.PROPOSED,
            ),
        ):
            with self.subTest(source=acquisition.source_contract):
                with self.assertRaises(ValueError):
                    CognitiveAcquisition.from_source(
                        source,
                        ordinal=acquisition.ordinal,
                        predecessor_acquisition_ref=(
                            acquisition.predecessor_acquisition_ref
                        ),
                        record=_record(
                            acquisition.source_ref,
                            acquisition.ordinal,
                            kind=wrong_kind,
                            status=wrong_status,
                        ),
                    )

        with self.assertRaises(TypeError):
            CognitiveAcquisition.from_source(
                object(),
                ordinal=0,
                predecessor_acquisition_ref=None,
                record=episode_acquisition.record,
            )

    def test_source_and_payload_tampering_fail_closed(self) -> None:
        source, acquisition = _acquisition_chain()[1]
        with self.assertRaises(ValueError):
            acquisition.assert_source(
                replace(source, decision_evidence_ref=_ref("different decision"))
            )

        payload = acquisition.to_payload()
        for mutation in (
            {**payload, "extra": True},
            {key: value for key, value in payload.items() if key != "source_ref"},
            {**payload, "contract": "ANG-CTR-COGNITIVE-ACQUISITION-001@9.0.0"},
            {**payload, "source_ref": _ref("forged-source")},
        ):
            with self.subTest(fields=tuple(sorted(mutation))):
                with self.assertRaises((TypeError, ValueError)):
                    CognitiveAcquisition.from_json(canonical_bytes(mutation))
        with self.assertRaises(ValueError):
            CognitiveAcquisition.from_json(acquisition.canonical_bytes() + b" ")

        # The exact source still passes after the negative checks.
        acquisition.assert_source(source)

    def test_projection_rejoin_rejects_acquisition_source_and_record_drift(self) -> None:
        _source, acquisition = _acquisition_chain()[2]
        projection = CognitiveGraphProjectionV2.from_acquisition(acquisition)

        drifted_acquisition = replace(
            acquisition,
            predecessor_acquisition_ref=_ref("different-predecessor"),
        )
        with self.assertRaises(ValueError):
            projection.assert_acquisition(drifted_acquisition)

        drifted_record = replace(acquisition.record, content="Different public view.")
        drifted_projection = replace(projection, record=drifted_record)
        with self.assertRaises(ValueError):
            drifted_projection.assert_acquisition(acquisition)

        payload = projection.to_payload()
        for mutation in (
            {**payload, "source_ref": _ref("wrong-source")},
            {**payload, "contract": "ANG-CTR-COGNITIVE-GRAPH-PROJECTION-001@0.1.0"},
            {**payload, "unexpected": "field"},
        ):
            with self.assertRaises((TypeError, ValueError)):
                CognitiveGraphProjectionV2.from_json(canonical_bytes(mutation))

        forged = CognitiveGraphProjectionV2.from_json(
            canonical_bytes({**payload, "acquisition_ref": _ref("wrong-acquisition")})
        )
        with self.assertRaises(ValueError):
            forged.assert_acquisition(acquisition)


if __name__ == "__main__":
    unittest.main()

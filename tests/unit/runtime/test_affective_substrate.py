from __future__ import annotations

import json
import unittest

from angler.runtime.affective_substrate import (
    EVENT_SUBSTRATE_CONTRACT,
    EVENT_SUBSTRATE_EPISTEMIC_STATUS,
    EventSubstrateObservation,
    project_canonical_episode,
)


def _ref(character: str) -> str:
    return "sha256:" + character * 64


def _episode() -> dict[str, object]:
    context = {
        "autonomous_initiative": {
            "formation": {
                "proposal": {
                    "commitment": {
                        "contract": "jenny.autonomous-commitment.v1",
                        "status": "ACTIVE",
                        "statement": "private commitment prose",
                        "rationale": "private rationale prose",
                        "evidence_keys": ["state-focus"],
                        "uncertainty": 0.35,
                    }
                }
            }
        },
        "intent_proposal": {
            "uncertainty": 0.4,
            "alternatives": [{"private": "alternative prose"}],
        },
        "memories": [
            {
                "record_ref": _ref("b"),
                "content": "private recalled prose",
                "semantic_distance": 0.125,
            }
        ],
        "phase_timings_ms": {
            "adaptive_human_router": 125.5,
            "choose_total": 140.25,
        },
    }
    return {
        "contract": "ANG-CTR-PERSISTENT-AUTONOMY-STATE-001@0.1.0",
        "genesis_ref": _ref("1"),
        "qualification_ref": _ref("2"),
        "observation": {
            "trigger_ref": "scheduler:test",
            "source": "SCHEDULER",
            "content": "private observation prose",
        },
        "choice": {
            "selected_affordance_id": "internal.authored-artifact",
            "rankings": [
                {"affordance_id": "internal.authored-artifact", "score": 0.75},
                {"affordance_id": "control.wait", "score": 0.25},
            ],
            "prediction": "private predicted prose",
            "uncertainty": 0.3,
            "wake_after_seconds": None,
            "action_payload": "private action prose",
            "context_json": json.dumps(
                context, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ),
        },
        "receipt": {
            "status": "COMPLETED",
            "output": "private output prose",
            "consequence": [["information_gain", 0.5]],
            "observable_consequence": {
                "version": "jenny.observable-consequence.v1",
                "request_ref": _ref("3"),
                "source_kind": "WORLD",
                "source_ref": _ref("4"),
                "observation_json": '{"private":"observed prose"}',
                "artifact_refs": [],
                "evidence_refs": [],
            },
        },
        "temporal": {
            "contract": "ANG-CTR-TEMPORAL-V2-001@0.2.0",
            "event_time_utc": "2026-09-04T12:00:00Z",
            "acquired_time_utc": "2026-09-04T12:00:00Z",
            "recorded_time_utc": "2026-09-04T12:00:00Z",
            "verified_time_utc": None,
            "valid_from_utc": None,
            "valid_until_utc": None,
            "timezone": "America/New_York",
            "moving_origin_ordinal": 9,
            "precision_ms": 1.0,
            "uncertainty_ms": 1.0,
            "clock_jump_detected": False,
            "source": "jenny.persistent-autonomy.v1",
        },
        "parent_state_ref": _ref("5"),
        "child_state_ref": _ref("6"),
        "reflection": "private reflection prose",
        "consolidation_proposal": "private consolidation prose",
        "update_ref": _ref("7"),
        "previous_event_ref": _ref("8"),
        "event_ref": _ref("9"),
    }


class AffectiveSubstrateTests(unittest.TestCase):
    def test_projects_provenance_without_private_semantic_text_or_emotion(self) -> None:
        sample = project_canonical_episode(
            episode_ref=_ref("a"), episode_payload=_episode()
        )
        payload = json.loads(sample.payload_json)

        self.assertEqual(payload["contract"], EVENT_SUBSTRATE_CONTRACT)
        self.assertEqual(
            payload["epistemic_status"], EVENT_SUBSTRATE_EPISTEMIC_STATUS
        )
        self.assertEqual(payload["moving_origin_ordinal"], 9)
        self.assertEqual(payload["commitment"]["status"], "ACTIVE")
        self.assertEqual(
            payload["commitment"]["value_kind"],
            "MODEL_AUTHORED_COMMITMENT_ESTIMATE",
        )
        self.assertEqual(
            payload["retrieval"]["observations"][0]["semantic_distance"],
            0.125,
        )
        self.assertEqual(
            payload["closure"]["declared_consequence"],
            [["information_gain", 0.5]],
        )
        self.assertEqual(
            payload["runtime"]["phase_timings_ms"]["choose_total"], 140.25
        )
        self.assertEqual(
            payload["temporal"]["value_kind"], "TRUSTED_CLOCK_MEASUREMENT"
        )
        for forbidden in (
            "private",
            "emotion",
            "valence",
            "arousal",
            "sentiment",
            "frustration",
        ):
            self.assertNotIn(forbidden, sample.payload_json.casefold())

    def test_mixed_rankings_are_retained_without_collapsing_to_one_value(self) -> None:
        sample = project_canonical_episode(
            episode_ref=_ref("a"), episode_payload=_episode()
        )
        prediction = json.loads(sample.payload_json)["prediction"]
        self.assertEqual(len(prediction["rankings"]), 2)
        self.assertEqual(
            [item["declared_score"] for item in prediction["rankings"]],
            [0.75, 0.25],
        )
        self.assertEqual(prediction["alternative_count"], 1)

    def test_sample_ref_detects_payload_tampering(self) -> None:
        sample = project_canonical_episode(
            episode_ref=_ref("a"), episode_payload=_episode()
        )
        payload = json.loads(sample.payload_json)
        payload["moving_origin_ordinal"] = 10
        with self.assertRaisesRegex(ValueError, "identity differs"):
            EventSubstrateObservation(
                sample.episode_ref,
                sample.event_ref,
                sample.ordinal,
                sample.sample_ref,
                json.dumps(payload, separators=(",", ":"), sort_keys=True),
            )

    def test_nonfinite_declared_measurement_is_rejected(self) -> None:
        episode = _episode()
        episode["receipt"]["consequence"][0][1] = float("nan")  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "finite"):
            project_canonical_episode(
                episode_ref=_ref("a"), episode_payload=episode
            )


if __name__ == "__main__":
    unittest.main()

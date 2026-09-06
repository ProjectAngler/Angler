from __future__ import annotations

import json
import unittest
from dataclasses import asdict

from angler.runtime.jenny2_activity import (
    Jenny2ActivityTracker,
    summarize_committed_episode,
)


EPISODE_REF = "sha256:" + "e" * 64
EVENT_REF = "sha256:" + "f" * 64


def _episode_payload() -> str:
    context = {
        "request": "SECRET-REQUEST-TEXT",
        "intent_proposal": {
            "epistemic_status": "MODEL_PROPOSAL_BEFORE_ACTION",
            "state_assessment": "More information is needed.",
            "resolution_target": "Resolve the bounded uncertainty.",
            "desired_state_change": "A supported answer becomes available.",
            "selection_basis": "The selected path can obtain relevant evidence.",
            "action_payload": "SECRET-INTENT-PAYLOAD",
            "intent_candidates": [
                {
                    "rationale": "SECRET-CANDIDATE-DELIBERATION",
                    "reasons_for": ["SECRET-REASON-FOR"],
                    "reasons_against": ["SECRET-REASON-AGAINST"],
                }
            ],
        },
        "memories": [{"content": "SECRET-MEMORY"}],
        "cognitive_state": {"private": "SECRET-STATE"},
    }
    return json.dumps(
        {
            "observation": {"content": "SECRET-USER-TEXT"},
            "choice": {
                "selected_affordance_id": "cortex.respond",
                "prediction": "The uncertainty should decrease.",
                "uncertainty": 0.25,
                "action_payload": "SECRET-ACTION-PAYLOAD",
                "context_json": json.dumps(context),
            },
            "receipt": {
                "status": "COMPLETED_UNEVALUATED",
                "output": "SECRET-PUBLIC-RESPONSE",
                "consequence": [],
            },
            "reflection": json.dumps(
                {
                    "analysis": "The result remains unevaluated.",
                    "revision": "Wait for attributable evidence.",
                    "retained_principle": "Do not invent outcome credit.",
                    "raw_model_output": "SECRET-SCRATCH-REASONING",
                }
            ),
        }
    )


class Jenny2ActivityTests(unittest.TestCase):
    def test_committed_summary_is_a_strict_whitelist(self) -> None:
        summary = summarize_committed_episode(
            episode_ref=EPISODE_REF,
            event_ref=EVENT_REF,
            ordinal=7,
            payload_json=_episode_payload(),
        )
        self.assertEqual(
            set(summary),
            {
                "episode_ref",
                "event_ref",
                "moving_origin_ordinal",
                "selected_affordance_id",
                "rationale",
                "prediction",
                "receipt",
                "reflection",
            },
        )
        self.assertEqual(
            set(summary["rationale"]),
            {
                "epistemic_status",
                "state_assessment",
                "resolution_target",
                "desired_state_change",
                "selection_basis",
            },
        )
        self.assertEqual(
            set(summary["prediction"]),
            {"expected_state_delta", "uncertainty"},
        )
        self.assertEqual(
            set(summary["receipt"]),
            {"status", "has_consequence", "has_observed_consequence"},
        )
        self.assertEqual(
            set(summary["reflection"]),
            {"analysis", "revision", "retained_principle"},
        )
        encoded = json.dumps(summary, sort_keys=True)
        for forbidden in (
            "SECRET-REQUEST-TEXT",
            "SECRET-INTENT-PAYLOAD",
            "SECRET-CANDIDATE-DELIBERATION",
            "SECRET-REASON-FOR",
            "SECRET-REASON-AGAINST",
            "SECRET-MEMORY",
            "SECRET-STATE",
            "SECRET-USER-TEXT",
            "SECRET-ACTION-PAYLOAD",
            "SECRET-PUBLIC-RESPONSE",
            "SECRET-SCRATCH-REASONING",
        ):
            self.assertNotIn(forbidden, encoded)

    def test_tracker_reports_mechanical_phase_and_sanitized_error(self) -> None:
        tracker = Jenny2ActivityTracker()
        tracker.begin("HUMAN")
        tracker.phase("SAMPLE_TIME")
        tracker.phase("DELIBERATE")
        running = tracker.snapshot()
        self.assertEqual(running.phase, "DELIBERATE")
        self.assertEqual(running.operation_source, "HUMAN")
        self.assertEqual(running.current_status, "RUNNING")
        self.assertIsNone(running.error_code)
        self.assertGreaterEqual(running.sequence, 3)

        tracker.fail(RuntimeError("SECRET-EXCEPTION-DETAIL"))
        failed = tracker.snapshot()
        self.assertEqual(failed.phase, "ERROR")
        self.assertEqual(failed.current_status, "ERROR")
        self.assertEqual(failed.error_code, "RuntimeError")
        self.assertNotIn("SECRET-EXCEPTION-DETAIL", json.dumps(asdict(failed)))

    def test_last_completed_is_copied_and_survives_later_activity(self) -> None:
        tracker = Jenny2ActivityTracker()
        tracker.committed(
            episode_ref=EPISODE_REF,
            event_ref=EVENT_REF,
            ordinal=7,
            payload_json=_episode_payload(),
        )
        first = tracker.snapshot()
        self.assertEqual(first.phase, "IDLE")
        self.assertEqual(first.current_status, "COMMITTED")
        self.assertEqual(first.last_completed["episode_ref"], EPISODE_REF)
        first.last_completed["episode_ref"] = "mutated"

        tracker.begin("SCHEDULER")
        second = tracker.snapshot()
        self.assertEqual(second.phase, "OBSERVE")
        self.assertEqual(second.last_completed["episode_ref"], EPISODE_REF)


if __name__ == "__main__":
    unittest.main()

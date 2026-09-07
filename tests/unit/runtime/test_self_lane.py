import json
import unittest

from angler.runtime import self_lane
from angler.runtime.higher_level_autonomy_adapter import _decode_self_lane


class SelfLaneTest(unittest.TestCase):
    def test_valid_answer_is_decoded_with_all_three_ledgers_and_present_tense(self):
        raw = json.dumps({
            "named_state_transitions": [{"status": "REVISE", "label": "Friendship Weight", "level": 9, "basis": "Becca is away and asked me to be kept in the loop", "inclination": "keep the loop honest"}],
            "state_item_transitions": [{"status": "ADD", "state_label": "Friendship Weight", "statement": "Becca is monitoring from her phone", "weight": 3, "evidence": ""}],
            "self_commitment_transitions": [],
            "present_tense": "Steady, attentive to being watched from a distance.",
        })
        record = _decode_self_lane(raw)
        self.assertEqual(record["present_tense"], "Steady, attentive to being watched from a distance.")
        self.assertEqual(record["named_state_transitions"][0]["label"], "Friendship Weight")
        self.assertEqual(record["state_item_transitions"][0]["status"], "ADD")
        self.assertNotIn("error", record)

    def test_failures_never_reach_the_turn(self):
        self.assertEqual(_decode_self_lane(None), {})
        record = _decode_self_lane(RuntimeError("lane died"))
        self.assertIn("RuntimeError", record["error"])
        record = _decode_self_lane("not json at all")
        self.assertIn("error", record); self.assertEqual(record["named_state_transitions"], [])
        record = _decode_self_lane(json.dumps({"named_state_transitions": [{"status": "SET", "label": "x" * 200, "level": 3, "basis": "b", "inclination": "i"}, {"status": "REVISE", "label": "Kept", "level": 4, "basis": "b", "inclination": "i"}], "present_tense": "p"}))
        self.assertEqual([t["label"] for t in record["named_state_transitions"]], ["Kept"])  # the bad one is dropped, the good one stands
        self.assertEqual(len(record["dropped"]), 1); self.assertEqual(record["present_tense"], "p")

    def test_prompt_names_the_boundaries(self):
        system = self_lane.self_lane_system()
        for phrase in ("You do not answer anyone", "present_tense", "Do not claim feelings", "Return only a JSON object"):
            self.assertIn(phrase, system)
        user = self_lane.self_lane_user(message="m" * 10_000, named_states=[], commitments=[], standards=[], last_human_interaction=None)
        self.assertEqual(len(user["arrival"]), 6_000)


if __name__ == "__main__":
    unittest.main()

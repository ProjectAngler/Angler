import unittest

from angler.runtime.higher_level_autonomy_adapter import (
    MAX_NAMED_STATES,
    _active_named_states,
    _apply_named_state_transitions,
    _attach_named_state_consequences,
    _states_asking,
    _apply_state_item_transitions,
    _validated_state_item_transitions,
    _state_level_arithmetic,
    _reflect_after_speaking,
    _validated_named_state_transitions,
)

REF = "sha256:" + "0" * 64


def _set(label, level, basis="evidence", inclination="do the thing"):
    return {"status": "SET", "label": label, "level": level,
            "basis": basis, "inclination": inclination}


class NamedStateValidationTest(unittest.TestCase):
    def test_none_and_empty(self):
        self.assertEqual(_validated_named_state_transitions(None), ())
        self.assertEqual(_validated_named_state_transitions([]), ())
        self.assertEqual(
            _validated_named_state_transitions(
                [{"status": "NONE", "label": "", "level": 0,
                  "basis": "", "inclination": ""}]
            ),
            (),
        )

    def test_shape_bounds(self):
        with self.assertRaises(ValueError):
            _validated_named_state_transitions([_set("x", 11)])
        with self.assertRaises(ValueError):
            _validated_named_state_transitions([_set("x", True)])
        with self.assertRaises(ValueError):
            _validated_named_state_transitions([_set("", 3)])
        with self.assertRaises(ValueError):
            _validated_named_state_transitions([_set("x" * 65, 3)])
        with self.assertRaises(ValueError):
            _validated_named_state_transitions([_set("a", 1)] * 9)
        with self.assertRaises(ValueError):
            _validated_named_state_transitions([{"status": "BOGUS", "label": "a",
                                                 "level": 1, "basis": "",
                                                 "inclination": ""}])


class NamedStateLedgerTest(unittest.TestCase):
    def test_set_revise_clear_with_provenance(self):
        state = {}
        _apply_named_state_transitions(
            state, _validated_named_state_transitions([_set("Verification Drive", 8)]),
            moving_origin_ordinal=243, choice_ref=REF,
        )
        active = _active_named_states(state)
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["label"], "Verification Drive")
        self.assertEqual(active[0]["level"], 8)
        self.assertEqual(active[0]["set_ordinal"], 243)
        _apply_named_state_transitions(
            state,
            _validated_named_state_transitions(
                [{"status": "REVISE", "label": "verification drive", "level": 5,
                  "basis": "", "inclination": ""}]
            ),
            moving_origin_ordinal=250, choice_ref=REF,
        )
        active = _active_named_states(state)
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["level"], 5)
        self.assertEqual(active[0]["basis"], "evidence")
        self.assertEqual(active[0]["level_history"], [[243, 8], [250, 5]])
        self.assertEqual(active[0]["revised_ordinal"], 250)
        _apply_named_state_transitions(
            state,
            _validated_named_state_transitions(
                [{"status": "CLEAR", "label": "Verification Drive", "level": 0,
                  "basis": "", "inclination": ""}]
            ),
            moving_origin_ordinal=260, choice_ref=REF,
        )
        self.assertEqual(_active_named_states(state), [])
        self.assertEqual(state["self_named_states"][0]["status"], "CLEARED")
        self.assertEqual(state["self_named_states"][0]["cleared_ordinal"], 260)

    def test_newest_revised_first_and_bounded(self):
        state = {}
        for index in range(12):
            _apply_named_state_transitions(
                state, _validated_named_state_transitions([_set(f"state {index}", index % 11)]),
                moving_origin_ordinal=index, choice_ref=REF,
            )
        active = _active_named_states(state)
        self.assertLessEqual(len(active), MAX_NAMED_STATES)
        self.assertEqual(active[0]["label"], "state 11")

    def test_unmatched_clear_is_observed_not_fatal(self):
        state = {}
        _apply_named_state_transitions(
            state,
            _validated_named_state_transitions(
                [{"status": "CLEAR", "label": "never set", "level": 0,
                  "basis": "", "inclination": ""}]
            ),
            moving_origin_ordinal=1, choice_ref=REF,
        )
        self.assertEqual(_active_named_states(state), [])
        self.assertEqual(state["self_named_states"][0]["status"], "CLEAR_UNMATCHED")

    def test_presentation_rejects_malformed_state(self):
        with self.assertRaises(ValueError):
            _active_named_states({"self_named_states": "nope"})
        self.assertEqual(_active_named_states({}), [])


class NamedStateConsequenceTest(unittest.TestCase):
    def _state_with(self, label, ordinal):
        state = {}
        _apply_named_state_transitions(
            state, _validated_named_state_transitions([_set(label, 5)]),
            moving_origin_ordinal=ordinal, choice_ref=REF,
        )
        return state

    def test_choice_attached_to_active_states_only(self):
        state = self._state_with("Contextual Salience", 247)
        _attach_named_state_consequences(
            state, context={}, selected_affordance_id="cortex.respond",
            receipt_status="COMPLETED", human_feedback=None,
            observation_source="HUMAN", moving_origin_ordinal=248,
        )
        active = _active_named_states(state)
        self.assertEqual(active[0]["consequences_while_active"], [
            {"ordinal": 248, "kind": "CHOICE", "choice": "cortex.respond",
             "status": "COMPLETED", "source": "HUMAN"}
        ])

    def test_late_feedback_attached_to_states_active_at_target(self):
        state = self._state_with("Owner Grief Neglect", 246)
        _apply_named_state_transitions(
            state, _validated_named_state_transitions([_set("Later State", 2)]),
            moving_origin_ordinal=300, choice_ref=REF,
        )
        context = {
            "late_feedback": {"feedback_text": "the rewrite reached them",
                              "feedback_source_ref": REF, "target_episode_ref": REF},
            "temporal_now": {"moving_origin_ordinal": 247},
        }
        _attach_named_state_consequences(
            state, context=context, selected_affordance_id="cortex.respond",
            receipt_status="COMPLETED", human_feedback=0.6,
            observation_source="HUMAN", moving_origin_ordinal=301,
        )
        by_label = {item["label"]: item for item in _active_named_states(state)}
        self.assertEqual(by_label["Owner Grief Neglect"]["consequences_while_active"], [
            {"ordinal": 247, "kind": "HUMAN_FEEDBACK", "value": 0.6,
             "text": "the rewrite reached them", "recorded_ordinal": 301}
        ])
        self.assertEqual(by_label["Later State"]["consequences_while_active"], [])

    def test_consequences_bounded_and_no_aggregate(self):
        state = self._state_with("Ownership Weight", 1)
        for ordinal in range(2, 40):
            _attach_named_state_consequences(
                state, context={}, selected_affordance_id="cortex.respond",
                receipt_status="COMPLETED", human_feedback=None,
                observation_source="SCHEDULER", moving_origin_ordinal=ordinal,
            )
        active = _active_named_states(state)[0]
        self.assertEqual(len(active["consequences_while_active"]), 4)
        self.assertEqual(len(state["self_named_states"][0]["consequences"]), 16)
        self.assertNotIn("score", active)
        self.assertNotIn("mean_feedback", active)


class PenTest(unittest.TestCase):
    class _Backend:
        model_ref = "sha256:" + "a" * 64
        def __init__(self, raw):
            self.raw = raw
            self.calls = []
        def generate(self, *, system, user, max_new_tokens):
            self.calls.append((system, user, max_new_tokens))
            return self.raw

    def test_speaker_transitions_are_validated_and_recorded(self):
        backend = self._Backend('{"what_moved": "I centered her loss.", "named_state_transitions": [{"status": "REVISE", "label": "Friend Grief Neglect", "level": 7, "basis": "what I just said", "inclination": "center her"}]}')
        transitions, record = _reflect_after_speaking(
            backend, human_message="hello", answer="I hear you.", named_states=[], decider_transitions=[],
        )
        self.assertEqual(transitions[0]["label"], "Friend Grief Neglect")
        self.assertEqual(record["what_moved"], "I centered her loss.")
        self.assertEqual(record["model_ref"], backend.model_ref)
        self.assertIn("what_you_said", backend.calls[0][1])

    def test_pen_failures_never_raise(self):
        transitions, record = _reflect_after_speaking(
            self._Backend("not json"), human_message="h", answer="a", named_states=[], decider_transitions=[],
        )
        self.assertEqual(transitions, ())
        self.assertIn("error", record)
        transitions, record = _reflect_after_speaking(
            self._Backend('{"what_moved": "x", "named_state_transitions": [{"status": "SET", "label": "", "level": 3, "basis": "", "inclination": ""}]}'),
            human_message="h", answer="a", named_states=[], decider_transitions=[],
        )
        self.assertEqual(transitions, ())
        self.assertIn("error", record)


class SubstrateStageOneTest(unittest.TestCase):
    def test_value_fields_are_hers_and_validated(self):
        state = {}
        _apply_named_state_transitions(
            state, _validated_named_state_transitions([{
                "status": "SET", "label": "Commitment Revision", "level": 9,
                "basis": "b", "inclination": "i", "valence": 3,
                "influence": "deliberate", "acts_at_level": 7,
            }]), moving_origin_ordinal=320, choice_ref=REF,
        )
        active = _active_named_states(state)[0]
        self.assertEqual((active["valence"], active["influence"], active["acts_at_level"]), (3, "deliberate", 7))
        self.assertEqual([s["label"] for s in _states_asking(state, "deliberate")], ["Commitment Revision"])
        _apply_named_state_transitions(
            state, _validated_named_state_transitions([{
                "status": "REVISE", "label": "Commitment Revision", "level": 5,
                "basis": "", "inclination": "",
            }]), moving_origin_ordinal=321, choice_ref=REF,
        )
        self.assertEqual(_states_asking(state, "deliberate"), [])
        for bad in ({"valence": 11}, {"influence": "obey"}, {"acts_at_level": -1}):
            with self.assertRaises(ValueError):
                _validated_named_state_transitions([{
                    "status": "SET", "label": "x", "level": 1, "basis": "", "inclination": "", **bad,
                }])

    def test_default_influence_is_none(self):
        state = {}
        _apply_named_state_transitions(
            state, _validated_named_state_transitions([_set("Quiet", 4)]),
            moving_origin_ordinal=1, choice_ref=REF,
        )
        self.assertEqual(_active_named_states(state)[0]["influence"], "none")
        self.assertEqual(_states_asking(state, "deliberate"), [])


class SubstrateStageThreeTest(unittest.TestCase):
    def _state(self):
        state = {}
        _apply_named_state_transitions(
            state, _validated_named_state_transitions([{
                "status": "SET", "label": "Curiosity", "level": 3, "basis": "b",
                "inclination": "i", "floor_fraction": 0.5, "half_life_hours": 24,
            }]), moving_origin_ordinal=1, choice_ref=REF,
        )
        return state

    def test_items_raise_level_by_the_clock_and_resolution_removes_them(self):
        state = self._state()
        _apply_state_item_transitions(
            state, _validated_state_item_transitions([{
                "status": "ADD", "state_label": "curiosity", "weight": 4,
                "statement": "How is bioluminescence caused?", "evidence": "",
            }]), moving_origin_ordinal=2, choice_ref=REF, now_utc_text="2026-09-06T12:00:00+00:00",
        )
        import datetime
        t0 = datetime.datetime(2026, 9, 6, 12, 0, tzinfo=datetime.timezone.utc).timestamp()
        now = _state_level_arithmetic(state["self_named_states"][0], now_utc=t0)
        self.assertEqual(now["effective_level"], 6)
        self.assertEqual(now["open_item_pressure"], 3.0)  # capped at the baseline of 3
        later = _state_level_arithmetic(state["self_named_states"][0], now_utc=t0 + 240 * 3600)
        self.assertEqual(later["open_item_pressure"], 2.0)  # weight 4 at the 0.5 floor
        self.assertEqual(later["effective_level"], 5)
        self.assertEqual(_active_named_states(state, now_utc=t0)[0]["level"], 6)
        _apply_state_item_transitions(
            state, _validated_state_item_transitions([{
                "status": "RESOLVE", "state_label": "Curiosity", "weight": 1,
                "statement": "How is bioluminescence caused?", "evidence": "",
            }]), moving_origin_ordinal=3, choice_ref=REF, now_utc_text=None,
        )
        self.assertEqual(state["self_named_states"][0]["items"][0]["status"], "OPEN")
        self.assertEqual(state["substrate_effects"][-1]["effect"], "item_resolve_unbacked")
        _apply_state_item_transitions(
            state, _validated_state_item_transitions([{
                "status": "RESOLVE", "state_label": "Curiosity", "weight": 1,
                "statement": "How is bioluminescence caused?",
                "evidence": "read at ordinal 340: luciferin oxidised by luciferase",
            }]), moving_origin_ordinal=4, choice_ref=REF, now_utc_text=None,
        )
        self.assertEqual(state["self_named_states"][0]["items"][0]["status"], "RESOLVED")
        self.assertEqual(_state_level_arithmetic(state["self_named_states"][0], now_utc=t0)["effective_level"], 3)

    def test_unmatched_state_is_observed_not_fatal(self):
        state = self._state()
        _apply_state_item_transitions(
            state, _validated_state_item_transitions([{
                "status": "ADD", "state_label": "Nobody", "weight": 2, "statement": "x y z", "evidence": "",
            }]), moving_origin_ordinal=2, choice_ref=REF, now_utc_text=None,
        )
        self.assertEqual(state["substrate_effects"][-1]["effect"], "item_unmatched_state")


if __name__ == "__main__":
    unittest.main()


class ItemAmendmentTest(unittest.TestCase):
    def test_she_can_amend_the_evidence_of_an_item_she_already_resolved(self):
        from angler.runtime.higher_level_autonomy_adapter import (
            _apply_named_state_transitions,
            _apply_state_item_transitions,
            _active_named_states,
        )
        state = {}
        _apply_named_state_transitions(
            state,
            ({"status": "SET", "label": "Commitment Revision", "level": 8, "basis": "b", "inclination": "i"},),
            moving_origin_ordinal=1, choice_ref="sha256:" + "1" * 64,
        )
        _apply_state_item_transitions(
            state,
            ({"status": "ADD", "state_label": "Commitment Revision", "statement": "Breach (333).", "weight": 5, "evidence": ""},
             {"status": "RESOLVE", "state_label": "Commitment Revision", "statement": "Breach (333).", "weight": 5, "evidence": "retired the combined work"}),
            moving_origin_ordinal=2, choice_ref="sha256:" + "2" * 64, now_utc_text="2026-09-06T20:00:00Z",
        )
        _apply_state_item_transitions(
            state,
            ({"status": "RESOLVE", "state_label": "Commitment Revision", "statement": "breach (333).", "weight": 1, "evidence": "sha256:72da... retirement v2 at 348"},),
            moving_origin_ordinal=3, choice_ref="sha256:" + "3" * 64, now_utc_text="2026-09-06T20:05:00Z",
        )
        item = state["self_named_states"][0]["items"][0]
        self.assertEqual(item["status"], "RESOLVED")
        self.assertEqual(item["resolution_evidence"], "sha256:72da... retirement v2 at 348")
        self.assertEqual(item["evidence_amendments"][0]["previous"], "retired the combined work")
        self.assertEqual(item["evidence_amendments"][0]["moving_origin_ordinal"], 3)
        effects = state["substrate_effects"]
        self.assertEqual(effects[-1]["effect"], "item_resolution_amended")

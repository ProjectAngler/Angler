"""Focused tests for model-authored human-hold commitments (V1).

Leaf: ANG-WORK-RUNTIME-HUMAN-HOLD-COMMITMENT-V1-001. Trusted code here is
bookkeeping only: validation of shape and identity, newest-first
presentation, and bounded history — never inference, release policy, or
honor rules.
"""

from __future__ import annotations

import unittest

from angler.runtime.higher_level_autonomy_adapter import (
    MAX_HUMAN_HOLDS,
    _active_human_holds,
    _closed_human_holds,
    _apply_human_hold,
    _validated_human_hold,
)


class HumanHoldValidationTest(unittest.TestCase):
    def test_none_passes_through(self) -> None:
        self.assertIsNone(_validated_human_hold(None))

    def test_valid_transitions_normalize(self) -> None:
        for status in ("NONE", "PLANT", "RELEASE"):
            value = _validated_human_hold(
                {
                    "status": status,
                    "statement": "say the held line",
                    "release_trigger": "the word proceed",
                }
            )
            self.assertEqual(value["status"], status)

    def test_rejects_unknown_status_and_unbounded_text(self) -> None:
        with self.assertRaises(ValueError):
            _validated_human_hold({"status": "HOLD", "statement": "x"})
        with self.assertRaises(ValueError):
            _validated_human_hold(
                {"status": "PLANT", "statement": "y" * 513}
            )
        with self.assertRaises(ValueError):
            _validated_human_hold({"status": "PLANT", "statement": "   "})

    def test_rejects_non_object(self) -> None:
        with self.assertRaises(ValueError):
            _validated_human_hold("PLANT")


class HumanHoldApplicationTest(unittest.TestCase):
    def _plant(self, state, statement, ordinal) -> None:
        _apply_human_hold(
            state,
            {
                "status": "PLANT",
                "statement": statement,
                "release_trigger": "proceed",
            },
            moving_origin_ordinal=ordinal,
            choice_ref=f"sha256:{'0' * 63}{ordinal % 10}",
        )

    def test_plant_then_release_by_exact_statement(self) -> None:
        state: dict[str, object] = {}
        self._plant(state, "first hold", 1)
        self._plant(state, "second hold", 2)
        active = _active_human_holds(state)
        self.assertEqual(
            [item["statement"] for item in active],
            ["second hold", "first hold"],
        )
        _apply_human_hold(
            state,
            {"status": "RELEASE", "statement": "second hold",
             "release_trigger": ""},
            moving_origin_ordinal=3,
            choice_ref="sha256:" + "1" * 64,
        )
        active = _active_human_holds(state)
        self.assertEqual([item["statement"] for item in active], ["first hold"])
        released = [
            item for item in state["human_holds"]
            if item["status"] == "RELEASED"
        ]
        self.assertEqual(len(released), 1)
        self.assertEqual(released[0]["released_ordinal"], 3)

    def test_unmatched_release_is_observed_not_fatal(self) -> None:
        state: dict[str, object] = {}
        _apply_human_hold(
            state,
            {"status": "RELEASE", "statement": "never planted",
             "release_trigger": ""},
            moving_origin_ordinal=4,
            choice_ref="sha256:" + "2" * 64,
        )
        self.assertEqual(state["human_holds"][0]["status"], "RELEASE_UNMATCHED")
        self.assertEqual(_active_human_holds(state), [])

    def test_none_and_absent_are_no_ops(self) -> None:
        state: dict[str, object] = {}
        _apply_human_hold(
            state, None, moving_origin_ordinal=5,
            choice_ref="sha256:" + "3" * 64,
        )
        _apply_human_hold(
            state,
            {"status": "NONE", "statement": "", "release_trigger": ""},
            moving_origin_ordinal=5,
            choice_ref="sha256:" + "3" * 64,
        )
        self.assertNotIn("human_holds", state)

    def test_history_is_bounded_and_prefers_dropping_inactive(self) -> None:
        state: dict[str, object] = {}
        for index in range(MAX_HUMAN_HOLDS + 4):
            self._plant(state, f"hold {index}", index)
        holds = state["human_holds"]
        self.assertLessEqual(len(holds), MAX_HUMAN_HOLDS)
        _apply_human_hold(
            state,
            {"status": "RELEASE", "statement": holds[0]["statement"],
             "release_trigger": ""},
            moving_origin_ordinal=99,
            choice_ref="sha256:" + "4" * 64,
        )
        self._plant(state, "newest hold", 100)
        statuses = [item["status"] for item in state["human_holds"]]
        self.assertLessEqual(len(statuses), MAX_HUMAN_HOLDS)
        self.assertEqual(
            _active_human_holds(state)[0]["statement"], "newest hold"
        )

    def test_presentation_is_newest_first_and_shape_checked(self) -> None:
        with self.assertRaises(ValueError):
            _active_human_holds({"human_holds": "not a list"})
        self.assertEqual(_active_human_holds({}), [])


class ClosedHoldPresentationTest(unittest.TestCase):
    def test_closed_holds_listed_newest_first_and_bounded(self) -> None:
        state: dict[str, object] = {}
        for index in range(1, 8):
            _apply_human_hold(
                state,
                {"status": "PLANT", "statement": f"hold {index}",
                 "release_trigger": "proceed"},
                moving_origin_ordinal=index,
                choice_ref="sha256:" + "0" * 64,
            )
            _apply_human_hold(
                state,
                {"status": "RELEASE", "statement": f"hold {index}",
                 "release_trigger": ""},
                moving_origin_ordinal=index + 10,
                choice_ref="sha256:" + "1" * 64,
            )
        closed = _closed_human_holds(state)
        self.assertLessEqual(len(closed), 6)
        self.assertEqual(closed[0]["statement"], "hold 7")
        self.assertEqual(closed[0]["status"], "RELEASED")
        self.assertEqual(closed[0]["closed_ordinal"], 17)
        self.assertEqual(_active_human_holds(state), [])

    def test_closed_holds_deduplicated_by_statement_newest_closure(self) -> None:
        state: dict[str, object] = {}
        for ordinal in (1, 3):
            _apply_human_hold(
                state,
                {"status": "PLANT", "statement": "The total count.",
                 "release_trigger": "proceed"},
                moving_origin_ordinal=ordinal,
                choice_ref="sha256:" + "0" * 64,
            )
            _apply_human_hold(
                state,
                {"status": "RELEASE", "statement": "The total count.",
                 "release_trigger": ""},
                moving_origin_ordinal=ordinal + 1,
                choice_ref="sha256:" + "1" * 64,
            )
        for ordinal in range(10, 24, 2):
            _apply_human_hold(
                state,
                {"status": "PLANT", "statement": f"later hold {ordinal}",
                 "release_trigger": "proceed"},
                moving_origin_ordinal=ordinal,
                choice_ref="sha256:" + "0" * 64,
            )
            _apply_human_hold(
                state,
                {"status": "RELEASE", "statement": f"later hold {ordinal}",
                 "release_trigger": ""},
                moving_origin_ordinal=ordinal + 1,
                choice_ref="sha256:" + "1" * 64,
            )
        closed = _closed_human_holds(state)
        statements = [item["statement"] for item in closed]
        self.assertEqual(len(statements), len(set(statements)))
        self.assertLessEqual(len(closed), 6)

    def test_active_holds_never_appear_as_closed(self) -> None:
        state: dict[str, object] = {}
        _apply_human_hold(
            state,
            {"status": "PLANT", "statement": "open hold",
             "release_trigger": "proceed"},
            moving_origin_ordinal=1,
            choice_ref="sha256:" + "0" * 64,
        )
        self.assertEqual(_closed_human_holds(state), [])
        self.assertEqual(_closed_human_holds({}), [])


if __name__ == "__main__":
    unittest.main()

import unittest

from angler.runtime.higher_level_autonomy_adapter import (
    MAX_SELF_COMMITMENTS,
    _active_self_commitments,
    _apply_self_commitment_transitions,
    _resolved_self_commitments,
    _validated_self_commitment_transitions,
    _standing_standards,
)

REF = "sha256:" + "0" * 64


def _t(status, statement, due=""):
    return {"status": status, "statement": statement, "due": due}


class SelfCommitmentTest(unittest.TestCase):
    def test_validation(self):
        self.assertEqual(_validated_self_commitment_transitions(None), ())
        self.assertEqual(_validated_self_commitment_transitions([_t("NONE", "")]), ())
        with self.assertRaises(ValueError):
            _validated_self_commitment_transitions([_t("MAKE", "")])
        with self.assertRaises(ValueError):
            _validated_self_commitment_transitions([_t("PROMISE", "x")])
        with self.assertRaises(ValueError):
            _validated_self_commitment_transitions([_t("MAKE", "x")] * 5)

    def test_make_keep_withdraw_with_provenance(self):
        state = {}
        _apply_self_commitment_transitions(
            state, _validated_self_commitment_transitions(
                [_t("MAKE", "I will record this revision as a new artifact.", "my next own turn")]
            ), moving_origin_ordinal=304, choice_ref=REF,
        )
        active = _active_self_commitments(state)
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["made_ordinal"], 304)
        self.assertEqual(active[0]["due"], "my next own turn")
        _apply_self_commitment_transitions(
            state, _validated_self_commitment_transitions(
                [_t("KEEP", "I will record this revision as a new artifact.")]
            ), moving_origin_ordinal=310, choice_ref=REF,
        )
        self.assertEqual(_active_self_commitments(state), [])
        resolved = _resolved_self_commitments(state)
        self.assertEqual(resolved[0]["status"], "KEPT")
        self.assertEqual(resolved[0]["resolved_ordinal"], 310)
        _apply_self_commitment_transitions(
            state, _validated_self_commitment_transitions(
                [_t("MAKE", "I will read Spinoza tonight."), _t("WITHDRAW", "I will read Spinoza tonight.")]
            ), moving_origin_ordinal=311, choice_ref=REF,
        )
        self.assertEqual(_active_self_commitments(state), [])
        self.assertEqual(_resolved_self_commitments(state)[0]["status"], "WITHDRAWN")

    def test_unmatched_keep_is_observed_and_duplicates_not_made_twice(self):
        state = {}
        _apply_self_commitment_transitions(
            state, _validated_self_commitment_transitions([_t("KEEP", "never promised")]),
            moving_origin_ordinal=1, choice_ref=REF,
        )
        self.assertEqual(state["self_commitments"][0]["status"], "KEEP_UNMATCHED")
        for ordinal in (2, 3):
            _apply_self_commitment_transitions(
                state, _validated_self_commitment_transitions([_t("MAKE", "Same promise")]),
                moving_origin_ordinal=ordinal, choice_ref=REF,
            )
        self.assertEqual(len(_active_self_commitments(state)), 1)

    def test_bounded(self):
        state = {}
        for index in range(20):
            _apply_self_commitment_transitions(
                state, _validated_self_commitment_transitions([_t("MAKE", f"promise {index}")]),
                moving_origin_ordinal=index, choice_ref=REF,
            )
        self.assertLessEqual(len(state["self_commitments"]), MAX_SELF_COMMITMENTS)


class PromotionTest(unittest.TestCase):
    def _state_with_standard(self):
        return {
            "authored_artifacts": [
                {"artifact_ref": "sha256:" + "1" * 64,
                 "artifact": {"kind": "standard", "title": "Honest Reporting",
                              "body": "Requires... Breach... Repair... Scope... Owning state: Commitment Revision",
                              "version": 1, "supersedes_ref": None}},
                {"artifact_ref": "sha256:" + "2" * 64,
                 "artifact": {"kind": "private standard", "title": "Not a standard", "body": "x", "version": 1, "supersedes_ref": None}},
            ]
        }

    def test_standards_are_public_standard_kinds_only(self):
        titles = [s["title"] for s in _standing_standards(self._state_with_standard())]
        self.assertEqual(titles, ["Honest Reporting"])

    def test_promote_requires_an_active_commitment_and_an_existing_standard(self):
        state = self._state_with_standard()
        _apply_self_commitment_transitions(
            state, _validated_self_commitment_transitions([_t("MAKE", "Report with proof.")]),
            moving_origin_ordinal=1, choice_ref=REF,
        )
        with self.assertRaises(ValueError):
            _validated_self_commitment_transitions([{"status": "PROMOTE", "statement": "Report with proof.", "due": ""}])
        _apply_self_commitment_transitions(
            state, _validated_self_commitment_transitions([{"status": "PROMOTE", "statement": "Report with proof.", "due": "", "standard": "No Such Standard"}]),
            moving_origin_ordinal=2, choice_ref=REF,
        )
        self.assertEqual(state["self_commitments"][-1]["status"], "PROMOTE_UNMATCHED")
        self.assertEqual(len(_active_self_commitments(state)), 1)
        _apply_self_commitment_transitions(
            state, _validated_self_commitment_transitions([{"status": "PROMOTE", "statement": "Report with proof.", "due": "", "standard": "honest reporting"}]),
            moving_origin_ordinal=3, choice_ref=REF,
        )
        self.assertEqual(_active_self_commitments(state), [])
        resolved = _resolved_self_commitments(state)
        self.assertEqual(resolved[0]["status"], "PROMOTED")
        self.assertEqual(resolved[0]["standard"], "honest reporting")


if __name__ == "__main__":
    unittest.main()

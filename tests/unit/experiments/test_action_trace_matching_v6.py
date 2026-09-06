from __future__ import annotations

import unittest

from experiments.runners.action_trace_matching_v6 import (
    CAUSAL_MARGIN_ARMS,
    DIAGNOSTIC_ARMS,
    EPOCHS,
    IDENTITY,
)


class ActionTraceMatchingV6Tests(unittest.TestCase):
    def test_identity_budget_and_diagnostics_are_frozen(self) -> None:
        self.assertEqual(IDENTITY, "angler.action-trace-matching.v6")
        self.assertEqual(EPOCHS, 8)
        self.assertEqual(
            CAUSAL_MARGIN_ARMS,
            ("reset_state", "coordinates_removed", "unrelated_evidence", "procedure_slots_removed"),
        )
        self.assertEqual(DIAGNOSTIC_ARMS, ("local_only", "persistent_only", "correspondence_removed"))
        self.assertTrue(set(CAUSAL_MARGIN_ARMS).isdisjoint(DIAGNOSTIC_ARMS))


if __name__ == "__main__":
    unittest.main()

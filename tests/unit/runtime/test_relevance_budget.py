"""Relevance budget: sections kept whole in relevance order under a per-stage
token budget; what is left out is named; nothing is rewritten."""
import os
import unittest

from angler.runtime import relevance_budget as rb


class BudgetTest(unittest.TestCase):
    def setUp(self):
        self._env = {k: os.environ.get(k) for k in ("JENNY2_BUDGET_ROUTER", "JENNY2_RELEVANCE_BUDGET", "JENNY2_BUDGET_SCALE")}

    def tearDown(self):
        for k, v in self._env.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v

    def test_everything_fits_under_a_generous_budget_and_the_manifest_says_so(self):
        ctx = {"observation": {"content": "hi"}, "named_states": [{"label": "Care"}], "history_channels": {"x": 1}}
        out = rb.fit(ctx, stage="router", turn_text="hi")
        self.assertEqual({k: v for k, v in out.items() if k != "context_budget"}, ctx)
        self.assertEqual(out["context_budget"]["omitted"], [])
        self.assertEqual(list(out)[:3], list(ctx))  # order preserved

    def test_low_relevance_sections_are_dropped_first_and_always_sections_never(self):
        os.environ["JENNY2_BUDGET_ROUTER"] = "1000"
        big = "word " * 2500  # ~3.5K tokens
        ctx = {
            "observation": {"content": "tell me about the river valley"},
            "named_states": [{"label": "River Valley Curiosity", "inclination": "read about valleys"}],
            "compute_calibration": {"blob": big},
            "history_channels": {"blob": big},
            "standing_standards": [{"title": "Honest Reporting"}],
        }
        out = rb.fit(ctx, stage="router", turn_text=ctx["observation"]["content"])
        self.assertIn("observation", out)
        self.assertIn("named_states", out)
        self.assertIn("standing_standards", out)
        omitted = {item["section"] for item in out["context_budget"]["omitted"]}
        self.assertEqual(omitted, {"compute_calibration", "history_channels"})
        self.assertLessEqual(out["context_budget"]["used_tokens"], 1000 + 60)

    def test_turn_overlap_raises_a_sections_rank(self):
        turn = "what did you write about the empty desk"
        low = rb.relevance("router", "qualitative_feedback", {"note": "nothing here"}, rb._words(turn))
        high = rb.relevance("router", "qualitative_feedback", {"note": "the empty desk prose write"}, rb._words(turn))
        self.assertGreater(high, low)

    def test_disabled_returns_the_context_unchanged(self):
        os.environ["JENNY2_RELEVANCE_BUDGET"] = "0"
        ctx = {"a": 1, "b": [1, 2, 3]}
        self.assertEqual(rb.fit(ctx, stage="cortex"), ctx)

    def test_budget_env_and_scale(self):
        os.environ["JENNY2_BUDGET_ROUTER"] = "2500"
        self.assertEqual(rb.budget_for("router"), 2500)
        os.environ.pop("JENNY2_BUDGET_ROUTER")
        os.environ["JENNY2_BUDGET_SCALE"] = "0.5"
        self.assertEqual(rb.budget_for("router"), rb.DEFAULT_BUDGETS["router"] // 2)


if __name__ == "__main__":
    unittest.main()

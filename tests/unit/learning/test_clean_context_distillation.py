from __future__ import annotations

import unittest

from experiments.runners.phase1_minimal_plastic_learner import (
    analysis_task_request,
    clean_teacher_prefixes,
    final_submission_request,
)
from angler.worlds import LearnerTask, PrecedenceConstraint


class CleanContextDistillationTests(unittest.TestCase):
    def test_corrected_trace_is_rebased_under_original_task_only(self) -> None:
        task = LearnerTask(
            instance_id="task-1",
            family_id="angler.relational-order",
            family_version="1.1.0",
            symbols=("amber", "birch"),
            constraints=(PrecedenceConstraint("amber", "birch"),),
            fact_statements=("amber is before birch.",),
            problem_statement="Order the symbols using the visible constraint.",
            prompt=(
                "Order the symbols using the visible constraint.\n"
                "Return only the final comma-separated symbol sequence."
            ),
        )
        accepted_analysis = "Amber precedes birch, so amber must appear first."

        analysis_prefix, submission_prefix = clean_teacher_prefixes(
            task,
            accepted_analysis,
        )

        self.assertEqual(
            analysis_prefix,
            ({"role": "user", "content": analysis_task_request(task)},),
        )
        self.assertEqual(
            submission_prefix,
            (
                {"role": "user", "content": analysis_task_request(task)},
                {"role": "assistant", "content": accepted_analysis},
                {"role": "user", "content": final_submission_request()},
            ),
        )
        self.assertNotIn("feedback", repr(analysis_prefix).lower())
        self.assertNotIn("feedback", repr(submission_prefix).lower())
        self.assertNotIn("return only", analysis_prefix[0]["content"].lower())
        self.assertIn("do not submit", analysis_prefix[0]["content"].lower())


if __name__ == "__main__":
    unittest.main()

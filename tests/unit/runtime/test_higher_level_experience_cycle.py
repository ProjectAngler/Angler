from __future__ import annotations

import unittest

from angler.runtime.higher_level_experience_cycle import (
    ConsequenceVector, ExecutionReceipt, HigherLevelExperienceCycle,
    MemoryCandidate, StructuredExperience, TemporalContext, UtilityMemory,
)


class _Memory:
    def __init__(self): self.stored = []
    def recall(self, request, *, limit):
        return (MemoryCandidate("sha256:"+"a"*64, "older useful attempt", .2), MemoryCandidate("sha256:"+"b"*64, "nearer attempt", .1))[:limit]
    def store(self, content, provenance_refs):
        self.stored.append((content, tuple(provenance_refs))); return "sha256:"+"c"*64


class _Temporal:
    def __init__(self): self.now = 3
    def context(self): return TemporalContext(self.now, "sha256:"+"d"*64)
    def advance(self, event_ref): self.now += 1; return TemporalContext(self.now, event_ref)


class _Experience:
    def __init__(self): self.reflect_calls = 0
    def generate_experience(self, request, memories, temporal):
        return StructuredExperience("Interpret constraints.", "VERIFY", "Try, then test assumptions.", "Evidence should improve.", ("Check constraints.",), .4)
    def reflect(self, request, receipt, consequence, experience): self.reflect_calls += 1; return "The approach made partial progress; revise the weak assumption."
    def consolidate(self, request, experience, reflection, consequence): return "Reusable procedure: verify constraints before committing."


class _Cortex:
    def __init__(self): self.inputs = []
    def execute(self, request, experience, memories):
        self.inputs.append((request, experience, tuple(memories)))
        return ExecutionReceipt("sha256:"+"e"*64, "public response", "COMPLETED")


def _consequence():
    return ConsequenceVector(.4, .5, .3, .7, .6, .8, .2, 1.0, .2)


class HigherLevelExperienceCycleTests(unittest.TestCase):
    def test_closed_cycle_orders_recall_utility_receipt_feedback_and_state(self):
        memory, experience, cortex = _Memory(), _Experience(), _Cortex()
        cycle = HigherLevelExperienceCycle(memory=memory, temporal=_Temporal(), utility=UtilityMemory(alpha=.5), experience_model=experience, cortex=cortex)
        prepared = cycle.prepare("new request")
        self.assertNotIn("objective_progress", repr(cortex.inputs))
        completed = cycle.observe(prepared, _consequence())
        self.assertEqual([span.stage for span in completed.spans], ["SEMANTIC_RECALL", "UTILITY_SELECTION", "EXPERIENCE_GENERATION", "CORTEX_EXECUTION", "OBJECTIVE_CONSEQUENCE", "REFLECTION", "CONSOLIDATION", "UTILITY_UPDATE", "TEMPORAL_ADVANCE"])
        self.assertTrue(all(value > 0 for value in completed.utility_state.values()))
        self.assertEqual(completed.next_temporal.now, 4)
        self.assertEqual(len(memory.stored), 1)

    def test_no_reflection_and_no_utility_are_real_removals(self):
        utility = UtilityMemory(alpha=1.0)
        exp = _Experience()
        cycle = HigherLevelExperienceCycle(memory=_Memory(), temporal=_Temporal(), utility=utility, experience_model=exp, cortex=_Cortex())
        done = cycle.observe(cycle.prepare("request", removal="NO_REFLECTION"), _consequence())
        self.assertEqual(done.reflection, "REMOVED")
        self.assertEqual(exp.reflect_calls, 0)
        utility2 = UtilityMemory(alpha=1.0)
        cycle2 = HigherLevelExperienceCycle(memory=_Memory(), temporal=_Temporal(), utility=utility2, experience_model=_Experience(), cortex=_Cortex())
        done2 = cycle2.observe(cycle2.prepare("request", removal="NO_UTILITY"), _consequence())
        self.assertEqual(done2.utility_state, {})

    def test_noncompleted_receipt_cannot_learn(self):
        class Clarifying(_Cortex):
            def execute(self, request, experience, memories): return ExecutionReceipt("sha256:"+"f"*64, "question", "CLARIFICATION_REQUIRED")
        cycle = HigherLevelExperienceCycle(memory=_Memory(), temporal=_Temporal(), utility=UtilityMemory(), experience_model=_Experience(), cortex=Clarifying())
        with self.assertRaisesRegex(RuntimeError, "only completed"):
            cycle.observe(cycle.prepare("request"), _consequence())


if __name__ == "__main__": unittest.main()

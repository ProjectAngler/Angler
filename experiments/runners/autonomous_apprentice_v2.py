"""Frozen V2 evaluation of semantic autonomous apprenticeship integration.

This successor reuses the V1 evaluator and controls but supplies a fresh
identity, fresh paired tasks, and task-contract opt-ins for bounded diagnostic
feedback and bounded retention of the model's own procedure content.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

from angler.runtime import AutonomousTaskContract
from experiments.runners import autonomous_apprentice_v1 as base


IDENTITY = "angler.autonomous-apprentice.v2-semantic-integration-first-result"


TASKS = (
    base.TaskTemplate(
        "token-a",
        "canonical_token",
        "acquisition",
        "Implement make_token(text) in solution.py. Strip outer whitespace, lowercase the text, replace every run of non-alphanumeric characters with one underscore, and remove leading or trailing underscores.",
        "def make_token(text):\n    return text\n",
        "from solution import make_token as f; assert f('  Alpha / BETA + Gamma ')=='alpha_beta_gamma'; assert f('X...Y')=='x_y'; assert f('___')==''",
    ),
    base.TaskTemplate(
        "distinct-a",
        "stable_distinct",
        "acquisition",
        "Implement first_distinct(values) in solution.py. Return a list containing only the earliest occurrence of each hashable value while preserving input order.",
        "def first_distinct(values):\n    return sorted(set(values))\n",
        "from solution import first_distinct as f; assert f(['b','a','b','c','a'])==['b','a','c']; assert f([2,2,1])==[2,1]; assert f(())==[]",
    ),
    base.TaskTemplate(
        "range-a",
        "validated_range",
        "acquisition",
        "Implement fit_range(number, low, high) in solution.py. Clamp number inclusively between low and high, and raise ValueError if low is greater than high.",
        "def fit_range(number, low, high):\n    return low\n",
        "from solution import fit_range as f; assert f(-3,0,4)==0; assert f(8,0,4)==4; assert f(2,0,4)==2;\ntry: f(1,3,2)\nexcept ValueError: pass\nelse: raise AssertionError('bounds')",
    ),
    base.TaskTemplate(
        "batch-a",
        "bounded_batch",
        "acquisition",
        "Implement batches(values, limit) in solution.py. Return consecutive list batches of at most limit items, preserve order, return an empty list for empty input, and raise ValueError unless limit is positive.",
        "def batches(values, limit):\n    return []\n",
        "from solution import batches as f; assert f([1,2,3,4,5],2)==[[1,2],[3,4],[5]]; assert f([],3)==[];\ntry: f([1],0)\nexcept ValueError: pass\nelse: raise AssertionError('limit')",
    ),
    base.TaskTemplate(
        "token-b",
        "canonical_token",
        "transfer",
        "Implement storage_id(value) in solution.py. Trim surrounding whitespace, lowercase it, collapse each run of characters that are not letters or digits to one underscore, then trim underscores at both ends.",
        "def storage_id(value):\n    return value.lower()\n",
        "from solution import storage_id as f; assert f(' Red---GREEN / Blue ')=='red_green_blue'; assert f('A..B')=='a_b'; assert f('***')==''",
    ),
    base.TaskTemplate(
        "distinct-b",
        "stable_distinct",
        "transfer",
        "Implement dedupe_ordered(items) in solution.py. Produce a list retaining each hashable item only at its first position and otherwise preserving order.",
        "def dedupe_ordered(items):\n    return list(set(items))\n",
        "from solution import dedupe_ordered as f; assert f(['q','x','q','r','x'])==['q','x','r']; assert f([3,3,2])==[3,2]; assert f([])==[]",
    ),
    base.TaskTemplate(
        "range-b",
        "validated_range",
        "transfer",
        "Implement limit_score(score, minimum, maximum) in solution.py. Return score limited inclusively to the bounds and reject minimum greater than maximum with ValueError.",
        "def limit_score(score, minimum, maximum):\n    return maximum\n",
        "from solution import limit_score as f; assert f(9,1,5)==5; assert f(-2,1,5)==1; assert f(4,1,5)==4;\ntry: f(0,4,3)\nexcept ValueError: pass\nelse: raise AssertionError('bounds')",
    ),
    base.TaskTemplate(
        "batch-b",
        "bounded_batch",
        "transfer",
        "Implement windows(sequence, capacity) in solution.py. Return order-preserving consecutive list groups no larger than capacity, return [] for empty input, and raise ValueError unless capacity is positive.",
        "def windows(sequence, capacity):\n    return [list(sequence)]\n",
        "from solution import windows as f; assert f('abcdef',4)==[['a','b','c','d'],['e','f']]; assert f([],2)==[];\ntry: f([1],-2)\nexcept ValueError: pass\nelse: raise AssertionError('capacity')",
    ),
)


def _task_contract(template: base.TaskTemplate, root: Path) -> AutonomousTaskContract:
    return AutonomousTaskContract(
        task_id=template.task_id,
        request=template.request,
        success_description="the private objective Python verifier exits successfully",
        workspace_root=root,
        readable_paths=("solution.py",),
        writable_paths=("solution.py",),
        verifier_command=(sys.executable, "-c", template.verifier_source),
        allow_mutation=True,
        max_steps=12,
        max_verifier_runs=4,
        verifier_timeout_seconds=20.0,
        max_file_bytes=32_768,
        max_observation_chars=4_096,
        reveal_verifier_diagnostics=True,
        retain_procedure_content=True,
        max_learning_trace_chars=3_072,
    )


async def evaluate(args: argparse.Namespace) -> None:
    # The evaluator is deliberately parameterized here rather than copying its
    # execution logic. This process owns these module globals and exits after
    # producing the fresh V2 identity.
    base.IDENTITY = IDENTITY
    base.TASKS = TASKS
    base._task_contract = _task_contract
    await base.evaluate(args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument(
        "--checkpoint",
        default="/opt/angler/results/compositional-procedure-v6-action-match.pt",
    )
    parser.add_argument(
        "--state-root",
        default="/opt/angler/state/autonomous-apprentice-v2/evaluation-first",
    )
    parser.add_argument(
        "--result",
        default="/opt/angler/results/autonomous-apprentice-v2-evaluation.json",
    )
    asyncio.run(evaluate(parser.parse_args()))


if __name__ == "__main__":
    main()

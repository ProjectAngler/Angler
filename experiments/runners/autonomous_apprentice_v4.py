"""Frozen V4 evaluation of outcome-correct autonomous apprenticeship."""

from __future__ import annotations

import argparse
import asyncio

from experiments.runners import autonomous_apprentice_v1 as base
from experiments.runners.autonomous_apprentice_v2 import _task_contract


IDENTITY = "angler.autonomous-apprentice.v4-outcome-credit-first-result"


def _verifier(import_line: str, body: str) -> str:
    return (
        import_line
        + "\n"
        + "def check(name, got, expected):\n"
        + "    if got != expected: raise AssertionError(f'{name}: got={got!r}; expected={expected!r}')\n"
        + body
    )


TASKS = (
    base.TaskTemplate("name-a", "canonical_name", "acquisition", "Implement label_code(text) in solution.py. Strip outer whitespace, lowercase text, collapse every run of non-alphanumeric characters to one dot, and remove leading or trailing dots.", "def label_code(text):\n    return text\n", _verifier("from solution import label_code as f", "check('mixed',f('  One / TWO + Three '),'one.two.three')\ncheck('run',f('A...B'),'a.b')\ncheck('empty',f('***'),'')")),
    base.TaskTemplate("order-a", "stable_order", "acquisition", "Implement earliest_values(values) in solution.py. Return a list retaining only the earliest occurrence of each hashable value while preserving order.", "def earliest_values(values):\n    return sorted(set(values))\n", _verifier("from solution import earliest_values as f", "check('words',f(['d','a','d','c','a']),['d','a','c'])\ncheck('numbers',f([5,5,2]),[5,2])\ncheck('empty',f(()),[])")),
    base.TaskTemplate("cap-a", "validated_cap", "acquisition", "Implement cap_amount(amount, minimum, maximum) in solution.py. Clamp amount inclusively to the bounds and raise ValueError when minimum is greater than maximum.", "def cap_amount(amount, minimum, maximum):\n    return minimum\n", _verifier("from solution import cap_amount as f", "check('low',f(-4,0,7),0)\ncheck('high',f(9,0,7),7)\ncheck('middle',f(3,0,7),3)\ntry: f(1,5,4)\nexcept ValueError: pass\nelse: raise AssertionError('reversed bounds must raise ValueError')")),
    base.TaskTemplate("block-a", "bounded_block", "acquisition", "Implement blocks(values, count) in solution.py. Return consecutive list blocks of at most count items, preserve order, return [] for empty input, and raise ValueError unless count is positive.", "def blocks(values, count):\n    return []\n", _verifier("from solution import blocks as f", "check('groups',f([1,2,3,4,5],2),[[1,2],[3,4],[5]])\ncheck('empty',f([],3),[])\ntry: f([1],0)\nexcept ValueError: pass\nelse: raise AssertionError('nonpositive count must raise ValueError')")),
    base.TaskTemplate("name-b", "canonical_name", "transfer", "Implement dotted_name(value) in solution.py. Trim outer whitespace, lowercase it, replace each run of characters that are not letters or digits with one dot, then trim edge dots.", "def dotted_name(value):\n    return value.lower()\n", _verifier("from solution import dotted_name as f", "check('mixed',f(' Red___GREEN / Blue '),'red.green.blue')\ncheck('run',f('X..Y'),'x.y')\ncheck('empty',f('___'),'')")),
    base.TaskTemplate("order-b", "stable_order", "transfer", "Implement preserve_unique(items) in solution.py. Produce a list keeping each hashable item only at its first input position.", "def preserve_unique(items):\n    return list(set(items))\n", _verifier("from solution import preserve_unique as f", "check('words',f(['p','x','p','q','x']),['p','x','q'])\ncheck('numbers',f([7,7,2]),[7,2])\ncheck('empty',f([]),[])")),
    base.TaskTemplate("cap-b", "validated_cap", "transfer", "Implement bounded(number, floor, ceiling) in solution.py. Limit number inclusively to floor and ceiling and raise ValueError for reversed bounds.", "def bounded(number, floor, ceiling):\n    return ceiling\n", _verifier("from solution import bounded as f", "check('high',f(8,1,6),6)\ncheck('low',f(-2,1,6),1)\ncheck('middle',f(4,1,6),4)\ntry: f(0,4,3)\nexcept ValueError: pass\nelse: raise AssertionError('reversed bounds must raise ValueError')")),
    base.TaskTemplate("block-b", "bounded_block", "transfer", "Implement segments(sequence, width) in solution.py. Return order-preserving consecutive list segments no larger than width, return [] for empty input, and raise ValueError unless width is positive.", "def segments(sequence, width):\n    return [list(sequence)]\n", _verifier("from solution import segments as f", "check('text',f('abcdef',4),[['a','b','c','d'],['e','f']])\ncheck('empty',f([],2),[])\ntry: f([1],-1)\nexcept ValueError: pass\nelse: raise AssertionError('nonpositive width must raise ValueError')")),
)


async def evaluate(args: argparse.Namespace) -> None:
    base.IDENTITY = IDENTITY
    base.TASKS = TASKS
    base._task_contract = _task_contract
    await base.evaluate(args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--checkpoint", default="/opt/angler/results/compositional-procedure-v6-action-match.pt")
    parser.add_argument("--state-root", default="/opt/angler/state/autonomous-apprentice-v4/evaluation-first")
    parser.add_argument("--result", default="/opt/angler/results/autonomous-apprentice-v4-evaluation.json")
    asyncio.run(evaluate(parser.parse_args()))


if __name__ == "__main__":
    main()

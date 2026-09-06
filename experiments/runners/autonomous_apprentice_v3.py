"""Frozen V3 evaluation after controller-owned action identity repair."""

from __future__ import annotations

import argparse
import asyncio

from experiments.runners import autonomous_apprentice_v1 as base
from experiments.runners.autonomous_apprentice_v2 import _task_contract


IDENTITY = "angler.autonomous-apprentice.v3-controller-identity-first-result"


TASKS = (
    base.TaskTemplate("key-a", "canonical_key", "acquisition", "Implement safe_key(text) in solution.py. Strip outer whitespace, lowercase text, collapse every run of non-alphanumeric characters to one hyphen, and trim leading or trailing hyphens.", "def safe_key(text):\n    return text\n", "from solution import safe_key as f; assert f('  North / SOUTH + East ')=='north-south-east'; assert f('A...B')=='a-b'; assert f('***')==''"),
    base.TaskTemplate("seen-a", "stable_seen", "acquisition", "Implement retain_first(items) in solution.py. Return a list keeping only the first occurrence of each hashable item while preserving order.", "def retain_first(items):\n    return list(set(items))\n", "from solution import retain_first as f; assert f(['c','a','c','b','a'])==['c','a','b']; assert f([4,4,2])==[4,2]; assert f([])==[]"),
    base.TaskTemplate("clip-a", "validated_clip", "acquisition", "Implement clip_value(value, floor, ceiling) in solution.py. Clamp value inclusively to the bounds and raise ValueError if floor is greater than ceiling.", "def clip_value(value, floor, ceiling):\n    return floor\n", "from solution import clip_value as f; assert f(-1,0,6)==0; assert f(9,0,6)==6; assert f(3,0,6)==3;\ntry: f(1,5,4)\nexcept ValueError: pass\nelse: raise AssertionError('bounds')"),
    base.TaskTemplate("group-a", "bounded_group", "acquisition", "Implement groups(values, size) in solution.py. Return consecutive list groups of at most size items, preserve order, return [] for empty input, and raise ValueError unless size is positive.", "def groups(values, size):\n    return []\n", "from solution import groups as f; assert f([1,2,3,4,5],3)==[[1,2,3],[4,5]]; assert f([],2)==[];\ntry: f([1],0)\nexcept ValueError: pass\nelse: raise AssertionError('size')"),
    base.TaskTemplate("key-b", "canonical_key", "transfer", "Implement route_key(value) in solution.py. Trim surrounding whitespace, lowercase it, replace each run of characters that are not letters or digits with one hyphen, then strip edge hyphens.", "def route_key(value):\n    return value.lower()\n", "from solution import route_key as f; assert f(' Red___GREEN / Blue ')=='red-green-blue'; assert f('X..Y')=='x-y'; assert f('___')==''"),
    base.TaskTemplate("seen-b", "stable_seen", "transfer", "Implement unique_in_order(values) in solution.py. Produce a list containing each hashable value only at its earliest input position.", "def unique_in_order(values):\n    return sorted(set(values))\n", "from solution import unique_in_order as f; assert f(['m','z','m','n','z'])==['m','z','n']; assert f([8,8,3])==[8,3]; assert f(())==[]"),
    base.TaskTemplate("clip-b", "validated_clip", "transfer", "Implement constrain(number, minimum, maximum) in solution.py. Limit number inclusively to minimum and maximum, raising ValueError for reversed bounds.", "def constrain(number, minimum, maximum):\n    return maximum\n", "from solution import constrain as f; assert f(7,1,5)==5; assert f(-2,1,5)==1; assert f(3,1,5)==3;\ntry: f(0,3,2)\nexcept ValueError: pass\nelse: raise AssertionError('bounds')"),
    base.TaskTemplate("group-b", "bounded_group", "transfer", "Implement partitions(sequence, width) in solution.py. Return order-preserving consecutive list partitions no larger than width, return [] for empty input, and raise ValueError unless width is positive.", "def partitions(sequence, width):\n    return [list(sequence)]\n", "from solution import partitions as f; assert f('abcde',2)==[['a','b'],['c','d'],['e']]; assert f([],4)==[];\ntry: f([1],-1)\nexcept ValueError: pass\nelse: raise AssertionError('width')"),
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
    parser.add_argument("--state-root", default="/opt/angler/state/autonomous-apprentice-v3/evaluation-first")
    parser.add_argument("--result", default="/opt/angler/results/autonomous-apprentice-v3-evaluation.json")
    asyncio.run(evaluate(parser.parse_args()))


if __name__ == "__main__":
    main()

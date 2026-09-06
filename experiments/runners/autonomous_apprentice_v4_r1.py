"""Technical recovery of V4 after exact-duplicate episode persistence."""

from __future__ import annotations

import argparse
import asyncio

from experiments.runners import autonomous_apprentice_v1 as base
from experiments.runners.autonomous_apprentice_v2 import _task_contract
from experiments.runners.autonomous_apprentice_v4 import TASKS


IDENTITY = "angler.autonomous-apprentice.v4-r1-idempotent-persistence-first-result"


async def evaluate(args: argparse.Namespace) -> None:
    base.IDENTITY = IDENTITY
    base.TASKS = TASKS
    base._task_contract = _task_contract
    await base.evaluate(args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--checkpoint", default="/opt/angler/results/compositional-procedure-v6-action-match.pt")
    parser.add_argument("--state-root", default="/opt/angler/state/autonomous-apprentice-v4-r1/evaluation-first")
    parser.add_argument("--result", default="/opt/angler/results/autonomous-apprentice-v4-r1-evaluation.json")
    asyncio.run(evaluate(parser.parse_args()))


if __name__ == "__main__":
    main()

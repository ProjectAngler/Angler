"""Lanes: split and rejoin her context pool (Becca, 2026-09-07: "split and
rejoin the context pool ... parallel states get distributed pool slices and
serial states recombine them into coherence").

Her model server admits a fixed number of concurrent requests; each is a
lane with its own slice of the token pool. Independent stages run in lanes
at once; serial stages run alone. The rejoin is not done here: every lane
returns a value, and the caller writes results into the single-writer state
in a fixed order, so the record stays coherent whatever the lanes' timing.

Orchestration only. A lane never decides anything; it carries one stage's
call. A lane failure returns the stage's own failure record; the turn goes
on. Latency-trace context is copied into each lane so every model call is
still attributed to its turn.
"""
from __future__ import annotations

import contextvars
import os
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor

DEFAULT_LANES = 3


def lane_count() -> int:
    try:
        return max(1, min(8, int(os.environ.get("JENNY2_LANES", str(DEFAULT_LANES)))))
    except ValueError:
        return DEFAULT_LANES


_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def _executor() -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=lane_count(), thread_name_prefix="jenny2-lane")
        return _pool


def run_lanes(tasks: Mapping[str, Callable[[], object]], *, timeout_seconds: float = 900.0) -> dict[str, object]:
    """Run independent callables at once, one per lane, and return their
    results by name in the caller's own order. With one lane configured, or
    one task, this is plain sequential execution. A task that raises returns
    its exception object under its name; the caller decides what that means."""

    names = list(tasks)
    if lane_count() <= 1 or len(names) <= 1:
        results: dict[str, object] = {}
        for name in names:
            try:
                results[name] = tasks[name]()
            except Exception as exc:  # noqa: BLE001 — a lane's failure is the stage's own record
                results[name] = exc
        return results
    executor = _executor()
    futures: dict[str, Future] = {}
    for name in names:
        context = contextvars.copy_context()
        futures[name] = executor.submit(context.run, tasks[name])
    results = {}
    for name in names:
        try:
            results[name] = futures[name].result(timeout=timeout_seconds)
        except Exception as exc:  # noqa: BLE001
            results[name] = exc
    return results

"""Low-overhead, content-free latency tracing for Jenny runtime operations."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Iterator, Mapping


_CURRENT_TRACE: ContextVar[dict[str, object] | None] = ContextVar(
    "jenny_latency_trace", default=None
)
_PHASE_STACK: ContextVar[tuple[str, ...]] = ContextVar(
    "jenny_latency_phase_stack", default=()
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@contextmanager
def capture_latency_trace(
    *, request_id: str, source: str
) -> Iterator[dict[str, object]]:
    """Capture one operation without retaining prompt or response contents."""

    started = perf_counter()
    trace: dict[str, object] = {
        "schema": "jenny2.latency-trace.v1",
        "request_id": request_id,
        "source": source,
        "started_at_utc": _utc_now(),
        "phases": [],
        "model_calls": [],
        "_started_monotonic": started,
    }
    trace_token = _CURRENT_TRACE.set(trace)
    stack_token = _PHASE_STACK.set(())
    try:
        yield trace
        trace["status"] = "COMPLETED"
    except BaseException as exc:
        trace["status"] = "ERROR"
        trace["error_type"] = type(exc).__name__
        raise
    finally:
        trace["finished_at_utc"] = _utc_now()
        trace["total_ms"] = round((perf_counter() - started) * 1000, 3)
        trace.pop("_started_monotonic", None)
        _PHASE_STACK.reset(stack_token)
        _CURRENT_TRACE.reset(trace_token)


@contextmanager
def latency_phase(name: str) -> Iterator[None]:
    """Record one nested phase when a request trace is active."""

    trace = _CURRENT_TRACE.get()
    if trace is None:
        yield
        return
    parent_stack = _PHASE_STACK.get()
    stack = (*parent_stack, name)
    token = _PHASE_STACK.set(stack)
    started = perf_counter()
    phase: dict[str, object] = {
        "name": name,
        "path": list(stack),
        "started_offset_ms": round(
            (started - float(trace.get("_started_monotonic", started))) * 1000, 3
        ),
    }
    try:
        yield
        phase["status"] = "COMPLETED"
    except BaseException as exc:
        phase["status"] = "ERROR"
        phase["error_type"] = type(exc).__name__
        raise
    finally:
        finished = perf_counter()
        phase["elapsed_ms"] = round((finished - started) * 1000, 3)
        phases = trace.get("phases")
        if isinstance(phases, list):
            phases.append(phase)
        _PHASE_STACK.reset(token)


def record_model_call(metrics: Mapping[str, object]) -> None:
    """Attach sanitized model-call measurements to the active request."""

    trace = _CURRENT_TRACE.get()
    if trace is None:
        return
    calls = trace.get("model_calls")
    if not isinstance(calls, list):
        return
    call = dict(metrics)
    call["sequence"] = len(calls) + 1
    call["phase_path"] = list(_PHASE_STACK.get())
    calls.append(call)


def annotate_latency_trace(values: Mapping[str, object]) -> None:
    trace = _CURRENT_TRACE.get()
    if trace is not None:
        trace.update(values)


def persist_latency_trace(root: Path, trace: Mapping[str, object]) -> tuple[str, Path]:
    """Write one immutable trace and return its content reference and path."""

    resolved = root.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    request_id = str(trace.get("request_id", "unknown"))
    request_digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    encoded_without_ref = json.dumps(
        dict(trace),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    trace_ref = "sha256:" + hashlib.sha256(encoded_without_ref).hexdigest()
    payload = dict(trace)
    payload["trace_ref"] = trace_ref
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    target = resolved / f"{request_digest}-{trace_ref[7:23]}.json"
    if target.exists():
        if target.read_bytes() != encoded:
            raise RuntimeError("latency trace identity collided with different bytes")
        return trace_ref, target
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(target)
    return trace_ref, target

"""One Unix system monotonic epoch for cross-process deadline arbitration."""

import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Iterator, Optional


_active = ContextVar("operation_timing", default=None)


class Timing:
    """Bounded operation-local inclusive spans, shared by nested calculation stages."""

    def __init__(self, origin: Optional[float] = None, limit: int = 500,
                 progress: Optional[Callable[[dict], None]] = None) -> None:
        self.origin = monotonic_now() if origin is None else origin
        self.limit = limit
        self.spans = []
        self.stack = []
        self.serial = 0
        self.truncated = False
        self.progress = progress
        self.aggregates = {}

    @contextmanager
    def activate(self) -> Iterator["Timing"]:
        """Install this recorder for the current calculation context."""
        token = _active.set(self)
        try:
            yield self
        finally:
            _active.reset(token)

    @contextmanager
    def span(self, name: str, kind: str = "step", **attributes: Any) -> Iterator[dict]:
        """Retain an inclusive stage even when its computation raises."""
        self.serial += 1
        started = monotonic_now()
        value = {"id": str(self.serial), "name": name, "kind": kind,
                 "start_ms": max(0.0, (started - self.origin) * 1000),
                 "state": "running", **attributes}
        if self.stack:
            value["parent_id"] = self.stack[-1]["id"]
        retained = len(self.spans) < self.limit
        if retained:
            self.spans.append(value)
        else:
            self.truncated = True
        self.stack.append(value)
        if retained and self.progress and kind in {"phase", "node", "target"}:
            self.progress(dict(value))
        try:
            yield value
            value["state"] = value.pop("node_state", "succeeded")
        except BaseException:
            value["state"] = "failed"
            raise
        finally:
            value["duration_ms"] = max(0.0, (monotonic_now() - started) * 1000)
            self.stack.pop()
            key = (name, kind)
            if key not in self.aggregates and len(self.aggregates) < 64:
                self.aggregates[key] = {"name": name, "kind": kind, "count": 0,
                                        "total_ms": 0.0, "max_ms": 0.0}
            if key not in self.aggregates:
                self.truncated = True
            aggregate = self.aggregates.get(key)
            if aggregate is not None:
                aggregate["count"] += 1
                aggregate["total_ms"] += value["duration_ms"]
                aggregate["max_ms"] = max(aggregate["max_ms"], value["duration_ms"])
            # Keep slow/error leaf spans even when an unusually large recipe fills the budget.
            if not retained and (not self.stack or self.stack[-1]["id"] in {s["id"] for s in self.spans}):
                parents = {span.get("parent_id") for span in self.spans}
                leaves = [span for span in self.spans if span["id"] not in parents
                          and span.get("state") not in {"running", "failed"}
                          and span["kind"] != "phase"]
                if leaves:
                    smallest = min(leaves, key=lambda span: span.get("duration_ms", 0))
                    if value["state"] == "failed" or value["duration_ms"] > smallest["duration_ms"]:
                        self.spans.remove(smallest)
                        self.spans.append(value)
            if retained and self.progress and kind in {"phase", "node", "target"}:
                self.progress(dict(value))

    def snapshot(self, complete: bool = False) -> dict:
        """Return JSON-safe measurements, never absolute monotonic timestamps."""
        return {"schema_version": 1,
                "status": "complete" if complete and not self.truncated else "partial",
                "total_ms": max(0.0, (monotonic_now() - self.origin) * 1000),
                "spans": [dict(value) for value in self.spans],
                "aggregates": list(self.aggregates.values()), "truncated": self.truncated}


@contextmanager
def span(name: str, kind: str = "step", **attributes: Any) -> Iterator[dict]:
    """Measure only inside an active operation; ordinary engine calls stay unchanged."""
    recorder = _active.get()
    if recorder is None:
        yield {}
    else:
        with recorder.span(name, kind, **attributes) as value:
            yield value


def timed(name: str) -> Callable:
    """Instrument API implementation work without changing its result or exception."""
    def decorate(function: Callable) -> Callable:
        @wraps(function)
        def measured(*args: Any, **kwargs: Any) -> Any:
            with span(name):
                return function(*args, **kwargs)
        return measured
    return decorate


def monotonic_now() -> float:
    """Read the same non-wall-clock epoch in the HTTP owner and spawned workers.

    Python 3.9 on macOS gives each process a different time.monotonic() origin.
    Unix CLOCK_MONOTONIC provides a shared OS epoch, matching this runtime's flock
    and process-isolation deployment requirements.

    Returns:
        Seconds from the operating system's monotonic epoch.
    """
    return time.clock_gettime(time.CLOCK_MONOTONIC)

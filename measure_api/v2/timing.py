"""One Unix system monotonic epoch for cross-process deadline arbitration."""

import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Dict, Iterator, Optional


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
        self.counters: Dict[str, float] = {}
        self.counter_limit = 64

    def counter(self, name: str, amount: float = 1.0) -> None:
        """Accumulate one numeric observation without allocating a span.

        Counters answer questions spans cannot afford to: how many NCC
        evaluations ran, how many megapixels were decoded, how many cache
        writes happened.  Names are bounded like aggregate names.

        Args:
            name: Stable counter identifier shared by every producer.
            amount: Increment added to the current value.
        """
        if name in self.counters:
            self.counters[name] += amount
            return
        if len(self.counters) >= self.counter_limit:
            self.truncated = True
            return
        self.counters[name] = amount

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Accumulate elapsed milliseconds and one call into ``name``.

        Args:
            name: Counter prefix; ``<name>_ms`` and ``<name>_count`` are updated.
        """
        started = monotonic_now()
        self.counter(name + "_count")
        try:
            yield
        finally:
            self.counter(name + "_ms", (monotonic_now() - started) * 1000.0)

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
                "aggregates": list(self.aggregates.values()),
                "counters": {name: round(value, 3) for name, value in self.counters.items()},
                "truncated": self.truncated}


@contextmanager
def span(name: str, kind: str = "step", **attributes: Any) -> Iterator[dict]:
    """Measure only inside an active operation; ordinary engine calls stay unchanged."""
    recorder = _active.get()
    if recorder is None:
        yield {}
    else:
        with recorder.span(name, kind, **attributes) as value:
            yield value


def current() -> Optional["Timing"]:
    """Return the recorder installed in this context, or ``None`` outside an operation.

    Work moved onto a thread starts with an empty context, so a producer that
    fans out must re-activate the recorder inside the worker to keep its spans
    and counters.
    """
    return _active.get()


def counter(name: str, amount: float = 1.0) -> None:
    """Record a numeric observation only inside an active operation."""
    recorder = _active.get()
    if recorder is not None:
        recorder.counter(name, amount)


@contextmanager
def measure(name: str) -> Iterator[None]:
    """Accumulate ``<name>_ms``/``<name>_count`` only inside an active operation."""
    recorder = _active.get()
    if recorder is None:
        yield
    else:
        with recorder.measure(name):
            yield


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

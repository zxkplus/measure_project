"""In-process operation ledger backed by one replaceable CPU subprocess."""

from __future__ import annotations

import multiprocessing
import queue
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional

OPERATION_CONTRACT_VERSION = "2.0"
TERMINAL_STATES = {"succeeded", "failed", "cancelled", "timed_out"}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _internal_error(message: str) -> Dict[str, Any]:
    return {
        "code": "internal.unknown",
        "category": "internal",
        "message": message,
        "retryable": False,
        "details": {},
        "trace_id": uuid.uuid4().hex,
    }


def _cancelled_error() -> Dict[str, Any]:
    return {
        "code": "operation.cancelled",
        "category": "cancelled",
        "message": "Operation was cancelled",
        "retryable": False,
        "details": {},
        "trace_id": uuid.uuid4().hex,
    }


def _timeout_error() -> Dict[str, Any]:
    return {
        "code": "operation.timed_out",
        "category": "timeout",
        "message": "Operation deadline exceeded",
        "retryable": False,
        "details": {},
        "trace_id": uuid.uuid4().hex,
    }


@dataclass
class _OperationRecord:
    operation_id: str
    kind: str
    payload: Dict[str, Any]
    timeout_ms: int
    client_request_id: Optional[str]
    accepted_monotonic: float
    accepted_at: str
    status: str = "queued"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    cancel_requested: bool = False
    completion: threading.Event = field(default_factory=threading.Event)
    on_success: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    on_abort: Optional[Callable[[Dict[str, Any]], None]] = None

    @property
    def deadline(self) -> float:
        return self.accepted_monotonic + self.timeout_ms / 1000.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "contract_version": OPERATION_CONTRACT_VERSION,
            "operation_id": self.operation_id,
            "trace_id": self.operation_id,
            "kind": self.kind,
            "status": self.status,
            "client_request_id": self.client_request_id,
            "accepted_at": self.accepted_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "timeout_ms": self.timeout_ms,
            "result": self.result,
            "error": self.error,
        }


def _process_entry(kind: str, payload: Mapping[str, Any], output: Any) -> None:
    """Run one registered task and return a serializable result to the parent."""
    try:
        from measure_api.v2.application import EngineError
        from measure_api.v2.worker import run_worker_task

        result = run_worker_task(kind, payload)
        output.put(("succeeded", result))
    except EngineError as exc:
        output.put(("failed", exc.error))
    except Exception as exc:  # pragma: no cover - defensive process guard
        output.put(
            (
                "failed",
                {
                    **_internal_error("Engine operation failed"),
                    "details": {
                        "exception_type": type(exc).__name__,
                        "diagnostic": "".join(
                            traceback.format_exception_only(type(exc), exc)
                        ).strip(),
                    },
                },
            )
        )


class OperationManager:
    """Queue operations and execute at most one in a CPU subprocess.

    Args:
        multiprocessing_context: Multiprocessing start method used for workers.
    """

    def __init__(self, *, multiprocessing_context: str = "spawn") -> None:
        self._context: Any = multiprocessing.get_context(multiprocessing_context)
        self._records: Dict[str, _OperationRecord] = {}
        self._pending: Deque[str] = deque()
        self._condition = threading.Condition()
        self._closed = False
        self._current_id: Optional[str] = None
        self._current_process: Optional[multiprocessing.Process] = None
        self._monitor = threading.Thread(
            target=self._run_monitor,
            name="v2-operation-monitor",
            daemon=True,
        )
        self._monitor.start()

    def submit(
        self,
        kind: str,
        payload: Mapping[str, Any],
        timeout_ms: int,
        *,
        client_request_id: Optional[str] = None,
        on_success: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        on_abort: Optional[Callable[[Dict[str, Any]], None]] = None,
        accepted_monotonic: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create a non-idempotent operation and enqueue it for execution.

        Args:
            kind: Registered worker task kind.
            payload: Pickle-safe task input.
            timeout_ms: Operation deadline measured from submission.
            client_request_id: Optional caller correlation identifier.
            on_success: Optional parent-process result transformation callback.
            on_abort: Optional rollback for a transformed result that misses its deadline.
            accepted_monotonic: Request acceptance time used to calculate the deadline.

        Returns:
            The newly submitted operation record.
        """
        if client_request_id is not None and not isinstance(client_request_id, str):
            from measure_api.v2.application import EngineError

            raise EngineError(
                "request.client_request_id_invalid",
                "request",
                "client_request_id must be a string",
            )
        operation_id = uuid.uuid4().hex
        submitted_monotonic = time.monotonic()
        if accepted_monotonic is None:
            accepted_monotonic = submitted_monotonic
        accepted_monotonic = min(accepted_monotonic, submitted_monotonic)
        accepted_at = datetime.now(timezone.utc) - timedelta(
            seconds=max(0.0, submitted_monotonic - accepted_monotonic)
        )
        record = _OperationRecord(
            operation_id=operation_id,
            kind=kind,
            payload={**payload, "operation_id": operation_id},
            timeout_ms=timeout_ms,
            client_request_id=client_request_id,
            accepted_monotonic=accepted_monotonic,
            accepted_at=accepted_at.isoformat(),
            on_success=on_success,
            on_abort=on_abort,
        )
        with self._condition:
            if self._closed:
                from measure_api.v2.application import EngineError

                raise EngineError(
                    "operation.manager_unavailable",
                    "operation",
                    "Operation manager is unavailable",
                    retryable=True,
                )
            self._records[operation_id] = record
            self._pending.append(operation_id)
            self._condition.notify_all()
        return record.as_dict()

    def get(self, operation_id: str) -> Dict[str, Any]:
        """Return one operation from the current process ledger.

        Args:
            operation_id: Server-generated operation identifier.

        Returns:
            The current operation record.
        """
        with self._condition:
            return self._record(operation_id).as_dict()

    def list(self, client_request_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List current-process operations, optionally by client request ID.

        Args:
            client_request_id: Optional caller correlation identifier filter.

        Returns:
            Current-process operation records matching the filter.
        """
        with self._condition:
            records = list(self._records.values())
            if client_request_id is not None:
                records = [
                    record
                    for record in records
                    if record.client_request_id == client_request_id
                ]
            return [record.as_dict() for record in records]

    def wait(self, operation_id: str, wait_ms: int) -> Dict[str, Any]:
        """Wait up to the caller's window without changing operation lifetime.

        Args:
            operation_id: Server-generated operation identifier.
            wait_ms: Maximum caller wait window in milliseconds.

        Returns:
            The operation record after completion or expiration of the wait window.
        """
        if wait_ms < 0:
            from measure_api.v2.application import EngineError

            raise EngineError("request.wait_invalid", "request", "wait_ms cannot be negative")
        with self._condition:
            record = self._record(operation_id)
            completion = record.completion
        completion.wait(wait_ms / 1000.0)
        return self.get(operation_id)

    def cancel(self, operation_id: str) -> Dict[str, Any]:
        """Request hard cancellation of a queued or running operation.

        Args:
            operation_id: Server-generated operation identifier.

        Returns:
            The operation record after recording the cancellation request.
        """
        with self._condition:
            record = self._record(operation_id)
            if record.status not in TERMINAL_STATES:
                record.cancel_requested = True
                self._condition.notify_all()
            return record.as_dict()

    def close(self) -> None:
        """Stop the manager and terminate any active CPU subprocess."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._monitor.join(timeout=2.0)
        process = self._current_process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=1.0)

    def _record(self, operation_id: str) -> _OperationRecord:
        record = self._records.get(operation_id)
        if record is None:
            from measure_api.v2.application import EngineError

            raise EngineError("operation.not_found", "operation", "Operation not found")
        return record

    def _run_monitor(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait(timeout=0.1)
                if self._closed:
                    return
                operation_id = self._pending.popleft()
                record = self._records[operation_id]

            if record.cancel_requested:
                self._complete(record, "cancelled", error=_cancelled_error())
                continue
            if time.monotonic() >= record.deadline:
                self._complete(record, "timed_out", error=_timeout_error())
                continue
            self._execute(record)

    def _execute(self, record: _OperationRecord) -> None:
        output = self._context.Queue(maxsize=1)
        process = self._context.Process(
            target=_process_entry,
            args=(record.kind, record.payload, output),
            daemon=True,
        )
        with self._condition:
            record.status = "running"
            record.started_at = _iso_now()
            self._current_id = record.operation_id
            self._current_process = process
        try:
            process.start()
        except Exception:
            self._complete(record, "failed", error=_internal_error("CPU worker failed to start"))
            self._clear_current()
            return

        response: Optional[Any] = None
        while process.is_alive():
            if record.cancel_requested:
                process.terminate()
                process.join(timeout=1.0)
                self._complete(record, "cancelled", error=_cancelled_error())
                self._clear_current()
                return
            if time.monotonic() >= record.deadline:
                process.terminate()
                process.join(timeout=1.0)
                self._complete(record, "timed_out", error=_timeout_error())
                self._clear_current()
                return
            try:
                response = output.get(timeout=0.01)
                break
            except queue.Empty:
                continue

        process.join(timeout=1.0)
        if response is None and time.monotonic() >= record.deadline:
            self._complete(record, "timed_out", error=_timeout_error())
            self._clear_current()
            return
        if response is None:
            try:
                response = output.get_nowait()
            except queue.Empty:
                response = None
        if response is None:
            self._complete(record, "failed", error=_internal_error("CPU worker exited"))
        elif response[0] == "failed":
            self._complete(record, "failed", error=response[1])
        else:
            result = response[1]
            if record.cancel_requested:
                self._complete(record, "cancelled", error=_cancelled_error())
                self._clear_current()
                return
            if time.monotonic() >= record.deadline:
                self._complete(record, "timed_out", error=_timeout_error())
                self._clear_current()
                return
            if record.on_success is not None:
                try:
                    result = record.on_success(result)
                except Exception as exc:
                    from measure_api.v2.application import EngineError

                    if not isinstance(exc, EngineError):
                        self._complete(
                            record,
                            "failed",
                            error=_internal_error("Operation completion handler failed"),
                        )
                        self._clear_current()
                        return
                    self._complete(record, "failed", error=exc.error)
                    self._clear_current()
                    return
            if record.cancel_requested or time.monotonic() >= record.deadline:
                if record.on_abort is not None:
                    try:
                        record.on_abort(result)
                    except Exception:
                        self._complete(
                            record,
                            "failed",
                            error=_internal_error("Operation rollback failed"),
                        )
                        self._clear_current()
                        return
                if record.cancel_requested:
                    self._complete(record, "cancelled", error=_cancelled_error())
                else:
                    self._complete(record, "timed_out", error=_timeout_error())
                self._clear_current()
                return
            self._complete(record, "succeeded", result=result)
        self._clear_current()

    def _complete(
        self,
        record: _OperationRecord,
        status: str,
        *,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._condition:
            if error is not None:
                error = dict(error)
                error["trace_id"] = record.operation_id
            record.status = status
            record.result = result
            record.error = error
            record.completed_at = _iso_now()
            record.completion.set()
            self._condition.notify_all()

    def _clear_current(self) -> None:
        with self._condition:
            self._current_id = None
            self._current_process = None

"""Hard-cancellable isolated jobs with retained, journalled public operation state."""

from __future__ import annotations

import asyncio
import copy
import ctypes
import fcntl
import sys
import threading
import hashlib
import json
import logging
import multiprocessing
from multiprocessing.connection import Connection
import os
import tempfile
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from measure_api.v2.errors import ServiceError
from measure_api.v2.files import SharedFiles, filesystem_lock, atomic_json
from measure_api.v2.models import CalculationRequest, ServiceSettings
from measure_api.v2.registry import CalculationContext, operations
from measure_api.v2.result_diagnostics import operation_diagnostics
from measure_api.v2.timing import monotonic_now

TERMINAL_STATES = {"succeeded", "failed", "cancelled", "timed_out"}


def utc_now() -> str:
    """Return an unambiguous operation timestamp.

    Returns:
        Timezone-aware UTC timestamp serialized in ISO 8601 format.
    """
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CommitGuard:
    """Arbitrate cancellation against a manifest commit across processes."""

    lock: Any
    committed: Any
    stopped: Any
    finalize: bool = True
    staging_directory: Optional[str] = None
    deadline: float = float("inf")


def configure_worker(settings: ServiceSettings) -> None:
    """Bound native parallelism and private allocations in a compute process.

    Args:
        settings: Deployment roots and resource limits shared by HTTP and compute workers.
    """
    import resource

    import cv2

    cv2.setNumThreads(1)
    if sys.platform.startswith("linux"):
        # Linux DATA limits heap allocation; OS resident-memory monitoring covers native buffers.
        _, hard = resource.getrlimit(resource.RLIMIT_DATA)
        limit = settings.max_worker_memory_bytes
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_DATA, (limit, hard))


def worker_job(connection: Connection, name: str, payload: dict, settings_payload: dict,
               identity: str, trace_id: str, guard: CommitGuard) -> None:
    """Validate in isolation, then execute only after HTTP admission confirms identity.

    Args:
        connection: Child-side process pipe for prepared-request and final-outcome messages.
        name: Registered calculation identifier selected by HTTP admission.
        payload: JSON-safe request or record to serialize without exposing partial content.
        settings_payload: JSON-safe deployment settings reconstructed in the isolated
            worker.
        identity: Runtime operation identifier used for retained state and owned staging.
        trace_id: Trace identity echoed in the public operation and error envelope.
        guard: Shared deadline, stop flag and death-safe commit lock for publication.
    """
    started = monotonic_now()
    compute_started = None
    parent_pid = os.getppid()
    def stop_on_parent_death():
        while True:
            threading.Event().wait(0.1)
            if os.getppid() != parent_pid:
                os._exit(1)
    threading.Thread(target=stop_on_parent_death, daemon=True).start()
    try:
        settings = ServiceSettings.model_validate(settings_payload)
        configure_worker(settings)
        operation = operations()[name]
        files = SharedFiles(settings, guard)
        request = operation.request_model.model_validate(payload)
        if operation.prepare is not None:
            request = operation.prepare(request, files)
            request = operation.request_model.model_validate(request)
        request = files.snapshot_request(request)
        connection.send({"phase": "prepared", "request": request.model_dump(mode="json"),
                         "preparation_ms": (monotonic_now() - started) * 1000,
                         "worker_started_monotonic": started})
        if connection.recv() != "execute":
            return
        context = CalculationContext(settings, identity, trace_id, guard)
        compute_started = monotonic_now()
        result = operation.result_model.model_validate(operation.execute(request, context))
        outcome = {"ok": True, "result": result.model_dump(mode="json"),
                   "compute_ms": (monotonic_now() - compute_started) * 1000,
                   "cache": context.files.cache_stats}
        context.files.finalize(outcome)
    except ServiceError as error:
        outcome = {"ok": False, "error": error.payload(trace_id), "status": error.status}
    except MemoryError:
        outcome = {"ok": False, "error": ServiceError(
            "operation.memory_limit", "Computation exceeded its memory budget", 429,
            retryable=True).payload(trace_id), "status": 429}
    except BaseException:
        outcome = {"ok": False, "error": ServiceError(
            "calculation.failed", "Calculation failed unexpectedly", 500,
        ).payload(trace_id), "status": 500}
    if compute_started is not None and "compute_ms" not in outcome:
        outcome["compute_ms"] = (monotonic_now() - compute_started) * 1000
    try:
        connection.send({"phase": "finished", **outcome})
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        connection.close()


def resident_bytes(pid: int) -> int:
    """Read OS resident memory without scheduling CPU work on the HTTP event loop.

    Args:
        pid: Compute process identifier whose resident memory is observed by the OS.

    Returns:
        Observed resident bytes, or zero when the OS cannot report process memory.
    """
    if sys.platform.startswith("linux"):
        try:
            pages = int(Path(f"/proc/{pid}/statm").read_text().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            return 0
    if sys.platform == "darwin":
        info = ctypes.create_string_buffer(96)
        library = ctypes.CDLL("/usr/lib/libproc.dylib")
        if library.proc_pidinfo(pid, 4, 0, info, len(info)) == len(info):
            return int.from_bytes(info.raw[8:16], sys.byteorder)
    return 0


class OperationRuntime:
    """One ASGI owner, bounded independent workers, and atomic operational journal."""

    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self.records: Dict[str, dict] = {}
        self.metadata: Dict[str, dict] = {}
        self.idempotency: Dict[tuple, tuple] = {}
        self.expired: Dict[str, float] = {}
        self.expired_traces: Dict[str, float] = {}
        self.jobs: Dict[str, dict] = {}
        self.tasks = set()
        self.ready = False
        self.pending = 0
        self.semaphore = None
        self.mp = multiprocessing.get_context("spawn")
        for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[variable] = "1"
        root_key = hashlib.sha256(json.dumps(settings.model_dump(mode="json")["shared_roots"],
                                             sort_keys=True).encode()).hexdigest()[:20]
        self.directory = settings.state_directory or Path(tempfile.gettempdir()) / (
            "measure-v2-state-" + root_key)
        self.settings = settings.model_copy(update={"state_directory": self.directory})

    async def start(self) -> None:
        """Recover interrupted work before opening admission."""
        self.semaphore = asyncio.Semaphore(self.settings.max_workers)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.owner = (self.directory / ".owner.lock").open("a+b")
        try:
            fcntl.flock(self.owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.owner.close()
            raise ServiceError("service.state_in_use", "Operation state already has an ASGI owner",
                               503) from error
        journal = self.directory / "journal.json"
        if journal.exists():
            try:
                saved = json.loads(journal.read_text())
            except (OSError, ValueError) as error:
                self.owner.close()
                raise ServiceError("service.journal_invalid", "Operational journal is unreadable",
                                   503) from error
            self.records = saved.get("records", {})
            self.metadata = saved.get("metadata", {})
            self.expired = saved.get("expired", {})
            self.expired_traces = saved.get("expired_traces", {})
            for record in self.records.values():
                if record["state"] not in TERMINAL_STATES:
                    recovered = self._completed(record["operation_id"])
                    if recovered is not None:
                        record.update(state="succeeded", result=recovered["result"],
                                      completed_at=utc_now())
                        record["diagnostics"].update(operation_diagnostics(recovered["result"]))
                        for metric in ["compute_ms", "cache"]:
                            if metric in recovered:
                                record["diagnostics"][metric] = recovered[metric]
                        self.metadata[record["operation_id"]]["terminal_at"] = time.time()
                        continue
                    record.update(state="failed", completed_at=utc_now(), error=ServiceError(
                        "worker.interrupted", "Operation interrupted by service restart", 503,
                        retryable=True).payload(record["trace_id"]))
                    self.metadata[record["operation_id"]]["terminal_at"] = time.time()
            for identity, meta in self.metadata.items():
                if meta.get("key"):
                    self.idempotency[tuple(meta["key"])] = (identity, meta["fingerprint"])
        for directory in self.directory.iterdir():
            if directory.is_dir() and len(directory.name) == 32:
                self._cleanup_owned(directory.name)
                shutil.rmtree(directory, ignore_errors=True)
        cache = self.directory / "decoded-cache"
        if cache.is_dir():
            with filesystem_lock(str(cache / ".lock")):
                for partial in cache.glob("*.partial"):
                    partial.unlink(missing_ok=True)
                entries = sorted(cache.glob("*.cache"), key=lambda item: item.stat().st_mtime_ns)
                total = sum(item.stat().st_size for item in entries)
                while entries and (len(entries) > self.settings.cache_max_entries
                                   or total > self.settings.cache_max_bytes):
                    oldest = entries.pop(0)
                    total -= oldest.stat().st_size
                    oldest.unlink()
        self._expire()
        try:
            self._persist()
        except ServiceError:
            self.owner.close()
            raise
        self.ready = True

    def _persist(self) -> None:
        stage = self.directory / "journal.partial"
        try:
            with stage.open("w") as target:
                json.dump({"records": self.records, "metadata": self.metadata,
                           "expired": self.expired, "expired_traces": self.expired_traces},
                          target, allow_nan=False)
                target.flush()
                os.fsync(target.fileno())
            os.replace(stage, self.directory / "journal.json")
        except OSError as error:
            raise ServiceError(
                "service.journal_unavailable", "Operational journal cannot be written", 503,
                retryable=True,
            ) from error

    def _checkpoint(self) -> bool:
        """Keep admitted work observable when its durable journal is temporarily unavailable."""
        try:
            self._persist()
            return True
        except ServiceError:
            logging.getLogger(__name__).warning("Operational journal checkpoint failed")
            return False

    def _expire(self) -> None:
        now = time.time()
        for identity, meta in list(self.metadata.items()):
            if meta.get("terminal_at") is not None and (
                    now - meta["terminal_at"] >= self.settings.operation_retention_seconds):
                record = self.records.pop(identity, None)
                if record is not None:
                    self.expired_traces[record["trace_id"]] = now
                self.metadata.pop(identity, None)
                self.expired[identity] = now
                if meta.get("key"):
                    self.idempotency.pop(tuple(meta["key"]), None)
        for identity, expired_at in list(self.expired.items()):
            if now - expired_at >= self.settings.expired_retention_seconds:
                self.expired.pop(identity)
        for trace_id, expired_at in list(self.expired_traces.items()):
            if now - expired_at >= self.settings.expired_retention_seconds:
                self.expired_traces.pop(trace_id)
        while len(self.expired_traces) > self.settings.max_retained_operations:
            self.expired_traces.pop(next(iter(self.expired_traces)))
        while len(self.expired) > self.settings.max_retained_operations:
            self.expired.pop(next(iter(self.expired)))

    async def close(self) -> None:
        """Stop admission, allow a bounded drain, and terminate every unfinished process."""
        self.ready = False
        if self.tasks:
            await asyncio.wait(self.tasks, timeout=self.settings.shutdown_wait_seconds)
        for identity in list(self.jobs):
            await self._stop(identity, "failed", "worker.interrupted")
        if self.tasks:
            await asyncio.gather(*list(self.tasks), return_exceptions=True)
        try:
            self._checkpoint()
        finally:
            self.owner.close()

    async def wait(self, identity: str, wait_ms: int) -> dict:
        """Poll with a bounded waiting window independent of computation deadline.

        Args:
            identity: Runtime operation identifier used for retained state and owned staging.
            wait_ms: Maximum poll wait; it does not extend the computation deadline.

        Returns:
            Retained operation record after completion or the bounded polling window elapses.
        """
        self.get(identity)
        job = self.jobs.get(identity)
        if job is not None and wait_ms:
            try:
                await asyncio.wait_for(asyncio.shield(job["task"]), wait_ms / 1000)
            except asyncio.TimeoutError:
                pass
        return self.get(identity)

    async def submit(self, name: str, request: CalculationRequest, trace_id: str,
                     caller: str = "anonymous") -> dict:
        """Validate before async acceptance; the budget starts before queue/preparation.

        Args:
            name: Registered calculation identifier defining schema and idempotency scope.
            request: Typed common scheduling controls; waiting never changes the operation
                deadline.
            trace_id: Trace identity echoed in the public operation and error envelope.
            caller: Upstream caller identity scoping idempotency keys for independent clients.

        Returns:
            Accepted/queryable operation record, or the scoped retained idempotent result.
        """
        started = monotonic_now()
        self._expire()
        if not self.ready:
            raise ServiceError("service.unavailable", "Compute workers are unavailable", 503,
                               retryable=True)
        if self.pending >= self.settings.max_pending:
            raise ServiceError("operation.capacity", "Operation admission limit reached", 429,
                               retryable=True)
        retained_retry = (caller, name, request.idempotency_key) in self.idempotency
        if len(self.records) >= self.settings.max_retained_operations and not retained_retry:
            raise ServiceError("operation.retention_capacity", "Retained operation limit reached",
                               429, retryable=True)
        identity = uuid.uuid4().hex
        record = {"operation_id": identity, "operation": name, "state": "queued",
                  "trace_id": trace_id, "result": None, "error": None,
                  "created_at": utc_now(), "completed_at": None,
                  "diagnostics": {"events": [{"state": "queued", "at": utc_now()}],
                                  "deadline_clock": "CLOCK_MONOTONIC",
                                  "admitted_monotonic": started,
                                  "deadline_monotonic": started + request.timeout_ms / 1000}}
        prepared = asyncio.get_running_loop().create_future()
        owned = self.directory / identity
        try:
            owned.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ServiceError("service.state_unavailable", "Operation staging is unavailable",
                               503, retryable=True) from error
        guard = CommitGuard(str(owned / "commit.lock"), self.mp.Value("b", False, lock=False),
                            self.mp.Value("b", False, lock=False),
                            staging_directory=str(owned / "stages"),
                            deadline=started + request.timeout_ms / 1000)
        job = {"prepared": prepared, "guard": guard, "process": None,
               "deadline": started + request.timeout_ms / 1000, "started": started,
               "name": name, "request": request, "caller": caller, "redirect": None}
        self.records[identity] = record
        self.metadata[identity] = {"terminal_at": None}
        self.jobs[identity] = job
        self.pending += 1
        try:
            self._persist()
        except ServiceError:
            self.pending -= 1
            self.jobs.pop(identity, None)
            self.records.pop(identity, None)
            self.metadata.pop(identity, None)
            self._cleanup_owned(identity)
            shutil.rmtree(owned, ignore_errors=True)
            raise
        task = asyncio.create_task(self._run(identity))
        job["task"] = task
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        await asyncio.shield(prepared)
        if job.get("preparation_error"):
            error = job["preparation_error"]
            raise ServiceError(error["code"], error["message"], job["status"],
                               error["details"], error["field_path"], error["retryable"])
        identity = job["redirect"] or identity
        if request.mode == "sync" and request.wait_ms:
            remaining = max(0, started + request.wait_ms / 1000 - monotonic_now())
            active = self.jobs.get(identity)
            if active:
                try:
                    await asyncio.wait_for(asyncio.shield(active["task"]), remaining)
                except asyncio.TimeoutError:
                    pass
        return self.get(identity)

    def _transition(self, identity: str, state: str) -> None:
        record = self.records[identity]
        record["state"] = state
        record["diagnostics"]["events"].append({"state": state, "at": utc_now()})
        durable = self._checkpoint()
        if identity in self.jobs:
            self.jobs[identity]["journal_durable"] = durable

    async def _run(self, identity: str) -> None:
        job = self.jobs[identity]
        record = self.records[identity]
        connection = None
        acquired = False
        try:
            remaining = max(0, job["deadline"] - monotonic_now())
            try:
                await asyncio.wait_for(self.semaphore.acquire(), remaining)
                acquired = True
                record["diagnostics"]["queue_ms"] = (monotonic_now()-job["started"]) * 1000
            except asyncio.TimeoutError:
                await self._stop(identity, "timed_out", "operation.timed_out")
                return
            if record["state"] in TERMINAL_STATES:
                return
            parent, child = self.mp.Pipe()
            connection = parent
            process = self.mp.Process(target=worker_job, args=(
                child, record["operation"], job["request"].model_dump(mode="json"),
                self.settings.model_dump(mode="json"), identity, record["trace_id"], job["guard"]))
            job["process"] = process
            job["process_started"] = monotonic_now()
            process.start()
            child.close()
            memory_checked = 0.0
            while record["state"] not in TERMINAL_STATES:
                if job.get("stopping"):
                    await asyncio.sleep(0.005)
                    continue
                if monotonic_now() - memory_checked >= 0.05:
                    memory_checked = monotonic_now()
                    usage = await asyncio.to_thread(resident_bytes, process.pid)
                    if usage > 0:
                        record["diagnostics"]["peak_worker_bytes"] = max(
                            usage, record["diagnostics"].get("peak_worker_bytes", 0))
                    if usage > self.settings.max_worker_memory_bytes:
                        await self._stop(identity, "failed", "operation.memory_limit")
                        break
                if monotonic_now() >= job["deadline"]:
                    await self._stop(identity, "timed_out", "operation.timed_out")
                    if record["state"] in TERMINAL_STATES:
                        break
                if connection.poll():
                    message = connection.recv()
                    if message["phase"] == "prepared":
                        if not self._admit_prepared(identity, message["request"]):
                            connection.send("discard")
                            break
                        connection.send("execute")
                        record["diagnostics"]["preparation_ms"] = message["preparation_ms"]
                        record["diagnostics"]["worker_started_monotonic"] = (
                            message["worker_started_monotonic"]
                        )
                        record["diagnostics"]["worker_startup_ms"] = (
                            (monotonic_now() - job["process_started"]) * 1000
                            - message["preparation_ms"]
                        )
                        record["diagnostics"]["admission_ms"] = (
                            monotonic_now() - job["started"]) * 1000
                        self._transition(identity, "running")
                        if not job["prepared"].done():
                            job["prepared"].set_result(True)
                    else:
                        recovered = self._completed(identity)
                        if recovered is not None:
                            message = recovered
                        if "compute_ms" in message:
                            record["diagnostics"]["compute_ms"] = message["compute_ms"]
                        if message["ok"]:
                            record["result"] = message["result"]
                            record["diagnostics"]["compute_ms"] = message["compute_ms"]
                            record["diagnostics"]["cache"] = message["cache"]
                            self._finish(identity, "succeeded")
                        else:
                            record["error"] = message["error"]
                            record["diagnostics"]["http_status"] = message["status"]
                            if not job["prepared"].done():
                                job.update(preparation_error=message["error"],
                                           status=message["status"])
                            state = ("timed_out"
                                     if message["error"]["code"] == "operation.timed_out"
                                     else "failed")
                            self._finish(identity, state)
                elif not process.is_alive():
                    recovered = self._completed(identity)
                    if recovered is not None:
                        record["result"] = recovered["result"]
                        record["diagnostics"]["compute_ms"] = recovered["compute_ms"]
                        record["diagnostics"]["cache"] = recovered["cache"]
                        self._finish(identity, "succeeded")
                        break
                    record["diagnostics"]["http_status"] = 503
                    record["error"] = ServiceError("worker.interrupted",
                        "Computation worker was interrupted", 503, retryable=True
                    ).payload(record["trace_id"])
                    self._finish(identity, "failed")
                else:
                    await asyncio.sleep(0.005)
        except ServiceError as error:
            record["error"] = error.payload(record["trace_id"])
            record["diagnostics"]["http_status"] = error.status
            if not job["prepared"].done():
                job.update(preparation_error=record["error"], status=error.status)
            self._finish(identity, "failed")
        except (EOFError, OSError, RuntimeError):
            if record["state"] not in TERMINAL_STATES:
                recovered = self._completed(identity)
                if recovered is not None:
                    record["result"] = recovered["result"]
                    for metric in ["compute_ms", "cache"]:
                        if metric in recovered:
                            record["diagnostics"][metric] = recovered[metric]
                    self._finish(identity, "succeeded")
                    return
                record["diagnostics"]["http_status"] = 503
                record["error"] = ServiceError("worker.interrupted",
                    "Computation worker was interrupted", 503, retryable=True
                ).payload(record["trace_id"])
                self._finish(identity, "failed")
        finally:
            process = job["process"]
            if process is not None:
                if process.is_alive():
                    process.terminate()
                await asyncio.to_thread(process.join)
                process.close()
            if connection is not None:
                connection.close()
            if acquired:
                self.semaphore.release()
            if not job["prepared"].done():
                job["prepared"].set_result(True)
            self._cleanup_owned(identity)
            if job.get("journal_durable", True):
                shutil.rmtree(self.directory / identity, ignore_errors=True)
            self.pending -= 1
            self.jobs.pop(identity, None)

    def _admit_prepared(self, identity: str, payload: dict) -> bool:
        job = self.jobs[identity]
        canonical = dict(payload)
        for control in ("mode", "wait_ms", "idempotency_key"):
            canonical.pop(control, None)
        fingerprint = hashlib.sha256(json.dumps(canonical, sort_keys=True,
                                                 separators=(",", ":")).encode()).hexdigest()
        key = (job["caller"], job["name"], job["request"].idempotency_key)
        if key[2] and key in self.idempotency:
            prior, digest = self.idempotency[key]
            if fingerprint != digest:
                error = ServiceError("operation.idempotency_conflict",
                                     "Idempotency key was used for different content", 409)
                job.update(preparation_error=error.payload(self.records[identity]["trace_id"]),
                           status=409)
            else:
                job["redirect"] = prior
            self.records.pop(identity)
            self.metadata.pop(identity)
            self._checkpoint()
            return False
        if key[2]:
            self.idempotency[key] = (identity, fingerprint)
            self.metadata[identity].update(key=key, fingerprint=fingerprint)
        try:
            self._persist()
        except ServiceError:
            if key[2]:
                self.idempotency.pop(key, None)
                self.metadata[identity] = {"terminal_at": None}
            raise
        return True

    def _finish(self, identity: str, state: str) -> None:
        record = self.records[identity]
        record["diagnostics"].update(
            operation_diagnostics(record.get("result"), record.get("error"))
        )
        record["completed_at"] = utc_now()
        self.metadata[identity]["terminal_at"] = time.time()
        self._transition(identity, state)

    async def _stop(self, identity: str, state: str, code: str) -> None:
        job = self.jobs.get(identity)
        if job is None or self.records[identity]["state"] in TERMINAL_STATES:
            return
        if not job.get("stop_task"):
            job["stopping"] = True
            job["stop_task"] = asyncio.create_task(self._stop_job(identity, state, code))
        await asyncio.shield(job["stop_task"])

    async def _stop_job(self, identity: str, state: str, code: str) -> None:
        """Kill a stalled commit, then inspect the durable completion boundary."""
        job = self.jobs[identity]
        guard = job["guard"]
        def mark_stopped():
            try:
                with filesystem_lock(guard.lock, blocking=False):
                    guard.stopped.value = True
            except BlockingIOError:
                # The worker may be in commit. Terminate first: flock is death-safe.
                pass
        await asyncio.to_thread(mark_stopped)
        process = job["process"]
        if process is not None and process.is_alive():
            process.terminate()
            await asyncio.to_thread(process.join)
        with filesystem_lock(guard.lock):
            recovered = self._completed(identity)
            guard.stopped.value = True
        if recovered is not None:
            record = self.records[identity]
            record.update(result=recovered["result"], error=None)
            record["diagnostics"].update(cache=recovered["cache"],
                                           compute_ms=recovered["compute_ms"])
            self._finish(identity, "succeeded")
            return
        if state == "cancelled":
            self._transition(identity, "cancelling")
        self.records[identity]["diagnostics"]["http_status"] = (
            429 if code == "operation.memory_limit" else 503 if state == "failed" else 409)
        self.records[identity]["error"] = ServiceError(
            code, "Operation stopped before completion", 503 if state == "failed" else 409,
            retryable=state == "failed").payload(self.records[identity]["trace_id"])
        self._finish(identity, state)

    def _completed(self, identity: str) -> Optional[dict]:
        """Recover a committed result even if its worker died before notifying HTTP."""
        directory = self.directory / identity
        try:
            completion = json.loads((directory / "completion.json").read_text())
        except (OSError, ValueError):
            return None
        manifests = completion["manifests"]
        if ((directory / "committed.json").exists()
                or manifests and all(Path(path).is_file() for path in manifests)):
            return completion["outcome"]
        return None

    def _cleanup_owned(self, identity: str) -> None:
        """Delete only runtime-registered namespaces lacking their completion marker."""
        directory = self.directory / identity / "owned-outputs"
        for entry in directory.glob("*.json"):
            try:
                namespace = Path(json.loads(entry.read_text())["path"])
                if not (namespace / "manifest.json").is_file():
                    shutil.rmtree(namespace, ignore_errors=True)
            except (OSError, ValueError, KeyError):
                continue

    def get(self, identity: str) -> dict:
        """Return retained immutable observations or explicit expiration.

        Args:
            identity: Runtime operation identifier used for retained state and owned staging.

        Returns:
            Independent copy of the retained record; missing/expired identities raise typed
            errors.
        """
        self._expire()
        if identity in self.expired:
            raise ServiceError("operation.expired", "Operation retention expired", 410)
        if identity not in self.records:
            raise ServiceError("operation.not_found", "Operation does not exist", 404)
        return copy.deepcopy(self.records[identity])

    def trace(self, trace_id: str) -> list:
        """Retrieve all retained operations sharing an operational trace identity.

        Args:
            trace_id: Trace identity echoed in the public operation and error envelope.

        Returns:
            Retained operation records matching the exact supplied trace identity.
        """
        self._expire()
        records = [copy.deepcopy(item) for item in self.records.values()
                   if item["trace_id"] == trace_id]
        if not records and trace_id in self.expired_traces:
            raise ServiceError("trace.expired", "Trace retention expired", 410)
        return records

    def list(self, state: Optional[str], limit: int) -> list:
        """List bounded retained operations newest first.

        Args:
            state: Optional public operation-state filter for retained records.
            limit: Maximum number of retained observations to return.

        Returns:
            Newest retained operation records matching the state filter, capped at limit.
        """
        self._expire()
        if state is not None and state not in TERMINAL_STATES | {"queued", "running", "cancelling"}:
            raise ServiceError("operation.invalid_state", "Unknown operation state")
        return [copy.deepcopy(item) for item in reversed(list(self.records.values()))
                if state is None or item["state"] == state][:limit]

    async def cancel(self, identity: str) -> dict:
        """Hard-stop uncommitted work; a completed commit wins the race.

        Args:
            identity: Runtime operation identifier used for retained state and owned staging.

        Returns:
            Current terminal record; an already committed output remains succeeded.
        """
        self.get(identity)
        await self._stop(identity, "cancelled", "operation.cancelled")
        return self.get(identity)

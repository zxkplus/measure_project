"""A bounded prewarmed process pool with individually replaceable workers."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from measure_api.v2.errors import ServiceError
from measure_api.v2.timing import monotonic_now


async def terminate_process(process: Any) -> None:
    """Escalate a hard stop to SIGKILL when the process ignores termination."""
    if process.is_alive():
        process.terminate()
    await asyncio.to_thread(process.join, 1.0)
    if process.is_alive():
        process.kill()
        await asyncio.to_thread(process.join)


@dataclass(eq=False)
class WarmWorker:
    """One exclusively leased compute process and its private command pipe."""

    process: Any
    connection: Any
    started: float
    ready: bool = False
    busy: bool = False
    uses: int = 0
    boot_spawn_ms: float = 0.0
    boot_setup_ms: float = 0.0
    disposal: Optional[asyncio.Task] = None


class WorkerPool:
    """Keep configured workers warm; an active job owns exactly one lease."""

    def __init__(self, context: Any, size: int, target: Callable, settings: dict) -> None:
        self.context, self.size = context, size
        self.target, self.settings = target, settings
        self.workers = []
        self.condition = asyncio.Condition()
        self.closed = False
        self.monitor: Optional[asyncio.Task] = None
        self.cursor = 0

    @property
    def ready_count(self) -> int:
        """Count warmed live processes, including exclusively leased workers."""
        return sum(worker.ready and worker.process is not None and worker.process.is_alive()
                   for worker in self.workers)

    def _spawn(self) -> WarmWorker:
        parent, child = self.context.Pipe()
        process = self.context.Process(target=self.target, args=(child, self.settings))
        worker = WarmWorker(process, parent, monotonic_now())
        try:
            process.start()
        except BaseException:
            parent.close()
            process.close()
            raise
        finally:
            child.close()
        return worker

    async def _warm(self, worker: WarmWorker) -> None:
        deadline = monotonic_now() + 30
        while monotonic_now() < deadline:
            if worker.connection.poll():
                message = worker.connection.recv()
                if message.get("phase") != "ready":
                    raise RuntimeError("Worker did not finish prewarming")
                worker.boot_spawn_ms = (message["started_monotonic"] - worker.started) * 1000
                worker.boot_setup_ms = message["setup_ms"]
                worker.ready = True
                async with self.condition:
                    self.condition.notify_all()
                return
            if not worker.process.is_alive():
                raise RuntimeError("Worker exited during prewarming")
            await asyncio.sleep(0.005)
        raise RuntimeError("Worker prewarming exceeded 30 seconds")

    async def start(self) -> None:
        """Prewarm every worker before exposing service readiness."""
        warmups = []
        try:
            for _ in range(self.size):
                self.workers.append(self._spawn())
            warmups = [asyncio.create_task(self._warm(worker)) for worker in self.workers]
            await asyncio.gather(*warmups)
        except BaseException as error:
            for task in warmups:
                task.cancel()
            await asyncio.gather(*warmups, return_exceptions=True)
            await self.close()
            if isinstance(error, asyncio.CancelledError):
                raise
            raise ServiceError("worker.startup_failed", "Compute workers could not prewarm", 503,
                               retryable=True) from error
        self.monitor = asyncio.create_task(self._monitor())

    async def acquire(self) -> WarmWorker:
        """Wait for one ready process without creating an unbounded task queue."""
        async with self.condition:
            while not self.closed:
                for offset in range(len(self.workers)):
                    index = (self.cursor + offset) % len(self.workers)
                    worker = self.workers[index]
                    if worker.ready and not worker.busy and worker.process.is_alive():
                        worker.busy = True
                        self.cursor = (index + 1) % len(self.workers)
                        return worker
                await self.condition.wait()
        raise ServiceError("service.unavailable", "Worker pool is stopping", 503, retryable=True)

    async def _dispose(self, worker: WarmWorker) -> None:
        if worker.disposal is None:
            worker.disposal = asyncio.create_task(self._dispose_process(worker))
        await asyncio.shield(worker.disposal)

    async def _dispose_process(self, worker: WarmWorker) -> None:
        worker.ready = False
        process = worker.process
        if process is not None:
            await terminate_process(process)
            process.close()
            worker.process = None
        worker.connection.close()

    async def release(self, worker: WarmWorker, identity: str, dispatched: bool,
                      memory_usage: Callable[[int], int], memory_limit: int) -> None:
        """Wait for cleanup acknowledgement before allowing another job to reuse the process."""
        reusable = not dispatched
        if dispatched and worker.process is not None and worker.process.is_alive():
            deadline = monotonic_now() + 5
            while monotonic_now() < deadline and worker.process.is_alive():
                if worker.connection.poll():
                    try:
                        message = worker.connection.recv()
                        reusable = (message.get("phase") == "idle"
                                    and message.get("operation_id") == identity)
                    except (EOFError, OSError):
                        pass
                    break
                await asyncio.sleep(0.005)
        if reusable and worker.process is not None and worker.process.is_alive():
            usage = await asyncio.to_thread(memory_usage, worker.process.pid)
            reusable = usage <= memory_limit
        if not reusable:
            await self._dispose(worker)
        worker.busy = False
        if dispatched:
            worker.uses += 1
        async with self.condition:
            self.condition.notify_all()

    async def _monitor(self) -> None:
        try:
            while not self.closed:
                for index, worker in enumerate(self.workers):
                    if worker.busy:
                        continue
                    if worker.process is None or not worker.process.is_alive():
                        await self._dispose(worker)
                        if self.closed:
                            return
                        try:
                            replacement = self._spawn()
                        except (OSError, RuntimeError):
                            logging.getLogger(__name__).exception(
                                "Worker replacement could not start")
                            await asyncio.sleep(0.25)
                            continue
                        self.workers[index] = replacement
                        try:
                            await self._warm(replacement)
                        except (EOFError, OSError, RuntimeError):
                            await self._dispose(replacement)
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            pass

    async def stop_replacing(self) -> None:
        """Stop creating replacement workers while accepted jobs drain at shutdown."""
        if self.monitor is not None:
            self.monitor.cancel()
            await self.monitor
            self.monitor = None

    async def close(self) -> None:
        """Stop replacement and reap all processes and pipe descriptors."""
        self.closed = True
        async with self.condition:
            self.condition.notify_all()
        await self.stop_replacing()
        await asyncio.gather(*(self._dispose(worker) for worker in self.workers))

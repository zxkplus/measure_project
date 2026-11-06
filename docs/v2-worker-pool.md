# Prewarmed V2 compute workers

The service keeps `max_workers` independent spawned processes warm. The default and example
configuration use five workers. One ASGI owner retains operation state and owns the pool; do not
increase the Uvicorn worker count. Readiness is exposed after every configured worker has loaded
its libraries and operation registry.

Each worker executes one operation at a time. A task has a fresh request, `CalculationContext`,
read-only image cache, output namespace, deadline and filesystem commit guard. After reporting its
outcome the worker drops task references, collects cyclic garbage and acknowledges `idle` with
that operation ID. The owner does not reuse the worker until this acknowledgement arrives.
The existing `max_pending` bound includes preparation and queued requests; the deadline still
starts at the HTTP admission boundary and includes waiting for a replacement worker.

Cancellation and deadlines terminate only the assigned worker. A worker that ignores SIGTERM is
killed after one second. Its replacement is prewarmed before reuse; other busy workers continue.
Queued cancellation releases the waiter without waiting for a busy worker to finish. Idle worker
crashes are also detected and replaced. Unexpected Python failures and memory errors retire the
worker; ordinary typed input or calculation errors can reuse it after cleanup.

Operation flags use private files under the existing task directory, protected by the same
filesystem commit lock. They are never reset for another task. This avoids passing process-start
only shared-memory objects over a persistent command pipe. Completion/manifest publication remains
the durable boundary: an already committed outcome wins cancellation, and restart recovery still
uses committed artifacts. The state database and journal export format are unchanged.

The parent-death monitor starts once per worker. Shutdown stops admission and replacement before
allowing the existing bounded drain; all remaining workers and pipe descriptors are then reaped.
Startup failure reaps partially warmed processes and releases the state owner lock.

## Memory and deployment

The default/example retains the 2 GiB per-worker budget. The example Compose memory limit changes
from 6 GiB to 12 GiB for five active workers plus the API owner. A container memory limit does not
reserve physical RAM; verify available host memory and existing services when applying it.
Native OpenCV/BLAS parallelism remains limited to one thread per worker. RSS is checked during
execution and before reuse; an over-budget worker is retired. Task-local image caches are dropped
between operations rather than accumulating a cross-request pixel cache.

The algorithm pool can handle five tasks concurrently. A business caller may retain a lower
concurrency limit; the measurement service's independent two-job scheduler does not need to change.

## Timing version 3

`diagnostics.timing_version` is 3 for pooled operations. Existing retained version-2 records are
left unchanged.

- `worker_spawn_ms` is zero for a task assigned to a prewarmed process.
- `worker_boot_spawn_ms` and `worker_boot_setup_ms` describe that worker's original warmup. They
  are not additional per-operation latency.
- `worker_checkout_ms` is the pool lease wait; `semaphore_wait_ms` retains the same wait observation.
- `worker_dispatch_ms` measures assignment to worker task entry. `worker_setup_ms` now covers only
  inexpensive per-task configuration with already imported modules.
- `worker_pid`, `worker_use_count`, `worker_reused` and `worker_python_threads` expose reuse and
  help detect accidental process or watchdog-thread growth.
- Input preparation, computation, persistence and cache metrics retain their existing meanings.

Use cold service startup, first-input cache creation and repeated warm-input requests as separate
measurements. A killed worker's next task may wait for replacement startup; do not report that
wait as ordinary hot dispatch latency.

## Acceptance

`tests/v2/test_worker_pool.py` covers 100 alternating-image tasks in the same process, stable Python
thread count, idle worker crash/replacement, five simultaneously blocked workers, queue cancellation,
single-worker hard cancellation, hard timeout of a SIGSTOP worker, and failed-startup cleanup.
`tests/v2/test_operation_lifecycle.py` retains filesystem commit-race, state-store outage, process/
service crash, retry, expiry and memory-limit checks and verifies unexpected task failures recycle
the worker. The full V2 suite also exercises sequential different recipes and asset families.

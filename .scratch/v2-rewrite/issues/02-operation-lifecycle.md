# 02: Make computations bounded, cancellable and retryable

**What to build:** An upstream caller can submit a long-running image computation, stop waiting
without losing it, query or cancel it, retry safely within retention, and distinguish interruption,
timeout and completed output. The same execution policy applies to every capability added later.

**Blocked by:** 01: Process shared images through the new FastAPI service.

**Status:** resolved

**Spec user stories:** 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92.

- [x] Deliver operation listing/query/cancellation, retained diagnostic/trace retrieval, and stable
  queued/running/cancelling/succeeded/failed/cancelled/timed_out transitions.

- [x] Distinguish operation timeout_ms from synchronous/poll wait_ms; queueing, input preparation,
  process startup and calculation consume the operation budget, and sync wait expiry returns a
  queryable operation.

- [x] Enforce configured queue, concurrent process and memory limits; reject excess work explicitly
  and keep health and operation HTTP requests responsive.

- [x] Validate required file references before accepting async work; snapshot configuration and
  privately decode/copy or pin verified immutable inputs so queued work cannot silently read
  rewritten assets.

- [x] Cancel or time out before commit without exposing a complete manifest; completed commits
  remain succeeded, with isolated retry namespaces and explicit output conflicts.

- [x] Deduplicate retained retries by caller, route, canonical parameters and verified input
  digests; a changed request under the same key conflicts and retention expiry does not promise
  permanent exactly-once behavior.

- [x] Bound decoded/prepared caches by verified content, algorithm version and relevant parameters;
  cache hits, misses and eviction cannot change observations or retain mutable prior-run results.

- [x] Handle startup, shutdown and process interruption explicitly; report interrupted jobs after
  restart, document operational retention/expiration, and preserve upstream-owned inputs and
  completed outputs.

- [x] Verify all lifecycle behaviors through real public requests using the image slice, including
  sync/async equivalence, restart, failure before manifest commit and deterministic
  cancellation/commit ordering.

## Answer

Implemented retained listing/query/cancel/diagnostic/trace endpoints and durable operational
journalling. Independent bounded processes enforce admission, worker memory and operation budgets
that include queueing, input preparation and startup. Required typed references are verified before
asynchronous acceptance. Caller/route/content-scoped retries use verified digests and explicit
retention/expiration outcomes; decoded caches are bounded, reverify inputs and return private data.

Manifest publication follows durable validated-result preparation and a death-safe commit gate.
Cancellation/deadline before publication stops the process and cleans owned incomplete output;
a published manifest preserves success even when its worker dies before reporting. Restart recovers
unfinished work as interruption and retains completed results without owning upstream artifacts.
The public Operation.prepare and context.stage hooks support transitive recipe inputs and temporary
compound calculations. Temporary outputs cannot mark the parent complete and expire at termination.

Verification: all 80 merged public HTTP tests passed on 2026-10-05, including real Uvicorn processes,
filesystem-gated cancellation/timeout, SIGKILL worker/service interruption and restart, idempotency,
retention, cache eviction, queue/memory bounds and unchanged capability-family calculations. The
implementer also reported 25 lifecycle/image tests passed under Python 3.9 after the lifespan-loop
semaphore correction. Broad legacy numerical regressions were not repeated for this service change.
No top-level README case-collision difference or user-owned image artifact was staged.

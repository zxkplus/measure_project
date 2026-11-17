# 02: Make computations bounded, cancellable and retryable

**What to build:** An upstream caller can submit a long-running image computation, stop waiting
without losing it, query or cancel it, retry safely within retention, and distinguish interruption,
timeout and completed output. The same execution policy applies to every capability added later.

**Blocked by:** 01: Process shared images through the new FastAPI service.

**Status:** needs-triage

**Spec user stories:** 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92.

- [ ] Deliver operation listing/query/cancellation, retained diagnostic/trace retrieval, and stable
  queued/running/cancelling/succeeded/failed/cancelled/timed_out transitions.

- [ ] Distinguish operation timeout_ms from synchronous/poll wait_ms; queueing, input preparation,
  process startup and calculation consume the operation budget, and sync wait expiry returns a
  queryable operation.

- [ ] Enforce configured queue, concurrent process and memory limits; reject excess work explicitly
  and keep health and operation HTTP requests responsive.

- [ ] Validate required file references before accepting async work; snapshot configuration and
  privately decode/copy or pin verified immutable inputs so queued work cannot silently read
  rewritten assets.

- [ ] Cancel or time out before commit without exposing a complete manifest; completed commits
  remain succeeded, with isolated retry namespaces and explicit output conflicts.

- [ ] Deduplicate retained retries by caller, route, canonical parameters and verified input
  digests; a changed request under the same key conflicts and retention expiry does not promise
  permanent exactly-once behavior.

- [ ] Bound decoded/prepared caches by verified content, algorithm version and relevant parameters;
  cache hits, misses and eviction cannot change observations or retain mutable prior-run results.

- [ ] Handle startup, shutdown and process interruption explicitly; report interrupted jobs after
  restart, document operational retention/expiration, and preserve upstream-owned inputs and
  completed outputs.

- [ ] Verify all lifecycle behaviors through real public requests using the image slice, including
  sync/async equivalence, restart, failure before manifest commit and deterministic
  cancellation/commit ordering.


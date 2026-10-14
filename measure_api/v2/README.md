# Stateless V2 computation service

This replacement service uses FastAPI and upstream-owned shared files. Start it with
`python -m measure_api.v2 --config deploy/v2-settings.json`. The old V2 engine and protocol are retired.
See [deployment guidance](../../docs/v2-production-engine.md) for mounts, permissions and limits,
and [the supported contract](../../docs/v2-supported-contract.md) for all families and applicability.
Upstream integrators should start with [the integration API guide](../../docs/v2-upstream-api.md),
then use [the full field reference](../../docs/v2-api-reference.md) and
[OpenAPI snapshot](../../docs/v2-openapi.json) for implementation and client generation.

## Runtime

Python 3.9 or later is supported. Install the fully resolved headless runtime using
`uv pip install -r measure_api/v2/requirements.lock` (or the equivalent pip command).
The lock is resolved for Python 3.9; the main project metadata also pins FastAPI, Pydantic
and Uvicorn to these compatible versions. Development HTTP tests require httpx and pytest.

Set `MEASURE_V2_SHARED_ROOT` to an existing readable/writable shared directory and run:

`uvicorn measure_api.v2.app:create_app --factory --host 0.0.0.0 --port 9004 --workers 1`

For multiple aliases, read-only roots and limits, set `MEASURE_V2_CONFIG` to a JSON `ServiceSettings`
file or pass settings to the public `create_app` factory when embedding. Only one ASGI process owns
the operation state. Calculations and input hashing run in bounded, independently cancellable
spawn-based processes. Health and capability endpoints use the
`/api/v2` prefix, as do `/docs` and `/openapi.json`.

## Caller contract

- A file reference is either `{root: "exchange", path: "images/input.png"}` or
  `{path: "/shared/images/input.png"}`. Absolute paths must still lie inside configured roots.
  An optional `sha256` verifies the exact expected content. The two pathname forms cannot mix.
- Upstream must atomically commit input files and retain them unchanged until terminal operation
  state. References are hashed before acceptance and verified again on use. V2 never deletes
  input assets.
- Every operation request has `mode` (`sync` or `async`), `wait_ms`, `timeout_ms` and optional
  `idempotency_key`. The operation budget includes admission, queueing, preparation and process startup.
- HTTP 200 returns a succeeded operation with a typed `result`. Async work and synchronous wait
  expiry return HTTP 202 plus `Location` and a queryable `operation_id`. Both use exactly the same
  calculation. Query `/operations/{operation_id}` for the retained state and result.
- Errors include `code`, `message`, `details`, `field_path`, `trace_id` and `retryable`. Terminal
  calculation failures also return the operation identity. `X-Trace-Id` correlates HTTP and worker
  results. Malformed JSON is 400, file/job absence 404, conflicts 409, validation 422, bounded
  admission 429 and unavailable readiness 503.
- Image-producing calls require `output.directory`, a directory file reference. Each accepted
  operation owns a new isolated directory beneath it. Files are staged on that filesystem,
  linked atomically without overwrite, and described by SHA-256, size, kind and format. A final
  `manifest.json` is committed last and is the multi-file completion marker. Upstream owns
  committed artifacts and their retention.
- Public points use x right/y down; pixel-center origin is top left. Angles are degrees, clockwise
  positive. Crop width/height are full output pixel dimensions. Mapping matrices explicitly go
  from source to destination and include the inverse; resize mapping accounts for pixel centers.
- PNG/JPEG GRAY8 and BGR8 pixels are accepted; four-channel and 16-bit pixels are rejected.
  Numeric preprocessing is a non-pickle NPY artifact; its separate PNG preview is display data.

## Operation extension seam

Add a module under `measure_api.v2.families` defining its own `OPERATIONS` list. Each entry is
an `Operation` with a public path, a `CalculationRequest` subclass, a `StrictModel` result,
`execute(request, context)`, summary, family and named output-port declarations. Discovery,
OpenAPI routing and worker dispatch share this registry; no central route edit is required.
The registry is immutable for each running process; adding a family requires service restart.

Keep family execution independent of HTTP. A recipe runner can find an `Operation` and directly
invoke its `execute` function with validated input and a `CalculationContext`. The context has
settings, operation identity, trace identity and `files` access. Use typed `FileReference` fields
for every input so recursive preparation pins digests; plain dictionaries cannot imply file reads.
Use `OutputSpec` for output directories so they are authorized without reading them as inputs.
`SharedFiles.read_bytes` and `read_image` reverify identity on each read. Nested typed models,
list values and dictionary values are traversed when preparing the request.

Use `ArtifactWriter(context.files, output, context.operation_id).commit(payloads)` to commit
`(name, kind, format, bytes)` payloads and obtain `(artifacts, manifest)` descriptors. One writer
owns one operation namespace; aggregate a multi-file result into a single commit. No call may
write directly into an input asset or a previous completed namespace. Final manifests are deferred until the worker validates the complete result and
commits its durable completion record; cancellation and deadlines gate this final publication.

Preprocessing definitions in `measure_api.v2.preprocessing` are a discriminated public model
shared by teaching and matching. Apply them once to source pixels, preserving floating-point
numeric data; `display_preview` is only for visualization.

## Complete contract acceptance

Run `python -m pytest tests/v2` for the public HTTP integration suite. It exercises every capability
family, route/OpenAPI discovery, real shared-file assets, bounded lifecycle/crash recovery, numerical
measurement validity, template/calibration/recipe composition, exported diagnostics, portable
packages and frozen replay. The installed CLI additionally runs through a real HTTP socket with
GUI/Flask/Waitress/matplotlib import sentinels and the locked-only runtime. Existing numerical and
retained V1 tests remain separate regression suites. Concrete evidence and external production
qualification gates are recorded in [the acceptance report](../../docs/v2-acceptance.md).

## Image contract acceptance

`python -m pytest tests/v2/test_image_service.py` exercises real shared PNG inputs and HTTP:
inspection, OpenAPI/discovery, all five preprocessing families, preview, rotated crop and inverse
mapping, alias/absolute paths, symlink escape, digest mismatch, input preservation, output
manifests, asynchronous query and HTTP health during large-image computation. Existing algorithm and retained V1 regressions remain separate from the retired V2 protocol.


## Operation lifecycle

The same independent process performs preparation and calculation. `max_workers` bounds concurrent
processes and `max_pending` bounds all admitted work, including preparation and queueing. CPU work
is hard-stopped on cancellation or deadline; `wait_ms` only stops HTTP waiting. Required references
are verified before async acceptance. Upstream must keep accepted inputs immutable through terminal
state; content changes are detected rather than silently replacing a queued asset.

- `GET /operations` lists retained records, newest first; optional `state` and bounded `limit` filter it.
- `GET /operations/{id}?wait_ms=...` optionally waits without extending the operation budget.
- `POST /operations/{id}:cancel` stops uncommitted work. A committed result remains succeeded.
- `GET /operations/{id}/diagnostics` returns timings, transitions, peak process bytes and cache counts.
- `GET /traces/{trace_id}` retrieves all retained operations with that trace identity.

`X-Caller-Id` namespaces idempotency keys (default `anonymous`). It is an integration identity, not an
authentication mechanism. Keys identify caller, operation route, canonical calculation parameters
(including timeout budget but excluding HTTP mode/wait), and verified input content digests. Reusing a
key with changed content returns 409. Operational retention defaults to 24 hours, followed by a
24-hour bounded expiration marker returning 410. After retention, a retry can create new work; this
is not permanent exactly-once delivery. `max_retained_operations` rejects admission when retained
state is full rather than silently evicting an unexpired retry identity.

Set `state_directory` to a service-owned persistent local directory for restart recovery. The default
is a deterministic directory under the OS temporary directory derived from shared root settings;
its availability follows the host's temporary-directory retention. One ASGI owner holds an OS lock.
The journal preserves terminal results and retry identities, and recovers unfinished work as
`worker.interrupted` after restart. Durable validated results are written before the final manifest;
a worker/service interruption after manifest publication recovers success. Graceful shutdown drains
for `shutdown_wait_seconds`, then kills unfinished workers. Parent-death monitoring stops orphaned
workers. Cleanup only removes registered uncommitted namespaces and service-owned stage/cache files;
completed upstream outputs and inputs are never removed.

A failed initial journal write returns retryable 503 `service.journal_unavailable`, rolls back
admission/staging and releases capacity. A failed preparation checkpoint also allows the same
idempotency key to be retried after storage recovers. Journal outages during admitted work are
logged while its current state remains queryable. If completion cannot be journaled, the durable
completion boundary stays available for restart recovery; committed output remains successful
during cancellation and shutdown. New admission requires a working journal.

`max_worker_memory_bytes` defaults to 2 GiB per process. Native CV/BLAS thread counts are one. Linux
heap allocation is additionally constrained through RLIMIT_DATA; both Linux and macOS monitor OS
resident memory and terminate over-budget workers. Encoded image dimensions are checked before
OpenCV allocation, with independent input-byte and pixel quotas. The resident monitor samples every
50 ms; deployment cgroup memory limits provide a hard host-level ceiling for transient native peaks.
Decoded caches default to 256 MiB and 64 entries, use verified content plus decoder/algorithm version,
return private pixel copies, and reverify the source on every access. Cache quota zero disables reuse;
cache corruption, eviction and restart do not change calculation observations.

## Preparation and temporary stage hooks

An operation can supply `prepare(request, files) -> request`. This runs inside the bounded preparation
process before HTTP acceptance and before the generic typed-reference snapshot. Use it to resolve
shared recipe/replay JSON, validate transitive required references, and inject verified digests into
the returned typed configuration. All dependency digests must remain in that configuration so they
participate in idempotency. The request model validates the hook's result before execution.

`context.stage(stage_id)` returns `(child_context, temporary_output)`. Invoke ordinary family
calculations with this context when correction/stitch/alignment must produce a temporary intermediate
file. Child writers inherit cancellation and deadline policy but cannot finalize the parent operation.
Stage identifiers must be unique within one computation. These descriptors expire at termination;
copy required payloads into one parent `ArtifactWriter.commit` before returning the final result.
The runtime deletes all temporary stage outputs on completion, failure, cancellation or restart.
Ordinary families should not access HTTP state or scheduler implementation details.

Lifecycle acceptance is `python -m pytest tests/v2/test_operation_lifecycle.py`. It includes real
HTTP server processes with explicitly gated OS file publication, worker/service SIGKILL, and restart,
in addition to synchronous/asynchronous calculation, waiting, queue/memory bounds, verified retry,
cache eviction, retention expiration and input/output ownership tests.

## Supplied recipes

`/recipe-validations`, `/recipe-previews`, `/runs` and `/run-batches` accept explicit upstream
recipe snapshots. The public schema, dependency bindings, physical sample mapping, confined
alignment and result provenance are described in [RECIPES.md](RECIPES.md). Their headless executor
calls registered atomic functions directly and shares the same worker lifecycle as individual
calculations. Complete batches retain per-item failures without stopping unrelated images.

Supplied-result overlays, shared sampling diagnostics, CSV/JSON reporting and operational timing
contracts are documented in [RESULTS.md](RESULTS.md).

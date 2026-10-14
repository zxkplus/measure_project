# FastAPI V2 deployment and upstream integration

The replacement keeps the `/api/v2` prefix and removes the former mutable modeling sessions,
model publication and POSIX shared-memory pool. Upstream owns recipes, templates, calibration
versions and business result archives. V2 computes explicit snapshots and publishes immutable
output files. GUI and V1 entrypoints remain separate.

## Install and start

Install Python 3.9+ and `pip install -r measure_api/v2/requirements.lock`. The runtime uses
headless OpenCV; do not install both `opencv-python` and `opencv-python-headless` in this environment.
No Flask, Waitress, PyQt, Tk or matplotlib is required. Start the service with:

```bash
python -m measure_api.v2 --config deploy/v2-settings.json --host 0.0.0.0 --port 9004
```

The JSON file has the exact `ServiceSettings` fields, including all limits shown in
[the example configuration](../deploy/v2-settings.json). Unknown fields and invalid bounds fail
startup. `MEASURE_V2_CONFIG`, `MEASURE_V2_HOST` and `MEASURE_V2_PORT` supply defaults. With no config,
`MEASURE_V2_SHARED_ROOT` supplies the single alias `shared`. Set a persistent private
`state_directory` for operational restart recovery. `uvicorn measure_api.v2.app:create_app --factory
--workers 1` also reads `MEASURE_V2_CONFIG`.

One ASGI process owns one state directory through an exclusive OS lock. CPU parallelism comes from
`max_workers` independent spawned processes; it is not Uvicorn worker count. This implementation
uses Unix process/file-lock facilities and supports Linux/macOS. Separate state directories isolate
additional service instances. `/health/live` checks HTTP responsiveness; `/health/ready` additionally
checks workers and configured readable/writable roots. OpenAPI and interactive docs are at
`/api/v2/openapi.json` and `/api/v2/docs`.

## Same-node container shared files

Create host directories and grant the service UID/GID read/traverse permissions on inputs and
write/traverse permissions on outputs and private state. Upstream also needs output read permission.
The example Compose defaults to UID/GID 10001, configurable with `MEASURE_V2_UID/GID`.

```bash
export MEASURE_V2_EXCHANGE_DIR=/srv/measure/exchange
export MEASURE_V2_ASSETS_DIR=/srv/measure/assets
export MEASURE_V2_OUTPUTS_DIR=/srv/measure/outputs
export MEASURE_V2_STATE_DIR=/srv/measure/state
mkdir -p "$MEASURE_V2_EXCHANGE_DIR" "$MEASURE_V2_ASSETS_DIR" \
  "$MEASURE_V2_OUTPUTS_DIR" "$MEASURE_V2_STATE_DIR"
# Provision ownership/ACLs for your upstream and configured service UID before starting.
docker compose -f deploy/docker-compose.v2.yaml up --build -d
```

`deploy/Dockerfile.v2` builds only the locked headless runtime. Supply `MEASURE_V2_IMAGE` to name
or use an existing deployment image. Adjust `MEASURE_V2_SETTINGS_FILE` for a node-specific config.
The host exchange directory may be `/workspace/exchange` in upstream and `/shared/exchange` in V2.
Send `{"root":"exchange","path":"images/part.png","sha256":"..."}` over HTTP; the alias resolves
against the V2 mount. An absolute upstream pathname only works if the V2 mount has the same pathname.
Absolute engine paths must still resolve inside configured roots. Assets and exchange are mounted
read-only; only `outputs` is writable. Private state must not be exposed as a shared root.

The service creates a new output namespace for each operation beneath `output.directory`. Verify
artifact hashes and wait for the final `manifest.json` before using a multi-file set. Outputs are
ordinary shared files; no `/dev/shm`, shared JPEG slots or model registry is required.

## HTTP integration and ownership

Atomically publish input images/assets, preferably with expected SHA-256, and retain them unchanged
until the operation is terminal. Recipe/template/calibration versions are explicit upstream values.
The service validates transitive references and pins their hashes before acceptance. It never deletes
upstream inputs. Upstream archives business outputs and owns their cleanup. Service cleanup touches
only its private cache/stages and registered incomplete output namespaces.

Every calculation accepts `mode`, `wait_ms`, `timeout_ms` and optional `idempotency_key`. HTTP 200
returns the completed operation; pending work returns 202 with `Location`. Poll the operation and
read its typed `result`; POST cancellation stops uncommitted CPU work. `wait_ms` is just the HTTP
waiting window. Timeout includes preparation, hashing, admission, queueing and process startup.
Use `X-Caller-Id` to namespace retries; it does not authenticate callers. The same key with changed
verified content/configuration returns 409. Retained results survive restart. Interrupted uncommitted
work becomes `worker.interrupted`; durable completed manifests recover success.

`max_pending` bounds preparation and queued/running work and returns 429 when saturated.
`max_retained_operations` prevents silent eviction of unexpired retry records. Retention expiry
returns 410 until the bounded marker expires, then 404; retry deduplication is not permanent.
`max_input_bytes` and `max_image_pixels` cap decoding. Per-worker memory bounds and one native CV/BLAS
thread per worker limit CPU work; resident memory is sampled, so the Compose container memory limit
provides the additional host-level ceiling. Cache limits bound decoded content and can be set to zero.
Container and host memory must cover all workers, cache and the ASGI owner. Graceful shutdown drains
for `shutdown_wait_seconds` and then stops unfinished processes.

## Runnable client and qualification

[The standard-library HTTP client](../scripts/v2_http_client.py) submits a JSON request and polls
until terminal state. It works without SDK dependencies:

```bash
python scripts/v2_http_client.py --base-url http://127.0.0.1:29004 \
  --route /image-inspections --request request.json
```

[The qualification harness](../scripts/v2_benchmark.py) consumes cases with real recipe/request
snapshots and optional measured ground truth. It records observations, repeatability on unchanged
files, target count, concurrency, wall latency, worker memory, input provenance and runtime version.
Use `--cases` and `--thresholds` to supply your real workload and agreed acceptance thresholds;
the recorded result distinguishes unavailable physical truth from numerical repeatability.

```bash
python scripts/v2_benchmark.py --base-url http://127.0.0.1:29004 \
  --cases qualification-cases.json --repeats 10 --concurrency 2 \
  --thresholds agreed-thresholds.json --inventory deployment-inventory.json \
  --output output/qualification.json
```

The existing four real sample JPEGs have no traceable physical ground truth. Synthetic HTTP
regressions establish numerical/contract behavior; neither those tests nor repeated reads of the
same image establish production dimensional accuracy or capture repeatability. Representative
camera inputs, truth measurements and explicit accuracy/repeatability/target-count/latency/memory/
concurrency thresholds are required before production qualification. Current evidence and route
coverage are recorded in the final acceptance report.

### Qualification case format

A cases file is a JSON array of `{name, route, request, observations, observation_units,
input_provenance, truth?}`. `observations` contains JSON pointers relative to the typed result.
`observation_units` maps each pointer to its unit; optional `truth` maps the same pointer to
`{value, unit, provenance?}` with matching units. Truth must come from independent traceable
measurements. The example [sample cases](../deploy/v2-sample-cases.json) use the root alias `samples`
for `data/sample`; configure that alias explicitly when running locally. These seeds select
visible features and provide no ground truth.

Threshold JSON supports `max_latency_p95_ms`, `max_peak_worker_bytes`, `max_same_file_stddev`,
`max_absolute_error`, `min_target_count` and `min_concurrency`. Numeric thresholds apply to all
selected observations/cases, so use separate case sets for incompatible physical units. Missing
evidence or absent agreed thresholds gives `pending`; only fully observed supplied thresholds
can pass. Capture-to-capture repeatability, detection misses and container peak memory require
a representative capture series and deployment instrumentation. Worker RMS fit residuals are
reported observations, not dimensional errors against truth.

`--inventory` attaches your service hardware, OS/container identity, worker/cache settings and
external container peak-memory observations to the report. The harness records its client platform
separately and does not assume the HTTP client runs on the service node. Full operation diagnostics
retain preparation/queue/compute and stage timings where provided by the service. The first call
of a case may already hit a cache warmed by another case; reported cache hit/miss counts provide
the observed cold/warm evidence rather than assuming the first case call is cold.

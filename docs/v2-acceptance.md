# V2 replacement acceptance evidence

Implementation uses the confirmed public FastAPI HTTP seam with real shared files and the retained
numerical regression suites. Production metrology qualification is separate: representative camera
truth, capture series and explicitly agreed accuracy/latency/memory/concurrency thresholds have not
been supplied. No physical accuracy or production throughput guarantee is inferred from synthetic
fixtures, fit residuals or repeated reads of an unchanged image.

## Contract and upstream workflow

The required-route manifest contains 65 Spec routes. All 65 are implemented; 55 are calculation
operations with public discovery schemas. Final coverage compares required methods and
normalized parameter paths against generated OpenAPI, and compares every advertised calculation
against its executable route and request/result schemas. The route audit covers discovery, files,
atomic probes/fits/geometry, calibration/plane/stereo, supplied recipes, evaluations, exports,
portable packages, replay and operational controls.

The single upstream integration scenario uses separate read-only image/asset aliases and a writable
output alias. It teaches a template, computes and copies an upstream calibration asset, validates and
previews an unsaved recipe, saves an explicit upstream recipe version, and measures two scaled
instances. Actual pixel widths are 20 and 24 px; scalar mapping returns 2.0 and 2.4 mm. Independent
range rules pass the first and fail the second while the operation and observations remain valid.
Sync and async target results agree. Physical vectors map back to source pixels, raster output has
the actual 220x110 dimensions, and NPZ sampling data/CSV/JSON exports retain separate physical units.
Every artifact is checked against its declared size, digest and final manifest. The same scenario
exports an embedded portable recipe and a frozen full run, imports the recipe and measures relocated
assets, then starts a fresh operational state. Its old job returns 404 while archived replay succeeds
with `equivalent: true`, no differences and the same physical measurements/conformity. Replay also
preserves the original input bytes.

## Public test coverage

| Contract family | Acceptance suite |
|---|---|
| Image input, preprocessing, preview, crop | `tests/v2/test_image_service.py` |
| Queue, cancellation, expiry, retries, memory/cache, crash/restart | `tests/v2/test_operation_lifecycle.py` |
| Edge/fit geometry, batches, validity and evaluation | `tests/v2/test_atomic_measurements.py` |
| Templates and target-scoped alignment | `tests/v2/test_template_service.py`, `test_alignment_service.py` |
| Scalar/camera assets and features | `tests/v2/test_calibrations.py` |
| Plane/stereo/distortion-aware stitching/projection | `tests/v2/test_plane_stereo.py` |
| Typed recipe DAG, previews, actual-size mapping, isolated targets/batches | `tests/v2/test_recipes.py` |
| Vector/raster/diagnostic/CSV/JSON supplied-result exports | `tests/v2/test_results.py` |
| Portable packages, legacy conversion and frozen replay | `tests/v2/test_portable_recipes.py` |
| Required routes, OpenAPI and advertised capabilities | `tests/v2/test_contract_coverage.py` |
| Installed CLI, locked-only imports, real HTTP/shared paths/client/harness | `tests/v2/test_deployment.py` |
| Complete upstream teaching-to-investigation scenario | `tests/v2/test_upstream_workflow.py` |

Final post-review commands and results:

| Runtime | Command | Result |
|---|---|---|
| CPython 3.11.15 | `MEASURE_V2_HEADLESS_PYTHON=/tmp/measure-v2-headless-env/bin/python python -m pytest tests/v2 -q` | 169 passed, 127.37s |
| CPython 3.9.6 | `python -m pytest tests/v2 -q` | 169 passed, 165.77s |
| CPython 3.11.15 | `python -m pytest measure_api/tests -o addopts='' -q` | 71 passed, 2 external-model skips, 2.47s |
| CPython 3.9.6 | `python -m pytest measure_api/tests -o addopts='' -q` | 71 passed, 2 external-model skips, 2.54s |

The malicious duplicate-ZIP fixture intentionally emits one stdlib duplicate-name warning. Retained
V1 suites have two pre-existing unregistered integration-marker warnings. These are disclosed and
are not interpreted as failing or silently skipped replacement cases.

The installed CLI test runs a real Uvicorn process and blocks GUI, Flask, Waitress and matplotlib
imports with an import sentinel. The deployment-only interpreter has only `requirements.lock`
installed: no SDK HTTP test or GUI dependencies. A second pathname prefix resolves to the same
shared input bytes; the caller sends only the configured alias and relative pathname. This tests
real filesystem/HTTP exchange across differing paths. It does not substitute for a running Docker
bind-mount check.

Existing numerical regression evidence is 219 passed under Python 3.9 with the missing
`bottleneck_6.bmp` fixture excluded. Retained V1/SDK tests passed 71 with 2 external-model skips under
Python 3.9 and 3.11. The untouched legacy `test_multi_target_workflow.py` has 5 failures and 13 setup
errors from stale constructor expectations; these were reproduced at baseline `3b46f24`. Its GUI
code and protocol expectations are outside the new V2 contract and have not been silently repaired
or counted as passing replacement tests.

## Operational and container checks

The lifecycle suite exercises actual admission saturation, deterministic cancellation before/after
publication, worker/service SIGKILL, restart recovery, deadline budget, cache eviction and immutable
input verification. Final publication is gated by result validation and durable completion. Existing
upstream committed outputs survive service cleanup. Worker cancellation terminates its isolated
process before incomplete owned output cleanup.

Compose configuration validates when the required host directories are supplied. Docker's daemon
socket is absent on this host, so image build, Linux-container startup and actual container mounts
are not claimed as tested. The operator must run the provided Dockerfile/Compose smoke on the target
node. Startup itself is verified natively with the locked headless runtime. One ASGI owner, bounded
CPU workers, root aliases, output permissions, private persistent state and memory/retention/cache
quotas are documented in the deployment guide.

## Real sample workload and qualification status

The four tracked repository images are GRAY8, 2448x2048, with 140–206 KB encoded JPEG content. They
have no traceable physical truth or supplied production calibration. Workload observations are
recorded on this host, using actual HTTP control and shared-file bytes; fixed files are each repeated
three times at concurrency 1 and 2 with two compute workers. There are separate inspection, atomic
line/circle and template-localized recipe cases. Localized sample cases report actual matched target
counts rather than treating an atomic fit as a target detector.

Reports retain input SHA-256, actual request snapshots, runtime identity, deployment inventory,
operation/quality diagnostics, cache hit/miss counts, stage timings, end-to-end latency and measured
peak worker resident bytes. P95 uses nearest rank on three observations per case and is a small
workload smoke statistic, not a production latency bound. The service/container total peak is not
measured on this native run. Fit RMS is quality against selected samples, not physical dimensional
error. Zero same-file standard deviation does not establish capture repeatability.

### Original pre-review workload on 2026-10-06 (Asia/Shanghai)

Measured host: Apple M5, 16 GiB RAM, macOS 26.6.2 arm64. Locked-only service runtime: CPython
3.11.15, NumPy 1.26.4, OpenCV 4.11.0 / headless wheel 4.11.0.86, SciPy 1.13.1, FastAPI 0.115.12,
Pydantic 2.11.7 and Uvicorn 0.34.3. Configuration: one ASGI owner, two compute workers,
`max_pending: 8`, decoded cache 256 MiB / 64 entries, native threads one per worker. This is a
native development-host run, not a Linux production/container benchmark.

Observed implementation SHA-256:

`eb6bfe9262212163a1a298fc50438b545def36e3af65cbbe362df839cd4ba10e`

Each concurrency setting completed 36/36 benchmark calculations (12 cases x 3 fixed-file repeats),
72/72 overall. Two prior template-teaching calls initialized actual shared template artifacts.
All recorded numeric observations repeat with same-file standard deviation zero; all localized
runs detect exactly one target per sample. Target count is null for atomic fits/inspection because
those endpoints do not localize objects.

| Image | Localized observed feature | Fit RMS (px) | Detected targets | Localized P95 ms, concurrency 1 / 2 | Peak worker MiB, max across both |
|---|---|---:|---:|---:|---:|
| `bottleneck_1.jpg` | line (-0.013423, 0.999910, -737.194404) | 0.557389 | 1 | 1161.93 / 1269.99 | 255.56 |
| `bottleneck_2.jpg` | line (-0.012469, 0.999922, -738.452930) | 0.592567 | 1 | 1152.82 / 1233.16 | 255.02 |
| `bottleopen_1.jpg` | inner mouth radius 320.432315 px | 0.175955 | 1 | 619.22 / 686.38 | 279.62 |
| `bottleopen_2.jpg` | inner mouth radius 320.391989 px | 0.184801 | 1 | 617.99 / 668.80 | 287.89 |

Inspection/atomic-measurement per-case P95 spans 262.97–272.63 ms at concurrency 1 and
326.06–370.64 ms at concurrency 2. Overall measured peak worker RSS is 301,875,200 bytes
(287.89 MiB). The report preserves actual queue/preparation/worker-startup/compute timing segments
and cache activity. These workload measurements have no supplied acceptance limits; both reports
explicitly retain `qualification.status: pending`. Physical dimensional error, capture-series
repeatability, missed-target rates and total service/container peak remain unmeasured.

Input identities:

| Image | Encoded bytes | SHA-256 |
|---|---:|---|
| `bottleneck_1.jpg` | 140249 | `c102bb71a132705705bc27ce9ddbe226cb20e56cee6ea6481f6c45028173f2e2` |
| `bottleneck_2.jpg` | 140260 | `9c77230c22148149cce0fba2979739cf8cd11d3235b0131078a7b457d81100fc` |
| `bottleopen_1.jpg` | 180657 | `aef35642789d09f0b40b1ff2803d70202f68678d191af0524e2c22c3eaae5650` |
| `bottleopen_2.jpg` | 205953 | `ff5fee2dc00cc5f3f28b6ae261b3dc9d441b980d01730129701e26dbde06786a` |

Raw evidence is generated under ignored `output/v2-qualification/`: `cases.json`, `inventory.json`,
`concurrency-1.json`, `concurrency-2.json` and immutable taught templates. The standard-library
qualification harness is runnable against supplied real recipe/request cases and optional matching-
unit ground truth. With no agreed thresholds it returns `qualification.status: pending` even when
all computations succeed. The deployment guide describes how to supply independent traceable truth
and qualification thresholds without changing the service contract.

Production qualification remains pending until the upstream supplies calibrated representative
camera captures, independent physical truth, capture-repeatability data and agreed dimensional
accuracy/target count/concurrency/latency/memory limits. Docker target-node qualification also
remains pending. These are external evidence gates, not advertised stub operations.


### Final independent review and post-review workload

Both independent review axes are closed after source fixes `7ccf62a` and `49baf8f`:
five Standards findings and four Spec findings resolved. The final reports and original findings
are retained separately in [the review record](v2-code-review.md). Full suites passed 169 cases on
each supported Python runtime, followed by five integrating HTTP checks (5.29s) and three
independent reviewer HTTP checks (3.78s). No numerical-core or V1 source was modified by these fixes.

The final-source native HTTP workload ran after both axes closed, using the same host, locked-only
runtime, configuration, four unchanged inputs and 12-case/three-repeat workload described above.
It preserves a new immutable local evidence snapshot under ignored
`output/v2-qualification-post-review/`; the original snapshot remains available unchanged.
Its source/runtime identity is equal across both reports and includes implementation SHA-256:

`a1847e48b96dcdf9827ad465560d4f11b3fce459778f4f109580dfe08bd8c6ef`

Independent checks confirm 36/36 succeeded at each concurrency setting, 72/72 overall, and
`measurement_valid: true` for all atomic and localized measurements. Every localized case has
`target_counts: [1, 1, 1]`. Every recorded numerical observation is identical to its original
snapshot and has zero unchanged-file standard deviation. Input SHA-256 values match the actual
four repository files. No independent truth was supplied, so every absolute-error field is null.
This numerical comparison establishes regression stability for these inputs, not physical accuracy.

| Image | Localized observed feature | Fit RMS (px) | Detected targets | Localized P95 ms, concurrency 1 / 2 | Peak worker MiB, max across both |
|---|---|---:|---:|---:|---:|
| `bottleneck_1.jpg` | line (-0.013423, 0.999910, -737.194404) | 0.557389 | 1 | 1131.76 / 1230.76 | 255.63 |
| `bottleneck_2.jpg` | line (-0.012469, 0.999922, -738.452930) | 0.592567 | 1 | 1133.62 / 1269.30 | 256.17 |
| `bottleopen_1.jpg` | inner mouth radius 320.432315 px | 0.175955 | 1 | 604.33 / 687.89 | 279.73 |
| `bottleopen_2.jpg` | inner mouth radius 320.391989 px | 0.184801 | 1 | 603.58 / 672.84 | 288.59 |

Inspection/atomic-fit per-case P95 spans 258.70–276.44 ms at concurrency 1 and 332.32–365.42 ms
at concurrency 2. Overall sampled peak worker RSS is 302,612,480 bytes (288.59 MiB).
Aggregate cache hits/misses/evictions are 58/2/0 at concurrency 1 and 60/0/0 at concurrency 2;
the same service cache persists between settings. Both reports preserve actual preparation/compute
segments and use `CLOCK_MONOTONIC` for deadline budgeting. Three repetitions per case remain a
small workload smoke; sampled worker RSS does not establish total service/container peak memory.
Both reports retain `qualification.status: pending` until representative capture data, traceable
physical truth and production thresholds are supplied. Actual Docker runtime/mount verification
also remains pending on the target node.

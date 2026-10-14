# Supplied-result rendering, reporting and diagnostics

`POST /api/v2/visualizations` and `/result-exports` accept observations supplied by upstream.
They never require the originating operation ID or an engine business-result database. Both
support the same bounded sync/async scheduling, caller-scoped retries, verified files and final
manifest contract as the other calculation families.

## Supplied records and files

Provide exactly one `result` or `result_file`. JSON files contain the **result itself**, rather
than an HTTP operation envelope. Accepted records are `RunResult`, `RunBatchResult` and
`AtomicSuppliedResult`. The latter has `operation` (the capabilities identifier), `value` (that
operation's public result), optional `node_id`, `target_id`, `label`, `evaluation_status` and
`space` (`pixel` by default; explicitly choose `physical` for a physical atomic result).

Preparation parses file JSON, validates successful nodes against their original atomic schemas,
and pins all source, template, calibration, curve and artifact references before admission and
idempotency. Failed/blocked nodes retain their structured failures and invalid observations.
Artifact descriptors must agree with actual digest and byte size. All declared upstream files
must remain available and immutable until the new calculation terminates. Missing files,
changed content or inconsistent descriptors produce explicit file/result errors. There is no
attempt to recover content from expired operations or infer replacements.

Reports use supplied evaluation outcomes. A valid measurement is never silently changed to a
business pass. Upstream labels and configuration versions are preserved independently of IDs.

## Vector and raster visualization

Vector overlays are the default response and need no output directory:

```json
{
  "result_file": {"root": "exchange", "path": "retained/result.json"},
  "display_frame": "source",
  "space": "pixel",
  "selected_nodes": ["width"],
  "visible_nodes": ["width", "circle"]
}
```

Each primitive has stable `id`, `item_id`, `target_id`, `node_id`, `port`, `type`, decimal
`geometry`, actual `frame`/`unit`, `label`, `valid`, `visible`, `selected` and `style`. Visibility
and selection belong to each request; they never mutate engine GUI preferences. Omit
`visible_nodes` to show all nodes, or supply `[]` to hide all. Hidden primitives remain in the
vector result so upstream can toggle them without calculating pixels again. Failed observations
produce invalid annotations, with no invented zero geometry. Scalar-only observations have no
spatial overlay unless actual endpoint/geometry observations accompany them.

Point, line, circle, rectangle and polyline observations are supported. Samples/curve arrays are
excluded from the normal overlay JSON. Vector coordinates preserve precision; raster rounding
only affects the optional drawing. Raster output adds `raster: true` and `output.directory`.
Supply `image`, or a single run's verified `measurement_image` is used automatically. The PNG has
the actual decoded image dimensions and BGR colors. The input image is never modified. With a
batch, supply the specific display image explicitly. The caller must declare that image's
actual coordinate frame using `display_frame`.

Pixel observations already use the run's published pixel metrology frame. A stitched canvas is
therefore drawn on a retained canvas image, with its declared canvas frame, rather than on an
individual raw camera image. A stitch run retains that image only when
`retain_stage_images: true` and `output.directory` were supplied; otherwise provide an
independently committed canvas image. `sources` and `image_stages` preserve camera/canvas
provenance. Corrected measurements map back to original source pixels in the recipe runner.

A differing display frame requires an explicit `mapping`, or a target's recorded
`physical_mapping` can be used to reverse physical observations into source pixels. Matrices and
inverses are validated. Lines retain analytic coefficients under invertible matrix mappings;
nonlinear line rendering requires measured endpoints. Similarity-mapped circles remain circles.
Affine/projective circle mappings use closed sampled curves (`curve_samples`, default 128,
range 16–4096); no average radius multiplier is used. Curves crossing the projective horizon
are rejected.

The supported nonlinear physical mapping is `details.model: opencv_pinhole_plane`, with recorded
`camera_matrix`, `distortion`, `corrected_to_physical` and `physical_to_corrected`. Physical curve
samples first map to corrected camera pixels, normalize through inverse K, and project through
the original OpenCV K/D lens model. Other nonlinear models return
`visualization.nonlinear_unsupported`; an unsupported mapping never becomes a fake homography.
Raster geometry must be in pixels. Primitive count is bounded at 100,000.

## Shared sampling diagnostics

Add `diagnostics: true` and `output.directory` to visualization. Full gray/smoothed/gradient
curves and their sampling coordinates are non-pickle NPZ outputs, with an optional headless
480x240 PNG curve preview. Selected edge positions, amplitudes, fit samples and residuals are
separate NPZ files. Supplied shared NPZ curves are verified and bounded before decoding; object
arrays/pickle metadata are rejected.

`diagnostics.json` records each item's target/node identity, frame/unit, selected observation
count, available numerical quality, array shape/dtype, structured failure, configured sampling
polygons and their recorded mapping to metrology. Repeated public sample/edge ports do not double
count a selected observation. Probe polygons describe continuous expected sampling support;
selected numeric arrays contain the actual subpixel observations. No synthetic zero residual or
quality is reported when a metric is absent.

All requested raster, curve, sampling and diagnostic files share one final manifest. Descriptor
`kind`, `format`, `schema_version`, `file`, `sha256` and `size_bytes` describe complete content.
Readers must wait for the manifest before using a multi-file set. A cancelled pre-manifest
attempt has no completed output; its owned files are cleaned after worker termination. A
completed upstream report remains upstream-owned after operation expiry or service restart.

## CSV and JSON reporting

```json
{
  "result_file": {"root": "exchange", "path": "retained/result.json"},
  "formats": ["csv", "json"],
  "output": {"directory": {"root": "exchange", "path": "reports"}}
}
```

CSV emits separate pixel and physical observations with `recipe_id`, `recipe_version`,
`configuration_sha256`, `item_id`, `target_id`, `node_id`, `label`, `operation`, `port`, `space`,
`frame`, `unit`, `state`, `valid`, `evaluation_status`, `value`, `reason`, `geometry_json` and
`error_json`. Invalid numeric values stay empty rather than zero. Geometry retains actual
coordinates/coefficients. Failed batch items have explicit identity/error rows. UTF-8 and standard
CSV quoting preserve Chinese, commas and newlines in labels. Numeric representations retain
full precision; GUI rounding and implicit scale multipliers are not applied.

JSON preserves the complete supplied typed result, including pixel/physical observations,
configuration, verified input/runtime provenance and structured failures. It can be archived by
upstream and supplied again later. Both formats commit together with a final manifest.

## Retained operational diagnostics and traces

`GET /operations/{id}/diagnostics` exposes actual events and available metrics:

- `queue_ms`: admission creation to acquisition of a compute-worker slot.
- `worker_startup_ms`: process launch to prepared delivery minus measured worker preparation.
  It includes IPC delivery and uses same-process monotonic durations on Python 3.9 and later.
- `preparation_ms`: worker-side registry/request/input snapshot preparation.
- `admission_ms`: total elapsed preparation/admission including queue and worker startup.
- `compute_ms`: execution and result validation after prepared admission; preparation and queue
  are excluded. Artifact staging inside execution is included; unavailable commit-only timing is
  omitted.
- `deadline_clock: CLOCK_MONOTONIC`, `admitted_monotonic`, `worker_started_monotonic` and
  `deadline_monotonic` use one Unix OS epoch across the HTTP owner and workers, including
  Python 3.9/macOS. They are operational monotonic values rather than wall-clock timestamps.
  Cancellation and pre-manifest deadline checks use this same clock.
- Actual `stage_timings`, cache counters and measured peak worker memory where available.
- Per-node numerical quality, exact selected sample count, at most 32 xy preview samples,
  truncation flag and structured failures. Full requested arrays belong in shared diagnostics.

`GET /traces/{trace_id}` returns the same retained diagnostics. Known expired operation and
trace records return 410 during the configured tombstone retention; unknown or forgotten IDs
return 404. Upstream retains business traces/results. Original operational expiry does not
prevent new visualization/report calculations from supplied records and unchanged shared files.

## Acceptance evidence

`tests/v2/test_results.py` exercises decimal geometry, independent display controls, exact
raster dimensions/pixels, projective and lens-distorted circle samples, physical-to-source
projection, rotated rectangles, CSV quoting/precision/nulls/evaluation, batch failures, numerical
NPZ diagnostics, probe/fit quality, digest/descriptor rejection, real queue versus compute timing,
trace expiration/restart, retained-file ownership and cancellation before a multi-file manifest.

Ticket 08 implementation acceptance: all 17 result-family HTTP tests pass. The integrated
123-test public V2 suite passed under Python 3.11 (85.60s) and Python 3.9 (109.85s), including
real lifecycle publication/deadline boundaries with the shared system monotonic clock.

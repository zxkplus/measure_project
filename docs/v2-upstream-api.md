# V2 upstream integration API

This document is for services implementing the measurement editor and production workflow.
It describes the implemented FastAPI V2 service, API version `2.0`, recipe/asset schema `1.0`.
The HTTP prefix is `/api/v2`. Requests use JSON control parameters and shared-file references;
images, templates and other large payloads are exchanged through directories on the same node.

The service implements 65 required routes, including 55 calculation operations. Use this guide
for integration behavior, the [complete field reference](v2-api-reference.md) for every request,
response and nested model, and the [OpenAPI snapshot](v2-openapi.json) for client generation.
The deployed service exposes its actual schemas at `GET /api/v2/openapi.json` and interactive
Swagger UI at `GET /api/v2/docs`. The live schema is authoritative for that deployment.
Only the URL prefix is shared with the retired V2; its sessions and protocol are incompatible.

## Contents

- [Responsibility and GUI mapping](#responsibility-and-gui-mapping)
- [Connection and shared directories](#connection-and-shared-directories)
- [Common HTTP and operation contract](#common-http-and-operation-contract)
- [Files, asset versions and publication](#files-asset-versions-and-publication)
- [Coordinates, observations and product decisions](#coordinates-observations-and-product-decisions)
- [End-to-end integration](#end-to-end-integration)
- [Recipe editor and batch behavior](#recipe-editor-and-batch-behavior)
- [Calibration and advanced image workflows](#calibration-and-advanced-image-workflows)
- [Errors, retries and operational records](#errors-retries-and-operational-records)
- [API families and further reference](#api-families-and-further-reference)
- [Runnable example and document verification](#runnable-example-and-document-verification)

## Responsibility and GUI mapping

| Concern | Upstream responsibility | V2 responsibility / API |
|---|---|---|
| Image acquisition and browsing | Camera/device control, input publication, preview display | Inspection, preprocessing, preview and crop |
| Project/recipe management | Drafts, names, versions, activation, permissions, authoritative storage | Validate and execute supplied snapshots |
| Template teaching | Choose reference/ROI/anchor, persist and version returned assets | Build, inspect and match portable templates |
| Calibration | Board selection/editing, accepted observations, camera/setup identity, asset lifecycle | Detect features, compute, validate, import/export calibration |
| Localization and correction | Select search policy and correction/alignment settings | Match independent targets, correct images, refine target-local alignment |
| Measurement tools | Editor geometry/parameters, stable node IDs, dependency graph | Edges, lines/circles, template points, geometric composition |
| Live editor feedback | Revision/token tracking, discard stale responses | Stateless selected-node preview with dependency closure |
| Product rules | Limits, units, invalid/no-target policy and business actions | Explicit range/variation/deviation evaluations |
| Results and overlays | Business archive, labels, display state, operator workflow | Supplied-result vectors/raster/diagnostics/CSV/JSON |
| Production execution | Triggering, job-to-product association, bounded submission, retries | Bounded sync/async computation and retained operation controls |
| Migration and investigation | Archive assets/runtime provenance, choose supported migration | Portable import/export, explicit legacy conversion, frozen replay |

There is no service-owned recipe CRUD, modeling session, current active recipe or template registry.
Saving an editor draft means saving its JSON/assets upstream. Changing a tool's label, visibility
or selection never implicitly mutates a saved measurement configuration in V2.

## Connection and shared directories

Example service origin: `http://measure-v2:9004`. A local process can use
`http://127.0.0.1:9004`; container-to-container callers use the service's reachable address.
All curl examples below set the origin separately from the API prefix:

```bash
BASE_URL=http://127.0.0.1:9004
curl -sS "$BASE_URL/api/v2/health/ready"
curl -sS "$BASE_URL/api/v2/capabilities"
curl -sS "$BASE_URL/api/v2/openapi.json" -o deployed-v2-openapi.json
```

Example engine configuration, corresponding to [deploy/v2-settings.json](../deploy/v2-settings.json):

```json
{
  "shared_roots": {
    "exchange": "/shared/exchange",
    "assets": "/shared/assets",
    "outputs": "/shared/outputs"
  },
  "writable_roots": ["outputs"],
  "state_directory": "/var/lib/measure-v2/state",
  "max_workers": 2,
  "max_pending": 16
}
```

| Alias | Example upstream mount | Engine mount | Engine permission | Contents |
|---|---|---|---|---|
| `exchange` | `/srv/measurement/images` | `/shared/exchange` | Read | Captures and reference images |
| `assets` | `/srv/measurement/assets` | `/shared/assets` | Read | Recipe, template, calibration and archived result files |
| `outputs` | `/srv/measurement/outputs` | `/shared/outputs` | Read/write | Isolated generated files and final manifests |

Each row must expose the same underlying files on both sides. The engine's pathname can differ
from upstream's pathname. Prefer alias-relative references, e.g.
`{"root":"exchange","path":"captures/frame-001.png"}`; upstream resolves that reference
using its own alias-to-directory map. Do not send an upstream-only absolute pathname.
Absolute engine paths are accepted only within configured shared roots.

Deploy one ASGI owner per persistent state directory. `max_workers` bounds spawned calculations;
`max_pending` bounds admitted work including queueing/preparation, not just running calculations.
Default file/pixel budgets are 256 MiB encoded per file and 64 Mi pixels; default worker-memory
budget is 2 GiB per worker. The operator can change these settings. They are resource limits,
not production throughput guarantees. See [deployment guidance](v2-production-engine.md) for
startup, mounts, permissions, state retention and container configuration.

## Common HTTP and operation contract

### Headers

| Header | Use |
|---|---|
| `Content-Type: application/json` | Calculation/control POST JSON |
| `X-Caller-Id` | Stable caller identity for idempotency scope; defaults to `anonymous` |
| `X-Trace-Id` | Optional upstream correlation value; generated if omitted and echoed in responses |

`X-Caller-Id` provides identity separation for retries; it is not authentication. The current
application defines no credential/security scheme. Supply deployment access controls as required
by the upstream environment rather than assuming these headers authorize a caller.
An operation keeps its original trace identity. A later GET can have a new HTTP response trace
header while its operation record still carries the original calculation trace.

### Shared calculation request fields

The 55 calculation POSTs take a flat object. Add these controls alongside family-specific fields;
do not wrap the family payload under `data` or `parameters` at the HTTP boundary.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `mode` | `sync` or `async` | `sync` | Wait for computation or return after verified preparation |
| `wait_ms` | integer, 0–60000 | `10000` | Sync waiting budget; does not stop work |
| `timeout_ms` | integer, 1–3600000 | `60000` | Total operation budget including admission, queue, preparation and worker startup |
| `idempotency_key` | string, 1–200 characters or null | null | Caller/route-scoped retry identity |

Requests and typed models reject unknown fields and non-finite values. OpenAPI uses nullable union
schemas: nullable does not mean omittable. Some semantic validators require exactly one of two
nullable fields, even though each is individually optional in OpenAPI.

Async admission still verifies and hashes required inputs before returning. In this implementation
preparation can wait for a worker slot; `mode: async` and `wait_ms: 0` do not promise an immediate
HTTP response. `wait_ms` is not a hard limit on the initial POST's full network duration. Choose
an upstream/proxy timeout that accommodates preparation and the requested operation budget.
A client disconnect or HTTP wait expiry does not cancel admitted work.

### Operation envelope

Calculation success and acceptance use the same envelope. This is a complete illustrative image
inspection response; operation IDs, hashes and timestamps are runtime-specific:

```json
{
  "operation_id": "example-operation",
  "operation": "image-inspections",
  "state": "succeeded",
  "trace_id": "example-trace",
  "result": {
    "metadata": {
      "width": 220, "height": 110, "channels": 1, "dtype": "uint8",
      "pixel_format": "GRAY8", "frame": "source", "unit": "px"
    },
    "source": {"root": "exchange", "path": "production.png"}
  },
  "error": null,
  "created_at": "2026-10-06T00:00:00+00:00",
  "completed_at": "2026-10-06T00:00:01+00:00",
  "diagnostics": {}
}
```

Actual verified `source` values contain SHA-256 content identities. Timestamps use ISO 8601 UTC;
diagnostic monotonic values are durations/deadline clocks and are not wall-clock timestamps.

| HTTP response | Body / upstream action |
|---|---|
| Calculation POST `200` | Terminal operation; check `state`, then family result validity and product evaluation |
| Calculation POST `202` | Nonterminal operation plus `Location: /api/v2/operations/<operation_id>`; persist ID and poll |
| Calculation POST error status | `{"error":{...},"operation_id":...}`; ID is optional and present for certain admitted failures |
| Operation GET `200` | Operation envelope, including still-running or terminal failed/cancelled/timed-out states |
| Discovery/diagnostics GET | Direct object, not a calculation envelope |

An async calculation that completes before the response can return `200`. Conversely, a sync
call can return `202` when its waiting budget ends. The original calculation continues in either
case. Initial POST calculation failures may use a non-2xx error envelope; later operation polling
returns the retained record with its `error`. Do not interpret polling HTTP `200` as successful
measurement. A terminal cancelled/timed-out POST can also carry an operation envelope.

### Polling, cancellation and discovery

| Endpoint | Response / parameters |
|---|---|
| `GET /health/live` | `{"status":"ok"}` |
| `GET /health/ready` | `{"status":"ready","workers":2}` or structured `503` |
| `GET /capabilities` | API/algorithm/runtime identity, coordinate convention, operation schemas and output ports |
| `GET /operations/{id}?wait_ms=1000` | Optional long-poll wait 0–60000 ms; returns the current retained record |
| `GET /operations?state=running&limit=100` | `{"operations":[...]}`, newest first; limit 1–1000, default 100 |
| `POST /operations/{id}:cancel` | No calculation body; returns the resulting operation record |
| `GET /operations/{id}/diagnostics` | Direct diagnostics object |
| `GET /traces/{trace_id}` | `{"trace_id":"...","operations":[...]}` |

All paths in this table are beneath `/api/v2`. Operational list results are bounded and are not
an upstream business-history pagination API.

```bash
curl -sS "$BASE_URL/api/v2/operations/$OPERATION_ID?wait_ms=1000"
curl -sS -X POST "$BASE_URL/api/v2/operations/$OPERATION_ID:cancel"
```

Nonterminal states: `queued`, `running`, `cancelling`. Terminal states: `succeeded`, `failed`,
`cancelled`, `timed_out`. A cancellation can lose the race against final publication: the API
preserves an already committed `succeeded` result. Read the returned state instead of assuming
the cancel request reversed a completed result. Queued/running work can also fail through worker
interruption or memory limits.

## Files, asset versions and publication

### FileReference and output directory

```json
{
  "image": {"root": "exchange", "path": "captures/frame-001.png"},
  "output": {"directory": {"root": "outputs", "path": "product-123"}}
}
```

`FileReference` has `path` (required), optional `root`, `sha256` and `provenance`.
Alias paths must be relative. Without `root`, `path` must be an absolute engine pathname.
These two forms are mutually exclusive. Paths and resolved symlinks must remain inside configured
roots. Output directories must resolve to a writable alias. A SHA-256, when supplied, is exactly
64 hexadecimal characters; do not send `"..."` as a digest.

Recommended upstream publication:

1. Write the complete capture/asset to a temporary pathname on the same shared filesystem.
2. Flush/close it, compute its digest and atomically publish it under a unique versioned pathname.
3. Submit the final reference with the actual digest; retain all transitive inputs unchanged
   until the operation reaches a terminal state.
4. Archive the returned result and any files needed for later visualization/export/replay.

The service verifies actual input content before acceptance and again when reading it. Nested
recipe/template/calibration/result dependencies are also prepared and verified. Input files are
never deleted by V2. A changed path's contents can create a digest/idempotency conflict; updating
bytes in place is not a way to revise an already accepted operation.

### Asset ownership

Template teaching returns `result.template`, a directly usable FileReference. Upstream can keep
that immutable generated path or copy the bytes atomically into a versioned `assets` pathname
and use the new reference/digest. Calibration calculations return `result.asset`, a typed JSON
value, plus canonical asset identity. Upstream can publish that value as versioned UTF-8 JSON or
use it inline under `calibration.value`. No registration or activation call is necessary.

`calibration` requires exactly one of `value` and `file`. Recipe calls require exactly one of
`recipe` and `recipe_file`. Result consumers require exactly one of `result` and `result_file`.
Replay requires exactly one of `frozen` and `frozen_file`.

The digest of a raw JSON file and the digest of canonical normalized calibration JSON are
different concepts. `FileReference.sha256` checks raw bytes. Calibration identity and
`configuration_sha256` identify normalized content/configuration; do not substitute one for
another. Preserve `recipe_id`, explicit `recipe_version`, `configuration_sha256`, input/asset
digests and runtime identity in the upstream business record.

### Output descriptors and completion

Every calculation that generates files uses an isolated namespace beneath `output.directory`.
For example, `outputs/product-123/<operation_id>/...`. Follow actual returned descriptors; do not
construct a filename or infer a namespace from a trace/product ID.

```text
FileDescriptor = {
  kind: string,
  format: string,
  schema_version: string,
  file: FileReference,
  sha256: 64-character content digest,
  size_bytes: integer
}
```

`result.artifacts` lists generated files. `result.manifest` describes `manifest.json`, committed
last, whose `files` list describes the complete artifact set. The manifest is the multi-file
completion marker. Staged files or a directory alone do not establish a completed output.
Verify descriptor size/digest where integrity is required. Completed artifacts are upstream-owned;
operation expiry and state cleanup do not remove them. Upstream decides asset/output retention.

Image inputs are PNG/JPEG with actual GRAY8 or BGR8 pixels. Four-channel and 16-bit inputs are
rejected. Template assets are safe numeric NPZ plus metadata. Preprocessing may output a numeric
NPY and separate display PNG: preserve numeric arrays for diagnostics; a clipped display preview
does not replace measurement pixels. Numeric NPY is not an accepted `image` input to PNG/JPEG
measurement calls. No HTTP multipart upload or binary download endpoint is provided by this V2.

## Coordinates, observations and product decisions

| Quantity | Convention |
|---|---|
| Pixel point | `{x,y}`; x right, y down; top-left pixel center is `(0,0)` |
| Angle | Degrees, clockwise positive; do not send legacy radians |
| Rotated ROI | Full integer `width`/`height`, source `center`, `angle_degrees` |
| Probe | `search_half_length` along search direction; `average_half_width` perpendicular |
| Pixel length | `px`, no GUI rounding in numeric exports |
| Physical length | Explicit `mm`, `cm`, `m` or `um`, with applicable calibration |
| Coordinate mapping | Named source/destination frames and `source_to_destination` direction |

Matrix mappings use homogeneous xy coordinates and include an inverse where defined. Pixel-center
resize/crop transforms can have half-pixel translations. Use returned mappings, not a width-ratio
guess. Nonlinear mappings describe their lens/plane model rather than fabricating an affine matrix.

Common positioned point: `{"valid":true,"frame":"source","unit":"px","point":{"x":10,"y":20}}`.
Common scalar: `{"valid":true,"frame":"source","unit":"px","value":20}`.
Invalid observations contain `valid: false`, a reason and null geometry/value where applicable;
do not replace missing measurements with zero. Lines use normalized `a*x+b*y+c=0`; circles use
center/radius. Composition requires compatible frames and units.

Upstream must keep these three decisions separate:

| Layer | Fields | Meaning |
|---|---|---|
| Execution | Operation `state` and `error` | Whether the service completed a calculation |
| Measurement | `result.valid`, target/node/observation `valid` | Whether required data/geometry is usable |
| Product | `evaluation_status` = `pass`, `fail`, `not_evaluated` | Outcome of explicitly supplied limits/policies |

For example, a run can be `succeeded`, measurements `valid: true`, and product evaluation `fail`
because a width exceeds the permitted range. An empty target set never supplies valid measurements,
even if the configured `no_targets` business policy is `pass`. `RunResult.status` is separately
`completed`, `partial_failure` or `no_targets`.

Similarity localization can normalize an object's patch for measurement, but reported pixel/physical
geometry recovers actual inspection size. A taught 20 px width at scale 1.2 measures 24 source px;
with a valid 0.1 mm/px calibration it measures 2.4 mm. Do not multiply these returned values by the
localization scale or calibration again. Selected samples are mapped before geometric refitting.

## End-to-end integration

The JSON examples below describe the runnable two-target example at the end of this document.
Use its fixture images or replace the image/ROI/geometry with actual product configuration.
File paths are illustrative committed assets; the service does not create the prerequisite input
files for a curl request. Optional digests are omitted for readability; include real digests in
production. HTTP operation IDs and output file paths always come from the responses.

### 1. Inspect an image

```bash
curl -sS -D inspection.headers -X POST "$BASE_URL/api/v2/image-inspections" \
  -H 'Content-Type: application/json' -H 'X-Caller-Id: upstream-service' \
  -H 'X-Trace-Id: product-123' \
  -d '{"image":{"root":"exchange","path":"reference.png"},"timeout_ms":60000}'
```

Read `operation.result.metadata` only after `state: succeeded`. For the example reference this is
80×80 GRAY8. The inspection image is 220×110; its calibration must apply to the inspection frame.
For independent atomic width measurement on the reference:

```json
{
  "image": {"root": "exchange", "path": "reference.png"},
  "probe": {
    "center": {"x": 40, "y": 40},
    "angle_degrees": 0,
    "search_half_length": 20,
    "average_half_width": 3
  },
  "polarity": "positive",
  "selection": "first"
}
```

POST this body to `/api/v2/measurements/edge-pairs`; read `result.width.value` in pixels.
For `selection: all`, inspect the `pairs` array: single-pair convenience ports can be null.

### 2. Teach and version a template

POST `/api/v2/template-builds`:

```json
{
  "image": {"root": "exchange", "path": "reference.png"},
  "roi": {"center": {"x": 40, "y": 40}, "width": 65, "height": 65, "angle_degrees": 0},
  "support_mode": "tight_roi",
  "preprocessing": {"type": "raw"},
  "output": {"directory": {"root": "outputs", "path": "teaching"}}
}
```

After success, take `result.template` and verify its manifest/artifacts. If the upstream asset
version is `template-v1`, copy the returned NPZ bytes into `assets/template-v1.npz` atomically.
The template retains the teaching image digest, support geometry, anchor and preprocessing.
Recipe `reference` must identify that same teaching content. A filename match alone is insufficient.

POST `/api/v2/template-matches` to inspect localization independently:

```json
{
  "image": {"root": "exchange", "path": "production.png"},
  "template": {"root": "assets", "path": "template-v1.npz"},
  "search": {
    "score_threshold": 0.85, "max_matches": 2,
    "scale_min": 1, "scale_max": 1.2, "scale_step": 0.2
  }
}
```

Default search is translation-only (angle 0°, scale 1). Supply angle/scale ranges explicitly when
needed. Each match contains score, angle/scale, anchor and reference-to-inspection mapping.
No match is a domain result with `valid: false`, an empty `matches` list and no stale geometry.
For single template-point measurement, use `/measurements/template-points` and `max_matches: 1`.

### 3. Compute and version scalar calibration

POST `/api/v2/calibrations/scales`:

```json
{
  "corners": [
    {"x": 5, "y": 5}, {"x": 15, "y": 5}, {"x": 25, "y": 5},
    {"x": 5, "y": 15}, {"x": 15, "y": 15}, {"x": 25, "y": 15}
  ],
  "square_size": 1,
  "unit": "mm",
  "applicability": {
    "camera_id": "upstream-camera", "setup_id": "fixture-A",
    "frame": "source", "image_size": {"width": 220, "height": 110}
  },
  "output": {"directory": {"root": "outputs", "path": "calibration"}}
}
```

These synthetic points encode 10 px spacing for 1 mm, hence 0.1 mm/px. Check `result.valid`,
then store `result.asset` as `assets/calibration-v1.json`. Scalar calibration assumes uniform
orthographic scale; it does not infer perspective or dimensional accuracy from this fixture.
In production the accepted board observations must come from the applicable camera/setup.

### 4. Assemble and validate a supplied recipe

Store the following Recipe JSON upstream. An editor may keep this as an unsaved in-memory value.

```json
{
  "schema_version": "1.0",
  "id": "upstream-part",
  "version": "upstream-v8",
  "source_frame": "source",
  "reference": {"root": "exchange", "path": "reference.png"},
  "localization": {
    "template": {"root": "assets", "path": "template-v1.npz"},
    "search": {
      "score_threshold": 0.85, "max_matches": 2,
      "scale_min": 1, "scale_max": 1.2, "scale_step": 0.2
    }
  },
  "metric": {
    "calibration": {"file": {"root": "assets", "path": "calibration-v1.json"}},
    "applicability": {
      "camera_id": "upstream-camera", "setup_id": "fixture-A",
      "frame": "source", "image_size": {"width": 220, "height": 110}
    }
  },
  "nodes": [
    {
      "id": "width",
      "operation": "measurements.edge-pairs",
      "parameters": {
        "probe": {
          "center": {"x": 40, "y": 40},
          "search_half_length": 20, "average_half_width": 3
        },
        "selection": "first", "polarity": "positive"
      }
    },
    {
      "id": "check",
      "operation": "evaluations.ranges",
      "space": "physical",
      "parameters": {"minimum": 1.9, "maximum": 2.2, "unit": "mm"},
      "inputs": {"value": {"node": "width", "port": "width", "space": "physical"}}
    }
  ],
  "required_outputs": [{"node": "check", "port": "value", "space": "physical"}],
  "target_policy": {"minimum": 2, "maximum": 2, "no_targets": "fail"}
}
```

POST `/api/v2/recipe-validations` with either `{"recipe": <the Recipe object>}` or:

```json
{"recipe_file": {"root": "assets", "path": "recipe-v8.json"}}
```

`result` reports `valid`, `issues`, deterministic `execution_order`, resolved recipe/asset
references and configuration hash. A semantic graph error can produce `succeeded` with
`valid: false`; do not publish it as ready. A schema/dynamic parameter error returns structured
`422`, for example `recipe.invalid_parameter` with a located node parameter `field_path`.

### 5. Preview an unsaved editor revision

POST `/api/v2/recipe-previews` using the entire current inline recipe, not only changed fields:

```json
{
  "recipe_file": {"root": "assets", "path": "recipe-v8.json"},
  "image": {"root": "exchange", "path": "production.png"},
  "selected_nodes": ["check"],
  "upstream_revision": "editor-8",
  "preview_token": "preview-8"
}
```

The file form above is useful for saved snapshots; substitute `recipe` for `recipe_file` when
previewing unsaved changes. The service validates the entire graph but executes selected nodes
and their dependency closure, e.g. `width` then `check`. It echoes revision/token unchanged and
saves no recipe. Upstream should apply the response only if both still match the editor's latest
request; late older previews must not replace current feedback.

### 6. Submit a production run and archive its result

POST `/api/v2/runs`:

```json
{
  "mode": "async",
  "timeout_ms": 60000,
  "idempotency_key": "product-123-recipe-v8-attempt-1",
  "recipe_file": {"root": "assets", "path": "recipe-v8.json"},
  "image": {"root": "exchange", "path": "production.png"},
  "output": {"directory": {"root": "outputs", "path": "product-123"}}
}
```

Retain the operation ID, poll until terminal, then associate the result with the product upstream.
For this fixture: two targets, source-pixel widths approximately 20 and 24 px, physical widths
2.0 and 2.4 mm, target business statuses `pass` and `fail`, and aggregate `evaluation_status: fail`.
Measurement validity remains true. Target IDs are local to the run, not stable cross-image part IDs.

Useful RunResult fields:

| Field | Upstream use |
|---|---|
| `recipe_id`, `recipe_version`, `configuration_sha256` | Associate exact configuration |
| `resolved_recipe`, `assets`, `sources`, `runtime_identity` | Archive numerical provenance |
| `status`, `valid`, `evaluation_status` | Separate execution content, usability and conformity |
| `targets[].target_id`, `valid`, `evaluation_status` | Per-target UI/product decision |
| `targets[].nodes[].id`, `state`, `valid`, `error` | Stable tool identity and partial failure |
| `nodes[].pixel_result`, `physical_result` | Distinct usable pixel and calibrated observations |
| `nodes[].result` | Authoritative node result; use named ports appropriate to the node |
| `mapping`, `metric_mapping`, `physical_mapping` on targets | Localization, recovered measurement and physical/display transforms |
| `measurement_image`, `image_stages` | Actual display image and correction/stitch provenance |
| `artifacts`, `manifest` | Shared output retention/integrity |

With output supplied, the run publishes pure `result.json`, described by an artifact of kind
`recipe_result`. That file is a RunResult, not the outer OperationResponse. Its own descriptor
and final manifest are omitted inside the file to avoid self-referential hashes. Upstream can also
archive the HTTP `result` value as a new pure JSON result file. Do not archive the outer envelope
and then pass it as a `result_file`.

### 7. Visualize and export retained results

Take a `result_file` from the returned `recipe_result` artifact or an upstream-published pure
RunResult JSON. POST `/api/v2/visualizations`:

```json
{
  "result_file": {"root": "assets", "path": "product-123-result.json"},
  "space": "physical",
  "display_frame": "source",
  "visible_nodes": ["width", "check"],
  "selected_nodes": ["width"],
  "raster": true,
  "diagnostics": true,
  "output": {"directory": {"root": "outputs", "path": "product-123-view"}}
}
```

The service uses the retained run's verified `measurement_image` unless `image` is supplied.
Physical observations map back into source pixels using their recorded mapping. Vector primitives
identify target/node/port, decimal geometry, frame/unit, validity, visibility and selection.
Omitting `visible_nodes` shows all; `[]` hides all. Hidden primitives remain in the vector result.
Raster/diagnostic generation needs output; vector-only visualization does not. Diagnostics export
full sampling/fit arrays as safe NPZ and manifest-bound files, not just UI-truncated sample previews.

POST `/api/v2/result-exports`:

```json
{
  "result_file": {"root": "assets", "path": "product-123-result.json"},
  "formats": ["csv", "json"],
  "output": {"directory": {"root": "outputs", "path": "product-123-reports"}}
}
```

CSV retains separate pixel/physical rows, stable IDs, units, validity and business status. JSON
preserves the typed result/provenance. These calls do not require an unexpired originating operation.
All referenced inputs/artifacts must still exist and match their recorded descriptors.
For atomic results, wrap the family result in `AtomicSuppliedResult`, for example:

```text
result = {
  operation: "measurements.edge-pairs",
  value: the successful atomic result object,
  space: "pixel",
  node_id: "width",
  target_id: "target-1"
}
```

### 8. Package, relocate and freeze a complete run

POST `/api/v2/recipe-exports` with `recipe: run.resolved_recipe`, `frozen_run: run`,
`packaging: embedded`, and `output.directory`. Select returned descriptors by kind:
`recipe_package` for the portable package, `frozen_run` for the frozen replay manifest.
Do not infer their paths. You can supply a pure `frozen_run_file` instead of inline `frozen_run`.

Embedded packaging includes transitive assets. Reference packaging keeps explicit linked shared
references; it does not copy their bytes. POST `/recipe-imports` with `package` and new `output`;
its validated `resolved_recipe` can execute directly using relocated assets. Reference imports
may use `root_bindings` and digest-preserving `asset_bindings`; original absolute paths require
explicit asset bindings. No import activates a recipe or creates upstream versions automatically.

POST `/api/v2/replays` with the actual frozen descriptor and a new output directory:

```text
{
  frozen_file: returned artifact.file of kind frozen_run,
  output: {directory: {root: "outputs", path: "product-123-replay"}}
}
```

Replay verifies exact runtime identity, resolved configuration and all original input digests.
It returns a new `run`, `equivalent` and bounded `differences`. Runtime identity includes source
digest, Python/architecture and numerical dependency/OpenCV-build identities; retain a compatible
runtime for forensic replay. Portable recipe relocation and frozen historical replay are distinct:
an embedded package can move assets, while replay still requires the frozen manifest's explicit
original input references. No historical operation record is required.
A preview cannot freeze as a complete run. Equivalence is numerical/business comparison within
the frozen tolerance, not a certification of production dimensional accuracy.

## Recipe editor and batch behavior

Use `GET /capabilities` to obtain operation IDs, request/result schemas and named output ports.
For example, the route `/geometry/lines:fit` uses ID `geometry.lines_fit`; do not derive IDs by
blindly replacing route separators. UI labels and ordering are independent of dependency IDs.

Each recipe node has stable `id`, `operation`, literal `parameters`, named `inputs`, optional
`label`, `enabled`, `required`, and `space`. Bindings are `{node, port, space}`. Input map keys
are actual atomic request field names. `parameters` and bound `inputs` cannot define the same
field. Parent-managed images/scheduling/output are not per-node controls. Arrays require explicit
binding lists; an array port never implicitly means its first result. Recipe nodes do not accept
recursive recipe execution or template-teaching operations.

Without localization a recipe runs once on the whole image. Template targets normalize to the
reference image when no explicit alignment is declared; taught tool coordinates use that reference.
`fixed_roi` targets rectify into ROI-local patches: tool coordinates then use that local patch,
not source-image coordinates. Explicit alignment/control points define further patch normalization.
Controls search only their own target; a neighboring target cannot supply a missing control.
`failure_policy: rigid_fallback` is explicit, not an automatic substitution for failed refinement.

| Batch endpoint | Input / behavior |
|---|---|
| `/measurement-batches` | One image, independent `measurements` with unique IDs, atomic operation IDs and literal parameters |
| `/run-batches` | One supplied recipe and unique-ID `items`, each with image and optional recipe override |

Both use the outer scheduling envelope and return ordered per-item outcomes. Child failures and
invalid observations do not erase successful independent siblings. An outer `succeeded` can carry
`valid: false` and failed item states. Batching is one admitted computation, not a promise that
every item is dispatched concurrently. Large batches still share one operation deadline/memory
budget. A batch's committed files share the final parent manifest.

Example `/run-batches` request:

```json
{
  "recipe_file": {"root": "assets", "path": "recipe-v8.json"},
  "items": [
    {"id": "product-123", "image": {"root": "exchange", "path": "frame-123.png"}},
    {"id": "product-124", "image": {"root": "exchange", "path": "frame-124.png"}}
  ],
  "output": {"directory": {"root": "outputs", "path": "batch-42"}}
}
```

Inspect `result.items[].state`, `valid`, `result` and `error`. Item ID and target ID have different
scopes. Missing per-item inputs can remain item failures while other usable items run.

Range limits are inclusive and require at least one bound. `invalid_policy` is `not_evaluated`
(default), `fail`, `error` or `ignore`. Aggregations still require enough usable observations;
`ignore` never makes an insufficient set valid. Diameter variation is max minus min supplied
diameter; centerline deviation is max minus min compatible center x. Sampled diameter spread
is `2*(max sampled radius-min sampled radius)`, not a standardized roundness metric.

## Calibration and advanced image workflows

`CalibrationApplicability` requires `camera_id`, `setup_id`, `image_size` (width/height) and `frame`.
The actual decoded dimensions and supplied frame/setup must match. An image is never implicitly
resized to fit a calibration. Camera intrinsic parameters alone cannot convert pixel distances
to physical length. Use a justified scalar model or map coordinates/samples through a plane model.

| Goal | APIs / integration rule |
|---|---|
| Manual/edited scale | `/calibrations/scales`; at least six usable unique noncollinear corners |
| Detect board observations | `/calibration-features/checkerboard-corners`, `/dot-grids`; check validity/ordering before accepting views |
| Camera intrinsics | `/calibrations/cameras`; at least three distinct usable planar views with explicit correspondence |
| Plane geometry | `/calibrations/plane-mappings`; supplied camera/view applicability and physical plane coordinates |
| Stereo rig | `/calibrations/stereo-rigs`; explicit camera assets, paired views and capture identities |
| Validation/storage | `/calibration-validations`, `/inspections`, `/imports`, `/exports`; normalized values, no server registration |
| Change physical origin | `/calibration-origin-adjustments`; creates a new returned asset, not in-place mutation |
| Image correction | `/image-undistortions`; separate corrected image and nonlinear source mapping |
| Stitch canvas | `/image-stitches`; returned canvas dimensions/frame, source/camera mappings and provenance |
| Geometry conversion | `/geometry/stereo-transforms`, `/camera-projections`, `/length-conversions` |

The abbreviated paths in this table use the `/api/v2` prefix; the complete routes/fields appear
in the [field reference](v2-api-reference.md). Detection output ordering is explicit: a partial
checkerboard lattice detection is not automatically a camera-calibration correspondence order.
Inspect accepted/rejected-view diagnostics and residuals before publishing calibration assets.

A recipe may declare up to eight correction/stitch `stages`. For normal correction, inverse
mapping recovers original source measurements; use a metric applicable to that recovered frame.
For separately supplied corrected images, declare their actual `source_frame` and matching metric.
Stitching must be the first stage and supplies paired sources explicitly. A plane metric on a
stitched canvas must use the same canonical calibration digest and canvas frame as the stitch stage.
After origin adjustment, update both stitch and metric asset references together. Scalar canvas
calibration must match actual canvas frame/dimensions.

For later raster display of a stitched run, request `retain_stage_images: true` with output when
running, or provide an independently published canvas image. Declare its canvas `display_frame`.
The run's original paired `sources` do not imply a retained final canvas raster. Projective/nonlinear
circle display uses mapped sample curves, not an arbitrary radius multiplier.
See [scalar/camera details](../measure_api/v2/CALIBRATIONS.md) and
[plane/stereo details](../measure_api/v2/PLANE_STEREO.md) for complete assumptions and correspondences.

## Errors, retries and operational records

### Error envelope

```json
{
  "error": {
    "code": "recipe.invalid_parameter",
    "message": "Recipe parameters are invalid",
    "details": {},
    "field_path": "recipe.nodes.0.parameters.selection",
    "trace_id": "product-123",
    "retryable": false
  }
}
```

The message/details are illustrative. Branch on stable `code` and `retryable`, preserve details
and trace identity, and show located `field_path` in the editor. An admitted failed operation may
also include `operation_id`; operation polling embeds `error` in the operation envelope.

| HTTP | Representative codes | Upstream action |
|---|---|---|
| `400` | `request.invalid_json` | Fix JSON encoding |
| `404` | `file.not_found`, `operation.not_found`, `trace.not_found` | Resolve missing files or unknown records |
| `409` | `file.changed`, `file.digest_mismatch`, `operation.idempotency_conflict`, `output.conflict` | Reconcile content/key/output identity before retry |
| `410` | `operation.expired`, `trace.expired` | Use the upstream archive; retry deduplication is no longer guaranteed |
| `422` | `request.validation`, `recipe.invalid_parameter`, `file.output_forbidden`, `result.invalid` | Fix typed parameters, permissions, references or supplied records |
| `429` | `operation.capacity`, `operation.retention_capacity`, `operation.memory_limit` | Back off when retryable; for memory/retention problems adjust workload/resources |
| `500` | `calculation.failed` | Record diagnostics/trace; do not assume a valid measurement |
| `503` | `service.not_ready`, `service.unavailable`, `service.state_unavailable`, `service.journal_unavailable`, `worker.interrupted` | Recover availability/state and retry only according to actual retryability/outcome |

The list is representative, not an exhaustive code registry. Numeric/domain failure can also be
a successful calculation containing invalid observations. Cancellation/timeout are terminal states
and should be handled as such; the service does not promise a separate HTTP `504` timeout contract.

### Idempotency and uncertain network outcomes

Keys are scoped by `X-Caller-Id` and calculation operation ID. Identity covers canonical prepared
parameters and verified transitive input content, including `timeout_ms`. `mode`, `wait_ms` and
the idempotency key itself are excluded from the parameter fingerprint. A same-key retry can change
sync/async waiting behavior and retrieve the retained original operation; changed recipe, image,
asset bytes or budget conflict with that retained key. Output destinations are calculation
parameters too. A new caller identity is a different retry namespace.

When a POST response is lost:

1. If the operation ID is known, query it; do not repeatedly submit new operations.
2. If the ID is unknown, resubmit the same request/key/caller within retention to reconcile it.
3. If content/parameters must change, deliberately allocate a new request identity/key.
4. A retained terminal failure remains that failure for the same key; a deliberate new attempt
   uses a new key after handling its cause. Never treat `retryable: true` as overriding deduplication.

Capacity/readiness failures can occur before acceptance. Return to bounded submission/backoff
with jitter based on the actual error. Async deduplication still prepares files; inputs must remain
available even when reconciling an old key. No guaranteed `Retry-After` header is defined.

Operation retention defaults to 24 hours after terminal completion, followed by a 24-hour bounded
expiration marker. Known expired IDs return `410` during that marker window; forgotten/unknown
IDs return `404`. After expiry a resubmission can create new work: this is finite retry deduplication,
not permanent exactly-once delivery. Upstream stores authoritative results and product state.

With a persistent `state_directory`, restart recovers committed results/retry identities and marks
unfinished calculations `worker.interrupted`. The retained journal is operational state, not an
upstream project/result database. Committed output files remain owned by upstream.

### Diagnostics

Record request trace/operation IDs and exact configuration/runtime provenance. Diagnostics include
queue, worker-startup, preparation, admission and compute durations; available stage timings/cache
counters; selected-sample/fit quality; and measured peak worker memory. Budget clocks use
`CLOCK_MONOTONIC`. `compute_ms` excludes queue/startup/preparation and is not end-to-end latency.
Fit RMS describes residuals against selected samples and is not physical dimensional accuracy.
Client-visible latency and total service/container peak memory require separate upstream monitoring.

## API families and further reference

The [complete route and schema catalog](v2-api-reference.md) covers all 65 routes and 222
OpenAPI components. The family guides explain algorithm/domain restrictions beyond schema bounds:

| Family | Integration details |
|---|---|
| Image and operation lifecycle | [Image/lifecycle guide](../measure_api/v2/README.md) |
| Edges, line/circle fits, composition, independent batches and product rules | [Atomic measurement guide](../measure_api/v2/ATOMIC_MEASUREMENTS.md) |
| Templates, similarity, rigid/control alignment and transforms | [Template/alignment guide](v2-template-alignment-contract.md) |
| Scalar/camera calibration | [Calibration guide](../measure_api/v2/CALIBRATIONS.md) |
| Plane/stereo, origins, stitching and projection | [Plane/stereo guide](../measure_api/v2/PLANE_STEREO.md) |
| Validation, preview, target-local execution and batches | [Recipe guide](../measure_api/v2/RECIPES.md) |
| Vector/raster/diagnostics and CSV/JSON | [Result guide](../measure_api/v2/RESULTS.md) |
| Packages, legacy conversion, frozen replay | [Portable/replay guide](../measure_api/v2/PORTABLE_RECIPES.md) |

Legacy conversion is explicit one-way migration using `/legacy-project-conversions`. Supported
JSON/core-numeric forms produce diagnostics and a validated recipe; unsupported/pickle/ambiguous
configuration is not silently reconstructed. Check `valid` and blocking diagnostics before saving
the new upstream version. UI layout, zoom, visibility and selection remain upstream preferences.

## Runnable example and document verification

The [executable example](v2/examples/quickstart.py) calls a real running HTTP service, atomically
publishes isolated synthetic inputs/versioned assets, teaches a template, calculates scale,
validates/previews a recipe, runs asynchronously, archives pure results, visualizes/exports,
packages/imports/relocates assets and replays the frozen run. Every invocation uses a unique
directory namespace and new keys. Request/response snapshots are written to
`client-records/<namespace>/` on the upstream side. They are useful for inspecting actual payloads.

Set up a shared demo root and a local config. Execute from the repository root with the installed
V2 runtime; this fixture generator also needs the runtime's NumPy/OpenCV dependencies:

```bash
python - <<'PY'
import json
from pathlib import Path

demo = (Path('output') / 'v2-api-demo').resolve()
roots = {name: demo / name for name in ('exchange', 'assets', 'outputs')}
for directory in roots.values():
    directory.mkdir(parents=True, exist_ok=True)
settings = {
    'shared_roots': {name: str(directory) for name, directory in roots.items()},
    'writable_roots': ['outputs'],
    'state_directory': str(demo / 'state'),
    'max_workers': 2,
    'max_pending': 16,
}
(demo / 'settings.json').write_text(json.dumps(settings))
PY
python -m measure_api.v2 --config output/v2-api-demo/settings.json --port 9004
```

In another terminal, using the same interpreter/environment:

```bash
python docs/v2/examples/quickstart.py \
  --base-url http://127.0.0.1:9004 --shared-dir output/v2-api-demo
```

Expected summary: 12 succeeded calculations, physical widths approximately `[2.0, 2.4]` mm,
aggregate product `fail`, and `replay_equivalent: true`. The synthetic demo proves integration
behavior only. The example uses the [standard-library HTTP client](../scripts/v2_http_client.py);
it can poll and preserve structured failures without importing the engine into upstream.
That example client returns terminal records; upstream must still inspect validity/conformity.
Its per-request network timeout is 75 seconds. Production callers requesting longer operation
budgets should implement configurable transport timeouts and reconciliation using the rules above.

Regenerate the schema snapshot and linked field reference after API changes:

```bash
python docs/v2/generate_reference.py
```

The generator uses the executable FastAPI models/registry without starting workers or reading
shared assets. The reference snapshot has 63 OpenAPI path entries plus the two automatically served
documentation/schema routes. Numerical/deployment evidence and pending production qualification
are described in [the acceptance record](v2-acceptance.md). Representative camera captures,
independent physical truth, capture repeatability and agreed production performance thresholds
remain prerequisites for production qualification; actual Docker runtime/mount validation must
be completed on the target node.

### Documentation verification on 2026-10-06 (Asia/Shanghai)

The example was executed against a real CLI/Uvicorn HTTP process using only the locked headless
runtime. All 12 calculations succeeded; the measured demo widths were 2.0 and 2.4005069037207925 mm,
product evaluation was `fail`, and frozen replay was equivalent with no differences.
The verifier checked all 12 saved operation records and nine output manifests, including every
declared artifact's actual byte size/digest. The live OpenAPI document matched the checked-in
snapshot exactly. All 15 JSON examples validated against their actual public request/result models;
all 287 route/schema anchors were unique and resolved, and local document links were checked.
Evidence is retained locally under ignored `output/v2-api-doc-verification-48ece3c9a7/`.
This verifies documentation/examples against the implementation; it adds no production-accuracy
or container-qualification claim.

# Measure API — Complete Endpoint Reference

Base URL: `http://<host>:<port>` (default `5000`)
Content-Type: `application/json` for all requests/responses
Traceability: Every response includes `X-Trace-Id` header — correlate with logs and call records

---

## Session Management

### `POST /api/session` — Create a new session

```json
// Request
{"project_dir": "/path/to/project/dir"}
// Response 200
{"session_id": "a1b2c3d4e5f6", "status": "created"}
// Error 400
{"error": "project_dir is required"}
```

### `GET /api/session/<sid>` — Get session status & phase

```json
// Response 200
{
  "phase": "has_measurements",  // phases: created | reference_loaded | template_ready | has_measurements | has_composed | measured
  "project_dir": "/path/...",
  "has_reference": true,
  "has_template": true,
  "template_shape": [1616, 1591],
  "num_measurements": 2,
  "num_composed": 0,
  "dag_valid": true
}
// Error 404
{"error": "Session 'xxx' not found"}
```

### `DELETE /api/session/<sid>` — Delete a session

```json
// Response 200
{"status": "deleted", "session_id": "a1b2c3d4e5f6"}
```

### `GET /api/sessions` — List all active sessions

```json
// Response 200
{"sessions": {"sid1": "phase1", "sid2": "phase2"}}
```

---

## Reference & Template

### `POST /api/session/<sid>/reference` — Load reference image

```json
// Request
{"image_path": "/abs/path/to/reference.jpg"}
// Response 200
{"width": 2448, "height": 2048, "path": "/abs/.../reference.jpg"}
```

### `POST /api/session/<sid>/template` — Define template ROI

```json
// Request — "center", "size", "angle_deg" are required; the rest are optional
{
  "center": [1237.8, 993.6],         // [row, col] of ROI center
  "size": [1616.7, 1591.5],          // [height, width] of ROI
  "angle_deg": 0.0,                  // rotation angle
  "preprocessor": "raw",             // raw | canny | sobel | clahe | threshold
  "match_score_threshold": 0.5,      // NCC score (0.0-1.0)
  "angle_range_deg": 30,             // HALF-range: searches ±30°
  "max_matches": 0,                  // 0=unlimited, 1=single, N=max
  "angle_step": 1.0,                 // fine-search step in degrees
  "coarse_angle_step": 5.0,          // coarse pass step (only when coarse_fine)
  "coarse_fine": true,               // two-pass search: coarse then refine
  "overlap": 0.3                     // min IoU between accepted matches (multi-target)
}
// Response 200
{"template_shape": [1616, 1591]}
```

The four trailing fields are commonly omitted from call code and silently take their
defaults. `angle_step` matters most: a coarse 1.0° step on a large ROI is slow, while a
too-coarse value can miss the true orientation. All four are persisted with the model.

---

## Measurement Objects — CRUD

Primitive measurement types — **the complete set**. Any other name returns 400.

> Machine-generated and always current: see
> [generated/object_contracts.md](generated/object_contracts.md). The table below is
> hand-written for readability; trust the generated one if they disagree.

| Type | Result | Required params | Optional params (default) |
|---|---|---|---|
| `FitCircle` | circle | `center`, `radius`, `measure_length1`, `measure_length2` | `num_measures`(12), `sigma`(1.0), `threshold`(30.0), `transition`("all"), `start_phi`(0.0), `end_phi`(2π), `radius_min`(null), `radius_max`(null) |
| `FitLine` | line | `start`, `end`, `measure_length1`, `measure_length2` | `num_measures`(10), `sigma`(1.0), `threshold`(30.0), `transition`("all") |
| `EdgePair` | point | `row`, `col`, `angle`, `length1`, `length2` | `sigma`(1.0), `threshold`(30.0), `interpolation`("linear") |
| `EdgePoint` | point | `row`, `col`, `angle`, `length1`, `length2` | `sigma`(1.0), `threshold`(30.0), `interpolation`("linear") |
| `TemplateMatchPoint` | point | `row`, `col` | `template_size`(40), `preprocessor_type`("raw"), `match_score_threshold`(0.5), `angle_range_half`(15.0), `angle_step`(1.0), `use_subpixel`(true), `pyramid_decimate`(0), `pyramid_max_template_size`(400) |

Unknown names are rejected with a 400 that lists the valid set — but only once you are in
an allowed phase (see Common Errors below). There is no `FitEdge`, `FitRectangle`,
`FindCircle` or `FindLine`. Single edges are `EdgePoint` / `EdgePair`; circles are always
`FitCircle`.

### `POST /api/session/<sid>/measurements` — Add + test on template

| Query param | Default | Description |
|---|---|---|
| `include_visual` | `false` | Return base64 PNG overlay of measurement on template |

```json
// Request
{
  "object_type": "FitCircle",
  "label": "circle_1",
  "params": {
    "center": [814.59, 760.89],       // [row, col]
    "radius": 484.15,
    "measure_length1": 60.0,          // search length outward
    "measure_length2": 10.0,          // search length inward (0=centered)
    "num_measures": 12,               // number of sampling lines
    "sigma": 1.0,                     // smoothing
    "threshold": 5.0,                 // edge threshold
    "transition": "negative",         // positive | negative | all
    "start_phi": 0.0,                 // start angle (radians)
    "end_phi": 6.283185307179586      // end angle (radians, 2π=full)
  }
}
// Response 200
{
  "label": "circle_1",
  "object_type": "FitCircle",
  "valid": true,
  "result": {
    "type": "circle",
    "valid": true,
    "center_row": 816.374,
    "center_col": 783.995,
    "radius": 478.641,
    "meta": {"num_points": 12, "mean_error": 3.9, "max_error": 9.86}
  },
  "quality": {
    "num_edges": 12,
    "expected_edges": 0,
    "coverage_ratio": 0.0,
    "edge_amplitude_mean": 45.2
  },
  "elapsed_ms": 4.6,
  "visual_b64": "..."  // only if include_visual=true
}
```

`quality` is **sparse**: every key is conditional, so test with `.get()` rather than
indexing. It is `{}` entirely when the measurement failed. Which keys appear depends on
`object_type`:

| Key | Present for | Source / caveat |
|---|---|---|
| `num_edges` | `FitCircle`, `FitLine`, `EdgePoint`, `EdgePair` | from `meta.num_points` |
| `expected_edges` | same four | from `meta.num_measures` — **the fitters do not record this**, so it is usually `0` |
| `coverage_ratio` | same four | `0.0` whenever `expected_edges == 0`; a zero ratio alone is **not** a defect signal |
| `rms` | never in practice | looked up as `meta.rms`/`meta.fit_rms`, which the engine does not emit — use `meta.mean_error` instead |
| `edge_amplitude_mean` / `_min` | all four primitives | absent when no amplitudes were recorded |

Composed types (`TwoPointsDistance`, …) get **no** quality keys — their `quality` is `{}`.
For fit accuracy read `result.meta.mean_error` and `meta.max_error` (pixels), which *are*
populated for `FitCircle` and `FitLine`.

### `PUT /api/session/<sid>/measurements/<label>` — Update params + re-test

Request body same as `POST` — only `params` is used (partial merge).

### `DELETE /api/session/<sid>/measurements/<label>` — Delete (cascades)

```json
// Response 200
{"status": "deleted", "label": "circle_1", "cascaded": [{"label": "dist_12", "reason": "depends on circle_1 via ...", "action": "removed"}]}
```

### `GET /api/session/<sid>/measurements` — List all

```json
// Response 200
{
  "measurements": [{"label": "circle_1", "object_type": "FitCircle", "params": {...}}],
  "composed": [{"label": "dist_12", "composed_type": "TwoPointsDistance", "dependencies": {...}}]
}
```

### `GET /api/session/<sid>/measurements/<label>` — Get single definition

### `POST /api/session/<sid>/measurements/test` — Test without saving

For parameter exploration before committing. Request same as POST.

---

## Composed Measurements

Composed types — **the complete set**. A composed type takes no tuning params, only
dependencies.

The name is `PointLineDistance`, **not** `PointToLineDistance`.

| Type | Result | Dependency slot | Slot accepts result types |
|---|---|---|---|
| `TwoPointsDistance` | distance | `point_a_label`, `point_b_label` | point, circle |
| `TwoPointsLine` | line | `point_a_label`, `point_b_label` | point, circle |
| `TwoLinesAngle` | angle | `line_a_label`, `line_b_label` | line |
| `PointLineDistance` | distance | `point_label` | point, circle |
| | | `line_label` | line |
| `PointCircleDistance` | distance | `point_label` | point, circle |
| | | `circle_label` | circle |

**Dependencies are type-checked at add time.** A `point` slot also accepts a fitted
circle, because `CircleResult` exposes `row`/`col` as aliases for its centre and the
shipped models depend on that. Any other mismatch is rejected:

```json
{"error": "Dependency 'line_1' (key=point_a_label) produces 'line' but TwoPointsDistance requires 'point'"}
```

A `FitCircle` therefore satisfies both `point_*` and `circle_label` slots. An `EdgePair`
exposes virtual endpoint labels `<label>_A` / `<label>_B`, which point slots may reference —
but **only those**: a valid EdgePair publishes no bare-label result, so referencing the pair
by its own name is rejected with a message pointing at both endpoints.

### `POST /api/session/<sid>/composed` — Add composed measurement

```json
// Request
{
  "composed_type": "TwoPointsDistance",
  "label": "dist_12",
  "dependencies": {
    "point_a_label": "circle_1",
    "point_b_label": "circle_2"
  }
}
// Response 200
{"label": "dist_12", "valid": true, "result": {"value": 123.45, "value_deg": null}, "quality": {...}}
```

### `DELETE /api/session/<sid>/composed/<label>` — Delete composed

### `GET /api/session/<sid>/composed` — List all composed

---

## DAG

### `GET /api/session/<sid>/dag?format=json` — Get dependency graph

```json
// Response 200
{
  "nodes": [
    {"label": "circle_1", "type": "FitCircle", "category": "primitive"},
    {"label": "circle_2", "type": "FitCircle", "category": "primitive"},
    {"label": "dist_12", "type": "TwoPointsDistance", "category": "composed"}
  ],
  "edges": [
    {"from": "circle_1", "to": "dist_12", "role": "point_a_label"},
    {"from": "circle_2", "to": "dist_12", "role": "point_b_label"}
  ],
  "execution_order": ["circle_1", "circle_2", "dist_12"],
  "is_valid": true
}
```

---

## Persistence

### `POST /api/session/<sid>/save` — Save project to disk

```json
// Response 200
{"saved_to": "/path/to/project", "files": ["<project_dir>/config.json", "<project_dir>/template.npz"]}
```

`save()` writes three files (only two are listed in `files`):

| File | Contents |
|---|---|
| `config.json` | `version: 2`, phase, `measurement_defs`, `composed_defs` |
| `template.npz` | template pixels + workflow state |
| `reference.png` | reference image (grayscale) |

> **This format is v1-only and is not interchangeable with the GUI / v2 model asset.**
> v2 `:publish` and the GUI write a manifest-v3.0 asset (`manifest.json` +
> `reference.npy` + `workflow.npz`). A v1 project directory cannot be imported by v2,
> and vice versa.

### `POST /api/session/<sid>/load` — Load saved project

Reads from the session's own `project_dir`; the request body is ignored.

```json
// Response 200
{"phase": "has_measurements", "has_template": true, "template_shape": [1616, 1591],
 "num_measurements": 2, "num_composed": 0}
```

---

## Measurement

### `POST /api/session/<sid>/measure` — Run full pipeline

| Query param | Default | Description |
|---|---|---|
| `include_visual` | `false` | Return base64 PNG of the **whole scene** — every detected target with all its measurements overlaid. Unlike `/measurements`, which overlays only that one tool. |

```json
// Request
{"inspection_image": "/abs/path/to/inspection.jpg"}
// Response 200
{
  "status": "ok",
  "elapsed_ms": 2727.3,
  "num_targets": 1,
  "targets": [
    {
      "target_id": 1,
      "score": 0.6522,
      "row": 1234.5,
      "col": 987.6,
      "rotation_deg": -5.1,
      "scale": 1.0,
      "valid": true,
      "measurements": {
        "circle_1": {
          "valid": true, "type": "circle", "label": "circle_1",
          "center_row": 816.374, "center_col": 783.995,
          "radius": 478.641,
          "meta": {"num_points": 12, "mean_error": 3.9, "max_error": 9.86,
                   "min_radius": 478.6, "max_radius": 478.7, "ellipticity": 2e-11}
        },
        "circle_2": {
          "valid": true, "type": "circle", "label": "circle_2",
          "center_row": 831.08, "center_col": 807.04,
          "radius": 611.24, "meta": {"num_points": 12}
        }
      }
    }
  ],
  "visual_b64": "..."  // only if include_visual=true
}
```

`target_id` is **1-based** and matches the numbering the GUI result table shows. It is not
a positional index into `targets`.

`measurements` is keyed by tool **label**, so the same names you used when adding tools come
back here. Values are plain dicts already mapped to original-image coordinates — no
rectification needed on the client side. Each valid entry carries `type`, `valid`, `label`
plus its value keys (`meta` too, for primitives). Note that a circle entry inside `/measure`
has **no** `row`/`col` keys (use `center_row`/`center_col`), unlike the `add_measurement`
response above.

Two failure shapes to code against:

- An **invalid** measurement appears as just `{"type": ..., "label": ..., "valid": false}` —
  no value keys. Check `valid` before reading anything else.
- If a tool **raised**, `measurements` gains a string-valued `"_error"` key naming the
  exception, and every tool registered after it was skipped — those labels are missing
  entirely. Treat `_error` as "results incomplete", and never assume a dict shape for it.

---

## Health

### `GET /api/health`

```json
// Response 200
{"status": "ok", "sessions": 0}
```

---

## Common Errors

Every error body has exactly one key:

```json
{"error": "Unknown primitive type 'FitEdge'. Supported: ['EdgePair', 'EdgePoint', 'FitCircle', 'FitLine', 'TemplateMatchPoint']"}
```

| Status | Meaning |
|---|---|
| `400` | Invalid request (missing fields, unknown object_type, duplicate label, state machine violation) |
| `404` | Session not found |
| `500` | Server error (check logs for trace_id) |

State-machine violations surface as **400**, not 409 — the message names the phase you are
in and the ones that would be accepted. Read `GET /api/session/<sid>` before retrying.

**Order of checks matters.** The phase guard runs *before* argument validation, so while
you are in a disallowed phase an unknown `object_type` or bad parameter will not be
reported — you only see the phase error. Fix the phase first.

---

## Config File (`config.yaml` / `config.local.yaml`)

Key settings that can be adjusted without code changes:

```yaml
log:
  level: "INFO"            # DEBUG | INFO | WARNING | ERROR
  directory: "logs"        # log output directory

call_records:
  enabled: true
  directory: "call_records"

server:
  port: 5000
  max_sessions: 10
```

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
// Request
{
  "center": [1237.8, 993.6],         // [row, col] of ROI center
  "size": [1616.7, 1591.5],          // [height, width] of ROI
  "angle_deg": 0.0,                  // rotation angle
  "preprocessor": "raw",             // raw | canny | sobel | clahe | threshold
  "match_score_threshold": 0.5,      // NCC score (0.0-1.0)
  "angle_range_deg": 30,             // ±search range
  "max_matches": 0                   // 0=unlimited, 1=single, N=max
}
// Response 200
{"template_shape": [1616, 1591]}
```

---

## Measurement Objects — CRUD

Measurement types: `FitCircle`, `FitLine`, `FitEdge`, `FitRectangle`, `FindCircle`, `FindLine`, etc.

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
    "expected_edges": 12,
    "coverage_ratio": 1.0,
    "rms": 3.9,
    "edge_amplitude_mean": 45.2,
    "edge_amplitude_min": 12.1
  },
  "elapsed_ms": 4.6,
  "visual_b64": "..."  // only if include_visual=true
}
```

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

Composed types: `TwoPointsDistance`, `TwoLinesAngle`, `PointToLineDistance`, etc.

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
{"saved_to": "/path/to/project", "files": ["config.json", "template.npz"]}
```

### `POST /api/session/<sid>/load` — Load saved project

```json
// Response 200
{"phase": "has_measurements", "has_template": true, "num_measurements": 2, ...}
```

---

## Measurement

### `POST /api/session/<sid>/measure` — Run full pipeline

| Query param | Default | Description |
|---|---|---|
| `include_visual` | `false` | Return base64 overview PNG |

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
      "target_id": 0,
      "score": 0.6522,
      "row": 1234.5,
      "col": 987.6,
      "rotation_deg": -5.1,
      "scale": 1.0,
      "valid": true,
      "measurements": {
        "circle_1": {
          "valid": true, "type": "circle",
          "center_row": 816.374, "center_col": 783.995,
          "radius": 478.641
        },
        "circle_2": {
          "valid": true, "type": "circle",
          "center_row": 831.08, "center_col": 807.04,
          "radius": 611.24
        }
      }
    }
  ],
  "visual_b64": "..."  // only if include_visual=true
}
```

---

## Health

### `GET /api/health`

```json
// Response 200
{"status": "ok", "sessions": 0}
```

---

## Common Errors

| Status | Meaning |
|---|---|
| `400` | Invalid request (missing fields, wrong parameters, state machine violation) |
| `404` | Session not found |
| `500` | Server error (check logs for trace_id) |

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

# Gantry pick HTTP API

Stateless HTTP service for a three-axis gantry.  Each calibration call returns
one 3x3 homogeneous matrix; the calling software stores both matrices and sends
them back to `/locate`.  The service computes XY only and never talks to a
robot/PLC; Z is deliberately ignored.

The service **never receives an image**.  Target detection belongs to the
caller — the v1 measurement API, the v2 `measurements.circles` operation, or the
equipment's own recipe — which passes the sub-pixel centre in as `(u, v)`.
Every request is `application/json`.

> 面向同事的完整流程、现场操作和排错说明见
> [`PICK_MODULE_GUIDE.md`](PICK_MODULE_GUIDE.md)。

## Run

```bash
python -m measure_api.pick_server
# or:
uvicorn --factory measure_api.pick_server:create_app --host 127.0.0.1 --port 8100
```

Dependencies: `fastapi`, `uvicorn` (plus the existing `numpy`/`opencv` engine
dependencies).

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `PICK_RUNS_DIR` | `data/pick_runs` | debug run root |
| `PICK_DEBUG` | `0` | start with the global debug switch on |
| `PICK_ANNOTATED_MAX_PX` | `2000` | annotated long edge, legacy image runs only |
| `PICK_LOCATE_KEEP_DAYS` | `2` | locate run retention |
| `PICK_LOCATE_KEEP_MAX` | `500` | locate run retention |
| `PICK_CALIBRATION_KEEP_DAYS` | `90` | vision/offset run retention |
| `PICK_CALIBRATION_KEEP_MAX` | `100` | vision/offset run retention |
| `PICK_CORS_ORIGINS` | empty | comma-separated browser origins |
| `PICK_HOST` / `PICK_PORT` | `127.0.0.1` / `8100` | bind address |

There is no server-side recipe file and no `/status` endpoint: the software
owns the two matrices and the detector.

## Coordinate conventions

| Name | Type | Meaning |
|---|---|---|
| image point | `[u, v]` | `u = column`, `v = row`, pixels, sub-pixel allowed |
| `image_size` | `[width, height]` | the frame the points were measured on |
| robot pose/point | `[x, y]` | controller units (normally mm); a trailing Z is accepted and ignored |

Every image point is validated against `image_size`.  A point outside
`[0, W-1] × [0, H-1]` is rejected with `400 point_out_of_image`, which also
catches a caller that sends `(row, col)` instead of `(col, row)`.

## Matrix conventions

All matrices are row-major 3x3.

### Vision matrix

```text
V = [[a00, a01, b0],
     [a10, a11, b1],
     [0,   0,   1 ]]
```

`/locate` uses only the upper-left 2x2:

```text
A = V[:2, :2]
camera_center_xy = robot_pose + A @ (image_centre - image_uv)
```

`b` is specific to the calibration target position and is **not** used by
`/locate`; it is kept for storage/audit.

The image centre is `((W-1)/2, (H-1)/2)`, so `image_size` must match the frames
the matrix was calibrated on.  Changing resolution, cropping or scaling
invalidates the matrix — `/calibrate/vision` echoes `image_size` and
`principal_point` so the caller can store them next to the matrix.

### Offset matrix

```text
O = [[1, 0, dx],
     [0, 1, dy],
     [0, 0, 1 ]]
```

```text
gripper_xy = camera_center_xy + (dx, dy)
```

## Endpoints

### `POST /api/v1/calibrate/vision`

`application/json`:

| Field | Type | Required | Meaning |
|---|---|---|---|
| `image_size` | `[w, h]` | yes | frame size the points were measured on |
| `image_points` | `[[u, v], ...]` | yes | 4 or more detected centres, in `robot_points` order |
| `robot_points` | `[[x, y], ...]` | yes | same length; a trailing Z is accepted |
| `principal_point` | `[u0, v0]` | no | defaults to the image centre |
| `confidences` | `[float, ...]` | no | per-point detector confidence |

```bash
curl -X POST http://127.0.0.1:8100/api/v1/calibrate/vision \
  -H 'content-type: application/json' \
  -d '{
        "image_size": [5472, 3648],
        "image_points": [[2458.040, 2279.497], [3206.108, 1334.445],
                         [3308.557, 2183.056], [2356.346, 1431.227]],
        "robot_points": [[10.928, 321.647], [20.928, 331.647],
                         [10.928, 331.647], [20.928, 321.647]]
      }'
```

Response:

```json
{
  "ok": true,
  "matrix": [
    [-0.001321362642334, -0.011627365736710, 40.681749989616],
    [ 0.011604128501497, -0.001396038632585, 296.3037814102695],
    [ 0.0,                0.0,                1.0]
  ],
  "rms_px": 0.207,
  "rms_mm": 0.00242,
  "principal_point": [2735.5, 1823.5],
  "image_size": [5472, 3648],
  "warnings": ["low_image_coverage"]
}
```

`rms_px` / `rms_mm` / `warnings` are the only quality signal now that the
service sees no image: store them next to the matrix, and re-calibrate when
`rms_px` grows.  A fit that fails the quality gate is `422 calibration_invalid`.

### `POST /api/v1/calibrate/offset`

`application/json`:

```json
{"samples": [{"place_pose": [118.0, 89.0], "center_pose": [110.0, 85.0]}]}
```

```json
{
  "ok": true,
  "matrix": [
    [1.0, 0.0, 8.0],
    [0.0, 1.0, 4.0],
    [0.0, 0.0, 1.0]
  ],
  "offset_xy": [8.0, 4.0],
  "std_xy": [0.0, 0.0],
  "num_samples": 1,
  "warnings": []
}
```

`offset_xy` is the mean of `place_pose - center_pose`.  Samples that disagree
by more than 0.5 mm per axis (`std_xy`) are rejected with
`422 offset_invalid` — averaging them would produce a plausible but wrong
grasp correction.

### `POST /api/v1/locate`

`application/json`:

| Field | Type | Required | Meaning |
|---|---|---|---|
| `image_size` | `[w, h]` | yes | must match the calibration frames |
| `image_uv` | `[u, v]` | yes | detected target centre |
| `robot_pose` | `[x, y]` | yes | flange XY at capture time |
| `vision_matrix` | 3x3 | yes | from `/calibrate/vision` |
| `offset_matrix` | 3x3 | no | from `/calibrate/offset` |
| `radius_px` | float | no | echoed into the debug run only |

```bash
curl -X POST http://127.0.0.1:8100/api/v1/locate \
  -H 'content-type: application/json' \
  -d '{
        "image_size": [5472, 3648],
        "image_uv": [2836.937, 1806.012],
        "robot_pose": [15.928, 326.647],
        "vision_matrix": [[-0.001321362642334, -0.011627365736710, 40.681749989616],
                          [ 0.011604128501497, -0.001396038632585, 296.3037814102695],
                          [ 0, 0, 1]],
        "offset_matrix": [[1, 0, 123.67973454793294],
                          [0, 1, 1.4771626790364583],
                          [0, 0, 1]]
      }'
```

```json
{
  "ok": true,
  "camera_center_xy": [15.8587, 325.4455],
  "gripper_xy": [139.5384, 326.9227],
  "delta_xy": [-0.0693, -1.2015]
}
```

- `offset_matrix` omitted: `gripper_xy` is `null`.
- `delta_xy` is `A @ (image_centre - image_uv)`: the relative move that puts
  the target at the image centre.
- A bad/singular matrix is `400 bad_matrix`; a point outside the frame is
  `400 point_out_of_image`.
- The response never includes Z.

## Debug switch

Debug is global; it is **not** a per-request parameter.

```http
GET /api/v1/debug
PUT /api/v1/debug   {"enabled": true}
```

- Off: no `pick_runs/<run_id>` directory.
- On: each call publishes a self-contained run directory and adds a single
  `run_id` to the business response.
- The switch is in memory and resets to `PICK_DEBUG` on restart.

## Debug artifact layout

```text
data/pick_runs/<run_id>/
├── manifest.json
├── inputs.json            # image_size, points, poses, matrices
├── result.json            # matrix/coordinates + per-sample diagnostics
├── points.json            # vision only: u, v, x, y, residual_px per point
└── README.txt
```

Runs carry no images: the service no longer decodes frames.  Runs written by
older versions (with `images/` and `overview.jpg`) still list and archive
correctly, and `pick_replay` handles both generations.

No `recipe_snapshot.json`: there is no server-side recipe.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/runs?limit=20` | recent runs |
| `GET` | `/api/v1/runs/{run_id}` | manifest |
| `GET` | `/api/v1/runs/{run_id}/archive` | ZIP for reproduction |
| `GET` | `/files/{run_id}/...` | artifact JSON (legacy runs also serve images) |

## Replay

```bash
python -m measure.pick_replay data/pick_runs/<run_id>
```

The CLI re-runs the math and compares the matrix or coordinates with
`result.json`.  Coordinate runs replay from `inputs.json` alone; legacy image
runs re-run the circle locator first.

## Errors

```json
{"ok": false, "error": "point_out_of_image", "message": "..."}
```

| HTTP | `error` | Meaning |
|---|---|---|
| 400 | `bad_request` | missing/invalid field, fewer than 4 points, bad `image_size` |
| 400 | `point_count_mismatch` | `image_points` and `robot_points` differ in length |
| 400 | `point_out_of_image` | a point lies outside `image_size` (often a row/col swap) |
| 400 | `bad_matrix` | matrix not 3x3, wrong last row, singular, or offset not a translation |
| 422 | `calibration_failed` | degenerate correspondences (collinear or singular) |
| 422 | `calibration_invalid` | fit quality gate failed; see `rms_px` and `warnings` |
| 422 | `offset_invalid` | no samples, or samples disagree beyond `std_xy` |
| 500 | `internal_error` | unexpected failure |
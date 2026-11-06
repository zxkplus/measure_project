# Gantry pick HTTP API

Stateless-in-business, stateful-in-calibration HTTP service for a three-axis
gantry.  The service computes XY coordinates only; the calling software moves
the gantry and captures images.  Z is deliberately ignored.

## Run

```bash
python -m measure_api.pick_server
# or:
uvicorn --factory measure_api.pick_server:create_app --host 127.0.0.1 --port 8100
```

Dependencies: `fastapi`, `uvicorn`, `python-multipart` (plus the existing
`numpy`/`opencv` engine dependencies).

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `PICK_RUNS_DIR` | `data/pick_runs` | debug run root |
| `PICK_RECIPE_PATH` | `data/pick_recipe.json` | active calibration recipe |
| `PICK_DEBUG` | `0` | start with the global debug switch on |
| `PICK_ANNOTATED_MAX_PX` | `2000` | annotated long edge |
| `PICK_LOCATE_KEEP_DAYS` | `2` | locate run retention |
| `PICK_LOCATE_KEEP_MAX` | `500` | locate run retention |
| `PICK_CALIBRATION_KEEP_DAYS` | `90` | vision/offset run retention |
| `PICK_CALIBRATION_KEEP_MAX` | `100` | vision/offset run retention |
| `PICK_CORS_ORIGINS` | empty | comma-separated browser origins |
| `PICK_HOST` / `PICK_PORT` | `127.0.0.1` / `8100` | bind address |

## Debug switch

Debug is global; it is **not** a per-request parameter.

```http
GET /api/v1/debug
PUT /api/v1/debug   {"enabled": true}
```

- Off: no `pick_runs/<run_id>` directory, no images, no `inputs.json`/`result.json`.
- On: every business call publishes a self-contained run directory.
- The active `pick_recipe.json` is always saved; otherwise a restart would lose
  the calibration.  Debug only controls the per-call evidence package.
- The switch is in memory and resets to `PICK_DEBUG` on restart.

## Endpoints

### `POST /api/v1/calibrate/vision`

`multipart/form-data`:

| Field | Type | Meaning |
|---|---|---|
| `images` | file, 4+ | same-size frames; order matches points |
| `robot_points` | JSON string | `[[x, y], ...]`, one per image |

```bash
curl -X POST http://127.0.0.1:8100/api/v1/calibrate/vision \
  -F "images=@LL.jpg" -F "images=@LT.jpg" \
  -F "images=@RB.jpg" -F "images=@RT.jpg" \
  -F 'robot_points=[[10.228,308.451],[26.428,309.193],[5.638,345.111],[22.058,346.681]]'
```

Response (debug off):

```json
{"ok": true, "rms_px": 1.889, "warnings": ["high_residual"]}
```

With debug on it additionally returns `run_id`, `overview`, and one entry per
image with `uv`, `confidence`, `residual_px`, `original`, and `annotated` URLs.

### `POST /api/v1/calibrate/offset`

`application/json`:

```json
{"samples": [{"place_pose": [118.0, 89.0], "center_pose": [110.0, 85.0]}]}
```

```json
{"ok": true, "offset_xy": [8.0, 4.0], "std_xy": [0.0, 0.0], "warnings": []}
```

With debug on it additionally returns `run_id`.

### `POST /api/v1/locate`

`multipart/form-data`:

| Field | Type | Meaning |
|---|---|---|
| `image` | file | current frame |
| `robot_pose` | JSON string | flange `[x, y]` at capture time |

```bash
curl -X POST http://127.0.0.1:8100/api/v1/locate \
  -F "image=@shot.jpg" -F 'robot_pose=[100.0,200.0]'
```

```json
{
  "ok": true,
  "camera_center_xy": [96.0, 203.0],
  "gripper_xy": [104.0, 207.0],
  "confidence": 0.93
}
```

`gripper_xy` is `null` when the offset has not been calibrated; the camera
coordinate is still returned so the offset workflow can centre the camera.

With debug on it additionally returns `run_id`, `annotated`, and `original`.

### Status and runs

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/status` | active recipe, readiness, debug state |
| `GET` | `/api/v1/runs?limit=20` | recent debug runs |
| `GET` | `/api/v1/runs/{run_id}` | run manifest |
| `GET` | `/api/v1/runs/{run_id}/archive` | ZIP for reproduction |
| `GET` | `/files/{run_id}/...` | original/annotated images and JSON |

Run IDs look like `20261106-150834-vision-a1b2c3`.

## Debug run layout

```text
data/pick_runs/<run_id>/
├── manifest.json
├── inputs.json
├── result.json
├── recipe_snapshot.json
├── README.txt
├── overview.jpg              # vision only
└── images/
    ├── 00_original.jpg
    ├── 00_annotated.jpg
    └── ...
```

The annotation draws the detected circle in green, its centre in red, and a
footer with `uv`, radius, confidence, and robot XY.  Locate annotations also
draw the principal point in blue and an arrow to the target.  Original images
are preserved byte-for-byte.

## Replay

Download the run ZIP and run:

```bash
python -m measure.pick_replay data/pick_runs/<run_id>
```

The CLI re-detects, re-fits, and compares the result with `result.json`.

## Error format

```json
{"ok": false, "error": "no_target", "message": "circle not found in image"}
```

| HTTP | Meaning |
|---|---|
| 400 | malformed request, count mismatch |
| 409 | vision calibration missing/invalid |
| 422 | unreadable image, no target, size mismatch, invalid calibration |
| 500 | unexpected failure |

# Gantry pick HTTP API

Stateless HTTP service for a three-axis gantry.  Each calibration call returns
one 3x3 homogeneous matrix; the calling software stores both matrices and sends
them back to `/locate`.  The service computes XY only and never talks to a
robot/PLC; Z is deliberately ignored.  The default locator accepts circles with a
radius of 6–3000 px (large enough for a 5 MP bottle mouth).

> 面向同事的完整流程、现场操作和排错说明见
> [`PICK_MODULE_GUIDE.md`](PICK_MODULE_GUIDE.md)。

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
| `PICK_DEBUG` | `0` | start with the global debug switch on |
| `PICK_ANNOTATED_MAX_PX` | `2000` | annotated source long edge |
| `PICK_LOCATE_KEEP_DAYS` | `2` | locate run retention |
| `PICK_LOCATE_KEEP_MAX` | `500` | locate run retention |
| `PICK_CALIBRATION_KEEP_DAYS` | `90` | vision/offset run retention |
| `PICK_CALIBRATION_KEEP_MAX` | `100` | vision/offset run retention |
| `PICK_CORS_ORIGINS` | empty | comma-separated browser origins |
| `PICK_HOST` / `PICK_PORT` | `127.0.0.1` / `8100` | bind address |

There is no server-side recipe file and no `/status` endpoint: the software
owns the two matrices.

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
camera_center_xy = robot_pose + A @ (image_centre - target_uv)
```

`b` is specific to the calibration target position and is **not** used by
`/locate`; it is kept for storage/audit.

The image centre is fixed to `((W-1)/2, (H-1)/2)`.  Changing resolution,
cropping or scaling invalidates the matrix.

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

Response:

```json
{
  "ok": true,
  "matrix": [
    [-0.00126349, -0.01169656, 40.699445],
    [ 0.01166881, -0.00132630, 297.676806],
    [ 0.0,         0.0,         1.0]
  ]
}
```

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
  ]
}
```

### `POST /api/v1/locate`

`multipart/form-data`:

| Field | Type | Required | Meaning |
|---|---|---|---|
| `image` | file | yes | current frame |
| `robot_pose` | JSON string | yes | flange `[x, y]` at capture time |
| `vision_matrix` | JSON string | yes | 3x3 from `/calibrate/vision` |
| `offset_matrix` | JSON string | no | 3x3 from `/calibrate/offset` |

```bash
V='[[-0.00126349,-0.01169656,40.699445],[0.01166881,-0.00132630,297.676806],[0,0,1]]'
O='[[1,0,8],[0,1,4],[0,0,1]]'

curl -X POST http://127.0.0.1:8100/api/v1/locate \
  -F "image=@shot.jpg" \
  -F 'robot_pose=[90.0,180.0]' \
  -F "vision_matrix=$V" \
  -F "offset_matrix=$O"
```

```json
{
  "ok": true,
  "camera_center_xy": [90.975, 181.975],
  "gripper_xy": [98.975, 185.975]
}
```

- `offset_matrix` omitted: `gripper_xy` is `null`.
- `vision_matrix` required; a bad/singular matrix is `400 bad_matrix`.
- Target not found / low confidence: `422`.
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
├── inputs.json            # includes the matrices used by /locate
├── result.json            # matrix/coordinates + diagnostics
├── README.txt
├── overview.jpg           # vision only
└── images/
    ├── 00_original.jpg
    ├── 00_annotated.jpg
    └── ...
```

No `recipe_snapshot.json`: there is no server-side recipe.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/runs?limit=20` | recent runs |
| `GET` | `/api/v1/runs/{run_id}` | manifest |
| `GET` | `/api/v1/runs/{run_id}/archive` | ZIP for reproduction |
| `GET` | `/files/{run_id}/...` | original/annotated images and JSON |

## Replay

```bash
python -m measure.pick_replay data/pick_runs/<run_id>
```

The CLI re-runs detection/calibration and compares the matrix or coordinates
with `result.json`.

## Errors

```json
{"ok": false, "error": "no_target", "message": "circle not found in image"}
```

| HTTP | Meaning |
|---|---|
| 400 | malformed request, count mismatch, bad/singular matrix |
| 422 | unreadable image, no target, low confidence, invalid calibration |
| 500 | unexpected failure |

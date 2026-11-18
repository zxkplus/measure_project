# Gantry pick operations in V2

The existing V2 application serves all four operations through its normal calculation
registry and prewarmed worker pool. Start V2 as usual; no additional FastAPI application,
process owner or listening port is required.

| Method | Path | Calculation ID | Result |
|---|---|---|---|
| POST | `/api/v2/calibrate/vision-points` | `calibrate.vision-points` | candidate `matrix`, quality and per-point residuals |
| POST | `/api/v2/calibrate/vision` | `calibrate.vision` | `matrix`: 3x3 vision affine |
| POST | `/api/v2/calibrate/offset` | `calibrate.offset` | `matrix`: 3x3 offset transform, `effective_model` and fit diagnostics |
| POST | `/api/v2/locate` | `locate` | `camera_center_xy`, optional `gripper_xy` / `gripper_delta_xy` |

All requests use `application/json`. Image inputs are upstream-owned `FileReference`
objects within the configured shared roots, as for other V2 calculations. Multipart
uploads from the standalone `/api/v1` service must be replaced with shared-file
references. There is no dependency on `measure_api.pick_server`.

## Vision calibration

Supply at least four same-size images, each containing the same stationary circular
target observed from a different flange XY pose. Image and robot-point ordering must
match. Use spatially distributed, noncollinear samples. Robot poses contain `[x, y]`
or `[x, y, z]`; Z is ignored. Coordinate units follow the controller, normally mm.

For a deployment with an `exchange` shared root:

```bash
BASE_URL=http://127.0.0.1:9004
curl -sS -X POST "$BASE_URL/api/v2/calibrate/vision" \
  -H 'Content-Type: application/json' \
  -d '{
    "images": [
      {"root": "exchange", "path": "calibration/LL.png"},
      {"root": "exchange", "path": "calibration/LT.png"},
      {"root": "exchange", "path": "calibration/RB.png"},
      {"root": "exchange", "path": "calibration/RT.png"}
    ],
    "robot_points": [
      [89.025, 193.025], [111.025, 193.025],
      [89.025, 207.025], [111.025, 207.025]
    ]
  }'
```

A successful V2 response contains `state: "succeeded"` and a result of this form
(matrix values below are illustrative):

```json
{
  "matrix": [
    [0.05, 0.0, 84.025],
    [0.0, 0.05, 188.025],
    [0.0, 0.0, 1.0]
  ]
}
```

The matrix is row-major `V = [[A, b], [0, 1]]`. Its upper-left 2x2 `A` maps pixel
displacements to flange displacements. The translation column `b` belongs to the
calibration target position and is retained for audit; location uses only `A`.
Store the complete `result.matrix` as the `vision_matrix` for later calls.

## Vision calibration from taught circle centres

Use `/api/v2/calibrate/vision-points` after upstream template matching and circle
fitting. This operation consumes the supplied centres directly and does not run
the default circle locator or discard samples through robust fitting.

```json
{
  "image_points": [[100, 100], [540, 100], [100, 380], [540, 380]],
  "robot_points": [[89.025, 193.025], [111.025, 193.025], [89.025, 207.025], [111.025, 207.025]],
  "image_size": [640, 480],
  "calib_z": 30
}
```

Coordinates must be finite, the two point lists must have equal length of at least
four, and each pixel centre must lie in the original image. `image_size` is
`[width, height]`, both greater than one; `calib_z` records the common capture plane.
The caller must ensure a single camera, resolution, height and stationary target.

The result contains `matrix`, `valid`, `warnings`, ordered `residuals` with
`residual_px` and `residual_mm`, plus `rms_px`, `max_px`, `rms_mm`, `max_mm`.
Robot residuals use controller units (normally mm). A solvable fit failing the
shared quality gate returns a **successful candidate calculation with
`valid: false`** and diagnostics; it must not be applied. Degenerate or singular
points produce `pick.calibration_failed`. Unlike the old image route, quality
rejection is inspectable in the candidate result. Independent centring verification
and explicit persistence remain the caller's responsibility.

## Gripper offset calibration

For each sample, `place_pose` is the flange pose when the gripper places the target,
and `center_pose` is the flange pose when the camera is centred over that target.
The operation fits `place_pose ~ matrix @ center_pose`.  With one sample, or when
the translation model explains every sample, `matrix` is the identity plus the
mean of `place_pose[:2] - center_pose[:2]`.

`model` selects the fit: `auto` (default) keeps translation while its residual
stays within 0.5 controller units, `translation` never fits a linear part, and
`affine` fits one from four or more samples (six recommended).

```bash
curl -sS -X POST "$BASE_URL/api/v2/calibrate/offset" \
  -H 'Content-Type: application/json' \
  -d '{
    "samples": [
      {"place_pose": [118.0, 89.0], "center_pose": [110.0, 85.0]}
    ]
  }'
```

The operation returns this value under `result`:

```json
{
  "matrix": [
    [1.0, 0.0, 8.0],
    [0.0, 1.0, 4.0],
    [0.0, 0.0, 1.0]
  ],
  "model": "auto",
  "effective_model": "translation",
  "rms_mm": 0.0,
  "max_mm": 0.0,
  "condition": 1.0,
  "rotation_deg": 0.0,
  "scale": 1.0,
  "shear": 0.0
}
```

Store `matrix` as `offset_matrix`.  `effective_model` says whether a linear part
was kept; `rms_mm` and `max_mm` are the residuals of that model.

## Location

Supply the flange pose at capture time, the current image and the caller's stored
matrices. The server does not retain an active calibration or recipe.

```bash
curl -sS -X POST "$BASE_URL/api/v2/locate" \
  -H 'Content-Type: application/json' \
  -d '{
    "image": {"root": "exchange", "path": "images/shot.png"},
    "robot_pose": [90.0, 180.0],
    "vision_matrix": [
      [0.05, 0.0, 84.025], [0.0, 0.05, 188.025], [0.0, 0.0, 1.0]
    ],
    "offset_matrix": [
      [1.0, 0.0, 8.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]
    ]
  }'
```

For a 640x480 image with the target at pixel `(300, 200)`, the successful operation
has this `result`:

```json
{
  "camera_center_xy": [90.975, 181.975],
  "gripper_xy": [98.975, 185.975],
  "gripper_delta_xy": [8.975, 5.975]
}
```

Both poses are **flange poses**, not absolute target coordinates:

```text
c = ((width - 1) / 2, (height - 1) / 2)
camera_center_xy = robot_pose[:2] + V[:2, :2] @ (c - target_uv)
gripper_xy = offset_matrix @ [camera_center_xy, 1]
gripper_delta_xy = gripper_xy - robot_pose[:2]
```

`offset_matrix` is applied as a full homogeneous transform.  With a translation
matrix this is `camera_center_xy + offset[:2, 2]`; with a fitted affine it is
not, so use `gripper_delta_xy` for a single-move command instead of adding
`delta_xy` and the offset.

Omit `offset_matrix` to calculate only camera centring; `gripper_xy` and
`gripper_delta_xy` are then JSON `null`. No response contains Z, and these
operations never command the robot/PLC.
The shared circle locator uses a default radius range of 6–3000 px.

## Scheduling, errors and diagnostics

These are native V2 calculations, not direct `{ok, matrix}` responses. All four
accept `mode`, `wait_ms`, `timeout_ms` and `idempotency_key`. HTTP 200 returns the
normal `OperationResponse` with a typed `result`. Async work or wait expiry returns
HTTP 202 with a `Location` header and an `operation_id`; query
`GET /api/v2/operations/{operation_id}?wait_ms=10000` until terminal state.

File references are pinned to SHA-256 during preparation and reverified on use.
The existing caller-scoped idempotency, deadlines, cancellation, worker replacement,
retained results, `X-Trace-Id` and operation diagnostics apply to all four routes.
No output directory or writable image root is required. These four calculations
do not create standalone `pick_runs` artifacts or use its global debug switch;
query `/api/v2/operations/{operation_id}/diagnostics` for V2 lifecycle diagnostics.

| HTTP | Code | Meaning |
|---|---|---|
| 422 | `request.validation` | Invalid counts, pose shape, matrix shape or non-finite numbers |
| 422 | `pick.calibration_failed` | Missing calibration target, differing image sizes or degenerate observations |
| 422 | `pick.calibration_invalid` | Fitted vision calibration failed its quality gate |
| 422 | `pick.offset_invalid` | Offset samples could not produce a valid correction |
| 400 | `pick.bad_matrix` | Singular/non-affine vision matrix, or an offset that is singular, mirrored or outside the plausible scale range |
| 422 | `pick.no_target` | Location target is absent or below the confidence gate |
| 404 / 409 / 422 | `file.*`, `image.*` | Normal V2 missing-file, content, boundary or image validation errors |

Calculation failures use the V2 `error` envelope with `operation_id`, rather than
returning a matrix or coordinates. Preparation/schema failures may occur before an
operation is accepted. Existing service capacity and availability errors also apply.

Keep camera orientation, resolution, pixel scale and camera-to-target-plane distance
consistent with calibration. The affine is a planar displacement model, not a 3D
camera calibration. It can convert same-plane pixel endpoint differences to physical
lengths using `norm(V[:2, :2] @ (p2 - p1))`, in the controller's unit. The translation
column must not be used to infer an arbitrary target's absolute robot coordinates.

Run the public HTTP acceptance tests with:

```bash
python -m pytest tests/v2/test_gantry_pick.py -q
```

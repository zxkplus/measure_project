# Shared-plane and stereo calculations

The `plane_stereo` calculation family is headless and independently callable. Every route uses
`CalculationRequest` scheduling controls, verified shared inputs, the standard operation envelope
and process executor. No calculation mutates or registers upstream calibration assets. Numerical
acceptance uses HTTP; the existing camera/stereo library regression suite remains applicable.
Synthetic/rasterized boards demonstrate mathematics and integration, not a production accuracy
threshold. Upstream must validate representative optics, target planes and capture synchronization.

## Public operations and callable seams

| POST route | Request model | Result model | Calculation callable |
| --- | --- | --- | --- |
| `/calibrations/plane-mappings` | `PlaneRequest` | `PlaneResult` | `compute_plane` |
| `/calibrations/stereo-rigs` | `StereoRequest` | `StereoResult` | `compute_stereo` |
| `/calibration-origin-adjustments` | `OriginRequest` | `PlaneStereoResult` | `adjust_origin` |
| `/image-stitches` | `StitchRequest` | `StitchResult` | `stitch_images` |
| `/geometry/stereo-transforms` | `StereoTransformRequest` | `StereoTransformResult` | `transform_stereo` |
| `/geometry/camera-projections` | `CameraProjectionRequest` | `CameraProjectionResult` | `project_points` |

These models and callables are public in `measure_api.v2.families.plane_stereo`. Recipe execution
may validate the models and call the registered operation with its child `CalculationContext`.
Asset kinds `plane` and `stereo` register via `CALIBRATION_ASSETS` and work with the common
calibration validation, inspection, import and export routes. Typed asset/result models expose
their complete payload in OpenAPI, including `PlaneParameters` and `StereoParameters`.

## Plane inputs and canvas

`PlaneRequest.views` contains one or two `PlaneViewInput` values. Each supplies `camera` (the
common `CalibrationInput` with exactly one inline `value` or pinned shared `file`), exact
`applicability`, `capture_id`, and exactly one `observation` or board `image` reference.
All views must share the capture identity and setup, with distinct cameras. An accepted
`CameraObservation.view_id` equals its capture identity. This pairing declaration identifies the
same physical board at the same instant; no capture pairing is inferred from filename order.

Accepted observations provide `image_size`, ordered `image_points` in source x/y pixels,
`object_points` in physical x/y/z and `ordering`. Boards currently require at least six unique
noncollinear z=0 correspondences in the declared request `unit` (`mm`, `cm`, `m`, `um`). Both
cameras in a pair must use identical object points and ordering. An optional observation `source`
is verified and dimension checked, and its pinned digest is retained. The caller is responsible
for the association of supplied manual observations with the declared camera/capture identity.

Image observations require `grid` (feature columns/rows) and positive physical `spacing`.
`symmetric`, `refine` and `blob` use the same dot-grid detection settings as intrinsic calibration.
`blob` exposes min/max area, circularity, convexity and dark/light color. Missing features return
an invalid domain outcome and no retained or previously computed asset.

`unit`, positive `pixel_size` (physical units per canvas pixel) and optional `destination_frame`
(default `plane_canvas`) define the shared canvas. Optional `output` commits canonical calibration
JSON and a manifest; compact asset values do not require file output. Camera intrinsics do not
supply a scene multiplier: physical scale comes from the accepted board coordinates.

The solve returns board-to-camera rotation/translation and per-point pixel reprojection errors.
Translation is in the enclosing plane asset's physical unit. Plane matrices operate on corrected
pinhole coordinates with K unchanged. Canvas bounds are the union of the corrected camera pixel
rectangle corners on the board plane. The origin is initially board x=0/y=0. A region crossing
its finite projective horizon is rejected rather than producing an unbounded canvas.

## Portable payload and mapping direction

`PlaneCalibration` has common schema version `1.0`, `kind: plane`, physical `unit`, primary
camera applicability and typed `PlaneParameters`. Parameters hold `views`, `canvas`, `origin`,
`physical_frame` (`board_plane`) and provenance. Each view embeds the immutable camera asset,
its canonical `camera_sha256`, capture identity, optional pinned source, board-to-camera pose,
and mutually inverse `camera_to_plane` / `plane_to_camera` projective matrices. These matrices
map corrected K-pinhole pixel sample coordinates to physical plane coordinates and back.
Imported assets validate poses, inverse matrices, camera digests, canvas geometry and mappings.

The canvas has width/height `image_size`, `pixel_size`, physical `unit`, pixel `frame`, physical
`bounds` `[x_min,y_min,x_max,y_max]` and inverse `plane_to_canvas` / `canvas_to_plane` matrices.
It includes the pixel-center endpoints; dimensions are ceil(span / pixel_size) + 1, with a
floating-point tolerance at integer boundaries. The physical origin need not coincide with the
canvas's top-left pixel. Canvas pixel units remain px; dimensions should be measured after
transforming endpoints or sample curves through `canvas_to_plane`.

Each view's `mapping` goes from original camera source pixels to shared canvas pixels. With
zero lens distortion it has `type: projective`, `matrix` and `inverse_matrix`. With nonzero
lens distortion it has `type: nonlinear` and no fabricated projective matrix. Details contain
K, D, corrected frame identity, `corrected_to_canvas`, its inverse, camera digest, physical unit
and a sample/curve policy. Correct original sample points using
`undistortPoints(source,K,D,P=K)` and then the corrected-to-canvas projective matrix. Reverse
canvas samples through canvas-to-plane and the view's board-to-camera pose and project using
K/D. No circle radius is preserved under projective correction; sample points/curves must be
mapped explicitly. No operation reconstructs arbitrary depth from 2D points.

## Stereo inputs and diagnostics

`StereoRequest` accepts `camera_a`, `camera_b`, exact `applicability_a`, `applicability_b`,
`pairs`, physical `unit`, `pixel_size`, optional destination frame and output. Camera identities
must be distinct and setup equal. Each `StereoPair` has one unique `capture_id`, and `a` / `b`
views whose explicit capture IDs must equal it. Each view has exactly one accepted observation
or board image. Image pairs use the same grid, spacing, refinement and blob options as planes.
Repeated numerical observation pairs are rejected even if renamed. At least three distinct
usable paired board poses are required. Rejected pairs retain their capture identity/reason;
accepted pairs retain observed/projected points, sources and errors.

`fix_intrinsics` defaults true. False performs actual intrinsic refinement with provided camera
values as initial guesses; it requires equal camera image dimensions and embeds newly calculated
camera values in the new output only. Original supplied camera values remain untouched.

`StereoCalibration` is a plane asset with two views and `StereoParameters`. Additional parameters
are proper `rotation`, `translation`, explicit `direction: camera_a_to_camera_b`, `baseline`,
essential/fundamental matrices and reprojection error. OpenCV's pose is forward A to B:
X_B = R X_A + T. T and baseline use the enclosing asset's unit. The inverse is
X_A = R-transpose (X_B - T). The first accepted paired board defines the shared physical plane;
its B pose is constrained by the solved stereo relationship. Provenance records original camera
identities, capture identities and whether intrinsics were refined.

Diagnostic errors are measured in source camera pixels. Every accepted pair includes both
cross-camera directions: solve board pose independently in B, transform to A using inverse R/T,
project into A with K/D and compare with observed A; and conversely A to B. Own-camera residuals
are also retained, so good monocular fits cannot conceal a bad relative pose. Aggregates include
mean, RMS, maximum and count, without invented manufacturing thresholds.

## Board reprojection validation and origin adjustment

The existing `CalibrationValidationRequest` additionally accepts nonempty typed `observations`,
each a `BoardValidationObservation` with `camera_id` plus the full accepted observation fields.
An explicit `unit` is required for physical observations and converted to the asset unit. Stereo
observations group by `view_id` (capture identity) and require both distinct rig cameras per group.
Plane observations are checked against the existing fixed plane pose; camera observations report
individual solve/reprojection errors. Optional `max_error_px` is the caller's acceptance policy;
it changes `valid` based on maximum observed error while preserving every per-point diagnostic.
No threshold is inferred when omitted. Malformed applicability, incomplete pairs and unusable
observations return actionable invalid calibration outcomes.

`OriginRequest` supplies the plane/stereo calibration, an `origin` point and explicit physical
`unit`, with optional output. The point is the location, in the OLD physical frame, that becomes
zero in the NEW frame. The calculation subtracts that point from all camera-to-plane mappings,
updates board-to-camera translations, recomputes canvas matrices and records cumulative origin
and `parent_sha256`. A stereo rig's relative R/T remains unchanged. Input files and assets remain
untouched; a supplied zero change may retain equivalent geometry while provenance still records
the parent computation. Revised plane validation observations use the revised physical frame.

## Stitching

`StitchRequest` accepts the supplied two-view plane or stereo `calibration`, exactly two `images`,
`blend`, optional `return_warped_views` and required `output`. Each image includes `camera_id`,
exact `applicability`, paired `capture_id` and verified shared `image`. Both product image views
must share their capture identity (distinct from the earlier calibration-board capture), match
rig cameras and have exact decoded dimensions. Mixed grayscale/BGR layouts are rejected.

The calculation corrects each source using its supplied K/D with K retained, warps the corrected
image and its valid support mask into the shared physical plane, and composites the two views.
`blend` supports `average`, `first` or `last`; the latter two follow input image order. Black input
pixels remain valid data, because support masks are warped independently of image intensities.
Canvas pixel count is checked against the configured deployment limit before allocation.

One atomic artifact set contains `stitched.png`, `coverage.png` (0, 1 or 2 supporting views), two
support masks and `mappings.json`. Optional warped view images are part of the same
commit. The final manifest marks completion. The result returns image metadata, pinned original
sources, per-view mappings, canvas, camera/calibration digests, capture identity and provenance.
The scene-plane assumption is explicit: targets off the calibrated board plane incur parallax.

## Independently callable 3D geometry

`StereoTransformRequest` has calibration, source/destination camera IDs, nonempty known 3D
`points`, input `unit` and `target_unit`. It supports both directions and same-camera identity.
The result states `camera:<id>` frames, transformed points and unit, source-to-destination 4x4
pose matrix, the matrix translation unit, and calibration digest. Convert input points to the
matrix unit before applying the pose, then convert to the requested output unit.

`CameraProjectionRequest` has supplied camera calibration, exact applicability, known 3D points,
physical input unit and `source_frame`. Points already in that camera's frame use
`camera:<camera_id>` and require no pose. Other object frames require `ProjectionPose` with a
proper rotation matrix, translation point and its unit, explicitly object-to-camera. The result
contains source/destination frames, pixel x/y coordinates, per-point inside-image flags and
camera digest. A point with nonpositive camera depth is rejected, not silently projected as if
visible. Known offscreen front-facing points remain meaningful and have inside_image false.

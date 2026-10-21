# Upstream-owned calibration calculations

All routes use the standard scheduling envelope and calculation worker. No call registers,
activates or mutates an upstream asset. Input and output FileReferences follow the shared-file
contract. Portable assets are canonical UTF-8 JSON, with schema version `1.0` and a SHA-256
content identity. Optional generation output and required export/correction output use the
atomic ArtifactWriter manifest contract.

## Calibration asset envelope

The common fields are `schema_version`, `kind`, `unit`, `applicability`, and `parameters`.
Units are `mm`, `cm`, `m`, or `um`. Applicability requires `camera_id`, `setup_id`,
`image_size` with width/height, and a named pixel `frame` (default `source`). Images are not
implicitly resized to fit an asset. The caller must use the exact applicable setup and frame.
Recipe metrics additionally bind applicability to the recovered sample frame: correction inverses
recover the original source; a separately supplied corrected image uses its declared corrected
frame. Stitched plane metrics use the same canonical calibration digest and canvas frame as the
stitch stage. When adjusting the plane origin, replace both recipe asset references together.
Scalar calibration of a stitched canvas requires its actual canvas frame and image dimensions.

Scalar kind `scalar` parameters contain positive `scale_per_px`, `spacing_px`, `square_size`,
and `assumption: uniform_orthographic`. The three numbers must be consistent. A scalar asset
states the caller's uniform/telecentric assumption; it does not estimate perspective or lens
parameters. Length conversion returns both the original pixel length and a separately named
physical length, target unit, frame and canonical calibration digest.

Camera kind `camera` parameters contain `camera_matrix` (positive pinhole 3 by 3 K),
`distortion` (4, 5, 8, 12 or 14 OpenCV coefficients), and `reprojection_error_px`.
Intrinsics alone do not establish physical scene scale. Length conversion of a camera or
plane asset rejects this unsupported operation: a plane calculation must transform endpoint
coordinates instead. A camera asset's physical unit is the unit of its calibration board and
pose diagnostics, not a pixel-length multiplier.

All asset consumers accept `calibration` with exactly one of `value` or `file`. File imports
validate and normalize the supplied JSON into typed values; the canonical normalized digest
may differ from a noncanonical input file's raw byte digest. Exported files use canonical
bytes, so their descriptor identity equals the returned canonical asset identity.

## Feature detection and calculation

Checkerboard detection accepts optional source ROI and inner-corner grid dimensions, plus
Shi-Tomasi and lattice refinement parameters. Corners are returned in source x/y coordinates.
The result reports detector ordering, method, refined corner count and spacing. A partial board
may fall back to the lattice detector and has no implicit camera correspondence ordering.
Manual scale calculation consumes supplied/edited corners and the physical square side; fewer
than six unique noncollinear points is an invalid observation, not a retained prior result.

Dot-grid detection consumes one image, feature grid dimensions and physical spacing. It emits
explicit paired image/object coordinates in row-major detector order. For asymmetric rows,
physical x is `(2 * column + row % 2) * spacing`. The caller can tune blob area, circularity,
convexity, color and subpixel refinement. A missing grid returns `valid: false`.

Camera calibration accepts exactly one of named image views or named accepted observations.
Image views require a grid and positive spacing. Observations require explicit paired physical
3D and image x/y coordinates, image dimensions, and ordering. Currently intrinsic calibration
uses planar board observations at z=0. Views must have at least six unique noncollinear pairs;
view IDs must be unique. Repeated observations are rejected rather than counted as new poses.
At least three distinct usable views are required. Production calibration should use enough
varied board poses and representative real camera images; synthetic acceptance does not prove
a production accuracy threshold.

Results include accepted/rejected view identities, rejection reasons, per-point residuals,
mean/RMS reprojection errors and pose vectors with an explicit translation unit. Image-based
accepted and rejected view diagnostics include verified source references. Compact assets
return as typed values and optionally as committed JSON.

## Image correction and mappings

Undistortion checks supplied applicability and decoded dimensions, then uses the existing
public camera correction computation. It creates a distinct corrected frame. The nonlinear
mapping contains K, D, corrected K, calibration digest and valid destination ROI. Its forward
point mapping is OpenCV undistortPoints with corrected K. A committed numeric NPZ stores
`source_x` and `source_y`, mapping each destination pixel center back to the source pixel.
This inverse map and the corrected PNG commit together under a final manifest. No affine
matrix is fabricated for nonlinear correction.

## Extension seam for plane/stereo

`calibration_assets` provides public asset/input/applicability models, canonical bytes and
digest, `validate_calibration`, `load_calibration`, and `check_applicability`. A calculation
family may expose `CALIBRATION_ASSETS` mapping each additional kind to its typed
CalibrationAsset subclass. Utilities discover these schemas and validate/import/inspect/export
all registered kinds without central route edits. Families should publish their full numeric
payload inside typed parameters and use the shared frame/unit conventions. Asset creation
helpers in the calibrations family return value, digest, size and optional committed artifact.

## Acceptance

Real HTTP tests cover manual and detected scale, ordered dot detection, image and accepted-point
intrinsics, incomplete/repeated/blank views, residuals, explicit unit conversion, portable asset
round-trip, schema discovery, applicability, actual image dimensions, atomic manifests, zero and
nonzero distortion point mappings. Existing pixel and camera/stereo regressions remain enabled.
The old small-grid fixture now uses the existing synthetic generator for distinct views; its
previous three identical images triggered a singular OpenCV 4.11 solve.

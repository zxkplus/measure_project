# 06: Calibrate shared planes and stereo rigs and stitch views

**What to build:** An upstream calibration workflow can establish a board-plane canvas and stereo
relationship, assess their errors, adjust a physical origin into a new asset and stitch paired views
for later measurement.

**Blocked by:** 05: Create scale and camera calibrations and correct images.

**Status:** resolved

**Spec user stories:** 54, 58, 59, 60, 61, 62, 63, 64.

- [x] Deliver board-plane mappings/canvas computation and stereo relative-pose calibration using
  explicitly paired capture identities and applicable camera assets.

- [x] Deliver per-point and aggregate board reprojection validation, calibration inspection and
  portable import/export for plane and stereo variants.

- [x] Deliver physical-origin adjustment as a new output without modifying the supplied calibration.

- [x] Deliver calibrated two-image stitching and declared mappings from source/corrected camera
  views to the shared plane canvas.

- [x] Deliver 3D camera-frame transforms and camera projection as independently callable geometry
  operations with explicit units and matrix direction.

- [x] Treat projective mappings as sample/curve mappings; do not silently preserve circle radius or
  claim arbitrary 3D reconstruction.

- [x] Return versioned calibration/image outputs using atomic commits and provenance linking paired
  images, cameras and supplied calibration versions.

- [x] Verify synthetic and representative paired camera cases, inconsistent pairing, canvas
  geometry, projection/reprojection residuals, origin adjustment and source asset preservation.

## Answer

Implemented shared-plane canvas calibration, paired stereo relative poses, board reprojection
validation, nonmutating physical-origin adjustment, calibrated image stitching, known 3D
camera-frame transformations and camera projections as independent FastAPI operations. Typed plane
and stereo asset variants extend the common inspection/import/export contract. Applicability,
pair identity, camera digests, units and forward/inverse matrix direction are checked explicitly;
nonlinear lens mappings retain their correction metadata and projective geometry uses samples.

Verification: all 64 public HTTP tests passed on the merged integration checkout on 2026-10-05,
covering plane/stereo together with image, template, alignment, atomic measurement and monocular
calibration families. The expanded common calibration-validation request remains compatible with
existing scalar/camera requests and supports paired board residual diagnostics. Stitching outputs
include independent support masks and one final manifest while preserving input sources. The
implementer reported 126 tests passed including numerical regressions and 22 calibration tests
passed under Python 3.9; those broad checks were not repeated without additional production changes.

Ticket 07's declared capability dependencies are now complete. Its compound execution must use
the shared preparation/staging context supplied by ticket 02, and only the parent final manifest
can mark a completed run; intermediate manifests cannot be treated as a completed parent result.
No top-level README case-collision difference or user-owned image artifact was staged.

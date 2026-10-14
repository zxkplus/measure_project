# 05: Create scale and camera calibrations and correct images

**What to build:** An upstream calibration editor can detect or manually amend calibration
observations, compute and inspect scale or camera assets, and correct a shared image while retaining
ownership of every input and generated calibration.

**Blocked by:** 01: Process shared images through the new FastAPI service.

**Status:** resolved

**Spec user stories:** 51, 52, 53, 54, 55, 56, 57, 60, 64.

- [x] Deliver checkerboard corner detection with ROI, lattice/corner refinement diagnostics and
  scale calculation from supplied or manually edited corners.

- [x] Deliver per-image dot-grid detection and intrinsic/distortion calibration from images or
  accepted observations with explicit ordering, physical coordinates, dimensions, rejected views and
  camera association.

- [x] Deliver image undistortion with nonlinear mapping metadata and applicable source/corrected
  frame identities.

- [x] Deliver scalar/monocular calibration validation, inspection and portable versioned
  import/export; inspection and conversion do not register, publish or mutate business assets.

- [x] Deliver explicit calibrated length conversion with separately identified pixel/physical
  results and supplied calibration provenance; intrinsics alone cannot establish millimeter scale.

- [x] Reject mismatched image/setup applicability, invalid physical dimensions, incomplete
  observations and unsupported geometry conversion with actionable outcomes.

- [x] Return generated assets as typed values or complete shared outputs with schema version, digest
  and size; use the shared-file commit contract.

- [x] Verify synthetic scale/intrinsic cases plus appropriate camera fixtures, manually edited
  corners, rejected views, reprojection residuals and undistortion frame mappings through public
  requests.

## Answer

Implemented scale and intrinsic calibration calculations as public FastAPI operations without
asset registration. Checkerboard and dot observations, supplied corner edits, camera views,
reprojection diagnostics, applicability checks, scalar length conversion, versioned asset
inspection/import/export and nonlinear image correction all follow the shared-file contract.
Generated calibration values include canonical content identity; optional files and correction
rasters/maps commit as complete artifact sets with a final manifest.

Verification: the merged integration checkout passed 20 public HTTP tests across calibration and
image-service acceptance on 2026-10-05. The implementer reported 82 passed tests including pixel,
camera and stereo numerical regressions. The existing small-grid camera fixture now generates
distinct observations instead of three identical images, avoiding a singular OpenCV 4.11 solve.
No production algorithm or user-owned image artifact changed in this ticket.

Plane and stereo asset families can extend the typed calibration asset registry, and generic
inspection/import/export utilities discover their schemas without central route edits. Ticket 06
is now unblocked.

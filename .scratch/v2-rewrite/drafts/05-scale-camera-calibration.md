# 05: Create scale and camera calibrations and correct images

**What to build:** An upstream calibration editor can detect or manually amend calibration
observations, compute and inspect scale or camera assets, and correct a shared image while retaining
ownership of every input and generated calibration.

**Blocked by:** 01: Process shared images through the new FastAPI service.

**Status:** needs-triage

**Spec user stories:** 51, 52, 53, 54, 55, 56, 57, 60, 64.

- [ ] Deliver checkerboard corner detection with ROI, lattice/corner refinement diagnostics and
  scale calculation from supplied or manually edited corners.

- [ ] Deliver per-image dot-grid detection and intrinsic/distortion calibration from images or
  accepted observations with explicit ordering, physical coordinates, dimensions, rejected views and
  camera association.

- [ ] Deliver image undistortion with nonlinear mapping metadata and applicable source/corrected
  frame identities.

- [ ] Deliver scalar/monocular calibration validation, inspection and portable versioned
  import/export; inspection and conversion do not register, publish or mutate business assets.

- [ ] Deliver explicit calibrated length conversion with separately identified pixel/physical
  results and supplied calibration provenance; intrinsics alone cannot establish millimeter scale.

- [ ] Reject mismatched image/setup applicability, invalid physical dimensions, incomplete
  observations and unsupported geometry conversion with actionable outcomes.

- [ ] Return generated assets as typed values or complete shared outputs with schema version, digest
  and size; use the shared-file commit contract.

- [ ] Verify synthetic scale/intrinsic cases plus appropriate camera fixtures, manually edited
  corners, rejected views, reprojection residuals and undistortion frame mappings through public
  requests.


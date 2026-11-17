# 06: Calibrate shared planes and stereo rigs and stitch views

**What to build:** An upstream calibration workflow can establish a board-plane canvas and stereo
relationship, assess their errors, adjust a physical origin into a new asset and stitch paired views
for later measurement.

**Blocked by:** 05: Create scale and camera calibrations and correct images.

**Status:** needs-triage

**Spec user stories:** 54, 58, 59, 60, 61, 62, 63, 64.

- [ ] Deliver board-plane mappings/canvas computation and stereo relative-pose calibration using
  explicitly paired capture identities and applicable camera assets.

- [ ] Deliver per-point and aggregate board reprojection validation, calibration inspection and
  portable import/export for plane and stereo variants.

- [ ] Deliver physical-origin adjustment as a new output without modifying the supplied calibration.

- [ ] Deliver calibrated two-image stitching and declared mappings from source/corrected camera
  views to the shared plane canvas.

- [ ] Deliver 3D camera-frame transforms and camera projection as independently callable geometry
  operations with explicit units and matrix direction.

- [ ] Treat projective mappings as sample/curve mappings; do not silently preserve circle radius or
  claim arbitrary 3D reconstruction.

- [ ] Return versioned calibration/image outputs using atomic commits and provenance linking paired
  images, cameras and supplied calibration versions.

- [ ] Verify synthetic and representative paired camera cases, inconsistent pairing, canvas
  geometry, projection/reprojection residuals, origin adjustment and source asset preservation.


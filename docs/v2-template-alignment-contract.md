# V2 template and alignment contract

These FastAPI operations use only supplied immutable shared files. No GUI module,
registration, session, recipe ownership or asset storage is required. Requests use
the common calculation scheduling controls, output namespace and manifest protocol.

## Teaching and assets

`POST /template-builds` accepts `image`, `roi` (center, full width/height and clockwise
angle), optional `anchor`, `support_mode`, `preprocessing` and `output`.
`tight_roi` rectifies exactly the supplied rectangle. `square_context` uses an
axis-aligned square of side max(width, height) around the same center, matching
GUI context semantics deliberately. Support must lie wholly inside the source.
The default anchor is the original ROI center; a different anchor remains a
reference-image point, even when it differs from the support center.

Preprocessing runs once on the full source image, followed by numeric support
extraction. Matching applies the same definition once to the full inspection
image. Sobel and CLAHE retain float32 values; PNG is a separate display preview.
The actual support dimensions, preprocessing, anchor, source content digest and
reference-to-support rectification mapping appear in `metadata`.

`template` is the directly usable file-reference output port. `artifacts` also
includes the numeric NPZ asset and PNG preview, followed by the final manifest.
NPZ schema 1.0 contains only non-pickle numeric pixels and a JSON metadata string;
portable assets can be retained or replaced by upstream without engine state.
`POST /template-inspections` validates both metadata and actual pixels, bounded
uncompressed size, version, finite data and invertible rigid rectification.
Constant spatial support, including uniform BGR color, cannot be taught.

## Localization and feature measurement

`POST /template-matches` accepts `image`, `template` and `search`. Search includes:

- `score_threshold`, `max_matches` (1 by default; 0 means up to the bounded 1000
  result limit), `overlap` (intersection over smaller rotated support area),
  optional axis-aligned `region` (x/y/full width/full height), and `subpixel`.
- `angle_min`, `angle_max`, `angle_step`: clockwise degrees from taught rectified
  support to inspection. Scale limits and step are positive dimensionless values.
  Single default angle 0 and scale 1 provide translation-only localization.
- `coarse_fine`, coarse angle/scale steps and `coarse_score_threshold` (0.3).
  Every coarse target neighborhood is refined through the configured fine grid;
  final observations always satisfy the final `score_threshold`.
- `pyramid_levels` controls powers-of-two decimation; the maximum template side
  also bounds the coarse resolution. Very small supports keep at least six
  samples along their smallest side. The actual `pyramid_scale` is returned.
  Anti-aliasing precedes decimation, and each coarse target is refined against
  original full-resolution numeric pixels. Pyramid candidates retain multiple
  targets, including odd-pixel translations.

Every match has center, mapped anchor, score, support-relative angle/scale,
reference-to-inspection `mapping` and support-to-inspection `support_mapping`.
Both matrices have their exact inverse and an explicit direction. The center
uses actual support pixel centers, not an assumed click or square. The full
reference mapping composes ROI rectification with detected support pose; callers
should use it to move taught geometry, especially for rotated teaching ROIs.
Overlap suppression runs across all poses. No match returns valid=false and an
empty matches list. Uninformative correlations are rejected, never reused.

`POST /measurements/template-points` shares localization controls but requires
max_matches=1. Its `point` output port uses the atomic positioned-point observation
shape (valid, reason, frame, unit, point). The supplied optional frame defaults to
source. This operation is also discoverable by the generic independent measurement
batch and directly callable by the recipe executor.

## Alignment and metrology

`POST /transforms/similarity-estimates` accepts paired source/destination xy points,
frame names, `estimate_scale`, optional robust RANSAC, `ransac_threshold`,
`max_residual`, and `minimum_inliers`. It estimates an orientation-preserving
similarity and returns residuals, accepted inliers, RMS, scale and clockwise angle.
Repeated/coincident controls, insufficient inliers or unacceptable residuals are
invalid observations with no mapping, rather than arbitrary transforms.

`POST /alignments/rigid` accepts an inspection image, `initial_mapping` from the
inspection frame to the desired aligned patch, full output width/height, `output`,
interpolation and explicit border policy/value. Similarity localization may
normalize scale. The result's `image` port references the actual aligned artifact;
`source` identifies the inspection input. `mapping` goes inspection-to-aligned;
`metric_mapping` goes aligned-to-original-inspection. Mapping normalized features
back through this inverse preserves actual dimensions. An aligned30px span under
scale0.5 still measures60px in original inspection pixels.

`POST /alignments/control-points` adds portable control templates, their desired
aligned-frame `reference_point`, local search radius/score threshold,
`minimum_controls`, robust/residual/scale options and `failure_policy`:
`fail` or `rigid_fallback`. Templates are constructed through the same independent
teaching interface. Each control search is confined to its own initially warped
target patch; adjacent targets cannot supply missing controls. If P is the initial
inspection-to-aligned mapping and Q maps desired controls to observed controls in
that patch, final alignment is inverse(Q) times P. The original image is warped
once with the final mapping, avoiding accumulated resampling. Results include each
control's match, inlier/residual, correction, fallback status and final inverse
metrology mapping. A failed refinement returns no output image or old geometry.

`POST /geometry/transforms` accepts a declared mapping and discriminated point,
line (a*x+b*y+c), circle, rectangle or polyline geometry. Points, polylines and
lines support invertible matrix mappings with finite-output checks. Circles and
rotated rectangles require a similarity: unrestricted affine/projective transforms
cannot claim preserved circle radius or metrology. The inverse must agree with
its forward matrix; mapped results identify the destination frame and unit.

## Acceptance

Real HTTP tests cover portable teaching/inspection, rectangle/square support,
arbitrary anchor, float preprocessing without repeated enhancement, translated,
clockwise-rotated/scaled and repeated targets, pyramid/odd-pixel sampling, score
and region/count limits, content replacement, weak matches, robust similarity,
typed geometry, correction direction, explicit fallback, adjacent-target isolation
and dimensional recovery after normalization. Existing numerical template and
transform regressions remain unchanged.

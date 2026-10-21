# 03: Measure edges and geometric dimensions through atomic APIs

**What to build:** An upstream measurement editor can measure a supplied image or point set, compose
returned geometry into dimensions, and evaluate supplied limits without creating a recipe or
registering assets. Results include valid geometry, selected samples and useful failure outcomes.

**Blocked by:** 01: Process shared images through the new FastAPI service.

**Status:** resolved

**Spec user stories:** 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 73, 76, 93.

- [x] Extract the relevant reusable measurement/composition behavior from GUI dependencies before
  extending it, with focused regressions preceding algorithm changes.

- [x] Deliver profiles, subpixel edge points/pairs, image-based lines/circles/arcs, fitting from
  supplied points, and independent measurement batches sharing an image.

- [x] Expose polarity, smoothing, threshold units, interpolation, probe extents/count,
  first/last/all selection, expected geometry, radius limits and arc ranges through validated
  discoverable parameters.

- [x] Select edge pairs after detection and pairing; return typed boundaries, center, width and
  separately typed multiple-pair arrays; missing observations never become valid zero dimensions.

- [x] Deliver two-point lines/distances, perpendicular point-to-line distances, smaller unsigned
  line angles, shortest point-to-circumference distances and circle dimensions with frame
  compatibility checks.

- [x] Use normalized a*x+b*y+c=0 coefficients consistently; the fitted-line then point-distance
  regression that previously produced about 100.5 pixels must produce about 40.5 pixels.

- [x] Deliver inclusive range, diameter-variation and centerline x-deviation calculations from
  supplied values and explicit unit/invalid-input policies; distinguish execution success,
  observation validity and evaluation status.

- [x] Return selected samples, requested/successful probe counts, amplitudes and genuine
  residual/RMS metrics where defined; exceptions and degenerate inputs cannot reuse a preceding
  successful result.

- [x] Verify numerical results and invalid cases through HTTP using synthetic lossless
  edges/lines/circles, including insufficient samples, coincident points, collinear circle inputs
  and explicit known tolerances.

## Answer

Implemented independent edge profiles/points/pairs, line/circle/arc observations, supplied-point
fits, geometric composition, business evaluations and shared-image measurement batches through
FastAPI. Reusable fit calculations were extracted from the legacy metrology classes without GUI
imports. Boundary selection occurs after pairing, normalized x/y line coefficients compose
correctly, and invalid samples/results do not retain a prior observation.

Verification: all 53 public HTTP tests passed on the merged integration checkout on 2026-10-05.
The horizontal fitted-line/point-distance regression now yields approximately 40.5 pixels rather
than 100.5. Shared-image batch acceptance includes an actual portable template-point calculation,
nested asset digest verification, independent failure outcomes and multiple numeric profiles
committed under a single complete parent manifest. This also completes ticket 04's deferred batch
acceptance. The implementer reported 21 atomic tests passing on both supported Python runtimes and
140 broad regression tests passing with the already absent sample fixture excluded. Broad
regressions were not repeated without additional production changes.

No top-level README case-collision difference or user-owned image artifact was staged.

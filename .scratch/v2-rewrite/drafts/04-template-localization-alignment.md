# 04: Build templates and localize aligned targets

**What to build:** An upstream teaching workflow can construct a portable template, match one or
several target instances, refine each pose using local controls and return mapped geometry or
template feature points without GUI callbacks or engine registration.

**Blocked by:** 01: Process shared images through the new FastAPI service.

**Status:** needs-triage

**Spec user stories:** 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 42, 50, 90, 93.

- [ ] Deliver template construction, inspection and single/multiple matching from shared
  upstream-owned files; outputs include actual support, anchor, preprocessing and versioned portable
  template metadata.

- [ ] Support explicit tight rectangular ROI and square-context modes; apply the same preprocessing
  definition exactly once during construction and matching.

- [ ] Expose rotation/scale ranges and steps, search regions, scores, maximum matches, overlap
  suppression and coarse/fine/pyramid controls.

- [ ] Deliver correspondence similarity estimation, rigid alignment, per-target control-point
  refinement and typed geometry transforms with explicit source/destination and matrix direction.

- [ ] Fix control-template initialization independently of GUI setup; expose minimum controls,
  robust fitting/residual settings and an explicit rigid-fallback versus failure policy.

- [ ] Deliver local template-feature point measurement and its variant in independent measurement
  batches.

- [ ] Keep control searches isolated to their own target instance; template construction, matching
  and alignment cannot depend on mutable state from an earlier request.

- [ ] Return enough inverse and metric-frame mappings for downstream measurements to recover actual
  target size after normalized localization; transformations do not claim unrestricted affine
  metrology.

- [ ] Verify translated, rotated, scaled and repeated targets, ROI support, weak matches,
  multi-target isolation, mapping direction and unchanged results after template-content replacement
  or cache eviction.


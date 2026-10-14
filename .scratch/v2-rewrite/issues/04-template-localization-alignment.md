# 04: Build templates and localize aligned targets

**What to build:** An upstream teaching workflow can construct a portable template, match one or
several target instances, refine each pose using local controls and return mapped geometry or
template feature points without GUI callbacks or engine registration.

**Blocked by:** 01: Process shared images through the new FastAPI service.

**Status:** resolved

**Spec user stories:** 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 42, 50, 90, 93.

- [x] Deliver template construction, inspection and single/multiple matching from shared
  upstream-owned files; outputs include actual support, anchor, preprocessing and versioned portable
  template metadata.

- [x] Support explicit tight rectangular ROI and square-context modes; apply the same preprocessing
  definition exactly once during construction and matching.

- [x] Expose rotation/scale ranges and steps, search regions, scores, maximum matches, overlap
  suppression and coarse/fine/pyramid controls.

- [x] Deliver correspondence similarity estimation, rigid alignment, per-target control-point
  refinement and typed geometry transforms with explicit source/destination and matrix direction.

- [x] Fix control-template initialization independently of GUI setup; expose minimum controls,
  robust fitting/residual settings and an explicit rigid-fallback versus failure policy.

- [x] Deliver local template-feature point measurement and its variant in independent measurement
  batches.

- [x] Keep control searches isolated to their own target instance; template construction, matching
  and alignment cannot depend on mutable state from an earlier request.

- [x] Return enough inverse and metric-frame mappings for downstream measurements to recover actual
  target size after normalized localization; transformations do not claim unrestricted affine
  metrology.

- [x] Verify translated, rotated, scaled and repeated targets, ROI support, weak matches,
  multi-target isolation, mapping direction and unchanged results after template-content replacement
  or cache eviction.

## Answer

Implemented portable teaching/inspection, rotation and scale localization with multiple targets,
coarse/fine and pyramid search, positioned template-feature observations, robust similarity,
rigid/control-point alignment and typed geometry transforms as FastAPI operations. The same
preprocessing definition is applied once, shared inputs retain verified content identity, and
alignment exposes inverse metrology mappings to recover true dimensions after normalization.
Control-point searches remain within their own target patches and explicitly fail or fall back.

Verification: all 32 merged HTTP tests across templates, alignment, images and calibrations passed
on 2026-10-05. The implementer reported 100 tests passed including existing numerical regressions.
The template-feature point operation exports the typed point port for recipe and batch dispatch.
Ticket 03 subsequently completed the combined measurement-batch HTTP acceptance using an actual
portable template asset and nested digest verification. All 53 merged public HTTP tests passed on
2026-10-05, completing this ticket's deferred final acceptance criterion.

No GUI runtime, template asset registration or prior-request geometry is used. No user-owned
image artifact changed in this ticket.

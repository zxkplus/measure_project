# 07: Validate, preview and execute complete supplied recipes

**What to build:** An upstream recipe editor can submit an unsaved configuration for validation or
selected-tool preview, then execute the same supplied version on one or many images and receive
per-target dimensions and evaluations. The engine invokes atomic calculations directly and does not
own recipe storage.

**Blocked by:** 03: Measure edges and geometric dimensions through atomic APIs; 04: Build templates
and localize aligned targets; 06: Calibrate shared planes and stereo rigs and stitch views.

**Status:** needs-triage

**Spec user stories:** 4, 5, 6, 7, 8, 9, 50, 51, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 93.

- [ ] Extract one GUI-independent recipe/multi-target execution implementation; reuse atomic
  functions without internal HTTP calls, GUI factories or duplicated scheduling behavior.

- [ ] Accept exactly one inline recipe or shared-file recipe reference; snapshot it, resolve
  explicit assets, echo upstream ID/version and return a configuration hash without inferring a
  latest version.

- [ ] Validate stable node IDs, named typed ports, single/array distinctions, enabled state, cycles,
  missing dependencies, required outputs, frames and template provenance before calculation.

- [ ] Preview selected nodes with dependency closure from an unsaved snapshot; echo upstream
  revision/preview token so the caller can reject stale responses.

- [ ] Execute declared correction/stitching, template matching or fixed ROI, per-target
  rigid/control refinement, atomic fitting and topologically ordered composed geometry.

- [ ] Map edge samples into the declared metric frame before fitting/dimension calculation; scaled
  normalization preserves actual size variation, and projective circle treatment uses supported
  mapped samples or explicit rejection.

- [ ] Return pixel and calibrated physical observations separately, apply supplied business
  policies, and separate operation state, measurement validity and conformity.

- [ ] Propagate invalid dependencies and required-output failures, continue unrelated targets,
  provide explicit no_targets behavior, and never reuse previous-request observations after
  exceptions.

- [ ] Deliver single-image and complete recipe batches with per-image/per-target outcomes; targets
  are local to a run and labels/display order do not alter dependency execution.

- [ ] Verify a headless HTTP teaching/preview/run sequence with calibrated measurements,
  scale-varying repeated targets, invalid dependencies, no targets, partial batches and control
  isolation.


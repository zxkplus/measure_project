# 07: Validate, preview and execute complete supplied recipes

**What to build:** An upstream recipe editor can submit an unsaved configuration for validation or
selected-tool preview, then execute the same supplied version on one or many images and receive
per-target dimensions and evaluations. The engine invokes atomic calculations directly and does not
own recipe storage.

**Blocked by:** 03: Measure edges and geometric dimensions through atomic APIs; 04: Build templates
and localize aligned targets; 06: Calibrate shared planes and stereo rigs and stitch views.

**Status:** resolved

**Spec user stories:** 4, 5, 6, 7, 8, 9, 50, 51, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 93.

- [x] Extract one GUI-independent recipe/multi-target execution implementation; reuse atomic
  functions without internal HTTP calls, GUI factories or duplicated scheduling behavior.

- [x] Accept exactly one inline recipe or shared-file recipe reference; snapshot it, resolve
  explicit assets, echo upstream ID/version and return a configuration hash without inferring a
  latest version.

- [x] Validate stable node IDs, named typed ports, single/array distinctions, enabled state, cycles,
  missing dependencies, required outputs, frames and template provenance before calculation.

- [x] Preview selected nodes with dependency closure from an unsaved snapshot; echo upstream
  revision/preview token so the caller can reject stale responses.

- [x] Execute declared correction/stitching, template matching or fixed ROI, per-target
  rigid/control refinement, atomic fitting and topologically ordered composed geometry.

- [x] Map edge samples into the declared metric frame before fitting/dimension calculation; scaled
  normalization preserves actual size variation, and projective circle treatment uses supported
  mapped samples or explicit rejection.

- [x] Return pixel and calibrated physical observations separately, apply supplied business
  policies, and separate operation state, measurement validity and conformity.

- [x] Propagate invalid dependencies and required-output failures, continue unrelated targets,
  provide explicit no_targets behavior, and never reuse previous-request observations after
  exceptions.

- [x] Deliver single-image and complete recipe batches with per-image/per-target outcomes; targets
  are local to a run and labels/display order do not alter dependency execution.

- [x] Verify a headless HTTP teaching/preview/run sequence with calibrated measurements,
  scale-varying repeated targets, invalid dependencies, no targets, partial batches and control
  isolation.

## Answer

Implemented one headless supplied-recipe executor with public validation, selected-node preview,
single-image runs and complete batches. Inline/file configuration expands before acceptance,
resolves calibration schemas, pins transitive assets and verifies teaching provenance. Static typed
ports, array/single distinctions, frames/units, required outputs and DAG ordering are validated;
preview executes dependency closure and echoes upstream revision/token without saving a draft.

Execution reuses atomic calculations directly, performs declared correction/stitch and target-local
alignment, maps actual observed samples before fitting/composition, and separates pixel/physical
observations, validity and business acceptance. Invalid dependencies and exceptions cannot reuse
old geometry; independent targets/items continue, with explicit no-target policy. Runtime stage
contexts remain temporary and only a single parent manifest completes retained outputs.

Verification: all 105 merged public HTTP tests passed on 2026-10-06. Recipe acceptance exercises
actual template normalization yielding 24 source pixels and 2.4 mm, mapped line/circle sample fits,
shared-file identity changes, preview closure/tokens, same-run control isolation, invalid policies,
partial batches, correction/stitch metrology and retained stage images. The implementer reported all
25 recipe HTTP tests passed under Python 3.11 and Python 3.9. The temporary-stage root canonicalization
handles macOS /var symlinks; no top-level README difference or user-owned image was staged.

Final snapshot verification: the retained stitched measurement_image reference is now computed
before result.json serialization. Both HTTP and the file snapshot retain the same verified canvas
reference and stage provenance. A new public HTTP test reads the snapshot file, reopens that canvas
through image inspection and measures its actual 20px edge pair without temporary aliases. All
26 recipe tests passed after the fix on the merged integration checkout. The parent independently
verified all 105 pre-fix HTTP tests under Python 3.9.6 (87.20s), and the implementer verified the
two updated stitching variants under Python 3.9 after the fix.

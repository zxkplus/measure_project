# V2 rewrite review and resolution

Review baseline: `3b46f24`. Reviewed implementation: `9f8f3eb`.
Diff: `git diff 3b46f24...9f8f3eb`. The two axes were reviewed independently.
The public HTTP seam was explicitly confirmed by the upstream integrator.

The reports retain reviewed source line numbers. Resolution is recorded separately
so implementation corrections do not erase the original findings. First follow-ups
refer to `7ccf62a`; final verification is recorded below after the supplemental fix.

## Standards

### Original report

Reviewed `git diff 3b46f24...9f8f3eb` and its commit list. Imported planning/tracker documents and unchanged legacy conventions are excluded. No configured lint gate enforces the items below.

**Documented-standard violations**

1. **Public annotations:** `measure_api/v2/files.py:352` declares `filesystem_lock` without a return type; `measure_api/v2/runtime.py:66` declares `worker_job(connection, ...)` without typing its process connection. AGENTS.md, Coding Style: “Type hints: Required for public function signatures.” Type both exported boundaries.

2. **Public docstrings:** 164 newly introduced public functions/methods with existing docstrings omit applicable Google-style sections, including `app.py:24`, `files.py:38`, `recipe_execution.py:484`, `portable_recipes.py:278` and `scripts/v2_http_client.py:21`. AGENTS.md, Coding Style: “Docstrings: Google style (triple-double-quoted, with Args:, Returns: sections).” Add meaningful parameter/result documentation. Every affected file, function, line and missing section is enumerated in [the original boundary audit](../.scratch/v2-rewrite/reviews/standards-audit.json); absent docstrings are not treated as violations.

3. **Line length:** AGENTS.md, Coding Style requires ≤100 characters. Violations: `families/calibrations.py:16`, `families/measurement_batches.py:144`, `tests/v2/test_alignment_service.py:107,183,185`, and `tests/v2/test_template_service.py:132,134,170,189` (family paths beneath `measure_api/v2`). Wrap these nine lines.

4. **Import groups:** `measure_api/v2/runtime.py:53–54` imports third-party `cv2` immediately before standard-library `resource`. AGENTS.md requires standard library → third-party → local, separated by blank lines. Reorder and separate this deferred group.

**Judgement call**

5. **Possible Duplicated Code:** `families/images.py:180–187`, `families/templates.py:94–98`, and `recipe_execution.py:400–408` independently implement the same rotated-ROI matrix: `cx=(width-1)/2`, followed by `[cos, sin, cx-cos*x-sin*y]`. Crop, teaching and fixed-ROI normalization must retain identical pixel-center conventions; changing that convention currently requires three edits. Extract a shared pure ROI-to-patch matrix helper and reuse it. Keep legacy conversion's deliberately different half-pixel convention separate.

Summary: four documented-standard findings; one heuristic smell. Worst Standards issue: incomplete public boundary documentation. No blocking architectural smell identified.

### First follow-up

Follow-up against `git diff 9f8f3eb...7ccf62a`; no source edits or test reruns.

**Resolved original findings**

- Public annotations: `filesystem_lock -> Iterator[None]` and `worker_job(connection: Connection)` are correct.
- All nine previously reported long lines are wrapped; the affected files contain no lines above 100 characters.
- Deferred `resource` / `cv2` imports now follow separated standard-library / third-party groups.
- Crop, template teaching and fixed-ROI execution genuinely call the same `roi_to_patch_matrix`. Its `(size-1)/2` geometry preserves the prior matrices. Legacy conversion deliberately retains its separate `size/2` convention.

**Original finding 2 remains partially open: documentation semantics**

All 164 audited boundaries now contain sections, but several newly added descriptions contradict the code:

- `measure_api/v2/app.py:24`: `create_app` incorrectly documents an exclusive-lock `Yields: None`; it returns a FastAPI application.
- `families/calibrations.py:103`: `convert_length` returns `request.target_unit`, not necessarily the calibration unit.
- `families/calibrations.py:177`: checkerboard detection returns detector-order source corners/provenance/diagnostics, no physical board points.
- `recipe_execution.py:496`: `align_target.mapping` is the localization reference-to-current-image pose; code inverts it to build the image-to-patch mapping. The new description states the opposite direction.
- `portable_recipes.py:508`: `output_reference.name` is the artifact filename, not a matrix/operation diagnostic name.
- `recipe_metrology.py:158`: `convert.operation` is a semantic operation identifier string, not a registered Operation object.
- `families/plane_stereo.py:363`: `plane_parameters.origin` records the already-established origin; this argument alone does not shift geometry or make a new zero.
- `families/plane_stereo.py:323`: `perspective` does not explicitly reject non-finite input/output, despite the new guarantee. Describe its actual horizon check without extending behavior.
- `families/plane_stereo.py:843`: board validation returns residuals/quality grouped by views or pairs; no accepted/rejected lists are returned.
- `portable_recipes.py:345`: export returns the validated original recipe and committed package descriptors; its returned recipe is not relocated and contains no compatibility diagnostics.

All unqualified paths above are beneath `measure_api/v2`. These are narrow corrections to the requested public-docstring remediation, not new implementation scope. Other sections reviewed provide meaningful parameter/result context. Four original findings are resolved; the documentation finding requires these corrections.

### Final independent Standards recheck

Final Standards follow-up: `git diff 7ccf62a...49baf8f`.

All five original Standards findings are resolved.

The ten documentation corrections now match their function bodies: the FastAPI factory documents its returned application; length conversion states the requested target unit; checkerboard detection describes source corners; alignment states the reference-to-current-image pose; output references describe artifact filenames; sample conversion accepts semantic identifier strings; plane origins are recorded metadata; projective helpers document their actual denominator checks; board validation describes residual grouping; and export describes the original recipe and committed package descriptors.

The added `roi_to_patch_matrix` and `check_metric_frame` helper docstrings accurately describe their inputs and behavior. Generator `geometry_values` correctly uses Yields; the nested lifespan no longer misclassifies the factory.

Previously resolved annotations, shared ROI implementation, import grouping and nine wrapped lines remain resolved. The revised static audit is clean across all 164 original documented boundaries.

No remaining actionable Standards finding within the requested review scope. No source edits or complete test reruns were performed.

## Spec

### Original report

Review range: `git diff 3b46f24...9f8f3eb`; commit list inspected. Four actionable Spec findings, independently reproduced through public HTTP with real temporary shared files:

- **[P1] Journal failure leaks admission capacity and breaks the error envelope.** `measure_api/v2/runtime.py:301–319` installs records/jobs and increments `pending` before `_persist`, without rollback. Creating a directory at `state_directory/journal.partial` makes a valid image inspection return plain HTTP 500; removing it still leaves HTTP 429 and an ownerless queued operation. Spec lines 522–528 require “explicit admission limits” and orderly lifecycle; lines 565–570 require “All errors contain stable code … trace ID and retryability.” Handle failed admission, terminal persistence and shutdown without leaking capacity or reversing committed success. Reproducer: `/tmp/v2_spec_probe.py`.

- **[P2] Invalid dynamic recipe parameters become unexpected server errors.** `measure_api/v2/recipe_execution.py:47` raises uncaught `ValidationError` during preparation; `runtime.py:106–109` converts it to `calculation.failed`/500. An edge-pair node with `selection: "banana"` produces this response from recipe validation, preview and run. Spec lines 480–483 require complete configuration validation; lines 568–569 specify “schema or algorithm validation is 422.” Preserve structured field diagnostics. Reproducer: `/tmp/v2_spec_probe.py`.

- **[P1] Valid template-point recipe results cannot be visualized or exported.** `recipe_metrology.py:185–192` replaces `match` with `localization_diagnostics`, absent from `families/templates.py:504–513`; `result_support.py:83` therefore rejects the service's own successful `RunResult` with 422 `result.invalid`. The result schema also restricts this mapped result to `px`, preventing physical-result round trips. Spec lines 549–555 require supplied-result visualization/export/replay; lines 404–408 require separate physical results. Reproducer: `/tmp/v2_spec_template_probe.py`.

- **[P1] Recipe metrology applies calibration from an incompatible frame.** `recipe_graph.py:252–253` checks asset against caller-supplied applicability but never against `recipe.source_frame`; `recipe_metrology.py:45–60` then labels that calibration as mapping from the recipe frame. A `source` recipe with a scalar calibration explicitly applicable to `other-cropped-frame` validates and returns valid 2 mm from a 20 px stripe, without conversion. Spec lines 396–410 require compatible frames and applicable calibration. Reject this mismatch, accounting explicitly for corrected-source and stitched-canvas provenance. Reproducer: `/tmp/v2_spec_frame_probe.py`.

No independent scope-creep finding. Documented production metrology and Docker qualification remain external evidence gates, not omitted API implementations.

### First follow-up

Reviewed `git diff 9f8f3eb...7ccf62a` and `.scratch/v2-rewrite/review-fixes.md`. Three original findings are resolved; the template-point visualization finding remains partially resolved.

1. **Resolved — journal lifecycle.** Admission now returns retryable structured 503 and rolls back records, jobs, pending capacity and owned directories. Independent public HTTP replay confirms recovery immediately accepts and completes subsequent work. Admitted transitions retain observable in-memory state; unsuccessful terminal checkpoints retain durable completion data for restart. Shutdown releases ownership. Added deterministic HTTP regressions exercise natural completion, cancellation and shutdown/restart after manifest publication, preserving committed success.

2. **Resolved — dynamic parameter validation.** Independent public HTTP probes against validation, preview and run now return 422 `recipe.invalid_parameter`, with `recipe.nodes.0.parameters.selection`, literal diagnostics and trace identity. Stage parameters use the same located conversion.

3. **Partially resolved — template-point result interoperability.** Strongly typed localization metadata and nonpixel units resolve supplied-result validation and JSON export. However, `measure_api/v2/result_rendering.py:50–52` treats the outer `point` observation as raw coordinates. Public visualization returns `type: "point"` with geometry `{valid, frame, unit, point: {x,y}}`, and adding `raster: true` returns HTTP 500 `calculation.failed` at raster coordinate access (`result_rendering.py:180–181`). This still violates Spec lines 549–551 (“Return vector overlays … optionally generate raster files”). Reproducer: `/tmp/v2_spec_template_raster_probe.py`, run using the fix worktree's `PYTHONPATH`. Recursively unwrap the typed point while preserving frame, units and validity; verify actual xy coordinates, raster output and physical-to-source display mapping. Edge-point results use the same nested shape.

4. **Resolved — metric frame applicability.** Independent public probes now reject unrelated calibration frames before measurement. Validation and the metrology mapper share the check. Plane/stereo metrics on a stitched canvas require its canonical calibration digest and canvas frame; scalar canvas metrics require matching frame/dimensions. Documentation correctly requires replacing both stitch and metric references after origin adjustment. Ordinary correction recovers original source samples; separately supplied corrected images use explicitly matching calibration frames.

No additional scope-creep finding. Production metrology and Docker qualification limitations remain accurately disclosed. Owner-reported complete suites: 167 passing on each Python version; no duplicate full-suite run performed here.

### Final independent Spec recheck

Final Spec recheck: `git diff 7ccf62a...49baf8f`, compared with the remaining item in `/tmp/v2-spec-followup.md`.

**All four original Spec findings are resolved.**

The final renderer change recursively unwraps positioned point observations while retaining inner validity, frame and units. It yields actual `{x, y}` geometry rather than an observation wrapper. The original public teaching/run/raster reproducer now returns HTTP 200 with source coordinates and committed raster/manifest descriptors; supplied-result JSON export also succeeds.

Independent focused HTTP verification: **3 passed in 3.78 seconds**. These cases verify template-point pixel-to-source and physical-to-source xy and actual green marker pixels, physical-to-physical vectors in millimeters, raster dimensions, first/all edge-point coordinates, and invalid inner observations becoming annotations without changing the raster. Existing strong result schemas remain intact; line/circle paths are unchanged.

The narrow patch does not modify journal lifecycle, located dynamic-parameter validation, metric-frame validation, stitch calibration identity or origin-reference rules, so the other three previously verified corrections remain resolved. No new actionable Spec error was found in this patch. No full-suite rerun was performed; the implementer reports 169 passing on both Python 3.11 and Python 3.9.

Production metrology qualification and actual Docker runtime verification remain external evidence gates as documented.

## Final resolution

Both fix commits (`7ccf62a`, `49baf8f`) are integrated. Independent final rechecks
closed every original finding. Complete final suites passed 169 cases on CPython 3.11
(127.37s) and 3.9 (165.77s); the integrating checkout additionally passed five focused
HTTP cases (5.29s). The final real-sample workload independently passed 72/72 calculations
with unchanged numeric observations. See [acceptance evidence](v2-acceptance.md). Reproducible HTTP regressions are retained in
`tests/v2/test_operation_lifecycle.py`, `test_recipes.py` and `test_results.py`.

Standards: five findings (four documented-standard classes and one heuristic);
worst issue: incomplete public boundary documentation. Spec: four findings;
worst issue: reporting valid physical dimensions from an incompatible calibration frame.

Final status: Standards 5/5 resolved (worst original issue: public boundary documentation); Spec 4/4 resolved (worst original issue: incompatible calibration frame producing valid physical dimensions). No actionable finding remains on either axis.

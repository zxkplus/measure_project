# Final dual-axis review fixes

Reviewed baseline: `3b46f24`; reviewed implementation head: `9f8f3eb`.
Initial fix commit: `7ccf62a`. Independent follow-up confirmed three Spec findings and
four Standards findings; the template-point raster path and ten documentation semantics
needed further corrections. The follow-up below records those corrections and verification.
Independent integrating reviewers have closed all findings on both review axes.
The original ten implementation tickets remain resolved.

## Initial Spec changes

1. Operational-journal admission failures return retryable structured 503
   `service.journal_unavailable`, roll back records/jobs/capacity and remove owned staging.
   Preparation checkpoint failure permits retrying the same idempotency key after recovery.
   Admitted state changes survive temporary checkpoint failure in memory. An unjournaled
   committed completion stays on disk for restart recovery; natural completion, cancellation
   and active-worker shutdown preserve committed success. Shutdown always releases the owner.
2. Invalid dynamic node/stage parameter types return 422 `recipe.invalid_parameter` with
   located parameter diagnostics, consistently through validation, preview and run.
3. Template-point recipe observations retain a strongly typed sampling-frame localization
   record separately from pixel/physical points. Physical units are accepted by the point
   result schema. A successful teaching/run result round-trips through visualization and
   export without allowing arbitrary extra fields.
4. Metric applicability binds to the recovered measurement source frame, including separately
   supplied corrected-source images. A stitched plane metric binds to the actual stitch
   calibration digest and canvas frame. Scalar canvas calibration binds to actual canvas
   dimensions/frame. Incompatible metrics fail before measurement. The documented origin
   adjustment workflow replaces both stitch and metric asset references together.

## Initial Standards changes

- Added the process-pipe argument annotation and filesystem-lock iterator return annotation.
- Documented all 164 audited existing public boundaries with applicable Google-style
  Args/Returns/Yields and parameter/result descriptions; new public helpers are also documented.
  Follow-up identified ten semantic inaccuracies, corrected individually below.
- Wrapped the nine reported long lines and all new changed Python lines to at most 100 columns.
- Separated deferred standard-library/third-party imports.
- Reused one pure pixel-center ROI-to-patch transform in image cropping, template teaching
  and fixed-ROI target normalization. Legacy conversion retains its own half-pixel convention.

## Initial verification

Behavior regressions were reproduced through public FastAPI HTTP before fixes. Fault tests
use real filesystem obstructions, controlled OS publication and the external filesystem
replace boundary; no production testing switch or calculation collaborator mock was added.

- Initial affected HTTP acceptance run: 84 passed in 57.73 seconds.
- Initial-patch Python 3.11 complete V2 HTTP/deployment/contract suite: 167 passed in 125.20 seconds.
- Initial-patch Python 3.9 complete V2 HTTP/deployment/contract suite: 167 passed in 163.10 seconds.
- Both full suites emit one expected warning while constructing a duplicate ZIP member.
- Static audit: all 164 reviewed docstrings complete; no missing public annotations; no
  Python lines over 100 characters across V2 production, V2 tests and V2 scripts.
- `git diff --check` passes, excluding the pre-existing case-colliding root README artifact.

No numeric-core or V1 source changed; their previously completed regressions are retained.
Physical accuracy/capture repeatability qualification still requires upstream truth and
acceptance thresholds. Actual Docker execution remains unavailable without the daemon.
The final sample qualification is run by the integrating agent after this source patch merges.

## Follow-up corrections

The follow-up starts from integrated source `944719f`. A public teaching/run/vector regression
reproduced the remaining nested-point geometry: a point primitive contained its outer framed
observation rather than xy coordinates. The shared geometry iterator now recursively unwraps
point observations, carrying their validity, frame and unit into the actual xy primitive.
No result schema was relaxed, and line/circle geometry paths are unchanged.

The enhanced real HTTP scenario checks pixel-to-source and physical-to-physical vector xy,
physical-to-source inverse drawing, committed raster dimensions and actual green marker pixels
at the independently known anchor. Separate first/all edge-point cases verify actual edge
coordinates, array overlays, raster pixels, and invalid inner observations producing annotations
without drawing. Initial focused template/edge/line/projective-circle acceptance: 5 passed.

The ten public documentation inaccuracies were corrected by reading each function:
FastAPI factory Returns, length target unit, checkerboard source-corner output, localization
pose direction, output artifact filename, operation identifier string, recorded plane origin,
actual homogeneous-horizon check, board residual grouping, and export's original recipe/package
result. No numerical behavior was changed to match a description. The geometry iterator's
section is now Yields; the factory's nested lifespan no longer affects its Returns documentation.

Follow-up verification:

- Python 3.11 complete V2 HTTP/deployment/contract suite: 169 passed in 127.37 seconds.
- Python 3.9 complete V2 HTTP/deployment/contract suite: 169 passed in 165.77 seconds.
- Each suite emits only the expected warning from constructing a duplicate ZIP member.
- Revised static audit checks generator sections at the function's own scope, excluding
  nested closures. All 164 original boundaries have applicable documented parameters and
  sections; public annotations and 100-column checks are clean across V2 production/tests/scripts.
- All ten listed semantic corrections were manually checked against their function bodies.
- `git diff --check` passes, excluding the case-colliding root README artifact.

Independent final rechecks closed all original review findings. Final-source workload evidence
is recorded by the integrating parent below; production qualification remains pending.

## Integrating checkout verification

Follow-up 49baf8f was merged after the initial source fix and integration evidence. All five
selected public HTTP cases passed (5.29s), asserting actual template-point pixel/physical xy,
physical-to-source raster pixels, first/all/invalid edge observations and retained line/projective
circle rendering. No full suite was repeated after both owner runtimes passed 169 cases.
Independent final rechecks closed all five Standards and four Spec findings. The Spec reviewer
also passed three focused public HTTP tests (3.78s). The integrating parent verified both
36-calculation reports from the final-source real-image workload: 72/72 succeeded, all localized
cases found one target, and every numerical observation matches the earlier snapshot. Runtime
identity and four input digests were independently checked. Source SHA-256 is
`a1847e48b96dcdf9827ad465560d4f11b3fce459778f4f109580dfe08bd8c6ef`.
Reports are retained in ignored output/v2-qualification-post-review/. No physical truth or
production thresholds were supplied, so both reports retain qualification.status: pending.
See docs/v2-code-review.md and docs/v2-acceptance.md for the final delivery evidence.

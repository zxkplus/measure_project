# V2 rewrite: coarse ticket proposal

**Status:** needs-triage

This is a review draft, not the published issue queue. The to-tickets workflow
requires approval of granularity and blocking edges before promotion to issues.
The configured tracker is local Markdown. Each ticket will be promoted as one
ready-for-agent issue after approval; the source specification is unchanged.

## Scope and slicing

Ten coarse tickets cover the agreed FastAPI computation service. Each includes
validated public requests, real calculations, observable outputs, errors and
HTTP acceptance tests. Upstream GUI implementation is outside this effort;
examples and protocol behavior make each slice usable by its upstream caller.
Refactor only the reusable calculations needed by a slice before extending them,
with focused behavioral regressions preceding core algorithm corrections.

The first slice is a working image service rather than an infrastructure-only
milestone. The operational policy slice is demonstrated on those real image
operations. All later operations use the same execution interface; they need
not wait for operational hardening to implement their own public behavior.
Intermediate replacement releases are for development, and the final cutover
waits for all capability and lifecycle acceptance.

## Proposed tickets

1. **Process shared images through the new FastAPI service** — blocked by none. [Draft](drafts/01-shared-file-image-service.md)

   An upstream caller can start the new headless service, inspect a committed shared image, apply
   preprocessing or a rotated crop, and receive typed metadata, coordinate mappings and complete
   output files through HTTP. This first usable slice establishes the file and execution contracts
   using real image operations.

2. **Make computations bounded, cancellable and retryable** — blocked by 01. [Draft](drafts/02-operation-lifecycle.md)

   An upstream caller can submit a long-running image computation, stop waiting without losing it,
   query or cancel it, retry safely within retention, and distinguish interruption, timeout and
   completed output. The same execution policy applies to every capability added later.

3. **Measure edges and geometric dimensions through atomic APIs** — blocked by 01. [Draft](drafts/03-atomic-dimensional-measurements.md)

   An upstream measurement editor can measure a supplied image or point set, compose returned
   geometry into dimensions, and evaluate supplied limits without creating a recipe or registering
   assets. Results include valid geometry, selected samples and useful failure outcomes.

4. **Build templates and localize aligned targets** — blocked by 01. [Draft](drafts/04-template-localization-alignment.md)

   An upstream teaching workflow can construct a portable template, match one or several target
   instances, refine each pose using local controls and return mapped geometry or template feature
   points without GUI callbacks or engine registration.

5. **Create scale and camera calibrations and correct images** — blocked by 01. [Draft](drafts/05-scale-camera-calibration.md)

   An upstream calibration editor can detect or manually amend calibration observations, compute and
   inspect scale or camera assets, and correct a shared image while retaining ownership of every
   input and generated calibration.

6. **Calibrate shared planes and stereo rigs and stitch views** — blocked by 05. [Draft](drafts/06-plane-stereo-stitching.md)

   An upstream calibration workflow can establish a board-plane canvas and stereo relationship,
   assess their errors, adjust a physical origin into a new asset and stitch paired views for later
   measurement.

7. **Validate, preview and execute complete supplied recipes** — blocked by 03, 04, 06. [Draft](drafts/07-recipe-preview-execution.md)

   An upstream recipe editor can submit an unsaved configuration for validation or selected-tool
   preview, then execute the same supplied version on one or many images and receive per-target
   dimensions and evaluations. The engine invokes atomic calculations directly and does not own
   recipe storage.

8. **Inspect and export measurement results** — blocked by 02, 07. [Draft](drafts/08-results-diagnostics-export.md)

   An upstream result viewer can display structured observations as selectable overlays, request
   raster and diagnostic files, and export CSV/JSON reports from supplied results even when it does
   not retain an engine business-result database.

9. **Convert legacy recipes and replay archived runs** — blocked by 08. [Draft](drafts/09-recipe-conversion-replay.md)

   An upstream asset owner can package/import a supplied recipe, convert supported GUI projects with
   explicit compatibility diagnostics, and reproduce an archived run using frozen configuration and
   unchanged files after the original operation expires.

10. **Replace old V2 and verify the full upstream workflow** — blocked by 09. [Draft](drafts/10-replace-v2-acceptance.md)

   A deployment operator can start the complete replacement at the V2 HTTP prefix and an upstream
   integrator can perform the entire teaching, calibration, preview, production run and
   investigation workflow using only the new contract.

## Why these blocking edges exist

- 01 establishes the actual HTTP, file, frame and execution contract used by
  02, 03, 04 and 05; those four slices can then progress independently.
- 06 requires 05's camera observations, calibration formats and corrected-image
  mappings for paired views and shared-plane calculations.
- 07 requires 03's measurements/evaluations, 04's localization and 06's complete
  correction/calibration variants to deliver the full supplied recipe path.
  05 is already a transitive dependency through 06; no duplicate edge is needed.
- 08 requires 07's complete per-target observations and 02's retained operation,
  diagnostic and expiration behavior. Basic quality values remain part of the
  atomic operation tickets, rather than being postponed to this slice.
- 09 requires 08's complete output descriptors/manifests for archived replay and
  transitively has the recipe, template and calibration formats it converts.
- 10 verifies and switches the complete replacement. 09 transitively gates all
  previous tickets, so a second list of every ancestor would add no dependency.

## Acceptance ownership

Every story from 1 through 94 is owned by at least one draft ticket. Overlap is
intentional where a shared contract needs both first delivery and integrated
verification. All routes belong to a slice: discovery/images to 01; operations
to 02 with rich diagnostics in 08; probes/fits/geometry/evaluations to 03;
templates/alignment/template points to 04; scalar/camera features, correction
and length conversion to 05; plane/stereo/stitching to 06; validation/preview/
runs/batches to 07; visualization/result export to 08; recipe formats/legacy
conversion/replay to 09; complete contract inventory and production cutover to
10. New template/calibration formats are delivered by their own calculation
slices, not deferred until recipe conversion.

## Review and publication

Confirm that this is the desired coarse granularity, that blockers genuinely
gate their dependants, and whether any tickets should be merged or split.
After approval, promote each draft separately in dependency order with
ready-for-agent status. Do not modify or close a parent issue. No source code,
runtime dependency, old V2 implementation or upstream asset is changed by this
proposal.


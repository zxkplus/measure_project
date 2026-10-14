# Rewrite V2 as a FastAPI measurement computation service

## Problem Statement

The upstream application needs to reproduce and improve the current desktop
measurement workflow using a supported integration contract. Existing V2
exposes only a subset of the project's capabilities, imposes an engine-owned
model/session lifecycle, and requires a shared-memory image protocol that does
not match the agreed deployment and ownership model. Additional usable
capabilities remain embedded in GUI code or available only through Python and
scripts. Coordinate inconsistencies and incomplete failure propagation can
also make a successful request produce an incorrect or misleading dimension.

Upstream owns recipes, templates, calibration assets, their versions and
business results. Both services run on the same node and exchange large files
through a shared directory. The user requires FastAPI for the HTTP server.
Upstream must be able to teach, inspect, calibrate and execute measurements
through HTTP parameters and file references without adopting V2 asset storage,
depending on GUI internals, or reimplementing measurement orchestration.

## Solution

Replace the existing V2 implementation and contract with a headless FastAPI
computation service. Expose independently callable operations for image
processing, template construction/matching, alignment, edge detection,
geometric fitting/composition, calibration, correction, stitching, evaluation,
visualization, file-format conversion, export and replay. Expose complete
recipe validation, preview and multi-target execution using those same
calculations for efficient integration.

HTTP transports compact configuration, typed observations, errors and file
references. The shared directory transports input pixels, portable templates,
calibrations, recipes and large output artifacts. Upstream remains the
authoritative owner of all business assets. V2 retains only bounded caches and
operational task records, writes requested output files, and returns enough
metadata for upstream to register and reproduce the result.

Use consistent coordinates, units, named dependency ports and failure
semantics. FastAPI and Pydantic provide the actual validation and generated
OpenAPI contract. Both synchronous and asynchronous requests use bounded
compute workers with explicit admission, timeout, cancellation and file-commit
behavior. Completion includes all agreed capability families, not only the
primary edge/line/circle objects.

## User Stories


1. As an upstream integrator, I want FastAPI HTTP endpoints, so that I can integrate a documented
    ASGI service with the upstream application.

2. As an upstream integrator, I want generated OpenAPI, so that I can generate clients and validate
    requests against the running service.

3. As an upstream UI developer, I want discoverable operation schemas and defaults, so that I can
    construct parameter panels consistently.

4. As an upstream asset owner, I want to maintain recipes and versions myself, so that V2 does not
    introduce a competing source of truth.

5. As an upstream asset owner, I want to maintain templates and calibrations myself, so that their
    registration and lifecycle remain part of my application.

6. As an upstream integrator, I want to call individual calculations without registering assets or
    opening a session, so that I can compose operations freely.

7. As an upstream integrator, I want full recipe execution as one operation, so that complete
    measurements do not require many HTTP round trips.

8. As an upstream integrator, I want to submit an inline recipe or a shared recipe file, so that I
    can choose the appropriate control payload size.

9. As an upstream operator, I want V2 to execute the exact supplied version, so that later asset
    edits cannot silently change my run.

10. As an upstream integrator, I want HTTP to carry parameters and file references, so that large
    data can remain on the shared filesystem.

11. As a deployment operator, I want shared-root aliases, so that different container mount paths do
    not change client requests.

12. As an upstream integrator, I want supported absolute references inside configured roots, so that
    deployments with identical mounts can use existing file paths.

13. As an upstream image producer, I want to commit input files before submission, so that
    calculations never read incomplete images.

14. As an upstream asset owner, I want explicit input retention responsibilities, so that
    asynchronous operations and replay keep valid inputs.

15. As an upstream integrator, I want missing, unreadable or changed input files reported
    distinctly, so that I can diagnose exchange failures.

16. As an upstream asset owner, I want optional expected content hashes checked, so that an
    operation cannot silently use a different asset version.

17. As an upstream integrator, I want image metadata inspection, so that I can confirm actual
    dimensions, channels and pixel representation.

18. As an upstream UI developer, I want image previews written to a shared output directory, so that
    I can display them using upstream file access.

19. As a measurement operator, I want raw, Canny, Sobel, CLAHE and threshold preprocessing, so that
    I can tune feature visibility.

20. As a measurement operator, I want rotated cropping and straightening with coordinate mappings,
    so that I can teach tools in a normalized target patch.

21. As an upstream asset owner, I want template construction to produce portable files, so that I
    can register and reuse them independently of V2 process lifetime.

22. As a measurement operator, I want template support and anchor metadata, so that I know exactly
    which image content defines localization.

23. As a measurement operator, I want ROI support or square-context template support, so that I can
    choose tighter localization or current GUI behavior deliberately.

24. As a measurement operator, I want template inspection, so that invalid or incompatible template
    files can be found before production execution.

25. As a measurement operator, I want single-target and multi-target template matching, so that I
    can inspect images containing one or several instances.

26. As a measurement operator, I want rotation and scale search controls, so that localization can
    tolerate pose changes without changing taught tools.

27. As a measurement operator, I want search regions, score limits, match-count limits and overlap
    suppression, so that I can control candidate selection.

28. As a measurement operator, I want coarse/fine and pyramid search controls, so that I can balance
    localization accuracy and runtime.

29. As a measurement operator, I want similarity estimation from correspondences, so that I can
    relocate taught geometry using known anchors.

30. As a measurement operator, I want rigid target alignment, so that translated or rotated targets
    can use one taught recipe.

31. As a measurement operator, I want per-target control-point refinement, so that local features
    can improve alignment independently of GUI callbacks.

32. As an upstream integrator, I want explicit refinement failure and fallback policies, so that
    alignment cannot silently degrade beyond my intended tolerance.

33. As an upstream integrator, I want typed geometry mapping between frames, so that overlays and
    measurements remain associated with the correct image.

34. As a measurement operator, I want gray, smoothed and gradient profiles, so that I can understand
    why a probe chooses or misses an edge.

35. As a measurement operator, I want selectable subpixel edge points and amplitudes, so that I can
    measure a specified boundary accurately.

36. As a measurement operator, I want explicit edge polarity and selection controls, so that a tool
    measures the intended boundary.

37. As a measurement operator, I want edge pairs with both endpoints, center and width, so that I
    can measure local thickness without hidden endpoint naming rules.

38. As an upstream integrator, I want independent measurements sharing one image in a batch, so that
    I can preview several tools without creating a recipe.

39. As a measurement operator, I want image-based line probing and fitting, so that I can define
    geometric reference edges.

40. As a measurement operator, I want image-based circle and arc probing with radius limits, so that
    I can measure circular boundaries while excluding unrelated edges.

41. As an upstream integrator, I want line and circle fitting from supplied points, so that existing
    point observations can use the same fitting algorithms.

42. As a measurement operator, I want local template-matched feature points, so that marked features
    can participate in dimensional measurements.

43. As a measurement operator, I want a line through two points, so that detected features can
    define a reference line.

44. As a measurement operator, I want two-point distance, so that I can measure width, height or
    feature spacing.

45. As a measurement operator, I want point-to-line distance, so that I can measure perpendicular
    offsets from a reference edge.

46. As a measurement operator, I want the smaller angle between two lines, so that I can measure
    relative inclination consistently.

47. As a measurement operator, I want point-to-circle distance, so that I can measure radial gaps.

48. As a measurement operator, I want radius, diameter and sampled radial spread, so that circular
    measurement outputs are explicit and correctly named.

49. As an upstream integrator, I want geometry frame compatibility checked, so that unrelated
    coordinates cannot produce a plausible but incorrect dimension.

50. As a measurement operator, I want dimensions preserved after scale-normalized localization, so
    that genuine size variation is not normalized away.

51. As a measurement operator, I want pixel and physical dimensions returned separately, so that
    calibration cannot obscure the original observation.

52. As a measurement operator, I want checkerboard corner detection with ROI and lattice
    diagnostics, so that I can establish a usable scale observation.

53. As a measurement operator, I want scale computation from manually edited corners, so that I can
    reproduce the current calibration interaction upstream.

54. As an upstream asset owner, I want generated calibrations returned as values or shared files, so
    that I can control their versions and applicability.

55. As a measurement operator, I want dot-grid detection per image, so that I can review or exclude
    observations before camera calibration.

56. As a measurement operator, I want intrinsic and distortion calibration from images or accepted
    observations, so that camera correction can be integrated.

57. As a measurement operator, I want undistortion with mapping metadata, so that corrected image
    measurements can be interpreted in their declared frame.

58. As a measurement operator, I want calibrated board-plane mappings, so that multiple camera views
    can share a physical plane and canvas.

59. As a measurement operator, I want stereo relative-pose calibration from paired observations, so
    that camera-frame transformations are available.

60. As a measurement operator, I want per-point and aggregate calibration validation errors, so that
    I can assess a calibration before using it.

61. As a measurement operator, I want a revised physical origin as a new output, so that I can
    adjust coordinates without mutating an existing calibration file.

62. As a measurement operator, I want calibrated two-image stitching, so that measurements can use
    an appropriate shared plane image.

63. As an upstream integrator, I want 3D camera-frame transformations and camera projections, so
    that existing stereo library calculations are externally callable.

64. As an upstream asset owner, I want calibration inspection and format conversion, so that I can
    transport and validate calibration assets.

65. As an upstream recipe editor, I want stable node IDs and named typed ports, so that labels and
    display order do not control dependencies.

66. As an upstream recipe editor, I want complete configuration and DAG validation, so that cycles,
    missing inputs and incompatible ports are found before execution.

67. As an upstream recipe editor, I want explicit node enabled state, so that hiding a graphic does
    not accidentally disable measurement.

68. As an upstream UI developer, I want selected-node preview with dependency closure, so that I can
    tune one tool without manually preparing its inputs.

69. As an upstream UI developer, I want unsaved preview configuration and echoed preview tokens, so
    that dragging and fast edits cannot display stale responses.

70. As an upstream operator, I want per-target measurement results, so that I can inspect each
    matched instance independently.

71. As an upstream operator, I want unrelated targets to continue after a target failure, so that
    partial results remain useful.

72. As an upstream integrator, I want invalid dependency propagation and required-output validity,
    so that a target is not called valid when a required dimension failed.

73. As an upstream integrator, I want operation completion, measurement validity and acceptance
    status separated, so that success is not mistaken for product conformity.

74. As an upstream integrator, I want explicit no-target outcomes, so that I can apply an expected-
    target policy rather than infer success from an empty list.

75. As an upstream operator, I want batches with per-image outcomes, so that one failed image does
    not hide the remaining measurements.

76. As an upstream integrator, I want range, diameter-variation and centerline-deviation
    calculations, so that current script metrics can be reused with supplied policies.

77. As an upstream UI developer, I want structured vector overlays, so that I can control selection
    and visibility without regenerating measurement pixels.

78. As an upstream operator, I want optional raster overlays and diagnostic files, so that difficult
    measurements can be inspected visually.

79. As an upstream integrator, I want CSV and JSON export from supplied results, so that upstream
    can deliver reports without an engine run database.

80. As an upstream asset owner, I want recipe packaging and conversion of supported GUI projects, so
    that I can register converted assets in my own storage.

81. As an upstream operator, I want replay from retained configuration and files, so that
    investigation does not depend on an expired V2 job record.

82. As an upstream integrator, I want synchronous and asynchronous execution to produce equivalent
    observations, so that integration mode does not change measurement behavior.

83. As an upstream integrator, I want a queryable operation after synchronous wait expiry, so that
    long calculations do not lose their result.

84. As an upstream operator, I want deadlines and cancellation, so that slow jobs can be stopped
    with a clear terminal outcome.

85. As a deployment operator, I want bounded workers and cache memory, so that HTTP responsiveness
    and node resources remain predictable.

86. As an upstream integrator, I want idempotent retry within stated retention, so that transient
    HTTP failures do not duplicate accepted work unexpectedly.

87. As an upstream asset owner, I want atomic output commits and complete-file manifests, so that
    registered assets never contain partial results.

88. As an upstream asset owner, I want V2 to preserve input and completed-output ownership, so that
    engine cleanup cannot remove business assets.

89. As an upstream operator, I want interrupted jobs reported explicitly after restart, so that I
    can decide whether to retry instead of receiving a fabricated success.

90. As an upstream integrator, I want cache identity based on verified content and parameters, so
    that reusing a path cannot return stale measurements.

91. As a deployment operator, I want liveness, readiness and orderly lifespan handling, so that
    startup and shutdown reflect worker and shared-directory availability.

92. As an upstream integrator, I want stable structured errors and trace IDs, so that I can display
    actionable failures and correlate diagnostics.

93. As a project maintainer, I want numerical regressions for known integration defects, so that the
    rewritten service does not reproduce coordinate or state errors.

94. As a project maintainer, I want all existing meaningful capabilities exposed without GUI runtime
    dependencies, so that the new V2 is a complete headless replacement.

## Implementation Decisions

1. **Replacement and ownership.** Replace the old V2 routing, implementation,
   public schemas, tests that enforce obsolete behavior, and startup contract.
   Retain the V2 HTTP prefix, not compatibility with old modeling sessions,
   engine-owned publication or shared-memory slot descriptors. Upstream owns
   recipe/template/calibration persistence and versions. No business asset
   registry, mandatory session, permanent result store or asset CRUD is added.

2. **HTTP implementation.** Use FastAPI, Pydantic v2 and Uvicorn. Provide an
   application factory, capability-family routers, dependency injection and
   lifespan-managed file settings, workers and caches. Lock dependencies that
   work with the declared Python runtime. Start with one ASGI worker and a
   bounded computation process pool. Increasing HTTP process count requires
   explicit operational-state routing and a shared resource budget.

3. **Schemas and discovery.** Typed request and response models validate
   finite numbers, limits, units, frames, unknown fields and discriminated
   operation variants. Stable operation identifiers and actual typed routes
   generate OpenAPI. Capability discovery uses the same operation/model
   registry, including available modes, parameter defaults and output ports.
   Interactive documentation exposes the generated contract.

4. **Calculation modules.** Extract GUI-independent geometry, alignment,
   atomic execution and recipe validation/execution modules. File exchange,
   portable formats, job admission, caches and diagnostics belong to service
   orchestration. Mathematical calculations receive typed data and return
   typed observations without HTTP requests or GUI side effects. The recipe
   runner invokes atomic functions directly. V2 must not import or initialize
   the GUI runtime or depend on private GUI factories.

5. **File references.** Accept root-alias/relative-path references and a
   mutually exclusive absolute-path form restricted to configured shared
   roots. Allow optional expected SHA-256 digests and upstream provenance.
   Normalize and check canonical paths and symlink destinations. Different
   container mount prefixes resolve through aliases. There is no required
   upload or image-ID registration step. Required referenced files must be
   checked before accepting asynchronous work.

6. **File ownership and commit.** Upstream commits inputs before submission
   using atomic rename and keeps them immutable through terminal operation
   state. Multi-file input manifests become visible after their payloads.
   V2 writes requested outputs into isolated attempt namespaces, stages on the
   destination filesystem and commits complete files atomically. A final
   manifest is the completion marker for a multi-file result. It never
   overwrites inputs or existing completed assets. Upstream registers and
   retains completed outputs. V2 cleans only owned staging/cache/job data.

7. **Input snapshots.** Snapshot control data and either obtain private
   decoded/copy buffers or pin and verify immutable referenced inputs before
   execution. If input availability or identity is violated, fail explicitly.
   Queued work must not silently follow a rewritten file or a newer recipe.
   Configuration and large-file input preparation count toward the deadline.

8. **Pixel formats.** Initially support existing GRAY8 and RGB/BGR8 processing
   through PNG/JPEG. Decode actual dimensions and channels rather than trusting
   supplied metadata. Lossless fixtures are preferred for numerical acceptance.
   Unsupported depth conversion is explicit. Float preprocessing results
   remain typed numeric arrays; an 8-bit display preview is a separate output.

9. **Coordinate contract.** Public points use named x/y values: x right, y
   down, with the origin at the top-left pixel center. Probe/pose angles use
   degrees: zero right, ninety down, visually clockwise positive. Adapters
   translate internal row/column, radians and OpenCV conventions. Probe extents
   distinguish search half-length and average half-width; box dimensions are
   full width/height. Normalize lines as a*x + b*y + c = 0.

10. **Frames and dimensions.** All observations declare frame and unit.
    Transform descriptions declare source, destination, type and direction.
    Undistortion has nonlinear mapping metadata. Geometry inputs require
    compatible frames or an explicit supported conversion. The default metric
    frame is source-image pixels; physical measurements require an applicable
    supplied calibration. Map edge samples into the metric frame before
    fitting and calculating distances to preserve actual size variation.

11. **Circle mapping and physical units.** Similarity transforms can convert
    radius analytically. General affine/projective mappings do not silently
    preserve a circle or radius: use mapped samples for refitting and sampled
    curves for rendering, or reject unsupported analytical conversion. Return
    raw pixel and physical results separately with calibration version/digest.
    Intrinsics alone do not establish millimeter scale. Plane and scalar
    mappings declare their camera/image/setup applicability.

12. **Image operations.** Provide image inspection, preview, preprocessing and
    axis-aligned/rotated crop calculations. Results expose metadata, transform
    provenance and shared output references. Preprocessing includes raw,
    Canny, Sobel, CLAHE and threshold options; template construction and
    matching apply the same preprocessing definition exactly once.

13. **Template operations.** Build portable templates from supplied image,
    support ROI, anchor and preprocessing; inspect and match supplied template
    files without registration. Expose actual support and ROI versus square
    context as explicit modes. Matching includes single/multiple instances,
    search region, score threshold, rotation/scale ranges and steps, maximum
    matches, overlap suppression, coarse/fine search and pyramid parameters.
    Tight rectangular support requires a tested core extension where current
    square-template assumptions do not suffice.

14. **Alignment operations.** Provide correspondence-based similarity
    estimation, rigid alignment and control-point refinement. The initial
    refinement contract describes the implemented similarity transform,
    minimum matched controls, RANSAC/residual settings and explicit rigid
    fallback versus error. Do not advertise unrestricted affine precision.
    Control templates must be constructed independently of broken GUI setup.
    Multi-target controls use independently cropped search regions and have
    their own target-isolation acceptance tests.

15. **Measurement operations.** Provide edge profiles, selectable subpixel
    edge points, edge pairs, image-based line/circle/arc fitting, local template
    feature points, supplied-point line/circle fitting, and independent batch
    measurements sharing an image. Expose probe geometry, counts, smoothing,
    threshold units, polarity, interpolation, selection, expected geometry,
    radius limits and arc range with discoverable defaults.

16. **Edge-pair outputs.** Return explicit boundary points, center and width.
    Atomic results can contain multiple pairs. Single-pair recipe nodes expose
    named edge_a, edge_b, center and width ports; array ports are typed
    separately. Pair selection is explicit. First/last selection occurs after
    detection/pairing rather than truncating the input edge set prematurely.
    Missing pairs produce invalid observations, not zero dimensions.

17. **Composed geometry.** Expose two-point line, two-point distance,
    perpendicular distance to an infinite line, smaller unsigned line angle
    in zero to ninety degrees, shortest point-to-circumference distance,
    circle dimensions and explicit calibrated length conversion. Coincident
    points and degenerate lines/circles have documented invalid outcomes.
    Sampled radial spread is identified by its actual computation, not an
    ellipse fit or standardized roundness measurement.

18. **Calibration operations.** Split checkerboard feature detection from
    scale computation using supplied/manual corners. Expose ROI, lattice and
    corner refinement diagnostics. Support dot-grid detection per image,
    intrinsic/distortion calibration from images or accepted observations,
    and camera correction. Point ordering, physical grid coordinates,
    observation rejection, dimensions and camera association are explicit.

19. **Stereo and plane operations.** Expose board-plane mappings and canvas,
    stereo relative pose, per-point/aggregate board reprojection validation,
    physical-origin adjustment, two-image stitching, 3D camera-frame mapping
    and camera projection. Input image pairs have explicit paired capture
    identity. Origin adjustment generates a new output rather than modifying
    supplied calibration files. All calibration inspection/import/export
    operations return values or files for upstream registration.

20. **Supplied recipe model.** Accept exactly one of inline recipe or a shared
    recipe reference. Asset references are shared files or compact typed
    values. Echo upstream recipe ID/version and calculate the resolved
    configuration hash; never infer a latest version. Recipe data includes
    correction/stitching, parent matching or fixed ROI, alignment, metric
    frame/calibration, target selection, required outputs and nodes.

21. **Dependencies and preview.** Stable node IDs and named typed ports define
    dependencies. Labels/display order are independent. Validate all missing
    references, cycles, frame/type mismatches, disabled required inputs and
    incompatible template provenance. Preview accepts a supplied unsaved
    snapshot, selected nodes with dependency closure, upstream revision and
    preview token. It echoes these values and never saves the recipe.

22. **Complete execution.** Validate/snapshot configuration and files; perform
    declared correction, matching/fixed ROI and per-target alignment; extract
    samples and fit in the metric frame; execute composed nodes in topological
    order; calculate physical outputs and supplied rules; commit requested
    results/diagnostics. Expose independent image batches and complete recipe
    batches with per-item outcomes. A target identifier is local to a run.

23. **Business calculations.** Expose inclusive range checks, multiple-diameter
    variation and edge-pair centerline x-deviation using supplied values and
    explicit units/invalid-input policy. Rule ownership stays upstream. These
    reproduce existing script calculations and do not claim standards-based
    certified roundness or verticality.

24. **Observation quality.** Report real selected samples, requested/successful
    probe counts, edge amplitudes, mean/max residual and calculated RMS where
    available. Missing metrics are absent/null with declared meaning, not
    invented zeros. Diagnostics can expose sampling curves and selected edges
    with frame mappings. Include per-stage timing separately from queue delay.

25. **Validity semantics.** Separate terminal execution state, measurement
    validity and evaluation status. Weak matches, insufficient samples and
    failed dependencies are typed domain outcomes. Required-node outcomes
    determine target validity. Exceptions cannot retain a prior successful
    observation. Unrelated targets continue; no matches yields no_targets,
    interpreted using optional expected-target rules. Invalid geometry is not
    returned as an apparently valid default scalar.

26. **Synchronous/asynchronous behavior.** Both modes submit to the same
    execution module. Async returns accepted operation identity/location.
    Sync waits within a separate bounded wait window; if still running, it
    returns the queryable operation. Queueing, input preparation, process
    startup and computation count toward the operation deadline. Operations
    have queued, running, cancelling, succeeded, failed, cancelled and
    timed_out states with stable terminal outcomes.

27. **Worker lifecycle.** Use bounded isolated computation processes and
    explicit admission limits for concurrency, queueing and memory. CPU work,
    large file reads and digest computation do not run on the ASGI event loop.
    FastAPI BackgroundTasks is not the measurement scheduler. Lifespan starts
    configured resources and handles orderly shutdown. Temporary job records
    may use an operational journal; interrupted jobs are reported rather than
    silently replayed after restart.

28. **Cancellation and commits.** Cancel/timeout before output commit abandons
    incomplete staging data. An already completed commit remains succeeded;
    it is not retroactively reported cancelled. Multi-file outputs are usable
    only after their manifest commits. Retried attempts have isolated output
    namespaces; preexisting completed destinations cause explicit conflicts.

29. **Operational retention and retries.** Document job/trace retention and
    cache quotas. Querying expired state returns an explicit expiration
    outcome. Idempotency keys deduplicate within that retention using caller,
    route and canonical request identity including verified input digests.
    Changed inputs with one key conflict. This does not promise perpetual
    exactly-once business execution; upstream retains terminal results.

30. **Cache correctness.** Decoded/prepared caches are disposable and bounded.
    Identify entries by verified content digest, algorithm version and
    relevant parameters, not pathname, label or modification time alone.
    Uncached/restarted execution can always compute from valid upstream files.
    Prepared matchers do not share mutable prior-run measurement state.

31. **Visualization, export and replay.** Return vector overlays as the normal
    interactive representation and optionally generate raster files. Large
    curves/arrays and CSV/JSON exports are shared-file outputs. Every completed
    file descriptor contains kind, format/schema version, path, digest and
    size. Upstream owns retained business traces and results. Replay accepts a
    supplied frozen manifest and unchanged inputs, even when old job metadata
    is gone; it does not replay mutation history. Record algorithm/runtime
    version and explain incompatible or unavailable replay inputs.

32. **Portable formats and conversion.** Recipe/template/calibration formats
    use versioned JSON with numeric/image payloads and explicit reference
    resolution. Format inspection/import/export are calculations, not
    registration. Supported GUI project conversion is one way and returns
    converted data plus unsupported-field diagnostics. New public packages
    do not depend on arbitrary pickle objects or a running GUI.

33. **Error contract.** All errors contain stable code, message, field path or
    structured detail, trace ID and retryability. Malformed JSON is 400;
    missing input or retained job is 404; expired operational state is 410;
    digest/output/idempotency conflict is 409; schema or algorithm validation
    is 422; capacity and unavailable readiness use 429/503 respectively.
    Distinguish unreadable/changed inputs and output write failures. A valid
    calculation request with no usable observation returns its domain outcome.

34. **Known defects.** Before correcting algorithms, add behavioral regressions
    for line-coordinate mismatch, matrix direction, edge-pair selection,
    control-template initialization, stale state after exceptions, composed
    failure propagation, duplicated execution behavior and scale-normalized
    dimensions. Preserve known-good low-level behavior while consolidating
    duplicate execution paths. A confirmed horizontal-line integration case
    that returned about 100.5 pixels must return about 40.5 pixels.

35. **API coverage.** All operations below are required. Paths are relative
    to the V2 HTTP prefix; these are public route contracts, not implementation
    file locations. Any intentional contract change must update OpenAPI,
    capabilities, client examples and request tests together. No capability
    family may be omitted from final acceptance.

| Family | Required routes |
| --- | --- |
| Discovery | GET /health/live, /health/ready, /capabilities, /openapi.json, /docs |
| Images | POST /image-inspections, /image-previews, /image-preprocessings, /image-crops |
| Templates | POST /template-builds, /template-inspections, /template-matches |
| Alignment | POST /transforms/similarity-estimates, /alignments/rigid, /alignments/control-points, /geometry/transforms |
| Probes | POST /edge-profiles, /measurements/edge-points, /measurements/edge-pairs |
| Fits/features | POST /measurements/lines, /measurements/circles, /measurements/template-points, /geometry/lines:fit, /geometry/circles:fit, /measurement-batches |
| Geometry | POST /geometry/two-point-lines, /geometry/point-distances, /geometry/point-line-distances, /geometry/line-angles, /geometry/point-circle-distances, /geometry/circle-dimensions, /geometry/length-conversions |
| Features/calibration | POST /calibration-features/checkerboard-corners, /calibration-features/dot-grids, /calibrations/scales, /calibrations/cameras, /calibrations/plane-mappings, /calibrations/stereo-rigs |
| Calibration utilities | POST /calibration-validations, /calibration-origin-adjustments, /calibration-inspections, /calibration-exports, /calibration-imports |
| Correction/stereo | POST /image-undistortions, /image-stitches, /geometry/stereo-transforms, /geometry/camera-projections |
| Recipe | POST /recipe-validations, /recipe-previews, /runs, /run-batches, /recipe-exports, /recipe-imports, /legacy-project-conversions |
| Evaluation | POST /evaluations/ranges, /evaluations/diameter-variation, /evaluations/centerline-deviation |
| Results | POST /visualizations, /result-exports, /replays |
| Operations | GET /operations, /operations/{id}, /operations/{id}/diagnostics, /traces/{trace_id}; POST /operations/{id}:cancel |

The operation response distinguishes terminal deadline from synchronous
waiting: timeout_ms is the operation budget, while wait_ms controls only
how long a synchronous request or bounded poll waits. File-only output
options are optional for calculations that return compact geometry; an
output location is required when raster/numeric/package files are requested.

36. **Delivery gates.** First complete schemas and file contracts, then extract
    and correct the shared core, deliver atomic operations, deliver supplied
    recipe execution, add all calibration/diagnostic/format capabilities, and
    replace old V2 startup/contracts. Each phase has public behavioral tests.
    Existing GUI and V1 may remain usable during implementation but are not
    runtime dependencies. Final acceptance requires full capability coverage
    and the complete upstream workflow, not partial primary-tool delivery.

## Testing Decisions

1. The primary proposed test seam is the public FastAPI HTTP contract with
   real temporary shared files and observable results, outputs and task state.
   This is the highest integration interface upstream actually uses. The seam
   has been presented to the user for confirmation; writing this specification
   does not imply the user has already approved that proposal.

2. A good test submits supported configuration and asserts a numerical,
   protocol or filesystem outcome a caller can rely on. It does not assert
   private helper calls, cache dictionary layout, router organization, private
   factory names or exact implementation sequence.

3. Use FastAPI/ASGI request tests exercising application lifespan and injected
   settings. Run calculations against real fixtures rather than universally
   mocking the computation module. Controlled clocks/workers/storage failures
   are reserved for lifecycle cases that need deterministic boundaries.

4. Keep existing pytest numerical algorithm tests as focused lower-level
   regression tests. Existing edge, line/circle, template, similarity,
   dependency, scale calibration and camera/stereo suites provide synthetic
   generators, degenerate inputs, serialization examples and known tolerances.
   Do not create a separate new public testing interface for every helper.

5. Existing HTTP tests provide prior art for full request sequences and
   checking route/schema consistency. Rewrite their obsolete assumptions
   rather than keeping shared-memory/session/publication behavior as required
   compatibility. Existing operational tests provide prior art for deadline,
   worker interruption, partial results and structured-error checks.

6. Assert generated OpenAPI agrees with actual routes, discriminators,
   parameters, statuses, units and outputs. Test Pydantic validation, unknown
   fields, nonfinite numbers, invalid geometry and consistent error envelopes.

7. Run a complete upstream-equivalent scenario: commit reference file, inspect
   and build a template, preview tools and compositions, validate a supplied
   recipe, commit its upstream version, execute a new image, render outputs,
   export and replay. Require no V2 session or asset registration.

8. Cover root aliases with different mount prefixes, allowed absolute paths,
   missing/unreadable files, symlink resolution, expected hash mismatch,
   rewritten input and output destinations outside the configured roots.

9. Assert incomplete input and output files cannot become accepted completed
   assets. Inject failure before final manifest commit and verify there is no
   usable completed manifest. Verify completed output preserves input files
   and upstream-owned retention after engine cleanup.

10. Verify synchronous/asynchronous numerical equivalence, queryable pending
    results after sync wait expiry, queue/input-preparation deadline accounting,
    admission limits and HTTP responsiveness while computation is running.

11. Verify stable terminal states, cancellation and timeout before/after commit,
    worker interruption, restart reporting and output conflicts. Use explicit
    synchronization around commits rather than flaky timing-only assertions.

12. Verify retry behavior through public operation identity and files: repeated
    valid idempotency keys identify the original accepted work; changed content
    conflicts; retention expiry does not imply perpetual deduplication.

13. Compare observations across cache hits, misses, eviction and service
    restart. Replacing content under a reused path must not reuse old template
    pixels or observations. Assert results, not cache implementation details.

14. Use lossless synthetic fixtures with known edge positions, thickness,
    straight lines, circles and feature coordinates. Check polarity, first/
    last/outermost pair behavior, no-edge cases, insufficient samples,
    coincident points and collinear circle inputs.

15. Add the fitted-horizontal-line plus point-distance regression through the
    public calculation interface. Combine generated line coefficients with
    point geometry so the historical row/column mismatch is exercised rather
    than bypassed with hand-authored coefficients.

16. Exercise translated, rotated, scaled and multiple targets, overlap
    suppression, ROI/square support, fine/pyramid search and control-point
    fallback. Confirm a larger target retains its measured size after
    scale-normalized alignment. Verify controls cannot cross target instances.

17. Test node output ports, arrays versus single values, dependency closure,
    cycles, missing/disabled nodes, label renaming, display reorder and stale
    assets. Confirm invalid observations propagate and cannot reuse a prior
    successful result in a subsequent request.

18. Confirm coordinate mapping and requested units with synthetic transforms
    and calibrated fixtures. Test nonlinear undistortion provenance and circle
    conversion restrictions. Pixel and physical values remain separately
    identifiable and traceable to the supplied calibration.

19. Test partial checkerboards, manual corner edits, lattice rejection,
    dot-grid ordering, rejected views, image dimensions, paired stereo
    captures, reprojection residuals, plane canvases and revised origins.
    Existing synthetic calibration tests are prior art, supplemented by
    appropriate real camera fixtures for acceptance.

20. Verify operation status, measurement validity and rule evaluation remain
    distinct for no targets, partial measurements, invalid required outputs,
    missing policy and threshold failure. Test the existing business metrics
    from explicit supplied values with unit and invalid-input checks.

21. Validate exported vector geometry, raster dimensions, curves, CSV/JSON
    values and file metadata. Replay archived inputs after operational record
    expiry and assert explicit errors when retained files are unavailable or
    incompatible. Verify replay does not modify source assets.

22. Verify the replacement imports and starts without GUI initialization,
    interactive windows or private GUI factories. Keep all tests headless and
    place generated artifacts only in ignored test-output locations.

23. Final integration acceptance uses representative upstream requests and
    actual camera data. Agree on dimensional accuracy, repeatability,
    supported target counts, concurrent workload, memory and latency before
    claiming production performance. Existing unit tests do not establish
    those guarantees.

## Out of Scope

- Upstream application implementation, GUI redesign, operator permissions,
  business project CRUD and UI layout/selection state.
- V2 ownership of recipe/template/calibration storage, draft editing, version
  publication, asset deletion or a permanent business result database.
- Compatibility with the old V2 public session, model publication and
  shared-memory contracts. Conversion of supported GUI project formats does
  not promise compatibility with every old V2 asset package.
- Required HTTP binary uploads/downloads, base64 transport for large payloads,
  shared-memory image pools or cross-node large-file exchange.
- New camera acquisition SDKs, PLC triggers, rejection control and a complete
  production-line control application.
- Arbitrary 3D reconstruction, unrestricted affine metrology, ellipse fitting,
  certified roundness/verticality or implicit millimeter conversion from
  intrinsic calibration alone.
- Silent support for higher-bit-depth inputs without an explicit conversion
  and verified algorithm contract.
- Unrestricted user scripting, server-path access outside configured shared
  roots, autonomous deletion of upstream assets and historical mutation replay.
- Cluster-wide permanent exactly-once execution, multi-node engine scheduling
  or multiple HTTP workers with uncoordinated local operation state.

## Further Notes

- This specification consolidates the final conversation decisions: FastAPI,
  upstream asset ownership, same-node deployment, HTTP control/file references
  and shared-directory large files. Earlier proposals for mandatory uploads
  and engine-owned recipe publication are superseded.
- No applicable domain glossary or ADR was found during repository inspection.
  Use the existing recipe, template, calibration, measurement object,
  dependency, target and result vocabulary. Operational state is distinct
  from upstream business asset ownership.
- Shared root mappings, input/output permissions, concrete format versions,
  job retention, queue/cache limits and production performance thresholds are
  deployment parameters to define during implementation; they do not reopen
  confirmed ownership or transport decisions.
- The requested tracker triage label is ready-for-agent. No project issue
  tracker configuration was found, so publication has not been performed.
  The to-spec skill requires setup-matt-pocock-skills when tracker setup is
  missing. The local specification is the reviewable artifact until that
  configuration is available.
- The public HTTP test-seam proposal has been sent for user feedback. Its
  proposed default is recorded in Testing Decisions rather than represented
  as an approval already received.
- This task creates the specification only. It does not delete current V2,
  modify measurement algorithms, change runtime dependencies or claim that
  the described contract has been implemented or production-validated.

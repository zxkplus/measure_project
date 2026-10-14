# Supported FastAPI V2 contract

The replacement has no server-owned recipe registry or modeling session. Upstream supplies
immutable configuration/files and retains authoritative versions. Only the `/api/v2` prefix is
shared with the retired V2 API. Start/deploy using [the deployment guide](v2-production-engine.md).
The runtime generates OpenAPI and capability schemas from the same executable operation registry.
For upstream integration, use [the API guide](v2-upstream-api.md),
[complete field reference](v2-api-reference.md) and [OpenAPI snapshot](v2-openapi.json).
The [required route manifest](v2-required-routes.json) records the 65 Spec requirements and the
HTTP discovery acceptance test compares them with actual OpenAPI routes. Presence alone is not
numerical acceptance; the family suites exercise the resulting observations and file effects.

## Required routes

| Family | Method | Route |
|---|---|---|
| Discovery | GET | `/api/v2/health/live` |
| Discovery | GET | `/api/v2/health/ready` |
| Discovery | GET | `/api/v2/capabilities` |
| Discovery | GET | `/api/v2/openapi.json` |
| Discovery | GET | `/api/v2/docs` |
| Images | POST | `/api/v2/image-inspections` |
| Images | POST | `/api/v2/image-previews` |
| Images | POST | `/api/v2/image-preprocessings` |
| Images | POST | `/api/v2/image-crops` |
| Templates | POST | `/api/v2/template-builds` |
| Templates | POST | `/api/v2/template-inspections` |
| Templates | POST | `/api/v2/template-matches` |
| Alignment | POST | `/api/v2/transforms/similarity-estimates` |
| Alignment | POST | `/api/v2/alignments/rigid` |
| Alignment | POST | `/api/v2/alignments/control-points` |
| Alignment | POST | `/api/v2/geometry/transforms` |
| Probes | POST | `/api/v2/edge-profiles` |
| Probes | POST | `/api/v2/measurements/edge-points` |
| Probes | POST | `/api/v2/measurements/edge-pairs` |
| Fits/features | POST | `/api/v2/measurements/lines` |
| Fits/features | POST | `/api/v2/measurements/circles` |
| Fits/features | POST | `/api/v2/measurements/template-points` |
| Fits/features | POST | `/api/v2/geometry/lines:fit` |
| Fits/features | POST | `/api/v2/geometry/circles:fit` |
| Fits/features | POST | `/api/v2/measurement-batches` |
| Geometry | POST | `/api/v2/geometry/two-point-lines` |
| Geometry | POST | `/api/v2/geometry/point-distances` |
| Geometry | POST | `/api/v2/geometry/point-line-distances` |
| Geometry | POST | `/api/v2/geometry/line-angles` |
| Geometry | POST | `/api/v2/geometry/point-circle-distances` |
| Geometry | POST | `/api/v2/geometry/circle-dimensions` |
| Geometry | POST | `/api/v2/geometry/length-conversions` |
| Features/calibration | POST | `/api/v2/calibration-features/checkerboard-corners` |
| Features/calibration | POST | `/api/v2/calibration-features/dot-grids` |
| Features/calibration | POST | `/api/v2/calibrations/scales` |
| Features/calibration | POST | `/api/v2/calibrations/cameras` |
| Features/calibration | POST | `/api/v2/calibrations/plane-mappings` |
| Features/calibration | POST | `/api/v2/calibrations/stereo-rigs` |
| Calibration utilities | POST | `/api/v2/calibration-validations` |
| Calibration utilities | POST | `/api/v2/calibration-origin-adjustments` |
| Calibration utilities | POST | `/api/v2/calibration-inspections` |
| Calibration utilities | POST | `/api/v2/calibration-exports` |
| Calibration utilities | POST | `/api/v2/calibration-imports` |
| Correction/stereo | POST | `/api/v2/image-undistortions` |
| Correction/stereo | POST | `/api/v2/image-stitches` |
| Correction/stereo | POST | `/api/v2/geometry/stereo-transforms` |
| Correction/stereo | POST | `/api/v2/geometry/camera-projections` |
| Recipe | POST | `/api/v2/recipe-validations` |
| Recipe | POST | `/api/v2/recipe-previews` |
| Recipe | POST | `/api/v2/runs` |
| Recipe | POST | `/api/v2/run-batches` |
| Recipe | POST | `/api/v2/recipe-exports` |
| Recipe | POST | `/api/v2/recipe-imports` |
| Recipe | POST | `/api/v2/legacy-project-conversions` |
| Evaluation | POST | `/api/v2/evaluations/ranges` |
| Evaluation | POST | `/api/v2/evaluations/diameter-variation` |
| Evaluation | POST | `/api/v2/evaluations/centerline-deviation` |
| Results | POST | `/api/v2/visualizations` |
| Results | POST | `/api/v2/result-exports` |
| Results | POST | `/api/v2/replays` |
| Operations | GET | `/api/v2/operations` |
| Operations | GET | `/api/v2/operations/{id}` |
| Operations | GET | `/api/v2/operations/{id}/diagnostics` |
| Operations | GET | `/api/v2/traces/{trace_id}` |
| Operations | POST | `/api/v2/operations/{id}:cancel` |

## Public data and coordinate conventions

Points are `{x,y}` with x right/y down, top-left pixel-center origin. Angles are clockwise degrees.
Probes explicitly separate search half-length from averaging half-width. Crops use full pixel
width/height. All transformations identify source and destination frames, forward direction and
inverse mapping where mathematically defined. Calibration identity includes camera/setup/image
size/frame applicability. Camera intrinsics alone do not establish physical scale.

Observation validity, operation state and product evaluation are independent. A successfully
computed result may contain missing edges, invalid geometry or a failed product rule. Recipe nodes
use stable IDs and named typed output ports; presentation labels/order do not define dependencies.
Physical geometry is built from mapped samples, preserving actual target size after localization
and normalization. Pixel and physical values remain separately traceable in run and export data.

File references use configured root aliases or restricted absolute engine paths; SHA-256 verifies
expected bytes. Template/calibration/recipe file references are transitively resolved and hashed.
Artifacts carry kind, format, schema version, byte size and digest. The final manifest marks the
complete artifact set. Upstream owns all completed files and input retention.

## Family guides

- [Image/lifecycle and extension contracts](../measure_api/v2/README.md)
- [Atomic measurements, geometry and evaluations](../measure_api/v2/ATOMIC_MEASUREMENTS.md)
- [Template localization and alignment](v2-template-alignment-contract.md)
- [Scalar/camera calibration](../measure_api/v2/CALIBRATIONS.md)
- [Plane/stereo/stitching/projection](../measure_api/v2/PLANE_STEREO.md)
- [Recipe validation, preview, target isolation and batches](../measure_api/v2/RECIPES.md)
- [Visualization and structured exports](../measure_api/v2/RESULTS.md)
- [Portable recipes, legacy conversion and archived replay](../measure_api/v2/PORTABLE_RECIPES.md)

## Applicability and qualification

The service accepts PNG/JPEG GRAY8/BGR8 images and explicitly rejects unsupported bit-depth/channel
layouts. Numeric preprocessing outputs are separate non-pickle arrays; previews do not replace
measurement pixels. Scalar calibration assumes a uniform plane scale. Plane/stereo assets require
valid supplied camera geometry and applicability; nonplanar objects cannot inherit a plane scale
without a justified model. Nonlinear/projective transformations map samples before fitting; they
cannot generally transform a circle by multiplying its radius by one scale.

One Unix ASGI owner manages one persistent state directory and bounded spawned calculations.
Runtime retention supports finite retry deduplication and restart diagnostics, not an upstream
business archive. Frozen replay depends on retained compatible runtime identity and input assets;
its archived manifest does not depend on a historical operation record. Legacy conversion rejects
unsupported/missing configuration rather than inventing defaults for unavailable capabilities.

Synthetic tests verify contract/numerical cases. Real samples can record latency, worker resident
memory, fit observations, localization target count and unchanged-file repeatability. Dimensional
accuracy and capture repeatability need traceable truth, calibrated representative camera captures
and agreed thresholds. See [the acceptance evidence](v2-acceptance.md) for what has actually been
verified and what remains unqualified.

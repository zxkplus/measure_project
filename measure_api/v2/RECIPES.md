# Supplied recipe calculations

The recipe family exposes `POST /recipe-validations`, `/recipe-previews`, `/runs` and
`/run-batches` beneath `/api/v2`. It owns no recipe drafts, names, publication, versions or
business rule storage. Public schema classes are in `recipe_models.py`; the single headless
executor is `recipe_execution.py`, with typed graph validation in `recipe_graph.py` and sample
mapping/refitting in `recipe_metrology.py`. It invokes the same registered atomic functions as
HTTP operations, without internal HTTP, GUI factories or a second worker scheduler.

## Snapshot and schema

Every call requires exactly one `recipe` value or `recipe_file` shared-file reference. Recipe
schema `1.0` requires upstream `id`, explicit `version` and `nodes`; no latest version is inferred.
Preparation loads file recipes, validates portable template provenance, materializes atomic typed
parameters, normalizes calibration family schemas, and pins all transitive file inputs before
admission/idempotency. Results echo `recipe_id`, `recipe_version`, `resolved_recipe`, verified
`assets` and `configuration_sha256`. The hash covers canonical resolved JSON, including labels and
asset digests, and excludes execution controls and preview tokens. Reusing an idempotency key
with changed recipe or asset bytes returns 409. Input files stay immutable until terminal state.

The request's `assets` and `source_recipe` fields carry normalized preparation provenance. The
preparer rebuilds assets from the supplied configuration, so caller values cannot bypass reads or
content verification. Calibration JSON paths are replaced by immutable normalized values while
original file digests remain in the assets list and idempotency identity.

Recipe fields:

- `reference`: explicit shared teaching image; every localization/control/local-point template must
  have the same source content digest. A pathname or a matching label is insufficient provenance.
- `source_frame`: final published pixel metrology frame, default `source`. Original camera pixels
  are recovered through correction inverses. Stitching requires the calibration's canvas frame.
- `stages`: ordered correction (`image-undistortions`) and/or stitching (`image-stitches`) with
  unique IDs and the corresponding atomic operation's parameters. `image` for correction is the
  current pipeline image. Stitching supplies its paired images/applicabilities/capture identity
  explicitly and must be first. The parent owns scheduling/output controls.
- `localization`: exactly one portable `template` plus `search`, or a `fixed_roi`. Template search
  returns bounded independent target poses, using the atomic score/angle/scale/pyramid contract.
  `max_targets` limits the selected matches. Omission runs once on the entire supplied image.
- `alignment`: patch dimensions, a pixel similarity `reference_to_patch`, and optional atomic
  `controls`/robust/residual/failure policy. For template poses, initial alignment is
  `reference_to_patch * inverse(reference_to_inspection)`. Without explicit alignment, template
  targets normalize into the supplied reference image's dimensions. A fixed ROI first rectifies
  source pixels into an ROI-local patch; its optional `reference_to_patch` acts on those ROI-local
  coordinates. Omission preserves that patch. Fixed ROI tools therefore use ROI-local coordinates.
- `metric`: supplied scalar/plane/stereo calibration plus exact `applicability`, and optional
  `physical_frame` (default `physical`). Camera intrinsics alone cannot provide physical scale.
  Without stitching, applicability's pixel frame must equal `source_frame`: correction stages
  recover original pixels through their inverse, so a `source` recipe still uses a `source`
  calibration. A separately supplied corrected image may declare `corrected-source` when its
  calibration applies to that exact frame. For a stitched plane metric, the metric and stitch
  stage must use the same canonical calibration digest and canvas frame. After an origin
  adjustment, upstream must replace both asset references together. A scalar canvas calibration
  is accepted only for the actual canvas frame and its calibrated dimensions.
- `target_policy`: optional inclusive `minimum`/`maximum` counts and `no_targets` business status
  (`not_evaluated`, `fail` or `pass`). This policy affects conformity; an empty set has no valid
  measurements regardless of the business status.
- `required_outputs`: explicit `{node, port, space}` bindings whose observation validity gates
  target validity, independently of optional diagnostics.

## Nodes and ports

A node has stable `id`, atomic `operation`, `parameters`, named `inputs`, `enabled` (default true),
`required` (default true), optional display `label`, and `space` (`pixel` by default). Operation
identifiers come from capabilities, e.g. `measurements.edge-pairs`, `geometry.point-distances`,
`evaluations.ranges`; route-style names normalize to the same identifiers. Nodes support atomic
measurements, geometric fits/composition and supplied business evaluations. Teaching and parent
matching belong to the recipe's declared stages/localization, and recursive recipe nodes are rejected.

`parameters` contains literal fields of the atomic request. `inputs` binds atomic request field
names to `{node, port, space}`, with space `pixel` or `physical`. An explicit list of bindings
constructs an array input; an array result port never silently becomes its first element. Scalar
ports preserve validity, reason, frame and unit even when the atomic result field is a number.
Image and output fields are parent-managed. Literals and bound fields cannot overlap. Literal
geometry declares its actual published frame/unit; it is never implicitly interpreted as taught
geometry or calibrated by changing a label. Physical-only compositions must request physical
required-output ports. Business limits use the supplied unit; pixel limits are never auto-scaled.

Validation checks the full configuration: duplicate IDs, unknown fields/operations/ports,
missing/disabled providers, single/array and observation types, incompatible frames/units, cycles,
required outputs, portable provenance, calibration applicability and stage/alignment structure.
Nodes run in deterministic topological order; labels and list/display order do not choose providers.
Invalid typed dynamic parameters return 422 `recipe.invalid_parameter`, including a located
`field_path` such as `recipe.nodes.1.parameters.selection`, consistently for validation, preview
and run. Malformed request/asset schemas return 422; semantic DAG validation returns `valid: false` and
structured `issues`. Runs reject invalid graphs before image calculation.

Example with an edge pair and a range check (forward references are allowed):

```json
{
  "image": {"root": "exchange", "path": "inspection.png"},
  "recipe": {
    "id": "bottle", "version": "upstream-42",
    "nodes": [
      {
        "id": "accepted", "operation": "evaluations.ranges",
        "parameters": {"minimum": 19.5, "maximum": 20.5, "unit": "px"},
        "inputs": {"value": {"node": "width", "port": "width"}}
      },
      {
        "id": "width", "operation": "measurements.edge-pairs",
        "parameters": {
          "probe": {"center": {"x": 50, "y": 50},
                    "search_half_length": 25, "average_half_width": 3},
          "selection": "first", "polarity": "positive"
        }
      }
    ],
    "required_outputs": [{"node": "accepted", "port": "value"}]
  }
}
```

## Preview, metrology and validity

Preview adds `selected_nodes`, `upstream_revision` and `preview_token`. The full snapshot is
validated, but only selected nodes and their dependency closure run. Revision/token are echoed
unchanged. Unselected required outputs do not fail a preview, and no configuration is saved.

Each run returns `status` (`completed`, `partial_failure`, `no_targets`), measurement `valid`,
`evaluation_status`, local `targets`, source references, image-stage provenance and stage timings.
The HTTP operation envelope separately states execution success/failure. No matches return an
empty target list and explicit `no_targets`. Target IDs have meaning only within that run.

Every target has fresh node outcomes with `state`, `valid`, authoritative `result`, separate
`pixel_result`/`physical_result`, and structured errors. Pixel observations recover actual source
size after normalization: a taught 20px pair detected at scale 1.2 returns 24 source px, and an
applicable 0.1mm/px asset gives 2.4mm. Selected line/circle samples map into the chosen frame before
refitting; residuals are recomputed in that frame. Projective circle metrology uses mapped real
samples, rather than applying a radius multiplier to a fitted source ellipse. Analytical supplied
geometry must already declare its actual frame/unit; arbitrary projective circle conversion is
not inferred. Edge amplitudes retain their original signal meaning while edge locations are mapped.

`metric_mapping` explicitly maps aligned samples back to pixel metrology. Correction chains expose
`type: nonlinear`, `details.model: patch_to_corrected_to_source`, a `patch_to_image` matrix and
`image_to_source` correction details. They do not invent an affine inverse for distorted pixels.
`physical_mapping` maps published pixel observations to physical coordinates, with exact inverse
matrices for scalar/zero-distortion plane/canvas mappings. Distorted plane mappings declare
`model: opencv_pinhole_plane`, `camera_matrix`, `distortion`, `corrected_to_physical` and
`physical_to_corrected`. The reverse first maps physical to corrected K-pixels, normalizes through
inverse K, and uses OpenCV projection with K/D. No mapping descriptors reference private files.

Required nodes/output observations gate target validity. Invalid domain observations preserve
null geometry/value; downstream typed observation bindings propagate invalidity. Exceptions clear
previous observations, create invalid dependency ports, and allow supplied evaluation
`invalid_policy` to run. Unsupported array dependencies become blocked. Independent nodes,
targets and batch items continue. `evaluation_status` is separate from measurement validity,
including a policy's deliberate `fail` on a missing value.

## Outputs and batches

Correction/stitch/alignment execute through `context.stage()` in private, cancellable directories.
Those manifests cannot finalize the parent operation. Temporary files are never returned as durable
references. With `output.directory`, a run commits `result.json`, while a complete batch commits
`batch.json`, using one final parent manifest after all requested results are validated.

`retain_stage_images: true` on a run/preview additionally retains declared correction/stitch PNGs
in that same parent namespace and requires `output.directory`. Source-frame measurements draw
on the original verified input (`measurement_image`). For a stitched canvas this field is null
unless images are retained, then it points to the final retained canvas image. Upstream may also
supply an independently committed canvas image for subsequent rendering. Original paired camera
images remain in `sources`; `image_stages` contains explicit canvas/camera mappings and provenance.
The saved calculation JSON records numerical results/provenance and the same durable
`measurement_image` reference as the HTTP result, including its final shared path and verified
content digest. It omits its own JSON descriptor and final manifest to avoid self-referential
hashes; returned artifact descriptors identify those committed files and the completion manifest.

A `run-batches` request supplies one common recipe and bounded unique-ID `items` with `image`.
Each item may override the recipe using exactly one inline/file form. Preparation failures (such
as missing image files) remain item outcomes, and usable items still execute the same full runner.
Each item result has independent target-local state. No child commits a parent business manifest.
Batches retain item outcomes in caller order; computation inside a recipe remains topological.

## Acceptance

`tests/v2/test_recipes.py` exercises the public FastAPI seam with real shared files: teaching,
validation, unsaved preview, calibrated runs, true scale normalization, repeated variable-scale
targets, missing targets, confined controls, partial batches, source-content retry conflicts,
invalid business policies, plane line refitting and a projected ellipse whose mapped samples
recover a physical circle. It also exercises declared undistortion/stitching, final manifests and
optional retention without temporary references. Synthetic metrology proves these integration
contracts; production accuracy still requires representative cameras, optics and agreed truth data.

Template-point nodes publish the compact located point separately in `pixel_result` and
`physical_result`, with each observation's own frame and unit. Strongly typed
`localization_diagnostics` preserve the original sampling-frame match, score, angle and scale;
that pose remains in aligned pixel coordinates even when the physical point is in millimeters.
The complete supplied RunResult can be visualized/exported without changing this metadata.

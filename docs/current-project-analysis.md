> Historical analysis of the pre-rewrite project. The V2 deployment contract is superseded by
> [the FastAPI deployment guide](v2-production-engine.md).

# Current project functionality and measurement flow

Analysis date: 2026-10-05. Evidence is the current source tree, not historical README claims.
This document records source inspection; it does not claim production metrology accuracy or a
complete interactive GUI acceptance test.

## GUI product model

The actual desktop application uses Python `tkinter` and `ttk`, not PyQt. `run_gui.py` imports
`measure_gui.app.main`, which creates a `tk.Tk`, constructs `MeasureApp`, and runs `mainloop`.
Sources: [run_gui.py](../run_gui.py#L15), [app.py](../measure_gui/app.py#L23),
[main](../measure_gui/app.py#L1374).

The product is an image-based, teach-once/inspect-many measurement workbench. A reference image
defines a reusable target template and measurement recipe; an inspection image can contain
multiple instances of that target. The recipe is applied in a normalized target patch for each
matched instance. Its visible workflow is reference teaching, optional physical pixel calibration,
project saving, inspection execution, and result review/export.
Sources: [GUI teaching help](../measure_gui/app.py#L1318),
[inspection action](../measure_gui/app.py#L535),
[actual per-target execution](../measure_gui/multi_target.py#L2316).

## GUI modules and exposed capabilities

| Module | Exposed capabilities | Source |
| --- | --- | --- |
| Main application and menus | New/open/save/save-as projects; reference and inspection image loading; recent projects; fit/100% image views; reset ROI; calibration and help | [app.py](../measure_gui/app.py#L101) |
| Tool panel | File operations, preprocessing (`raw`, `canny`, `sobel`, `clahe`, `threshold`), matching score, angle search range and step, maximum matches, NMS overlap, pyramid downsampling, recipe list, edit/delete/reorder/display toggles, execute and debug-image checkbox | [tool_panel.py](../measure_gui/tool_panel.py#L102) |
| Reference/inspection canvas | Separate tabs; reference ROI/control-point editing; inspection visual overlays; zoom and pan | [app.py](../measure_gui/app.py#L193), [image_canvas.py](../measure_gui/image_canvas.py#L31) |
| Template view | Straightened target image; add edge point, edge pair, fitted line, fitted circle, local template-match point; interactive placement, editing, deletion and overlays | [template_view.py](../measure_gui/template_view.py#L120) |
| Parameter dialogs | Sampling half-length/width, smoothing sigma, threshold, interpolation, probe counts, line/circle search dimensions, edge polarity; local-template score/rotation/subpixel/pyramid options | [dialogs.py](../measure_gui/dialogs.py#L119), [template dialog](../measure_gui/dialogs.py#L241) |
| Composed measurements | Two-point distance, two-line angle, point-line distance, point-circle boundary distance; references to prior recipe labels | [tool_panel.py](../measure_gui/tool_panel.py#L270) |
| Calibration dialog | Load checkerboard image, enter known square side in mm, select ROI, tune corner detection, manually add/move/delete corners, review and apply a scalar mm/px ratio | [calibration_dialog.py](../measure_gui/calibration_dialog.py#L105), [corner edits](../measure_gui/calibration_dialog.py#L301) |
| Result panel | Target list with ID/score/rotation/center/validity; selected target measurement table with value/error/ellipticity/validity; text summary; CSV export | [result_panel.py](../measure_gui/result_panel.py#L47) |
| Project manager | Portable project directory, workflow pixels/definitions, image copies, GUI/ROI/matching/alignment/calibration state and recent-project list; legacy NPZ loading | [project_manager.py](../measure_gui/project_manager.py#L1) |

`TwoPointsLine` also exists in the backend registry and executor, but the current tool-panel
construction exposes only the four composed buttons listed above. It should not be described as
a visible GUI action. Sources: [backend registry](../measure_gui/multi_target.py#L1478),
[backend executor](../measure_gui/multi_target.py#L2430),
[visible buttons](../measure_gui/tool_panel.py#L274).

## Actual teaching and inspection sequence

1. Load a reference image as grayscale. This creates a fresh GUI workflow and clears recipe and
   results. Source: [app.py](../measure_gui/app.py#L254).
2. Define a rotated rectangular target ROI by center, size and angle. Confirming the ROI creates a
   template; the explicit create/update action does the same. The template view displays a cropped,
   straightened patch, while parent localization teaches a square template around the original
   reference center with size `max(height, width)`. Sources:
   [ROI confirmation](../measure_gui/app.py#L825),
   [template construction](../measure_gui/app.py#L903),
   [alignment teaching](../measure_gui/alignment.py#L138).
3. Place primitive tools in straightened-template coordinates. Each tool is stored as a type,
   label and parameter dictionary in the workflow. Add composed dimensions by referencing tool
   labels; an edge pair exposes virtual labels `<label>_A` and `<label>_B` for its two endpoints.
   Sources: [tool callbacks](../measure_gui/app.py#L1004),
   [composed label expansion](../measure_gui/tool_panel.py#L533).
4. Optionally calibrate mm per pixel using a known checkerboard square dimension. The GUI stores
   a scalar conversion factor in the app and active workflow; it does not apply full camera
   intrinsic/extrinsic calibration or perspective correction in this action. Sources:
   [calibration application](../measure_gui/app.py#L614),
   [scale application](../measure_gui/calibration_dialog.py#L324).
5. Save a project containing the taught recipe and state. Source:
   [project save](../measure_gui/project_manager.py#L81).
6. Load an inspection image as grayscale and execute. The GUI checks that a template and
   inspection image exist, synchronizes score/count/overlap/angle range, calls `measure`, then
   displays per-target results, overlays, summary and elapsed time. Source:
   [inspection loading](../measure_gui/app.py#L291),
   [execution](../measure_gui/app.py#L535).
7. Parent template matching returns multiple candidate instances. Each instance is cropped and
   straightened by the alignment strategy; primitive and composed measurements execute in recipe
   insertion order; results are mapped for display on the inspection image. Sources:
   [multi-target matching](../measure_gui/multi_target.py#L2263),
   [per-target execution](../measure_gui/multi_target.py#L2316).
8. Review targets and export CSV. Dimensions remain pixel values internally; summaries, table and
   CSV format lengths as `pixel_value * scale_mm_per_px` when calibration exists. Sources:
   [result table](../measure_gui/result_panel.py#L193),
   [CSV](../measure_gui/result_panel.py#L273),
   [length conversion](../measure_gui/multi_target.py#L3117).

## Project format

The current project manifest version is 3. A directory holds `project.json`, `workflow.npz`,
`reference.png`, optional `inspection.png`, and optional `calibration_board.png`.
The manifest stores matching controls, ROI, canvas zoom/pan, template-view tools, list order and
visibility, layout/tab state, alignment controls and calibration metadata. The workflow stores
reference/template pixels and measurement definitions. Sources:
[format constants](../measure_gui/project_manager.py#L32),
[manifest fields](../measure_gui/project_manager.py#L143),
[workflow save](../measure_gui/multi_target.py#L2679).

Loading restores recipe and GUI state, and prefers the project-local inspection-image copy for
portability. The legacy standalone `.npz` path restores the recipe list and template, but not the
template tool overlays. Sources: [project load](../measure_gui/project_manager.py#L168),
[inspection resolution](../measure_gui/project_manager.py#L308),
[legacy load](../measure_gui/project_manager.py#L335).

## Important implementation boundaries and current inconsistencies

- `measure_gui/multi_target.py` defines `MultiTargetWorkflow` twice (lines 500 and 1501). Python's
  later declaration is the effective class. The active `measure` method starts at line 2263.
  The desktop app explicitly imports this GUI workflow; the similarly named unified-package and
  root-level implementations should not be substituted when explaining the GUI path.
  Sources: [app import](../measure_gui/app.py#L37),
  [first declaration](../measure_gui/multi_target.py#L500),
  [later declaration](../measure_gui/multi_target.py#L1501).
- The GUI workflow executes definitions sequentially, without the dependency scheduler from
  `MeasurementWorkflow`; composed inputs must already be in `raw_results`.
  Successful `EdgePair` execution produces virtual endpoints, rather than a standalone width
  result. A two-point distance can reference those endpoints to produce width.
  Sources: [ordered loop](../measure_gui/multi_target.py#L2371),
  [virtual endpoints](../measure_gui/multi_target.py#L2383),
  [two-point distance](../measure_gui/multi_target.py#L2461).
- GUI move-up/down updates only the Treeview order; it does not call workflow reorder methods.
  Execution parameter synchronization does not synchronize recipe order. Therefore the visible
  ordering and actual execution order can differ.
  Sources: [tree reorder](../measure_gui/tool_panel.py#L517),
  [execution synchronization](../measure_gui/app.py#L1131).
- Visibility toggles affect template overlays, not measurement execution. Selecting a target
  updates the table and status text; the app's selection callback does not currently zoom to or
  highlight the target. Sources: [visibility callback](../measure_gui/app.py#L1110),
  [selection callback](../measure_gui/app.py#L1250).
- The main GUI exposes a "multi-point affine" alignment mode and a backend refinement strategy,
  but its current confirmation callback is not compatible with that strategy: it calls an absent
  `set_parent_search_angle_range` method and then calls `teach` without required ROI geometry.
  Do not describe this GUI path as verified usable. The backend uses
  `cv2.estimateAffinePartial2D` (a similarity transform), not unrestricted six-parameter affine
  correction, and falls back to rigid alignment if insufficient control points succeed.
  Sources: [GUI callback](../measure_gui/app.py#L1172),
  [required teach signature](../measure_gui/alignment.py#L116),
  [backend refinement](../measure_gui/alignment.py#L443).
- The `all_valid` target flag is updated for primitive failures and exceptions, but composed-result
  failure branches do not update it. A target can therefore be shown valid despite an invalid
  composed dimension. The validity indicator also does not encode dimensional tolerance or
  product acceptance criteria. Sources: [primitive failures](../measure_gui/multi_target.py#L2393),
  [composed branches](../measure_gui/multi_target.py#L2461),
  [target validity](../measure_gui/multi_target.py#L2565).
- Point coordinates, line endpoints and circle centers are mapped back to the inspection image.
  Line coefficients remain patch coefficients, circle radius and composed distances remain patch
  values. This is consistent for scalar dimensions under rigid alignment, but further care is
  required if scaled or affine alignment is used.
  Source: [result mapping](../measure_gui/multi_target.py#L2944).
- Execute runs `measure` synchronously in the GUI callback; there is no worker-thread dispatch in
  this action. Long matching jobs can block GUI interaction.
  Source: [execute callback](../measure_gui/app.py#L545).
- Recreating the rectangular template clears existing measurement tools and definitions.
  Changing preprocessing requires template recreation. Search angle step and pyramid controls
  are passed during template construction, whereas execution synchronization updates only score,
  maximum count, NMS overlap and angle range.
  Sources: [template reset](../measure_gui/app.py#L980),
  [preprocessing change](../measure_gui/app.py#L1142),
  [template parameters](../measure_gui/app.py#L953).
- Debug-image output is enabled only when a project directory exists. Its timestamped directory
  includes template, target boxes, straightened patches, overlays and sampling-profile images.
  Sources: [debug toggle](../measure_gui/app.py#L1096),
  [debug outputs](../measure_gui/multi_target.py#L2283),
  [per-target debug](../measure_gui/multi_target.py#L2540).

## Verification performed

Read-only import/introspection using `.venv/bin/python` confirmed
`hasattr(MultiPointAlignment(), "set_parent_search_angle_range") == False`, the required
`teach(reference_image, center, size, angle_deg, ...)` signature, and
`MultiTargetWorkflow.measure.__code__.co_firstlineno == 2263`.
No source code was modified, and no interactive desktop acceptance test was performed.

## Measurement implementation and orchestration

The primary algorithm chain is:

1. Configure a measurement ROI from the reference image.
2. Locate the product in an inspection image and align its image patch.
3. Extract subpixel edge points or template-matched feature points.
4. Fit lines/circles, then calculate dependent distances and angles.
5. Return pixel-space results with validity and diagnostic metadata.
6. Apply the optional mm/pixel ratio when formatting GUI results.

### One-dimensional edge measurement

`Halcon1DMeasure` extracts a rotated rectangle with `cv2.warpAffine`.
Its output dimensions are twice the ROI half-length and half-width. It
averages each column into a one-dimensional gray profile, optionally smooths
it with a Gaussian, computes the gradient, selects positive/negative peaks
above the threshold, and refines peak positions with a quadratic fit. Peaks
are mapped back to image coordinates. Current peak detection and refinement
use vectorized helpers, unlike the older algorithm description in CLAUDE.md.

Positive transitions mean dark to bright; negative transitions mean bright to
dark. The threshold applies to the gradient magnitude. Sigma, ROI geometry,
transition polarity, and edge selection determine which physical boundary
is actually measured.

Sources: [ROI extraction](../measure/measure1D.py#L98),
[edge detection](../measure/measure1D.py#L131),
[vectorized signal processing](../measure/signal_ops.py#L15).

### Geometric primitives and composed measurements

| Object | Implementation and output |
| --- | --- |
| EdgePoint | One selected subpixel edge point. |
| EdgePair | A pair of boundaries; core output is its center with endpoints and width in metadata. GUI exposes virtual endpoints ending in _A and _B. |
| FitLine | Places probes perpendicular to a taught segment, selects edges nearest that segment, and fits normalized line coefficients using SVD. |
| FitCircle | Places radial probes around a taught circle or arc, filters by radius limits, chooses edges nearest the expected radius, and fits a circle using algebraic least squares. |
| TemplateMatchPoint | Matches a small local feature template and returns its location and score. |
| TwoPointsLine | Derives a line from two points. |
| TwoPointsDistance | Euclidean point-to-point distance. |
| PointLineDistance | Perpendicular distance to an infinite line. |
| TwoLinesAngle | Smaller unsigned angle, between 0 and 90 degrees. |
| PointCircleDistance | Absolute difference between point-to-center distance and radius. |

A circle diameter can be derived as twice its fitted radius. The additional
circle metric called ellipticity is implemented as twice the difference
between the largest and smallest sampled radial distances from the fitted
center; it is not an ellipse fit or a complete standardized roundness analysis.
A fitted line's displayed length is based on fitted endpoint span, including
the low-level fit's extension, and should not automatically be interpreted as
a complete object's physical length.

Sources: [line sampling/fitting](../measure/measure2D.py#L166),
[circle sampling/fitting](../measure/measure2D.py#L649),
[circle radial spread](../measure/measure2D.py#L775),
[edge-pair output](../measure/measure_workflow.py#L1141),
[composed objects](../measure/measure_workflow.py#L1596).

### Template matching and localization

TemplatePoint uses normalized cross correlation
(`cv2.matchTemplate(TM_CCOEFF_NORMED)`), optional subpixel refinement,
rotation/scale search, coarse-to-fine search, optional image downsampling,
score filtering, and non-maximum suppression for multiple instances.
Preprocessors include raw gray, Canny, Sobel, CLAHE, and binary thresholding.

The generic MeasurementWorkflow binds dependencies and topologically sorts
the object graph. Optional localization anchors estimate a similarity
transform: one anchor supplies translation; two or more supply rotation,
uniform scale, and translation. Localization failure invalidates the workflow.
Its calibrate methods relocate the taught image geometry; they do not perform
a physical mm conversion.

The GUI uses a different workflow: detect parent-template matches, crop and
straighten each product, execute definitions in stored order, calculate
composed measurements in that patch, then map geometric coordinates back to
the inspection image. Its ordinary single-box alignment uses translation and
rotation. Do not assume all capabilities of the generic workflow are active
in this GUI path.

Sources: [TemplatePoint](../measure/measure_template.py#L278),
[similarity estimation](../measure/measure_workflow.py#L230),
[dependency resolution](../measure/measure_workflow.py#L2020),
[generic execution](../measure/measure_workflow.py#L2133),
[GUI execution](../measure_gui/multi_target.py#L2263).

### Pixel-to-physical calibration

CheckerboardScaleCalibration detects checkerboard corners, rejects corners
inconsistent with the lattice, estimates neighboring-corner spacing, and
computes:

`scale_mm_per_px = square_size_mm / spacing_px`

`physical_length_mm = measured_length_px * scale_mm_per_px`

It supports partial boards and manually curated corners. This is a uniform
scale calibration intended for telecentric/orthographic imaging; it performs
no lens-distortion correction or perspective compensation. The scale is
stored with the GUI project and used in formatted results; underlying
measurement values remain in pixels.

Separate CameraCalibration and StereoRigCalibration classes provide
dot-grid camera intrinsics/distortion calibration and a two-camera board-plane
stitching/relative-pose workflow. These capabilities are available in the
library/scripts but are not automatically inserted into the GUI's ordinary
checkerboard scale and measurement path. Their existence does not establish
a general three-dimensional dimensional measurement product.

Sources: [scale assumptions](../measure/pixel_calibration.py#L1),
[scale computation](../measure/pixel_calibration.py#L246),
[GUI physical formatting](../measure_gui/multi_target.py#L3117),
[camera calibration](../measure/measure_calibration.py#L45),
[stereo rig](../measure/measure_calibration.py#L393).

## Product and service boundaries

A saved model is a reusable measurement recipe containing image templates,
ROI geometry, tool parameters, and relationships. Teaching is configuration
and parameter tuning, not neural-network training. The product separates
image localization, geometric measurement, unit conversion, and business
acceptance rules.

The legacy MeasureProject SDK and session-based Flask service reuse the GUI
workflow and factories, expose teaching and measurement operations, allow
individual-tool previews, and provide dependency inspection, persistence,
diagnostics, logs, and recorded-call replay. Their project format is distinct
from the GUI directory manifest.

The v2 headless production engine builds the generic MeasurementWorkflow
directly. Its lifecycle is modeling session, object CRUD, preview, publish,
measurement using a published asset, and package import/export. Operations
use a serial replaceable CPU subprocess with polling, timeout, and cancellation.
The HTTP image contract accepts JPEG through the configured shared-memory
pool. Result distances are explicitly in pixels and angles in degrees.
The v2 object factory does not create the GUI's parent-template multi-target
alignment workflow or localization-anchor TemplatePointObject instances.
GUI project assets and v2 published assets should not be treated as the same
model format.

Sources: [legacy SDK](../measure_api/project.py#L1),
[SDK GUI-workflow import](../measure_api/project.py#L43),
[v2 object factory](../measure_api/v2/worker.py#L71),
[v2 lifecycle](../measure_api/v2/application.py#L372),
[v2 result units](../measure_api/v2/worker.py#L429),
[v2 deployment limits](v2-production-engine.md).

Application-specific scripts add business acceptance checks:

- Bottle diameter: reads a selected point-distance output, checks each
  diameter against configured limits, and checks diameter variation across
  images taken at different rotations.
- Bottle verticality: finds paired edge endpoints at different heights,
  calculates their center-column positions, and checks the range of those
  positions against a configured threshold.

These scripts use pixel thresholds and implement specific inspection metrics.
The generic GUI's valid flag indicates computational validity rather than
product conformity. The examined GUI entry point loads images from files and
does not expose an integrated camera/PLC acquisition and rejection pipeline.

Sources: [diameter rules](../scripts/bottle_diameter_measurement.py#L80),
[diameter variation](../scripts/bottle_diameter_measurement.py#L152),
[verticality metric](../scripts/bottle_verticality_measurement.py#L85),
[inference recipe loading](../scripts/measure_inference.py#L29),
[GUI image execution](../measure_gui/app.py#L535).

## Verification and accuracy limitations

A focused existing regression selection passed:

```text
.venv/bin/python -m pytest tests/test_measure_workflow.py \
  tests/test_pixel_calibration.py -q \
  -k 'not VisualDemo and not RotationScaleTemplate and not MultiTargetTemplate'
78 passed, 26 deselected
```

This validates the selected synthetic workflow and calibration cases. It does
not establish real-camera dimensional accuracy, repeatability, full GUI
integration, rotation/multi-target coverage, or production throughput.

A separate read-only synthetic integration probe exposed a line-coordinate
convention mismatch. The low-level fit uses coefficients for
`a*col + b*row + c = 0`, while PointLineDistance consumes them as
`a*row + b*col + c = 0`. FitLineObject copies the coefficients unchanged.
For a horizontal fitted edge at row approximately 99.5 and a point at
(row=140, col=200), the expected perpendicular distance is approximately
40.5 px, but the current composition returns approximately 100.5 px.
The GUI uses the same row-first distance formula before mapping results.
This is a confirmed integration limitation, not just a hypothetical risk.

Sources: [low-level line coefficients](../measure/measure2D.py#L253),
[wrapper transfer](../measure/measure_workflow.py#L1321),
[point-line formula](../measure/measure_workflow.py#L1763),
[GUI point-line formula](../measure_gui/multi_target.py#L2483).

No production code was changed as part of this analysis.

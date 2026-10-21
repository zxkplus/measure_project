# Teaching Phase — Building a Measurement Model

## Overview

The Teaching phase creates a reusable measurement model. Start with a reference image (a "good" sample), define where and how to measure, tune each measurement tool interactively, then save. The model is later used in the Measurement phase against production images.

## Step-by-Step Workflow

### 1. Create Session

Every measurement model lives inside a session. The session tracks state and prevents invalid operations.

```
POST /api/session  {"project_dir": "/data/bottle_project"}
→ Get session_id "a1b2c3d4"
```

The `project_dir` is a path on the server filesystem where the project will be saved/loaded.

### 2. Load Reference Image

The reference image is a high-quality "golden" sample used to define the template and measurement tools.

```
POST /api/session/a1b2c3d4/reference  {"image_path": "/data/reference.jpg"}
```

The image must exist on the server filesystem. Supported formats: PNG, JPEG, BMP. Grayscale recommended.

### 3. Define Template ROI

The template is the region of interest used for locating the part in inspection images via NCC template matching.

Common parameters:
- **center, size**: Define a rotated rectangle on the reference image
- **angle_deg**: Typically 0 if the part is upright
- **angle_range_deg**: How much rotation variation to expect — this is the **half**-range (30 means ±30°)
- **max_matches**: For single-part inspection, set to 1 for multiple parts, set to 0 for unlimited
- **preprocessor**: `raw` (raw pixels) works for most cases; `canny` for edge-based matching; `sobel` gradient-based
- **angle_step** (default 1.0): fine-search step in degrees. Smaller = more precise orientation, slower.
- **coarse_fine** (default true) + **coarse_angle_step** (default 5.0): two-pass search — coarse sweep then refine. Usually leave both at their defaults.
- **overlap** (default 0.3): minimum IoU between accepted matches; only matters when `max_matches != 1`.

```
POST /api/session/a1b2c3d4/template
{
  "center": [1237.8, 993.6],
  "size": [1616.7, 1591.5],
  "angle_deg": 0,
  "preprocessor": "raw",
  "match_score_threshold": 0.5,
  "angle_range_deg": 30,
  "max_matches": 0,
  "angle_step": 1.0
}
```

Omitting the trailing fields is fine — they take the defaults above — but be deliberate
about `angle_step`: it is persisted with the model and directly trades speed against
orientation accuracy.

### 4. Add & Tune Measurement Tools

Each measurement tool defines a geometric feature to measure. Tools are tested immediately on the template image so you get instant feedback.

**The complete set of primitive measurement types** — no others exist:

| Type | What it measures | Key parameters |
|---|---|---|
| `FitCircle` | Circle fit from radial edge samples | center, radius, measure_length1, measure_length2, num_measures, sigma, threshold, transition |
| `FitLine` | Line fit from edge samples along a segment | start, end, measure_length1, measure_length2, num_measures, sigma, threshold, transition |
| `EdgePoint` | One edge crossing | row, col, angle, length1, length2, sigma, threshold, interpolation |
| `EdgePair` | Two opposing edges (a wall / a gap) | row, col, angle, length1, length2, sigma, threshold, interpolation |
| `TemplateMatchPoint` | Locate a small patch by NCC | row, col, template_size, preprocessor_type, match_score_threshold, angle_range_half, angle_step |

There is no `FitEdge`, `FindCircle`, `FitRectangle` or `FindLine`. A single edge is
`EdgePoint`; two opposing edges are `EdgePair`; circles are always `FitCircle`.
Exact required-vs-optional params and defaults are in
[api_reference.md](api_reference.md).

**Adding with immediate feedback:**
```python
import requests

resp = requests.post(
    f"http://localhost:5001/api/session/{sid}/measurements?include_visual=true",
    json={
        "object_type": "FitCircle",
        "label": "outer_diameter",
        "params": {
            "center": [814.59, 760.89],
            "radius": 484.15,
            "num_measures": 12,
            "sigma": 1.0,
            "threshold": 5.0,
            "transition": "negative",
            "start_phi": 0.0,
            "end_phi": 6.283185307179586,
        }
    }
)
result = resp.json()
print(f"Valid: {result['valid']}, Quality: {result['quality']}")
```

The response includes:
- `valid`: Whether the measurement found the feature
- `result`: The geometric values (center, radius for circles; a,b,c for lines)
- `quality`: Quality metrics (see below)
- `visual_b64`: If requested, an image overlay for visual inspection
- `elapsed_ms`: Time taken

**Key tuning parameters for FitCircle:**
- `num_measures`: Number of radial sampling lines (more = better fit, slower). Start with 12-24.
- `sigma`: Gaussian smoothing before edge detection. Higher = smoother but less precise. Range 0.5-5.0.
- `threshold`: Edge gradient threshold. Higher = fewer but stronger edges. Start with 5-20.
- `transition`: Edge polarity — `negative` (dark→bright), `positive` (bright→dark), `all` (both)
- `measure_length1/2`: Search window along each sampling line

### 5. Test Parameters Without Saving

Use the test endpoint to explore parameter variations without affecting the model:

```
POST /api/session/a1b2c3d4/measurements/test?include_visual=true
{
  "object_type": "FitCircle",
  "label": "test_circle",
  "params": {  // same params structure as add_measurement
    "center": [814.59, 760.89],
    "radius": 484.15,
    "num_measures": 24,   // try more sampling lines
    "sigma": 0.5,          // try less smoothing
    "threshold": 8.0,      // try higher threshold
    ...
  }
}
```

This is the primary workflow for parameter exploration during model building.

### 6. Update Existing Measurements

Use PUT to adjust parameters of an already-added tool:

```python
requests.put(
    f"http://localhost:5000/api/session/{sid}/measurements/outer_diameter",
    json={"params": {"sigma": 2.0, "threshold": 10.0}}
)
```

Partial merge — only specified params are updated; the rest stay as defined.

### 7. Add Composed (Derived) Measurements

Composed measurements derive values from other measurements. For example, the distance between two circle centers:

```python
requests.post(
    f"http://localhost:5000/api/session/{sid}/composed",
    json={
        "composed_type": "TwoPointsDistance",
        "label": "wall_thickness",
        "dependencies": {
            "point_a_label": "outer_diameter",
            "point_b_label": "inner_diameter"
        }
     }
)
```

All composed types (5 total):

| Composed Type | Description | Required Dependencies |
|---|---|---|
| `TwoPointsDistance` | Distance between two point results | point_a_label, point_b_label |
| `TwoPointsLine` | Line through two point results | point_a_label, point_b_label |
| `TwoLinesAngle` | Angle between two lines | line_a_label, line_b_label |
| `PointLineDistance` | Distance from a point to a line | point_label, line_label |
| `PointCircleDistance` | Distance from a point to a circle center | point_label, circle_label |

The name is `PointLineDistance`, not `PointToLineDistance`.

Slots are type-checked when you add the composed tool, so a wrong reference fails with 400
instead of surfacing much later at run time. A `point` slot accepts a fitted circle as well
as a point (`CircleResult.row/col` alias its centre); `line_label` and `circle_label` accept
only their own kind. An `EdgePair` is addressable **only** through its virtual endpoints
`<label>_A` / `<label>_B` — its bare label publishes no result and is rejected.

Full parameter tables and defaults: [generated/object_contracts.md](generated/object_contracts.md).

### 8. Validate DAG

The DAG endpoint shows dependency relationships and validates that all references are resolvable:

```
GET /api/session/a1b2c3d4/dag
```

Check `is_valid: true` and inspect `execution_order` — primitives execute first, then composed in dependency order.

### 9. Save Model

```python
requests.post(f"http://localhost:5000/api/session/{sid}/save")
```

Saves `config.json` (measurement + composed definitions), `template.npz` (template pixels)
and `reference.png` to the project directory. The model is ready for the Measurement phase.

> This v1 layout is **not** interchangeable with the GUI / v2 model asset (`manifest.json`
> + `reference.npy` + `workflow.npz`). A v1 directory cannot be imported by v2.

## Quality Metrics Reference

| Metric | Range | Meaning |
|---|---|---|
| `num_edges` | 0 to N | Number of edge points actually detected (`meta.num_points`) |
| `expected_edges` | N | Intended number of sampling lines — **usually `0` in practice**, because the fitters do not record `num_measures` in `meta` |
| `coverage_ratio` | 0.0-1.0 | num_edges / expected_edges; forced to `0.0` when `expected_edges == 0`, so a zero alone does not mean failure |
| `edge_amplitude_mean` | 0-255 | Average edge gradient strength (conditional) |
| `edge_amplitude_min` | 0-255 | Minimum edge gradient strength (conditional) |

**Use `result.meta.mean_error` / `meta.max_error` for fit quality.** They are populated for
`FitCircle` and `FitLine` and are in pixels. The `rms` key is documented by the quality
module but the engine never emits `meta.rms`, so it is absent in real responses.

**All keys are conditional.** `quality` is `{}` when the measurement failed, and composed
types never populate it. Always use `.get()` and cross-check against `valid`.

**Rule of thumb:** `meta.num_points` close to your `num_measures`, and `meta.mean_error < 2`
pixels, is excellent for circles/lines.

# Measurement Phase — Batch Inspection

## Overview

The Measurement phase uses a saved model to run inspection on production images. Results include detected targets, their measurement values, and quality metrics. This is typically called by a production pipeline for each incoming image.

## Step-by-Step Workflow

### 1. Create Session & Load Model

Load a previously saved project by creating a session and loading:

```python
import requests

BASE = "http://localhost:5000"

# Create session with the project directory
resp = requests.post(f"{BASE}/api/session", json={
    "project_dir": "/data/bottle_project"
})
sid = resp.json()["session_id"]

# Load saved project
requests.post(f"{BASE}/api/session/{sid}/load")
```

The `project_dir` must match the directory used during save. The load restores template, measurements, and composed definitions.

### 2. Verify Session State

```python
resp = requests.get(f"{BASE}/api/session/{sid}")
state = resp.json()
assert state["phase"] in ("has_measurements", "has_composed"), "Model not ready"
assert state["dag_valid"], "DAG has errors"
```

### 3. Run Measurement on an Inspection Image

```python
resp = requests.post(
    f"{BASE}/api/session/{sid}/measure?include_visual=true",
    json={"inspection_image": "/data/inspections/part_001.jpg"}
)
result = resp.json()
```

### 4. Interpret Results

The response contains one entry per detected target (controlled by `max_matches` during template definition):

```json
{
  "status": "ok",
  "elapsed_ms": 2727.3,
  "num_targets": 1,
  "targets": [
    {
      "target_id": 0,
      "score": 0.6522,
      "row": 1234.5,
      "col": 987.6,
      "rotation_deg": -5.1,
      "scale": 1.0,
      "measurements": {
        "outer_diameter": {
          "valid": true,
          "center_row": 816.374,
          "center_col": 783.995,
          "radius": 478.641
        },
        "inner_diameter": {
          "valid": true,
          "center_row": 831.08,
          "center_col": 807.04,
          "radius": 611.24
        },
        "wall_thickness": {
          "valid": true,
          "value": 134.28
        }
      }
    }
  ]
}
```

**Key fields:**
- `target.score`: NCC template match score (0.0-1.0). Higher = better match. 0.5+ is typical.
- `target.row`, `target.col`: Location of the detected part in the original image.
- `target.rotation_deg`: Part rotation relative to template.
- `target.measurements`: Object keyed by measurement label. Each contains:
  - `valid`: Whether the measurement was successful on this target
  - For circles: `center_row`, `center_col`, `radius`
  - For lines: `a`, `b`, `c`, `start_row/col`, `end_row/col`
  - For distances/angles: `value`, `value_deg`
- `visual_b64`: Base64-encoded PNG overlay showing detected features (if requested)

### 5. Batch Processing Pattern

```python
import base64
import os

for img_file in sorted(os.listdir("/data/inspections")):
    img_path = os.path.join("/data/inspections", img_file)

    resp = requests.post(
        f"{BASE}/api/session/{sid}/measure?include_visual=true",
        json={"inspection_image": img_path}
    )
    result = resp.json()

    if result["status"] != "ok":
        print(f"FAIL: {img_file} — {result.get('error')}")
        continue

    # Save visual overlay
    if result.get("visual_b64"):
        png = base64.b64decode(result["visual_b64"])
        with open(f"results/{img_file}_overlay.png", "wb") as f:
            f.write(png)

    # Extract measurement values
    for target in result["targets"]:
        for label, m in target["measurements"].items():
            if not m.get("valid"):
                print(f"  {img_file}: {label} measurement FAILED")
                continue
            # Circle measurements
            if "radius" in m:
                print(f"  {label}: center=({m['center_row']:.1f}, {m['center_col']:.1f}) r={m['radius']:.2f}")
            # Distance/angle measurements
            if "value" in m:
                unit = "deg" if m.get("value_deg") else "pixels"
                print(f"  {label}: {m['value']:.2f} {unit}")
```

### 6. Clean Up

```python
requests.delete(f"{BASE}/api/session/{sid}")
```

Sessions hold server resources. Delete when done, especially for long-running batch processes.

## Inspection Image Requirements

- Supported formats: JPEG, PNG, BMP (grayscale recommended)
- Images must exist on the server filesystem (the server reads them directly)
- Resolution: ideally similar to the reference image (the template is built at the reference resolution)

## Quality Monitoring

For production use, monitor these indicators:

- **Match score** (`target.score`): Set a minimum threshold. Drop below 0.4-0.5 may indicate parts outside spec.
- **Measurement validity** (`m.valid`): Failed measurements often indicate defects or part misalignment.
- **Scale** (`target.scale`): Should be close to 1.0. Significant deviation means the part is at a different distance from the camera.

## Multi-Target Mode

If `max_matches` is set to 0 (unlimited) or > 1 during template definition, multiple targets may be returned:

```json
{"num_targets": 3, "targets": [...]}
```

Process each target independently. Useful for multi-cavity molds or batch inspection with multiple parts per image.

## Error Recovery

| Error | Likely Cause | Fix |
|---|---|---|
| `FileNotFoundError` | Image path doesn't exist | Check path on server filesystem |
| `RuntimeError: Operation not allowed in phase...` | State machine violation | Check session phase — call the right endpoint in order |
| Target `score` too low | Part differs from template | Reconsider angle_range_deg or template definition |
| Measurement `valid: false` | Feature not found at expected location | Increase measure_length or adjust threshold |

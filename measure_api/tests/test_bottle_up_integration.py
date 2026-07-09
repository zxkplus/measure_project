"""
Integration test using the bottle_up project data.

Sets up a MeasureProject matching the ``bottle_up`` configuration
(reference image, ROI, two FitCircle measurements), runs the full
measurement pipeline on every inspection image in the ``内外直径/``
directory, saves result overlays and a JSON summary to
``/media/industai/data11/data/results/``.

This test is marked ``integration`` and is not included in the default
test suite (pytest -m integration ...).
"""

from __future__ import annotations

import base64
import json
import os
import tempfile

import pytest

from measure_api.project import MeasureProject

# ---------------------------------------------------------------------------
# Paths (edit these if data is relocated)
# ---------------------------------------------------------------------------

DATA_DIR = "/media/industai/data11/data"
INSP_DIR = os.path.join(DATA_DIR, "内外直径")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "output", "bottle_up_results")

REFERENCE_FILENAME = "20260701-143514.jpg"
SKIP_PREFIX = "20260701-143514"


def _data_available() -> bool:
    ref = os.path.join(INSP_DIR, REFERENCE_FILENAME)
    if not os.path.isfile(ref):
        return False
    for f in os.listdir(INSP_DIR):
        if f.endswith(".jpg") and SKIP_PREFIX not in f:
            return True
    return False


pytestmark = pytest.mark.skipif(
    not _data_available(),
    reason=f"Bottle-up project data not found in {INSP_DIR}",
)


# ===========================================================================
# Test
# ===========================================================================


@pytest.mark.integration
def test_bottle_up_measurement_pipeline():
    """End-to-end integration test on real bottle-up production images."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Create project & load reference
    # ------------------------------------------------------------------
    project_dir = tempfile.mkdtemp(prefix="bottle_up_integration_")
    proj = MeasureProject(project_dir)

    ref_path = os.path.join(INSP_DIR, REFERENCE_FILENAME)
    ref_info = proj.load_reference(ref_path)
    assert ref_info["width"] > 0
    assert ref_info["height"] > 0

    # ------------------------------------------------------------------
    # 2. Set template (ROI from bottle_up project.json)
    # ------------------------------------------------------------------
    proj.set_template(
        center=(1237.8, 993.6),
        size=(1616.7, 1591.5),
        angle_deg=0.0,
        preprocessor="raw",
        match_score_threshold=0.5,
        angle_range_deg=30,
        max_matches=0,
    )
    assert proj.has_template

    # ------------------------------------------------------------------
    # 3. Add the two FitCircle measurements
    # ------------------------------------------------------------------
    r1 = proj.add_measurement("FitCircle", "circle_1", {
        "center": (814.59, 760.89),
        "radius": 484.15,
        "measure_length1": 60.0,
        "measure_length2": 10.0,
        "num_measures": 12,
        "sigma": 1.0,
        "threshold": 5.0,
        "transition": "negative",
        "start_phi": 0.0,
        "end_phi": 6.283185307179586,
    }, include_visual=True)
    assert "valid" in r1
    assert "quality" in r1

    r2 = proj.add_measurement("FitCircle", "circle_2", {
        "center": (831.08, 807.04),
        "radius": 611.24,
        "measure_length1": 120.0,
        "measure_length2": 10.0,
        "num_measures": 12,
        "sigma": 1.0,
        "threshold": 5.0,
        "transition": "all",
        "start_phi": 0.0,
        "end_phi": 6.283185307179586,
    }, include_visual=True)
    assert "valid" in r2

    # ------------------------------------------------------------------
    # 4. Collect inspection images
    # ------------------------------------------------------------------
    insp_images = sorted([
        f for f in os.listdir(INSP_DIR)
        if f.endswith(".jpg") and SKIP_PREFIX not in f
    ])
    assert len(insp_images) > 0, f"No inspection images in {INSP_DIR}"

    # ------------------------------------------------------------------
    # 5. Run measurement on each image
    # ------------------------------------------------------------------
    summary: list[dict] = []

    for img_file in insp_images:
        img_path = os.path.join(INSP_DIR, img_file)
        result = proj.measure(img_path, include_visual=True)

        # Save annotated PNG
        if result.get("visual_b64"):
            png_bytes = base64.b64decode(result["visual_b64"])
            out = os.path.join(RESULTS_DIR, f"result_{img_file}".replace(".jpg", ".png"))
            with open(out, "wb") as f:
                f.write(png_bytes)

        # Collect structured data
        entry: dict = {
            "image": img_file,
            "status": result.get("status"),
            "elapsed_ms": result.get("elapsed_ms"),
            "num_targets": result.get("num_targets", 0),
        }

        if result.get("targets"):
            t0 = result["targets"][0]
            # Target metadata
            entry["target"] = {
                "score": t0.get("score", 0),
                "row": round(t0.get("row", 0), 2),
                "col": round(t0.get("col", 0), 2),
                "rotation_deg": round(t0.get("rotation_deg", 0), 2),
                "scale": round(t0.get("scale", 1.0), 4),
            }
            # Per-measurement values
            measurements = {}
            for label, m in t0.get("measurements", {}).items():
                flat = {}
                for key in ("valid", "type", "center_row", "center_col",
                            "radius", "value", "value_deg"):
                    if key in m:
                        val = m[key]
                        flat[key] = round(val, 3) if isinstance(val, float) else val
                measurements[label] = flat
            entry["measurements"] = measurements

        summary.append(entry)

    # ------------------------------------------------------------------
    # 6. Save summary JSON
    # ------------------------------------------------------------------
    summary_path = os.path.join(RESULTS_DIR, "measurement_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 7. Verify at least some targets were detected
    # ------------------------------------------------------------------
    total_targets = sum(s["num_targets"] for s in summary)
    assert total_targets > 0, (
        f"No targets detected in any of {len(insp_images)} inspection images."
    )

    _print_summary(proj, summary)


# ===========================================================================
# Helper
# ===========================================================================


def _print_summary(proj: MeasureProject, summary: list[dict]) -> None:
    n_total = len(summary)
    n_with = sum(1 for s in summary if s["num_targets"] > 0)
    total_targets = sum(s["num_targets"] for s in summary)

    print(f"\n{'='*60}")
    print(f"  bottle_up Integration Test Results")
    print(f"{'='*60}")
    print(f"  Template shape       : {proj.template_shape}")
    print(f"  Inspection images     : {n_total}")
    print(f"  Images with targets   : {n_with}")
    print(f"  Total targets found   : {total_targets}")
    print(f"  Results directory     : {RESULTS_DIR}")
    print()

    for s in summary:
        img = s["image"]
        ms = s["elapsed_ms"]
        nt = s["num_targets"]
        if nt > 0 and "target" in s:
            t = s["target"]
            print(f"  {img:30s}  {ms:>8.1f}ms  {nt} target(s)  "
                  f"score={t['score']:.4f}  rot={t['rotation_deg']:.1f}deg")
        else:
            print(f"  {img:30s}  {ms:>8.1f}ms  {nt} target(s)  NO DETECTIONS")

    print(f"\n  Results saved to {RESULTS_DIR}")
    print(f"{'='*60}\n")

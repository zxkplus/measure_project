"""
REST API integration test — walks through the full bottle_up pipeline
via the Flask HTTP endpoints, mirroring the SDK-based test.

Requirements:
  - Real bottle_up project data under ``/media/industai/data11/data/``
  - Flask test client (no need to start a real server)

Usage:
  pytest measure_api/tests/test_bottle_up_api_integration.py -v -m integration -s
"""

from __future__ import annotations

import base64
import json
import os
import tempfile

import pytest

from measure_api.server import create_app
from measure_api.config import Config

# ---------------------------------------------------------------------------
# Paths (edit these if data is relocated)
# ---------------------------------------------------------------------------

DATA_DIR = "/media/industai/data11/data"
INSP_DIR = os.path.join(DATA_DIR, "内外直径")
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "output", "bottle_up_api_results",
)

REFERENCE_FILENAME = "20260701-143514.jpg"
SKIP_PREFIX = "20260701-143514"
MAX_MATCHES = 0  # unlimited


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
# Fixtures
# ===========================================================================


@pytest.fixture(scope="module")
def api_client():
    """Create a Flask test client with real-data configuration."""
    Config.reset()
    cfg = Config.load_from_dict({
        "log": {
            "directory": "/tmp/measure_api_api_test_logs",
            "level": "INFO",
            "console_output": False,
            "backup_days": 1,
        },
        "server": {
            "host": "127.0.0.1",
            "port": 0,
            "max_sessions": 10,
        },
        "call_records": {
            "enabled": False,
        },
    })
    app = create_app(cfg)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
    Config.reset()


@pytest.fixture(scope="module")
def project_dir():
    """Temporary directory for the project."""
    d = tempfile.mkdtemp(prefix="bottle_up_api_")
    yield d


# ===========================================================================
# Helper wrappers
# ===========================================================================


def _post(client, url, data: dict, query: str = ""):
    """POST a JSON payload, return the parsed response body."""
    full_url = f"{url}{query}"
    resp = client.post(
        full_url,
        data=json.dumps(data),
        content_type="application/json",
    )
    body = resp.get_json()
    assert resp.status_code == 200, (
        f"POST {full_url} returned {resp.status_code}: {body}"
    )
    return body, resp.headers.get("X-Trace-Id", "")


def _get(client, url, expect=200):
    """GET a JSON resource, return the parsed response body."""
    resp = client.get(url)
    body = resp.get_json()
    assert resp.status_code == expect, (
        f"GET {url} returned {resp.status_code}: {body}"
    )
    return body


def _delete(client, url, expect=200):
    """DELETE a resource, return the parsed response body."""
    resp = client.delete(url)
    body = resp.get_json()
    assert resp.status_code == expect, (
        f"DELETE {url} returned {resp.status_code}: {body}"
    )
    return body


# ===========================================================================
# Test
# ===========================================================================


@pytest.mark.integration
def test_bottle_up_measurement_pipeline_via_api(api_client, project_dir):
    """
    End-to-end integration test using the REST API endpoints on real
    bottle-up production images.

    Validates:
      - Session lifecycle
      - Reference loading
      - Template definition
      - Measurement CRUD with test feedback
      - Full measurement pipeline
      - Visual overlay generation
      - Session state reporting
      - Session deletion
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ref_path = os.path.join(INSP_DIR, REFERENCE_FILENAME)

    # ── 1. Create session ────────────────────────────────────────────────
    body, trace_id = _post(api_client, "/api/session", {
        "project_dir": project_dir,
    })
    assert body["status"] == "created"
    sid = body["session_id"]
    assert sid, f"No session_id in response: {body}"
    assert trace_id, "Missing X-Trace-Id header"
    print(f"\n  Session created: {sid}  trace={trace_id}")

    # ── 2. Get session status (initial) ─────────────────────────────────
    status = _get(api_client, f"/api/session/{sid}")
    assert status["phase"] == "created"
    assert status["project_dir"] == os.path.abspath(project_dir)

    # ── 3. Load reference image ──────────────────────────────────────────
    body, _ = _post(api_client, f"/api/session/{sid}/reference", {
        "image_path": ref_path,
    })
    assert body["width"] > 0
    assert body["height"] > 0
    print(f"  Reference loaded: {body['width']}x{body['height']}")

    # ── 4. Set template ──────────────────────────────────────────────────
    body, _ = _post(api_client, f"/api/session/{sid}/template", {
        "center": [1237.8, 993.6],
        "size": [1616.7, 1591.5],
        "angle_deg": 0.0,
        "preprocessor": "raw",
        "match_score_threshold": 0.5,
        "angle_range_deg": 30,
        "max_matches": MAX_MATCHES,
    })
    assert "template_shape" in body
    print(f"  Template set: {body['template_shape']}")

    # ── 5. Add circle_1 measurement ─────────────────────────────────────
    circle_1_params = {
        "center": [814.59, 760.89],
        "radius": 484.15,
        "measure_length1": 60.0,
        "measure_length2": 10.0,
        "num_measures": 12,
        "sigma": 1.0,
        "threshold": 5.0,
        "transition": "negative",
        "start_phi": 0.0,
        "end_phi": 6.283185307179586,
    }
    body, _ = _post(api_client, f"/api/session/{sid}/measurements?include_visual=true", {
        "object_type": "FitCircle",
        "label": "circle_1",
        "params": circle_1_params,
    })
    assert body["valid"], f"circle_1 not valid: {body}"
    assert "quality" in body
    assert "visual_b64" in body
    print(f"  circle_1 added: valid={body['valid']}  quality={body['quality']}")

    # ── 6. Add circle_2 measurement ─────────────────────────────────────
    circle_2_params = {
        "center": [831.08, 807.04],
        "radius": 611.24,
        "measure_length1": 120.0,
        "measure_length2": 10.0,
        "num_measures": 12,
        "sigma": 1.0,
        "threshold": 5.0,
        "transition": "all",
        "start_phi": 0.0,
        "end_phi": 6.283185307179586,
    }
    body, _ = _post(api_client, f"/api/session/{sid}/measurements?include_visual=true", {
        "object_type": "FitCircle",
        "label": "circle_2",
        "params": circle_2_params,
    })
    assert body["valid"], f"circle_2 not valid: {body}"
    print(f"  circle_2 added: valid={body['valid']}  quality={body['quality']}")

    # ── 7. List measurements ─────────────────────────────────────────────
    body = _get(api_client, f"/api/session/{sid}/measurements")
    assert len(body["measurements"]) == 2
    labels = {m["label"] for m in body["measurements"]}
    assert labels == {"circle_1", "circle_2"}, f"Unexpected labels: {labels}"
    print(f"  Listed {len(body['measurements'])} measurements, "
          f"{len(body['composed'])} composed")

    # ── 8. Get individual measurement ────────────────────────────────────
    body = _get(api_client, f"/api/session/{sid}/measurements/circle_1")
    assert body["label"] == "circle_1"
    assert body["object_type"] == "FitCircle"

    # ── 9. Update measurement params & re-test ───────────────────────────
    body, _ = _post(api_client, f"/api/session/{sid}/measurements/test?include_visual=true", {
        "object_type": "FitCircle",
        "label": "circle_1_test",
        "params": {**circle_1_params, "sigma": 2.0},
    })
    assert "valid" in body
    print(f"  test_measurement (sigma=2.0): valid={body['valid']}")

    # ── 10. GET session status (teaching phase) ──────────────────────────
    status = _get(api_client, f"/api/session/{sid}")
    assert status["phase"] == "has_measurements"
    assert status["num_measurements"] == 2

    # ── 11. DAG ──────────────────────────────────────────────────────────
    dag = _get(api_client, f"/api/session/{sid}/dag")
    assert dag["is_valid"]
    assert len(dag["nodes"]) == 2
    assert len(dag["execution_order"]) == 2
    print(f"  DAG: {len(dag['nodes'])} nodes, valid={dag['is_valid']}")

    # ── 12. Save project ─────────────────────────────────────────────────
    body, _ = _post(api_client, f"/api/session/{sid}/save", {})
    assert "saved_to" in body
    assert os.path.isdir(body["saved_to"])
    print(f"  Project saved to: {body['saved_to']}")

    # ── 13. Collect inspection images ────────────────────────────────────
    insp_images = sorted([
        f for f in os.listdir(INSP_DIR)
        if f.endswith(".jpg") and SKIP_PREFIX not in f
    ])
    assert len(insp_images) > 0, f"No inspection images in {INSP_DIR}"
    print(f"  Inspection images: {len(insp_images)}")

    # ── 14. Run measurement on each image ────────────────────────────────
    summary: list[dict] = []

    for img_file in insp_images:
        img_path = os.path.join(INSP_DIR, img_file)

        body, _ = _post(
            api_client,
            f"/api/session/{sid}/measure?include_visual=true",
            {"inspection_image": img_path},
        )
        assert body["status"] == "ok", f"measure failed for {img_file}: {body}"

        # Save annotated PNG
        if body.get("visual_b64"):
            png_bytes = base64.b64decode(body["visual_b64"])
            out = os.path.join(RESULTS_DIR, f"result_{img_file}".replace(".jpg", ".png"))
            with open(out, "wb") as f:
                f.write(png_bytes)

        # Collect structured data
        entry: dict = {
            "image": img_file,
            "status": body.get("status"),
            "elapsed_ms": body.get("elapsed_ms"),
            "num_targets": body.get("num_targets", 0),
        }

        if body.get("targets"):
            t0 = body["targets"][0]
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

    # ── 15. Save summary JSON ────────────────────────────────────────────
    summary_path = os.path.join(RESULTS_DIR, "api_measurement_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary saved to: {summary_path}")

    # ── 16. Verify at least some targets were detected ───────────────────
    total_targets = sum(s["num_targets"] for s in summary)
    assert total_targets > 0, (
        f"No targets detected in any of {len(insp_images)} inspection images."
    )

    # ── 17. Delete session ───────────────────────────────────────────────
    body = _delete(api_client, f"/api/session/{sid}")
    assert body["status"] == "deleted"

    # ── 18. Verify session gone ──────────────────────────────────────────
    _get(api_client, f"/api/session/{sid}", expect=404)

    # ── Print summary ────────────────────────────────────────────────────
    _print_api_summary(summary)

    print(f"  API integration test PASSED — see {RESULTS_DIR}")
    print(f"{'='*60}\n")


# ===========================================================================
# Helper
# ===========================================================================


def _print_api_summary(summary: list[dict]) -> None:
    """Print a formatted summary of the API integration test results."""
    n_total = len(summary)
    n_with = sum(1 for s in summary if s["num_targets"] > 0)
    total_targets = sum(s["num_targets"] for s in summary)

    print(f"\n{'='*60}")
    print(f"  bottle_up API Integration Test Results")
    print(f"{'='*60}")
    print(f"  Method               : HTTP (Flask test client)")
    print(f"  Inspection images     : {n_total}")
    print(f"  Images with targets   : {n_with}")
    print(f"  Total targets found   : {total_targets}")
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

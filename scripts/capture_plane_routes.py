"""Capture plane/stereo route answers so a refactor can be compared field by field.

These routes are not exercised by the exported production data, so their
regression evidence has to be a recorded baseline: this script posts a fixed set
of synthetic requests through the in-process app and dumps the normalised
results.

    python scripts/capture_plane_routes.py --out tests/output/plane/before.json
    python scripts/capture_plane_routes.py --out tests/output/plane/after.json
    python scripts/replay_v2_debug_package.py --compare <before> <after>

The plane/stereo routes are not exercised by the exported production data, so
this capture is their regression evidence. Stitched and coverage canvases are
recorded as sha256 digests of the committed PNG bytes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hashlib

import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from measure_api.v2.app import create_app          # noqa: E402
from measure_api.v2.models import ServiceSettings  # noqa: E402
from tests.v2 import test_plane_stereo as fixtures  # noqa: E402
from scripts.replay_v2_debug_package import _prune  # noqa: E402


def _settings(tmp: Path) -> ServiceSettings:
    roots = {}
    for alias in ("exchange", "assets", "outputs"):
        (tmp / alias).mkdir(parents=True, exist_ok=True)
        roots[alias] = tmp / alias
    return ServiceSettings(shared_roots=roots, writable_roots=["outputs"],
                           state_directory=tmp / "state", max_workers=2)


def capture(tmp: Path) -> dict:
    """Post the fixed requests and return their normalised results."""
    out = {}
    (tmp / "exchange").mkdir(parents=True, exist_ok=True)
    for camera_id, value in (("A", 20), ("B", 80)):
        cv2.imwrite(str(tmp / "exchange" / f"{camera_id}.png"),
                    np.full((80, 100), value, np.uint8))
    with TestClient(create_app(_settings(tmp))) as client:
        def post(route, payload):
            response = client.post("/api/v2" + route, json=payload)
            body = response.json()
            if response.status_code != 200:
                return {"status": response.status_code, "error": body}
            return body["result"]

        out["plane"] = post("/calibrations/plane-mappings", fixtures.plane_request())
        out["stereo"] = post("/calibrations/stereo-rigs", fixtures.stereo_request())

        plane_asset = {"value": out["plane"]["asset"]}
        out["plane_validation"] = post("/calibration-validations", {
            "calibration": plane_asset, "unit": "mm",
            "observations": [{**fixtures.observation(), "camera_id": "A"}],
            "max_error_px": 1,
        })
        out["origin"] = post("/calibration-origin-adjustments", {
            "calibration": plane_asset, "origin": {"x": 5, "y": 3}, "unit": "mm",
            "output": {"directory": {"root": "outputs", "path": "shifted"}},
        })
        stereo_asset = {"value": out["stereo"]["asset"]}
        out["stereo_forward"] = post("/geometry/stereo-transforms", {
            "calibration": stereo_asset, "source_camera": "A", "destination_camera": "B",
            "points": [{"x": 0, "y": 0, "z": 0.1}], "unit": "m", "target_unit": "mm",
        })
        out["stereo_reverse"] = post("/geometry/stereo-transforms", {
            "calibration": stereo_asset, "source_camera": "B", "destination_camera": "A",
            "points": [{"x": 10, "y": 0, "z": 100}], "unit": "mm", "target_unit": "cm",
        })
        images = [{"camera_id": cam, "applicability": fixtures.app(cam),
                   "capture_id": "product-1",
                   "image": {"root": "exchange", "path": f"{cam}.png"}}
                  for cam in ("A", "B")]
        for blend in ("average", "first", "last"):
            stitched = post("/image-stitches", {
                "calibration": plane_asset, "images": images, "blend": blend,
                "output": {"directory": {"root": "outputs", "path": f"stitch-{blend}"}},
            })
            files = {entry["kind"]: entry["file"] for entry in stitched.get("artifacts", [])
                     if isinstance(entry, dict)}
            out[f"stitch_{blend}_pixels"] = {
                kind: hashlib.sha256((tmp / "outputs" / files[kind]["path"]).read_bytes()
                                     ).hexdigest()
                for kind in ("stitched", "coverage") if kind in files
            }
            stitched.pop("artifacts", None)
            stitched.pop("manifest", None)
            out[f"stitch_{blend}"] = stitched
        out["camera_projection"] = post("/geometry/camera-projections", {
            "calibration": {"value": fixtures.camera()}, "applicability": fixtures.app(),
            "points": [{"x": 1, "y": 2, "z": 10}], "unit": "cm", "source_frame": "camera:A",
        })
    return _prune(out)


def main() -> int:
    """Parse arguments and write the captured baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import tempfile
    with tempfile.TemporaryDirectory(prefix="plane-capture-") as scratch:
        captured = capture(Path(scratch))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(captured, ensure_ascii=False, indent=1, sort_keys=True),
                        encoding="utf-8")
    print(f"wrote {args.out} ({len(captured)} route(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
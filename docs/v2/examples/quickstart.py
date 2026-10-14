"""Exercise the upstream HTTP/shared-file workflow with isolated synthetic demo inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict
import uuid

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from v2_http_client import V2Client


def main() -> None:
    """Run an HTTP teaching-to-replay example against an already started V2 service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9004")
    parser.add_argument("--shared-dir", type=Path, required=True)
    args = parser.parse_args()
    shared = args.shared_dir.resolve()
    roots = {name: shared / name for name in ("exchange", "assets", "outputs")}
    if not all(path.is_dir() for path in roots.values()):
        raise ValueError("Create exchange, assets and outputs before starting the service")
    namespace = "api-demo-" + uuid.uuid4().hex
    for root in roots.values():
        (root / namespace).mkdir(exist_ok=False)
    records = shared / "client-records" / namespace
    records.mkdir(parents=True, exist_ok=False)
    client = V2Client(args.base_url, caller_id="upstream-api-demo")
    if client.request("/health/ready")["status"] != "ready":
        raise RuntimeError("V2 is not ready")
    counter = 0

    def publish(root: str, name: str, content: bytes) -> Dict[str, str]:
        path = roots[root] / namespace / name
        temporary = path.with_name(path.name + ".partial")
        temporary.write_bytes(content)
        os.replace(temporary, path)
        return {"root": root, "path": f"{namespace}/{name}",
                "sha256": hashlib.sha256(content).hexdigest()}

    def publish_json(root: str, name: str, value: Dict[str, Any]) -> Dict[str, str]:
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        return publish(root, name, data)

    def calculate(route: str, request: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal counter
        counter += 1
        payload = {"timeout_ms": 60000, "idempotency_key": f"{namespace}-{counter}", **request}
        operation = client.calculate(route, payload)
        (records / f"{counter:02d}-request.json").write_text(json.dumps(payload, indent=2))
        (records / f"{counter:02d}-response.json").write_text(json.dumps(operation, indent=2))
        if operation["state"] != "succeeded":
            raise RuntimeError(json.dumps(operation, allow_nan=False))
        print(f"{route}: {operation['state']}")
        return operation["result"]

    # Fixture pixels are solely for contract integration, with no physical accuracy claim.
    taught = np.zeros((80, 80), np.uint8)
    taught[20:60, 30:50] = 255
    taught[22:29, 15:23] = 120
    taught[52:59, 55:63] = 190
    inspection = np.zeros((110, 220), np.uint8)
    inspection[10:90, 10:90] = taught
    inspection[5:101, 115:211] = cv2.resize(taught, (96, 96))
    reference = publish("exchange", "reference.png", cv2.imencode(".png", taught)[1].tobytes())
    image = publish("exchange", "production.png", cv2.imencode(".png", inspection)[1].tobytes())
    output = {"directory": {"root": "outputs", "path": namespace}}
    applicability = {"camera_id": "upstream-camera", "setup_id": "fixture-A", "frame": "source",
                     "image_size": {"width": 220, "height": 110}}
    calculate("/image-inspections", {"image": reference})
    teaching = calculate("/template-builds", {
        "image": reference,
        "roi": {"center": {"x": 40, "y": 40}, "width": 65, "height": 65},
        "output": output,
    })
    source = teaching["template"]
    template_bytes = (roots[source["root"]] / source["path"]).read_bytes()
    template = publish("assets", "template-v1.npz", template_bytes)
    calibration = calculate("/calibrations/scales", {
        "corners": [{"x": x, "y": y} for y in (5, 15) for x in (5, 15, 25)],
        "square_size": 1, "unit": "mm", "applicability": applicability, "output": output,
    })
    if not calibration["valid"]:
        raise RuntimeError("Synthetic calibration is invalid")
    calibration_file = publish_json("assets", "calibration-v1.json", calibration["asset"])
    recipe = {
        "id": "upstream-part", "version": "unsaved-edit", "reference": reference,
        "localization": {"template": template, "search": {
            "score_threshold": 0.85, "max_matches": 2,
            "scale_min": 1, "scale_max": 1.2, "scale_step": 0.2,
        }},
        "metric": {"calibration": {"file": calibration_file}, "applicability": applicability},
        "nodes": [
            {"id": "width", "operation": "measurements.edge-pairs", "parameters": {
                "probe": {"center": {"x": 40, "y": 40},
                          "search_half_length": 20, "average_half_width": 3},
                "selection": "first", "polarity": "positive",
            }},
            {"id": "check", "operation": "evaluations.ranges", "space": "physical",
             "parameters": {"minimum": 1.9, "maximum": 2.2, "unit": "mm"},
             "inputs": {"value": {"node": "width", "port": "width", "space": "physical"}}},
        ],
        "required_outputs": [{"node": "check", "port": "value", "space": "physical"}],
        "target_policy": {"minimum": 2, "maximum": 2, "no_targets": "fail"},
    }
    validated = calculate("/recipe-validations", {"recipe": recipe})
    if not validated["valid"]:
        raise RuntimeError(json.dumps(validated))
    preview = calculate("/recipe-previews", {
        "recipe": recipe, "image": image, "selected_nodes": ["check"],
        "upstream_revision": "editor-8", "preview_token": "preview-8",
    })
    if preview["preview_token"] != "preview-8" or preview["upstream_revision"] != "editor-8":
        raise RuntimeError("Preview identity mismatch")
    recipe["version"] = "upstream-v8"
    recipe_file = publish_json("assets", "recipe-v8.json", recipe)
    run = calculate("/runs", {"mode": "async", "recipe_file": recipe_file,
                              "image": image, "output": output})
    if not run["valid"] or len(run["targets"]) != 2 or run["evaluation_status"] != "fail":
        raise RuntimeError("Unexpected synthetic run outcome")
    widths = sorted(target["nodes"][0]["physical_result"]["width"]["value"]
                    for target in run["targets"])
    if not np.allclose(widths, [2.0, 2.4], atol=0.02, rtol=0):
        raise RuntimeError(f"Unexpected widths: {widths}")
    result_file = publish_json("assets", "run-result.json", run)
    calculate("/visualizations", {
        "result_file": result_file, "space": "physical", "display_frame": "source",
        "raster": True, "diagnostics": True, "selected_nodes": ["width"], "output": output,
    })
    calculate("/result-exports", {
        "result_file": result_file, "formats": ["csv", "json"], "output": output,
    })
    package_result = calculate("/recipe-exports", {
        "recipe": run["resolved_recipe"], "frozen_run": run,
        "packaging": "embedded", "output": output,
    })
    package = next(item["file"] for item in package_result["artifacts"]
                   if item["kind"] == "recipe_package")
    frozen = next(item["file"] for item in package_result["artifacts"]
                  if item["kind"] == "frozen_run")
    imported = calculate("/recipe-imports", {"package": package, "output": output})
    relocated = calculate("/runs", {
        "recipe": imported["resolved_recipe"], "image": image, "output": output,
    })
    if not imported["valid"] or not relocated["valid"]:
        raise RuntimeError("Relocated recipe is invalid")
    replay = calculate("/replays", {"frozen_file": frozen, "output": output})
    if not replay["equivalent"] or replay["differences"]:
        raise RuntimeError("Replay differs")
    print(json.dumps({"operations": counter, "physical_widths_mm": widths,
                      "evaluation_status": run["evaluation_status"],
                      "replay_equivalent": replay["equivalent"],
                      "client_records": str(records)}, indent=2))


if __name__ == "__main__":
    main()

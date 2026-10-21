"""One upstream-owned teaching, calibration, production and investigation scenario."""

import csv
import hashlib
import json
import shutil

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from measure_api.v2.app import create_app
from measure_api.v2.models import ServiceSettings


def _calculate(client, route, request):
    response = client.post("/api/v2" + route, json=request)
    assert response.status_code in {200, 202}, response.text
    operation = response.json()
    while operation["state"] in {"queued", "running", "cancelling"}:
        operation = client.get(
            f"/api/v2/operations/{operation['operation_id']}?wait_ms=10000",
        ).json()
    assert operation["state"] == "succeeded", operation
    return operation


def _verify_artifacts(result, roots):
    manifest = result["manifest"]
    path = roots[manifest["file"]["root"]] / manifest["file"]["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
    marker = json.loads(path.read_text())
    assert marker["files"] == result["artifacts"]
    for descriptor in result["artifacts"]:
        artifact = roots[descriptor["file"]["root"]] / descriptor["file"]["path"]
        assert artifact.stat().st_size == descriptor["size_bytes"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == descriptor["sha256"]


def test_upstream_teaching_calibration_preview_production_export_package_and_replay(tmp_path):
    roots = {alias: tmp_path / alias for alias in ["exchange", "assets", "outputs"]}
    for path in roots.values():
        path.mkdir()
    taught = np.zeros((80, 80), np.uint8)
    taught[20:60, 30:50] = 255
    taught[22:29, 15:23] = 120
    taught[52:59, 55:63] = 190
    cv2.imwrite(str(roots["exchange"] / "reference.png"), taught)
    inspection = np.zeros((110, 220), np.uint8)
    inspection[10:90, 10:90] = taught
    inspection[5:101, 115:211] = cv2.resize(taught, (96, 96))
    cv2.imwrite(str(roots["exchange"] / "production.png"), inspection)
    initial_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                      for name, path in [("reference", roots["exchange"] / "reference.png"),
                                         ("production", roots["exchange"] / "production.png")]}
    reference = {"root": "exchange", "path": "reference.png",
                 "sha256": initial_hashes["reference"]}
    image = {"root": "exchange", "path": "production.png",
             "sha256": initial_hashes["production"]}
    output = {"directory": {"root": "outputs", "path": "computations"}}
    applicability = {"camera_id": "upstream-camera", "setup_id": "fixture-A",
                     "frame": "source", "image_size": {"width": 220, "height": 110}}
    settings = ServiceSettings(
        shared_roots=roots, writable_roots=["outputs"], state_directory=tmp_path / "state",
        max_workers=2,
    )
    with TestClient(create_app(settings)) as client:
        inspected = _calculate(client, "/image-inspections", {"image": reference})["result"]
        assert inspected["metadata"]["width"] == 80
        teaching = _calculate(client, "/template-builds", {
            "image": reference, "roi": {"center": {"x": 40, "y": 40},
                                        "width": 65, "height": 65}, "output": output,
        })["result"]
        _verify_artifacts(teaching, roots)
        source = teaching["template"]
        shutil.copyfile(roots[source["root"]] / source["path"],
                        roots["assets"] / "template-v1.npz")
        template = {"root": "assets", "path": "template-v1.npz"}
        calibration = _calculate(client, "/calibrations/scales", {
            "corners": [{"x": x, "y": y} for y in [5, 15] for x in [5, 15, 25]],
            "square_size": 1, "unit": "mm", "applicability": applicability, "output": output,
        })["result"]
        assert calibration["valid"]
        assert calibration["asset"]["parameters"]["scale_per_px"] == pytest.approx(0.1)
        _verify_artifacts(calibration, roots)
        (roots["assets"] / "calibration-v1.json").write_text(json.dumps(calibration["asset"]))
        recipe = {
            "id": "upstream-part", "version": "unsaved-edit",
            "reference": reference,
            "localization": {"template": template, "search": {
                "score_threshold": 0.85, "max_matches": 2,
                "scale_min": 1, "scale_max": 1.2, "scale_step": 0.2,
            }},
            "metric": {"calibration": {"file": {"root": "assets",
                                                 "path": "calibration-v1.json"}},
                       "applicability": applicability},
            "nodes": [
                {"id": "width", "operation": "measurements.edge-pairs", "parameters": {
                    "probe": {"center": {"x": 40, "y": 40}, "search_half_length": 20,
                              "average_half_width": 3},
                    "selection": "first", "polarity": "positive",
                }},
                {"id": "check", "operation": "evaluations.ranges", "space": "physical",
                 "parameters": {"minimum": 1.9, "maximum": 2.2, "unit": "mm"},
                 "inputs": {"value": {"node": "width", "port": "width", "space": "physical"}}},
            ],
            "required_outputs": [{"node": "check", "port": "value", "space": "physical"}],
            "target_policy": {"minimum": 2, "maximum": 2, "no_targets": "fail"},
        }
        validation = _calculate(client, "/recipe-validations", {"recipe": recipe})["result"]
        assert validation["valid"]
        assert validation["execution_order"] == ["width", "check"]
        preview = _calculate(client, "/recipe-previews", {
            "recipe": recipe, "image": image, "selected_nodes": ["check"],
            "upstream_revision": "editor-8", "preview_token": "preview-8",
        })["result"]
        assert preview["preview_token"] == "preview-8"
        assert preview["recipe_version"] == "unsaved-edit"
        assert len(preview["targets"]) == 2
        assert not (roots["assets"] / "recipe-v8.json").exists()
        recipe["version"] = "upstream-v8"
        (roots["assets"] / "recipe-v8.json").write_text(json.dumps(recipe))
        request = {"recipe_file": {"root": "assets", "path": "recipe-v8.json"},
                   "image": image, "output": output}
        sync_operation = _calculate(client, "/runs", request)
        run = sync_operation["result"]
        async_run = _calculate(client, "/runs", {**request, "mode": "async"})["result"]
        assert run["valid"] and async_run["valid"]
        assert run["recipe_version"] == "upstream-v8"
        assert run["targets"] == async_run["targets"]
        assert run["evaluation_status"] == "fail"
        assert sorted(target["evaluation_status"] for target in run["targets"]) == ["fail", "pass"]
        widths = sorted(target["nodes"][0]["physical_result"]["width"]["value"]
                        for target in run["targets"])
        assert widths == pytest.approx([2, 2.4], abs=0.02)
        pixel_widths = sorted(target["nodes"][0]["pixel_result"]["width"]["value"]
                              for target in run["targets"])
        assert pixel_widths == pytest.approx([20, 24], abs=0.2)
        _verify_artifacts(run, roots)
        snapshot = next(item["file"] for item in run["artifacts"]
                        if item["kind"] == "recipe_result")
        visual = _calculate(client, "/visualizations", {
            "result_file": snapshot, "space": "physical", "display_frame": "source",
            "raster": True, "diagnostics": True, "selected_nodes": ["width"], "output": output,
        })["result"]
        _verify_artifacts(visual, roots)
        assert {item["target_id"] for item in visual["overlays"]} == {
            target["target_id"] for target in run["targets"]}
        assert all(item["frame"] == "source" and item["unit"] == "px"
                   for item in visual["overlays"] if item["type"] != "annotation")
        raster = next(item["file"] for item in visual["artifacts"]
                      if item["kind"] == "raster_overlay")
        assert cv2.imread(str(roots[raster["root"]] / raster["path"])).shape == (110, 220, 3)
        assert any(item["format"] == "npz" for item in visual["artifacts"])
        exported = _calculate(client, "/result-exports", {
            "result_file": snapshot, "formats": ["csv", "json"], "output": output,
        })["result"]
        _verify_artifacts(exported, roots)
        csv_file = next(item["file"] for item in exported["artifacts"] if item["format"] == "csv")
        with (roots[csv_file["root"]] / csv_file["path"]).open() as file:
            rows = list(csv.DictReader(file))
        exported_widths = sorted(float(row["value"]) for row in rows
                                if row["node_id"] == "width" and row["port"] == "width"
                                and row["space"] == "physical")
        assert exported_widths == pytest.approx([2, 2.4], abs=0.02)
        assert {row["unit"] for row in rows if row["space"] == "physical"} == {"mm"}
        json_file = next(item["file"] for item in exported["artifacts"] if item["format"] == "json")
        archived_result = json.loads((roots[json_file["root"]] / json_file["path"]).read_text())
        assert archived_result["targets"] == run["targets"]
        assert archived_result["evaluation_status"] == "fail"
        diagnostics = client.get(
            f"/api/v2/operations/{sync_operation['operation_id']}/diagnostics",
        ).json()
        assert diagnostics["compute_ms"] >= 0 and diagnostics["preparation_ms"] >= 0
        trace = client.get(f"/api/v2/traces/{sync_operation['trace_id']}").json()
        assert any(item["operation_id"] == sync_operation["operation_id"]
                   for item in trace["operations"])
        packaged = _calculate(client, "/recipe-exports", {
            "recipe": run["resolved_recipe"], "frozen_run_file": json_file,
            "packaging": "embedded", "output": output,
        })["result"]
        _verify_artifacts(packaged, roots)
        package = next(item["file"] for item in packaged["artifacts"]
                       if item["kind"] == "recipe_package")
        frozen = next(item["file"] for item in packaged["artifacts"]
                      if item["kind"] == "frozen_run")
        imported = _calculate(client, "/recipe-imports", {
            "package": package, "output": output,
        })["result"]
        assert imported["valid"]
        _verify_artifacts(imported, roots)
        assert imported["resolved_recipe"]["version"] == "upstream-v8"
        relocated = _calculate(client, "/runs", {
            "recipe": imported["resolved_recipe"], "image": image, "output": output,
        })["result"]
        assert relocated["valid"]
        assert sorted(target["nodes"][0]["physical_result"]["width"]["value"]
                      for target in relocated["targets"]) == pytest.approx([2, 2.4], abs=0.02)
        assert relocated["evaluation_status"] == "fail"
        for name in initial_hashes:
            filename = "reference.png" if name == "reference" else "production.png"
            assert hashlib.sha256((roots["exchange"] / filename).read_bytes()).hexdigest() == (
                initial_hashes[name])
    # A fresh operational state owns no historical job; archived files are sufficient.
    replay_settings = settings.model_copy(update={"state_directory": tmp_path / "fresh-state"})
    with TestClient(create_app(replay_settings)) as fresh:
        assert fresh.get(f"/api/v2/operations/{sync_operation['operation_id']}").status_code == 404
        replayed = _calculate(
            fresh, "/replays", {"frozen_file": frozen, "output": output})["result"]
        assert replayed["equivalent"] and replayed["differences"] == []
        assert replayed["run"]["valid"]
        assert replayed["run"]["evaluation_status"] == "fail"
        assert replayed["run"]["targets"] == run["targets"]
        _verify_artifacts(replayed["run"], roots)
    for name in initial_hashes:
        filename = "reference.png" if name == "reference" else "production.png"
        assert hashlib.sha256((roots["exchange"] / filename).read_bytes()).hexdigest() == (
            initial_hashes[name])

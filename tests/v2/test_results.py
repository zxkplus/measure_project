"""Supplied-result visualization, reporting and diagnostic HTTP acceptance."""

import csv
import hashlib
import io
import json
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from measure_api.v2.app import create_app
from measure_api.v2.models import ServiceSettings


@pytest.fixture
def client(tmp_path):
    cv2.imwrite(str(tmp_path / "image.png"), np.zeros((60, 80), np.uint8))
    with TestClient(create_app(ServiceSettings(shared_roots={"exchange": tmp_path}))) as value:
        yield value


def post(client, path, payload):
    response = client.post("/api/v2" + path, json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["state"] == "succeeded"
    return response.json()


def atomic(value, operation="geometry.lines_fit"):
    return {"operation": operation, "value": value, "node_id": "line", "label": "Measured"}


def test_vectors_keep_decimal_geometry_and_independent_visibility(client, tmp_path):
    line = post(
        client,
        "/geometry/lines:fit",
        {
            "points": [{"x": 10.25, "y": 20.5}, {"x": 60.25, "y": 20.5}],
        },
    )["result"]
    request = {
        "result": atomic(line),
        "image": {"root": "exchange", "path": "image.png"},
        "raster": True,
        "output": {"directory": {"root": "exchange", "path": "out"}},
    }
    rendered = post(client, "/visualizations", request)["result"]
    vector = next(item for item in rendered["overlays"] if item["type"] == "line")
    assert vector["geometry"]["start"] == pytest.approx({"x": 65.25, "y": 20.5})
    assert vector["frame"] == "source" and vector["unit"] == "px"
    assert vector["node_id"] == "line" and vector["visible"] and not vector["selected"]
    raster = next(item for item in rendered["artifacts"] if item["kind"] == "raster_overlay")
    pixels = cv2.imread(str(tmp_path / raster["file"]["path"]))
    assert pixels.shape == (60, 80, 3)
    assert pixels[20, 30].tolist() == [0, 255, 0]
    hidden = post(
        client, "/visualizations", {**request, "visible_nodes": [], "selected_nodes": ["line"]}
    )["result"]
    second = next(item for item in hidden["overlays"] if item["type"] == "line")
    assert second["id"] == vector["id"] and not second["visible"] and second["selected"]
    path = next(item for item in hidden["artifacts"] if item["kind"] == "raster_overlay")
    assert not cv2.imread(str(tmp_path / path["file"]["path"])).any()
    assert not cv2.imread(str(tmp_path / "image.png")).any()


def test_projective_circle_is_sampled_and_nonlinear_inverse_uses_lens_model(client):
    circle = post(
        client,
        "/geometry/circles:fit",
        {
            "points": [
                {"x": 10, "y": 0},
                {"x": 0, "y": 10},
                {"x": -10, "y": 0},
                {"x": 0, "y": -10},
            ],
        },
    )["result"]
    mapping = {
        "source_frame": "source",
        "destination_frame": "display",
        "type": "projective",
        "matrix": [[1, 0, 0], [0, 1, 0], [0.01, 0, 1]],
        "inverse_matrix": [[1, 0, 0], [0, 1, 0], [-0.01, 0, 1]],
    }
    vectors = post(
        client,
        "/visualizations",
        {
            "result": atomic(circle, "geometry.circles_fit"),
            "display_frame": "display",
            "mapping": mapping,
        },
    )["result"]["overlays"]
    curve = next(item for item in vectors if item["port"] == "circle")
    assert curve["type"] == "polyline" and curve["geometry"]["closed"]
    assert curve["geometry"]["points"][0] == pytest.approx({"x": 9.09090909090909, "y": 0})
    assert curve["geometry"]["points"][32] == pytest.approx({"x": 0, "y": 10}, abs=1e-8)
    circle["frame"] = circle["circle"]["frame"] = "physical"
    circle["unit"] = circle["circle"]["unit"] = "mm"
    circle["samples"] = []
    nonlinear = {
        "source_frame": "source",
        "destination_frame": "physical",
        "type": "nonlinear",
        "unit": "mm",
        "details": {
            "model": "opencv_pinhole_plane",
            "camera_matrix": [[100, 0, 40], [0, 100, 30], [0, 0, 1]],
            "distortion": [0.1, 0, 0, 0, 0],
            "corrected_to_physical": [[1, 0, -40], [0, 1, -30], [0, 0, 1]],
            "physical_to_corrected": [[1, 0, 40], [0, 1, 30], [0, 0, 1]],
        },
    }
    vectors = post(
        client,
        "/visualizations",
        {"result": atomic(circle, "geometry.circles_fit"), "mapping": nonlinear},
    )["result"]["overlays"]
    curve = next(item for item in vectors if item["port"] == "circle")
    assert curve["type"] == "polyline" and curve["unit"] == "px"
    assert curve["geometry"]["points"][0] == pytest.approx({"x": 50.01, "y": 30})
    assert curve["geometry"]["points"][32] == pytest.approx({"x": 40, "y": 40.01})


def measured_recipe(client, tmp_path, output=None):
    image = np.zeros((60, 80), np.uint8)
    image[:, 30:54] = 255
    cv2.imwrite(str(tmp_path / "image.png"), image)
    applicability = {
        "camera_id": "c",
        "setup_id": "s",
        "frame": "source",
        "image_size": {"width": 80, "height": 60},
    }
    recipe = {
        "id": "part",
        "version": "42",
        "metric": {
            "applicability": applicability,
            "calibration": {
                "value": {
                    "kind": "scalar",
                    "unit": "mm",
                    "applicability": applicability,
                    "parameters": {"scale_per_px": 0.1, "spacing_px": 10, "square_size": 1},
                }
            },
        },
        "nodes": [
            {
                "id": "width",
                "label": "宽度,第一\n行",
                "operation": "measurements.edge-pairs",
                "parameters": {
                    "probe": {
                        "center": {"x": 42, "y": 30},
                        "search_half_length": 25,
                        "average_half_width": 3,
                    },
                    "selection": "first",
                },
            },
            {
                "id": "missing",
                "operation": "measurements.edge-points",
                "required": False,
                "parameters": {
                    "probe": {
                        "center": {"x": 10, "y": 30},
                        "search_half_length": 6,
                        "average_half_width": 3,
                    }
                },
            },
            {
                "id": "policy",
                "operation": "evaluations.ranges",
                "space": "physical",
                "parameters": {"minimum": 0, "maximum": 2, "unit": "mm"},
                "inputs": {"value": {"node": "width", "port": "width", "space": "physical"}},
            },
        ],
    }
    payload = {"recipe": recipe, "image": {"root": "exchange", "path": "image.png"}}
    if output is not None:
        payload["output"] = output
    return post(client, "/runs", payload)


def test_csv_json_export_preserves_spaces_precision_invalidity_and_evaluation(client, tmp_path):
    supplied = measured_recipe(client, tmp_path)["result"]
    report = post(
        client,
        "/result-exports",
        {
            "result": supplied,
            "formats": ["csv", "json"],
            "output": {"directory": {"root": "exchange", "path": "reports"}},
        },
    )["result"]
    files = {item["format"]: item for item in report["artifacts"]}
    rows = list(csv.DictReader(io.StringIO((tmp_path / files["csv"]["file"]["path"]).read_text())))
    pixel = next(
        row
        for row in rows
        if row["node_id"] == "width" and row["port"] == "width" and row["space"] == "pixel"
    )
    physical = next(
        row
        for row in rows
        if row["node_id"] == "width" and row["port"] == "width" and row["space"] == "physical"
    )
    assert float(pixel["value"]) == pytest.approx(24)
    assert float(physical["value"]) == pytest.approx(2.4)
    assert (pixel["unit"], physical["unit"]) == ("px", "mm")
    assert physical["label"] == "宽度,第一\n行"
    assert physical["target_id"] == "target-1" and physical["recipe_version"] == "42"
    assert all(row["value"] == "" for row in rows if row["node_id"] == "missing")
    assert next(row for row in rows if row["node_id"] == "policy")["evaluation_status"] == "fail"
    restored = json.loads((tmp_path / files["json"]["file"]["path"]).read_text())
    assert restored == supplied
    for descriptor in [*report["artifacts"], report["manifest"]]:
        data = (tmp_path / descriptor["file"]["path"]).read_bytes()
        assert descriptor["schema_version"] == "1.0"
        assert descriptor["size_bytes"] == len(data)
        assert descriptor["sha256"] == hashlib.sha256(data).hexdigest()
    marker = json.loads((tmp_path / report["manifest"]["file"]["path"]).read_text())
    assert marker["files"] == report["artifacts"]


def test_curve_and_selected_sample_diagnostics_are_shared_numeric_files(client, tmp_path):
    image = np.zeros((60, 80), np.uint8)
    image[:, 40:] = 255
    cv2.imwrite(str(tmp_path / "image.png"), image)
    profile = post(
        client,
        "/edge-profiles",
        {
            "image": {"root": "exchange", "path": "image.png"},
            "probe": {
                "center": {"x": 40, "y": 30},
                "search_half_length": 25,
                "average_half_width": 3,
            },
            "inline_samples": False,
            "output": {"directory": {"root": "exchange", "path": "profiles"}},
        },
    )["result"]
    rendered = post(
        client,
        "/visualizations",
        {
            "result": atomic(profile, "edge-profiles"),
            "diagnostics": True,
            "output": {"directory": {"root": "exchange", "path": "debug"}},
        },
    )["result"]
    curves = next(item for item in rendered["artifacts"] if item["kind"] == "curves")
    with np.load(tmp_path / curves["file"]["path"], allow_pickle=False) as payload:
        assert payload["gray"].shape == (50,)
        assert payload["gray"][20] == 0 and payload["gray"][30] == 255
        assert payload["sample_points"][0] == pytest.approx([15, 30])
    chart = next(item for item in rendered["artifacts"] if item["kind"] == "curve_preview")
    assert cv2.imread(str(tmp_path / chart["file"]["path"])).shape == (240, 480, 3)
    supplied = measured_recipe(client, tmp_path)["result"]
    rendered = post(
        client,
        "/visualizations",
        {
            "result": supplied,
            "diagnostics": True,
            "output": {"directory": {"root": "exchange", "path": "selected"}},
        },
    )["result"]
    samples = next(item for item in rendered["artifacts"] if item["kind"] == "selected_samples")
    with np.load(tmp_path / samples["file"]["path"], allow_pickle=False) as payload:
        values = payload[next(key for key in payload.files if "pairs" in key and "edge_a" in key)]
        assert values[0, :2] == pytest.approx([29.5, 30])
    index = next(item for item in rendered["artifacts"] if item["kind"] == "sampling_diagnostics")
    metadata = json.loads((tmp_path / index["file"]["path"]).read_text())
    assert any(
        entry["node_id"] == "width" and entry["selected_sample_count"] > 0
        for entry in metadata["nodes"]
    )
    assert "gray" not in rendered and "sample_points" not in rendered


def test_retained_diagnostics_have_real_stage_quality_and_distinct_queue_compute(client, tmp_path):
    run = measured_recipe(client, tmp_path)
    diagnostics = client.get(f"/api/v2/operations/{run['operation_id']}/diagnostics").json()
    assert diagnostics["stage_timings"] == run["result"]["stage_timings"]
    width = next(node for node in diagnostics["nodes"] if node["node_id"] == "width")
    assert width["selected_sample_count"] > 0
    assert [29.5, 30] in width["selected_sample_preview"]
    assert width["quality"] is None
    trace = client.get(f"/api/v2/traces/{run['trace_id']}").json()
    assert trace["operations"][0]["diagnostics"]["stage_timings"] == diagnostics["stage_timings"]

    import os
    from concurrent.futures import ThreadPoolExecutor
    from test_operation_lifecycle import real_server, wait_for_boundary

    queue_root = tmp_path / "queue"
    queue_root.mkdir()
    cv2.imwrite(str(queue_root / "image.png"), np.zeros((60, 80), np.uint8))
    os.mkfifo(queue_root / "release.fifo")
    with real_server(queue_root, "before-payload") as (remote, _):
        first = remote.post(
            "/api/v2/image-previews",
            json={
                "mode": "async",
                "image": {"root": "exchange", "path": "image.png"},
                "output": {"directory": {"root": "exchange", "path": "outputs"}},
            },
        )
        assert first.status_code == 202
        wait_for_boundary(queue_root)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                remote.post,
                "/api/v2/image-inspections",
                json={"image": {"root": "exchange", "path": "image.png"}},
            )
            time.sleep(0.25)
            with (queue_root / "release.fifo").open("wb") as release:
                release.write(b"1")
            second = future.result(timeout=10)
        assert second.status_code == 200, second.text
        metrics = second.json()["diagnostics"]
        assert metrics["queue_ms"] >= 150
        assert metrics["compute_ms"] < metrics["queue_ms"]
        assert metrics["preparation_ms"] >= 0 and metrics["worker_startup_ms"] > 0


def test_supplied_files_render_and_export_after_original_operation_and_trace_expire(tmp_path):
    settings = ServiceSettings(
        shared_roots={"exchange": tmp_path}, operation_retention_seconds=0.15
    )
    with TestClient(create_app(settings)) as client:
        run = measured_recipe(client, tmp_path)
        (tmp_path / "saved.json").write_text(json.dumps(run["result"]))
        time.sleep(0.2)
        assert client.get(f"/api/v2/operations/{run['operation_id']}").status_code == 410
        trace = client.get(f"/api/v2/traces/{run['trace_id']}")
        assert trace.status_code == 410 and trace.json()["error"]["code"] == "trace.expired"
        rendered = post(
            client,
            "/visualizations",
            {
                "result_file": {"root": "exchange", "path": "saved.json"},
                "raster": True,
                "output": {"directory": {"root": "exchange", "path": "kept"}},
            },
        )["result"]
        reported = post(
            client,
            "/result-exports",
            {
                "result_file": {"root": "exchange", "path": "saved.json"},
                "formats": ["csv", "json"],
                "output": {"directory": {"root": "exchange", "path": "kept"}},
            },
        )["result"]
        time.sleep(0.2)
    with TestClient(create_app(settings)) as restarted:
        assert restarted.get("/api/v2/health/ready").status_code == 200
        assert restarted.get(f"/api/v2/traces/{run['trace_id']}").status_code == 410
    assert all(
        (tmp_path / item["file"]["path"]).is_file()
        for item in [
            *rendered["artifacts"],
            rendered["manifest"],
            *reported["artifacts"],
            reported["manifest"],
        ]
    )


def test_result_file_and_nested_numeric_digests_are_checked_before_async_admission(
    client, tmp_path
):
    line = post(
        client,
        "/geometry/lines:fit",
        {
            "points": [{"x": 10, "y": 20}, {"x": 40, "y": 20}],
        },
    )["result"]
    (tmp_path / "result.json").write_text(json.dumps(atomic(line)))
    request = {
        "result_file": {"root": "exchange", "path": "result.json"},
        "output": {"directory": {"root": "exchange", "path": "out"}},
        "idempotency_key": "report",
    }
    original = post(client, "/result-exports", request)
    assert post(client, "/result-exports", request)["operation_id"] == original["operation_id"]
    line["line"]["c"] = -30
    (tmp_path / "result.json").write_text(json.dumps(atomic(line)))
    changed = client.post("/api/v2/result-exports", json={**request, "mode": "async"})
    assert changed.status_code == 409
    assert changed.json()["error"]["code"] == "operation.idempotency_conflict"
    supplied = measured_recipe(client, tmp_path)["result"]
    supplied["resolved_recipe"]["reference"] = {
        "root": "exchange",
        "path": "image.png",
        "sha256": "0" * 64,
    }
    rejected = client.post("/api/v2/visualizations", json={"result": supplied, "mode": "async"})
    assert (
        rejected.status_code == 409 and rejected.json()["error"]["code"] == "file.digest_mismatch"
    )


def test_rotated_rectangle_and_polyline_overlays_preserve_declared_frame(client):
    mapping = {
        "source_frame": "source",
        "destination_frame": "source",
        "type": "identity",
        "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "inverse_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    transformed = post(
        client,
        "/geometry/transforms",
        {
            "mapping": mapping,
            "geometry": [
                {
                    "type": "rectangle",
                    "center": {"x": 30, "y": 20},
                    "width": 10,
                    "height": 4,
                    "angle_degrees": 90,
                },
                {"type": "polyline", "points": [{"x": 10.2, "y": 12.3}, {"x": 40.4, "y": 22.5}]},
            ],
        },
    )["result"]
    rendered = post(
        client, "/visualizations", {"result": atomic(transformed, "geometry.transforms")}
    )["result"]
    rectangle = next(item for item in rendered["overlays"] if item["type"] == "rectangle")
    assert rectangle["geometry"]["points"][0] == pytest.approx({"x": 32, "y": 15})
    assert rectangle["geometry"]["points"][2] == pytest.approx({"x": 28, "y": 25})
    polyline = next(item for item in rendered["overlays"] if item["type"] == "polyline")
    assert polyline["geometry"]["points"][0] == pytest.approx({"x": 10.2, "y": 12.3})
    assert not polyline["geometry"]["closed"]


def test_fit_quality_failures_and_probe_provenance_are_inspectable(client, tmp_path):
    image = np.zeros((60, 80), np.uint8)
    image[30:] = 255
    cv2.imwrite(str(tmp_path / "image.png"), image)
    recipe = {
        "id": "quality",
        "version": "1",
        "nodes": [
            {
                "id": "line",
                "operation": "measurements.lines",
                "parameters": {
                    "start": {"x": 10, "y": 30},
                    "end": {"x": 70, "y": 30},
                    "search_half_length": 8,
                    "average_half_width": 2,
                    "probe_count": 5,
                },
            },
            {
                "id": "outside",
                "required": False,
                "operation": "measurements.edge-points",
                "parameters": {
                    "probe": {
                        "center": {"x": -40, "y": -20},
                        "search_half_length": 5,
                        "average_half_width": 2,
                    }
                },
            },
        ],
    }
    run = post(
        client, "/runs", {"recipe": recipe, "image": {"root": "exchange", "path": "image.png"}}
    )
    diagnostics = client.get(f"/api/v2/operations/{run['operation_id']}/diagnostics").json()
    line = next(node for node in diagnostics["nodes"] if node["node_id"] == "line")
    assert (
        line["quality"]["requested_probe_count"] == line["quality"]["successful_probe_count"] == 5
    )
    assert line["selected_sample_count"] == 5
    assert line["quality"]["rms_error"] == pytest.approx(0, abs=1e-8)
    outside = next(node for node in diagnostics["nodes"] if node["node_id"] == "outside")
    assert outside["error"]["code"] == "image.roi_outside"
    rendered = post(
        client,
        "/visualizations",
        {
            "result": run["result"],
            "diagnostics": True,
            "output": {"directory": {"root": "exchange", "path": "diagnostics"}},
        },
    )["result"]
    index = next(item for item in rendered["artifacts"] if item["kind"] == "sampling_diagnostics")
    metadata = json.loads((tmp_path / index["file"]["path"]).read_text())
    line = next(node for node in metadata["nodes"] if node["node_id"] == "line")
    assert len(line["probes"]["areas"]) == 5
    assert line["probes"]["frame"] == "source"
    assert line["selected_sample_count"] == 5
    assert (
        next(item for item in rendered["overlays"] if item["node_id"] == "outside")["valid"]
        is False
    )


def test_cancelling_multi_file_report_before_manifest_removes_attempt(tmp_path):
    import os
    from test_operation_lifecycle import real_server, wait_for_boundary

    os.mkfifo(tmp_path / "release.fifo")
    supplied = atomic(
        {"valid": True, "frame": "source", "unit": "px", "value": 24}, "geometry.point-distances"
    )
    with real_server(tmp_path, "before-manifest") as (remote, _):
        accepted = remote.post(
            "/api/v2/result-exports",
            json={
                "mode": "async",
                "result": supplied,
                "formats": ["csv", "json"],
                "output": {"directory": {"root": "exchange", "path": "reports"}},
            },
        )
        assert accepted.status_code == 202, accepted.text
        wait_for_boundary(tmp_path)
        assert not list((tmp_path / "reports").rglob("manifest.json"))
        cancelled = remote.post(f"/api/v2/operations/{accepted.json()['operation_id']}:cancel")
        assert cancelled.json()["state"] == "cancelled", cancelled.text
        remote.get(f"/api/v2/operations/{accepted.json()['operation_id']}?wait_ms=5000")
        assert not list((tmp_path / "reports").rglob("result.*"))
        diagnostics = remote.get(
            f"/api/v2/operations/{accepted.json()['operation_id']}/diagnostics"
        ).json()
        assert diagnostics["failure"]["code"] == "operation.cancelled"


def test_batch_reports_keep_failed_item_identity_and_physical_only_node(client, tmp_path):
    good = measured_recipe(client, tmp_path)["result"]
    supplied = {
        "valid": False,
        "items": [
            {"id": "frame-1", "state": "succeeded", "valid": good["valid"], "result": good},
            {
                "id": "frame-2",
                "state": "failed",
                "valid": False,
                "error": {"code": "file.not_found", "message": "Missing frame"},
            },
        ],
    }
    exported = post(
        client,
        "/result-exports",
        {
            "result": supplied,
            "formats": ["csv"],
            "output": {"directory": {"root": "exchange", "path": "batches"}},
        },
    )["result"]
    rows = list(
        csv.DictReader(
            io.StringIO((tmp_path / exported["artifacts"][0]["file"]["path"]).read_text())
        )
    )
    failed = next(row for row in rows if row["item_id"] == "frame-2")
    assert failed["state"] == "failed" and failed["value"] == "" and failed["valid"] == "False"
    assert json.loads(failed["error_json"])["code"] == "file.not_found"
    policy = next(row for row in rows if row["node_id"] == "policy")
    assert policy["space"] == "physical" and policy["evaluation_status"] == "fail"
    visualized = post(client, "/visualizations", {"result": supplied})["result"]
    assert all(item["item_id"] == "frame-1" for item in visualized["overlays"])


def test_malformed_supplied_geometry_is_rejected_without_a_completed_manifest(client, tmp_path):
    supplied = measured_recipe(client, tmp_path)["result"]
    next(node for node in supplied["targets"][0]["nodes"] if node["id"] == "width")["pixel_result"][
        "edge_a"
    ]["point"] = {"x": "wrong", "y": 30}
    response = client.post(
        "/api/v2/visualizations",
        json={
            "result": supplied,
            "mode": "async",
            "raster": True,
            "output": {"directory": {"root": "exchange", "path": "invalid"}},
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "result.invalid"
    assert not list((tmp_path / "invalid").rglob("manifest.json"))


def test_numeric_diagnostics_reject_pickle_payloads_and_descriptor_size_mismatch(client, tmp_path):
    archive = io.BytesIO()
    np.savez(archive, gray=np.array([{"unsafe": True}], dtype=object))
    data = archive.getvalue()
    (tmp_path / "unsafe.npz").write_bytes(data)
    descriptor = {
        "kind": "curves",
        "format": "npz",
        "schema_version": "1.0",
        "file": {"root": "exchange", "path": "unsafe.npz"},
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }
    supplied = atomic(
        {
            "valid": True,
            "frame": "source",
            "unit": "px",
            "gray": [],
            "smoothed": [],
            "gradient": [],
            "sample_points": [],
            "threshold_gray": 30,
            "artifacts": [descriptor],
        },
        "edge-profiles",
    )
    payload = {
        "result": supplied,
        "diagnostics": True,
        "output": {"directory": {"root": "exchange", "path": "unsafe-output"}},
    }
    response = client.post("/api/v2/visualizations", json=payload)
    assert response.status_code == 422 and response.json()["error"]["code"] == "diagnostics.invalid"
    assert not list((tmp_path / "unsafe-output").rglob("manifest.json"))
    descriptor["size_bytes"] += 1
    payload["mode"] = "async"
    response = client.post("/api/v2/visualizations", json=payload)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "result.artifact_metadata_mismatch"


def test_physical_run_overlays_automatically_project_to_retained_source_pixels(client, tmp_path):
    supplied = measured_recipe(client, tmp_path)["result"]
    rendered = post(
        client,
        "/visualizations",
        {
            "result": supplied,
            "space": "physical",
            "raster": True,
            "output": {"directory": {"root": "exchange", "path": "physical"}},
        },
    )["result"]
    edge = next(
        item
        for item in rendered["overlays"]
        if item["node_id"] == "width" and item["port"] == "edge_a"
    )
    assert edge["frame"] == "source" and edge["unit"] == "px"
    assert edge["geometry"] == pytest.approx({"x": 29.5, "y": 30})
    raster = next(item for item in rendered["artifacts"] if item["kind"] == "raster_overlay")
    pixels = cv2.imread(str(tmp_path / raster["file"]["path"]))
    assert pixels[30, 30].tolist() == [0, 255, 0]


def test_atomic_physical_scalar_report_declares_space_and_preserves_full_precision(
    client, tmp_path
):
    supplied = {
        **atomic(
            {"valid": True, "frame": "physical", "unit": "mm", "value": 1.2345678901234567},
            "geometry.point-distances",
        ),
        "space": "physical",
    }
    exported = post(
        client,
        "/result-exports",
        {
            "result": supplied,
            "formats": ["csv"],
            "output": {"directory": {"root": "exchange", "path": "precise"}},
        },
    )["result"]
    rows = list(
        csv.DictReader(
            io.StringIO((tmp_path / exported["artifacts"][0]["file"]["path"]).read_text())
        )
    )
    assert len(rows) == 1 and rows[0]["space"] == "physical"
    assert rows[0]["value"] == "1.2345678901234567"
    assert rows[0]["unit"] == "mm" and rows[0]["evaluation_status"] == "not_evaluated"


def test_http_worker_commit_deadline_uses_one_system_monotonic_clock_across_processes(
    client, tmp_path
):
    inspected = post(
        client,
        "/image-inspections",
        {
            "image": {"root": "exchange", "path": "image.png"},
            "timeout_ms": 3000,
        },
    )
    diagnostics = client.get(f"/api/v2/operations/{inspected['operation_id']}/diagnostics").json()
    assert diagnostics["deadline_clock"] == "CLOCK_MONOTONIC"
    assert diagnostics["admitted_monotonic"] <= diagnostics["worker_started_monotonic"]
    assert diagnostics["worker_started_monotonic"] < diagnostics["deadline_monotonic"]
    assert diagnostics["deadline_monotonic"] - diagnostics["admitted_monotonic"] == pytest.approx(3)


def test_expired_multi_file_report_never_publishes_a_manifest(tmp_path):
    import os
    from test_operation_lifecycle import real_server, wait_for_boundary

    os.mkfifo(tmp_path / "release.fifo")
    supplied = atomic(
        {"valid": True, "frame": "source", "unit": "px", "value": 24}, "geometry.point-distances"
    )
    with real_server(tmp_path, "before-manifest") as (remote, _):
        accepted = remote.post(
            "/api/v2/result-exports",
            json={
                "mode": "async",
                "timeout_ms": 1500,
                "result": supplied,
                "formats": ["csv", "json"],
                "output": {"directory": {"root": "exchange", "path": "expired"}},
            },
        )
        assert accepted.status_code == 202, accepted.text
        wait_for_boundary(tmp_path)
        final = remote.get(
            f"/api/v2/operations/{accepted.json()['operation_id']}?wait_ms=5000"
        ).json()
        assert final["state"] == "timed_out" and final["error"]["code"] == "operation.timed_out"
        assert final["diagnostics"]["deadline_clock"] == "CLOCK_MONOTONIC"
        assert not list((tmp_path / "expired").rglob("manifest.json"))
        assert not list((tmp_path / "expired").rglob("result.*"))


@pytest.mark.parametrize("selection, expected_x", [("first", [19.5]), ("all", [19.5, 39.5])])
def test_edge_point_observations_and_arrays_render_as_source_xy(
    client, tmp_path, selection, expected_x
):
    pixels = np.zeros((60, 80), np.uint8)
    pixels[:, 20:40] = 255
    cv2.imwrite(str(tmp_path / "image.png"), pixels)
    measured = post(client, "/measurements/edge-points", {
        "image": {"root": "exchange", "path": "image.png"},
        "probe": {"center": {"x": 40, "y": 30},
                  "search_half_length": 30, "average_half_width": 3},
        "selection": selection,
    })["result"]
    request = {
        "result": atomic(measured, "measurements.edge-points"),
        "image": {"root": "exchange", "path": "image.png"},
        "raster": True, "output": {"directory": {"root": "exchange", "path": "points"}},
    }
    rendered = post(client, "/visualizations", request)["result"]
    points = [overlay for overlay in rendered["overlays"] if overlay["type"] == "point"]
    assert len(points) == len(expected_x)
    assert [point["geometry"]["x"] for point in points] == pytest.approx(expected_x, abs=0.02)
    for point in points:
        assert set(point["geometry"]) == {"x", "y"}
        assert point["geometry"]["y"] == pytest.approx(30, abs=0.02)
        assert point["frame"] == "source" and point["unit"] == "px" and point["valid"]
    artifact = next(a for a in rendered["artifacts"] if a["kind"] == "raster_overlay")
    raster = cv2.imread(str(tmp_path / artifact["file"]["path"]))
    assert any(np.all(raster[30, round(x)] == [0, 255, 0]) for x in expected_x)
    assert np.any(raster != cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR))
    if selection == "first":
        measured["point"].update(valid=False, reason="invalid_local_point")
        invalid = post(client, "/visualizations", request)["result"]
        assert len(invalid["overlays"]) == 1
        assert invalid["overlays"][0]["type"] == "annotation"
        assert not invalid["overlays"][0]["valid"]
        assert invalid["overlays"][0]["geometry"]["reason"] == "invalid_local_point"
        artifact = next(a for a in invalid["artifacts"] if a["kind"] == "raster_overlay")
        assert np.array_equal(
            cv2.imread(str(tmp_path / artifact["file"]["path"])),
            cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR),
        )

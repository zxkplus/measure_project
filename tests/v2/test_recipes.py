"""Public HTTP acceptance for upstream-owned recipe snapshots."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from measure_api.v2.app import create_app
from measure_api.v2.models import ServiceSettings


@pytest.fixture
def client(tmp_path):
    pixels = np.zeros((100, 160), np.uint8)
    pixels[:, 40:60] = 255
    cv2.imwrite(str(tmp_path / "stripes.png"), pixels)
    with TestClient(create_app(ServiceSettings(shared_roots={"exchange": tmp_path}))) as value:
        yield value


IMAGE = {"root": "exchange", "path": "stripes.png"}


def recipe():
    return {
        "id": "upstream-part",
        "version": "unsaved-7",
        "nodes": [
            {
                "id": "check",
                "operation": "evaluations.ranges",
                "parameters": {"maximum": 21},
                "inputs": {"value": {"node": "width", "port": "width"}},
            },
            {
                "id": "width",
                "operation": "measurements.edge-pairs",
                "parameters": {
                    "probe": {
                        "center": {"x": 50, "y": 50},
                        "search_half_length": 25,
                        "average_half_width": 3,
                    },
                    "selection": "first",
                    "polarity": "positive",
                },
            },
        ],
        "required_outputs": [{"node": "check", "port": "value"}],
    }


def result(client, route, payload):
    response = client.post("/api/v2" + route, json=payload)
    assert response.status_code == 200, response.text
    return response.json()["result"]


def test_validate_forward_typed_bindings_without_saving_recipe(client):
    validated = result(client, "/recipe-validations", {"recipe": recipe()})
    assert validated["valid"]
    assert validated["execution_order"] == ["width", "check"]
    assert validated["recipe_id"] == "upstream-part"
    assert validated["recipe_version"] == "unsaved-7"
    assert len(validated["configuration_sha256"]) == 64


def test_preview_executes_only_dependency_closure_and_echoes_token(client):
    supplied = recipe()
    supplied["nodes"].append(
        {
            "id": "unrelated",
            "operation": "geometry.point-distances",
            "parameters": {
                "point_a": {"valid": True, "point": {"x": 0, "y": 0}},
                "point_b": {"valid": True, "point": {"x": 3, "y": 4}},
            },
        }
    )
    preview = result(
        client,
        "/recipe-previews",
        {
            "image": IMAGE,
            "recipe": supplied,
            "selected_nodes": ["check"],
            "upstream_revision": "editor-42",
            "preview_token": "p-99",
        },
    )
    assert preview["upstream_revision"] == "editor-42"
    assert preview["preview_token"] == "p-99"
    assert [node["id"] for node in preview["targets"][0]["nodes"]] == ["width", "check"]
    assert preview["targets"][0]["nodes"][0]["result"]["width"]["value"] == pytest.approx(20)
    assert preview["targets"][0]["nodes"][1]["result"]["evaluation_status"] == "pass"


def scalar_calibration(width=160, height=100):
    return {
        "value": {
            "kind": "scalar",
            "unit": "mm",
            "applicability": {
                "camera_id": "camera-1",
                "setup_id": "setup-1",
                "frame": "source",
                "image_size": {"width": width, "height": height},
            },
            "parameters": {"scale_per_px": 0.1, "spacing_px": 10, "square_size": 1},
        }
    }


def test_normalized_template_target_preserves_actual_24px_and_2_4mm(client, tmp_path):
    taught = np.zeros((100, 160), np.uint8)
    taught[25:75, 40:60] = 255
    taught[28:35, 65:75] = 180
    taught[63:71, 25:33] = 100
    cv2.imwrite(str(tmp_path / "taught.png"), taught)
    built = result(
        client,
        "/template-builds",
        {
            "image": {"root": "exchange", "path": "taught.png"},
            "roi": {"center": {"x": 50, "y": 50}, "width": 61, "height": 71},
            "output": {"directory": {"root": "exchange", "path": "templates"}},
        },
    )
    actual = cv2.warpAffine(taught, np.array([[1.2, 0, 0], [0, 1.2, -10.0]]), (160, 100))
    cv2.imwrite(str(tmp_path / "actual.png"), actual)
    supplied = recipe()
    supplied["reference"] = {"root": "exchange", "path": "taught.png"}
    supplied["localization"] = {
        "template": built["template"],
        "search": {"score_threshold": 0.8, "scale_min": 1.2, "scale_max": 1.2},
    }
    supplied["alignment"] = {
        "width": 160,
        "height": 100,
        "reference_to_patch": {
            "source_frame": "reference",
            "destination_frame": "aligned",
            "type": "identity",
            "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "inverse_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
    }
    supplied["metric"] = {
        "calibration": scalar_calibration(),
        "applicability": {
            "camera_id": "camera-1",
            "setup_id": "setup-1",
            "frame": "source",
            "image_size": {"width": 160, "height": 100},
        },
    }
    measured = result(
        client, "/runs", {"recipe": supplied, "image": {"root": "exchange", "path": "actual.png"}}
    )
    assert measured["valid"], str(measured["targets"])
    width = measured["targets"][0]["nodes"][0]
    assert width["pixel_result"]["width"]["value"] == pytest.approx(24, abs=0.15)
    assert width["physical_result"]["width"]["value"] == pytest.approx(2.4, abs=0.015)
    assert width["pixel_result"]["width"]["frame"] == "source"
    assert width["physical_result"]["width"]["unit"] == "mm"


@pytest.mark.parametrize(
    "change,code",
    [
        ("duplicate", "duplicate_id"),
        ("disabled", "invalid_binding"),
        ("array", "invalid_binding"),
        ("cycle", "cycle"),
        ("unknown_port", "invalid_binding"),
        ("frame", "invalid_binding"),
    ],
)
def test_validation_rejects_unsafe_dependency_configuration(client, change, code):
    supplied = recipe()
    if change == "duplicate":
        supplied["nodes"].append(supplied["nodes"][1])
    elif change == "disabled":
        supplied["nodes"][1]["enabled"] = False
    elif change == "array":
        supplied["nodes"][0]["inputs"]["value"]["port"] = "pairs"
    elif change == "unknown_port":
        supplied["nodes"][0]["inputs"]["value"]["port"] = "label"
    elif change == "cycle":
        supplied["nodes"][1] = {
            "id": "width",
            "operation": "evaluations.ranges",
            "parameters": {"maximum": 21},
            "inputs": {"value": {"node": "check", "port": "value"}},
        }
    else:
        supplied["nodes"][1]["parameters"]["frame"] = "unrelated-camera"
        supplied["nodes"][0] = {
            "id": "check",
            "operation": "geometry.point-distances",
            "parameters": {},
            "inputs": {
                "point_a": {"node": "width", "port": "edge_a"},
                "point_b": {"node": "width", "port": "edge_b"},
            },
        }
    checked = result(client, "/recipe-validations", {"recipe": supplied})
    assert not checked["valid"]
    assert code in [issue["code"] for issue in checked["issues"]]


def test_invalid_dependency_preserves_invalid_policy_and_fresh_results(client, tmp_path):
    good = result(client, "/runs", {"recipe": recipe(), "image": IMAGE})
    assert good["valid"]
    cv2.imwrite(str(tmp_path / "stripes.png"), np.zeros((100, 160), np.uint8))
    supplied = recipe()
    supplied["nodes"][0]["parameters"]["invalid_policy"] = "fail"
    invalid = result(client, "/runs", {"recipe": supplied, "image": IMAGE})
    assert not invalid["valid"]
    assert invalid["targets"][0]["nodes"][0]["result"]["width"] is None
    assert invalid["targets"][0]["nodes"][1]["result"]["value"] is None
    assert invalid["evaluation_status"] == "fail"


def test_recipe_file_batch_keeps_independent_image_failures_and_one_manifest(client, tmp_path):
    import json

    (tmp_path / "recipe.json").write_text(json.dumps(recipe()))
    (tmp_path / "broken.png").write_bytes(b"not an image")
    measured = result(
        client,
        "/run-batches",
        {
            "recipe_file": {"root": "exchange", "path": "recipe.json"},
            "items": [
                {"id": "good", "image": IMAGE},
                {"id": "bad", "image": {"root": "exchange", "path": "broken.png"}},
                {"id": "missing", "image": {"root": "exchange", "path": "absent.png"}},
            ],
            "output": {"directory": {"root": "exchange", "path": "batch-output"}},
        },
    )
    assert not measured["valid"]
    assert [item["state"] for item in measured["items"]] == ["succeeded", "failed", "failed"]
    assert measured["items"][0]["result"]["valid"]
    assert measured["manifest"]["kind"] == "manifest"
    assert measured["items"][0]["result"]["sources"][0]["sha256"]
    assert "__stages__" not in str(measured)


def test_physical_composition_and_inclusive_evaluation_bind_calibrated_ports(client):
    supplied = recipe()
    supplied["required_outputs"][0]["space"] = "physical"
    supplied["nodes"][0] = {
        "id": "distance",
        "operation": "geometry.point-distances",
        "inputs": {
            "point_a": {"node": "width", "port": "edge_a", "space": "physical"},
            "point_b": {"node": "width", "port": "edge_b", "space": "physical"},
        },
    }
    supplied["nodes"].append(
        {
            "id": "check",
            "operation": "evaluations.ranges",
            "parameters": {"minimum": 2, "maximum": 2, "unit": "mm"},
            "inputs": {"value": {"node": "distance", "port": "value", "space": "physical"}},
        }
    )
    supplied["metric"] = {
        "calibration": scalar_calibration(),
        "applicability": {
            "camera_id": "camera-1",
            "setup_id": "setup-1",
            "frame": "source",
            "image_size": {"width": 160, "height": 100},
        },
    }
    measured = result(client, "/runs", {"image": IMAGE, "recipe": supplied})
    assert measured["valid"], str(measured["targets"])
    assert measured["evaluation_status"] == "pass"
    distance = next(node for node in measured["targets"][0]["nodes"] if node["id"] == "distance")
    assert distance["physical_result"]["value"] == 2
    assert distance["physical_result"]["unit"] == "mm"
    assert distance["pixel_result"] is None


def test_template_provenance_must_match_the_supplied_teaching_reference(client, tmp_path):
    built = result(
        client,
        "/template-builds",
        {
            "image": IMAGE,
            "roi": {"center": {"x": 50, "y": 50}, "width": 41, "height": 31},
            "output": {"directory": {"root": "exchange", "path": "templates"}},
        },
    )
    other = np.zeros((100, 160), np.uint8)
    other[:, 20:30] = 255
    cv2.imwrite(str(tmp_path / "other.png"), other)
    supplied = recipe()
    supplied["reference"] = {"root": "exchange", "path": "other.png"}
    supplied["localization"] = {"template": built["template"]}
    response = client.post("/api/v2/recipe-validations", json={"recipe": supplied})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "recipe.template_provenance_mismatch"


def test_repeated_scaled_targets_are_independent_and_no_matches_follow_policy(client, tmp_path):
    taught = np.zeros((80, 80), np.uint8)
    taught[20:60, 30:50] = 255
    taught[22:29, 15:23] = 120
    taught[52:59, 55:63] = 190
    cv2.imwrite(str(tmp_path / "taught-multi.png"), taught)
    built = result(
        client,
        "/template-builds",
        {
            "image": {"root": "exchange", "path": "taught-multi.png"},
            "roi": {"center": {"x": 40, "y": 40}, "width": 65, "height": 65},
            "output": {"directory": {"root": "exchange", "path": "templates"}},
        },
    )
    inspection = np.zeros((110, 220), np.uint8)
    inspection[10:90, 10:90] = taught
    inspection[5:101, 115:211] = cv2.resize(taught, (96, 96))
    cv2.imwrite(str(tmp_path / "many.png"), inspection)
    supplied = recipe()
    supplied["reference"] = {"root": "exchange", "path": "taught-multi.png"}
    supplied["nodes"][1]["parameters"]["probe"] = {
        "center": {"x": 40, "y": 40},
        "search_half_length": 20,
        "average_half_width": 3,
    }
    supplied["localization"] = {
        "template": built["template"],
        "search": {
            "score_threshold": 0.85,
            "max_matches": 2,
            "scale_min": 1,
            "scale_max": 1.2,
            "scale_step": 0.2,
        },
    }
    supplied["target_policy"] = {"minimum": 2, "maximum": 2, "no_targets": "fail"}
    measured = result(
        client, "/runs", {"recipe": supplied, "image": {"root": "exchange", "path": "many.png"}}
    )
    assert measured["valid"], str(measured["targets"])
    assert len(measured["targets"]) == 2
    widths = sorted(
        target["nodes"][0]["pixel_result"]["width"]["value"] for target in measured["targets"]
    )
    assert widths == pytest.approx([20, 24], abs=0.2)
    assert len({target["target_id"] for target in measured["targets"]}) == 2
    assert measured["evaluation_status"] == "fail"
    cv2.imwrite(str(tmp_path / "many.png"), np.zeros_like(inspection))
    missing = result(
        client, "/runs", {"recipe": supplied, "image": {"root": "exchange", "path": "many.png"}}
    )
    assert missing["status"] == "no_targets"
    assert missing["targets"] == []
    assert not missing["valid"]
    assert missing["evaluation_status"] == "fail"


def test_control_alignment_is_confined_to_each_fixed_roi_and_other_item_continues(client, tmp_path):
    reference = np.zeros((70, 90), np.uint8)
    reference[:, 35:55] = 255
    rng = np.random.default_rng(9)
    controls = []
    for x, y in [(20, 20), (65, 50)]:
        reference[y - 4 : y + 5, x - 4 : x + 5] = rng.integers(0, 255, (9, 9), np.uint8)
    cv2.imwrite(str(tmp_path / "control-reference.png"), reference)
    for x, y in [(20, 20), (65, 50)]:
        built = result(
            client,
            "/template-builds",
            {
                "image": {"root": "exchange", "path": "control-reference.png"},
                "roi": {"center": {"x": x, "y": y}, "width": 9, "height": 9},
                "output": {"directory": {"root": "exchange", "path": "controls"}},
            },
        )
        controls.append(
            {
                "template": built["template"],
                "reference_point": {"x": x, "y": y},
                "search_radius": 100,
                "score_threshold": 0.95,
            }
        )
    inspection = np.zeros((70, 180), np.uint8)
    inspection[:, 90:] = reference
    cv2.imwrite(str(tmp_path / "neighbors.png"), inspection)
    left = recipe()
    left["reference"] = {"root": "exchange", "path": "control-reference.png"}
    left["nodes"][1]["parameters"]["probe"] = {
        "center": {"x": 45, "y": 35},
        "search_half_length": 30,
        "average_half_width": 3,
    }
    left["localization"] = {
        "fixed_roi": {"center": {"x": 44.5, "y": 34.5}, "width": 90, "height": 70}
    }
    left["alignment"] = {
        "width": 90,
        "height": 70,
        "controls": controls,
        "reference_to_patch": {
            "source_frame": "reference",
            "destination_frame": "aligned",
            "type": "identity",
            "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "inverse_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
    }
    import copy

    right = copy.deepcopy(left)
    right["localization"]["fixed_roi"]["center"]["x"] = 134.5
    measured = result(
        client,
        "/run-batches",
        {
            "recipe": left,
            "items": [
                {"id": "left", "image": {"root": "exchange", "path": "neighbors.png"}},
                {
                    "id": "right",
                    "image": {"root": "exchange", "path": "neighbors.png"},
                    "recipe": right,
                },
            ],
        },
    )
    assert measured["items"][0]["result"]["targets"][0]["nodes"] == []
    assert not measured["items"][0]["valid"]
    assert measured["items"][1]["valid"], str(measured)
    assert measured["items"][1]["result"]["targets"][0]["nodes"][0]["result"]["width"][
        "value"
    ] == pytest.approx(20, abs=0.01)


def test_bound_nodes_validate_literal_bounds_and_invalid_business_intervals(client):
    supplied = recipe()
    supplied["nodes"][0]["parameters"].update(minimum=30, maximum=20)
    response = result(client, "/recipe-validations", {"recipe": supplied})
    assert not response["valid"]
    assert any(issue["node"] == "check" for issue in response["issues"])


def test_resolved_asset_content_changes_conflict_on_same_recipe_idempotency_key(client, tmp_path):
    import json

    (tmp_path / "recipe.json").write_text(json.dumps(recipe()))
    payload = {
        "recipe_file": {"root": "exchange", "path": "recipe.json"},
        "image": IMAGE,
        "idempotency_key": "snapshot-1",
    }
    first = client.post("/api/v2/runs", json=payload)
    assert first.status_code == 200
    updated = recipe()
    updated["nodes"][0]["parameters"]["maximum"] = 19
    (tmp_path / "recipe.json").write_text(json.dumps(updated))
    response = client.post("/api/v2/runs", json=payload)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "operation.idempotency_conflict"


def test_image_line_samples_are_refitted_in_plane_units_before_composed_distance(client, tmp_path):
    camera = {
        "kind": "camera",
        "unit": "mm",
        "applicability": {
            "camera_id": "camera-1",
            "setup_id": "setup-1",
            "frame": "source",
            "image_size": {"width": 160, "height": 100},
        },
        "parameters": {
            "camera_matrix": [[100, 0, 80], [0, 100, 50], [0, 0, 1]],
            "distortion": [0, 0, 0, 0, 0],
        },
    }
    objects = [(x, y, 0) for y in range(3) for x in range(4)]
    projected = [{"x": 40 + x * 2, "y": 20 + y * 2} for x, y, _ in objects]
    plane = result(
        client,
        "/calibrations/plane-mappings",
        {
            "unit": "mm",
            "pixel_size": 0.5,
            "views": [
                {
                    "camera": {"value": camera},
                    "applicability": camera["applicability"],
                    "capture_id": "plane-1",
                    "observation": {
                        "view_id": "plane-1",
                        "image_size": camera["applicability"]["image_size"],
                        "ordering": "explicit_correspondences",
                        "object_points": [{"x": x, "y": y, "z": z} for x, y, z in objects],
                        "image_points": projected,
                    },
                }
            ],
        },
    )
    pixels = np.zeros((100, 160), np.uint8)
    pixels[40:] = 255
    cv2.imwrite(str(tmp_path / "line-plane.png"), pixels)
    supplied = {
        "id": "plane-part",
        "version": "1",
        "metric": {
            "calibration": {"value": plane["asset"]},
            "applicability": camera["applicability"],
        },
        "nodes": [
            {
                "id": "line",
                "operation": "measurements.lines",
                "parameters": {
                    "start": {"x": 20, "y": 40},
                    "end": {"x": 140, "y": 40},
                    "probe_count": 8,
                    "search_half_length": 10,
                    "average_half_width": 2,
                    "polarity": "positive",
                },
            },
            {
                "id": "distance",
                "operation": "geometry.point-line-distances",
                "parameters": {
                    "point": {
                        "valid": True,
                        "frame": "physical",
                        "unit": "mm",
                        "point": {"x": 30, "y": 15},
                    }
                },
                "inputs": {"line": {"node": "line", "port": "line", "space": "physical"}},
            },
        ],
        "required_outputs": [{"node": "distance", "port": "value", "space": "physical"}],
    }
    measured = result(
        client,
        "/runs",
        {"image": {"root": "exchange", "path": "line-plane.png"}, "recipe": supplied},
    )
    assert measured["valid"], str(measured["targets"])
    line = measured["targets"][0]["nodes"][0]["physical_result"]
    assert len(line["samples"]) == 8
    assert line["samples"][0]["y"] == pytest.approx(9.75, abs=0.01)
    assert line["diagnostics"]["rms_error"] < 1e-5
    assert measured["targets"][0]["nodes"][1]["result"]["value"] == pytest.approx(5.25, abs=0.01)


def test_projected_circle_uses_mapped_edge_samples_for_physical_fit(client, tmp_path):
    applicability = {
        "camera_id": "camera-1",
        "setup_id": "setup-1",
        "frame": "source",
        "image_size": {"width": 240, "height": 200},
    }
    matrix = np.array([[220, 0, 120], [0, 220, 100], [0, 0, 1]], np.float64)
    rotation, translation = np.array([0.6, 0.2, 0.1]), np.array([0.0, 0.0, 100.0])
    camera = {
        "kind": "camera",
        "unit": "mm",
        "applicability": applicability,
        "parameters": {"camera_matrix": matrix.tolist(), "distortion": [0, 0, 0, 0, 0]},
    }
    objects = np.array([(x, y, 0) for y in [-10, 0, 10] for x in [-10, 0, 10]], np.float64)
    positions = cv2.projectPoints(objects, rotation, translation, matrix, np.zeros(5))[0].reshape(
        -1, 2
    )
    plane = result(
        client,
        "/calibrations/plane-mappings",
        {
            "unit": "mm",
            "pixel_size": 0.5,
            "views": [
                {
                    "camera": {"value": camera},
                    "applicability": applicability,
                    "capture_id": "circle-plane",
                    "observation": {
                        "view_id": "circle-plane",
                        "image_size": applicability["image_size"],
                        "ordering": "explicit_correspondences",
                        "object_points": [{"x": x, "y": y, "z": z} for x, y, z in objects],
                        "image_points": [{"x": x, "y": y} for x, y in positions],
                    },
                }
            ],
        },
    )
    angles = np.linspace(0, 2 * np.pi, 1000)
    boundary = np.column_stack([10 * np.cos(angles), 10 * np.sin(angles), np.zeros(len(angles))])
    projected = cv2.projectPoints(boundary, rotation, translation, matrix, np.zeros(5))[0].reshape(
        -1, 2
    )
    image = np.zeros((200, 240), np.uint8)
    cv2.fillPoly(image, [np.rint(projected).astype(np.int32)], 255)
    cv2.imwrite(str(tmp_path / "ellipse.png"), image)
    supplied = {
        "id": "circle-part",
        "version": "1",
        "metric": {"calibration": {"value": plane["asset"]}, "applicability": applicability},
        "nodes": [
            {
                "id": "circle",
                "operation": "measurements.circles",
                "parameters": {
                    "center": {"x": 120, "y": 100},
                    "radius": 20,
                    "radius_min": 14,
                    "radius_max": 27,
                    "search_half_length": 9,
                    "average_half_width": 1,
                    "probe_count": 48,
                },
            }
        ],
    }
    measured = result(
        client, "/runs", {"image": {"root": "exchange", "path": "ellipse.png"}, "recipe": supplied}
    )
    assert measured["valid"], str(measured["targets"])
    circle = measured["targets"][0]["nodes"][0]
    assert circle["physical_result"]["circle"]["radius"] == pytest.approx(10, abs=0.3)
    assert circle["physical_result"]["diagnostics"]["rms_error"] < 0.2
    assert circle["pixel_result"]["diagnostics"]["rms_error"] > 0.8


def test_declared_undistortion_recovers_original_pixels_and_commits_only_final_result(client):
    app = {
        "camera_id": "camera-1",
        "setup_id": "setup-1",
        "frame": "source",
        "image_size": {"width": 160, "height": 100},
    }
    camera = {
        "kind": "camera",
        "unit": "mm",
        "applicability": app,
        "parameters": {
            "camera_matrix": [[100, 0, 80], [0, 100, 50], [0, 0, 1]],
            "distortion": [0, 0, 0, 0, 0],
        },
    }
    supplied = recipe()
    supplied["stages"] = [
        {
            "id": "correction",
            "operation": "image-undistortions",
            "parameters": {"calibration": {"value": camera}, "applicability": app},
        }
    ]
    measured = result(
        client,
        "/runs",
        {
            "image": IMAGE,
            "recipe": supplied,
            "output": {"directory": {"root": "exchange", "path": "final"}},
        },
    )
    assert measured["valid"], str(measured["targets"])
    assert measured["targets"][0]["nodes"][0]["pixel_result"]["width"]["value"] == pytest.approx(
        20, abs=0.1
    )
    assert measured["targets"][0]["metric_mapping"]["type"] == "nonlinear"
    assert "correction_ms" in measured["stage_timings"]
    assert measured["manifest"]["kind"] == "manifest"
    assert "__stages__" not in str(measured)


@pytest.mark.parametrize("retained", [False, True])
def test_declared_stitch_measurement_returns_canvas_pixels_and_plane_dimensions(
    client, tmp_path, retained
):
    cameras, views = [], []
    for camera_id, offset in [("A", 40), ("B", 60)]:
        app = {
            "camera_id": camera_id,
            "setup_id": "setup-1",
            "frame": "source",
            "image_size": {"width": 160, "height": 100},
        }
        camera = {
            "kind": "camera",
            "unit": "mm",
            "applicability": app,
            "parameters": {
                "camera_matrix": [[100, 0, 80], [0, 100, 50], [0, 0, 1]],
                "distortion": [0, 0, 0, 0, 0],
            },
        }
        cameras.append(camera)
        views.append(
            {
                "camera": {"value": camera},
                "applicability": app,
                "capture_id": "board-1",
                "observation": {
                    "view_id": "board-1",
                    "image_size": app["image_size"],
                    "ordering": "explicit_correspondences",
                    "object_points": [{"x": x, "y": y, "z": 0} for y in range(3) for x in range(4)],
                    "image_points": [
                        {"x": offset + x * 2, "y": 20 + y * 2} for y in range(3) for x in range(4)
                    ],
                },
            }
        )
    plane = result(
        client, "/calibrations/plane-mappings", {"unit": "mm", "pixel_size": 0.5, "views": views}
    )
    images = []
    for camera, stripe in zip(cameras, [40, 60]):
        camera_id = camera["applicability"]["camera_id"]
        pixels = np.zeros((100, 160), np.uint8)
        pixels[:, stripe : stripe + 20] = 255
        filename = camera_id + ".png"
        cv2.imwrite(str(tmp_path / filename), pixels)
        images.append(
            {
                "camera_id": camera_id,
                "capture_id": "product-1",
                "applicability": camera["applicability"],
                "image": {"root": "exchange", "path": filename},
            }
        )
    supplied = recipe()
    supplied["source_frame"] = "plane_canvas"
    supplied["nodes"][1]["parameters"]["probe"]["center"]["x"] = 70
    supplied["stages"] = [
        {
            "id": "stitch",
            "operation": "image-stitches",
            "parameters": {"calibration": {"value": plane["asset"]}, "images": images},
        }
    ]
    supplied["metric"] = {
        "calibration": {"value": plane["asset"]},
        "applicability": cameras[0]["applicability"],
    }
    payload = {"image": images[0]["image"], "recipe": supplied}
    if retained:
        payload.update(
            retain_stage_images=True,
            output={"directory": {"root": "exchange", "path": "canvas-output"}},
        )
    measured = result(client, "/runs", payload)
    assert measured["valid"], str(measured["targets"])
    pair = measured["targets"][0]["nodes"][0]
    assert pair["pixel_result"]["width"]["value"] == pytest.approx(20, abs=0.01)
    assert pair["pixel_result"]["frame"] == "plane_canvas"
    assert pair["physical_result"]["width"]["value"] == pytest.approx(10, abs=0.01)
    assert "__stages__" not in str(measured)
    if retained:
        import json

        snapshot_file = next(
            artifact["file"]
            for artifact in measured["artifacts"]
            if artifact["kind"] == "recipe_result"
        )
        snapshot = json.loads((tmp_path / snapshot_file["path"]).read_text())
        assert snapshot["measurement_image"] == measured["measurement_image"]
        assert snapshot["measurement_image"]["root"] == "exchange"
        assert len(snapshot["measurement_image"]["sha256"]) == 64
        assert snapshot["image_stages"] == measured["image_stages"]
        assert "__stages__" not in str(snapshot)
        inspected = result(client, "/image-inspections", {"image": snapshot["measurement_image"]})
        assert inspected["metadata"]["width"] == snapshot["image_stages"][-1]["metadata"]["width"]
        canvas_pair = result(
            client,
            "/measurements/edge-pairs",
            {
                "image": snapshot["measurement_image"],
                "frame": "plane_canvas",
                **supplied["nodes"][1]["parameters"],
            },
        )
        assert canvas_pair["width"]["value"] == pytest.approx(20, abs=0.01)

    scalar = scalar_calibration()
    scalar["value"]["applicability"].update(
        camera_id="A", frame="plane_canvas",
        image_size=plane["asset"]["parameters"]["canvas"]["image_size"],
    )
    scalar["value"]["parameters"].update(scale_per_px=0.5, spacing_px=2)
    supplied["metric"] = {
        "calibration": scalar, "applicability": scalar["value"]["applicability"]
    }
    scalar_run = result(client, "/runs", {"recipe": supplied, "image": images[0]["image"]})
    assert scalar_run["valid"], scalar_run
    assert scalar_run["targets"][0]["nodes"][0]["physical_result"]["width"]["value"] == (
        pytest.approx(10, abs=0.01)
    )
    scalar["value"]["applicability"]["frame"] = "source"
    scalar_mismatch = result(client, "/recipe-validations", {"recipe": supplied})
    assert not scalar_mismatch["valid"]
    supplied["metric"]["applicability"] = cameras[0]["applicability"]
    coarser = result(
        client, "/calibrations/plane-mappings", {"unit": "mm", "pixel_size": 1.0, "views": views}
    )
    supplied["metric"]["calibration"] = {"value": coarser["asset"]}
    mismatched = result(client, "/recipe-validations", {"recipe": supplied})
    assert not mismatched["valid"]
    assert any(issue["code"] == "metric_invalid" for issue in mismatched["issues"])
    rejected = client.post("/api/v2/runs", json={"recipe": supplied, "image": images[0]["image"]})
    assert rejected.status_code == 422, rejected.text


def test_node_exception_does_not_block_invalid_policy_or_unrelated_nodes(client):
    supplied = recipe()
    supplied["nodes"][1]["parameters"]["probe"]["center"]["x"] = 1
    supplied["nodes"][0]["parameters"]["invalid_policy"] = "fail"
    supplied["nodes"].append(
        {
            "id": "unrelated",
            "operation": "geometry.point-distances",
            "parameters": {
                "point_a": {"valid": True, "point": {"x": 0, "y": 0}},
                "point_b": {"valid": True, "point": {"x": 3, "y": 4}},
            },
        }
    )
    measured = result(client, "/runs", {"recipe": supplied, "image": IMAGE})
    assert not measured["valid"]
    by_id = {node["id"]: node for node in measured["targets"][0]["nodes"]}
    assert by_id["width"]["state"] == "failed"
    assert by_id["check"]["state"] == "succeeded"
    assert by_id["check"]["result"]["evaluation_status"] == "fail"
    assert by_id["unrelated"]["result"]["value"] == 5


def test_requested_stage_image_is_retained_with_parent_manifest(client, tmp_path):
    app = {
        "camera_id": "camera-1",
        "setup_id": "setup-1",
        "frame": "source",
        "image_size": {"width": 160, "height": 100},
    }
    camera = {
        "kind": "camera",
        "unit": "mm",
        "applicability": app,
        "parameters": {
            "camera_matrix": [[100, 0, 80], [0, 100, 50], [0, 0, 1]],
            "distortion": [0, 0, 0, 0, 0],
        },
    }
    supplied = recipe()
    supplied["stages"] = [
        {
            "id": "correct",
            "operation": "image-undistortions",
            "parameters": {"calibration": {"value": camera}, "applicability": app},
        }
    ]
    measured = result(
        client,
        "/runs",
        {
            "image": IMAGE,
            "recipe": supplied,
            "retain_stage_images": True,
            "output": {"directory": {"root": "exchange", "path": "retained"}},
        },
    )
    retained = next(a for a in measured["artifacts"] if a["kind"] == "stage_image")
    verified = result(client, "/image-inspections", {"image": retained["file"]})
    assert verified["metadata"]["width"] == 160
    assert len(measured["artifacts"]) == 2
    assert "__stages__" not in str(measured)


def test_numeric_array_ports_are_not_scalar_parameters(client):
    supplied = recipe()
    supplied["nodes"][0] = {
        "id": "profile",
        "operation": "edge-profiles",
        "parameters": {
            "probe": {
                "center": {"x": 50, "y": 50},
                "search_half_length": 25,
                "average_half_width": 3,
            }
        },
    }
    supplied["nodes"][1]["inputs"] = {"sigma": {"node": "profile", "port": "gray"}}
    supplied["required_outputs"] = []
    checked = result(client, "/recipe-validations", {"recipe": supplied})
    assert not checked["valid"]
    assert "invalid_binding" in [issue["code"] for issue in checked["issues"]]


def test_one_target_control_failure_does_not_reuse_neighbor_or_stop_other_target(client, tmp_path):
    reference = np.zeros((70, 90), np.uint8)
    reference[:, 35:55] = 255
    rng = np.random.default_rng(18)
    for x, y in [(20, 20), (65, 50)]:
        reference[y - 4 : y + 5, x - 4 : x + 5] = rng.integers(0, 255, (9, 9), np.uint8)
    source = {"root": "exchange", "path": "target-reference.png"}
    cv2.imwrite(str(tmp_path / source["path"]), reference)
    parent = result(
        client,
        "/template-builds",
        {
            "image": source,
            "roi": {"center": {"x": 44.5, "y": 34.5}, "width": 90, "height": 70},
            "output": {"directory": {"root": "exchange", "path": "teaching"}},
        },
    )
    controls = []
    for x, y in [(20, 20), (65, 50)]:
        built = result(
            client,
            "/template-builds",
            {
                "image": source,
                "roi": {"center": {"x": x, "y": y}, "width": 9, "height": 9},
                "output": {"directory": {"root": "exchange", "path": "teaching"}},
            },
        )
        controls.append(
            {
                "template": built["template"],
                "reference_point": {"x": x, "y": y},
                "search_radius": 100,
                "score_threshold": 0.95,
            }
        )
    inspection = np.zeros((70, 190), np.uint8)
    inspection[:, :90] = reference
    inspection[:, 100:] = reference
    inspection[46:55, 61:70] = 0
    cv2.imwrite(str(tmp_path / "targets.png"), inspection)
    supplied = recipe()
    supplied["reference"] = source
    supplied["localization"] = {
        "template": parent["template"],
        "search": {"max_matches": 2, "score_threshold": 0.8},
    }
    supplied["alignment"] = {
        "width": 90,
        "height": 70,
        "controls": controls,
        "reference_to_patch": {
            "source_frame": "reference",
            "destination_frame": "aligned",
            "type": "identity",
            "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "inverse_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
    }
    supplied["nodes"][1]["parameters"]["probe"] = {
        "center": {"x": 45, "y": 35},
        "search_half_length": 30,
        "average_half_width": 3,
    }
    measured = result(
        client, "/runs", {"recipe": supplied, "image": {"root": "exchange", "path": "targets.png"}}
    )
    assert len(measured["targets"]) == 2
    assert sorted(target["valid"] for target in measured["targets"]) == [False, True]
    good = next(target for target in measured["targets"] if target["valid"])
    failed = next(target for target in measured["targets"] if not target["valid"])
    assert failed["nodes"] == []
    assert good["nodes"][0]["pixel_result"]["width"]["value"] == pytest.approx(20, abs=0.05)


@pytest.mark.parametrize("route", ["recipe-validations", "recipe-previews", "runs"])
def test_invalid_dynamic_parameter_has_a_field_diagnostic(client, route):
    supplied = recipe()
    supplied["nodes"][1]["parameters"]["selection"] = "banana"
    payload = {"recipe": supplied}
    if route != "recipe-validations":
        payload["image"] = IMAGE
    if route == "recipe-previews":
        payload.update(selected_nodes=["width"], upstream_revision="invalid-1",
                       preview_token="invalid-1")
    response = client.post("/api/v2/" + route, json=payload)
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "recipe.invalid_parameter"
    assert error["field_path"] == "recipe.nodes.1.parameters.selection"
    assert error["details"]["errors"][0]["type"] == "literal_error"


def test_template_point_pixel_and_physical_results_can_be_supplied_for_render_and_export(
    client, tmp_path
):
    pixels = np.zeros((100, 160), np.uint8)
    pixels[30:50, 40:100] = np.random.default_rng(123).integers(
        20, 230, (20, 60), dtype=np.uint8
    )
    cv2.imwrite(str(tmp_path / "feature.png"), pixels)
    image = {"root": "exchange", "path": "feature.png"}
    output = {"directory": {"root": "exchange", "path": "feature-output"}}
    taught = result(client, "/template-builds", {
        "image": image,
        "roi": {"center": {"x": 69.5, "y": 39.5}, "width": 60, "height": 20},
        "anchor": {"x": 45, "y": 35}, "output": output,
    })
    supplied = {
        "id": "feature", "version": "1", "reference": image,
        "metric": {"calibration": scalar_calibration(),
                   "applicability": scalar_calibration()["value"]["applicability"]},
        "nodes": [{"id": "anchor", "operation": "measurements.template-points",
                   "parameters": {"template": taught["template"],
                                  "search": {"score_threshold": 0.95}}}],
    }
    run = result(client, "/runs", {"recipe": supplied, "image": image})
    assert run["valid"]
    node = run["targets"][0]["nodes"][0]
    assert node["pixel_result"]["point"]["point"] == pytest.approx({"x": 45, "y": 35}, abs=0.02)
    assert node["physical_result"]["point"]["point"] == pytest.approx(
        {"x": 4.5, "y": 3.5}, abs=0.002
    )
    for space, display_frame, expected, raster in [
        ("pixel", "source", {"x": 45, "y": 35}, True),
        ("physical", "physical", {"x": 4.5, "y": 3.5}, False),
        ("physical", "source", {"x": 45, "y": 35}, True),
    ]:
        request = {"result": run, "space": space, "display_frame": display_frame}
        vectors = result(client, "/visualizations", request)
        points = [overlay for overlay in vectors["overlays"] if overlay["type"] == "point"]
        assert len(points) == 1
        assert set(points[0]["geometry"]) == {"x", "y"}
        assert points[0]["geometry"] == pytest.approx(expected, abs=0.02)
        assert points[0]["frame"] == display_frame
        assert points[0]["unit"] == ("mm" if display_frame == "physical" else "px")
        assert points[0]["valid"]
        if raster:
            rendered = result(client, "/visualizations", {
                **request, "image": image, "raster": True, "output": output,
            })
            artifact = next(a for a in rendered["artifacts"] if a["kind"] == "raster_overlay")
            overlay_pixels = cv2.imread(str(tmp_path / artifact["file"]["path"]))
            assert overlay_pixels.shape == (100, 160, 3)
            assert overlay_pixels[35, 45].tolist() == [0, 255, 0]
            assert np.any(overlay_pixels != cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR))
    exported = result(client, "/result-exports", {"result": run, "output": output})
    assert exported["manifest"]
    assert node["pixel_result"]["localization_diagnostics"]["sampling_frame"] == "aligned"
    assert node["physical_result"]["localization_diagnostics"] == (
        node["pixel_result"]["localization_diagnostics"]
    )


@pytest.mark.parametrize("source_frame", ["source", "corrected-source"])
def test_metric_rejects_an_unrelated_frame_and_accepts_the_actual_source_frame(
    client, source_frame
):
    supplied = recipe()
    supplied["source_frame"] = source_frame
    calibration = scalar_calibration()
    calibration["value"]["applicability"]["frame"] = "other-cropped-frame"
    supplied["metric"] = {
        "calibration": calibration,
        "applicability": calibration["value"]["applicability"],
    }
    validated = result(client, "/recipe-validations", {"recipe": supplied})
    assert not validated["valid"]
    assert any(issue["code"] == "metric_invalid" for issue in validated["issues"])
    rejected = client.post("/api/v2/runs", json={"recipe": supplied, "image": IMAGE})
    assert rejected.status_code == 422, rejected.text
    calibration["value"]["applicability"]["frame"] = source_frame
    measured = result(client, "/runs", {"recipe": supplied, "image": IMAGE})
    assert measured["valid"], measured
    width = measured["targets"][0]["nodes"][0]
    assert width["pixel_result"]["frame"] == source_frame
    assert width["physical_result"]["width"]["value"] == pytest.approx(2)

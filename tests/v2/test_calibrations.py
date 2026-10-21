"""Calibration editor acceptance through the public HTTP computation contract."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from measure_api.v2.app import create_app
from measure_api.v2.models import ServiceSettings


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(ServiceSettings(
        shared_roots={"exchange": tmp_path}, max_workers=1,
    ))) as api:
        yield api


def calculate(client, route, payload):
    response = client.post("/api/v2" + route, json=payload)
    assert response.status_code == 200, response.text
    return response.json()["result"]


def applicability():
    return {"camera_id": "camera-A", "setup_id": "fixture-1",
            "image_size": {"width": 640, "height": 480}, "frame": "source"}


def manual_scale(client):
    return calculate(client, "/calibrations/scales", {
        "corners": [{"x": 30 + x * 40, "y": 30 + y * 40}
                    for y in range(4) for x in range(5)],
        "square_size": 2, "unit": "mm", "applicability": applicability(),
    })


def test_manually_edited_checkerboard_creates_explicit_scale_and_length(client):
    result = manual_scale(client)
    assert result["valid"]
    assert result["asset"]["parameters"]["scale_per_px"] == pytest.approx(0.05)
    converted = calculate(client, "/geometry/length-conversions", {
        "calibration": {"value": result["asset"]}, "length_px": 120,
        "applicability": applicability(), "target_unit": "cm",
    })
    assert converted["pixel_length"] == 120
    assert converted["physical_length"] == pytest.approx(0.6)
    assert converted["unit"] == "cm"
    assert converted["calibration_sha256"] == result["sha256"]


def test_detect_checkerboard_and_dot_grid_in_full_source_coordinates(client, tmp_path):
    board = np.full((300, 400), 220, np.uint8)
    for row in range(6):
        for col in range(7):
            board[40 + row * 30:70 + row * 30, 50 + col * 30:80 + col * 30] = (
                255 if (row + col) % 2 else 0
            )
    cv2.imwrite(str(tmp_path / "board.png"), board)
    result = calculate(client, "/calibration-features/checkerboard-corners", {
        "image": {"root": "exchange", "path": "board.png"},
        "grid": {"columns": 6, "rows": 5},
        "roi": {"x": 45, "y": 35, "width": 225, "height": 195},
    })
    assert result["valid"]
    assert len(result["corners"]) == 30
    assert min(p["x"] for p in result["corners"]) == pytest.approx(79.5, abs=1)
    assert min(p["y"] for p in result["corners"]) == pytest.approx(69.5, abs=1)
    assert result["diagnostics"]["spacing_px"] == pytest.approx(30, abs=1)
    dots = np.full((300, 400), 255, np.uint8)
    for row in range(4):
        for col in range(5):
            cv2.circle(dots, (80 + 40 * col, 70 + 40 * row), 9, 0, -1)
    cv2.imwrite(str(tmp_path / "dots.png"), dots)
    result = calculate(client, "/calibration-features/dot-grids", {
        "image": {"root": "exchange", "path": "dots.png"},
        "grid": {"columns": 5, "rows": 4}, "spacing": 2, "unit": "mm",
    })
    assert result["valid"]
    assert len(result["image_points"]) == 20
    assert result["ordering"] == "row_major"
    assert result["object_points"][6] == {"x": 2, "y": 2, "z": 0}
    assert result["image_points"][6] == pytest.approx({"x": 120, "y": 110}, abs=0.1)


def camera_observations():
    matrix = np.array([[820, 0, 320], [0, 800, 240], [0, 0, 1]], np.float64)
    objects = np.array([(x * 20, y * 20, 0) for y in range(5) for x in range(6)], np.float32)
    observations = []
    for index, rotation in enumerate([[0.15, 0.2, 0.1], [-0.25, 0.1, 0],
                                      [0.2, -0.2, 0.1], [0.1, 0.35, -0.1],
                                      [-0.1, -0.3, 0.2], [0.3, 0.15, -0.1]]):
        image, _ = cv2.projectPoints(objects, np.array(rotation),
                                     np.array([-40, -35, 450 + index * 20], np.float64),
                                     matrix, np.zeros(5))
        observations.append({
            "view_id": f"pose-{index}", "image_size": {"width": 640, "height": 480},
            "image_points": [{"x": float(p[0]), "y": float(p[1])}
                             for p in image.reshape(-1, 2)],
            "object_points": [{"x": float(p[0]), "y": float(p[1]), "z": 0} for p in objects],
            "ordering": "row_major",
        })
    return observations


def test_camera_from_accepted_observations_reports_each_view_and_cannot_scale_lengths(client):
    observations = camera_observations()
    rejected = {**observations[0], "view_id": "incomplete", "image_points": []}
    result = calculate(client, "/calibrations/cameras", {
        "observations": observations + [rejected], "unit": "mm",
        "applicability": applicability(),
    })
    assert result["valid"]
    parameters = result["asset"]["parameters"]
    assert parameters["camera_matrix"][0][0] == pytest.approx(820, abs=0.2)
    assert parameters["camera_matrix"][1][1] == pytest.approx(800, abs=0.2)
    assert parameters["reprojection_error_px"] < 0.001
    assert len(result["diagnostics"]["accepted_views"]) == 6
    assert result["diagnostics"]["rejected_views"][0]["view_id"] == "incomplete"
    assert len(result["diagnostics"]["accepted_views"][0]["residuals_px"]) == 30
    response = client.post("/api/v2/geometry/length-conversions", json={
        "calibration": {"value": result["asset"]}, "length_px": 120,
        "applicability": applicability(),
    })
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "calibration.scale_required"


def test_calibration_roundtrip_is_upstream_owned_and_rejects_wrong_setup(client, tmp_path):
    scale = manual_scale(client)
    exported = calculate(client, "/calibration-exports", {
        "calibration": {"value": scale["asset"]},
        "output": {"directory": {"root": "exchange", "path": "calibrations"}},
    })
    descriptor = exported["artifacts"][0]
    assert descriptor["sha256"] == scale["sha256"]
    original = (tmp_path / descriptor["file"]["path"]).read_bytes()
    imported = calculate(client, "/calibration-imports", {
        "calibration": {"file": descriptor["file"]},
    })
    assert imported["asset"] == scale["asset"]
    assert imported["sha256"] == scale["sha256"]
    inspected = calculate(client, "/calibration-inspections", {
        "calibration": {"file": descriptor["file"]},
    })
    assert inspected["diagnostics"]["physical_scale_available"]
    validated = calculate(client, "/calibration-validations", {
        "calibration": {"value": scale["asset"]},
        "applicability": {**applicability(), "camera_id": "other-camera"},
    })
    assert not validated["valid"]
    assert validated["diagnostics"]["code"] == "calibration.applicability_mismatch"
    assert (tmp_path / descriptor["file"]["path"]).read_bytes() == original


def test_undistortion_commits_nonlinear_mapping_with_camera_and_frame_provenance(client, tmp_path):
    calibration = calculate(client, "/calibrations/cameras", {
        "observations": camera_observations(), "unit": "mm",
        "applicability": applicability(),
    })["asset"]
    image = np.zeros((480, 640, 3), np.uint8)
    image[100:200, 100:200] = [30, 120, 240]
    cv2.imwrite(str(tmp_path / "image.png"), image)
    result = calculate(client, "/image-undistortions", {
        "image": {"root": "exchange", "path": "image.png"},
        "calibration": {"value": calibration}, "applicability": applicability(),
        "destination_frame": "corrected-camera-A",
        "output": {"directory": {"root": "exchange", "path": "corrected"}},
    })
    assert result["mapping"]["type"] == "nonlinear"
    assert result["mapping"]["source_frame"] == "source"
    assert result["mapping"]["destination_frame"] == "corrected-camera-A"
    assert result["metadata"]["frame"] == "corrected-camera-A"
    artifacts = {artifact["kind"]: artifact for artifact in result["artifacts"]}
    output = cv2.imread(str(tmp_path / artifacts["undistorted"]["file"]["path"]))
    assert output.shape == image.shape
    assert output[150, 150].tolist() == [30, 120, 240]
    with np.load(tmp_path / artifacts["mapping"]["file"]["path"], allow_pickle=False) as maps:
        assert maps["source_x"].shape == (480, 640)
        assert maps["source_x"][240, 320] == pytest.approx(320, abs=0.01)
        assert maps["source_y"][240, 320] == pytest.approx(240, abs=0.01)
    assert result["mapping"]["details"]["inverse_map_direction"] == "destination_to_source"


def test_image_camera_calibration_rejects_views_individually_and_invalid_corners_do_not_scale(
    client, tmp_path,
):
    matrix = np.array([[820, 0, 320], [0, 800, 240], [0, 0, 1]], np.float64)
    objects = np.array([(x * 20, y * 20, 0) for y in range(5) for x in range(6)], np.float32)
    views = []
    for index, rotation in enumerate([[0.15, 0.2, 0.1], [-0.25, 0.1, 0],
                                      [0.2, -0.2, 0.1], [0.1, 0.35, -0.1]]):
        points, _ = cv2.projectPoints(objects, np.array(rotation),
                                      np.array([-40, -35, 450 + index * 20], np.float64),
                                      matrix, np.zeros(5))
        image = np.full((480, 640), 255, np.uint8)
        for point in points.reshape(-1, 2):
            cv2.circle(image, tuple(np.rint(point).astype(int)), 6, 0, -1)
        name = f"view-{index}.png"
        cv2.imwrite(str(tmp_path / name), image)
        views.append({"view_id": str(index), "image": {"root": "exchange", "path": name}})
    cv2.imwrite(str(tmp_path / "blank.png"), np.zeros((480, 640), np.uint8))
    result = calculate(client, "/calibrations/cameras", {
        "images": views + [{"view_id": "blank", "image": {
            "root": "exchange", "path": "blank.png",
        }}], "grid": {"columns": 6, "rows": 5}, "spacing": 20,
        "unit": "mm", "applicability": applicability(),
    })
    assert result["valid"]
    assert len(result["diagnostics"]["accepted_views"]) == 4
    assert len(result["diagnostics"]["accepted_views"][0]["source"]["sha256"]) == 64
    assert result["diagnostics"]["rejected_views"][0]["view_id"] == "blank"
    assert result["asset"]["parameters"]["reprojection_error_px"] < 0.5
    invalid = calculate(client, "/calibrations/scales", {
        "corners": [{"x": 0, "y": 0}], "square_size": 2,
        "applicability": applicability(),
    })
    assert not invalid["valid"]
    assert invalid["asset"] is None


def test_scale_outputs_and_invalid_asset_diagnostics_are_complete(client, tmp_path):
    request = {
        "corners": [{"x": 30 + x * 40, "y": 30 + y * 40}
                    for y in range(4) for x in range(5)],
        "square_size": 2, "applicability": applicability(),
        "output": {"directory": {"root": "exchange", "path": "scales"}},
    }
    result = calculate(client, "/calibrations/scales", request)
    assert result["artifacts"][0]["sha256"] == result["sha256"]
    assert result["artifacts"][0]["size_bytes"] == result["size_bytes"]
    duplicate = calculate(client, "/calibrations/scales", {
        **request, "corners": [{"x": 30, "y": 30}] * 8,
    })
    assert not duplicate["valid"]
    invalid = {**result["asset"], "parameters": {
        **result["asset"]["parameters"], "scale_per_px": -1,
    }}
    checked = calculate(client, "/calibration-validations", {"calibration": {"value": invalid}})
    assert not checked["valid"]
    assert checked["diagnostics"]["code"] == "calibration.invalid_asset"
    assert not list((tmp_path / "scales").rglob("*.partial"))


def test_camera_requires_distinct_usable_views_and_undistortion_checks_actual_dimensions(
    client, tmp_path,
):
    observation = camera_observations()[0]
    result = calculate(client, "/calibrations/cameras", {
        "observations": [{**observation, "view_id": str(i)} for i in range(3)],
        "applicability": applicability(),
    })
    assert not result["valid"]
    assert len(result["diagnostics"]["rejected_views"]) == 2
    asset = {
        "kind": "camera", "unit": "mm", "applicability": applicability(),
        "parameters": {"camera_matrix": [[820, 0, 320], [0, 800, 240], [0, 0, 1]],
                       "distortion": [0.1, -0.05, 0, 0, 0]},
    }
    cv2.imwrite(str(tmp_path / "wrong.png"), np.zeros((20, 30), np.uint8))
    response = client.post("/api/v2/image-undistortions", json={
        "image": {"root": "exchange", "path": "wrong.png"},
        "calibration": {"value": asset}, "applicability": applicability(),
        "output": {"directory": {"root": "exchange", "path": "bad-correction"}},
    })
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "calibration.image_size_mismatch"
    assert not (tmp_path / "bad-correction").exists()


def test_nonzero_distortion_mapping_agrees_with_corrected_point_geometry(client, tmp_path):
    image = np.full((480, 640), 80, np.uint8)
    cv2.imwrite(str(tmp_path / "distorted.png"), image)
    asset = {
        "kind": "camera", "unit": "mm", "applicability": applicability(),
        "parameters": {"camera_matrix": [[820, 0, 320], [0, 800, 240], [0, 0, 1]],
                       "distortion": [0.15, -0.02, 0.001, -0.001, 0]},
    }
    result = calculate(client, "/image-undistortions", {
        "image": {"root": "exchange", "path": "distorted.png"},
        "calibration": {"value": asset}, "applicability": applicability(), "alpha": 1,
        "output": {"directory": {"root": "exchange", "path": "nonlinear"}},
    })
    details = result["mapping"]["details"]
    with np.load(tmp_path / result["artifacts"][1]["file"]["path"], allow_pickle=False) as maps:
        source = np.array([[[maps["source_x"][80, 100], maps["source_y"][80, 100]]]], np.float64)
    corrected = cv2.undistortPoints(source, np.array(details["camera_matrix"], np.float64),
                                     np.array(details["distortion"], np.float64),
                                     P=np.array(details["corrected_camera_matrix"], np.float64))
    assert corrected.reshape(-1) == pytest.approx([100, 80], abs=0.02)
    assert source.reshape(-1) != pytest.approx([100, 80], abs=1)


def test_openapi_exposes_typed_scale_camera_parameters_and_all_calibration_routes(client):
    schema = client.get("/api/v2/openapi.json").json()
    models = schema["components"]["schemas"]
    assert "ScalarParameters" in models
    assert "CameraParameters" in models
    assert "camera_matrix" in models["CameraParameters"]["properties"]
    for route in ["calibration-features/checkerboard-corners", "calibration-features/dot-grids",
                  "calibrations/scales", "calibrations/cameras", "calibration-validations",
                  "calibration-inspections", "calibration-imports", "calibration-exports",
                  "image-undistortions", "geometry/length-conversions"]:
        assert f"/api/v2/{route}" in schema["paths"]


def test_calibration_validation_rejects_inconsistent_scale_and_nonfinite_intrinsics(client):
    scale = manual_scale(client)["asset"]
    inconsistent = {**scale, "parameters": {
        **scale["parameters"], "scale_per_px": 100,
    }}
    result = calculate(client, "/calibration-validations", {
        "calibration": {"value": inconsistent},
    })
    assert not result["valid"]
    bad_camera = {
        "kind": "camera", "unit": "mm", "applicability": applicability(),
        "parameters": {"camera_matrix": [[-820, 0, 320], [0, 800, 240], [0, 0, 1]],
                       "distortion": [0, 0, 0, 0, 0]},
    }
    result = calculate(client, "/calibration-validations", {
        "calibration": {"value": bad_camera},
    })
    assert not result["valid"]
    assert result["diagnostics"]["code"] == "calibration.invalid_asset"

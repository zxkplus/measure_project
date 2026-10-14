"""Plane and stereo workflows exercised through real public HTTP requests."""

import copy
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from measure_api.v2.app import create_app
from measure_api.v2.models import ServiceSettings


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(ServiceSettings(shared_roots={"exchange": tmp_path},
                                              max_workers=1))) as api:
        yield api


def calculate(client, route, payload):
    response = client.post("/api/v2" + route, json=payload)
    assert response.status_code == 200, response.text
    return response.json()["result"]


def app(camera="A"):
    return {"camera_id": camera, "setup_id": "fixture", "frame": "source",
            "image_size": {"width": 100, "height": 80}}


def camera(camera_id="A", distortion=None):
    return {"kind": "camera", "unit": "mm", "applicability": app(camera_id),
            "parameters": {"camera_matrix": [[100, 0, 50], [0, 100, 40], [0, 0, 1]],
                           "distortion": distortion or [0, 0, 0, 0, 0]}}


def observation(camera_id="A", pose=0, capture="capture-0"):
    obj = np.array([(x * 5, y * 5, 0) for y in range(4) for x in range(5)], np.float64)
    rotation = np.array([pose * 0.08, -pose * 0.05, pose * 0.03], np.float64)
    translation = np.array([-10 + (10 if camera_id == "B" else 0), -8, 100 + pose * 10],
                           np.float64)
    img, _ = cv2.projectPoints(obj, rotation, translation,
                              np.array(camera()["parameters"]["camera_matrix"], np.float64),
                              np.zeros(5))
    return {"view_id": capture, "image_size": app()["image_size"],
            "object_points": [{"x": float(x), "y": float(y), "z": float(z)} for x, y, z in obj],
            "image_points": [{"x": float(x), "y": float(y)} for x, y in img.reshape(-1, 2)],
            "ordering": "explicit_correspondences"}


def plane_request():
    return {"unit": "mm", "pixel_size": 1,
            "views": [{"camera": {"value": camera(cam)}, "applicability": app(cam),
                       "capture_id": "capture-0", "observation": observation(cam)}
                      for cam in ["A", "B"]]}


def test_plane_calibration_exposes_physical_and_canvas_mappings_and_residuals(client):
    result = calculate(client, "/calibrations/plane-mappings", plane_request())
    assert result["valid"]
    parameters = result["asset"]["parameters"]
    assert parameters["canvas"]["image_size"] == {"width": 110, "height": 80}
    assert parameters["canvas"]["pixel_size"] == 1
    assert parameters["canvas"]["unit"] == "mm"
    assert parameters["views"][0]["camera_to_plane"][0][2] == pytest.approx(-40, abs=1e-5)
    assert parameters["views"][1]["camera_to_plane"][0][2] == pytest.approx(-50, abs=1e-5)
    assert result["diagnostics"]["mean_error_px"] < 1e-5
    assert len(result["diagnostics"]["views"][0]["residuals_px"]) == 20
    assert parameters["views"][0]["mapping"]["direction"] == "source_to_destination"
    assert parameters["views"][0]["camera_sha256"]


def stereo_request():
    return {"camera_a": {"value": camera("A")}, "camera_b": {"value": camera("B")},
            "applicability_a": app("A"), "applicability_b": app("B"),
            "unit": "mm", "pixel_size": 1,
            "pairs": [{"capture_id": f"capture-{i}",
                       "a": {"capture_id": f"capture-{i}",
                             "observation": observation("A", i, f"capture-{i}")},
                       "b": {"capture_id": f"capture-{i}",
                             "observation": observation("B", i, f"capture-{i}")}}
                      for i in range(4)]}


def test_stereo_pairs_establish_explicit_a_to_b_pose_and_rejection_diagnostics(client):
    result = calculate(client, "/calibrations/stereo-rigs", stereo_request())
    assert result["valid"]
    parameters = result["asset"]["parameters"]
    assert parameters["direction"] == "camera_a_to_camera_b"
    assert parameters["translation"] == pytest.approx([10, 0, 0], abs=0.001)
    assert np.array(parameters["rotation"]) == pytest.approx(np.eye(3), abs=0.00001)
    assert parameters["baseline"] == pytest.approx(10, abs=0.001)
    assert result["diagnostics"]["mean_error_px"] < 0.001
    assert len(result["diagnostics"]["accepted_pairs"]) == 4
    assert len(result["diagnostics"]["accepted_pairs"][0]["a_from_b_residuals_px"]) == 20
    assert len(parameters["provenance"]["capture_ids"]) == 4
    request = stereo_request()
    request["pairs"][0]["b"]["capture_id"] = "other-capture"
    response = client.post("/api/v2/calibrations/stereo-rigs", json=request)
    assert response.status_code == 422
    duplicate = stereo_request()
    duplicate["pairs"] = [duplicate["pairs"][0]] * 3
    response = client.post("/api/v2/calibrations/stereo-rigs", json=duplicate)
    assert response.status_code == 422


def test_origin_adjustment_is_a_new_asset_and_validation_reports_real_residuals(client, tmp_path):
    plane = calculate(client, "/calibrations/plane-mappings", plane_request())
    exported = calculate(client, "/calibration-exports", {
        "calibration": {"value": plane["asset"]},
        "output": {"directory": {"root": "exchange", "path": "calibrations"}},
    })
    file = exported["artifacts"][0]["file"]
    original = (tmp_path / file["path"]).read_bytes()
    shifted = calculate(client, "/calibration-origin-adjustments", {
        "calibration": {"file": file}, "origin": {"x": 5, "y": 3}, "unit": "mm",
        "output": {"directory": {"root": "exchange", "path": "shifted"}},
    })
    assert shifted["sha256"] != plane["sha256"]
    assert shifted["asset"]["parameters"]["views"][0]["camera_to_plane"][0][2] == pytest.approx(-45)
    assert shifted["asset"]["parameters"]["origin"] == {"x": 5, "y": 3}
    assert shifted["asset"]["parameters"]["provenance"]["parent_sha256"] == plane["sha256"]
    assert shifted["artifacts"][0]["sha256"] == shifted["sha256"]
    assert (tmp_path / file["path"]).read_bytes() == original
    observed = {**observation(), "camera_id": "A"}
    observed["image_points"][0]["x"] += 4
    checked = calculate(client, "/calibration-validations", {
        "calibration": {"value": plane["asset"]}, "observations": [observed], "unit": "mm",
        "max_error_px": 1,
    })
    assert not checked["valid"]
    assert checked["diagnostics"]["views"][0]["residuals_px"][0] == pytest.approx(4, abs=0.0001)
    assert checked["diagnostics"]["max_error_px"] == pytest.approx(4, abs=0.0001)
    schema = client.get("/api/v2/openapi.json").json()["components"]["schemas"]
    assert "PlaneParameters" in schema
    assert "StereoParameters" in schema


def test_camera_frame_transform_and_projection_are_explicit_and_unit_correct(client):
    stereo = calculate(client, "/calibrations/stereo-rigs", stereo_request())
    forward = calculate(client, "/geometry/stereo-transforms", {
        "calibration": {"value": stereo["asset"]}, "source_camera": "A",
        "destination_camera": "B", "points": [{"x": 0, "y": 0, "z": 0.1}],
        "unit": "m", "target_unit": "mm",
    })
    assert forward["points"][0] == pytest.approx({"x": 10, "y": 0, "z": 100}, abs=0.001)
    assert forward["source_frame"] == "camera:A"
    assert forward["destination_frame"] == "camera:B"
    reverse = calculate(client, "/geometry/stereo-transforms", {
        "calibration": {"value": stereo["asset"]}, "source_camera": "B",
        "destination_camera": "A", "points": forward["points"], "unit": "mm",
        "target_unit": "cm",
    })
    assert reverse["points"][0] == pytest.approx({"x": 0, "y": 0, "z": 10}, abs=0.0001)
    projected = calculate(client, "/geometry/camera-projections", {
        "calibration": {"value": camera()}, "applicability": app(),
        "points": [{"x": 1, "y": 2, "z": 10}], "unit": "cm", "source_frame": "camera:A",
    })
    assert projected["points"][0] == pytest.approx({"x": 60, "y": 60})
    assert projected["unit"] == "px"
    response = client.post("/api/v2/geometry/camera-projections", json={
        "calibration": {"value": camera()}, "applicability": app(),
        "points": [{"x": 1, "y": 2, "z": 0}], "unit": "mm", "source_frame": "camera:A",
    })
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "geometry.behind_camera"


def test_calibrated_pair_stitch_commits_canvas_masks_and_preserves_sources(client, tmp_path):
    plane = calculate(client, "/calibrations/plane-mappings", plane_request())
    images = []
    for cam, value in [("A", 20), ("B", 80)]:
        cv2.imwrite(str(tmp_path / f"{cam}.png"), np.full((80, 100), value, np.uint8))
        images.append({"camera_id": cam, "applicability": app(cam), "capture_id": "product-1",
                       "image": {"root": "exchange", "path": f"{cam}.png"}})
    result = calculate(client, "/image-stitches", {
        "calibration": {"value": plane["asset"]}, "images": images, "blend": "average",
        "output": {"directory": {"root": "exchange", "path": "stitched"}},
    })
    artifacts = {a["kind"]: a for a in result["artifacts"]}
    canvas = cv2.imread(str(tmp_path / artifacts["stitched"]["file"]["path"]), 0)
    assert canvas.shape == (80, 110)
    assert canvas[40, 5] == 80
    assert canvas[40, 15] == 50
    assert canvas[40, 105] == 20
    coverage = cv2.imread(str(tmp_path / artifacts["coverage"]["file"]["path"]), 0)
    assert coverage[40, 5] == 1
    assert coverage[40, 15] == 2
    assert result["metadata"]["frame"] == "plane_canvas"
    assert result["sources"][0]["sha256"]
    assert result["mappings"][0]["source_frame"] == "source"
    assert result["mappings"][0]["destination_frame"] == "plane_canvas"
    manifest = json.loads((tmp_path / result["manifest"]["file"]["path"]).read_text())
    assert len(manifest["files"]) >= 4
    assert cv2.imread(str(tmp_path / "A.png"), 0)[40, 15] == 20
    images[1]["capture_id"] = "wrong-capture"
    response = client.post("/api/v2/image-stitches", json={
        "calibration": {"value": plane["asset"]}, "images": images,
        "output": {"directory": {"root": "exchange", "path": "bad"}},
    })
    assert response.status_code == 422
    assert not (tmp_path / "bad").exists()


def test_portable_assets_reject_inconsistent_pose_and_mappings_before_stitch(client):
    plane = calculate(client, "/calibrations/plane-mappings", plane_request())["asset"]
    inconsistent = copy.deepcopy(plane)
    inconsistent["parameters"]["views"][0]["board_to_camera_translation"][0] += 20
    checked = calculate(client, "/calibration-validations", {
        "calibration": {"value": inconsistent},
    })
    assert not checked["valid"]
    assert checked["diagnostics"]["code"] == "calibration.invalid_asset"
    inconsistent = copy.deepcopy(plane)
    inconsistent["parameters"]["views"][0]["mapping"]["matrix"][0][2] += 5
    checked = calculate(client, "/calibration-validations", {
        "calibration": {"value": inconsistent},
    })
    assert not checked["valid"]


def test_plane_images_detect_shared_grid_and_blank_board_is_a_domain_outcome(client, tmp_path):
    cv2.imwrite(str(tmp_path / "blank.png"), np.full((80, 100), 255, np.uint8))
    result = calculate(client, "/calibrations/plane-mappings", {
        "views": [{"camera": {"value": camera()}, "applicability": app(),
                   "capture_id": "blank", "image": {"root": "exchange", "path": "blank.png"}}],
        "grid": {"columns": 5, "rows": 4}, "spacing": 5, "unit": "mm", "pixel_size": 1,
    })
    assert not result["valid"]
    assert result["asset"] is None
    assert result["diagnostics"]["reason"] == "calibration.board_not_found"


def test_stereo_calibrates_real_shared_image_pairs_and_exports_portable_pose(client, tmp_path):
    matrix = np.array([[820, 0, 320], [0, 800, 240], [0, 0, 1]], np.float64)
    objects = np.array([(x * 20, y * 20, 0) for y in range(5) for x in range(6)], np.float64)
    cameras = {}
    for cam in ["A", "B"]:
        value = camera(cam)
        value["applicability"]["image_size"] = {"width": 640, "height": 480}
        value["parameters"]["camera_matrix"] = matrix.tolist()
        cameras[cam] = value
    pairs = []
    for index, rotation in enumerate([[0.15, 0.2, 0.1], [-0.25, 0.1, 0],
                                      [0.2, -0.2, 0.1], [0.1, 0.35, -0.1]]):
        pair = {"capture_id": f"image-{index}"}
        for cam in ["A", "B"]:
            pixels = cv2.projectPoints(objects, np.array(rotation),
                                       np.array([-40 + (40 if cam == "B" else 0), -35,
                                                 450 + index * 20], np.float64),
                                       matrix, np.zeros(5))[0].reshape(-1, 2)
            image = np.full((480, 640), 255, np.uint8)
            for point in pixels:
                cv2.circle(image, tuple(np.rint(point).astype(int)), 6, 0, -1)
            name = f"{cam}-{index}.png"
            cv2.imwrite(str(tmp_path / name), image)
            pair[cam.lower()] = {"capture_id": pair["capture_id"],
                                 "image": {"root": "exchange", "path": name}}
        pairs.append(pair)
    request = {"camera_a": {"value": cameras["A"]}, "camera_b": {"value": cameras["B"]},
               "applicability_a": cameras["A"]["applicability"],
               "applicability_b": cameras["B"]["applicability"],
               "grid": {"columns": 6, "rows": 5}, "spacing": 20, "unit": "mm",
               "pixel_size": 1, "pairs": pairs,
               "output": {"directory": {"root": "exchange", "path": "stereo-assets"}}}
    result = calculate(client, "/calibrations/stereo-rigs", request)
    assert result["valid"]
    assert result["asset"]["parameters"]["baseline"] == pytest.approx(40, abs=3)
    assert result["diagnostics"]["mean_error_px"] < 1
    assert result["diagnostics"]["accepted_pairs"][0]["a_source"]["sha256"]
    imported = calculate(client, "/calibration-imports", {
        "calibration": {"file": result["artifacts"][0]["file"]},
    })
    assert imported["sha256"] == result["sha256"]
    assert imported["asset"] == result["asset"]
    checked = calculate(client, "/calibration-validations", {
        "calibration": {"value": result["asset"]}, "unit": "mm", "observations": [
            {**observation(cam, 0), "camera_id": cam} for cam in ["A"]],
    })
    assert not checked["valid"]
    assert checked["diagnostics"]["code"] == "calibration.unpaired_capture"


def test_stereo_validation_measures_cross_camera_error_and_refinement_is_nonmutating(client):
    request = stereo_request()
    original = copy.deepcopy(request["camera_a"]["value"])
    request["fix_intrinsics"] = False
    result = calculate(client, "/calibrations/stereo-rigs", request)
    assert result["valid"]
    assert result["asset"]["parameters"]["provenance"]["fix_intrinsics"] is False
    assert request["camera_a"]["value"] == original
    observations = [{**observation(cam, 0), "camera_id": cam} for cam in ["A", "B"]]
    checked = calculate(client, "/calibration-validations", {
        "calibration": {"value": result["asset"]}, "unit": "mm", "observations": observations,
    })
    assert checked["valid"]
    assert checked["diagnostics"]["max_error_px"] < 0.01
    assert len(checked["diagnostics"]["pairs"][0]["a_from_b_residuals_px"]) == 20
    observations[1]["image_points"][0]["x"] += 4.4
    checked = calculate(client, "/calibration-validations", {
        "calibration": {"value": result["asset"]}, "unit": "mm", "observations": observations,
        "max_error_px": 1,
    })
    assert not checked["valid"]
    assert checked["diagnostics"]["max_error_px"] > 3


def test_distorted_plane_maps_samples_nonlinearly_and_stitch_uses_declared_correction(
    client, tmp_path,
):
    request = plane_request()
    distortion = [0.1, -0.02, 0.001, -0.001, 0]
    for view in request["views"]:
        view["camera"]["value"]["parameters"]["distortion"] = distortion
        observed = view["observation"]
        cam = view["applicability"]["camera_id"]
        obj = np.array([[p["x"], p["y"], p["z"]] for p in observed["object_points"]])
        pixels = cv2.projectPoints(obj, np.zeros(3),
                                   np.array([-10 + (10 if cam == "B" else 0), -8, 100.],
                                            np.float64),
                                   np.array(camera()["parameters"]["camera_matrix"], np.float64),
                                   np.array(distortion))[0].reshape(-1, 2)
        observed["image_points"] = [{"x": float(x), "y": float(y)} for x, y in pixels]
    plane = calculate(client, "/calibrations/plane-mappings", request)
    assert plane["valid"]
    mapping = plane["asset"]["parameters"]["views"][0]["mapping"]
    assert mapping["type"] == "nonlinear"
    assert mapping["matrix"] is None
    images = []
    for cam in ["A", "B"]:
        cv2.imwrite(str(tmp_path / f"distorted-{cam}.png"), np.full((80, 100), 100, np.uint8))
        images.append({"camera_id": cam, "applicability": app(cam), "capture_id": "distorted",
                       "image": {"root": "exchange", "path": f"distorted-{cam}.png"}})
    stitched = calculate(client, "/image-stitches", {
        "calibration": {"value": plane["asset"]}, "images": images,
        "return_warped_views": True,
        "output": {"directory": {"root": "exchange", "path": "distorted-result"}},
    })
    assert all(m["type"] == "nonlinear" for m in stitched["mappings"])
    assert len(stitched["artifacts"]) == 7
    artifact = next(a for a in stitched["artifacts"] if a["kind"] == "stitched")
    canvas = cv2.imread(str(tmp_path / artifact["file"]["path"]), 0)
    assert canvas[40, 55] == 100


def test_repeated_stereo_observations_and_revised_stereo_origins_remain_traceable(client):
    request = stereo_request()
    for index, pair in enumerate(request["pairs"]):
        for cam in ["a", "b"]:
            pair[cam]["observation"] = observation(cam.upper(), 0, pair["capture_id"])
    result = calculate(client, "/calibrations/stereo-rigs", request)
    assert not result["valid"]
    assert len(result["diagnostics"]["accepted_pairs"]) == 1
    assert len(result["diagnostics"]["rejected_pairs"]) == 3
    stereo = calculate(client, "/calibrations/stereo-rigs", stereo_request())
    shifted = calculate(client, "/calibration-origin-adjustments", {
        "calibration": {"value": stereo["asset"]}, "origin": {"x": 0.5, "y": 0.3},
        "unit": "cm",
    })
    assert shifted["asset"]["kind"] == "stereo"
    assert (shifted["asset"]["parameters"]["translation"]
            == stereo["asset"]["parameters"]["translation"])
    assert shifted["asset"]["parameters"]["origin"] == {"x": 5, "y": 3}
    assert shifted["sha256"] != stereo["sha256"]

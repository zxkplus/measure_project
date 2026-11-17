"""HTTP tests for the stateless gantry pick service (JSON in, JSON out)."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from measure.pick_api import GantryPick
from measure_api.pick_server import PickSettings, create_app

WIDTH = 640
HEIGHT = 480
IMAGE_CENTRE = np.array([(WIDTH - 1) * 0.5, (HEIGHT - 1) * 0.5])
ROBOT_PER_PIXEL = np.array([[0.05, 0.0], [0.0, 0.05]])
ROBOT_ORIGIN = np.array([100.0, 200.0])
CORNER_UV = [(100.0, 100.0), (540.0, 100.0), (100.0, 380.0), (540.0, 380.0)]

REAL_IMAGE_SIZE = [5472, 3648]
REAL_IMAGE_POINTS = [
    [2458.040224337917, 2279.4969844857173],
    [3206.1075863362894, 1334.4446890724007],
    [3308.557276553789, 2183.0556801280945],
    [2356.3462968677204, 1431.227404097654],
]
REAL_ROBOT_POINTS = [
    [10.928000450134277, 321.6470031738281],
    [20.928000450134277, 331.6470031738281],
    [10.928000450134277, 331.6470031738281],
    [20.928000450134277, 321.6470031738281],
]
REAL_MATRIX = [
    [-0.0013213626423343333, -0.011627365736710232, 40.681749989616],
    [0.011604128501496913, -0.0013960386325849925, 296.3037814102695],
    [0.0, 0.0, 1.0],
]
OFFSET_T = (123.67973454793294, 1.4771626790364583)


def _robot_point(uv):
    return (ROBOT_PER_PIXEL @ (np.asarray(uv) - IMAGE_CENTRE)) + ROBOT_ORIGIN


def _settings(tmp_path: Path):
    return PickSettings(
        runs_dir=tmp_path / "runs",
        debug_default=False,
        annotated_max_px=800,
    )


def _post_vision(client, points=None, robot_points=None, image_size=None):
    points = CORNER_UV if points is None else points
    robot_points = (
        [_robot_point(uv).tolist() for uv in points]
        if robot_points is None
        else robot_points
    )
    return client.post(
        "/api/v1/calibrate/vision",
        json={
            "image_size": image_size or [WIDTH, HEIGHT],
            "image_points": [list(p) for p in points],
            "robot_points": robot_points,
        },
    )


def _post_offset(client, samples=None):
    samples = samples or [
        {"place_pose": [108, 184], "center_pose": [100, 180]}
    ]
    return client.post("/api/v1/calibrate/offset", json={"samples": samples})


def _post_locate(
    client,
    image_uv=(300.0, 200.0),
    pose=(90.0, 180.0),
    vision_matrix=None,
    offset_matrix=None,
    image_size=None,
):
    body = {
        "image_size": image_size or [WIDTH, HEIGHT],
        "image_uv": list(image_uv),
        "robot_pose": list(pose),
        "vision_matrix": vision_matrix,
    }
    if offset_matrix is not None:
        body["offset_matrix"] = offset_matrix
    return client.post("/api/v1/locate", json=body)


def _run_dirs(settings):
    if not settings.runs_dir.is_dir():
        return []
    return [
        item
        for item in settings.runs_dir.iterdir()
        if item.is_dir() and not item.name.startswith(".")
    ]


def _manifest(settings, run_id):
    return json.loads(
        (settings.runs_dir / run_id / "manifest.json").read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_debug_off_returns_matrices_and_coordinates(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))

    vision = _post_vision(client)
    assert vision.status_code == 200
    vision_payload = vision.json()
    assert set(vision_payload) == {
        "ok",
        "matrix",
        "rms_px",
        "rms_mm",
        "principal_point",
        "image_size",
        "warnings",
    }
    assert vision_payload["image_size"] == [WIDTH, HEIGHT]
    assert vision_payload["principal_point"] == pytest.approx(
        [(WIDTH - 1) * 0.5, (HEIGHT - 1) * 0.5]
    )
    vision_matrix = vision_payload["matrix"]
    assert np.asarray(vision_matrix).shape == (3, 3)

    offset = _post_offset(client)
    assert offset.status_code == 200
    offset_payload = offset.json()
    assert set(offset_payload) == {
        "ok",
        "matrix",
        "offset_xy",
        "std_xy",
        "num_samples",
        "warnings",
    }
    assert offset_payload["matrix"] == [
        [1.0, 0.0, 8.0],
        [0.0, 1.0, 4.0],
        [0.0, 0.0, 1.0],
    ]

    located = _post_locate(
        client,
        vision_matrix=vision_matrix,
        offset_matrix=offset_payload["matrix"],
    )
    assert located.status_code == 200
    payload = located.json()
    assert set(payload) == {
        "ok",
        "camera_center_xy",
        "gripper_xy",
        "delta_xy",
    }
    assert payload["gripper_xy"] == pytest.approx(
        (
            payload["camera_center_xy"][0] + 8.0,
            payload["camera_center_xy"][1] + 4.0,
        ),
        abs=1e-9,
    )
    assert payload["delta_xy"] == pytest.approx(
        (
            payload["camera_center_xy"][0] - 90.0,
            payload["camera_center_xy"][1] - 180.0,
        ),
        abs=1e-9,
    )
    assert _run_dirs(settings) == []


def test_real_package_reproduces_matrix_over_http(tmp_path):
    """The 2026-11-16 calibration set, replayed through the JSON endpoint."""
    client = TestClient(create_app(_settings(tmp_path)))
    response = client.post(
        "/api/v1/calibrate/vision",
        json={
            "image_size": REAL_IMAGE_SIZE,
            "image_points": REAL_IMAGE_POINTS,
            "robot_points": REAL_ROBOT_POINTS,
        },
    )
    assert response.status_code == 200
    assert np.allclose(response.json()["matrix"], REAL_MATRIX, atol=1e-12)

    located = client.post(
        "/api/v1/locate",
        json={
            "image_size": REAL_IMAGE_SIZE,
            "image_uv": [2836.937, 1806.012],
            "robot_pose": [15.928, 326.647],
            "vision_matrix": REAL_MATRIX,
            "offset_matrix": [
                [1.0, 0.0, OFFSET_T[0]],
                [0.0, 1.0, OFFSET_T[1]],
                [0.0, 0.0, 1.0],
            ],
        },
    )
    assert located.status_code == 200
    assert located.json()["camera_center_xy"] == pytest.approx(
        [15.8587, 325.4455], abs=1e-4
    )
    assert located.json()["gripper_xy"] == pytest.approx(
        [139.5384, 326.9227], abs=1e-4
    )


def test_offset_is_optional_for_locate(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    vision = _post_vision(client).json()

    located = _post_locate(client, vision_matrix=vision["matrix"])
    assert located.status_code == 200
    assert located.json()["gripper_xy"] is None
    assert located.json()["camera_center_xy"]


def test_coordinates_match_core_math(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    vision = _post_vision(client).json()
    offset = _post_offset(client).json()
    response = _post_locate(
        client,
        vision_matrix=vision["matrix"],
        offset_matrix=offset["matrix"],
    )
    expected = GantryPick().locate(
        (300.0, 200.0),
        (90.0, 180.0),
        vision["matrix"],
        offset["matrix"],
        image_size=(WIDTH, HEIGHT),
    )
    assert response.json()["camera_center_xy"] == pytest.approx(
        expected.camera_center_xy, abs=1e-9
    )
    assert response.json()["gripper_xy"] == pytest.approx(
        expected.gripper_xy, abs=1e-9
    )


# ---------------------------------------------------------------------------
# Debug artifacts (no images any more)
# ---------------------------------------------------------------------------


def test_debug_on_writes_point_runs(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    assert client.put("/api/v1/debug", json={"enabled": True}).json()["enabled"]

    vision = _post_vision(client)
    assert vision.status_code == 200
    vision_payload = vision.json()
    assert vision_payload["run_id"]
    assert "overview" not in vision_payload
    vision_run = vision_payload["run_id"]
    manifest = _manifest(settings, vision_run)
    assert manifest["type"] == "vision"
    assert manifest["overview"] is None
    assert manifest["images"] == []
    assert manifest["points"] == "points.json"

    run_dir = settings.runs_dir / vision_run
    assert not (run_dir / "images").exists()
    points = json.loads((run_dir / "points.json").read_text(encoding="utf-8"))
    assert len(points) == 4
    assert set(points[0]) >= {"index", "u", "v", "x", "y", "residual_px"}
    assert client.get(f"/api/v1/runs/{vision_run}").status_code == 200

    offset = _post_offset(client)
    assert offset.status_code == 200
    assert offset.json()["run_id"]
    offset_manifest = _manifest(settings, offset.json()["run_id"])
    assert offset_manifest["type"] == "offset"
    assert offset_manifest["points"] is None

    located = _post_locate(
        client,
        vision_matrix=vision_payload["matrix"],
        offset_matrix=offset.json()["matrix"],
    )
    assert located.status_code == 200
    locate_run = located.json()["run_id"]
    locate_manifest = _manifest(settings, locate_run)
    assert locate_manifest["type"] == "locate"
    assert locate_manifest["images"] == []
    assert locate_manifest["points"] is None
    inputs = json.loads(
        (settings.runs_dir / locate_run / "inputs.json").read_text(encoding="utf-8")
    )
    assert inputs["image_uv"] == [300.0, 200.0]
    assert inputs["image_size"] == [WIDTH, HEIGHT]

    runs = client.get("/api/v1/runs").json()["runs"]
    assert len(runs) == 3
    assert all(item["thumbnail"] is None for item in runs)

    archive = client.get(f"/api/v1/runs/{vision_run}/archive")
    assert archive.status_code == 200
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        names = set(bundle.namelist())
    assert "manifest.json" in names
    assert "inputs.json" in names
    assert "result.json" in names
    assert "points.json" in names
    assert "recipe_snapshot.json" not in names


def test_vision_failure_records_a_failed_run(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    client.put("/api/v1/debug", json={"enabled": True})

    collinear = [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]
    response = _post_vision(client, robot_points=collinear)

    assert response.status_code == 422
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error"] == "calibration_failed"
    assert payload["run_id"]
    manifest = _manifest(settings, payload["run_id"])
    assert manifest["status"] == "failed"


def test_locate_failure_records_a_failed_run(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    client.put("/api/v1/debug", json={"enabled": True})
    _post_vision(client)

    singular = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    response = _post_locate(client, vision_matrix=singular)

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"] == "bad_matrix"
    assert payload["run_id"]
    manifest = _manifest(settings, payload["run_id"])
    assert manifest["status"] == "failed"


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


def test_bad_vision_matrix_is_400(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    response = _post_locate(client, vision_matrix=[[1, 2], [3, 4]])
    assert response.status_code == 400
    assert response.json()["error"] == "bad_matrix"
    assert _run_dirs(settings) == []


def test_missing_vision_matrix_is_400(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    response = client.post(
        "/api/v1/locate",
        json={
            "image_size": [WIDTH, HEIGHT],
            "image_uv": [300.0, 200.0],
            "robot_pose": [90.0, 180.0],
        },
    )
    assert response.status_code == 400
    assert response.json()["error"] == "bad_request"


def test_multipart_images_are_no_longer_accepted(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    response = client.post(
        "/api/v1/calibrate/vision",
        files=[("images", ("a.png", b"not-an-image", "image/png"))],
        data={"robot_points": "[[1, 2]]"},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "bad_request"
    assert _run_dirs(settings) == []


def test_validation_errors_do_not_create_runs(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))

    response = _post_vision(
        client,
        points=CORNER_UV[:2],
        robot_points=[[1, 2], [3, 4]],
    )
    assert response.status_code == 400
    assert response.json()["error"] == "bad_request"
    assert _run_dirs(settings) == []

    # An out-of-image point is rejected before a debug run is opened.
    vision = _post_vision(client).json()
    locate = _post_locate(
        client, image_uv=(700.0, 200.0), vision_matrix=vision["matrix"]
    )
    assert locate.status_code == 400
    assert locate.json()["error"] == "point_out_of_image"
    assert _run_dirs(settings) == []


def test_point_count_mismatch_is_400(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    response = _post_vision(
        client,
        robot_points=[_robot_point(uv).tolist() for uv in CORNER_UV[:3]],
    )
    assert response.status_code == 400
    assert response.json()["error"] == "point_count_mismatch"


def test_row_col_swap_is_rejected(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    swapped = [(v, u) for u, v in CORNER_UV]
    response = _post_vision(client, points=swapped)
    assert response.status_code == 400
    assert response.json()["error"] == "point_out_of_image"


def test_inconsistent_offset_samples_are_422(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    response = _post_offset(
        client,
        samples=[
            {"place_pose": [20.0, 20.0], "center_pose": [20.0, 20.0]},
            {"place_pose": [139.0, 327.0], "center_pose": [15.8, 326.2]},
        ],
    )
    assert response.status_code == 422
    assert response.json()["error"] == "offset_invalid"
    assert "offset_inconsistent" in response.json()["warnings"]


def test_off_switch_stops_new_runs(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    client.put("/api/v1/debug", json={"enabled": True})
    vision = _post_vision(client).json()
    before = len(_run_dirs(settings))

    client.put("/api/v1/debug", json={"enabled": False})
    response = _post_locate(client, vision_matrix=vision["matrix"])
    assert response.status_code == 200
    assert "run_id" not in response.json()
    assert len(_run_dirs(settings)) == before


def test_status_endpoint_is_gone(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    assert client.get("/api/v1/status").status_code == 404

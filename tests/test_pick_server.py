"""HTTP tests for the stateless gantry pick service."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import cv2
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


def _image(centre, radius=45):
    image = np.full((HEIGHT, WIDTH), 210, dtype=np.uint8)
    cv2.circle(
        image,
        (int(round(centre[0])), int(round(centre[1]))),
        radius,
        40,
        -1,
    )
    return cv2.GaussianBlur(image, (0, 0), 1.5)


def _png(image):
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


def _robot_point(uv):
    return (ROBOT_PER_PIXEL @ (np.asarray(uv) - IMAGE_CENTRE)) + ROBOT_ORIGIN


def _settings(tmp_path: Path):
    return PickSettings(
        runs_dir=tmp_path / "runs",
        debug_default=False,
        annotated_max_px=800,
    )


def _upload_vision(client, images=None, points=None):
    images = images or [_image(uv) for uv in CORNER_UV]
    points = points or [_robot_point(uv).tolist() for uv in CORNER_UV]
    files = [
        ("images", (f"{index}.png", _png(image), "image/png"))
        for index, image in enumerate(images)
    ]
    return client.post(
        "/api/v1/calibrate/vision",
        files=files,
        data={"robot_points": json.dumps(points)},
    )


def _upload_offset(client):
    return client.post(
        "/api/v1/calibrate/offset",
        json={"samples": [{"place_pose": [108, 184], "center_pose": [100, 180]}]},
    )


def _upload_locate(
    client,
    image,
    pose=(90.0, 180.0),
    vision_matrix=None,
    offset_matrix=None,
):
    data = {
        "robot_pose": json.dumps(list(pose)),
        "vision_matrix": json.dumps(vision_matrix),
    }
    if offset_matrix is not None:
        data["offset_matrix"] = json.dumps(offset_matrix)
    return client.post(
        "/api/v1/locate",
        files=[("image", ("shot.png", _png(image), "image/png"))],
        data=data,
    )


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


def test_debug_off_returns_matrices_and_coordinates(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))

    vision = _upload_vision(client)
    assert vision.status_code == 200
    vision_payload = vision.json()
    assert set(vision_payload) == {"ok", "matrix"}
    vision_matrix = vision_payload["matrix"]
    assert np.asarray(vision_matrix).shape == (3, 3)

    offset = _upload_offset(client)
    assert offset.status_code == 200
    offset_payload = offset.json()
    assert set(offset_payload) == {"ok", "matrix"}
    offset_matrix = offset_payload["matrix"]
    assert offset_matrix == [
        [1.0, 0.0, 8.0],
        [0.0, 1.0, 4.0],
        [0.0, 0.0, 1.0],
    ]

    located = _upload_locate(
        client,
        _image((300.0, 200.0)),
        vision_matrix=vision_matrix,
        offset_matrix=offset_matrix,
    )
    assert located.status_code == 200
    payload = located.json()
    assert set(payload) == {"ok", "camera_center_xy", "gripper_xy"}
    assert payload["gripper_xy"] == pytest.approx(
        (
            payload["camera_center_xy"][0] + 8.0,
            payload["camera_center_xy"][1] + 4.0,
        ),
        abs=1e-9,
    )
    assert _run_dirs(settings) == []


def test_debug_on_writes_and_serves_artifacts(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    assert client.put("/api/v1/debug", json={"enabled": True}).json()["enabled"]

    vision = _upload_vision(client)
    assert vision.status_code == 200
    vision_payload = vision.json()
    assert "matrix" in vision_payload
    assert vision_payload["run_id"]
    assert "overview" not in vision_payload
    assert "images" not in vision_payload
    vision_run = vision_payload["run_id"]
    manifest = _manifest(settings, vision_run)
    assert manifest["type"] == "vision"
    assert manifest["overview"] == "overview.jpg"
    assert manifest["images"][0]["annotated"]

    assert client.get(f"/api/v1/runs/{vision_run}").status_code == 200
    assert client.get(f"/files/{vision_run}/overview.jpg").status_code == 200
    assert (
        client.get(
            f"/files/{vision_run}/{manifest['images'][0]['annotated']}"
        ).status_code
        == 200
    )

    offset = _upload_offset(client)
    assert offset.status_code == 200
    assert offset.json()["matrix"]
    assert offset.json()["run_id"]
    assert "offset_xy" not in offset.json()

    located = _upload_locate(
        client,
        _image((300.0, 200.0)),
        vision_matrix=vision_payload["matrix"],
        offset_matrix=offset.json()["matrix"],
    )
    assert located.status_code == 200
    locate_payload = located.json()
    assert locate_payload["run_id"]
    assert "annotated" not in locate_payload
    assert "original" not in locate_payload
    locate_run = locate_payload["run_id"]
    locate_manifest = _manifest(settings, locate_run)
    assert locate_manifest["type"] == "locate"
    assert locate_manifest["images"][0]["annotated"]
    assert (
        client.get(
            f"/files/{locate_run}/{locate_manifest['images'][0]['annotated']}"
        ).status_code
        == 200
    )

    runs = client.get("/api/v1/runs").json()["runs"]
    assert len(runs) == 3
    archive = client.get(f"/api/v1/runs/{vision_run}/archive")
    assert archive.status_code == 200
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        names = set(bundle.namelist())
    assert "manifest.json" in names
    assert "inputs.json" in names
    assert "result.json" in names
    assert "recipe_snapshot.json" not in names


def test_offset_is_optional_for_locate(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    vision = _upload_vision(client).json()

    located = _upload_locate(
        client,
        _image((300.0, 200.0)),
        vision_matrix=vision["matrix"],
    )
    assert located.status_code == 200
    assert located.json()["gripper_xy"] is None
    assert located.json()["camera_center_xy"]


def test_locate_failure_saves_annotated_when_debug(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    client.put("/api/v1/debug", json={"enabled": True})
    vision = _upload_vision(client).json()

    blank = np.full((HEIGHT, WIDTH), 210, dtype=np.uint8)
    response = _upload_locate(
        client,
        blank,
        vision_matrix=vision["matrix"],
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["ok"] is False
    assert payload["run_id"]
    manifest = _manifest(settings, payload["run_id"])
    assert manifest["status"] == "failed"
    assert manifest["images"][0]["annotated"]
    assert (
        client.get(
            f"/files/{payload['run_id']}/{manifest['images'][0]['annotated']}"
        ).status_code
        == 200
    )


def test_vision_failure_saves_overview_when_debug(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    client.put("/api/v1/debug", json={"enabled": True})

    blank = np.full((HEIGHT, WIDTH), 210, dtype=np.uint8)
    images = [_image(uv) for uv in CORNER_UV]
    images[2] = blank
    response = _upload_vision(client, images=images)

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"] == "no_target"
    assert payload["image_index"] == 2
    assert payload["run_id"]
    manifest = _manifest(settings, payload["run_id"])
    assert manifest["status"] == "failed"
    assert manifest["overview"] == "overview.jpg"
    assert (
        client.get(f"/files/{payload['run_id']}/overview.jpg").status_code
        == 200
    )


def test_bad_vision_matrix_is_400(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    response = _upload_locate(
        client,
        _image((300.0, 200.0)),
        vision_matrix=[[1, 2], [3, 4]],
    )
    assert response.status_code == 400
    assert response.json()["error"] == "bad_matrix"
    assert _run_dirs(settings) == []


def test_missing_vision_matrix_is_400(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    response = client.post(
        "/api/v1/locate",
        files=[("image", ("shot.png", _png(_image((300.0, 200.0))), "image/png"))],
        data={"robot_pose": "[90, 180]"},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "bad_request"


def test_validation_errors_do_not_create_runs(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))

    response = client.post(
        "/api/v1/calibrate/vision",
        files=[
            ("images", ("a.png", _png(_image(CORNER_UV[0])), "image/png"))
        ],
        data={"robot_points": "[[1, 2]]"},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "bad_request"
    assert _run_dirs(settings) == []


def test_off_switch_stops_new_runs(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    client.put("/api/v1/debug", json={"enabled": True})
    vision = _upload_vision(client).json()
    before = len(_run_dirs(settings))

    client.put("/api/v1/debug", json={"enabled": False})
    response = _upload_locate(
        client,
        _image((300.0, 200.0)),
        vision_matrix=vision["matrix"],
    )
    assert response.status_code == 200
    assert "run_id" not in response.json()
    assert len(_run_dirs(settings)) == before


def test_status_endpoint_is_gone(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    assert client.get("/api/v1/status").status_code == 404


def test_coordinates_match_core_math(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    vision = _upload_vision(client).json()
    offset = _upload_offset(client).json()
    image = _image((300.0, 200.0))
    response = _upload_locate(
        client,
        image,
        vision_matrix=vision["matrix"],
        offset_matrix=offset["matrix"],
    )
    expected = GantryPick().locate(
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
        (90.0, 180.0),
        vision["matrix"],
        offset["matrix"],
    )
    assert response.json()["camera_center_xy"] == pytest.approx(
        expected.camera_center_xy, abs=1e-9
    )
    assert response.json()["gripper_xy"] == pytest.approx(
        expected.gripper_xy, abs=1e-9
    )

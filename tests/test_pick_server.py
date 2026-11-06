"""HTTP tests for the gantry pick service."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

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
        recipe_path=tmp_path / "recipe.json",
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


def _upload_locate(client, image, pose=(90.0, 180.0)):
    return client.post(
        "/api/v1/locate",
        files=[("image", ("shot.png", _png(image), "image/png"))],
        data={"robot_pose": json.dumps(list(pose))},
    )


def _run_dirs(settings):
    if not settings.runs_dir.is_dir():
        return []
    return [
        item
        for item in settings.runs_dir.iterdir()
        if item.is_dir() and not item.name.startswith(".")
    ]


def test_debug_off_keeps_business_only(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))

    vision = _upload_vision(client)
    assert vision.status_code == 200
    payload = vision.json()
    assert payload["ok"] is True
    assert "run_id" not in payload
    assert "overview" not in payload

    offset = client.post(
        "/api/v1/calibrate/offset",
        json={"samples": [{"place_pose": [108, 184], "center_pose": [100, 180]}]},
    )
    assert offset.status_code == 200
    assert offset.json()["offset_xy"] == [8.0, 4.0]
    assert "run_id" not in offset.json()

    locate = _upload_locate(client, _image((300.0, 200.0)))
    assert locate.status_code == 200
    assert locate.json()["gripper_xy"] is not None
    assert "run_id" not in locate.json()

    assert _run_dirs(settings) == []
    status = client.get("/api/v1/status").json()
    assert status["ready"] is True
    assert status["debug"] is False


def test_debug_on_writes_and_serves_artifacts(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    assert client.put("/api/v1/debug", json={"enabled": True}).json()["enabled"]

    vision = _upload_vision(client)
    assert vision.status_code == 200
    vision_payload = vision.json()
    assert vision_payload["ok"] is True
    assert vision_payload["run_id"]
    assert vision_payload["overview"].startswith("/files/")
    assert len(vision_payload["images"]) == 4

    overview = client.get(vision_payload["overview"])
    assert overview.status_code == 200
    assert overview.headers["content-type"].startswith("image/jpeg")

    annotated = client.get(vision_payload["images"][0]["annotated"])
    assert annotated.status_code == 200

    offset = client.post(
        "/api/v1/calibrate/offset",
        json={"samples": [{"place_pose": [108, 184], "center_pose": [100, 180]}]},
    )
    assert offset.status_code == 200
    assert offset.json()["run_id"]

    locate = _upload_locate(client, _image((300.0, 200.0)))
    assert locate.status_code == 200
    locate_payload = locate.json()
    assert locate_payload["run_id"]
    assert locate_payload["gripper_xy"] is not None
    assert client.get(locate_payload["annotated"]).status_code == 200

    runs = client.get("/api/v1/runs").json()["runs"]
    assert len(runs) == 3
    assert all(item["url"].startswith("/api/v1/runs/") for item in runs)

    manifest = client.get(f"/api/v1/runs/{locate_payload['run_id']}")
    assert manifest.status_code == 200
    assert manifest.json()["run"]["type"] == "locate"

    archive = client.get(f"/api/v1/runs/{vision_payload['run_id']}/archive")
    assert archive.status_code == 200
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        names = set(bundle.namelist())
    assert "manifest.json" in names
    assert "inputs.json" in names
    assert "result.json" in names

    status = client.get("/api/v1/status").json()
    assert status["debug"] is True
    assert status["vision"]["run_id"] == vision_payload["run_id"]
    assert status["offset"]["run_id"] == offset.json()["run_id"]


def test_locate_failure_saves_annotated_when_debug(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    client.put("/api/v1/debug", json={"enabled": True})
    assert _upload_vision(client).status_code == 200

    blank = np.full((HEIGHT, WIDTH), 210, dtype=np.uint8)
    response = _upload_locate(client, blank)

    assert response.status_code == 422
    payload = response.json()
    assert payload["ok"] is False
    assert payload["run_id"]
    assert payload["annotated"].startswith("/files/")
    assert client.get(payload["annotated"]).status_code == 200
    assert client.get(f"/api/v1/runs/{payload['run_id']}").json()["run"]["status"] == "failed"


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
    assert payload["overview"].startswith("/files/")
    assert client.get(payload["overview"]).status_code == 200
    assert client.get("/api/v1/status").json()["vision"] is None
    assert not settings.recipe_path.exists()


def test_validation_errors_do_not_create_runs(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))

    response = client.post(
        "/api/v1/calibrate/vision",
        files=[("images", ("a.png", _png(_image(CORNER_UV[0])), "image/png"))],
        data={"robot_points": "[[1, 2]]"},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "bad_request"
    assert _run_dirs(settings) == []


def test_missing_vision_is_409(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    response = _upload_locate(client, _image((300.0, 200.0)))
    assert response.status_code == 409
    assert response.json()["error"] == "vision_not_calibrated"


def test_offset_before_vision_is_409(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    response = client.post(
        "/api/v1/calibrate/offset",
        json={"samples": [{"place_pose": [1, 2], "center_pose": [3, 4]}]},
    )
    assert response.status_code == 409
    assert response.json()["error"] == "vision_not_calibrated"


def test_off_switch_stops_new_runs(tmp_path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    client.put("/api/v1/debug", json={"enabled": True})
    assert _upload_vision(client).status_code == 200
    before = len(_run_dirs(settings))

    client.put("/api/v1/debug", json={"enabled": False})
    response = _upload_locate(client, _image((300.0, 200.0)))
    assert response.status_code == 200
    assert "run_id" not in response.json()
    assert len(_run_dirs(settings)) == before


def test_recipe_survives_app_restart(tmp_path):
    settings = _settings(tmp_path)
    first = TestClient(create_app(settings))
    assert _upload_vision(first).status_code == 200
    assert first.post(
        "/api/v1/calibrate/offset",
        json={"samples": [{"place_pose": [108, 184], "center_pose": [100, 180]}]},
    ).status_code == 200
    assert settings.recipe_path.is_file()

    second = TestClient(create_app(settings))
    status = second.get("/api/v1/status").json()
    assert status["ready"] is True
    locate = _upload_locate(second, _image((300.0, 200.0)))
    assert locate.status_code == 200
    assert locate.json()["gripper_xy"] is not None

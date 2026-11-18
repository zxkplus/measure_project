"""Gantry calibration and location through the existing V2 operation runtime."""

import hashlib
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from measure_api.v2.app import create_app
from measure_api.v2.models import ServiceSettings


IMAGE_CENTER = np.array([319.5, 239.5])
ROBOT_PER_PIXEL = np.array([[0.05, 0.0], [0.0, 0.05]])
ROBOT_ORIGIN = np.array([100.0, 200.0])
CORNERS = [(100, 100), (540, 100), (100, 380), (540, 380)]
VISION_MATRIX = [[0.05, 0.0, 84.025], [0.0, 0.05, 188.025], [0.0, 0.0, 1.0]]
OFFSET_MATRIX = [[1.0, 0.0, 8.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]]


def circle_image(center, shape=(480, 640)):
    image = np.full(shape, 210, np.uint8)
    cv2.circle(image, center, 45, 40, -1)
    return cv2.GaussianBlur(image, (0, 0), 1.5)


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    root = tmp_path_factory.mktemp("gantry-pick")
    for index, point in enumerate(CORNERS):
        cv2.imwrite(str(root / f"corner-{index}.png"), circle_image(point))
    cv2.imwrite(str(root / "shot.png"), circle_image((300, 200)))
    cv2.imwrite(str(root / "blank.png"), np.full((480, 640), 210, np.uint8))
    cv2.imwrite(str(root / "wrong-size.png"), circle_image((150, 150), (300, 400)))
    with TestClient(create_app(ServiceSettings(
        shared_roots={"exchange": root}, writable_roots=[], max_workers=1,
        state_directory=root / "state",
    ))) as client:
        yield client, root


def reference(name):
    return {"root": "exchange", "path": name}


def vision_payload():
    return {
        "images": [reference(f"corner-{index}.png") for index in range(4)],
        "robot_points": [
            (ROBOT_PER_PIXEL @ (np.asarray(point) - IMAGE_CENTER) + ROBOT_ORIGIN).tolist()
            for point in CORNERS
        ],
    }


def calculate(client, route, payload):
    response = client.post("/api/v2" + route, json=payload)
    assert response.status_code == 200, response.text
    operation = response.json()
    assert operation["state"] == "succeeded"
    assert response.headers["X-Trace-Id"] == operation["trace_id"]
    return operation


def test_vision_offset_and_location_form_one_v2_workflow(service):
    client, root = service
    original = (root / "shot.png").read_bytes()
    vision = calculate(client, "/calibrate/vision", vision_payload())
    matrix = vision["result"]["matrix"]
    assert np.asarray(matrix) == pytest.approx(np.asarray(VISION_MATRIX), abs=1e-4)
    assert set(vision["result"]) == {"matrix"}
    offset = calculate(client, "/calibrate/offset", {
        "samples": [
            {"place_pose": [118, 89], "center_pose": [110, 85]},
            {"place_pose": [128, 99], "center_pose": [120, 95]},
        ],
    })
    assert offset["result"]["matrix"] == OFFSET_MATRIX
    located = calculate(client, "/locate", {
        "image": reference("shot.png"), "robot_pose": [90, 180],
        "vision_matrix": matrix, "offset_matrix": offset["result"]["matrix"],
    })
    result = located["result"]
    assert result["camera_center_xy"] == pytest.approx([90.975, 181.975], abs=1e-3)
    assert result["gripper_xy"] == pytest.approx([98.975, 185.975], abs=1e-3)
    assert set(result) == {"camera_center_xy", "gripper_xy", "gripper_delta_xy"}
    assert result["gripper_delta_xy"] == pytest.approx([8.975, 5.975], abs=1e-3)
    assert located["operation"] == "locate"
    assert (root / "shot.png").read_bytes() == original
    assert not (root / "pick_runs").exists()
    assert not (root / "pick_recipe.json").exists()


def test_multi_point_offset_returns_an_affine_transform(service):
    client, _ = service
    centres = np.array([[0.0, 0.0], [200.0, 0.0], [0.0, 150.0], [200.0, 150.0],
                        [100.0, 0.0], [0.0, 75.0], [200.0, 75.0], [100.0, 150.0]])
    theta = np.radians(1.5)
    linear = 1.01 * np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])
    shift = np.array([8.0, 4.0])
    samples = [
        {"center_pose": list(centre), "place_pose": list(linear @ centre + shift)}
        for centre in centres
    ]

    payload = calculate(client, "/calibrate/offset",
                        {"samples": samples, "model": "affine"})["result"]
    assert payload["model"] == "affine"
    assert payload["effective_model"] == "affine"
    assert payload["rotation_deg"] == pytest.approx(1.5, abs=1e-6)
    matrix = np.asarray(payload["matrix"])

    located = calculate(client, "/locate", {
        "image": reference("shot.png"), "robot_pose": [90, 180],
        "vision_matrix": VISION_MATRIX, "offset_matrix": payload["matrix"],
    })["result"]
    camera = located["camera_center_xy"]
    expected = matrix @ np.array([camera[0], camera[1], 1.0])
    assert located["gripper_xy"] == pytest.approx(expected[:2].tolist(), abs=1e-9)
    assert located["gripper_delta_xy"] == pytest.approx(
        (expected[:2] - [90.0, 180.0]).tolist(), abs=1e-9)


def test_mirrored_affine_offset_is_rejected(service):
    client, _ = service
    centres = [[0.0, 0.0], [200.0, 0.0], [0.0, 150.0], [200.0, 150.0]]
    samples = [
        {"center_pose": centre,
         "place_pose": [-centre[0] + 8.0, centre[1] + 4.0]}
        for centre in centres
    ]
    response = client.post("/api/v2/calibrate/offset", json={
        "samples": samples, "model": "affine"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "pick.offset_implausible"


def test_location_ignores_translation_column_and_z_and_offset_is_optional(service):
    client, _ = service
    matrix = [row[:] for row in VISION_MATRIX]
    matrix[0][2] += 1000
    matrix[1][2] -= 500
    result = calculate(client, "/locate", {
        "image": reference("shot.png"), "robot_pose": [90, 180, 500],
        "vision_matrix": matrix,
    })["result"]
    assert result["camera_center_xy"] == pytest.approx([90.975, 181.975], abs=1e-3)
    assert result["gripper_xy"] is None
    offset = calculate(client, "/calibrate/offset", {
        "samples": [{"place_pose": [118, 89, 50], "center_pose": [110, 85, 75]}],
    })
    assert offset["result"]["matrix"] == OFFSET_MATRIX


def test_location_uses_async_query_trace_and_verified_idempotency(service):
    client, root = service
    payload = {
        "image": {**reference("shot.png"),
                  "sha256": hashlib.sha256((root / "shot.png").read_bytes()).hexdigest()},
        "robot_pose": [90, 180], "vision_matrix": VISION_MATRIX,
        "mode": "async", "idempotency_key": "pick-location",
    }
    headers = {"X-Caller-Id": "pick-client", "X-Trace-Id": "pick-trace"}
    accepted = client.post("/api/v2/locate", json=payload, headers=headers)
    assert accepted.status_code in {200, 202}, accepted.text
    identity = accepted.json()["operation_id"]
    if accepted.status_code == 202:
        assert accepted.headers["Location"] == f"/api/v2/operations/{identity}"
    completed = client.get(f"/api/v2/operations/{identity}?wait_ms=10000").json()
    assert completed["state"] == "succeeded"
    assert completed["trace_id"] == "pick-trace"
    assert completed["result"]["camera_center_xy"] == pytest.approx([90.975, 181.975], abs=1e-3)
    retry = client.post("/api/v2/locate", json={**payload, "mode": "sync"}, headers=headers)
    assert retry.status_code == 200, retry.text
    assert retry.json()["operation_id"] == identity
    changed = client.post("/api/v2/locate", json={**payload, "robot_pose": [91, 180]},
                          headers=headers)
    assert changed.status_code == 409
    diagnostics = client.get(f"/api/v2/operations/{identity}/diagnostics").json()
    assert diagnostics["timing_version"] == 3


@pytest.mark.parametrize("change", [
    {"images": [reference("corner-0.png")] * 3},
    {"robot_points": [[0, 0]] * 3},
    {"robot_points": [[0]] * 4},
])
def test_vision_rejects_invalid_counts_and_poses(service, change):
    client, _ = service
    response = client.post("/api/v2/calibrate/vision", json={**vision_payload(), **change})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request.validation"


@pytest.mark.parametrize("route,payload", [
    ("/calibrate/offset", {"samples": []}),
    ("/calibrate/offset", {"samples": [{"place_pose": [0], "center_pose": [0, 0]}]}),
    ("/locate", {"image": reference("shot.png"), "robot_pose": [90, 180]}),
    ("/locate", {"image": reference("shot.png"), "robot_pose": [90, 180],
                 "vision_matrix": [[1, 0], [0, 1]]}),
    ("/locate", {"image": reference("shot.png"), "robot_pose": [90, 180],
                 "vision_matrix": VISION_MATRIX, "offset_matrix": [[1, 0], [0, 1]]}),
])
def test_invalid_requests_use_the_v2_validation_envelope(service, route, payload):
    client, _ = service
    response = client.post("/api/v2" + route, json=payload)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "request.validation"
    assert error["field_path"]
    assert error["trace_id"] == response.headers["X-Trace-Id"]


@pytest.mark.parametrize("field,matrix", [
    ("vision_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 1]]),
    ("vision_matrix", [[1, 0, 0], [0, 1, 0], [1, 0, 1]]),
    ("offset_matrix", [[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
])
def test_invalid_matrix_semantics_are_reported_as_pick_errors(service, field, matrix):
    client, _ = service
    response = client.post("/api/v2/locate", json={
        "image": reference("shot.png"), "robot_pose": [90, 180],
        "vision_matrix": VISION_MATRIX, field: matrix,
    })
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "pick.bad_matrix"
    assert response.json()["operation_id"]


@pytest.mark.parametrize("name", ["blank.png", "wrong-size.png"])
def test_bad_calibration_frames_return_a_typed_failure(service, name):
    client, _ = service
    payload = vision_payload()
    payload["images"][0] = reference(name)
    response = client.post("/api/v2/calibrate/vision", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "pick.calibration_failed"
    assert response.json()["operation_id"]


def test_degenerate_and_poor_quality_calibrations_do_not_return_a_matrix(service):
    client, _ = service
    degenerate = client.post("/api/v2/calibrate/vision", json={
        **vision_payload(), "robot_points": [[index, 0] for index in range(4)],
    })
    assert degenerate.status_code == 422
    assert degenerate.json()["error"]["code"] == "pick.calibration_failed"
    payload = vision_payload()
    payload["robot_points"][-1][1] += 2
    poor_quality = client.post("/api/v2/calibrate/vision", json=payload)
    assert poor_quality.status_code == 422
    assert poor_quality.json()["error"]["code"] == "pick.calibration_invalid"
    assert poor_quality.json()["error"]["details"]["rms_px"] > 1
    assert "result" not in poor_quality.json()


def test_vision_ignores_trailing_z_and_nonfinite_numbers_are_rejected(service):
    client, _ = service
    payload = vision_payload()
    payload["robot_points"] = [point + [100 + index * 10]
                               for index, point in enumerate(payload["robot_points"])]
    vision = calculate(client, "/calibrate/vision", payload)
    assert np.asarray(vision["result"]["matrix"]) == pytest.approx(
        np.asarray(VISION_MATRIX), abs=1e-4,
    )
    response = client.post("/api/v2/locate", content=json.dumps({
        "image": reference("shot.png"), "robot_pose": [float("nan"), 180],
        "vision_matrix": VISION_MATRIX,
    }), headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request.validation"


def test_no_target_and_file_failures_do_not_return_coordinates(service):
    client, _ = service
    payload = {"image": reference("blank.png"), "robot_pose": [90, 180],
               "vision_matrix": VISION_MATRIX}
    missing = client.post("/api/v2/locate", json=payload)
    assert missing.status_code == 422
    assert missing.json()["error"]["code"] == "pick.no_target"
    assert "result" not in missing.json()
    for image, status, code in [
        (reference("absent.png"), 404, "file.not_found"),
        ({**reference("shot.png"), "sha256": "0" * 64}, 409, "file.digest_mismatch"),
        (reference("../outside.png"), 422, "file.outside_roots"),
    ]:
        response = client.post("/api/v2/locate", json={**payload, "image": image})
        assert response.status_code == status, response.text
        assert response.json()["error"]["code"] == code


def test_gantry_operations_are_discovered_and_typed_in_the_existing_v2_app(service):
    client, _ = service
    schema = client.get("/api/v2/openapi.json").json()
    operations = {item["id"]: item for item in client.get("/api/v2/capabilities").json()[
        "operations"
    ]}
    for route, identity in [
        ("/calibrate/vision", "calibrate.vision"),
        ("/calibrate/offset", "calibrate.offset"),
        ("/locate", "locate"),
    ]:
        assert schema["paths"]["/api/v2" + route]["post"]["operationId"] == identity
        assert operations[identity]["family"] == "gantry_pick"
        assert operations[identity]["modes"] == ["sync", "async"]
    properties = schema["components"]["schemas"]["GantryLocateRequest"]["properties"]
    matrix_schema = properties["vision_matrix"]
    assert matrix_schema["minItems"] == matrix_schema["maxItems"] == 3
    assert matrix_schema["items"]["minItems"] == matrix_schema["items"]["maxItems"] == 3
    assert client.get("/api/v2/health/ready").status_code == 200
    assert client.post("/api/v1/locate", json={}).status_code == 404


def test_vision_points_use_supplied_centres_without_image_detection(service):
    from measure.robot_calibration import fit_image_to_robot
    client, _ = service
    payload = {
        "image_points": CORNERS,
        "robot_points": vision_payload()["robot_points"],
        "image_size": [640, 480], "calib_z": 30,
    }
    result = calculate(client, "/calibrate/vision-points", payload)["result"]
    expected = fit_image_to_robot(
        CORNERS, payload["robot_points"], image_size=[640, 480], calib_z=30,
        robust=False,
    )
    assert result["valid"]
    assert np.allclose(result["matrix"], expected.matrix, atol=1e-10)
    assert result["rms_px"] < 1e-8
    assert len(result["residuals"]) == 4
    assert all(r["residual_mm"] < 1e-8 for r in result["residuals"])
    changed = dict(payload, image_points=[(u + 2, v - 3) for u, v in CORNERS])
    other = calculate(client, "/calibrate/vision-points", changed)["result"]
    assert not np.allclose(result["matrix"], other["matrix"])


@pytest.mark.parametrize("patch", [
    {"image_points": [[10, 10]] * 3},
    {"robot_points": [[0, 0]] * 5},
    {"image_size": [1, 480]},
    {"image_points": [[-1, 0]] * 4},
    {"calib_z": "NaN"},
])
def test_vision_points_reject_invalid_input(service, patch):
    client, _ = service
    payload = {"image_points": CORNERS, "robot_points": vision_payload()["robot_points"],
               "image_size": [640, 480], "calib_z": 30, **patch}
    response = client.post("/api/v2/calibrate/vision-points", json=payload)
    assert response.status_code == 422


def test_vision_points_reject_collinear_robot_motion(service):
    client, _ = service
    response = client.post("/api/v2/calibrate/vision-points", json={
        "image_points": CORNERS, "robot_points": [[i, i] for i in range(4)],
        "image_size": [640, 480], "calib_z": 30,
    })
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "pick.calibration_failed"


def test_vision_points_return_rejected_quality_as_candidate_diagnostics(service):
    client, _ = service
    robot_points = vision_payload()["robot_points"]
    robot_points[-1][1] += 2
    result = calculate(client, "/calibrate/vision-points", {
        "image_points": CORNERS, "robot_points": robot_points,
        "image_size": [640, 480], "calib_z": 30,
    })["result"]
    assert result["valid"] is False
    assert result["warnings"]
    assert result["rms_px"] > 1
    assert len(result["residuals"]) == 4
    assert np.isfinite(result["matrix"]).all()

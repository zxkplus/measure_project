"""Upstream image-service acceptance through real HTTP and shared files."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


def test_inspect_shared_image_reports_actual_pixels_and_discovery(tmp_path):
    from measure_api.v2.app import create_app
    from measure_api.v2.models import ServiceSettings

    image = np.zeros((40, 60, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "image.png"), image)
    settings = ServiceSettings(shared_roots={"exchange": tmp_path}, max_workers=1)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v2/image-inspections",
            json={"image": {"root": "exchange", "path": "image.png"}},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["state"] == "succeeded"
        assert body["result"]["metadata"] == {
            "width": 60, "height": 40, "channels": 3,
            "dtype": "uint8", "pixel_format": "BGR8", "frame": "source", "unit": "px",
        }
        assert body["trace_id"] == response.headers["X-Trace-Id"]
        assert client.get("/api/v2/health/ready").status_code == 200
        capabilities = client.get("/api/v2/capabilities").json()
        assert any(item["path"] == "/image-inspections" for item in capabilities["operations"])
        schema = client.get("/api/v2/openapi.json").json()
        assert "/api/v2/image-inspections" in schema["paths"]
        assert client.get("/api/v2/docs").status_code == 200


def test_shared_file_errors_are_structured_and_async_inputs_are_checked(tmp_path):
    from measure_api.v2.app import create_app
    from measure_api.v2.models import ServiceSettings

    import hashlib

    cv2.imwrite(str(tmp_path / "image.png"), np.zeros((12, 20), dtype=np.uint8))
    outside = tmp_path.parent / "outside.png"
    cv2.imwrite(str(outside), np.zeros((12, 20), dtype=np.uint8))
    (tmp_path / "escape.png").symlink_to(outside)
    settings = ServiceSettings(shared_roots={"exchange": tmp_path}, max_workers=1)
    with TestClient(create_app(settings)) as client:
        for reference, expected_status, code in [
            ({"root": "exchange", "path": "missing.png"}, 404, "file.not_found"),
            ({"root": "exchange", "path": "escape.png"}, 422, "file.outside_roots"),
            ({"root": "exchange", "path": "../outside.png"}, 422, "file.outside_roots"),
            ({"root": "exchange", "path": "image.png", "sha256": "0" * 64},
             409, "file.digest_mismatch"),
            ({"root": "exchange", "path": str(tmp_path / "image.png")},
             422, "request.validation"),
            ({"path": "image.png"}, 422, "request.validation"),
        ]:
            response = client.post("/api/v2/image-inspections",
                                   json={"image": reference, "mode": "async"})
            assert response.status_code == expected_status, response.text
            assert response.json()["error"]["code"] == code
            assert response.json()["error"]["trace_id"] == response.headers["X-Trace-Id"]
        absolute = client.post("/api/v2/image-inspections",
                               json={"image": {"path": str(tmp_path / "image.png")}})
        assert absolute.status_code == 200
        expected = hashlib.sha256((tmp_path / "image.png").read_bytes()).hexdigest()
        assert absolute.json()["result"]["source"]["sha256"] == expected
        malformed = client.post("/api/v2/image-inspections", content="{",
                                headers={"Content-Type": "application/json"})
        assert malformed.status_code == 400
        assert malformed.json()["error"]["code"] == "request.invalid_json"


def test_preview_atomically_commits_a_complete_output_manifest(tmp_path):
    from measure_api.v2.app import create_app
    from measure_api.v2.models import ServiceSettings

    import hashlib
    import json

    image = np.full((40, 60, 3), (20, 60, 100), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "image.png"), image)
    settings = ServiceSettings(shared_roots={"exchange": tmp_path}, max_workers=1)
    request = {
        "image": {"root": "exchange", "path": "image.png"},
        "output": {"directory": {"root": "exchange", "path": "outputs"}},
        "max_width": 30, "max_height": 30,
    }
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v2/image-previews", json=request)
        assert response.status_code == 200, response.text
        result = response.json()["result"]
        assert (result["metadata"]["width"], result["metadata"]["height"]) == (30, 20)
        preview = result["artifacts"][0]
        assert preview["kind"] == "preview"
        path = tmp_path / preview["file"]["path"]
        pixels = cv2.imread(str(path))
        assert pixels.shape == (20, 30, 3)
        assert pixels[10, 10].tolist() == [20, 60, 100]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == preview["sha256"]
        manifest = tmp_path / result["manifest"]["file"]["path"]
        marker = json.loads(manifest.read_text())
        assert marker["operation_id"] == response.json()["operation_id"]
        assert marker["files"] == result["artifacts"]
        assert not list((tmp_path / "outputs").rglob("*.partial"))
        repeated = client.post("/api/v2/image-previews", json=request)
        assert repeated.status_code == 200
        assert repeated.json()["result"]["artifacts"][0]["file"] != preview["file"]
        assert np.array_equal(cv2.imread(str(tmp_path / "image.png")), image)


def test_preprocessing_keeps_numeric_data_separate_from_display_preview(tmp_path):
    from measure_api.v2.app import create_app
    from measure_api.v2.models import ServiceSettings

    image = np.zeros((40, 40), dtype=np.uint8)
    image[:, 20:] = 255
    cv2.imwrite(str(tmp_path / "step.png"), image)
    settings = ServiceSettings(shared_roots={"exchange": tmp_path}, max_workers=1)
    with TestClient(create_app(settings)) as client:
        for options, dtype, expected in [
            ({"type": "raw"}, "float32", 255),
            ({"type": "sobel", "kernel_size": 3}, "float32", 1020),
            ({"type": "canny", "low_threshold": 30, "high_threshold": 80}, "uint8", 255),
            ({"type": "clahe", "clip_limit": 2}, "float32", 255),
            ({"type": "threshold", "threshold": 127, "mode": "binary"}, "uint8", 255),
        ]:
            response = client.post("/api/v2/image-preprocessings", json={
                "image": {"root": "exchange", "path": "step.png"},
                "preprocessing": options,
                "output": {"directory": {"root": "exchange", "path": "outputs"}},
            })
            assert response.status_code == 200, response.text
            result = response.json()["result"]
            artifacts = {item["kind"]: item for item in result["artifacts"]}
            numeric = np.load(tmp_path / artifacts["numeric"]["file"]["path"], allow_pickle=False)
            display = cv2.imread(str(tmp_path / artifacts["preview"]["file"]["path"]), 0)
            assert str(numeric.dtype) == dtype
            assert numeric.max() == pytest.approx(expected, abs=0.001)
            assert display.dtype == np.uint8
            assert result["metadata"]["dtype"] == dtype
            assert result["mapping"]["matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_rotated_crop_maps_pixel_centers_in_both_directions(tmp_path):
    from measure_api.v2.app import create_app
    from measure_api.v2.models import ServiceSettings

    image = np.array([[4 * y + x for x in range(40)] for y in range(40)], dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "ramp.png"), image)
    settings = ServiceSettings(shared_roots={"exchange": tmp_path}, max_workers=1)
    with TestClient(create_app(settings)) as client:
        for angle, first_pixel in [(0, 84), (90, 83)]:
            response = client.post("/api/v2/image-crops", json={
                "image": {"root": "exchange", "path": "ramp.png"},
                "roi": {"center": {"x": 10, "y": 20}, "width": 5,
                        "height": 3, "angle_degrees": angle},
                "output": {"directory": {"root": "exchange", "path": "crops"}},
            })
            assert response.status_code == 200, response.text
            result = response.json()["result"]
            pixels = cv2.imread(str(tmp_path / result["artifacts"][0]["file"]["path"]), 0)
            assert pixels.shape == (3, 5)
            assert int(pixels[0, 0]) == first_pixel
            assert int(pixels[1, 2]) == 90
            forward = np.array(result["mapping"]["matrix"])
            inverse = np.array(result["mapping"]["inverse_matrix"])
            assert forward @ [10, 20, 1] == pytest.approx([2, 1, 1])
            assert inverse @ [2, 1, 1] == pytest.approx([10, 20, 1])
            if angle == 90:
                assert inverse @ [3, 1, 1] == pytest.approx([10, 21, 1])
        rejected = client.post("/api/v2/image-crops", json={
            "image": {"root": "exchange", "path": "ramp.png"},
            "roi": {"center": {"x": 0, "y": 0}, "width": 5, "height": 3},
            "output": {"directory": {"root": "exchange", "path": "crops"}},
        })
        assert rejected.status_code == 422
        assert rejected.json()["error"]["code"] == "image.roi_outside"


def test_async_image_work_remains_queryable_and_http_health_stays_responsive(tmp_path):
    from measure_api.v2.app import create_app
    from measure_api.v2.models import ServiceSettings

    import time

    image = np.zeros((2048, 2048), dtype=np.uint8)
    image[:, 1024:] = 100
    cv2.imwrite(str(tmp_path / "large.png"), image)
    settings = ServiceSettings(shared_roots={"exchange": tmp_path}, max_workers=1)
    request = {
        "image": {"root": "exchange", "path": "large.png"},
        "preprocessing": {"type": "sobel"}, "mode": "async",
        "output": {"directory": {"root": "exchange", "path": "outputs"}},
    }
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v2/image-preprocessings", json=request)
        assert response.status_code == 202
        location = response.headers["Location"]
        started = time.monotonic()
        assert client.get("/api/v2/health/live").status_code == 200
        assert time.monotonic() - started < 1.0
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            operation = client.get(location).json()
            if operation["state"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)
        assert operation["state"] == "succeeded", operation
        sync = client.post("/api/v2/image-preprocessings", json={**request, "mode": "sync"})
        assert sync.status_code == 200
        first = operation["result"]["artifacts"][0]
        second = sync.json()["result"]["artifacts"][0]
        assert first["sha256"] == second["sha256"]
        assert client.get(location).json()["result"] == operation["result"]


def test_pixel_format_validation_and_readiness_reflect_real_shared_files(tmp_path):
    from measure_api.v2.app import create_app
    from measure_api.v2.models import ServiceSettings

    for name, image in [("rgba.png", np.zeros((10, 10, 4), np.uint8)),
                        ("deep.png", np.zeros((10, 10), np.uint16))]:
        cv2.imwrite(str(tmp_path / name), image)
    settings = ServiceSettings(shared_roots={"exchange": tmp_path}, max_workers=1)
    with TestClient(create_app(settings)) as client:
        for name in ["rgba.png", "deep.png"]:
            response = client.post("/api/v2/image-inspections",
                                   json={"image": {"root": "exchange", "path": name}})
            assert response.status_code == 422
            assert response.json()["error"]["code"] == "image.unsupported_pixels"
        invalid = client.post("/api/v2/image-inspections", json={
            "image": {"root": "exchange", "path": "deep.png"}, "unexpected": True,
        })
        assert invalid.status_code == 422
        assert invalid.json()["error"]["field_path"] == "body.unexpected"
    absent_settings = ServiceSettings(shared_roots={"exchange": tmp_path / "absent"}, max_workers=1)
    with TestClient(create_app(absent_settings)) as client:
        assert client.get("/api/v2/health/live").status_code == 200
        assert client.get("/api/v2/health/ready").status_code == 503


def test_outputs_cannot_escape_roots_or_replace_upstream_inputs(tmp_path):
    from measure_api.v2.app import create_app
    from measure_api.v2.models import ServiceSettings

    image = np.zeros((10, 10), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "source.png"), image)
    original = (tmp_path / "source.png").read_bytes()
    outside = tmp_path.parent / "output-outside"
    outside.mkdir(exist_ok=True)
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)
    settings = ServiceSettings(shared_roots={"exchange": tmp_path}, max_workers=1)
    with TestClient(create_app(settings)) as client:
        for path, status, code in [("escape", 422, "file.outside_roots"),
                                   ("source.png", 409, "output.conflict")]:
            response = client.post("/api/v2/image-previews", json={
                "image": {"root": "exchange", "path": "source.png"},
                "output": {"directory": {"root": "exchange", "path": path}},
            })
            assert response.status_code == status, response.text
            assert response.json()["error"]["code"] == code
        assert (tmp_path / "source.png").read_bytes() == original
        assert not list(outside.iterdir())


def test_openapi_declares_real_error_and_async_response_contracts(tmp_path):
    from measure_api.v2.app import create_app
    from measure_api.v2.models import ServiceSettings

    settings = ServiceSettings(shared_roots={"exchange": tmp_path}, max_workers=1)
    with TestClient(create_app(settings)) as client:
        schema = client.get("/api/v2/openapi.json").json()
        responses = schema["paths"]["/api/v2/image-inspections"]["post"]["responses"]
        assert responses["422"]["content"]["application/json"]["schema"]["$ref"].endswith(
            "/ErrorResponse"
        )
        assert "content" in responses["202"]
        missing = client.post("/api/v2/image-inspections",
                              json={"image": {"root": "exchange", "path": "absent.png"}})
        assert missing.status_code == 404
        assert set(missing.json()["error"]) == {
            "code", "message", "details", "field_path", "trace_id", "retryable",
        }

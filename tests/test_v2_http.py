"""HTTP contract tests for the production v2 adapter."""

from __future__ import annotations

import json
import time

import numpy as np

from measure_api.v2.application import EngineApplication
from measure_api.v2.http import create_production_app


class _ImageReader:
    def snapshot(self, _descriptor: dict) -> np.ndarray:
        return np.zeros((32, 32), dtype=np.uint8)


class _SlowPreviewImageReader(_ImageReader):
    def snapshot(self, descriptor: dict) -> np.ndarray:
        if descriptor["uuid"] == "slow-frame":
            time.sleep(0.02)
        return super().snapshot(descriptor)


def _image_descriptor() -> dict:
    return {
        "uuid": "frame-http",
        "camera_sn": "camera-http",
        "type": "jpg",
        "height": 32,
        "width": 32,
        "channel": 1,
        "jpeg_size": 123,
        "bin_index": 1,
        "timestamp": 1786243200,
    }


def test_http_adapter_exposes_only_versioned_headless_contract(tmp_path) -> None:
    application = EngineApplication(tmp_path)
    http = create_production_app(application, _ImageReader())
    http.testing = True
    client = http.test_client()
    try:
        created = client.post(
            "/api/v2/modeling-sessions", json={"image": _image_descriptor()}
        )
        assert created.status_code == 201
        session_id = created.get_json()["session_id"]

        obj = client.post(
            f"/api/v2/modeling-sessions/{session_id}/objects",
            json={
                "object_type": "EdgePoint",
                "label": "edge",
                "parameters": {
                    "row": 10,
                    "col": 10,
                    "angle": 0,
                    "length1": 8,
                    "length2": 3,
                },
            },
        )
        assert obj.status_code == 201

        completed = client.post(
            f"/api/v2/modeling-sessions/{session_id}:preview",
            json={"wait_ms": 15_000},
        )
        assert completed.status_code == 200
        assert completed.get_json()["status"] == "succeeded"

        pending = client.post(
            f"/api/v2/modeling-sessions/{session_id}:preview",
            json={"wait_ms": 0},
        )
        assert pending.status_code == 202
        assert pending.headers["Location"].endswith(
            pending.get_json()["operation_id"]
        )
        client.post(
            f"/api/v2/operations/{pending.get_json()['operation_id']}:cancel"
        )

        immutable = client.put(
            f"/api/v2/modeling-sessions/{session_id}/objects/"
            f"{obj.get_json()['object_id']}",
            json={"object_id": "replacement"},
        )
        assert immutable.status_code == 400
        assert immutable.get_json()["error"]["code"] == "object.field_immutable"

        unknown = client.post(
            f"/api/v2/modeling-sessions/{session_id}:preview",
            json={"unexpected": True},
        )
        assert unknown.status_code == 400
        assert unknown.get_json()["error"]["code"] == "request.field_unknown"

        document = client.get("/api/v2/openapi.json").get_json()
        serialized = json.dumps(document)
        assert "/api/v2/measurements" in document["paths"]
        assert all(document["paths"].values())
        assert "image_path" not in serialized
        assert "include_visual" not in serialized
        assert "capabilities" not in serialized
        assert client.get("/api/v2/health").get_json() == {"status": "ok"}
        missing = client.get("/api/v2/capabilities")
        assert missing.status_code == 404
        assert missing.get_json()["error"]["code"] == "request.route_unavailable"
    finally:
        application.close()


def test_openapi_is_an_authoritative_object_and_route_contract() -> None:
    from measure_api.v2.http import openapi_document

    document = openapi_document()
    schemas = document["components"]["schemas"]
    variants = schemas["MeasurementObjectCreate"]["oneOf"]
    by_type = {
        variant["properties"]["object_type"]["const"]: variant
        for variant in variants
    }

    assert set(by_type) == {
        "EdgePoint",
        "EdgePair",
        "FitLine",
        "FitCircle",
        "TemplateMatchPoint",
        "TwoPointsLine",
        "TwoPointsDistance",
        "PointLineDistance",
        "TwoLinesAngle",
        "PointCircleDistance",
    }
    assert by_type["EdgePoint"]["properties"]["parameters"]["properties"][
        "threshold"
    ] == {"type": "number", "minimum": 0, "default": 30.0}
    radius_schema = by_type["FitCircle"]["properties"]["parameters"][
        "properties"
    ]["radius_min"]
    assert {item.get("type") for item in radius_schema["anyOf"]} == {"number", "null"}
    assert by_type["TwoPointsDistance"]["x-input-result-types"] == ["point", "point"]
    assert by_type["TwoPointsDistance"]["x-result-type"] == "distance"
    assert schemas["MeasurementObjectResult"]["oneOf"]
    assert document["paths"]["/api/v2/measurements"]["post"]["responses"]["202"]


def test_http_deadline_includes_shared_memory_snapshot_time(tmp_path) -> None:
    application = EngineApplication(tmp_path)
    http = create_production_app(application, _SlowPreviewImageReader())
    http.testing = True
    client = http.test_client()
    try:
        created = client.post(
            "/api/v2/modeling-sessions", json={"image": _image_descriptor()}
        )
        session_id = created.get_json()["session_id"]
        descriptor = _image_descriptor()
        descriptor["uuid"] = "slow-frame"

        response = client.post(
            f"/api/v2/modeling-sessions/{session_id}:preview",
            json={"image": descriptor, "timeout_ms": 1, "wait_ms": 15_000},
        )

        assert response.status_code == 200
        assert response.get_json()["status"] == "timed_out"
    finally:
        application.close()


def test_http_publish_measure_export_import_and_republish(tmp_path) -> None:
    application = EngineApplication(tmp_path)
    http = create_production_app(application, _ImageReader())
    http.testing = True
    client = http.test_client()
    try:
        created = client.post(
            "/api/v2/modeling-sessions", json={"image": _image_descriptor()}
        )
        session_id = created.get_json()["session_id"]
        client.post(
            f"/api/v2/modeling-sessions/{session_id}/objects",
            json={
                "object_type": "EdgePoint",
                "label": "edge",
                "parameters": {
                    "row": 16,
                    "col": 16,
                    "angle": 0,
                    "length1": 8,
                    "length2": 3,
                },
            },
        )
        publication = client.post(
            f"/api/v2/modeling-sessions/{session_id}:publish",
            json={"calibration_model_id": "http-source", "wait_ms": 15_000},
        ).get_json()
        assert publication["status"] == "succeeded"
        asset_ref = publication["result"]["asset_ref"]

        measurement = client.post(
            "/api/v2/measurements",
            json={
                "calibration_model_id": "http-source",
                "asset_ref": asset_ref,
                "image": _image_descriptor(),
                "wait_ms": 15_000,
            },
        ).get_json()
        assert measurement["status"] == "succeeded"

        exported = client.post(
            "/api/v2/models/http-source:export",
            json={"asset_ref": asset_ref, "wait_ms": 15_000},
        ).get_json()
        assert exported["status"] == "succeeded"
        package = application.read_export(exported["result"]["download_ref"])

        imported = client.post(
            "/api/v2/modeling-sessions:import?wait_ms=15000",
            data=package,
            content_type="application/octet-stream",
        ).get_json()
        assert imported["status"] == "succeeded"
        imported_session_id = imported["result"]["session_id"]

        republished = client.post(
            f"/api/v2/modeling-sessions/{imported_session_id}:publish",
            json={"calibration_model_id": "http-imported", "wait_ms": 15_000},
        ).get_json()
        assert republished["status"] == "succeeded"
    finally:
        application.close()

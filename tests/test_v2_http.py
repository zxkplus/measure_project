"""HTTP contract tests for the production v2 adapter."""

from __future__ import annotations

import json

import numpy as np

from measure_api.v2.application import EngineApplication
from measure_api.v2.http import create_production_app


class _ImageReader:
    def snapshot(self, _descriptor: dict) -> np.ndarray:
        return np.zeros((32, 32), dtype=np.uint8)


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

        document = client.get("/api/v2/openapi.json").get_json()
        serialized = json.dumps(document)
        assert "/api/v2/measurements" in document["paths"]
        assert "image_path" not in serialized
        assert "include_visual" not in serialized
        assert "capabilities" not in serialized
        assert client.get("/api/v2/health").get_json() == {"status": "ok"}
        missing = client.get("/api/v2/capabilities")
        assert missing.status_code == 404
        assert missing.get_json()["error"]["code"] == "request.route_unavailable"
    finally:
        application.close()

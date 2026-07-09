"""Tests for Flask REST server."""

from __future__ import annotations

import json
import os
import tempfile

import cv2
import numpy as np
import pytest

from measure_api.config import Config
from measure_api.server import create_app


@pytest.fixture
def app():
    """Create a Flask test app with minimal config."""
    Config.reset()
    cfg = Config.load_from_dict({
        "log": {"directory": "/tmp", "level": "CRITICAL", "console_output": False},
        "server": {"max_sessions": 10},
        "call_records": {"enabled": False},
    })
    flask_app = create_app(cfg)
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()


@pytest.fixture
def project_dir():
    """Create a temp directory with test images."""
    tmp = tempfile.mkdtemp(prefix="measure_api_server_test_")
    img = np.zeros((100, 100), dtype=np.uint8) + 100
    cv2.circle(img, (50, 50), 30, 200, -1)
    cv2.imwrite(os.path.join(tmp, "ref.png"), img)
    cv2.imwrite(os.path.join(tmp, "insp.png"), img)
    return tmp


# ===========================================================================
# Health
# ===========================================================================


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"


# ===========================================================================
# Sessions
# ===========================================================================


def test_create_session(client, project_dir):
    resp = client.post("/api/session", json={"project_dir": project_dir})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "session_id" in data
    assert data["status"] == "created"


def test_create_session_missing_dir(client):
    resp = client.post("/api/session", json={})
    assert resp.status_code == 400


def test_get_session(client, project_dir):
    create_resp = client.post("/api/session", json={"project_dir": project_dir})
    sid = create_resp.get_json()["session_id"]
    resp = client.get(f"/api/session/{sid}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "phase" in data


def test_get_session_not_found(client):
    resp = client.get("/api/session/nonexistent")
    assert resp.status_code == 404


def test_delete_session(client, project_dir):
    create_resp = client.post("/api/session", json={"project_dir": project_dir})
    sid = create_resp.get_json()["session_id"]
    resp = client.delete(f"/api/session/{sid}")
    assert resp.status_code == 200
    # Verify gone
    resp = client.get(f"/api/session/{sid}")
    assert resp.status_code == 404


def test_list_sessions(client, project_dir):
    client.post("/api/session", json={"project_dir": project_dir})
    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "sessions" in data
    assert len(data["sessions"]) >= 1


# ===========================================================================
# Reference image
# ===========================================================================


def test_load_reference(client, project_dir):
    sid = client.post("/api/session", json={"project_dir": project_dir}).get_json()["session_id"]
    resp = client.post(f"/api/session/{sid}/reference", json={
        "image_path": os.path.join(project_dir, "ref.png"),
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["width"] == 100
    assert data["height"] == 100


def test_load_reference_not_found(client, project_dir):
    sid = client.post("/api/session", json={"project_dir": project_dir}).get_json()["session_id"]
    resp = client.post(f"/api/session/{sid}/reference", json={
        "image_path": "nonexistent.png",
    })
    assert resp.status_code == 400


# ===========================================================================
# Template
# ===========================================================================


def test_set_template(client, project_dir):
    sid = client.post("/api/session", json={"project_dir": project_dir}).get_json()["session_id"]
    client.post(f"/api/session/{sid}/reference", json={
        "image_path": os.path.join(project_dir, "ref.png"),
    })
    resp = client.post(f"/api/session/{sid}/template", json={
        "center": [50, 50],
        "size": [60, 60],
        "angle_deg": 0,
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert "template_shape" in data


# ===========================================================================
# Measurements CRUD
# ===========================================================================


def _setup_ready_session(client, project_dir):
    """Helper: create session with template ready."""
    sid = client.post("/api/session", json={"project_dir": project_dir}).get_json()["session_id"]
    client.post(f"/api/session/{sid}/reference", json={
        "image_path": os.path.join(project_dir, "ref.png"),
    })
    client.post(f"/api/session/{sid}/template", json={
        "center": [50, 50],
        "size": [60, 60],
        "angle_deg": 0,
    })
    return sid


def test_add_measurement(client, project_dir):
    sid = _setup_ready_session(client, project_dir)
    resp = client.post(f"/api/session/{sid}/measurements", json={
        "object_type": "FitCircle",
        "label": "c1",
        "params": {
            "center": [25, 25],
            "radius": 20,
            "measure_length1": 15,
            "measure_length2": 5,
        },
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["label"] == "c1"


def test_list_measurements(client, project_dir):
    sid = _setup_ready_session(client, project_dir)
    client.post(f"/api/session/{sid}/measurements", json={
        "object_type": "FitCircle",
        "label": "c1",
        "params": {
            "center": [25, 25],
            "radius": 20,
            "measure_length1": 15,
            "measure_length2": 5,
        },
    })
    resp = client.get(f"/api/session/{sid}/measurements")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["measurements"]) == 1


def test_delete_measurement(client, project_dir):
    sid = _setup_ready_session(client, project_dir)
    client.post(f"/api/session/{sid}/measurements", json={
        "object_type": "FitCircle",
        "label": "c1",
        "params": {"center": [25, 25], "radius": 20, "measure_length1": 15, "measure_length2": 5},
    })
    resp = client.delete(f"/api/session/{sid}/measurements/c1")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "deleted"


def test_trace_id_header(client, project_dir):
    """Response has X-Trace-Id header."""
    resp = client.get("/api/health")
    assert "X-Trace-Id" in resp.headers
    assert resp.headers["X-Trace-Id"].startswith("trk-")


def test_save_load(client, project_dir):
    sid = _setup_ready_session(client, project_dir)
    # Add a measurement
    client.post(f"/api/session/{sid}/measurements", json={
        "object_type": "FitCircle",
        "label": "c1",
        "params": {"center": [25, 25], "radius": 20, "measure_length1": 15, "measure_length2": 5},
    })
    # Save
    resp = client.post(f"/api/session/{sid}/save")
    assert resp.status_code == 200
    save_data = resp.get_json()
    assert "saved_to" in save_data

    # Reload
    resp = client.post(f"/api/session/{sid}/load")
    assert resp.status_code == 200
    load_data = resp.get_json()
    assert load_data["num_measurements"] == 1


def test_measure_no_image(client, project_dir):
    """Measure without setting inspection image yields 400."""
    sid = _setup_ready_session(client, project_dir)
    resp = client.post(f"/api/session/{sid}/measure", json={})
    assert resp.status_code == 400


def test_endpoint_not_found(client):
    resp = client.get("/api/nonexistent")
    assert resp.status_code == 404

"""Bounded inclusive timing and public operation diagnostics."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from measure_api.v2.app import create_app
from measure_api.v2.timing import Timing, span
from tests.v2.test_operation_lifecycle import settings_for, inspection


def test_nested_spans_preserve_exception_and_context_isolation():
    recorder = Timing()
    with recorder.activate(), span("execute", "phase"):
        with pytest.raises(ValueError), span("bad", "node", node_id="tool-a", space="pixel"):
            raise ValueError("original error")
        with ThreadPoolExecutor() as pool:
            # New threads do not inherit the active operation.
            assert pool.submit(lambda: run_unscoped()).result() == {}
    with span("outside") as value:
        assert value == {}
    snapshot = recorder.snapshot(complete=True)
    parent, child = snapshot["spans"]
    assert child["parent_id"] == parent["id"]
    assert child["state"] == "failed"
    assert parent["duration_ms"] >= child["duration_ms"] >= 0
    assert snapshot["status"] == "complete"


def run_unscoped():
    with span("unscoped") as value:
        return value


def test_bounded_timing_retains_failed_nodes_and_counts_all_work():
    recorder = Timing(limit=5)
    with recorder.activate(), span("execute", "phase"):
        for i in range(1000):
            with span("fit", "node", node_id=str(i)):
                pass
        with pytest.raises(ValueError), span("broken", "node", node_id="error"):
            raise ValueError()
    snapshot = recorder.snapshot(complete=True)
    assert len(snapshot["spans"]) == 5
    assert snapshot["truncated"] and snapshot["status"] == "partial"
    assert any(s.get("node_id") == "error" for s in snapshot["spans"])
    assert next(a["count"] for a in snapshot["aggregates"] if a["name"] == "fit") == 1000
    ids = {s["id"] for s in snapshot["spans"]}
    assert all(s.get("parent_id", s["id"]) in ids for s in snapshot["spans"])


def test_public_timing_survives_restart_with_partial_durable_tail(tmp_path):
    settings = settings_for(tmp_path, state_directory=tmp_path / "runtime")
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v2/image-inspections", json=inspection())
        assert response.status_code == 200, response.text
        operation = response.json()
        timing = operation["diagnostics"]["timing"]
        assert timing["schema_version"] == 1
        assert timing["status"] == "complete"
        assert timing["total_ms"] >= 0
        spans = timing["spans"]
        assert {"preparation", "execution", "result_validate", "result_encode", "terminal_persist"} <= {s["name"] for s in spans}
        assert all(s["start_ms"] >= 0 and s["duration_ms"] >= 0 for s in spans)
        identity = operation["operation_id"]
    with TestClient(create_app(settings)) as client:
        restored = client.get(f"/api/v2/operations/{identity}").json()
        assert restored["state"] == "succeeded"
        assert restored["diagnostics"]["timing"]["status"] == "partial"


def test_forced_cancellation_keeps_last_known_running_phase(tmp_path):
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor

    os.mkfifo(tmp_path / "blocked.png")
    with TestClient(create_app(settings_for(tmp_path))) as client, ThreadPoolExecutor() as pool:
        pending = pool.submit(client.post, "/api/v2/image-inspections", json={
            "image": {"root": "exchange", "path": "blocked.png"}, "mode": "async",
        })
        deadline = time.monotonic() + 5
        observed = None
        while time.monotonic() < deadline:
            records = client.get("/api/v2/operations").json()["operations"]
            if records and records[0]["diagnostics"].get("timing", {}).get("spans"):
                observed = records[0]
                break
            time.sleep(0.01)
        assert observed is not None
        identity = observed["operation_id"]
        canceled = client.post(f"/api/v2/operations/{identity}:cancel").json()
        assert canceled["state"] == "cancelled"
        timing = canceled["diagnostics"]["timing"]
        assert timing["status"] == "partial"
        preparation = next(s for s in timing["spans"] if s["name"] == "preparation")
        assert preparation["state"] == "running"
        assert "duration_ms" not in preparation
        assert pending.result(timeout=5).json()["state"] == "cancelled"

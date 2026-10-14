"""Lifecycle acceptance using public HTTP and real shared files."""

import time
from contextlib import contextmanager

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from measure_api.v2.app import create_app
from measure_api.v2.models import ServiceSettings


def settings_for(tmp_path, **limits):
    cv2.imwrite(str(tmp_path / "image.png"), np.full((32, 48), 90, np.uint8))
    return ServiceSettings(shared_roots={"exchange": tmp_path}, max_workers=1, **limits)


def inspection(**controls):
    return {"image": {"root": "exchange", "path": "image.png"}, **controls}


def test_completed_work_has_listing_diagnostics_trace_and_commit_wins(tmp_path):
    with TestClient(create_app(settings_for(tmp_path))) as client:
        response = client.post("/api/v2/image-inspections", json=inspection())
        assert response.status_code == 200, response.text
        identity = response.json()["operation_id"]
        listing = client.get("/api/v2/operations")
        assert listing.status_code == 200
        assert listing.json()["operations"][0]["operation_id"] == identity
        assert client.get(f"/api/v2/operations/{identity}/diagnostics").status_code == 200
        trace = client.get(f"/api/v2/traces/{response.json()['trace_id']}")
        assert trace.status_code == 200
        assert trace.json()["operations"][0]["state"] == "succeeded"
        cancelled = client.post(f"/api/v2/operations/{identity}:cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["state"] == "succeeded"


def test_retry_identity_is_scoped_by_caller_and_verified_content(tmp_path):
    with TestClient(create_app(settings_for(tmp_path))) as client:
        request = inspection(idempotency_key="capture-1")
        first = client.post("/api/v2/image-inspections", json=request,
                            headers={"X-Caller-Id": "camera-a"})
        retry = client.post("/api/v2/image-inspections", json={**request, "mode": "async"},
                            headers={"X-Caller-Id": "camera-a"})
        assert retry.json()["operation_id"] == first.json()["operation_id"]
        other = client.post("/api/v2/image-inspections", json=request,
                            headers={"X-Caller-Id": "camera-b"})
        assert other.json()["operation_id"] != first.json()["operation_id"]
        cv2.imwrite(str(tmp_path / "image.png"), np.full((32, 48), 100, np.uint8))
        changed = client.post("/api/v2/image-inspections", json=request,
                              headers={"X-Caller-Id": "camera-a"})
        assert changed.status_code == 409
        assert changed.json()["error"]["code"] == "operation.idempotency_conflict"


def terminal(client, identity, seconds=15):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = client.get(f"/api/v2/operations/{identity}").json()
        if result["state"] in {"succeeded", "failed", "cancelled", "timed_out"}:
            return result
        time.sleep(0.02)
    raise AssertionError(result)


def test_deadline_covers_worker_startup_and_preparation(tmp_path):
    with TestClient(create_app(settings_for(tmp_path))) as client:
        response = client.post("/api/v2/image-inspections", json=inspection(timeout_ms=1))
        assert response.status_code in {200, 202}, response.text
        result = terminal(client, response.json()["operation_id"])
        assert result["state"] == "timed_out"
        assert result["error"]["code"] == "operation.timed_out"


def test_cancellation_stops_work_before_output_commit(tmp_path):
    settings = settings_for(tmp_path)
    cv2.imwrite(str(tmp_path / "image.png"), np.full((4096, 4096), 90, np.uint8))
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v2/image-preprocessings", json={
            **inspection(mode="async"), "preprocessing": {"type": "sobel"},
            "output": {"directory": {"root": "exchange", "path": "outputs"}},
        })
        assert response.status_code == 202, response.text
        identity = response.json()["operation_id"]
        cancelled = client.post(f"/api/v2/operations/{identity}:cancel")
        assert cancelled.status_code == 200, cancelled.text
        result = terminal(client, identity)
        assert result["state"] == "cancelled", result
        assert not list((tmp_path / "outputs").rglob("manifest.json"))
        following = client.post("/api/v2/image-inspections", json=inspection())
        assert following.status_code == 200, following.text


def test_restart_retains_results_and_retry_identity_without_owning_outputs(tmp_path):
    settings = settings_for(tmp_path, state_directory=tmp_path / "runtime")
    request = {**inspection(idempotency_key="preview-1"),
               "output": {"directory": {"root": "exchange", "path": "outputs"}}}
    with TestClient(create_app(settings)) as client:
        first = client.post("/api/v2/image-previews", json=request)
        assert first.status_code == 200, first.text
        identity = first.json()["operation_id"]
    with TestClient(create_app(settings)) as client:
        restored = client.get(f"/api/v2/operations/{identity}").json()
        assert restored["result"] == first.json()["result"]
        repeated = client.post("/api/v2/image-previews", json=request)
        assert repeated.json()["operation_id"] == identity
        assert (tmp_path / first.json()["result"]["manifest"]["file"]["path"]).exists()


def test_retention_expiry_is_explicit_and_allows_new_retry(tmp_path):
    settings = settings_for(tmp_path, operation_retention_seconds=0.2)
    with TestClient(create_app(settings)) as client:
        first = client.post("/api/v2/image-inspections", json=inspection(idempotency_key="one"))
        identity = first.json()["operation_id"]
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            expired = client.get(f"/api/v2/operations/{identity}")
            if expired.status_code == 410:
                break
            time.sleep(0.02)
        assert expired.status_code == 410
        repeated = client.post("/api/v2/image-inspections", json=inspection(idempotency_key="one"))
        assert repeated.json()["operation_id"] != identity


def test_bounded_admission_and_queued_cancellation_keep_health_responsive(tmp_path):
    import os
    from concurrent.futures import ThreadPoolExecutor

    settings = settings_for(tmp_path, max_pending=1)
    os.mkfifo(tmp_path / "blocked.png")
    with TestClient(create_app(settings)) as client, ThreadPoolExecutor() as threads:
        blocked = threads.submit(client.post, "/api/v2/image-inspections", json={
            "image": {"root": "exchange", "path": "blocked.png"}, "mode": "async",
        })
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            records = client.get("/api/v2/operations").json()["operations"]
            if records:
                break
            time.sleep(0.02)
        assert records[0]["state"] == "queued"
        assert client.get("/api/v2/health/live").status_code == 200
        excess = client.post("/api/v2/image-inspections", json=inspection())
        assert excess.status_code == 429
        identity = records[0]["operation_id"]
        assert client.post(f"/api/v2/operations/{identity}:cancel").json()["state"] == "cancelled"
        assert blocked.result(timeout=5).json()["state"] == "cancelled"


def test_small_memory_budget_is_a_queryable_failure_not_http_stall(tmp_path):
    settings = settings_for(tmp_path, max_worker_memory_bytes=16 * 1024 * 1024)
    with TestClient(create_app(settings)) as client:
        result = client.post("/api/v2/image-inspections", json=inspection())
        assert result.status_code == 429, result.text
        assert result.json()["error"]["code"] == "operation.memory_limit"
        assert client.get("/api/v2/health/live").status_code == 200


def test_decoded_cache_is_bounded_and_never_hides_rewritten_input(tmp_path):
    settings = settings_for(tmp_path, state_directory=tmp_path / "runtime",
                            cache_max_entries=1, cache_max_bytes=4096)
    with TestClient(create_app(settings)) as client:
        cold = client.post("/api/v2/image-inspections", json=inspection()).json()
        warm = client.post("/api/v2/image-inspections", json=inspection()).json()
        assert warm["result"] == cold["result"]
        assert cold["diagnostics"]["cache"]["misses"] == 1
        assert warm["diagnostics"]["cache"]["hits"] == 1
        cv2.imwrite(str(tmp_path / "other.png"), np.full((32, 48), 99, np.uint8))
        other = client.post("/api/v2/image-inspections", json={
            "image": {"root": "exchange", "path": "other.png"}}).json()
        assert other["diagnostics"]["cache"]["evictions"] == 1
        evicted = client.post("/api/v2/image-inspections", json=inspection()).json()
        assert evicted["result"] == cold["result"]
        assert evicted["diagnostics"]["cache"]["misses"] == 1
        cv2.imwrite(str(tmp_path / "image.png"), np.full((12, 14), 25, np.uint8))
        changed = client.post("/api/v2/image-inspections", json=inspection()).json()
        assert changed["result"]["metadata"]["width"] == 14
        assert changed["result"]["source"]["sha256"] != cold["result"]["source"]["sha256"]
        stale = client.post("/api/v2/image-inspections", json={
            "image": cold["result"]["source"]})
        assert stale.status_code == 409


def test_public_poll_wait_expiry_does_not_cancel_calculation(tmp_path):
    with TestClient(create_app(settings_for(tmp_path))) as client:
        submitted = client.post("/api/v2/image-inspections", json=inspection(wait_ms=0))
        assert submitted.status_code == 202, submitted.text
        identity = submitted.json()["operation_id"]
        final = client.get(f"/api/v2/operations/{identity}?wait_ms=5000")
        assert final.status_code == 200
        assert final.json()["state"] == "succeeded"


def wait_for_boundary(tmp_path):
    import json

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            return json.loads((tmp_path / "boundary.json").read_text())
        except (FileNotFoundError, ValueError):
            time.sleep(0.02)
    raise AssertionError("Filesystem boundary was not reached")


@contextmanager
def real_server(tmp_path, boundary=None):
    import os
    from pathlib import Path
    import socket
    import subprocess
    import sys

    import httpx

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = dict(os.environ, MEASURE_LIFECYCLE_ROOT=str(tmp_path), MEASURE_LIFECYCLE_PORT=str(port),
               PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    if boundary:
        env["MEASURE_LIFECYCLE_BOUNDARY"] = boundary
    script = Path(__file__).with_name("lifecycle_server.py")
    with (tmp_path / "server.log").open("a") as log:
        process = subprocess.Popen([sys.executable, str(script)], env=env,
                                   stdout=log, stderr=log)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    try:
                        if client.get("/api/v2/health/ready").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if process.poll() is not None:
                        raise AssertionError((tmp_path / "server.log").read_text())
                    time.sleep(0.02)
                else:
                    raise AssertionError((tmp_path / "server.log").read_text())
                yield client, process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def test_cancel_at_filesystem_boundary_removes_uncommitted_attempt(tmp_path):
    import os

    settings_for(tmp_path)
    os.mkfifo(tmp_path / "release.fifo")
    with real_server(tmp_path, "before-payload") as (client, _):
        submitted = client.post("/api/v2/image-previews", json={
            **inspection(mode="async"),
            "output": {"directory": {"root": "exchange", "path": "outputs"}},
        })
        assert submitted.status_code == 202, submitted.text
        wait_for_boundary(tmp_path)
        identity = submitted.json()["operation_id"]
        cancelled = client.post(f"/api/v2/operations/{identity}:cancel")
        assert cancelled.json()["state"] == "cancelled", cancelled.text
        final = client.get(f"/api/v2/operations/{identity}?wait_ms=5000")
        assert final.json()["state"] == "cancelled"
        assert not (tmp_path / "outputs" / identity).exists()


def test_worker_interruption_after_manifest_recovers_success(tmp_path):
    import os
    import signal

    settings_for(tmp_path)
    os.mkfifo(tmp_path / "release.fifo")
    with real_server(tmp_path, "after-manifest") as (client, _):
        submitted = client.post("/api/v2/image-previews", json={
            **inspection(mode="async"),
            "output": {"directory": {"root": "exchange", "path": "outputs"}},
        })
        worker = wait_for_boundary(tmp_path)
        identity = submitted.json()["operation_id"]
        os.kill(worker["pid"], signal.SIGKILL)
        result = terminal(client, identity)
        assert result["state"] == "succeeded", result
        assert client.post(f"/api/v2/operations/{identity}:cancel").json()["state"] == "succeeded"
        assert (tmp_path / result["result"]["manifest"]["file"]["path"]).exists()


def test_service_crash_reports_interruption_and_preserves_committed_outputs(tmp_path):
    import os
    import signal

    settings_for(tmp_path)
    original = (tmp_path / "image.png").read_bytes()
    os.mkfifo(tmp_path / "release.fifo")
    with real_server(tmp_path, "before-payload") as (client, process):
        successful = client.post("/api/v2/image-inspections", json=inspection())
        submitted = client.post("/api/v2/image-previews", json={
            **inspection(mode="async"),
            "output": {"directory": {"root": "exchange", "path": "outputs"}},
        })
        wait_for_boundary(tmp_path)
        identity = submitted.json()["operation_id"]
        os.kill(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    with real_server(tmp_path) as (client, _):
        recovered = client.get(f"/api/v2/operations/{identity}").json()
        assert recovered["state"] == "failed"
        assert recovered["error"]["code"] == "worker.interrupted"
        previous = client.get(f"/api/v2/operations/{successful.json()['operation_id']}").json()
        assert previous["state"] == "succeeded"
        assert not (tmp_path / "outputs" / identity).exists()
        assert (tmp_path / "image.png").read_bytes() == original


def test_cancel_after_manifest_wins_even_if_worker_has_not_reported(tmp_path):
    import os

    settings_for(tmp_path)
    os.mkfifo(tmp_path / "release.fifo")
    with real_server(tmp_path, "after-manifest") as (client, _):
        submitted = client.post("/api/v2/image-previews", json={
            **inspection(mode="async"),
            "output": {"directory": {"root": "exchange", "path": "outputs"}},
        })
        wait_for_boundary(tmp_path)
        identity = submitted.json()["operation_id"]
        result = client.post(f"/api/v2/operations/{identity}:cancel", timeout=2)
        assert result.status_code == 200, result.text
        assert result.json()["state"] == "succeeded"


def test_deadline_can_stop_a_stalled_final_commit_before_manifest(tmp_path):
    import os

    settings_for(tmp_path)
    os.mkfifo(tmp_path / "release.fifo")
    with real_server(tmp_path, "before-manifest") as (client, _):
        submitted = client.post("/api/v2/image-previews", json={
            **inspection(mode="async", timeout_ms=1500),
            "output": {"directory": {"root": "exchange", "path": "outputs"}},
        })
        wait_for_boundary(tmp_path)
        identity = submitted.json()["operation_id"]
        result = terminal(client, identity, seconds=5)
        assert result["state"] == "timed_out", result
        assert not list((tmp_path / "outputs").rglob("manifest.json"))


def test_operation_budget_includes_time_waiting_for_a_worker(tmp_path):
    import os
    from concurrent.futures import ThreadPoolExecutor

    settings = settings_for(tmp_path, max_pending=2)
    os.mkfifo(tmp_path / "blocked.png")
    with TestClient(create_app(settings)) as client, ThreadPoolExecutor() as threads:
        blocked = threads.submit(client.post, "/api/v2/image-inspections", json={
            "image": {"root": "exchange", "path": "blocked.png"}, "mode": "async",
        })
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            records = client.get("/api/v2/operations").json()["operations"]
            if records:
                break
            time.sleep(0.02)
        expired = client.post("/api/v2/image-inspections", json=inspection(timeout_ms=20))
        try:
            assert expired.json().get("state") == "timed_out", expired.text
        finally:
            client.post(f"/api/v2/operations/{records[0]['operation_id']}:cancel")
        assert blocked.result(timeout=5).json()["state"] == "cancelled"


def test_journal_admission_failure_is_typed_and_releases_capacity(tmp_path):
    state = tmp_path / "runtime"
    settings = settings_for(tmp_path, state_directory=state, max_pending=1)
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        obstruction = state / "journal.partial"
        obstruction.mkdir()
        failed = client.post("/api/v2/image-inspections", json=inspection(),
                             headers={"X-Trace-Id": "storage-probe"})
        assert failed.status_code == 503, failed.text
        assert failed.json()["error"]["code"] == "service.journal_unavailable"
        assert failed.json()["error"]["trace_id"] == "storage-probe"
        assert failed.json()["error"]["retryable"]
        assert client.get("/api/v2/operations").json()["operations"] == []
        assert not [item for item in state.iterdir() if item.is_dir() and len(item.name) == 32]
        obstruction.rmdir()
        recovered = client.post("/api/v2/image-inspections", json=inspection())
        assert recovered.status_code == 200, recovered.text
        assert recovered.json()["state"] == "succeeded"


@pytest.mark.parametrize("completion", ["cancel", "finish", "shutdown"])
def test_journal_outage_preserves_committed_success_across_shutdown_and_restart(
    tmp_path, completion
):
    import os

    settings_for(tmp_path)
    os.mkfifo(tmp_path / "release.fifo")
    obstruction = tmp_path / "runtime" / "journal.partial"
    saved_result = None
    try:
        with real_server(tmp_path, "after-manifest") as (client, _):
            response = client.post("/api/v2/image-previews", json={
                **inspection(mode="async"),
                "output": {"directory": {"root": "exchange", "path": "outputs"}},
            })
            assert response.status_code == 202, response.text
            wait_for_boundary(tmp_path)
            obstruction.mkdir()
            identity = response.json()["operation_id"]
            if completion == "cancel":
                completed = client.post(f"/api/v2/operations/{identity}:cancel")
                assert completed.status_code == 200, completed.text
                observed = completed.json()
            elif completion == "finish":
                with (tmp_path / "release.fifo").open("wb") as release:
                    release.write(b"x")
                observed = terminal(client, identity)
            if completion != "shutdown":
                assert observed["state"] == "succeeded"
                saved_result = observed["result"]
                assert client.get(f"/api/v2/operations/{identity}").json()["result"] == saved_result
    finally:
        if obstruction.exists():
            obstruction.rmdir()
    with real_server(tmp_path) as (client, _):
        recovered = client.get(f"/api/v2/operations/{identity}")
        assert recovered.status_code == 200, recovered.text
        assert recovered.json()["state"] == "succeeded"
        if saved_result is not None:
            assert recovered.json()["result"] == saved_result
        assert recovered.json()["result"]["metadata"]["width"] == 48
        manifest = recovered.json()["result"]["manifest"]["file"]
        assert (tmp_path / manifest["path"]).is_file()
        following = client.post("/api/v2/image-inspections", json=inspection())
        assert following.status_code == 200, following.text


def test_journal_failure_after_admission_is_queryable_and_capacity_recovers(tmp_path, monkeypatch):
    import os
    from pathlib import Path

    settings = settings_for(tmp_path, state_directory=tmp_path / "runtime", max_pending=1)
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        original_replace = os.replace
        admitted = False

        def unavailable_after_admission(source, destination, *args, **kwargs):
            nonlocal admitted
            if Path(destination).name == "journal.json":
                if admitted:
                    raise OSError("Controlled journal storage outage")
                admitted = True
            return original_replace(source, destination, *args, **kwargs)

        with monkeypatch.context() as fault:
            fault.setattr(os, "replace", unavailable_after_admission)
            failed = client.post(
                "/api/v2/image-inspections", json=inspection(idempotency_key="storage-retry")
            )
            assert failed.status_code == 503, failed.text
            assert failed.json()["error"]["code"] == "service.journal_unavailable"
            records = client.get("/api/v2/operations").json()["operations"]
            assert len(records) == 1
            assert records[0]["state"] == "failed"
        recovered = client.post(
            "/api/v2/image-inspections", json=inspection(idempotency_key="storage-retry")
        )
        assert recovered.status_code == 200, recovered.text

"""Prewarmed process reuse, independent cancellation and replacement acceptance."""

import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from measure_api.v2.app import create_app
from measure_api.v2.models import ServiceSettings


def settings_for(tmp_path, workers=1):
    cv2.imwrite(str(tmp_path / "first.png"), np.full((32, 48), 90, np.uint8))
    cv2.imwrite(str(tmp_path / "second.png"), np.full((19, 27), 20, np.uint8))
    return ServiceSettings(shared_roots={"exchange": tmp_path}, max_workers=workers,
                           state_directory=tmp_path / "runtime")


def inspect(client, image="first.png", **controls):
    return client.post("/api/v2/image-inspections", json={
        "image": {"root": "exchange", "path": image}, **controls,
    })


def wait_until(predicate, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.02)
    raise AssertionError("Worker-pool condition did not complete")


def test_ready_pool_reuses_process_and_keeps_task_images_and_threads_isolated(tmp_path):
    with TestClient(create_app(settings_for(tmp_path))) as client:
        pool = client.app.state.runtime.pool
        assert pool.ready_count == 1
        pid = pool.workers[0].process.pid
        for index in range(100):
            name = "first.png" if index % 2 == 0 else "second.png"
            response = inspect(client, name)
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["diagnostics"]["worker_pid"] == pid
            assert result["diagnostics"]["worker_use_count"] == index + 1
            assert result["diagnostics"]["worker_python_threads"] == 2
            assert result["diagnostics"]["worker_spawn_ms"] == 0
            expected = 48 if index % 2 == 0 else 27
            assert result["result"]["metadata"]["width"] == expected
            assert not list((tmp_path / "runtime").glob("????????????????????????????????"))
        assert pool.ready_count == 1
        assert pool.workers[0].process.pid == pid
    assert all(worker.process is None for worker in pool.workers)


def test_idle_worker_crash_is_replaced_without_restarting_healthy_workers(tmp_path):
    with TestClient(create_app(settings_for(tmp_path, 2))) as client:
        pool = client.app.state.runtime.pool
        original = {worker.process.pid for worker in pool.workers}
        victim = next(iter(original))
        os.kill(victim, signal.SIGKILL)
        wait_until(lambda: pool.ready_count == 2 and victim not in {
            worker.process.pid for worker in pool.workers if worker.process is not None})
        current = {worker.process.pid for worker in pool.workers}
        assert original - {victim} <= current
        response = inspect(client)
        assert response.status_code == 200, response.text


def test_five_busy_workers_cancel_and_replace_only_the_target(tmp_path):
    settings = settings_for(tmp_path, 5)
    for index in range(5):
        os.mkfifo(tmp_path / f"blocked-{index}.png")
    with TestClient(create_app(settings)) as client, ThreadPoolExecutor(6) as threads:
        pool = client.app.state.runtime.pool
        original = {worker.process.pid for worker in pool.workers}
        assert len(original) == pool.ready_count == 5
        calls = [threads.submit(inspect, client, f"blocked-{index}.png", mode="async")
                 for index in range(5)]
        records = []
        try:
            def all_busy():
                nonlocal records
                records = client.get("/api/v2/operations").json()["operations"]
                return (len(records) == 5 and all(
                    record["diagnostics"].get("worker_pid") for record in records))
            wait_until(all_busy)
            assert {r["diagnostics"]["worker_pid"] for r in records} == original
            queued_call = threads.submit(inspect, client, mode="async")
            queued_record = wait_until(lambda: next((r for r in client.get(
                "/api/v2/operations").json()["operations"]
                if r["state"] == "queued" and not r["diagnostics"].get("worker_pid")), None))
            response = client.post(f"/api/v2/operations/{queued_record['operation_id']}:cancel")
            assert response.json()["state"] == "cancelled"
            assert queued_call.result(timeout=1).json()["state"] == "cancelled"
            queued = inspect(client, timeout_ms=30)
            assert queued.json()["state"] == "timed_out", queued.text
            target = records[0]
            victim = target["diagnostics"]["worker_pid"]
            cancelled = client.post(f"/api/v2/operations/{target['operation_id']}:cancel")
            assert cancelled.json()["state"] == "cancelled", cancelled.text
            following = inspect(client)
            assert following.status_code == 200, following.text
            assert following.json()["diagnostics"]["worker_pid"] not in original
            assert original - {victim} <= {
                worker.process.pid for worker in pool.workers if worker.process is not None}
        finally:
            for record in records:
                client.post(f"/api/v2/operations/{record['operation_id']}:cancel")
            for call in calls:
                assert call.result(timeout=10).json()["state"] == "cancelled"
        wait_until(lambda: pool.ready_count == 5)



def test_deadline_kills_an_unresponsive_worker_and_replaces_it(tmp_path):
    os.mkfifo(tmp_path / "blocked.png")
    with TestClient(create_app(settings_for(tmp_path))) as client, ThreadPoolExecutor(1) as threads:
        blocked = threads.submit(inspect, client, "blocked.png", mode="async", timeout_ms=500)
        record = wait_until(lambda: next((r for r in client.get(
            "/api/v2/operations").json()["operations"]
            if r["diagnostics"].get("worker_pid")), None))
        victim = record["diagnostics"]["worker_pid"]
        os.kill(victim, signal.SIGSTOP)
        started = time.monotonic()
        response = blocked.result(timeout=4)
        assert response.json()["state"] == "timed_out", response.text
        assert time.monotonic() - started < 3
        following = inspect(client)
        assert following.status_code == 200, following.text
        assert following.json()["diagnostics"]["worker_pid"] != victim



def test_failed_prewarm_reaps_processes_and_releases_state_owner(tmp_path, monkeypatch):
    from measure_api.v2.errors import ServiceError
    from measure_api.v2.worker_pool import WorkerPool

    async def fail(worker_pool, worker):
        raise RuntimeError("Controlled prewarm failure")

    monkeypatch.setattr(WorkerPool, "_warm", fail)
    app = create_app(settings_for(tmp_path, 2))
    with pytest.raises(ServiceError) as caught:
        with TestClient(app):
            pass
    assert caught.value.code == "worker.startup_failed"
    assert app.state.runtime.owner.closed
    assert all(worker.process is None for worker in app.state.runtime.pool.workers)

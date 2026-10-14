"""Deployment acceptance through the installed CLI and real HTTP sockets."""

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import cv2
import numpy as np


def _http(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return response.status, json.load(response)


def test_cli_starts_headless_with_shared_aliases_and_retires_old_protocol(tmp_path):
    upstream = tmp_path / "upstream-mount"
    upstream.mkdir()
    engine = tmp_path / "engine-mount"
    engine.symlink_to(upstream, target_is_directory=True)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    image = np.zeros((30, 50), dtype=np.uint8)
    cv2.imwrite(str(upstream / "input.png"), image)
    digest = hashlib.sha256((upstream / "input.png").read_bytes()).hexdigest()
    config = tmp_path / "settings.json"
    config.write_text(json.dumps({
        "shared_roots": {"exchange": str(engine), "outputs": str(outputs)},
        "writable_roots": ["outputs"], "state_directory": str(tmp_path / "state"),
        "max_workers": 1, "max_pending": 2, "cache_max_bytes": 0,
    }))
    sentinel = tmp_path / "import-sentinel"
    sentinel.mkdir()
    (sentinel / "sitecustomize.py").write_text(
        "import sys\n"
        "class BlockGUI:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.split('.')[0] in "
        "{'measure_gui','PyQt5','PyQt6','PySide6','tkinter','flask','waitress','matplotlib'}:\n"
        "            raise ImportError('V2 imported GUI/legacy dependency: ' + fullname)\n"
        "sys.meta_path.insert(0, BlockGUI())\n"
    )
    env = dict(os.environ)
    repo = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join([str(sentinel), repo])
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    executable = os.environ.get("MEASURE_V2_HEADLESS_PYTHON", sys.executable)
    with (tmp_path / "server.log").open("w+") as log:
        server = subprocess.Popen([
            executable, "-m", "measure_api.v2", "--config", str(config),
            "--host", "127.0.0.1", "--port", str(port),
        ], cwd=repo, env=env, stdout=log, stderr=log)
        try:
            for _ in range(150):
                if server.poll() is not None:
                    log.seek(0)
                    raise AssertionError(log.read())
                try:
                    if _http(base, "/api/v2/health/ready")[0] == 200:
                        break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.1)
            else:
                raise AssertionError("CLI did not become ready")
            payload = {"image": {"root": "exchange", "path": "input.png", "sha256": digest}}
            status, sync = _http(base, "/api/v2/image-inspections", payload)
            assert status == 200
            assert sync["result"]["metadata"]["width"] == 50
            status, accepted = _http(
                base, "/api/v2/image-inspections", {**payload, "mode": "async"})
            assert status in {200, 202}
            _, complete = _http(
                base, f"/api/v2/operations/{accepted['operation_id']}?wait_ms=10000")
            assert complete["state"] == "succeeded"
            assert complete["result"] == sync["result"]
            _, preview = _http(base, "/api/v2/image-previews", {
                **payload, "output": {"directory": {"root": "outputs", "path": "generated"}},
            })
            manifest = preview["result"]["manifest"]["file"]
            assert manifest["root"] == "outputs"
            assert (outputs / manifest["path"]).is_file()
            assert hashlib.sha256((upstream / "input.png").read_bytes()).hexdigest() == digest
            request_file = tmp_path / "request.json"
            request_file.write_text(json.dumps(payload))
            client_output = subprocess.check_output([
                executable, "scripts/v2_http_client.py", "--base-url", base,
                "--route", "/image-inspections", "--request", str(request_file),
            ], cwd=repo, env=env, text=True)
            assert json.loads(client_output)["result"]["metadata"]["height"] == 30
            cases_file = tmp_path / "cases.json"
            cases_file.write_text(json.dumps([{
                "name": "shared-image", "route": "/image-inspections", "request": payload,
                "observations": ["/metadata/width"],
                "observation_units": {"/metadata/width": "px"},
            }]))
            report_file = tmp_path / "qualification.json"
            subprocess.check_output([
                executable, "scripts/v2_benchmark.py", "--base-url", base,
                "--cases", str(cases_file), "--repeats", "2", "--concurrency", "1",
                "--output", str(report_file),
            ], cwd=repo, env=env, text=True)
            report = json.loads(report_file.read_text())
            assert report["summary"][0]["successes"] == 2
            measured = report["summary"][0]["observations"]["/metadata/width"]
            assert measured["values"] == [50, 50]
            assert measured["same_file_stddev"] == 0
            assert measured["absolute_error_max"] is None
            assert report["qualification"]["status"] == "pending"
            for path in ["/api/v2/health", "/api/v2/modeling-sessions", "/api/v2/measurements"]:
                try:
                    _http(base, path, {})
                except urllib.error.HTTPError as error:
                    assert error.code == 404
                else:
                    raise AssertionError(f"Obsolete route remains: {path}")
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()

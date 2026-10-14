"""Required route and advertised schema acceptance through public HTTP discovery."""

import json
from pathlib import Path
import re

from fastapi.testclient import TestClient

from measure_api.v2.app import create_app
from measure_api.v2.models import ServiceSettings


def _normalize(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "{parameter}", path)


def test_required_spec_routes_and_all_advertised_calculations_have_openapi(tmp_path):
    required = json.loads((Path(__file__).resolve().parents[2]
                           / "docs/v2-required-routes.json").read_text())
    with TestClient(create_app(ServiceSettings(
        shared_roots={"exchange": tmp_path},
        state_directory=tmp_path.parent / (tmp_path.name + "-state"),
    ))) as client:
        response = client.get("/api/v2/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        paths = {_normalize(path): methods for path, methods in schema["paths"].items()}
        for route in required:
            if route["path"] in {"/api/v2/docs", "/api/v2/openapi.json"}:
                assert client.get(route["path"]).status_code == 200
                continue
            path = _normalize(route["path"])
            assert path in paths, f"Missing Spec route: {route}"
            assert route["method"].lower() in paths[path], f"Missing Spec method: {route}"
        discovery = client.get("/api/v2/capabilities").json()
        advertised = discovery["operations"]
        assert len({item["id"] for item in advertised}) == len(advertised)
        calculation_paths = {_normalize("/api/v2" + item["path"]) for item in advertised}
        routed_calculations = {path for path, methods in paths.items()
                               if "post" in methods and not path.endswith(":cancel")}
        assert calculation_paths == routed_calculations
        for item in advertised:
            declared = paths[_normalize("/api/v2" + item["path"])]["post"]
            assert declared["operationId"] == item["id"]
            assert item["modes"] == ["sync", "async"]
            assert set(item["request_schema"]["properties"]) >= {
                "mode", "wait_ms", "timeout_ms", "idempotency_key",
            }
            assert item["result_schema"]["type"] == "object"
            assert "200" in declared["responses"] and "202" in declared["responses"]
        obsolete = {"/api/v2/modeling-sessions", "/api/v2/measurements", "/api/v2/health"}
        assert not obsolete.intersection(paths)

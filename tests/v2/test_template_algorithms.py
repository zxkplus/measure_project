"""Both matchers are reachable through the same v2 contract.

Shape scoring reports a mean gradient-direction cosine, correlation a normalized
intensity coefficient. The two are not comparable, so these tests compare
positions rather than scores, and they pin the reporting of the policy fields
that only the correlation pipeline implemented.
"""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from measure_api.v2.app import create_app
from measure_api.v2.models import ServiceSettings


def ref(path):
    return {"root": "exchange", "path": path}


def call(client, route, payload):
    response = client.post("/api/v2" + route, json=payload)
    assert response.status_code == 200, response.text
    return response.json()["result"]


@pytest.fixture
def client(tmp_path):
    settings = ServiceSettings(shared_roots={"exchange": tmp_path}, max_workers=1)
    with TestClient(create_app(settings)) as value:
        yield value


def _teach(client, tmp_path):
    """Teach one textured support and return the patch plus its template asset."""
    rng = np.random.default_rng(7)
    patch = rng.integers(20, 230, (40, 40), dtype=np.uint8)
    patch = cv2.GaussianBlur(patch, (3, 3), 1.0)
    cv2.circle(patch, (20, 20), 10, 10, -1)
    reference = np.zeros((160, 200), np.uint8)
    reference[60:100, 80:120] = patch
    cv2.imwrite(str(tmp_path / "reference.png"), reference)
    built = call(client, "/template-builds", {
        "image": ref("reference.png"),
        "roi": {"center": {"x": 99.5, "y": 79.5}, "width": 40, "height": 40},
        "preprocessing": {"type": "raw"},
        "output": {"directory": ref("templates")},
    })
    asset = next(item for item in built["artifacts"] if item["kind"] == "template")
    inspection = np.zeros((200, 260), np.uint8)
    inspection[120:160, 160:200] = patch
    cv2.imwrite(str(tmp_path / "inspection.png"), inspection)
    return patch, asset["file"]


def _match(client, asset, algorithm, **search):
    return call(client, "/template-matches", {
        "image": ref("inspection.png"), "template": asset,
        "search": {"score_threshold": 0.5, "algorithm": algorithm, **search},
    })


def test_shape_finds_the_same_target_as_correlation(client, tmp_path):
    _, asset = _teach(client, tmp_path)
    correlated = _match(client, asset, "ncc")
    shaped = _match(client, asset, "shape")
    assert correlated["valid"] and shaped["valid"]
    assert shaped["matches"][0]["center"] == pytest.approx(
        correlated["matches"][0]["center"], abs=2.0)
    assert shaped["diagnostics"]["algorithm"] == "shape"
    assert shaped["diagnostics"]["search_plan"]["resample"] == "shape"


def test_correlation_stays_the_default(client, tmp_path):
    _, asset = _teach(client, tmp_path)
    result = call(client, "/template-matches", {
        "image": ref("inspection.png"), "template": asset,
        "search": {"score_threshold": 0.5},
    })
    assert result["valid"]
    assert result["diagnostics"]["search_plan"]["resample"] != "shape"


def test_unknown_algorithm_is_rejected(client, tmp_path):
    _, asset = _teach(client, tmp_path)
    response = client.post("/api/v2/template-matches", json={
        "image": ref("inspection.png"), "template": asset,
        "search": {"algorithm": "surf"},
    })
    assert response.status_code == 422


def test_shape_reports_policy_fields_it_cannot_honour(client, tmp_path):
    _, asset = _teach(client, tmp_path)
    result = _match(client, asset, "shape", fine_search_policy="whole",
                    fine_angle_policy="hierarchical")
    ignored = result["diagnostics"]["search_plan"]["ignored_fields"]
    assert "fine_search_policy" in ignored
    assert "fine_angle_policy" in ignored


def test_shape_falls_back_when_a_scale_range_is_requested(client, tmp_path):
    _, asset = _teach(client, tmp_path)
    result = _match(client, asset, "shape", scale_min=0.9, scale_max=1.1,
                    scale_step=0.1)
    assert result["diagnostics"].get("shape_scale_fallback") == "multi_scale_uses_correlation"

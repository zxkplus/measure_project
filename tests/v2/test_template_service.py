"""Template teaching and localization acceptance through the public HTTP contract."""

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


def teach(client, tmp_path, *, mode="tight_roi", preprocessing=None):
    rng = np.random.default_rng(123)
    patch = rng.integers(20, 230, (20, 60), dtype=np.uint8)
    reference = np.zeros((100, 160), np.uint8)
    reference[30:50, 40:100] = patch
    cv2.imwrite(str(tmp_path / "reference.png"), reference)
    result = call(client, "/template-builds", {
        "image": ref("reference.png"),
        "roi": {"center": {"x": 69.5, "y": 39.5}, "width": 60, "height": 20},
        "anchor": {"x": 45, "y": 35}, "support_mode": mode,
        "preprocessing": preprocessing or {"type": "raw"},
        "output": {"directory": ref("templates")},
    })
    return patch, result


def test_teach_inspect_and_match_rectangular_support_and_arbitrary_anchor(client, tmp_path):
    patch, built = teach(client, tmp_path)
    asset = next(item for item in built["artifacts"] if item["kind"] == "template")
    inspected = call(client, "/template-inspections", {"template": asset["file"]})
    assert inspected["metadata"]["schema_version"] == "1.0"
    assert inspected["metadata"]["support"]["width"] == 60
    assert inspected["metadata"]["support"]["height"] == 20
    assert inspected["metadata"]["anchor"] == {"x": 45, "y": 35}
    image = np.zeros((160, 240), np.uint8)
    image[70:90, 100:160] = patch
    cv2.imwrite(str(tmp_path / "inspection.png"), image)
    result = call(client, "/template-matches", {
        "image": ref("inspection.png"), "template": asset["file"],
        "search": {"score_threshold": 0.95},
    })
    assert result["valid"]
    match = result["matches"][0]
    assert match["center"] == pytest.approx({"x": 129.5, "y": 79.5}, abs=0.1)
    assert match["anchor"] == pytest.approx({"x": 105, "y": 75}, abs=0.1)
    assert match["mapping"]["matrix"][0][2] == pytest.approx(60, abs=0.1)
    assert match["mapping"]["matrix"][1][2] == pytest.approx(40, abs=0.1)
    assert result["matches"][0]["score"] > 0.99


def test_repeated_targets_survive_pyramid_search_and_local_point_measurement(client, tmp_path):
    patch, built = teach(client, tmp_path)
    asset = built["artifacts"][0]["file"]
    image = np.zeros((240, 360), np.uint8)
    image[50:70, 80:140] = patch
    image[150:170, 240:300] = patch
    cv2.imwrite(str(tmp_path / "inspection.png"), image)
    request = {"image": ref("inspection.png"), "template": asset,
               "search": {"score_threshold": 0.9, "max_matches": 0,
                          "pyramid_levels": 1, "pyramid_max_template_size": 30}}
    result = call(client, "/template-matches", request)
    assert len(result["matches"]) == 2
    centers = sorted((m["center"]["x"], m["center"]["y"]) for m in result["matches"])
    assert centers[0] == pytest.approx((109.5, 59.5), abs=0.2)
    assert centers[1] == pytest.approx((269.5, 159.5), abs=0.2)
    local = call(client, "/measurements/template-points", {
        **request, "search": {"score_threshold": 0.9,
                               "region": {"x": 200, "y": 120, "width": 140, "height": 90}},
    })
    assert local["point"]["point"] == pytest.approx({"x": 245, "y": 155}, abs=0.2)
    assert local["valid"]


def test_clockwise_rotation_and_scale_have_invertible_metric_mappings(client, tmp_path):
    patch, built = teach(client, tmp_path)
    image = np.zeros((240, 300), np.uint8)
    # Independent rendering: reference support maps to inspection at clockwise20, scale1.2.
    theta = np.deg2rad(20)
    matrix = np.array([[1.2*np.cos(theta), -1.2*np.sin(theta), 120],
                       [1.2*np.sin(theta), 1.2*np.cos(theta), 80]], np.float64)
    image = cv2.warpAffine(patch, matrix, (300, 240))
    cv2.imwrite(str(tmp_path / "pose.png"), image)
    result = call(client, "/template-matches", {
        "image": ref("pose.png"), "template": built["artifacts"][0]["file"],
        "search": {"score_threshold": 0.85, "angle_min": 10, "angle_max": 30,
                   "angle_step": 1, "scale_min": 1.1, "scale_max": 1.3, "scale_step": 0.02,
                   "coarse_fine": True, "coarse_angle_step": 5, "coarse_scale_step": 0.1},
    })
    assert result["valid"]
    found = result["matches"][0]
    assert found["angle_degrees"] == pytest.approx(20, abs=1)
    assert found["scale"] == pytest.approx(1.2, abs=0.02)
    forward = np.array(found["support_mapping"]["matrix"])
    inverse = np.array(found["support_mapping"]["inverse_matrix"])
    assert forward @ [0, 0, 1] == pytest.approx([120, 80, 1], abs=0.8)
    assert inverse @ forward @ [10, 5, 1] == pytest.approx([10, 5, 1], abs=1e-6)
    assert np.linalg.norm(forward[:2, 0]) == pytest.approx(1.2, abs=0.02)


def test_square_context_float_preprocessing_and_fresh_file_content(client, tmp_path):
    patch, tight = teach(client, tmp_path, preprocessing={"type": "sobel", "kernel_size": 3})
    inspected = call(client, "/template-inspections", {"template": tight["artifacts"][0]["file"]})
    assert inspected["metadata"]["dtype"] == "float32"
    # Fresh asset can be used in a new request without teaching state in the engine.
    result = call(client, "/template-matches", {
        "image": ref("reference.png"), "template": tight["artifacts"][0]["file"],
        "search": {"score_threshold": 0.8},
    })
    assert result["valid"]
    assert result["matches"][0]["center"] == pytest.approx({"x": 69.5, "y": 39.5}, abs=0.2)
    _, square = teach(client, tmp_path, mode="square_context")
    assert square["metadata"]["support"]["width"] == 60
    assert square["metadata"]["support"]["height"] == 60
    (tmp_path / "current.npz").write_bytes(
        (tmp_path / square["artifacts"][0]["file"]["path"]).read_bytes()
    )
    first = call(client, "/template-inspections", {"template": ref("current.npz")})
    (tmp_path / "current.npz").write_bytes(
        (tmp_path / tight["artifacts"][0]["file"]["path"]).read_bytes()
    )
    second = call(client, "/template-inspections", {"template": ref("current.npz")})
    assert second["metadata"]["support"]["height"] == 20
    assert first["template"]["sha256"] != second["template"]["sha256"]
    cv2.imwrite(str(tmp_path / "blank.png"), np.zeros((120, 160), np.uint8))
    weak = call(client, "/template-matches", {
        "image": ref("blank.png"), "template": ref("current.npz"),
    })
    assert not weak["valid"] and weak["matches"] == []


def test_match_count_search_region_and_invalid_portable_asset(client, tmp_path):
    patch, built = teach(client, tmp_path)
    image = np.zeros((160, 300), np.uint8)
    image[25:45, 30:90] = patch
    image[100:120, 180:240] = patch
    cv2.imwrite(str(tmp_path / "two.png"), image)
    request = {"image": ref("two.png"), "template": built["artifacts"][0]["file"]}
    single = call(client, "/template-matches", request)
    assert len(single["matches"]) == 1
    restricted = call(client, "/template-matches", {
        **request, "search": {"max_matches": 0, "score_threshold": 0.95,
                               "region": {"x": 150, "y": 80, "width": 140, "height": 60}},
    })
    assert len(restricted["matches"]) == 1
    assert restricted["matches"][0]["center"] == pytest.approx({"x": 209.5, "y": 109.5}, abs=0.2)
    (tmp_path / "bad.npz").write_bytes(b"not a portable template")
    invalid = client.post("/api/v2/template-inspections", json={"template": ref("bad.npz")})
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "template.invalid"
    reversed_range = client.post("/api/v2/template-matches", json={
        **request, "search": {"angle_min": 10, "angle_max": -10},
    })
    assert reversed_range.status_code == 422


def test_replacement_template_content_is_used_for_matching_and_constant_teaching_is_rejected(
    client, tmp_path
):
    patch, first = teach(client, tmp_path)
    first_file = tmp_path / first["artifacts"][0]["file"]["path"]
    (tmp_path / "current.npz").write_bytes(first_file.read_bytes())
    old_image = np.zeros((150, 220), np.uint8)
    old_image[61:81, 101:161] = patch
    cv2.imwrite(str(tmp_path / "old.png"), old_image)
    request = {"template": ref("current.npz"), "image": ref("old.png"),
               "search": {"score_threshold": 0.95, "pyramid_levels": 1}}
    assert call(client, "/template-matches", request)["valid"]
    replacement = np.random.default_rng(66).integers(0, 255, (20, 60), np.uint8)
    reference = np.zeros((100, 160), np.uint8)
    reference[30:50, 40:100] = replacement
    cv2.imwrite(str(tmp_path / "new_reference.png"), reference)
    second = call(client, "/template-builds", {
        "image": ref("new_reference.png"),
        "roi": {"center": {"x": 69.5, "y": 39.5}, "width": 60, "height": 20},
        "output": {"directory": ref("new_templates")},
    })
    (tmp_path / "current.npz").write_bytes(
        (tmp_path / second["artifacts"][0]["file"]["path"]).read_bytes()
    )
    stale = call(client, "/template-matches", request)
    assert not stale["valid"] and stale["matches"] == []
    old_image[61:81, 101:161] = replacement
    cv2.imwrite(str(tmp_path / "new.png"), old_image)
    fresh = call(client, "/template-matches", {**request, "image": ref("new.png")})
    assert fresh["valid"] and fresh["matches"][0]["score"] > 0.99
    cv2.imwrite(str(tmp_path / "constant.png"), np.zeros((100, 160), np.uint8))
    invalid = client.post("/api/v2/template-builds", json={
        "image": ref("constant.png"),
        "roi": {"center": {"x": 69.5, "y": 39.5}, "width": 60, "height": 20},
        "output": {"directory": ref("constant_templates")},
    })
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "template.uninformative"


def test_rotated_teaching_support_composes_reference_mapping_without_anchor_drift(client, tmp_path):
    patch = np.random.default_rng(51).integers(0, 255, (20, 60), np.uint8)
    reference = np.zeros((130, 170), np.uint8)
    reference[30:90, 60:80] = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(str(tmp_path / "rotated_reference.png"), reference)
    built = call(client, "/template-builds", {
        "image": ref("rotated_reference.png"),
        "roi": {"center": {"x": 69.5, "y": 59.5}, "width": 60, "height": 20,
                "angle_degrees": 90},
        "anchor": {"x": 70, "y": 50}, "output": {"directory": ref("rotated_templates")},
    })
    result = call(client, "/template-matches", {
        "image": ref("rotated_reference.png"), "template": built["template"],
        "search": {"score_threshold": 0.95, "angle_min": 90, "angle_max": 90},
    })
    assert result["valid"]
    assert result["matches"][0]["anchor"] == pytest.approx({"x": 70, "y": 50}, abs=0.2)
    assert np.array(result["matches"][0]["mapping"]["matrix"]) == pytest.approx(np.eye(3), abs=0.2)

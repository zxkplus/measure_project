"""Alignment matrices and metric geometry acceptance at the HTTP boundary."""

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
    with TestClient(create_app(ServiceSettings(shared_roots={"exchange": tmp_path},
                                                max_workers=1))) as value:
        yield value


def test_similarity_maps_known_correspondences_and_typed_geometry(client):
    result = call(client, "/transforms/similarity-estimates", {
        "source_points": [{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 0, "y": 10}],
        "destination_points": [{"x": 7, "y": 9}, {"x": 7, "y": 29}, {"x": -13, "y": 9}],
        "source_frame": "reference", "destination_frame": "inspection",
    })
    assert result["valid"]
    assert result["scale"] == pytest.approx(2)
    assert result["angle_degrees"] == pytest.approx(90)
    mapped = call(client, "/geometry/transforms", {
        "mapping": result["mapping"], "geometry": [
            {"type": "point", "point": {"x": 2, "y": 3}},
            {"type": "circle", "center": {"x": 0, "y": 0}, "radius": 4},
            {"type": "line", "a": 0, "b": 1, "c": 0},
        ],
    })
    assert mapped["geometry"][0]["point"] == pytest.approx({"x": 1, "y": 13})
    assert mapped["geometry"][1]["radius"] == pytest.approx(8)
    line = mapped["geometry"][2]
    assert abs(line["a"] * 7 + line["b"] * 15 + line["c"]) < 1e-6
    assert mapped["frame"] == "inspection"
    failed = call(client, "/transforms/similarity-estimates", {
        "source_points": [{"x": 2, "y": 2}, {"x": 2, "y": 2}],
        "destination_points": [{"x": 1, "y": 1}, {"x": 5, "y": 4}],
    })
    assert not failed["valid"]
    assert failed["mapping"] is None


def identity_mapping():
    return {"source_frame": "inspection", "destination_frame": "aligned", "type": "identity",
            "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "inverse_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}


def test_control_refinement_uses_inverse_correction_and_explicit_fallback(client, tmp_path):
    rng = np.random.default_rng(44)
    reference = np.zeros((100, 140), np.uint8)
    control_positions = [(25, 30), (90, 60), (55, 75)]
    templates = []
    for index, (x, y) in enumerate(control_positions):
        reference[y-5:y+6, x-5:x+6] = rng.integers(0, 255, (11, 11), np.uint8)
    cv2.imwrite(str(tmp_path / "reference.png"), reference)
    for x, y in control_positions:
        built = call(client, "/template-builds", {
            "image": ref("reference.png"),
            "roi": {"center": {"x": x, "y": y}, "width": 11, "height": 11},
            "output": {"directory": ref("controls")},
        })
        templates.append({"template": built["artifacts"][0]["file"],
                          "reference_point": {"x": x, "y": y}, "search_radius": 10,
                          "score_threshold": 0.95})
    inspection = cv2.warpAffine(reference, np.array([[1, 0, 5], [0, 1, -3]], np.float64),
                                (140, 100))
    cv2.imwrite(str(tmp_path / "inspection.png"), inspection)
    request = {"image": ref("inspection.png"), "initial_mapping": identity_mapping(),
               "width": 140, "height": 100, "controls": templates, "minimum_controls": 3,
               "max_residual": 0.5, "output": {"directory": ref("alignments")}}
    result = call(client, "/alignments/control-points", request)
    assert result["valid"] and not result["used_fallback"]
    assert np.array(result["mapping"]["matrix"]) @ [30, 27, 1] == pytest.approx(
        [25, 30, 1], abs=0.15,
    )
    aligned = cv2.imread(str(tmp_path / result["artifacts"][0]["file"]["path"]), 0)
    assert aligned[30, 25] == pytest.approx(int(reference[30, 25]), abs=5)
    cv2.imwrite(str(tmp_path / "blank.png"), np.zeros_like(reference))
    failed = call(client, "/alignments/control-points", {
        **request, "image": ref("blank.png"), "failure_policy": "fail",
    })
    assert not failed["valid"]
    assert failed["mapping"] is None and failed["artifacts"] == []
    fallback = call(client, "/alignments/control-points", {
        **request, "image": ref("blank.png"), "failure_policy": "rigid_fallback",
    })
    assert fallback["valid"] and fallback["used_fallback"]
    assert fallback["mapping"]["matrix"] == identity_mapping()["matrix"]


def test_rigid_normalization_preserves_actual_size_and_forbids_affine_circle_metrology(
    client, tmp_path
):
    image = np.zeros((100, 140), np.uint8)
    image[30:60, 40:100] = 200
    cv2.imwrite(str(tmp_path / "metric.png"), image)
    mapping = identity_mapping()
    mapping["type"] = "similarity"
    mapping["matrix"] = [[0.5, 0, -20], [0, 0.5, -15], [0, 0, 1]]
    mapping["inverse_matrix"] = [[2, 0, 40], [0, 2, 30], [0, 0, 1]]
    result = call(client, "/alignments/rigid", {
        "image": ref("metric.png"), "initial_mapping": mapping,
        "width": 30, "height": 15, "output": {"directory": ref("metric_outputs")},
    })
    inverse = np.array(result["metric_mapping"]["matrix"])
    assert inverse @ [0, 0, 1] == pytest.approx([40, 30, 1])
    assert inverse @ [30, 0, 1] == pytest.approx([100, 30, 1])
    # A normalized30px feature still covers60px in the metrology frame.
    mapped = call(client, "/geometry/transforms", {
        "mapping": result["metric_mapping"],
        "geometry": [{"type": "rectangle", "center": {"x": 15, "y": 7.5},
                      "width": 30, "height": 15}],
    })
    assert mapped["geometry"][0]["width"] == pytest.approx(60)
    affine = identity_mapping()
    affine["type"] = "affine"
    affine["matrix"] = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
    affine["inverse_matrix"] = [[0.5, 0, 0], [0, 1, 0], [0, 0, 1]]
    rejected = client.post("/api/v2/geometry/transforms", json={
        "mapping": affine, "geometry": [{"type": "circle", "center": {"x": 0, "y": 0},
                                          "radius": 10}],
    })
    assert rejected.status_code == 422
    assert rejected.json()["error"]["code"] == "transform.similarity_required"


def test_robust_correspondences_reject_outliers_and_enforce_residuals(client):
    payload = {"source_points": [{"x": 0, "y": 0}, {"x": 20, "y": 0},
                                  {"x": 0, "y": 20}, {"x": 20, "y": 20}],
               "destination_points": [{"x": 5, "y": -3}, {"x": 25, "y": -3},
                                       {"x": 5, "y": 17}, {"x": 80, "y": 90}],
               "robust": True, "minimum_inliers": 3, "ransac_threshold": 1,
               "max_residual": 0.1}
    robust = call(client, "/transforms/similarity-estimates", payload)
    assert robust["valid"] and robust["inliers"] == [True, True, True, False]
    assert np.array(robust["mapping"]["matrix"]) @ [3, 4, 1] == pytest.approx([8, 1, 1])
    strict = call(client, "/transforms/similarity-estimates", {**payload, "robust": False})
    assert not strict["valid"] and strict["mapping"] is None


def test_control_search_cannot_borrow_features_from_neighbor_target(client, tmp_path):
    rng = np.random.default_rng(9)
    reference = np.zeros((70, 90), np.uint8)
    points = [(20, 20), (65, 50)]
    for x, y in points:
        reference[y-4:y+5, x-4:x+5] = rng.integers(0, 255, (9, 9), np.uint8)
    cv2.imwrite(str(tmp_path / "reference.png"), reference)
    controls = []
    for x, y in points:
        built = call(client, "/template-builds", {
            "image": ref("reference.png"),
            "roi": {"center": {"x": x, "y": y}, "width": 9, "height": 9},
            "output": {"directory": ref("teaching")},
        })
        controls.append({"template": built["artifacts"][0]["file"],
                         "reference_point": {"x": x, "y": y}, "search_radius": 100,
                         "score_threshold": 0.95})
    inspection = np.zeros((70, 180), np.uint8)
    inspection[:, 90:] = reference
    cv2.imwrite(str(tmp_path / "neighbors.png"), inspection)
    request = {"image": ref("neighbors.png"), "initial_mapping": identity_mapping(),
               "width": 90, "height": 70, "controls": controls,
               "output": {"directory": ref("isolated")}}
    left = call(client, "/alignments/control-points", request)
    assert not left["valid"] and all(not c["valid"] for c in left["controls"])
    right_mapping = identity_mapping()
    right_mapping["matrix"][0][2] = -90
    right_mapping["inverse_matrix"][0][2] = 90
    right = call(
        client, "/alignments/control-points", {**request, "initial_mapping": right_mapping}
    )
    assert right["valid"] and all(c["valid"] for c in right["controls"])
    assert np.array(right["mapping"]["matrix"]) @ [110, 20, 1] == pytest.approx(
        [20, 20, 1], abs=0.1
    )

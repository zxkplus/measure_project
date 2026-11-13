"""Acceptance for one-way V1 project migration into executable V2 recipes."""

import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from measure_api.project import MeasureProject
from measure_api.v2.app import create_app
from measure_api.v2.models import ServiceSettings


OUTPUT = {"directory": {"root": "exchange", "path": "outputs"}}
IMAGE = {"root": "exchange", "path": "v1/reference.png"}
CONFIG = {"root": "exchange", "path": "v1/config.json"}
MANIFEST = {"root": "exchange", "path": "v1/manifest.json"}
REFERENCE = {"root": "exchange", "path": "v1/reference.png"}
REFERENCE_NPY = {"root": "exchange", "path": "v1/reference.npy"}


def _axis_aligned_image():
    image = np.full((160, 200), 40, np.uint8)
    image[30:130, 80:120] = 200
    image[70:100, 90:110] = 60
    return image


def _rotated_image():
    patch = np.zeros((80, 80), np.uint8)
    patch[:, 30:50] = 255
    patch[5:12, 57:66] = 140
    matrix = np.array(
        [[0.866025403784, -0.5, 85.35898384864],
         [0.5, 0.866025403784, 45.35898384864]]
    )
    return cv2.warpAffine(patch, matrix, (180, 160))


def _build_project(root, *, angle_deg=0, image=None):
    image = image if image is not None else _axis_aligned_image()
    cv2.imwrite(str(root / "ref.png"), image)
    project = MeasureProject(str(root / "v1"))
    project.load_reference(str(root / "ref.png"))
    project.set_template(
        center=(80, 100) if angle_deg == 0 else (100, 100),
        size=(80, 80),
        angle_deg=angle_deg,
        angle_range_deg=10,
    )
    project.add_measurement(
        "EdgePair", "pair",
        {"row": 40, "col": 40, "angle": 0, "length1": 25, "length2": 2,
         "sigma": 1, "threshold": 30},
    )
    project.add_measurement(
        "TemplateMatchPoint", "tpl",
        {"row": 40, "col": 40, "template_size": 20},
    )
    project.add_composed(
        "TwoPointsDistance", "dist",
        {"point_a_label": "pair_A", "point_b_label": "pair_B"},
    )
    project.add_composed(
        "TwoPointsLine", "line",
        {"point_a_label": "pair_A", "point_b_label": "pair_B"},
    )
    project.save()
    assert (root / "v1" / "manifest.json").is_file()
    return project


@pytest.fixture
def v1_fixture(tmp_path):
    project = _build_project(tmp_path)
    return tmp_path, project


@pytest.fixture
def client(v1_fixture):
    root, _ = v1_fixture
    with TestClient(create_app(ServiceSettings(shared_roots={"exchange": root}))) as value:
        yield value


def post(client, path, payload):
    response = client.post("/api/v2" + path, json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def payload(**overrides):
    value = {
        "config": CONFIG,
        "model_manifest": MANIFEST,
        "reference": REFERENCE,
        "reference_npy": REFERENCE_NPY,
        "recipe_id": "migrated",
        "recipe_version": "1",
        "output": OUTPUT,
    }
    value.update(overrides)
    return value


def migrate(client, **overrides):
    return post(client, "/legacy-v1-project-conversions", payload(**overrides))["result"]


def mutate(root, name, mutate_fn):
    path = root / "v1" / name
    value = json.loads(path.read_text())
    mutate_fn(value)
    path.write_text(json.dumps(value))


def node_by_label(recipe, label):
    return next(item for item in recipe["nodes"] if item["label"] == label)


def outcome_by_id(run, identity):
    return next(item for item in run["targets"][0]["nodes"] if item["id"] == identity)


def test_migrates_real_v1_project_and_executes(client, v1_fixture):
    root, project = v1_fixture
    result = migrate(client)
    assert result["valid"], result["diagnostics"]
    assert result["migrated_from"] == "v1-api-project"
    assert result["summary"]["nodes"] == 4
    assert result["summary"]["templates"] == 2
    assert result["configuration_sha256"]
    assert len(result["resolved_recipe"]["nodes"]) == 4
    pair_node = node_by_label(result["resolved_recipe"], "pair")
    template_node = node_by_label(result["resolved_recipe"], "tpl")
    assert pair_node["operation"] == "measurements.edge-pairs"
    assert template_node["operation"] == "measurements.template-points"

    run = post(client, "/runs", {
        "recipe": result["resolved_recipe"], "image": IMAGE,
    })["result"]
    assert run["status"] == "completed"
    pair = outcome_by_id(run, node_by_label(result["resolved_recipe"], "pair")["id"])["result"]
    distance = outcome_by_id(run, node_by_label(result["resolved_recipe"], "dist")["id"])["result"]
    template = outcome_by_id(run, node_by_label(result["resolved_recipe"], "tpl")["id"])["result"]
    assert pair["valid"]
    assert template["valid"]
    assert pair["pairs"][0]["width"]["value"] == pytest.approx(distance["value"], abs=1e-6)

    legacy = project.measure(str(root / "ref.png"))
    assert legacy["targets"][0]["valid"]
    measured = legacy["targets"][0]["measurements"]
    assert measured["pair_A"]["valid"] and measured["pair_B"]["valid"]
    assert pair["pairs"][0]["edge_a"]["point"]["x"] == pytest.approx(
        measured["pair_A"]["col"], abs=0.5
    )
    assert pair["pairs"][0]["edge_b"]["point"]["x"] == pytest.approx(
        measured["pair_B"]["col"], abs=0.5
    )
    expected_distance = np.hypot(
        measured["pair_B"]["col"] - measured["pair_A"]["col"],
        measured["pair_B"]["row"] - measured["pair_A"]["row"],
    )
    assert distance["value"] == pytest.approx(expected_distance, abs=1.0)


def test_npy_only_reference_is_staged(client):
    result = migrate(client, reference=None)
    assert result["valid"], result["diagnostics"]
    assert result["summary"]["templates"] == 2
    run = post(client, "/runs", {
        "recipe": result["resolved_recipe"], "image": IMAGE,
    })["result"]
    assert run["status"] == "completed"


def test_missing_manifest_blocks_with_diagnostic(client):
    result = migrate(client, model_manifest=None)
    assert not result["valid"]
    assert "v1.missing_model_manifest" in {item["code"] for item in result["diagnostics"]}


def test_unsupported_alignment_blocks(client, v1_fixture):
    root, _ = v1_fixture
    mutate(root, "manifest.json", lambda value: value["alignment"].update(mode="obb"))
    result = migrate(client)
    assert not result["valid"]
    assert "v1.unsupported_alignment" in {item["code"] for item in result["diagnostics"]}


def test_unknown_config_field_blocks(client, v1_fixture):
    root, _ = v1_fixture
    mutate(root, "config.json", lambda value: value.update(surprise=1))
    result = migrate(client)
    assert not result["valid"]
    assert "v1.unknown_field" in {item["code"] for item in result["diagnostics"]}


def test_duplicate_label_blocks(client, v1_fixture):
    root, _ = v1_fixture
    mutate(root, "config.json", lambda value: value["measurement_defs"][1].update(label="pair"))
    result = migrate(client)
    assert not result["valid"]
    assert "v1.duplicate_label" in {item["code"] for item in result["diagnostics"]}


def test_unresolved_dependency_blocks(client, v1_fixture):
    root, _ = v1_fixture

    def mutate_deps(value):
        value["composed_defs"][0]["deps"]["point_a_label"] = "missing"

    mutate(root, "config.json", mutate_deps)
    result = migrate(client)
    assert not result["valid"]
    assert "v1.unresolved_dependency" in {item["code"] for item in result["diagnostics"]}


def test_empty_project_blocks(client, v1_fixture):
    root, _ = v1_fixture

    def clear(value):
        value["measurement_defs"] = []
        value["composed_defs"] = []

    mutate(root, "config.json", clear)
    result = migrate(client)
    assert not result["valid"]
    assert "v1.empty_project" in {item["code"] for item in result["diagnostics"]}


def test_advisory_reject_blocks(client):
    result = migrate(client, advisory_policy="reject")
    assert not result["valid"]
    assert "advisory.rejected" in {item["code"] for item in result["diagnostics"]}


def test_physical_scale_without_applicability_is_dropped(client, v1_fixture):
    root, _ = v1_fixture
    mutate(root, "manifest.json", lambda value: value.update(physical_scale_mm=0.1))
    result = migrate(client)
    assert result["valid"], result["diagnostics"]
    assert result["resolved_recipe"]["metric"] is None
    assert "v1.physical_units_dropped" in {item["code"] for item in result["diagnostics"]}
    assert result["summary"]["dropped_physical_units"]


def test_physical_scale_with_applicability_builds_metric(client, v1_fixture):
    root, _ = v1_fixture
    mutate(root, "manifest.json", lambda value: value.update(physical_scale_mm=0.1))
    result = migrate(client, applicability={
        "camera_id": "cam-1",
        "setup_id": "line-a",
        "frame": "source",
        "image_size": {"width": 200, "height": 160},
    })
    assert result["valid"], result["diagnostics"]
    metric = result["resolved_recipe"]["metric"]
    assert metric["applicability"]["camera_id"] == "cam-1"
    assert metric["calibration"]["value"]["parameters"]["scale_per_px"] == pytest.approx(0.1)
    assert "v1.calibration_derived" in {item["code"] for item in result["diagnostics"]}


def test_reference_digest_mismatch_blocks(client, v1_fixture):
    root, _ = v1_fixture
    pixels = np.load(str(root / "v1" / "reference.npy"))
    pixels[0, 0] = (int(pixels[0, 0]) + 1) % 256
    np.save(str(root / "v1" / "reference.npy"), pixels)
    result = migrate(client, reference=None)
    assert not result["valid"]
    assert "v1.reference_digest_mismatch" in {item["code"] for item in result["diagnostics"]}


def test_reference_pixel_mismatch_blocks(client, v1_fixture):
    root, _ = v1_fixture
    pixels = np.load(str(root / "v1" / "reference.npy"))
    pixels[0, 0] = (int(pixels[0, 0]) + 1) % 256
    np.save(str(root / "v1" / "reference.npy"), pixels)
    result = migrate(client)
    assert not result["valid"]
    assert "v1.reference_pixel_mismatch" in {item["code"] for item in result["diagnostics"]}


def test_idempotency_key_pins_file_contents(client, v1_fixture):
    root, _ = v1_fixture
    first = migrate(client, idempotency_key="migrate-pin")
    assert first["valid"], first["diagnostics"]
    mutate(root, "config.json", lambda value: value.update(surprise=1))
    response = client.post(
        "/api/v2/legacy-v1-project-conversions",
        json=payload(idempotency_key="migrate-pin"),
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "operation.idempotency_conflict"


def test_missing_reference_forms_is_rejected(client):
    response = client.post(
        "/api/v2/legacy-v1-project-conversions",
        json=payload(reference=None, reference_npy=None),
    )
    assert response.status_code == 422


def test_rotated_roi_preserves_patch_mapping(client, tmp_path):
    project = _build_project(tmp_path, angle_deg=30, image=_rotated_image())
    result = post(client, "/legacy-v1-project-conversions", {
        "config": CONFIG,
        "model_manifest": MANIFEST,
        "reference": REFERENCE,
        "recipe_id": "rotated",
        "recipe_version": "1",
        "output": OUTPUT,
    })["result"]
    assert result["valid"], result["diagnostics"]
    assert result["resolved_recipe"]["alignment"]["width"] == 80
    assert result["resolved_recipe"]["alignment"]["height"] == 80
    run = post(client, "/runs", {
        "recipe": result["resolved_recipe"], "image": IMAGE,
    })["result"]
    pair = outcome_by_id(run, node_by_label(result["resolved_recipe"], "pair")["id"])["result"]
    distance = outcome_by_id(run, node_by_label(result["resolved_recipe"], "dist")["id"])["result"]
    legacy = project.measure(str(tmp_path / "ref.png"))
    assert legacy["targets"][0]["valid"]
    measured = legacy["targets"][0]["measurements"]
    assert pair["valid"] and measured["pair_A"]["valid"] and measured["pair_B"]["valid"]
    assert pair["pairs"][0]["edge_a"]["point"]["x"] == pytest.approx(
        measured["pair_A"]["col"], abs=0.5
    )
    assert pair["pairs"][0]["edge_b"]["point"]["x"] == pytest.approx(
        measured["pair_B"]["col"], abs=0.5
    )
    expected_distance = np.hypot(
        measured["pair_B"]["col"] - measured["pair_A"]["col"],
        measured["pair_B"]["row"] - measured["pair_A"]["row"],
    )
    assert distance["value"] == pytest.approx(expected_distance, abs=1.0)
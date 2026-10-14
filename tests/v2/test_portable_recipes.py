"""Public HTTP acceptance for portable upstream recipes and frozen replay."""

import hashlib
import io
import json
import shutil
import time
import zipfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from measure_api.v2.app import create_app
from measure_api.v2.models import ServiceSettings


@pytest.fixture
def client(tmp_path):
    pixels = np.zeros((100, 160), np.uint8)
    pixels[:, 40:60] = 255
    cv2.imwrite(str(tmp_path / "stripes.png"), pixels)
    with TestClient(create_app(ServiceSettings(shared_roots={"exchange": tmp_path}))) as value:
        yield value


IMAGE = {"root": "exchange", "path": "stripes.png"}
OUTPUT = {"directory": {"root": "exchange", "path": "outputs"}}


def recipe():
    return {
        "id": "upstream",
        "version": "7",
        "nodes": [
            {
                "id": "width",
                "operation": "measurements.edge-pairs",
                "parameters": {
                    "probe": {
                        "center": {"x": 50, "y": 50},
                        "search_half_length": 25,
                        "average_half_width": 3,
                    },
                    "selection": "first",
                    "polarity": "positive",
                },
            }
        ],
    }


def post(client, path, body):
    response = client.post("/api/v2" + path, json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_run_records_actual_shared_algorithm_runtime_identity(client):
    capabilities = client.get("/api/v2/capabilities").json()
    recorded = post(client, "/runs", {"recipe": recipe(), "image": IMAGE})["result"]
    identity = recorded["runtime_identity"]
    assert identity == capabilities["runtime_identity"]
    assert identity["algorithm_version"] == capabilities["algorithm_version"]
    assert len(identity["implementation_sha256"]) == 64
    assert identity["dependencies"]["numpy"] == np.__version__
    assert identity["dependencies"]["opencv"] == cv2.__version__
    assert identity["python_version"]


def test_portable_package_relocates_reference_and_keeps_numerical_result(client, tmp_path):
    supplied = recipe()
    supplied["reference"] = IMAGE
    exported = post(
        client,
        "/recipe-exports",
        {
            "recipe": supplied,
            "output": OUTPUT,
        },
    )["result"]
    package = next(item for item in exported["artifacts"] if item["kind"] == "recipe_package")
    other = tmp_path / "relocated"
    other.mkdir()
    shutil.copyfile(tmp_path / package["file"]["path"], other / "supplied.zip")
    cv2.imwrite(str(other / "image.png"), cv2.imread(str(tmp_path / "stripes.png"), 0))
    with TestClient(create_app(ServiceSettings(shared_roots={"new": other}))) as moved:
        imported = post(
            moved,
            "/recipe-imports",
            {
                "package": {"root": "new", "path": "supplied.zip"},
                "output": {"directory": {"root": "new", "path": "assets"}},
            },
        )["result"]
        assert imported["valid"]
        assert imported["resolved_recipe"]["reference"]["root"] == "new"
        run = post(
            moved,
            "/runs",
            {
                "recipe": imported["resolved_recipe"],
                "image": {"root": "new", "path": "image.png"},
            },
        )["result"]
        assert run["targets"][0]["nodes"][0]["result"]["width"]["value"] == pytest.approx(20)
        assert imported["source_package"]["sha256"] == package["sha256"]


def test_reference_package_explicit_alias_binding_pins_external_asset(client, tmp_path):
    supplied = recipe()
    supplied["reference"] = IMAGE
    exported = post(
        client,
        "/recipe-exports",
        {
            "recipe": supplied,
            "output": OUTPUT,
            "packaging": "references",
        },
    )["result"]
    package = exported["artifacts"][0]["file"]
    manifest = json.loads((tmp_path / package["path"]).read_bytes())
    manifest["assets"][0]["source"]["root"] = "old-machine"
    (tmp_path / "external.json").write_text(json.dumps(manifest))
    imported = post(
        client,
        "/recipe-imports",
        {
            "package": {"root": "exchange", "path": "external.json"},
            "output": OUTPUT,
            "root_bindings": {"old-machine": "exchange"},
        },
    )["result"]
    assert imported["resolved_recipe"]["reference"]["path"] == "stripes.png"
    pixels = cv2.imread(str(tmp_path / "stripes.png"), 0)
    pixels[0, 0] = 1
    cv2.imwrite(str(tmp_path / "stripes.png"), pixels)
    changed = client.post(
        "/api/v2/recipe-imports",
        json={
            "package": {"root": "exchange", "path": "external.json"},
            "output": OUTPUT,
            "root_bindings": {"old-machine": "exchange"},
        },
    )
    assert changed.status_code == 409
    assert changed.json()["error"]["code"] == "file.digest_mismatch"


@pytest.mark.parametrize("attack", ["traversal", "duplicate", "digest", "expanded"])
def test_unsafe_packages_fail_without_completed_output(client, tmp_path, attack):
    supplied = recipe()
    supplied["reference"] = IMAGE
    exported = post(client, "/recipe-exports", {"recipe": supplied, "output": OUTPUT})["result"]
    package = tmp_path / exported["artifacts"][0]["file"]["path"]
    with zipfile.ZipFile(package) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members["manifest.json"])
    name = "assets/" + manifest["assets"][0]["name"]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for key, payload in members.items():
            archive.writestr(key, b"changed" if attack == "digest" and key == name else payload)
        if attack == "traversal":
            archive.writestr("../escape.png", b"invalid")
        if attack == "duplicate":
            archive.writestr("manifest.json", members["manifest.json"])
        if attack == "expanded":
            archive.writestr("assets/huge", b"0" * (270 * 1024 * 1024))
    (tmp_path / "attack.zip").write_bytes(buffer.getvalue())
    response = client.post(
        "/api/v2/recipe-imports",
        json={
            "package": {"root": "exchange", "path": "attack.zip"},
            "output": {"directory": {"root": "exchange", "path": "invalid-import"}},
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "recipe_package.invalid"
    assert not (tmp_path / "invalid-import").exists()


def test_package_rejects_object_npz_even_when_not_used_by_recipe(client, tmp_path):
    exported = post(client, "/recipe-exports", {"recipe": recipe(), "output": OUTPUT})["result"]
    package = tmp_path / exported["artifacts"][0]["file"]["path"]
    with zipfile.ZipFile(package) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    payload = io.BytesIO()
    np.savez(payload, metadata=np.array({"forbidden": "pickle"}, dtype=object))
    content = payload.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    manifest["assets"].append(
        {
            "id": "asset-0",
            "name": "asset-0.npz",
            "kind": "recipe_asset",
            "format": "npz",
            "sha256": digest,
            "size_bytes": len(content),
            "source": {"root": "old", "path": "unsafe.npz", "sha256": digest},
        }
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("assets/asset-0.npz", content)
    (tmp_path / "objects.zip").write_bytes(buffer.getvalue())
    response = client.post(
        "/api/v2/recipe-imports",
        json={
            "package": {"root": "exchange", "path": "objects.zip"},
            "output": {"directory": {"root": "exchange", "path": "unsafe"}},
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "recipe_package.unsafe_payload"
    assert not (tmp_path / "unsafe").exists()


def test_frozen_replay_after_operation_expiration_uses_new_ids_and_same_values(tmp_path):
    pixels = np.zeros((100, 160), np.uint8)
    pixels[:, 40:60] = 255
    path = tmp_path / "stripes.png"
    cv2.imwrite(str(path), pixels)
    original_bytes = path.read_bytes()
    settings = ServiceSettings(
        shared_roots={"exchange": tmp_path}, operation_retention_seconds=0.1
    )
    with TestClient(create_app(settings)) as client:
        run = post(client, "/runs", {"recipe": recipe(), "image": IMAGE, "output": OUTPUT})
        exported = post(
            client,
            "/recipe-exports",
            {
                "recipe": run["result"]["resolved_recipe"],
                "frozen_run": run["result"],
                "output": OUTPUT,
            },
        )["result"]
        frozen = next(
            item["file"] for item in exported["artifacts"] if item["kind"] == "frozen_run"
        )
        time.sleep(0.15)
        assert client.get("/api/v2/operations/" + run["operation_id"]).status_code == 410
        replay = post(
            client,
            "/replays",
            {
                "frozen_file": frozen,
                "output": {"directory": {"root": "exchange", "path": "replays"}},
            },
        )
        assert replay["operation_id"] != run["operation_id"]
        assert replay["result"]["equivalent"] is True
        assert replay["result"]["run"]["targets"][0]["nodes"][0]["result"]["width"][
            "value"
        ] == pytest.approx(20)
        assert replay["result"]["run"]["manifest"]["file"]["path"].startswith("replays/")
        assert path.read_bytes() == original_bytes


@pytest.mark.parametrize("failure", ["changed", "missing", "runtime", "configuration"])
def test_replay_explicitly_rejects_changed_missing_or_incompatible_inputs(
    client, tmp_path, failure
):
    run = post(client, "/runs", {"recipe": recipe(), "image": IMAGE})["result"]
    exported = post(
        client,
        "/recipe-exports",
        {
            "recipe": run["resolved_recipe"],
            "frozen_run": run,
            "output": OUTPUT,
        },
    )["result"]
    descriptor = next(item for item in exported["artifacts"] if item["kind"] == "frozen_run")
    frozen = json.loads((tmp_path / descriptor["file"]["path"]).read_bytes())
    codes = {
        "changed": "file.digest_mismatch",
        "missing": "file.not_found",
        "runtime": "replay.incompatible_version",
        "configuration": "replay.configuration_mismatch",
    }
    if failure == "changed":
        (tmp_path / "stripes.png").write_bytes(b"different")
    elif failure == "missing":
        (tmp_path / "stripes.png").unlink()
    elif failure == "runtime":
        frozen["runtime_identity"]["dependencies"]["opencv"] = "unsupported-runtime"
    else:
        frozen["recipe"]["version"] = "changed"
    response = client.post(
        "/api/v2/replays",
        json={
            "frozen": frozen,
            "output": {"directory": {"root": "exchange", "path": "bad-replay"}},
        },
    )
    assert response.status_code == (404 if failure == "missing" else 409)
    assert response.json()["error"]["code"] == codes[failure]
    assert not (tmp_path / "bad-replay").exists()


def test_unknown_historical_producer_cannot_be_inferred_from_current_service(client):
    run = post(client, "/runs", {"recipe": recipe(), "image": IMAGE})["result"]
    run.pop("runtime_identity")
    response = client.post(
        "/api/v2/recipe-exports",
        json={
            "recipe": run["resolved_recipe"],
            "frozen_run": run,
            "output": OUTPUT,
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "replay.unknown_producer"


def test_declared_embedded_image_payload_must_be_decodable(client, tmp_path):
    supplied = recipe()
    supplied["reference"] = IMAGE
    exported = post(client, "/recipe-exports", {"recipe": supplied, "output": OUTPUT})["result"]
    with zipfile.ZipFile(tmp_path / exported["artifacts"][0]["file"]["path"]) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    asset = manifest["assets"][0]
    invalid = b"not-a-png"
    digest = hashlib.sha256(invalid).hexdigest()
    asset.update(sha256=digest, size_bytes=len(invalid))
    asset["source"]["sha256"] = digest
    manifest["recipe"]["reference"]["sha256"] = digest
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("assets/" + asset["name"], invalid)
    (tmp_path / "invalid-image.zip").write_bytes(buffer.getvalue())
    response = client.post(
        "/api/v2/recipe-imports",
        json={
            "package": {"root": "exchange", "path": "invalid-image.zip"},
            "output": OUTPUT,
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "recipe_package.unsafe_payload"


def test_gui_conversion_preserves_rotated_roi_and_forward_virtual_pair_ports(client, tmp_path):
    # Independent taught patch: 20 px bright band, asymmetry removes NCC ambiguity.
    patch = np.zeros((80, 80), np.uint8)
    patch[:, 30:50] = 255
    patch[5:12, 57:66] = 140
    # Legacy image-from-patch rotation is clockwise 30deg around patch pixel (40,40).
    matrix = np.array(
        [[0.866025403784, -0.5, 85.35898384864], [0.5, 0.866025403784, 45.35898384864]]
    )
    source = cv2.warpAffine(patch, matrix, (180, 160))
    cv2.imwrite(str(tmp_path / "reference.png"), source)
    project = {
        "version": 3,
        "roi": {
            "center_row": 100,
            "center_col": 100,
            "width": 80,
            "height": 80,
            "angle_deg": 30,
            "confirmed": True,
            "alignment_mode": "single_box",
        },
        "matching": {
            "match_score_threshold": 0.8,
            "angle_range_deg": 0,
            "angle_step_deg": 1,
            "max_matches": 1,
            "preprocessor_type": "raw",
        },
        "template_view": {
            "zoom_factor": 8,
            "offset_x": 999,
            "tools": [
                {
                    "object_type": "TwoPointsDistance",
                    "label": "distance",
                    "params": {"point_a_label": "pair_A", "point_b_label": "pair_B"},
                },
                {
                    "object_type": "EdgePair",
                    "label": "pair",
                    "params": {
                        "row": 40,
                        "col": 40,
                        "angle": 0,
                        "length1": 25,
                        "length2": 3,
                        "sigma": 1,
                        "threshold": 30,
                    },
                },
            ],
        },
        "tool_list_order": ["distance", "pair"],
        "gui": {"width": 1900},
    }
    (tmp_path / "project.json").write_text(json.dumps(project))
    converted = post(
        client,
        "/legacy-project-conversions",
        {
            "project": {"root": "exchange", "path": "project.json"},
            "recipe_id": "rotated",
            "recipe_version": "migrated-1",
            "output": OUTPUT,
        },
    )["result"]
    assert converted["valid"], converted["diagnostics"]
    nodes = converted["resolved_recipe"]["nodes"]
    distance = next(node for node in nodes if node["label"] == "distance")
    assert distance["inputs"]["point_a"]["port"] == "edge_a"
    assert distance["inputs"]["point_b"]["port"] == "edge_b"
    run = post(
        client,
        "/runs",
        {
            "recipe": converted["resolved_recipe"],
            "image": {"root": "exchange", "path": "reference.png"},
        },
    )["result"]
    outcomes = run["targets"][0]["nodes"]
    measured = next(node for node in outcomes if node["id"] == distance["id"])
    assert measured["result"]["value"] == pytest.approx(20, abs=0.35), json.dumps(run)
    pair = next(node for node in outcomes if node["id"] != distance["id"])
    first = pair["pixel_result"]["edge_a"]["point"]
    second = pair["pixel_result"]["edge_b"]["point"]
    assert first["y"] == pytest.approx(94.75, abs=0.7)
    assert second["y"] == pytest.approx(104.75, abs=0.7)
    assert first["x"] < second["x"]
    assert any(item["code"] == "ignored_ui_state" for item in converted["diagnostics"])


@pytest.mark.parametrize(
    "problem", ["duplicate", "unknown", "parameter", "multi_point", "matching"]
)
def test_unsupported_legacy_configuration_blocks_ready_recipe(client, tmp_path, problem):
    tools = [
        {
            "object_type": "EdgePair",
            "label": "pair",
            "params": {
                "row": 50,
                "col": 50,
                "angle": 0,
                "length1": 25,
                "length2": 3,
            },
        }
    ]
    project = {"version": 3, "template_view": {"tools": tools}}
    expected = {
        "duplicate": "ambiguous_labels",
        "unknown": "unsupported_tool",
        "parameter": "unsupported_parameter",
        "multi_point": "unsupported_multi_point",
        "matching": "unsupported_field",
    }
    if problem == "duplicate":
        tools.append(tools[0])
    elif problem == "unknown":
        tools[0]["object_type"] = "UnsupportedMeasurement"
    elif problem == "parameter":
        tools[0]["params"]["arbitrary_algorithm"] = "hidden"
    elif problem == "multi_point":
        project["roi"] = {"alignment_mode": "multi_point"}
    else:
        project["matching"] = {"unknown_threshold_mode": "unsafe"}
    (tmp_path / "project.json").write_text(json.dumps(project))
    converted = post(
        client,
        "/legacy-project-conversions",
        {
            "project": {"root": "exchange", "path": "project.json"},
            "reference": IMAGE,
            "recipe_id": "migrated",
            "recipe_version": "1",
            "output": OUTPUT,
        },
    )["result"]
    assert not converted["valid"]
    assert converted["resolved_recipe"] is None
    assert any(
        item["code"] == expected[problem] and item["severity"] == "error"
        for item in converted["diagnostics"]
    )
    assert converted["manifest"] is None


def test_pickle_only_legacy_metadata_is_explicitly_blocked_without_loading_objects(
    client, tmp_path
):
    np.savez(
        tmp_path / "workflow.npz", workflow_meta=np.array({"must": "not load"}, dtype=object)
    )
    converted = post(
        client,
        "/legacy-project-conversions",
        {
            "project": {"root": "exchange", "path": "workflow.npz"},
            "recipe_id": "legacy",
            "recipe_version": "1",
            "output": OUTPUT,
        },
    )["result"]
    assert converted["valid"] is False
    assert converted["diagnostics"][0]["code"] == "pickle_metadata_requires_safe_manifest"
    assert converted["manifest"] is None


def test_core_unicode_npz_converts_to_executable_recipe_with_known_20px_width(client, tmp_path):
    parameters = {
        "center_row": 50,
        "center_col": 50,
        "bbox_width": 80,
        "bbox_height": 80,
        "bbox_angle_deg": 0,
        "angle_range": [0, 0],
        "match_score_threshold": 0.8,
        "max_matches": 1,
    }
    definitions = [
        {
            "measure_type": "EdgePair",
            "label": "pair",
            "params": {
                "row": 40,
                "col": 40,
                "angle": 0,
                "length1": 25,
                "length2": 3,
            },
        }
    ]
    pixels = cv2.imread(str(tmp_path / "stripes.png"), 0)
    np.savez(
        tmp_path / "safe.npz",
        version=1,
        template_params=json.dumps(parameters),
        measure_defs=json.dumps(definitions),
        preprocessor_json=json.dumps({"type": "raw"}),
        template_image=pixels[10:90, 10:90],
        edge_template=pixels[10:90, 10:90].astype(np.float32),
    )
    converted = post(
        client,
        "/legacy-project-conversions",
        {
            "project": {"root": "exchange", "path": "safe.npz"},
            "reference": IMAGE,
            "recipe_id": "core",
            "recipe_version": "1",
            "output": OUTPUT,
        },
    )["result"]
    assert converted["valid"], converted["diagnostics"]
    run = post(client, "/runs", {"recipe": converted["resolved_recipe"], "image": IMAGE})[
        "result"
    ]
    assert run["targets"][0]["nodes"][0]["result"]["width"]["value"] == pytest.approx(20)


def test_portable_numeric_template_and_calibration_relocate_and_keep_mm(client, tmp_path):
    pixels = cv2.imread(str(tmp_path / "stripes.png"), 0)
    pixels[20:27, 67:74] = 170
    pixels[70:76, 24:31] = 110
    cv2.imwrite(str(tmp_path / "stripes.png"), pixels)
    template = post(
        client,
        "/template-builds",
        {
            "image": IMAGE,
            "roi": {"center": {"x": 50, "y": 50}, "width": 80, "height": 80},
            "output": OUTPUT,
        },
    )["result"]["template"]
    applicability = {
        "camera_id": "c1",
        "setup_id": "s1",
        "frame": "source",
        "image_size": {"width": 160, "height": 100},
    }
    calibration = {
        "kind": "scalar",
        "unit": "mm",
        "applicability": applicability,
        "parameters": {"scale_per_px": 0.1, "square_size": 1, "spacing_px": 10},
    }
    (tmp_path / "calibration.json").write_text(json.dumps(calibration))
    supplied = recipe()
    supplied.update(
        reference=IMAGE,
        localization={"template": template},
        metric={
            "calibration": {"file": {"root": "exchange", "path": "calibration.json"}},
            "applicability": applicability,
        },
    )
    exported = post(client, "/recipe-exports", {"recipe": supplied, "output": OUTPUT})["result"]
    moved = tmp_path / "new-node"
    moved.mkdir()
    shutil.copyfile(tmp_path / exported["artifacts"][0]["file"]["path"], moved / "part.zip")
    with TestClient(create_app(ServiceSettings(shared_roots={"next": moved}))) as upstream:
        imported = post(
            upstream,
            "/recipe-imports",
            {
                "package": {"root": "next", "path": "part.zip"},
                "output": {"directory": {"root": "next", "path": "imported"}},
            },
        )["result"]
        run = post(
            upstream,
            "/runs",
            {
                "recipe": imported["resolved_recipe"],
                "image": imported["resolved_recipe"]["reference"],
            },
        )["result"]
        width = run["targets"][0]["nodes"][0]["physical_result"]["width"]
        assert width["unit"] == "mm"
        assert width["value"] == pytest.approx(2.0, abs=0.02)
        assert len(imported["assets"]) == 3
        kinds = {item["format"] for item in imported["artifacts"]}
        assert {"npz", "png", "json"} <= kinds


def test_malformed_legacy_tool_container_returns_blocking_compatibility_diagnostic(
    client, tmp_path
):
    (tmp_path / "malformed.json").write_text(
        json.dumps({"version": 3, "template_view": {"tools": "corrupt"}})
    )
    converted = post(
        client,
        "/legacy-project-conversions",
        {
            "project": {"root": "exchange", "path": "malformed.json"},
            "reference": IMAGE,
            "recipe_id": "legacy",
            "recipe_version": "1",
            "output": OUTPUT,
        },
    )["result"]
    assert converted["valid"] is False
    assert converted["diagnostics"][0]["code"] == "malformed_configuration"
    assert converted["manifest"] is None


def test_numeric_payload_declared_storage_is_bounded_before_array_allocation(tmp_path):
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header,
        {
            "descr": "<f8",
            "fortran_order": False,
            "shape": (10000000,),
        },
    )
    npz = io.BytesIO()
    with zipfile.ZipFile(npz, "w") as archive:
        archive.writestr("pixels.npy", header.getvalue())
    payload = npz.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {
        "format": "measure-recipe-package",
        "packaging": "embedded",
        "recipe": recipe(),
        "source_configuration_sha256": "0" * 64,
        "assets": [
            {
                "id": "asset-0",
                "name": "asset-0.npz",
                "kind": "recipe_asset",
                "format": "npz",
                "sha256": digest,
                "size_bytes": len(payload),
                "source": {"root": "old", "path": "numeric.npz", "sha256": digest},
            }
        ],
    }
    with zipfile.ZipFile(tmp_path / "bounded.zip", "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("assets/asset-0.npz", payload)
    with TestClient(
        create_app(
            ServiceSettings(shared_roots={"exchange": tmp_path}, max_input_bytes=1024 * 1024)
        )
    ) as client:
        response = client.post(
            "/api/v2/recipe-imports",
            json={
                "package": {"root": "exchange", "path": "bounded.zip"},
                "output": OUTPUT,
            },
        )
        assert response.status_code == 422
        assert "Declared numeric storage exceeds" in response.json()["error"]["message"]


def test_export_rejects_forged_resolved_run_configuration(client):
    run = post(client, "/runs", {"recipe": recipe(), "image": IMAGE})["result"]
    supplied_recipe = run["resolved_recipe"]
    run = json.loads(json.dumps(run))
    run["resolved_recipe"]["version"] = "forged"
    response = client.post(
        "/api/v2/recipe-exports",
        json={
            "recipe": supplied_recipe,
            "frozen_run": run,
            "output": OUTPUT,
        },
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "replay.configuration_mismatch"


def test_json_report_exports_preserve_producer_identity_for_frozen_replay(client, tmp_path):
    run = post(client, "/runs", {"recipe": recipe(), "image": IMAGE})["result"]
    report = post(client, "/result-exports", {"result": run, "output": OUTPUT})["result"]
    run_file = report["artifacts"][0]["file"]
    stored = json.loads((tmp_path / run_file["path"]).read_bytes())
    assert stored["runtime_identity"] == run["runtime_identity"]
    exported = post(
        client,
        "/recipe-exports",
        {
            "recipe": run["resolved_recipe"],
            "frozen_run_file": run_file,
            "output": OUTPUT,
        },
    )["result"]
    frozen = next(
        item["file"] for item in exported["artifacts"] if item["kind"] == "frozen_run"
    )
    replay = post(client, "/replays", {"frozen_file": frozen, "output": OUTPUT})["result"]
    assert replay["equivalent"] is True
    visual = post(client, "/visualizations", {"result": replay["run"]})["result"]
    assert visual["overlays"]


def test_portable_relocation_preserves_explicit_upstream_asset_version_provenance(client):
    supplied = recipe()
    provenance = {"asset_id": "teaching-source", "version": "upstream-17"}
    supplied["reference"] = {**IMAGE, "provenance": provenance}
    exported = post(client, "/recipe-exports", {"recipe": supplied, "output": OUTPUT})["result"]
    imported = post(
        client,
        "/recipe-imports",
        {
            "package": exported["artifacts"][0]["file"],
            "output": OUTPUT,
        },
    )["result"]
    assert imported["resolved_recipe"]["reference"]["provenance"] == provenance
    assert imported["assets"][0]["provenance"] == provenance

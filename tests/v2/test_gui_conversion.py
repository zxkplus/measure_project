"""GUI model assets convert into runnable V2 recipe bundles."""

import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from measure import MultiTargetWorkflow
from measure_api.v2.app import create_app
from measure_api.v2.gui_conversion import (
    GuiConversionError,
    convert_gui_model,
    load_gui_model,
    reference_to_patch,
)
from measure_api.v2.models import ServiceSettings


def _model(tmp_path, *, alignment_mode=None, radius=60.0):
    """Write a taught GUI model asset and return it with its inspection image."""
    pixels = np.zeros((300, 400), np.uint8)
    cv2.circle(pixels, (200, 150), int(radius), 255, -1)
    taught = MultiTargetWorkflow()
    taught.teach_template(
        pixels,
        center=(150.0, 200.0),
        size=(200.0, 200.0),
        angle_deg=0.0,
        match_score_threshold=0.5,
        angle_range=(-15.0, 15.0),
        angle_step=5.0,
        max_matches=1,
    )
    taught.add_measurement(
        "FitCircle",
        "circle_1",
        center=(100.0, 100.0),
        radius=float(radius),
        measure_length1=12.0,
        measure_length2=3.0,
        num_measures=24,
        sigma=1.0,
        threshold=20.0,
        transition="all",
        selection="nearest",
    )
    directory = tmp_path / "model"
    taught.save(str(directory))
    if alignment_mode is not None:
        payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        payload["alignment"]["mode"] = alignment_mode
        (directory / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    np.save(str(directory / "reference.npy"), pixels)
    return directory, pixels, taught


def _client(bundle):
    app = create_app(ServiceSettings(shared_roots={"gui": bundle.directory}))
    return TestClient(app)


def test_conversion_publishes_a_validated_self_contained_bundle(tmp_path):
    directory, _, _ = _model(tmp_path)
    bundle = convert_gui_model(directory, tmp_path / "bundle", recipe_id="synthetic")

    assert bundle.recipe.id == "synthetic"
    assert bundle.recipe.localization is not None
    assert bundle.recipe.alignment is not None
    for name in (
        "recipe.json",
        "reference.png",
        "localization.npz",
        "run-request.json",
        "bundle.json",
        "README.txt",
    ):
        assert (bundle.directory / name).is_file(), name
    request = json.loads((bundle.directory / "run-request.json").read_text(encoding="utf-8"))
    assert request["recipe_file"] == {"root": "gui", "path": "recipe.json"}
    assert request["image"]["root"] == "gui"
    # The staging namespace never leaks into the published bundle.
    assert not (bundle.directory / ".gui-conversion-staging").exists()


def test_runs_executes_the_converted_model_and_matches_the_gui_engine(tmp_path):
    directory, pixels, taught = _model(tmp_path)
    inspection = tmp_path / "inspection.png"
    cv2.imwrite(str(inspection), pixels)
    bundle = convert_gui_model(
        directory, tmp_path / "bundle", recipe_id="synthetic", image=inspection
    )
    request = json.loads((bundle.directory / "run-request.json").read_text(encoding="utf-8"))

    with _client(bundle) as client:
        response = client.post("/api/v2/runs", json=request)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["state"] == "succeeded"
    result = body["result"]
    assert result["status"] == "completed"
    assert result["valid"] is True
    assert len(result["targets"]) == 1
    node = result["targets"][0]["nodes"][0]
    assert (node["state"], node["valid"]) == ("succeeded", True)
    circle = node["result"]["circle"]
    assert circle["valid"] is True

    expected = taught.measure(pixels)[0].measurements["circle_1"]
    assert circle["center"]["x"] == pytest.approx(expected["center_col"], abs=1.5)
    assert circle["center"]["y"] == pytest.approx(expected["center_row"], abs=1.5)
    assert circle["radius"] == pytest.approx(expected["radius"], abs=1.5)


def test_conversion_keeps_gui_patch_geometry(tmp_path):
    directory, _, _ = _model(tmp_path)
    model = load_gui_model(directory)
    roi = model.config["roi"]
    matrix = reference_to_patch(roi)

    # The GUI patch centre is size/2, so the box centre maps there exactly.
    mapped = matrix @ np.array([roi["center_col"], roi["center_row"], 1.0])
    assert mapped[0] == pytest.approx(roi["width"] / 2.0)
    assert mapped[1] == pytest.approx(roi["height"] / 2.0)


@pytest.mark.parametrize("mode", ["none", "fixed_roi"])
def test_conversion_accepts_search_free_alignment_modes(tmp_path, mode):
    directory, _, _ = _model(tmp_path, alignment_mode=mode)
    bundle = convert_gui_model(directory, tmp_path / "bundle", recipe_id="synthetic")
    assert (bundle.recipe.localization is not None) is (mode == "fixed_roi")
    assert (bundle.recipe.alignment is not None) is (mode == "fixed_roi")


def test_conversion_rejects_an_unsupported_alignment_mode(tmp_path):
    directory, _, _ = _model(tmp_path, alignment_mode="multi_point")
    with pytest.raises(GuiConversionError) as error:
        convert_gui_model(directory, tmp_path / "bundle")
    assert any(
        item.code == "unsupported_alignment_mode" for item in error.value.diagnostics
    )


def test_target_result_to_dict_accepts_plain_mapping_results(tmp_path):
    _, pixels, taught = _model(tmp_path)
    payload = taught.measure(pixels)[0].to_dict()
    assert payload["measurements"]["circle_1"]["center_col"] == pytest.approx(200.0, abs=1.5)

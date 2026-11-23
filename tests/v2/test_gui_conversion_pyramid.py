"""GUI pyramid settings must survive conversion into V2 search semantics.

``pyramid_decimate`` is only an on/off switch in the GUI engine (the level
number is ignored) and it disables decimation entirely when zero. V2's size cap
is always active, so a naive conversion silently decimates models the operator
taught at full resolution.
"""

import json

import cv2
import numpy as np

from fastapi.testclient import TestClient

from measure import MultiTargetWorkflow
from measure_api.v2.app import create_app
from measure_api.v2.gui_conversion import convert_gui_model
from measure_api.v2.models import ServiceSettings

SHARED = "gui"
TEMPLATE_SIDE = 600.0


def _model(tmp_path, *, decimate, cap=400, side=TEMPLATE_SIDE):
    """Teach a GUI model with a template larger than the size cap."""
    pixels = np.zeros((900, 900), np.uint8)
    cv2.circle(pixels, (450, 450), 200, 255, -1)
    taught = MultiTargetWorkflow()
    taught.teach_template(
        pixels,
        center=(450.0, 450.0),
        size=(side, side),
        angle_deg=0.0,
        match_score_threshold=0.5,
        angle_range=(-15.0, 15.0),
        angle_step=5.0,
        max_matches=1,
        decimate_exponent=decimate,
        coarse_template_side=cap,
    )
    taught.add_measurement(
        "FitCircle", "circle_1", center=(300.0, 300.0), radius=60.0,
        measure_length1=12.0, measure_length2=3.0, num_measures=24,
        sigma=1.0, threshold=20.0, transition="all", selection="nearest",
    )
    directory = tmp_path / "model"
    taught.save(str(directory))
    np.save(str(directory / "reference.npy"), pixels)
    return directory, pixels


def _recipe(bundle):
    return json.loads((bundle.directory / "recipe.json").read_text(encoding="utf-8"))


def test_disabled_pyramid_converts_to_no_downsampling(tmp_path):
    directory, _ = _model(tmp_path, decimate=0)
    bundle = convert_gui_model(directory, tmp_path / "bundle", recipe_id="synthetic")
    search = _recipe(bundle)["localization"]["search"]

    assert search["pyramid_levels"] == 0
    assert search["pyramid_max_template_size"] >= TEMPLATE_SIDE
    factor = min(2.0 ** -search["pyramid_levels"],
                 search["pyramid_max_template_size"] / TEMPLATE_SIDE)
    assert factor == 1.0


def test_enabled_pyramid_keeps_the_taught_cap(tmp_path):
    directory, _ = _model(tmp_path, decimate=1, cap=300)
    bundle = convert_gui_model(directory, tmp_path / "bundle", recipe_id="synthetic")
    search = _recipe(bundle)["localization"]["search"]

    assert search["pyramid_levels"] == 0
    assert search["pyramid_max_template_size"] == 300


def test_conversion_flags_the_score_estimator_change(tmp_path):
    directory, _ = _model(tmp_path, decimate=1)
    bundle = convert_gui_model(directory, tmp_path / "bundle", recipe_id="synthetic")

    codes = {item.code for item in bundle.diagnostics}
    assert "score_estimator_changed" in codes


def test_v2_does_not_resample_a_model_taught_with_pyramid_off(tmp_path):
    directory, _ = _model(tmp_path, decimate=0)
    bundle = convert_gui_model(directory, tmp_path / "bundle", recipe_id="synthetic")
    recipe = _recipe(bundle)
    template = recipe["localization"]["template"]

    settings = ServiceSettings(shared_roots={SHARED: bundle.directory}, max_workers=1)
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v2/template-matches", json={
            "template": {"root": template["root"], "path": template["path"]},
            "image": {"root": SHARED, "path": "reference.png"},
            "search": recipe["localization"]["search"],
        })
    assert response.status_code == 200, response.text
    plan = response.json()["result"]["diagnostics"]["search_plan"]
    assert plan["resample"] == "none"
    assert plan["coarse_poses"] == 7  # -15..15 step 5, no decimation

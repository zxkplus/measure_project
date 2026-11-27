"""A template-matching measurement tool keeps its own matcher settings.

These tools build their own TemplatePoint from the stored parameters, which is a
second construction path next to the localization one. When that path ignored the
new settings, a tool taught with "shape" selected ran correlation forever while
the dialog showed otherwise.
"""

import cv2
import numpy as np
import pytest

from measure.multi_target_workflow import MultiTargetWorkflow


def _textured(size=60, seed=3):
    rng = np.random.RandomState(seed)
    patch = cv2.GaussianBlur((rng.rand(size, size) * 255).astype(np.uint8), (5, 5), 1.5)
    cv2.circle(patch, (size // 2, size // 2), size // 4, 0, -1)
    return patch


def _reference(patch, top=120, left=160):
    image = np.zeros((300, 400), np.uint8)
    image[top:top + patch.shape[0], left:left + patch.shape[1]] = patch
    return image


def _workflow():
    """Teach a 60x60 template whose straightened patch is a coordinate frame."""
    patch = _textured()
    workflow = MultiTargetWorkflow()
    workflow.teach_template(_reference(patch), center=(150.0, 190.0),
                            size=(60.0, 60.0), angle_deg=0.0,
                            angle_range=(-5.0, 5.0), coarse_fine=True,
                            coarse_angle_step=5.0)
    return workflow, patch


def _add_tool(workflow, label, **overrides):
    params = {"row": 30.0, "col": 30.0, "template_size": 40,
              "angle_range_half": 5.0}
    params.update(overrides)
    workflow.add_measurement("TemplateMatchPoint", label, **params)
    return workflow._template_match_points[label]


def test_tool_defaults_to_correlation_with_strict_polarity():
    workflow, _ = _workflow()
    point = _add_tool(workflow, "p")
    assert point.algorithm == "ncc"
    assert point.metric == "use_polarity"
    assert point.min_verify_level == 0


@pytest.mark.parametrize("algorithm", ["ncc", "shape"])
def test_the_matcher_choice_reaches_the_tools_own_template_point(algorithm):
    workflow, _ = _workflow()
    point = _add_tool(workflow, "p", algorithm=algorithm)
    assert point.algorithm == algorithm


def test_the_tool_measures_with_the_selected_matcher():
    workflow, patch = _workflow()
    inspection = np.zeros((300, 400), np.uint8)
    inspection[130:130 + patch.shape[0], 170:170 + patch.shape[1]] = patch
    correlated = _add_tool(workflow, "ncc", algorithm="ncc").measure(inspection)
    shaped = _add_tool(workflow, "shape", algorithm="shape").measure(inspection)
    assert correlated["valid"] and shaped["valid"]
    # Same physical position; the scores are on different scales by design.
    assert shaped["matched_row"] == pytest.approx(correlated["matched_row"], abs=1.5)
    assert shaped["matched_col"] == pytest.approx(correlated["matched_col"], abs=1.5)


def test_polarity_and_verification_level_reach_the_tool(tmp_path):
    workflow, _ = _workflow()
    _add_tool(workflow, "p", metric="ignore_global_polarity", min_verify_level=1)
    model = tmp_path / "model"
    workflow.save(str(model))
    restored = MultiTargetWorkflow.load(str(model))
    point = restored._template_match_points["p"]
    assert point.metric == "ignore_global_polarity"
    assert point.min_verify_level == 1


def test_the_matcher_choice_survives_a_save_and_load(tmp_path):
    workflow, _ = _workflow()
    _add_tool(workflow, "shape_tool", algorithm="shape")
    model = tmp_path / "model"
    workflow.save(str(model))
    restored = MultiTargetWorkflow.load(str(model))
    assert restored._template_match_points["shape_tool"].algorithm == "shape"


def test_a_tool_stored_before_the_settings_existed_still_loads(tmp_path):
    """An asset with no matcher keys must keep behaving as correlation."""
    workflow, _ = _workflow()
    _add_tool(workflow, "p", algorithm="shape")
    model = tmp_path / "model"
    workflow.save(str(model))
    import json
    path = model / "manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    for obj in document["objects"]:
        for key in ("algorithm", "metric", "min_verify_level"):
            obj.get("parameters", {}).pop(key, None)
    path.write_text(json.dumps(document), encoding="utf-8")
    restored = MultiTargetWorkflow.load(str(model))
    point = restored._template_match_points["p"]
    assert point.algorithm == "ncc"
    assert point.metric == "use_polarity"
    assert point.min_verify_level == 0

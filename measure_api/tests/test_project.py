"""Tests for MeasureProject SDK."""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from measure_api.project import (
    PHASE_CREATED,
    PHASE_HAS_COMPOSED,
    PHASE_HAS_MEASUREMENTS,
    PHASE_MEASURED,
    PHASE_REF_LOADED,
    PHASE_TEMPLATE_READY,
    MeasureProject,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def proj(temp_dir):
    """Create a MeasureProject with a temp dir."""
    return MeasureProject(temp_dir)


@pytest.fixture
def ref_path(temp_dir):
    """Create a small synthetic reference image."""
    img = np.zeros((200, 200), dtype=np.uint8) + 100
    cv2.circle(img, (100, 100), 60, 200, -1)
    path = os.path.join(temp_dir, "ref.png")
    cv2.imwrite(path, img)
    return path


@pytest.fixture
def insp_path(temp_dir):
    """Create a small synthetic inspection image."""
    img = np.zeros((200, 200), dtype=np.uint8) + 100
    cv2.circle(img, (100, 100), 60, 200, -1)
    cv2.circle(img, (100, 100), 40, 50, -1)
    path = os.path.join(temp_dir, "insp.png")
    cv2.imwrite(path, img)
    return path


@pytest.fixture
def ready_project(proj, ref_path):
    """A project ready for measurements (template set, 2 circles)."""
    proj.load_reference(ref_path)
    proj.set_template(
        center=(100, 100),
        size=(160, 160),
        angle_deg=0,
        preprocessor="raw",
    )
    return proj


# ===========================================================================
# Phase transitions
# ===========================================================================


def test_initial_phase(proj):
    assert proj.phase == _phase_label(PHASE_CREATED)


def test_phase_ref_loaded(proj, ref_path):
    proj.load_reference(ref_path)
    assert proj.phase == _phase_label(PHASE_REF_LOADED)


def test_phase_template_ready(proj, ref_path):
    proj.load_reference(ref_path)
    proj.set_template(center=(100, 100), size=(160, 160), angle_deg=0)
    assert proj.phase == _phase_label(PHASE_TEMPLATE_READY)
    assert proj.has_template
    assert proj.template_shape == (160, 160)


def test_phase_has_measurements(ready_project):
    r = ready_project.add_measurement(
        "FitCircle", "c1",
        {"center": (100, 100), "radius": 60, "measure_length1": 30, "measure_length2": 10},
    )
    assert r["valid"] is not None  # may be valid or not depending on image content
    assert ready_project.phase == _phase_label(PHASE_HAS_MEASUREMENTS)


def test_phase_has_composed(ready_project):
    c1 = ready_project.add_measurement("FitCircle", "c1", {
        "center": (100, 100), "radius": 60, "measure_length1": 30, "measure_length2": 10,
    })
    c2 = ready_project.add_measurement("FitCircle", "c2", {
        "center": (100, 100), "radius": 40, "measure_length1": 20, "measure_length2": 10,
    })
    comp = ready_project.add_composed(
        "PointCircleDistance", "wall",
        {"point_label": "c1", "circle_label": "c2"},
    )
    assert ready_project.phase == _phase_label(PHASE_HAS_COMPOSED)


# ===========================================================================
# CRUD
# ===========================================================================


def test_add_measurement(ready_project):
    r = ready_project.add_measurement(
        "FitCircle", "c1",
        {"center": (100, 100), "radius": 60, "measure_length1": 30, "measure_length2": 10},
    )
    assert "label" in r
    assert "valid" in r
    assert "object_type" in r


def test_add_duplicate_label(ready_project):
    ready_project.add_measurement(
        "FitCircle", "c1",
        {"center": (100, 100), "radius": 60, "measure_length1": 30, "measure_length2": 10},
    )
    with pytest.raises(ValueError, match="Duplicate"):
        ready_project.add_measurement(
            "FitCircle", "c1",
            {"center": (100, 100), "radius": 60, "measure_length1": 30, "measure_length2": 10},
        )


def test_update_measurement(ready_project):
    ready_project.add_measurement(
        "FitCircle", "c1",
        {"center": (100, 100), "radius": 60, "measure_length1": 30, "measure_length2": 10},
    )
    r = ready_project.update_measurement("c1", {"threshold": 15.0})
    assert "valid" in r


def test_get_measurement(ready_project):
    ready_project.add_measurement(
        "FitCircle", "c1",
        {"center": (100, 100), "radius": 60, "measure_length1": 30, "measure_length2": 10},
    )
    m = ready_project.get_measurement("c1")
    assert m["label"] == "c1"
    assert m["object_type"] == "FitCircle"
    assert "params" in m


def test_get_measurement_not_found(ready_project):
    with pytest.raises(ValueError, match="not found"):
        ready_project.get_measurement("nonexistent")


def test_list_measurements(ready_project):
    ready_project.add_measurement(
        "FitCircle", "c1",
        {"center": (100, 100), "radius": 60, "measure_length1": 30, "measure_length2": 10},
    )
    lst = ready_project.list_measurements()
    assert len(lst["measurements"]) == 1
    assert len(lst["composed"]) == 0


def test_remove_measurement(ready_project):
    ready_project.add_measurement(
        "FitCircle", "c1",
        {"center": (100, 100), "radius": 60, "measure_length1": 30, "measure_length2": 10},
    )
    r = ready_project.remove_measurement("c1")
    assert r["status"] == "deleted"
    lst = ready_project.list_measurements()
    assert len(lst["measurements"]) == 0


def test_test_measurement(ready_project):
    """Test without saving."""
    r = ready_project.test_measurement(
        "FitCircle", "test_circle",
        {"center": (100, 100), "radius": 60, "measure_length1": 30, "measure_length2": 10},
    )
    assert "valid" in r
    # Should NOT be in the project
    lst = ready_project.list_measurements()
    assert len(lst["measurements"]) == 0


# ===========================================================================
# Composed measurements
# ===========================================================================


def test_add_composed(ready_project):
    ready_project.add_measurement("FitCircle", "c1", {
        "center": (100, 100), "radius": 60, "measure_length1": 30, "measure_length2": 10,
    })
    ready_project.add_measurement("FitCircle", "c2", {
        "center": (100, 100), "radius": 40, "measure_length1": 20, "measure_length2": 10,
    })
    r = ready_project.add_composed(
        "PointCircleDistance", "wall",
        {"point_label": "c1", "circle_label": "c2"},
    )
    assert "valid" in r
    assert "dependencies" in r


def test_add_composed_missing_dep(ready_project):
    ready_project.add_measurement("FitCircle", "c1", {
        "center": (100, 100), "radius": 60, "measure_length1": 30, "measure_length2": 10,
    })
    with pytest.raises(ValueError, match="not found"):
        ready_project.add_composed(
            "TwoPointsDistance", "gap",
            {"point_a_label": "nonexistent", "point_b_label": "c2"},
        )


def test_remove_composed(ready_project):
    ready_project.add_measurement("FitCircle", "c1", {
        "center": (100, 100), "radius": 60, "measure_length1": 30, "measure_length2": 10,
    })
    ready_project.add_measurement("FitCircle", "c2", {
        "center": (100, 100), "radius": 40, "measure_length1": 20, "measure_length2": 10,
    })
    ready_project.add_composed(
        "PointCircleDistance", "wall",
        {"point_label": "c1", "circle_label": "c2"},
    )
    r = ready_project.remove_composed("wall")
    assert r["status"] == "deleted"


# ===========================================================================
# Cascade delete
# ===========================================================================


def test_cascade_delete(ready_project):
    ready_project.add_measurement("FitCircle", "c1", {
        "center": (100, 100), "radius": 60, "measure_length1": 30, "measure_length2": 10,
    })
    ready_project.add_measurement("FitCircle", "c2", {
        "center": (100, 100), "radius": 40, "measure_length1": 20, "measure_length2": 10,
    })
    ready_project.add_composed(
        "PointCircleDistance", "wall",
        {"point_label": "c1", "circle_label": "c2"},
    )
    r = ready_project.remove_measurement("c1")
    assert r["status"] == "deleted"
    assert len(r.get("cascaded", [])) == 1
    assert r["cascaded"][0]["label"] == "wall"


# ===========================================================================
# DAG
# ===========================================================================


def test_dag_empty(ready_project):
    dag = ready_project.get_dag()
    assert len(dag["nodes"]) == 0
    assert len(dag["edges"]) == 0
    assert dag["execution_order"] == []
    assert dag["is_valid"] is True


def test_dag_with_measurements(ready_project):
    ready_project.add_measurement("FitCircle", "c1", {
        "center": (100, 100), "radius": 60, "measure_length1": 30, "measure_length2": 10,
    })
    ready_project.add_measurement("FitCircle", "c2", {
        "center": (100, 100), "radius": 40, "measure_length1": 20, "measure_length2": 10,
    })
    ready_project.add_composed(
        "PointCircleDistance", "wall",
        {"point_label": "c1", "circle_label": "c2"},
    )
    dag = ready_project.get_dag()
    assert len(dag["nodes"]) == 3
    assert len(dag["edges"]) == 2
    assert dag["is_valid"] is True
    # c1, c2 should be before wall
    ci1 = dag["execution_order"].index("c1")
    ci2 = dag["execution_order"].index("c2")
    wi = dag["execution_order"].index("wall")
    assert ci1 < wi
    assert ci2 < wi


def test_dag_text_format(ready_project):
    dag = ready_project.get_dag(format="text")
    assert "text" in dag
    assert "DAG Summary" in dag["text"]


# ===========================================================================
# Save / Load
# ===========================================================================


def test_save_load(ready_project):
    # Add measurements
    ready_project.add_measurement("FitCircle", "c1", {
        "center": (100, 100), "radius": 60, "measure_length1": 30, "measure_length2": 10,
    })
    ready_project.add_measurement("FitCircle", "c2", {
        "center": (100, 100), "radius": 40, "measure_length1": 20, "measure_length2": 10,
    })
    ready_project.add_composed(
        "PointCircleDistance", "wall",
        {"point_label": "c1", "circle_label": "c2"},
    )

    # Save
    save_info = ready_project.save()
    assert "saved_to" in save_info
    assert os.path.isdir(save_info["saved_to"])

    # Load into a new project
    proj2 = MeasureProject(ready_project.project_dir)
    load_info = proj2.load()
    assert load_info["num_measurements"] == 2
    assert load_info["num_composed"] == 1


def test_save_load_empty_project(proj):
    """Save and load an empty project (no template)."""
    save_info = proj.save()
    proj2 = MeasureProject(proj.project_dir)
    load_info = proj2.load()
    assert load_info["num_measurements"] == 0


# ===========================================================================
# Validation
# ===========================================================================


def test_validate_dag(ready_project):
    assert ready_project._validate_dag() is True
    ready_project.add_measurement("FitCircle", "c1", {
        "center": (100, 100), "radius": 60, "measure_length1": 30, "measure_length2": 10,
    })
    assert ready_project._validate_dag() is True
    # Manually inject a broken composed def to test DAG validation
    ready_project._composed_defs.append({
        "composed_type": "TwoPointsDistance",
        "label": "gap",
        "deps": {"point_a_label": "c1", "point_b_label": "nonexistent"},
    })
    ready_project._update_phase()
    assert ready_project._validate_dag() is False


# ===========================================================================
# Status
# ===========================================================================


def test_status(proj):
    s = proj.status()
    assert "phase" in s
    assert "has_reference" in s
    assert "has_template" in s


# ===========================================================================
# Helpers
# ===========================================================================


def _phase_label(phase: str) -> str:
    labels = {
        PHASE_CREATED: "created",
        PHASE_REF_LOADED: "reference_loaded",
        PHASE_TEMPLATE_READY: "template_ready",
        PHASE_HAS_MEASUREMENTS: "has_measurements",
        PHASE_HAS_COMPOSED: "has_composed",
        PHASE_MEASURED: "measured",
    }
    return labels.get(phase, phase)

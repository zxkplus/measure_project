"""Headless tests for GUI project save/load (``measure_gui.project_manager``).

``ProjectManager.save_project`` / ``load_project`` take a live ``MeasureApp``,
which needs a display.  These tests drive them with a recording stand-in instead,
so the project.json <-> model-manifest split is verifiable without tkinter.

The stand-in only has to expose the attributes and methods those two functions
actually touch; every call is recorded so tests can assert what was restored and
from where.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from measure.multi_target_workflow import MultiTargetWorkflow
from measure_gui.project_manager import ProjectManager

# A reference image with enough texture for template matching to lock on.
REF_H, REF_W = 200, 240
ROI_CENTER = (100.0, 120.0)
ROI_SIZE = (120.0, 140.0)          # (height, width)


def _reference(seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(30, 90, (REF_H, REF_W), dtype=np.uint8)
    image[60:140, 70:130] = rng.integers(180, 255, (80, 60), dtype=np.uint8)
    return image


class _RecordingCanvas:
    """Stand-in for ImageCanvas that records ROI/view interactions."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.image = None
        self.roi_center = None
        self.roi_size = None
        self.roi_angle = None
        self.roi_confirmed = False
        self.control_points: list = []

    def load_image(self, image):
        self.image = image
        self.calls.append(("load_image", image.shape))

    def set_roi(self, center, size, angle_deg):
        self.roi_center, self.roi_size, self.roi_angle = center, size, angle_deg
        self.calls.append(("set_roi", tuple(center), tuple(size), float(angle_deg)))

    def confirm_roi(self):
        self.roi_confirmed = True
        self.calls.append(("confirm_roi",))

    def set_control_points(self, points):
        self.control_points = list(points)
        self.calls.append(("set_control_points", list(points)))

    def get_control_points(self):
        return list(self.control_points)

    def get_view_state(self):
        return {"zoom": 1.5, "offset_x": 3.0, "offset_y": 4.0}

    def set_view_state(self, state):
        self.calls.append(("set_view_state", dict(state)))


class _RecordingToolPanel:
    """Stand-in for ToolPanel."""

    def __init__(self):
        self.calls: list[tuple] = []
        self._tool_visibility: dict = {}
        self._preproc_var = _Var("raw")
        self._score_var = _Var(0.5)
        self._angle_var = _Var("±30°")
        self._max_matches_var = _Var(0)
        self._overlap_var = _Var(0.3)
        self._angle_step_var = _Var(1.0)
        self._alignment_var = _Var("旋转框")

    def clear_tool_list(self):
        self.calls.append(("clear_tool_list",))

    def restore_tool_list(self, tools, order):
        self.calls.append(("restore_tool_list", [t["label"] for t in tools],
                           list(order)))

    def get_tool_list_order(self):
        return ["line_1"]

    def set_matching_params(self, preprocessor_type, score_threshold,
                            angle_range_half, max_matches, overlap=0.3,
                            angle_step_deg=1.0):
        self.calls.append(("set_matching_params", preprocessor_type,
                           float(score_threshold), float(angle_range_half),
                           int(max_matches), float(overlap), float(angle_step_deg)))

    def get_alignment_mode(self):
        return "multi_point" if self._alignment_var.get() == "多点仿射" else "single_box"

    def set_alignment_mode(self, mode):
        self._alignment_var.set("多点仿射" if mode == "multi_point" else "旋转框")
        self.calls.append(("set_alignment_mode", mode))

    def set_template_created(self, flag):
        self.calls.append(("set_template_created", bool(flag)))


class _Var:
    """Minimal tkinter.StringVar replacement."""

    def __init__(self, value=None):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class _RecordingTemplateView:
    def __init__(self):
        self.calls: list[tuple] = []

    def load_template(self, image, clear_tools=True):
        self.calls.append(("load_template", image.shape, bool(clear_tools)))

    def get_state(self):
        return {"scale": 2.0, "zoom_factor": 1.0, "offset_x": 0.0, "offset_y": 0.0}

    def set_state(self, state):
        self.calls.append(("set_state", dict(state)))

    def set_tool_visibility(self, visibility):
        self.calls.append(("set_tool_visibility", dict(visibility)))


class FakeApp:
    """Stand-in exposing exactly what save_project/load_project touch."""

    def __init__(self, workflow: MultiTargetWorkflow | None = None):
        self._workflow = workflow
        self._reference_image = None
        self._inspection_image = None
        self._reference_image_path = "/somewhere/ref.png"
        self._inspection_image_path = None
        self._calibration = None
        self._calibration_board_image = None
        self._current_project_dir = None
        self._current_project_path = None
        self._created_at = None
        self._status_text = _Var("")
        self.root = _FakeRoot()
        self.tool_panel = _RecordingToolPanel()
        self.template_view = _RecordingTemplateView()
        self.result_panel = type("RP", (), {"_export_csv": lambda self: None})()
        self._ref_canvas = _RecordingCanvas()
        self._insp_canvas = _RecordingCanvas()
        self.gui_state = {"window_geometry": "1200x800+0+0"}
        self.calls: list[tuple] = []

    # --- state collectors used by save_project -----------------------------
    def get_matching_state(self):
        return {
            "preprocessor_type": self.tool_panel._preproc_var.get(),
            "match_score_threshold": self.tool_panel._score_var.get(),
            "angle_range_deg": 30.0,
            "angle_step_deg": self.tool_panel._angle_step_var.get(),
            "max_matches": self.tool_panel._max_matches_var.get(),
            "overlap": self.tool_panel._overlap_var.get(),
        }

    def _get_roi_state(self):
        canvas = self._ref_canvas
        if canvas.roi_center is None:
            return {}
        return {
            "center_row": canvas.roi_center[0],
            "center_col": canvas.roi_center[1],
            "height": canvas.roi_size[0],
            "width": canvas.roi_size[1],
            "angle_deg": canvas.roi_angle,
            "confirmed": canvas.roi_confirmed,
            "alignment_mode": self.tool_panel.get_alignment_mode(),
        }

    def get_alignment_state(self):
        return {"mode": self.tool_panel.get_alignment_mode()}

    def get_calibration_state(self):
        return dict(self._calibration) if self._calibration else {}

    def get_gui_state(self):
        return dict(self.gui_state)

    def get_tool_list_order(self):
        return self.tool_panel.get_tool_list_order()

    def restore_gui_state(self, gui_state):
        self.calls.append(("restore_gui_state", dict(gui_state)))

    def _update_calibration_status(self):
        self.calls.append(("update_calibration_status",))


class _FakeRoot:
    def __init__(self):
        self.title_text = None

    def title(self, text):
        self.title_text = text


@pytest.fixture
def taught_workflow():
    wf = MultiTargetWorkflow()
    wf.teach_template(_reference(), center=ROI_CENTER, size=ROI_SIZE,
                      angle_deg=12.5, match_score_threshold=0.4,
                      angle_range=(-25.0, 25.0), angle_step=1.5,
                      max_matches=3, overlap=0.25)
    wf.add_measurement("FitLine", "line_1", start=(10, 20), end=(60, 90),
                       measure_length1=8.0, measure_length2=3.0)
    return wf


@pytest.fixture(autouse=True)
def _isolate_recent_projects_file(tmp_path, monkeypatch):
    """Keep save/load off the real ~/.measure_gui_recent.json."""
    monkeypatch.setattr(ProjectManager, "RECENT_FILE",
                        str(tmp_path / ".measure_gui_recent.json"))


def _saved_project(tmp_path, workflow):
    """Save a project with a fully-taught workflow and return paths + app."""
    app = FakeApp(workflow)
    app._reference_image = workflow.reference_image
    app._ref_canvas.set_roi(ROI_CENTER, ROI_SIZE, 12.5)
    app._ref_canvas.confirm_roi()
    project_dir = str(tmp_path / "proj")
    ProjectManager.save_project(project_dir, app)
    return project_dir, app


# ---------------------------------------------------------------------------
# Save side
# ---------------------------------------------------------------------------


class TestSaveProject:
    def test_save_writes_both_project_json_and_model_manifest(self, tmp_path,
                                                              taught_workflow):
        project_dir, _ = _saved_project(tmp_path, taught_workflow)

        names = set(os.listdir(project_dir))
        assert {"project.json", "manifest.json", "reference.npy"} <= names

    def test_project_json_does_not_duplicate_measurement_config(self, tmp_path,
                                                               taught_workflow):
        """ROI/matching/alignment live in the model manifest, not project.json."""
        project_dir, _ = _saved_project(tmp_path, taught_workflow)

        project = json.load(open(os.path.join(project_dir, "project.json")))
        model = json.load(open(os.path.join(project_dir, "manifest.json")))

        # project.json must no longer carry any copy of the measurement config
        for key in ("matching", "roi", "alignment"):
            assert key not in project, f"project.json still duplicates '{key}'"

        # ...the model asset owns it instead (note its own naming: the ROI lives
        # under 'alignment', the matcher settings under 'matching')
        assert model["alignment"]["box_center"] == [ROI_CENTER[0], ROI_CENTER[1]]
        assert model["alignment"]["box_size"] == [ROI_SIZE[0], ROI_SIZE[1]]
        assert model["alignment"]["mode"] == "single_box"
        assert model["matching"]["angle_range"] == [-25.0, 25.0]
        assert model["matching"]["match_score_threshold"] == pytest.approx(0.4)
        assert model["matching"]["max_matches"] == 3
        assert model["matching"]["overlap"] == pytest.approx(0.25)
        assert model["matching"]["angle_step"] == pytest.approx(1.5)

    def test_project_json_keeps_purely_gui_state(self, tmp_path, taught_workflow):
        project_dir, _ = _saved_project(tmp_path, taught_workflow)
        project = json.load(open(os.path.join(project_dir, "project.json")))

        assert set(project) >= {
            "version", "project_name", "created_at", "updated_at",
            "reference_image_path", "view", "gui", "calibration",
            "tool_list_order", "tool_visibility",
        }

    def test_gui_interaction_state_is_preserved_separately(self, tmp_path,
                                                          taught_workflow):
        """'confirmed' is GUI interaction state the model asset does not carry."""
        project_dir, _ = _saved_project(tmp_path, taught_workflow)
        project = json.load(open(os.path.join(project_dir, "project.json")))

        assert project["view"]["roi_confirmed"] is True


# ---------------------------------------------------------------------------
# Load side
# ---------------------------------------------------------------------------


class TestLoadProject:
    def test_roi_restored_from_the_model_manifest(self, tmp_path, taught_workflow):
        project_dir, _ = _saved_project(tmp_path, taught_workflow)
        app = FakeApp()
        ProjectManager.load_project(project_dir, app)

        # patch frame origin is (ROI_CENTER - ROI_SIZE/2)
        assert app._ref_canvas.roi_center == pytest.approx(ROI_CENTER)
        assert app._ref_canvas.roi_size == pytest.approx(ROI_SIZE)
        assert app._ref_canvas.roi_angle == pytest.approx(12.5)

    def test_matching_params_restored_from_the_model_manifest(self, tmp_path,
                                                             taught_workflow):
        project_dir, _ = _saved_project(tmp_path, taught_workflow)
        app = FakeApp()
        ProjectManager.load_project(project_dir, app)

        calls = [c for c in app.tool_panel.calls if c[0] == "set_matching_params"]
        assert len(calls) == 1
        _, preproc, threshold, half_range, max_matches, overlap, step = calls[0]
        assert preproc == "raw"
        assert threshold == pytest.approx(0.4)
        assert half_range == pytest.approx(25.0)   # range pair -> half range
        assert max_matches == 3
        assert overlap == pytest.approx(0.25)
        assert step == pytest.approx(1.5)

    def test_roi_confirmed_flag_still_comes_from_project_json(self, tmp_path,
                                                             taught_workflow):
        project_dir, _ = _saved_project(tmp_path, taught_workflow)
        app = FakeApp()
        ProjectManager.load_project(project_dir, app)

        assert app._ref_canvas.roi_confirmed is True
        assert ("confirm_roi",) in app._ref_canvas.calls

    def test_single_source_of_truth_no_second_roi_write(self, tmp_path,
                                                       taught_workflow):
        """set_roi must be applied exactly once — not once per config source."""
        project_dir, _ = _saved_project(tmp_path, taught_workflow)
        app = FakeApp()
        ProjectManager.load_project(project_dir, app)

        roi_calls = [c for c in app._ref_canvas.calls if c[0] == "set_roi"]
        assert len(roi_calls) == 1, f"ROI applied {len(roi_calls)} times"

    def test_load_survives_a_save_load_save_cycle(self, tmp_path, taught_workflow):
        """Round-tripping twice must not drift the measurement config."""
        first, _ = _saved_project(tmp_path, taught_workflow)

        app = FakeApp()
        ProjectManager.load_project(first, app)
        second_dir = str(tmp_path / "proj2")
        ProjectManager.save_project(second_dir, app)

        a = json.load(open(os.path.join(first, "manifest.json")))
        b = json.load(open(os.path.join(second_dir, "manifest.json")))
        assert a["alignment"] == b["alignment"]
        assert a["matching"] == b["matching"]
        assert a["objects"] == b["objects"]


# ---------------------------------------------------------------------------
# Backwards compatibility with old project.json files
# ---------------------------------------------------------------------------


class TestLegacyProjectCompatibility:
    def test_old_project_json_with_duplicated_config_still_loads(self, tmp_path,
                                                                taught_workflow):
        """Projects saved before the split keep working (fallback path)."""
        project_dir, _ = _saved_project(tmp_path, taught_workflow)
        path = os.path.join(project_dir, "project.json")
        project = json.load(open(path))

        # re-add the legacy duplicated blocks, with values that DISAGREE with
        # the model manifest, to prove which source wins
        project["roi"] = {"center_row": 1.0, "center_col": 2.0,
                          "height": 3.0, "width": 4.0, "angle_deg": 5.0,
                          "confirmed": True, "alignment_mode": "single_box"}
        project["matching"] = {"preprocessor_type": "canny",
                              "match_score_threshold": 0.99,
                              "angle_range_deg": 7.0, "max_matches": 42,
                              "overlap": 0.9, "angle_step_deg": 0.1}
        project.pop("view", None)
        json.dump(project, open(path, "w"))

        app = FakeApp()
        ProjectManager.load_project(project_dir, app)

        # the model manifest is authoritative for geometry/matching ...
        assert app._ref_canvas.roi_center == pytest.approx(ROI_CENTER)
        calls = [c for c in app.tool_panel.calls if c[0] == "set_matching_params"]
        assert calls[0][2] == pytest.approx(0.4)
        # ... and the legacy roi.confirmed is still honoured for the GUI flag
        assert app._ref_canvas.roi_confirmed is True

    def test_missing_model_manifest_reports_actionably(self, tmp_path,
                                                     taught_workflow):
        project_dir, _ = _saved_project(tmp_path, taught_workflow)
        os.remove(os.path.join(project_dir, "manifest.json"))

        app = FakeApp()
        with pytest.raises(Exception) as excinfo:
            ProjectManager.load_project(project_dir, app)
        assert "migrate_models_to_manifest" in str(excinfo.value)

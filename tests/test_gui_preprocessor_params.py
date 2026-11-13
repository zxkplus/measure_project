"""GUI tests for the explicit "apply" gate on preprocessing parameters.

The parameters end up inside the model (they change the template pixels), so
editing a widget must not be enough -- an operator has to press the apply button
and be told what happened.  These tests pin that contract.
"""

from __future__ import annotations

import os

import pytest

_tk = pytest.importorskip("tkinter")


def _display_available() -> bool:
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False
    try:
        root = _tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


gui_only = pytest.mark.skipif(not _display_available(),
                              reason="requires an X/Wayland display")


@pytest.fixture()
def panel():
    from measure_gui.tool_panel import ToolPanel

    root = _tk.Tk()
    root.withdraw()
    widget = ToolPanel(root)
    widget.pack()
    root.update()
    try:
        yield widget
    finally:
        try:
            root.destroy()
        except Exception:
            pass


CLAHE_APPLIED = {"type": "clahe", "clip_limit": 6.0, "tile_grid_size": [4, 6]}


@gui_only
class TestApplyGate:
    def test_default_committed_config_is_raw(self, panel):
        assert panel.get_preprocessor_config() == {"type": "raw"}

    def test_switching_type_builds_that_types_controls(self, panel):
        panel._preproc_var.set("clahe")
        panel._on_preproc_type_changed()
        assert set(panel._preproc_vars) == {"clip_limit", "tile_x", "tile_y"}

        panel._preproc_var.set("threshold")
        panel._on_preproc_type_changed()
        assert set(panel._preproc_vars) == {"threshold", "mode"}

    def test_switching_type_does_not_commit(self, panel):
        panel._preproc_var.set("clahe")
        panel._on_preproc_type_changed()
        assert panel.get_preprocessor_config() == {"type": "raw"}
        assert "应用" in panel._preproc_status.get()

    def test_editing_parameters_does_not_commit(self, panel):
        panel._preproc_var.set("clahe")
        panel._on_preproc_type_changed()
        panel._preproc_vars["clip_limit"].set(6.0)
        panel._preproc_vars["tile_x"].set(4)
        panel._preproc_vars["tile_y"].set(6)

        # Still the old config: the widgets are a draft.
        assert panel.get_preprocessor_config() == {"type": "raw"}

        panel._on_apply_preprocessor()

        assert panel.get_preprocessor_config() == CLAHE_APPLIED
        assert "已应用" in panel._preproc_status.get()

    def test_apply_notifies_the_application(self, panel):
        seen = []
        panel.on_preprocessor_apply = seen.append

        panel._preproc_var.set("clahe")
        panel._on_preproc_type_changed()
        panel._preproc_vars["clip_limit"].set(6.0)
        panel._preproc_vars["tile_x"].set(4)
        panel._preproc_vars["tile_y"].set(6)
        panel._on_apply_preprocessor()

        assert seen == [CLAHE_APPLIED]

    def test_editing_a_parameter_marks_the_draft_dirty(self, panel):
        panel.apply_preprocessor_config(CLAHE_APPLIED)
        panel._preproc_status.set("")

        panel._preproc_vars["clip_limit"].set(9.0)

        # Exercises the variable trace.  A broken callback leaves this empty,
        # because Tk swallows exceptions raised inside trace handlers.
        assert "应用" in panel._preproc_status.get()
        assert panel.get_preprocessor_config() == CLAHE_APPLIED

    def test_draft_survives_type_switching_back(self, panel):
        panel._preproc_var.set("clahe")
        panel._on_preproc_type_changed()
        panel._preproc_vars["clip_limit"].set(6.0)
        panel._on_apply_preprocessor()

        panel._preproc_var.set("raw")
        panel._on_preproc_type_changed()
        panel._preproc_var.set("clahe")
        panel._on_preproc_type_changed()

        assert panel._preproc_vars["clip_limit"].get() == pytest.approx(6.0)

    def test_restoring_a_config_populates_widgets(self, panel):
        panel.apply_preprocessor_config(CLAHE_APPLIED)

        assert panel._preproc_var.get() == "clahe"
        assert panel._preproc_vars["clip_limit"].get() == pytest.approx(6.0)
        assert panel._preproc_vars["tile_x"].get() == 4
        assert panel._preproc_vars["tile_y"].get() == 6
        assert panel.get_preprocessor_config() == CLAHE_APPLIED
        # Nothing pending right after a restore.
        assert panel._preproc_status.get() == ""

    def test_restore_leaves_matching_parameters_alone(self, panel):
        panel.set_matching_params(
            "clahe", 0.62, 20.0, 2, overlap=0.2, angle_step_deg=0.5,
            preprocessor={"type": "clahe", "clip_limit": 8.0,
                          "tile_grid_size": [4, 4]},
        )
        assert panel.get_preprocessor_config()["clip_limit"] == pytest.approx(8.0)
        assert panel._preproc_vars["tile_y"].get() == 4
        assert panel._score_var.get() == pytest.approx(0.62)
        assert panel._max_matches_var.get() == 2

    def test_legacy_type_only_restore_still_works(self, panel):
        """Older callers pass only a type; the built-in defaults fill the rest."""
        panel.set_matching_params("canny", 0.5, 30.0, 0)
        assert panel._preproc_var.get() == "canny"
        assert set(panel._preproc_vars) == {"t1", "t2"}
        assert panel.get_preprocessor_config() == {
            "type": "canny", "t1": 50.0, "t2": 150.0
        }


@gui_only
class TestApplicationWiring:
    def test_teach_path_uses_the_applied_config(self):
        from measure_gui.app import MeasureApp

        root = _tk.Tk()
        root.withdraw()
        app = MeasureApp(root)
        try:
            app.tool_panel.apply_preprocessor_config(CLAHE_APPLIED)
            app._on_preprocessor_apply(CLAHE_APPLIED)

            preprocessor = app._build_display_and_teach_preprocessor()

            assert preprocessor.serialize() == CLAHE_APPLIED
        finally:
            root.destroy()

    def test_apply_while_a_template_exists_warns(self):
        from measure_gui.app import MeasureApp
        from measure.multi_target_workflow import MultiTargetWorkflow

        root = _tk.Tk()
        root.withdraw()
        app = MeasureApp(root)
        try:
            import numpy as np

            reference = np.full((120, 140), 180, dtype=np.uint8)
            workflow = MultiTargetWorkflow()
            workflow.teach_template(reference, center=(60.0, 70.0),
                                    size=(40.0, 40.0), angle_deg=0.0)
            app._workflow = workflow

            app._on_preprocessor_apply(CLAHE_APPLIED)

            status = app._status_text.get()
            assert "已保存" in status
            assert "重新创建模板" in status
        finally:
            root.destroy()
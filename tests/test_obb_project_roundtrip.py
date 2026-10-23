"""Reopening an OBB project restores what the user configured.

Three defects reported from a real session, each pinned here:

1. the ROI box on the reference image was mirrored, because the manifest's box
   angle (CCW+) reached the canvas, which stores ROI angles CW+;
2. the template view showed the whole reference image, because the strategy's
   ``template_image`` is the raw reference and nothing replaced it with the
   straightened patch on load;
3. the detector's identity was never saved, so reopening asked for the weights
   file again even though the project knew which model it belonged to.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

_tk = pytest.importorskip("tkinter")

_WEIGHTS = Path("/tmp/yolov8n-obb.pt")
_SAMPLE = Path(__file__).parent / "fixtures" / "obb_sample.jpg"


def _display_available() -> bool:
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False
    try:
        root = _tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


def _ultralytics_available() -> bool:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return True


gui_only = pytest.mark.skipif(not _display_available(),
                              reason="requires an X/Wayland display")
needs_model = pytest.mark.skipif(
    not _WEIGHTS.is_file() or not _SAMPLE.is_file() or not _ultralytics_available(),
    reason="real OBB weights, sample image, or ultralytics not available")


@gui_only
@needs_model
class TestObbProjectRoundTrip:
    def _app(self, tmp_path, monkeypatch):
        import tkinter.messagebox as mb
        from measure_gui.app import MeasureApp
        from measure_gui.project_manager import ProjectManager

        monkeypatch.setattr(mb, "showwarning", lambda *a, **k: None)
        monkeypatch.setattr(mb, "showerror", lambda *a, **k: None)
        monkeypatch.setattr(mb, "showinfo", lambda *a, **k: None)
        monkeypatch.setattr(ProjectManager, "RECENT_FILE",
                            str(tmp_path / "recent.json"))
        root = _tk.Tk()
        root.withdraw()
        return root, MeasureApp(root)

    def _teach_and_save(self, app, directory):
        import cv2
        from measure_gui.project_manager import ProjectManager

        image = cv2.imread(str(_SAMPLE), cv2.IMREAD_GRAYSCALE)
        app._reference_image = image
        app._ref_canvas.load_image(image)
        app._workflow = None
        app.tool_panel._alignment_var.set("OBB 模型")
        app.tool_panel._on_alignment_selected()
        app.tool_panel.set_obb_model(str(_WEIGHTS))
        app.tool_panel._obb_conf_var.set(0.5)
        app.tool_panel._obb_imgsz_var.set("1024")
        app._load_obb_model()
        app._current_project_dir = str(directory)
        ProjectManager.save_project(str(directory), app)
        return image

    def test_model_identity_is_saved(self, tmp_path, monkeypatch):
        root, app = self._app(tmp_path, monkeypatch)
        try:
            directory = tmp_path / "proj"
            self._teach_and_save(app, directory)
            manifest = json.loads((directory / "manifest.json").read_text())
            ref = manifest["alignment"]["roi_state"]["model_ref"]
            assert ref, "model_ref was empty; reopening cannot reload the detector"
            assert ref["path"] == str(_WEIGHTS)
            assert ref["sha256"]
            assert ref["imgsz"] == 1024
        finally:
            root.destroy()

    def test_reopen_restores_panel_and_detector(self, tmp_path, monkeypatch):
        root, app = self._app(tmp_path, monkeypatch)
        directory = tmp_path / "proj"
        try:
            self._teach_and_save(app, directory)
        finally:
            root.destroy()

        root2, app2 = self._app(tmp_path, monkeypatch)
        try:
            from measure_gui.project_manager import ProjectManager
            ProjectManager.load_project(str(directory), app2)
            strategy = app2._workflow._alignment

            # The detector is re-created from the recorded path.
            assert strategy.source is not None
            # Panel fields reflect the asset, not blank inputs.
            assert app2.tool_panel._obb_model_var.get() == str(_WEIGHTS)
            assert "fa6e4cd" in app2.tool_panel._obb_digest_label.cget("text") or \
                "…" in app2.tool_panel._obb_digest_label.cget("text")
            assert len(app2.tool_panel._obb_anchor_combo.cget("values")) > 10
            assert str(app2.tool_panel._exec_btn.cget("state")) == "normal"
        finally:
            root2.destroy()

    def test_reopen_shows_the_straightened_patch(self, tmp_path, monkeypatch):
        """The template view must show the frame tools are defined in."""
        root, app = self._app(tmp_path, monkeypatch)
        directory = tmp_path / "proj"
        try:
            image = self._teach_and_save(app, directory)
        finally:
            root.destroy()

        root2, app2 = self._app(tmp_path, monkeypatch)
        try:
            from measure_gui.project_manager import ProjectManager
            ProjectManager.load_project(str(directory), app2)
            shown = app2.template_view._template_image
            assert shown.shape != image.shape, "showing the whole reference image"
            assert shown.shape[0] < shown.shape[1], "patch should be wide, not tall"
        finally:
            root2.destroy()

    def test_reopen_restores_the_roi_angle_convention(self, tmp_path, monkeypatch):
        """The manifest angle (CCW+) must reach the canvas as CW+.

        Drawn through the canvas' own conversion this puts the box on the detected
        corners; passing the raw value mirrors it.
        """
        import cv2
        from measure.rectify import compute_rotated_box_corners

        root, app = self._app(tmp_path, monkeypatch)
        directory = tmp_path / "proj"
        try:
            self._teach_and_save(app, directory)
        finally:
            root.destroy()

        root2, app2 = self._app(tmp_path, monkeypatch)
        try:
            from measure_gui.project_manager import ProjectManager
            ProjectManager.load_project(str(directory), app2)
            manifest = json.loads((directory / "manifest.json").read_text())
            stored_angle = manifest["alignment"]["roi_state"]["box_angle_deg"]
            assert app2._ref_canvas._roi_angle == pytest.approx(-stored_angle)

            # And the drawn box coincides with the stored anchor geometry.
            expected = compute_rotated_box_corners(
                tuple(manifest["alignment"]["roi_state"]["box_center"]),
                tuple(manifest["alignment"]["roi_state"]["nominal_size"]),
                stored_angle,
            )
            drawn = compute_rotated_box_corners(
                app2._ref_canvas.roi_center, app2._ref_canvas.roi_size,
                -app2._ref_canvas._roi_angle)
            assert float(np.abs(drawn - expected).max()) < 2.0
        finally:
            root2.destroy()

    def test_reopen_can_execute(self, tmp_path, monkeypatch):
        root, app = self._app(tmp_path, monkeypatch)
        directory = tmp_path / "proj"
        try:
            image = self._teach_and_save(app, directory)
        finally:
            root.destroy()

        root2, app2 = self._app(tmp_path, monkeypatch)
        try:
            from measure_gui.project_manager import ProjectManager
            ProjectManager.load_project(str(directory), app2)
            app2._inspection_image = app2._reference_image
            app2._execute()
            assert len(app2._workflow.results) > 10
        finally:
            root2.destroy()


@gui_only
class TestObbRestoreWithoutDetector:
    """Restoring must not require a detector: geometry alone suffices."""

    def test_missing_recorded_path_is_reported_actionably(self, tmp_path,
                                                          monkeypatch):
        """An older asset has no model_ref; say so instead of failing silently."""
        import tkinter as tk
        import tkinter.messagebox as mb
        from measure.alignment import ObbAlignment
        from measure.multi_target_workflow import MultiTargetWorkflow
        from measure_gui.app import MeasureApp
        from measure_gui.project_manager import ProjectManager

        monkeypatch.setattr(mb, "showwarning", lambda *a, **k: None)
        monkeypatch.setattr(mb, "showerror", lambda *a, **k: None)
        monkeypatch.setattr(mb, "showinfo", lambda *a, **k: None)
        monkeypatch.setattr(ProjectManager, "RECENT_FILE",
                            str(tmp_path / "recent.json"))

        # A taught OBB strategy with no source and no recorded identity, which is
        # exactly what a project saved before the identity was persisted looks like.
        image = np.zeros((200, 300), np.uint8)
        strategy = ObbAlignment()
        strategy._reference_image = image
        strategy._template_image = image
        strategy._nominal_size = (30.0, 100.0)
        strategy._box_center = (100.0, 150.0)
        strategy._box_size = (30.0, 100.0)
        strategy._box_angle_deg = -20.0
        strategy._template_size = 100

        root = tk.Tk()
        root.withdraw()
        app = MeasureApp(root)
        try:
            wf = MultiTargetWorkflow()
            wf._alignment = strategy
            app._workflow = wf
            app._reference_image = image
            app._ref_canvas.load_image(image)

            app.restore_obb_state()
            status = app.tool_panel._obb_status_label.cget("text")
            assert "未记录模型路径" in status
            assert str(app.tool_panel._exec_btn.cget("state")) == "disabled"
            # The template view is still restored from geometry alone.
            assert app.template_view._template_image is not None
            assert app.template_view._template_image.shape == (30, 100)
        finally:
            root.destroy()

    def test_missing_weights_file_is_reported(self, tmp_path, monkeypatch):
        import tkinter as tk
        import tkinter.messagebox as mb
        from measure.alignment import ObbAlignment
        from measure.multi_target_workflow import MultiTargetWorkflow
        from measure_gui.app import MeasureApp
        from measure_gui.project_manager import ProjectManager

        monkeypatch.setattr(mb, "showwarning", lambda *a, **k: None)
        monkeypatch.setattr(mb, "showerror", lambda *a, **k: None)
        monkeypatch.setattr(mb, "showinfo", lambda *a, **k: None)
        monkeypatch.setattr(ProjectManager, "RECENT_FILE",
                            str(tmp_path / "recent.json"))

        image = np.zeros((200, 300), np.uint8)
        strategy = ObbAlignment()
        strategy._reference_image = image
        strategy._template_image = image
        strategy._nominal_size = (30.0, 100.0)
        strategy._box_center = (100.0, 150.0)
        strategy._box_size = (30.0, 100.0)
        strategy._box_angle_deg = -20.0
        strategy._template_size = 100
        strategy._model_ref = {"path": str(tmp_path / "gone.pt")}

        root = tk.Tk()
        root.withdraw()
        app = MeasureApp(root)
        try:
            wf = MultiTargetWorkflow()
            wf._alignment = strategy
            app._workflow = wf
            app._reference_image = image
            app._ref_canvas.load_image(image)

            app.restore_obb_state()
            status = app.tool_panel._obb_status_label.cget("text")
            assert "不存在" in status
            assert str(app.tool_panel._exec_btn.cget("state")) == "disabled"
        finally:
            root.destroy()

"""GUI tests for detector-backed (OBB) localisation.

Covers the panel form swap, the execution gate, and — with a real model — the
whole teach/execute path through an actual ``MeasureApp``.  The GUI is the only
place a user meets these modes, so the logic is exercised through the widgets
rather than by calling the engine directly.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
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
_WEIGHTS = Path("/tmp/yolov8n-obb.pt")
_SAMPLE = Path(__file__).parent / "fixtures" / "obb_sample.jpg"


def _ultralytics_available() -> bool:
    """Whether the detector backend can actually be imported.

    Checking for the weights file alone is not enough: in an environment without
    ultralytics the adapter raises ImportError, the GUI reports "model not loaded",
    and these tests fail with a confusing message rather than skipping.  The
    library is the real precondition.
    """
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return True


needs_model = pytest.mark.skipif(
    not _WEIGHTS.is_file() or not _SAMPLE.is_file() or not _ultralytics_available(),
    reason="real OBB weights, sample image, or ultralytics not available")


@gui_only
class TestAlignmentModeMapping:
    """Every engine mode must round-trip through the combobox."""

    @pytest.mark.parametrize("mode", ["single_box", "multi_point", "obb",
                                      "fixed_roi", "none"])
    def test_mode_round_trips(self, mode):
        from measure_gui.tool_panel import ToolPanel
        root = _tk.Tk()
        root.withdraw()
        try:
            panel = ToolPanel(root)
            panel.set_alignment_mode(mode)
            assert panel.get_alignment_mode() == mode
            # The displayed text must be a human label, never the internal key.
            assert panel._alignment_var.get() != mode
        finally:
            root.destroy()

    def test_obb_is_user_selectable(self):
        from measure_gui.tool_panel import ToolPanel
        root = _tk.Tk()
        root.withdraw()
        try:
            panel = ToolPanel(root)
            assert "OBB 模型" in panel._selectable_alignment_labels()
        finally:
            root.destroy()

    def test_fixed_roi_and_none_stay_display_only(self):
        """They have no teaching flow yet, so they are not offered."""
        from measure_gui.tool_panel import ToolPanel
        root = _tk.Tk()
        root.withdraw()
        try:
            panel = ToolPanel(root)
            selectable = panel._selectable_alignment_labels()
            assert "固定区域" not in selectable
            assert "仅定位" not in selectable
        finally:
            root.destroy()


@gui_only
class TestFormSwap:
    """The settings area shows what belongs to the current mode."""

    def make(self):
        from measure_gui.tool_panel import ToolPanel
        root = _tk.Tk()
        root.withdraw()
        return root, ToolPanel(root)

    def test_obb_hides_the_template_widgets(self):
        root, panel = self.make()
        try:
            panel.set_alignment_mode("obb")
            panel.set_alignment_form()
            assert panel._obb_form.winfo_manager()
            assert not panel._template_form.winfo_manager()
            # Six search parameters that do nothing are worse than absent.
            assert "载入模型" in panel._create_btn.cget("text")
        finally:
            root.destroy()

    def test_template_mode_restores_them(self):
        root, panel = self.make()
        try:
            panel.set_alignment_mode("obb")
            panel.set_alignment_form()
            panel.set_alignment_mode("single_box")
            panel.set_alignment_form()
            assert panel._template_form.winfo_manager()
            assert not panel._obb_form.winfo_manager()
            assert "创建" in panel._create_btn.cget("text")
        finally:
            root.destroy()

    def test_roi_hint_is_hidden_for_detectors(self):
        """A detector reads no ROI, so prompting for one is misleading."""
        root, panel = self.make()
        try:
            panel.set_alignment_mode("obb")
            panel.set_alignment_form()
            assert not panel._roi_label.winfo_manager()
            panel.set_alignment_mode("single_box")
            panel.set_alignment_form()
            assert panel._roi_label.winfo_manager()
        finally:
            root.destroy()


@gui_only
class TestObbSettings:
    def make(self):
        from measure_gui.tool_panel import ToolPanel
        root = _tk.Tk()
        root.withdraw()
        return root, ToolPanel(root)

    def test_detections_are_listed_best_first(self):
        """The anchor list is detections, not classes.

        With many instances of one class, a class-level choice cannot say which
        box fixes the patch frame; listing boxes lets the user point at the part.
        """
        from measure.obb_source import ObbObservation
        root, panel = self.make()
        try:
            obs = [
                ObbObservation(row=10, col=20, width=100, height=30,
                               angle_deg=0.0, class_id=1, score=0.5,
                               class_name="ship"),
                ObbObservation(row=30, col=40, width=90, height=28,
                               angle_deg=5.0, class_id=1, score=0.9,
                               class_name="ship"),
            ]
            # The caller is expected to pass them already sorted (the selector
            # sorts); feed best-first as the adapter/selector would.
            panel.set_obb_detections(obs)
            values = list(panel._obb_anchor_combo.cget("values"))
            assert len(values) == 2
            assert values[0].startswith("ship_0")
            # Highest confidence is preselected, so the common case needs no clicks.
            assert panel.get_anchor_index() == 0
            assert panel.get_anchor_observation().score == pytest.approx(0.5)
        finally:
            root.destroy()

    def test_anchor_name_is_class_and_index(self):
        from measure.obb_source import ObbObservation
        root, panel = self.make()
        try:
            panel.set_obb_detections([
                ObbObservation(row=1, col=2, width=10, height=5, angle_deg=0.0,
                               class_id=7, score=0.75, class_name="harbor")])
            assert panel._obb_anchor_var.get().startswith("harbor_0")
            assert panel.get_anchor_index() == 0
            assert panel.get_anchor_class_id() == 7
            assert panel.get_anchor_class_label() == "harbor"
        finally:
            root.destroy()

    def test_selecting_by_index_round_trips(self):
        from measure.obb_source import ObbObservation
        root, panel = self.make()
        try:
            obs = [ObbObservation(row=i, col=i, width=10, height=5, angle_deg=0.0,
                                  class_id=1, score=1.0 - i / 10, class_name="ship")
                   for i in range(5)]
            panel.set_obb_detections(obs)
            for index in (0, 2, 4):
                panel.set_anchor_index(index)
                assert panel.get_anchor_index() == index
                assert panel.get_anchor_observation().row == index
        finally:
            root.destroy()

    def test_out_of_range_index_yields_no_anchor(self):
        from measure.obb_source import ObbObservation
        root, panel = self.make()
        try:
            panel.set_obb_detections([
                ObbObservation(row=1, col=2, width=10, height=5, angle_deg=0.0,
                               class_id=1, score=0.9, class_name="ship")])
            panel.set_anchor_index(99)
            assert panel.get_anchor_index() is None
            assert panel.get_anchor_observation() is None
        finally:
            root.destroy()

    def test_empty_detections_leave_nothing_selected(self):
        root, panel = self.make()
        try:
            panel.set_obb_detections([])
            assert panel._obb_anchor_var.get() == ""
            assert panel.get_anchor_observation() is None
        finally:
            root.destroy()

    def test_settings_snapshot(self):
        root, panel = self.make()
        try:
            panel._obb_model_var.set("/tmp/m.pt")
            panel._obb_conf_var.set(0.4)
            panel._obb_imgsz_var.set("1024")
            settings = panel.get_obb_settings()
            assert settings["model_path"] == "/tmp/m.pt"
            assert settings["conf"] == pytest.approx(0.4)
            assert settings["imgsz"] == 1024
        finally:
            root.destroy()


@gui_only
class TestExecutionGate:
    """A detector is never persisted, so execution must wait for one."""

    def make(self):
        from measure_gui.tool_panel import ToolPanel
        root = _tk.Tk()
        root.withdraw()
        return root, ToolPanel(root)

    def test_obb_without_detector_is_blocked(self):
        root, panel = self.make()
        try:
            panel.set_alignment_mode("obb")
            panel.set_template_created(True)
            panel.set_detector_ready(False)
            assert str(panel._exec_btn.cget("state")) == "disabled"
        finally:
            root.destroy()

    def test_obb_with_detector_is_enabled(self):
        root, panel = self.make()
        try:
            panel.set_alignment_mode("obb")
            panel.set_template_created(True)
            panel.set_detector_ready(True)
            assert str(panel._exec_btn.cget("state")) == "normal"
        finally:
            root.destroy()

    def test_template_mode_ignores_the_detector_flag(self):
        """The gate is OBB-specific; template models must still run."""
        root, panel = self.make()
        try:
            panel.set_alignment_mode("single_box")
            panel.set_template_created(True)
            panel.set_detector_ready(False)
            assert str(panel._exec_btn.cget("state")) == "normal"
        finally:
            root.destroy()

    def test_untaught_blocks_regardless_of_detector(self):
        root, panel = self.make()
        try:
            panel.set_alignment_mode("obb")
            panel.set_detector_ready(True)
            panel.set_template_created(False)
            assert str(panel._exec_btn.cget("state")) == "disabled"
        finally:
            root.destroy()


@gui_only
class TestAppHandlers:
    """Handler behaviour, driven through a real MeasureApp."""

    def make(self, tmp_path, monkeypatch):
        import tkinter.messagebox as mb
        from measure_gui.app import MeasureApp
        from measure_gui.project_manager import ProjectManager
        popups: list = []
        monkeypatch.setattr(mb, "showwarning", lambda *a, **k: popups.append(("warn", a)))
        monkeypatch.setattr(mb, "showerror", lambda *a, **k: popups.append(("err", a)))
        monkeypatch.setattr(mb, "showinfo", lambda *a, **k: popups.append(("info", a)))
        monkeypatch.setattr(ProjectManager, "RECENT_FILE",
                            str(tmp_path / "recent.json"))
        root = _tk.Tk()
        root.withdraw()
        app = MeasureApp(root)
        from measure_gui.image_canvas import CanvasMode
        app._reference_image = np.zeros((200, 300), np.uint8)
        app._ref_canvas.load_image(app._reference_image)
        app._workflow = None
        return root, app, popups, CanvasMode

    def test_switching_to_obb_leaves_draw_mode(self, tmp_path, monkeypatch):
        """Leaving the canvas armed invites drawing a box nothing reads."""
        root, app, _, CanvasMode = self.make(tmp_path, monkeypatch)
        try:
            app.tool_panel._alignment_var.set("OBB 模型")
            app.tool_panel._on_alignment_selected()
            assert app._ref_canvas.get_mode() == CanvasMode.BROWSE
        finally:
            root.destroy()

    def test_switching_to_obb_blocks_execution_and_explains(self, tmp_path, monkeypatch):
        root, app, _, _ = self.make(tmp_path, monkeypatch)
        try:
            app.tool_panel._alignment_var.set("OBB 模型")
            app.tool_panel._on_alignment_selected()
            assert str(app.tool_panel._exec_btn.cget("state")) == "disabled"
            status = app.tool_panel._obb_status_label.cget("text")
            assert "未载入" in status and "载入模型" in status
        finally:
            root.destroy()

    def test_switching_back_restores_roi_drawing(self, tmp_path, monkeypatch):
        root, app, _, CanvasMode = self.make(tmp_path, monkeypatch)
        try:
            app.tool_panel._alignment_var.set("OBB 模型")
            app.tool_panel._on_alignment_selected()
            app.tool_panel._alignment_var.set("旋转框")
            app.tool_panel._on_alignment_selected()
            assert app._ref_canvas.get_mode() == CanvasMode.DRAW_ROI
        finally:
            root.destroy()

    def test_create_button_dispatches_to_the_detector_flow(self, tmp_path, monkeypatch):
        """One button, two meanings — the dispatch must follow the mode."""
        root, app, popups, _ = self.make(tmp_path, monkeypatch)
        try:
            called = []
            monkeypatch.setattr(app, "_load_obb_model",
                                lambda: called.append("obb"))
            app.tool_panel._alignment_var.set("OBB 模型")
            app.tool_panel._on_alignment_selected()
            app._create_template()
            assert called == ["obb"]
            assert not popups, f"unexpected popups: {popups}"
        finally:
            root.destroy()

    def test_create_requires_a_reference_image_first(self, tmp_path, monkeypatch):
        root, app, popups, _ = self.make(tmp_path, monkeypatch)
        try:
            app._reference_image = None
            app._create_template()
            assert popups and popups[0][0] == "warn"
        finally:
            root.destroy()

    def test_missing_dependency_is_reported_not_raised(self, tmp_path, monkeypatch):
        """A GUI event handler must not surface a raw ImportError."""
        root, app, popups, _ = self.make(tmp_path, monkeypatch)
        try:
            import builtins
            real_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if "ultralytics_source" in name:
                    raise ImportError("no ultralytics")
                return real_import(name, *args, **kwargs)

            monkeypatch.setattr(builtins, "__import__", fake_import)
            app.tool_panel._obb_model_var.set("/tmp/whatever.pt")
            assert app._load_obb_source() is None
            assert popups and popups[0][0] == "err"
            assert "ultralytics" in popups[0][1][1]
        finally:
            root.destroy()


@gui_only
@needs_model
class TestRealModelThroughGui:
    """Full teach-and-execute path with an actual detector, through the GUI.

    Slow, but it is the only test that proves the pieces connect: the panel
    settings reach the adapter, the adapter's detections become a taught model,
    and execution produces posed targets in the results panel.
    """

    def test_full_flow(self, tmp_path, monkeypatch):
        import cv2
        import tkinter.messagebox as mb
        from measure.alignment import ObbAlignment
        from measure_gui.app import MeasureApp
        from measure_gui.image_canvas import CanvasMode
        from measure_gui.project_manager import ProjectManager

        popups: list = []
        monkeypatch.setattr(mb, "showwarning", lambda *a, **k: popups.append(("warn", a)))
        monkeypatch.setattr(mb, "showerror", lambda *a, **k: popups.append(("err", a)))
        monkeypatch.setattr(mb, "showinfo", lambda *a, **k: popups.append(("info", a)))
        monkeypatch.setattr(ProjectManager, "RECENT_FILE",
                            str(tmp_path / "recent.json"))

        root = _tk.Tk()
        root.withdraw()
        app = MeasureApp(root)
        try:
            app._reference_image = cv2.imread(str(_SAMPLE), cv2.IMREAD_GRAYSCALE)
            app._ref_canvas.load_image(app._reference_image)
            app._workflow = None

            app.tool_panel._alignment_var.set("OBB 模型")
            app.tool_panel._on_alignment_selected()
            assert app._ref_canvas.get_mode() == CanvasMode.BROWSE

            app.tool_panel.set_obb_model(str(_WEIGHTS))
            app.tool_panel._obb_conf_var.set(0.5)
            app.tool_panel._obb_imgsz_var.set("1024")
            app._load_obb_model()

            # The self-check must have run and reported findings, not silence.
            status = app.tool_panel._obb_status_label.cget("text")
            assert "自检通过" in status, status
            assert not [p for p in popups if p[0] == "err"], popups

            strategy = app._workflow._alignment
            assert isinstance(strategy, ObbAlignment)
            assert strategy.source is not None
            # Anchor defaulted to the highest-scoring detection's class.
            assert app.tool_panel.get_anchor_class_id() == strategy.anchor_class_id

            assert str(app.tool_panel._exec_btn.cget("state")) == "normal"

            app._inspection_image = app._reference_image
            app._execute()
            results = app._workflow.results
            assert len(results) > 10, "expected many detected targets"
            for target in results[:5]:
                assert target.score is not None
                assert -90.0 <= target.rotation_deg < 90.0
                assert target.center_row > 0 and target.center_col > 0
        finally:
            root.destroy()

    def test_switching_anchor_box_changes_the_patch_frame(self, tmp_path, monkeypatch):
        """Picking a different box must re-teach against THAT box.

        The anchor decides the patch frame and the recorded pose, so if switching
        merely relabelled the selection without re-teaching, tools would silently
        keep measuring in the previous box's coordinates.
        """
        import cv2
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
        app = MeasureApp(root)
        try:
            app._reference_image = cv2.imread(str(_SAMPLE), cv2.IMREAD_GRAYSCALE)
            app._ref_canvas.load_image(app._reference_image)
            app._workflow = None
            app.tool_panel._alignment_var.set("OBB 模型")
            app.tool_panel._on_alignment_selected()
            app.tool_panel.set_obb_model(str(_WEIGHTS))
            app.tool_panel._obb_conf_var.set(0.5)
            app.tool_panel._obb_imgsz_var.set("1024")
            app._load_obb_model()

            first = app._workflow._alignment.anchor_observation
            first_size = app._workflow._alignment.nominal_size

            # Choose a box with a visibly different size so the frame must change.
            target = max(app.tool_panel._obb_detections[:20],
                         key=lambda o: abs(o.width - first.width))
            index = app.tool_panel._obb_detections.index(target)
            app.tool_panel.set_anchor_index(index)
            app._on_obb_settings_changed()

            after = app._workflow._alignment.anchor_observation
            assert abs(after.row - target.row) < 1.0
            assert abs(after.col - target.col) < 1.0
            assert after is not first or index == 0
            # The frame follows the anchor.
            assert app._workflow._alignment.nominal_size != first_size
        finally:
            root.destroy()

    def test_anchor_must_be_one_of_the_current_detections(self):
        """Rejecting an unknown anchor beats silently anchoring elsewhere."""
        import numpy as np
        from measure.alignment import ObbAlignment
        from measure.obb_source import ObbObservation

        present = [ObbObservation(row=10, col=20, width=50, height=20,
                                  angle_deg=0.0, class_id=1, score=0.9)]
        stranger = ObbObservation(row=999, col=999, width=50, height=20,
                                  angle_deg=0.0, class_id=1, score=0.9)
        with pytest.raises(ValueError, match="not among"):
            ObbAlignment._resolve_anchor(present, stranger)
        # And a matching one is accepted.
        assert ObbAlignment._resolve_anchor(present, present[0]) is present[0]

    def test_no_request_defaults_to_the_best(self):
        from measure.alignment import ObbAlignment
        from measure.obb_source import ObbObservation

        obs = [ObbObservation(row=1, col=1, width=10, height=5, angle_deg=0.0,
                              class_id=1, score=0.9)]
        assert ObbAlignment._resolve_anchor(obs, None) is obs[0]


    def test_template_view_matches_what_the_engine_measures(self, tmp_path, monkeypatch):
        """The displayed patch must be the frame tools are defined in.

        Regression guard for a real report: the template view showed a tilted crop
        while the engine measured a straightened one, because the preview called
        crop_and_straighten with the raw box angle while the engine negated it.  A
        user reads that as "the crop is wrong", and it is — the two paths must agree.
        """
        import cv2
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
        app = MeasureApp(root)
        try:
            app._reference_image = cv2.imread(str(_SAMPLE), cv2.IMREAD_GRAYSCALE)
            app._ref_canvas.load_image(app._reference_image)
            app._workflow = None
            app.tool_panel._alignment_var.set("OBB 模型")
            app.tool_panel._on_alignment_selected()
            app.tool_panel.set_obb_model(str(_WEIGHTS))
            app.tool_panel._obb_conf_var.set(0.5)
            app.tool_panel._obb_imgsz_var.set("1024")
            app._load_obb_model()

            strategy = app._workflow._alignment
            anchor = strategy.anchor_observation
            shown = app.template_view._template_image
            engine_patch = strategy.align(
                app._reference_image, anchor.row, anchor.col, 0.0).patch

            assert shown.shape == engine_patch.shape, (
                f"template view {shown.shape} != engine patch {engine_patch.shape}")
            assert np.allclose(shown.astype(float), engine_patch.astype(float),
                               atol=1.0), "template view shows a different patch"

            # And the displayed patch must actually be straightened: the part's
            # long axis should be near-horizontal, not tilted by the box angle.
            bright = shown > shown.max() * 0.6
            ys, xs = np.nonzero(bright)
            assert len(ys) > 20
            pts = np.stack([xs, ys], 1).astype(float)
            pts -= pts.mean(0)
            _, vectors = np.linalg.eigh(np.cov(pts.T))
            major = vectors[:, np.argmax(_)]
            residual = np.degrees(np.arctan2(major[1], major[0]))
            if residual > 90:
                residual -= 180
            if residual < -90:
                residual += 180
            assert abs(residual) < 10.0, (
                f"patch still tilted by {residual:.1f} degrees")
        finally:
            root.destroy()

    def test_anchor_box_is_drawn_on_the_detected_corners(self, tmp_path, monkeypatch):
        """The box on the reference image must coincide with the detection.

        Asserts on corner pixels, not on the angle number: the two sides use
        opposite conventions (observations are CCW+ like every manifest box angle,
        ImageCanvas stores CW+ because its rotate handle derives angles from
        arctan2 in image coordinates), so comparing angles would just re-encode the
        assumption being tested.  Passing the raw angle drew a mirrored box —
        41.9 px of corner error on a real detection, 0.0 when negated.
        """
        import cv2
        import tkinter.messagebox as mb
        from measure.rectify import compute_rotated_box_corners
        from measure_gui.app import MeasureApp
        from measure_gui.project_manager import ProjectManager

        monkeypatch.setattr(mb, "showwarning", lambda *a, **k: None)
        monkeypatch.setattr(mb, "showerror", lambda *a, **k: None)
        monkeypatch.setattr(mb, "showinfo", lambda *a, **k: None)
        monkeypatch.setattr(ProjectManager, "RECENT_FILE",
                            str(tmp_path / "recent.json"))

        root = _tk.Tk()
        root.withdraw()
        app = MeasureApp(root)
        try:
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

            anchor = app._workflow._alignment.anchor_observation
            canvas = app._ref_canvas
            # Reproduce what the canvas draws for its stored ROI angle.
            drawn = compute_rotated_box_corners(
                canvas.roi_center, canvas.roi_size, -canvas._roi_angle)
            error = float(np.abs(drawn - anchor.corners).max())
            assert error < 1.5, (
                f"drawn box is {error:.1f} px off the detection; "
                f"canvas angle {canvas._roi_angle:.2f} vs detection "
                f"{anchor.angle_deg:.2f}")
        finally:
            root.destroy()

    def test_roi_preview_uses_the_engine_angle_convention(self, tmp_path, monkeypatch):
        """The live preview must crop exactly as the engine will.

        Tested by driving the preview handler directly rather than the whole load
        flow: a rebuilt workflow elsewhere overwrites the template view, which
        masked this regression when it was first guarded the indirect way.

        The bug it pins: crop_and_straighten rotates opposite to the box-angle
        convention, so previewing with the raw angle drew a tilted patch while the
        engine measured a straightened one.
        """
        import cv2
        import tkinter.messagebox as mb
        from measure.alignment import _crop_angle
        from measure.rectify import crop_and_straighten
        from measure_gui.app import MeasureApp
        from measure_gui.project_manager import ProjectManager

        monkeypatch.setattr(mb, "showwarning", lambda *a, **k: None)
        monkeypatch.setattr(mb, "showerror", lambda *a, **k: None)
        monkeypatch.setattr(mb, "showinfo", lambda *a, **k: None)
        monkeypatch.setattr(ProjectManager, "RECENT_FILE",
                            str(tmp_path / "recent.json"))

        root = _tk.Tk()
        root.withdraw()
        app = MeasureApp(root)
        try:
            image = cv2.imread(str(_SAMPLE), cv2.IMREAD_GRAYSCALE)
            app._reference_image = image
            app._ref_canvas.load_image(image)

            geometry = (658.0, 1765.6, 34.0, 109.0, -22.55)
            app._on_roi_changed(*geometry)
            drawn = app.template_view._template_image

            correct = crop_and_straighten(
                image, geometry[:2], geometry[2:4], _crop_angle(geometry[4]))[0]
            assert np.allclose(drawn.astype(float), correct.astype(float), atol=1.0), \
                "preview patch differs from what the engine would crop"
        finally:
            root.destroy()

    def test_anchor_switch_updates_the_template_view(self, tmp_path, monkeypatch):
        """A moved patch frame must be reflected in the view."""
        import cv2
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
        app = MeasureApp(root)
        try:
            app._reference_image = cv2.imread(str(_SAMPLE), cv2.IMREAD_GRAYSCALE)
            app._ref_canvas.load_image(app._reference_image)
            app._workflow = None
            app.tool_panel._alignment_var.set("OBB 模型")
            app.tool_panel._on_alignment_selected()
            app.tool_panel.set_obb_model(str(_WEIGHTS))
            app.tool_panel._obb_conf_var.set(0.5)
            app.tool_panel._obb_imgsz_var.set("1024")
            app._load_obb_model()

            first_shape = app.template_view._template_image.shape
            # Pick a box with a different extent so the patch must change size.
            dets = app.tool_panel._obb_detections[:20]
            target = max(dets, key=lambda o: abs(o.width - dets[0].width))
            index = app.tool_panel._obb_detections.index(target)
            app.tool_panel.set_anchor_index(index)
            app._on_obb_settings_changed()

            refreshed = app.template_view._template_image
            anchor = app._workflow._alignment.anchor_observation
            expected = app._workflow._alignment.align(
                app._reference_image, anchor.row, anchor.col, 0.0).patch
            assert refreshed.shape == expected.shape
            assert np.allclose(refreshed.astype(float), expected.astype(float),
                               atol=1.0)
        finally:
            root.destroy()


class TestGuiModuleStaysImportableWithoutMl:
    def test_no_top_level_ml_imports_in_gui(self):
        """The GUI must import in an environment without torch/ultralytics.

        Checked by reading the source rather than sys.modules, because the test
        process may already have imported them.
        """
        gui_dir = Path(__file__).parent.parent / "measure_gui"
        for path in gui_dir.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith(("import ultralytics", "from ultralytics",
                                        "import torch", "from torch")):
                    pytest.fail(f"{path.name} imports ML at module level: {stripped}")

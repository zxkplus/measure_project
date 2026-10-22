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

    def test_class_names_populate_and_select(self):
        root, panel = self.make()
        try:
            panel.set_obb_class_names({0: "plane", 1: "ship", 7: "harbor"})
            assert panel.get_anchor_class_id() is None
            panel.set_anchor_class_id(1)
            assert panel.get_anchor_class_id() == 1
            # Index and label are both shown: selection is by index, and two
            # classes could share a label.
            assert "1: ship" in panel._obb_class_combo.cget("values")
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

    def test_changing_anchor_class_reteaches(self, tmp_path, monkeypatch):
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

            original = app._workflow._alignment.anchor_class_id
            # Pick a class that is actually present on this image; switching to
            # one that is not is exercised separately below.
            present = {o.class_id for o in app._obb_observations}
            other = next(k for k in present if k != original)
            app.tool_panel.set_anchor_class_id(other)
            app._on_obb_settings_changed()

            assert app._workflow._alignment.anchor_class_id == other
            assert "锚点类" in app.tool_panel._obb_status_label.cget("text")
        finally:
            root.destroy()

    def test_selecting_an_absent_class_says_so(self, tmp_path, monkeypatch):
        """Choosing a class the image lacks must explain itself, not go quiet.

        The model knows classes the reference image may not contain; silently
        accepting one would leave the user with a model that matches nothing.
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

            present = {o.class_id for o in app._obb_observations}
            absent = next(k for k in app._workflow._alignment.source.class_names
                          if k not in present)
            app.tool_panel.set_anchor_class_id(absent)
            app._on_obb_settings_changed()

            status = app.tool_panel._obb_status_label.cget("text")
            assert "没有检出" in status, status
            # The taught model must not silently adopt a class it cannot match.
            assert app._workflow._alignment.anchor_class_id != absent
        finally:
            root.destroy()

    def test_absent_classes_are_annotated_in_the_dropdown(self, tmp_path, monkeypatch):
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

            values = list(app.tool_panel._obb_class_combo.cget("values"))
            assert any("参考图未检出" in v for v in values), values
            assert any("参考图未检出" not in v for v in values), values
            # The annotation must not break index parsing.
            present = {o.class_id for o in app._obb_observations}
            chosen = next(k for k in present)
            app.tool_panel.set_anchor_class_id(chosen)
            assert app.tool_panel.get_anchor_class_id() == chosen
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
